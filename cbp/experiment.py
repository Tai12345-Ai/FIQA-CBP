from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.metadata
import platform
import sys
import time
import numpy as np
import pandas as pd

from cbp.analysis import mean_matched_evidence_test
from cbp.baselines.continuous import train_multi_context_continuous_policy
from cbp.baselines.image_policy import (
    predict_image_policy,
    train_multi_context_image_policy,
)
from cbp.baselines.ranked import solve_ranked_cell_policy
from cbp.certification.certificate import certify_one_sided
from cbp.config import ContextConfig, ExperimentConfig
from cbp.data.pairs import identity_balanced_sample_weights
from cbp.data.splits import (
    SPLIT_NAMES,
    assert_identity_disjoint,
    cap_identities,
    eligible_capture_mask,
    make_identity_splits,
)
from cbp.fr.cosine import l2_normalize, sample_impostor_scores
from cbp.fr.embeddings import EmbeddingTable
from cbp.metrics.identity import (
    evaluate_one_sided,
    identity_components,
    paired_identity_multiplier_bootstrap_all,
)
from cbp.policy.quality import context_quality_score, decoded_profile
from cbp.policy.solver import PolicySolution, solve_policy
from cbp.representation.partitions import (
    Partition,
    embedding_kmeans_partition,
    evidence_features,
    feature_partition,
    projection_diagnostics,
    random_partition,
)
from cbp.router.train import predict_router, save_router, train_router
from cbp.router.transfer import router_transfer_metrics
from cbp.statistics.finite_state import build_evidence_operators, hard_matcher_from_embeddings
from cbp.statistics.identity_kernel import estimate_identity_cluster_moments
from cbp.statistics.moments import CellMoments, aggregate_cell_moments
from cbp.statistics.state_space import StateSpace, build_state_space
from cbp.statistics.targets import estimate_genuine_utility
from cbp.utils.io import ensure_dir, jsonable, save_json, save_npz, sha256_file
from cbp.utils.seed import seed_everything


@dataclass
class ContextRuntime:
    config: ContextConfig
    threshold: float
    threshold_pairs: int
    threshold_empirical_fmr: float
    identity_budget: int
    state_space: StateSpace
    operators: object


@dataclass
class CandidateEvaluation:
    name: str
    family: str
    partition: Partition
    moments: dict[str, CellMoments]
    solutions: dict[str, PolicySolution]
    dev_metrics: dict[str, object]
    projection: dict[str, dict[str, float]]
    fit_feasible: bool
    dev_feasible: bool
    selection_score: float

    def report(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "k": self.partition.k,
            "state_to_cell": self.partition.labels,
            "fit_feasible": self.fit_feasible,
            "dev_feasible": self.dev_feasible,
            "selection_score": self.selection_score,
            "solutions": self.solutions,
            "dev_metrics": self.dev_metrics,
            "projection_diagnostics": self.projection,
            "moments": {
                name: {
                    "p": m.p,
                    "u": m.u,
                    "ell": m.ell,
                    "B_eigenvalues": np.linalg.eigvalsh(m.B),
                    "mu_g": m.mu_g,
                    "mu_I": m.mu_I,
                    "counts": m.counts,
                    "n_fit_identities": m.n_fit_identities,
                    "pair_count": m.pair_count,
                }
                for name, m in self.moments.items()
            },
        }


def _reset_cuda_peak_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _cuda_peak_memory() -> int | None:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return None


def _save_torch_checkpoint(model, path: Path, metadata: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    config = model.config() if hasattr(model, "config") else None
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": config,
            "metadata": metadata,
        },
        path,
    )


def _package_versions() -> dict[str, str | None]:
    names = ["numpy", "scipy", "pandas", "scikit-learn", "torch", "cvxpy", "PyYAML"]
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _prepare_splits(table: EmbeddingTable, cfg: ExperimentConfig) -> tuple[EmbeddingTable, dict[str, np.ndarray], dict]:
    mask = eligible_capture_mask(table.identity_ids, cfg.min_captures_per_identity)
    dropped = int(np.sum(~mask))
    table = table.subset(np.flatnonzero(mask)).normalized()

    if table.split_labels is not None:
        labels = np.asarray(table.split_labels).astype(str)
        if not set(np.unique(labels)).issubset(set(SPLIT_NAMES)):
            raise ValueError(f"split_labels must use only {SPLIT_NAMES}")
        assert_identity_disjoint(table.identity_ids, labels)
    else:
        splits = make_identity_splits(table.identity_ids, cfg.splits.to_dict(), cfg.seed)
        labels = splits.labels
        table.split_labels = labels

    split_indices: dict[str, np.ndarray] = {}
    split_report = {"dropped_samples_for_min_captures": dropped, "splits": {}}
    for offset, name in enumerate(SPLIT_NAMES):
        idx = np.flatnonzero(labels == name)
        idx = cap_identities(
            idx,
            table.identity_ids,
            cfg.max_identities_per_split,
            cfg.seed + 1009 * offset,
        )
        if len(idx) == 0:
            raise ValueError(f"split is empty: {name}")
        split_indices[name] = idx
        split_report["splits"][name] = {
            "samples": int(len(idx)),
            "identities": int(len(np.unique(table.identity_ids[idx]))),
        }
    return table, split_indices, split_report


def _context_state_model(
    base: StateSpace,
    repr_table: EmbeddingTable,
    threshold: float,
    alpha: float,
) -> StateSpace:
    weights = identity_balanced_sample_weights(repr_table.identity_ids)
    g_sample = estimate_genuine_utility(repr_table.embeddings, repr_table.identity_ids, threshold)
    g_num = np.bincount(base.labels, weights=weights * g_sample, minlength=len(base.centroids))
    g = np.divide(g_num, base.p, out=np.zeros_like(g_num), where=base.p > 0)
    matcher = hard_matcher_from_embeddings(base.centroids, threshold)
    return StateSpace(
        centroids=base.centroids,
        labels=base.labels,
        p=base.p,
        g=g,
        matcher=matcher,
    )


def _heldout_phenomenon(
    runtime: ContextRuntime,
    dev_table: EmbeddingTable,
    permutations: int,
    match_quantile: float,
    seed: int,
) -> dict:
    state_labels = runtime.state_space.assign(dev_table.embeddings)
    weights = identity_balanced_sample_weights(dev_table.identity_ids)
    p = np.bincount(state_labels, weights=weights, minlength=len(runtime.state_space.centroids))
    g_sample = estimate_genuine_utility(dev_table.embeddings, dev_table.identity_ids, runtime.threshold)
    g_num = np.bincount(state_labels, weights=weights * g_sample, minlength=len(p))
    observed = np.flatnonzero(p > 0)
    if len(observed) < 4:
        return {"status": "INSUFFICIENT_OBSERVED_STATES", "n_observed_states": int(len(observed))}
    p_obs = p[observed]
    p_obs /= p_obs.sum()
    g_obs = g_num[observed] / p[observed]
    matcher = runtime.state_space.matcher[np.ix_(observed, observed)]
    operators = build_evidence_operators(matcher, alpha=runtime.config.target_fmr, p=p_obs)
    features = evidence_features(
        g_obs,
        operators,
        runtime.identity_budget,
        spectral_dim=min(8, len(observed)),
    )
    result = mean_matched_evidence_test(
        g_obs,
        operators.r,
        features[:, 2:],
        p=p_obs,
        match_quantile=match_quantile,
        permutations=permutations,
        seed=seed,
    )
    out = result.to_dict()
    out.update({"status": "OK", "n_observed_states": int(len(observed))})
    return out


def _partition_candidates(
    runtimes: dict[str, ContextRuntime],
    k_candidates: list[int],
    spectral_dim: int,
    seed: int,
) -> list[tuple[str, str, Partition]]:
    first = next(iter(runtimes.values()))
    n_states = len(first.state_space.centroids)
    mean_features = []
    evidence_blocks = []
    for runtime in runtimes.values():
        mean_features.extend([runtime.state_space.g, runtime.operators.r])
        evidence_blocks.append(
            evidence_features(
                runtime.state_space.g,
                runtime.operators,
                runtime.identity_budget,
                spectral_dim=spectral_dim,
            )
        )
    mean_x = np.column_stack(mean_features)
    evidence_x = np.column_stack(evidence_blocks)

    out: list[tuple[str, str, Partition]] = []
    for index, k in enumerate(sorted(set(k_candidates))):
        if k > n_states:
            continue
        out.extend([
            (
                f"cbp_evidence_k{k}",
                "cbp_evidence",
                feature_partition(evidence_x, k, seed + 100 * index + 1, f"cbp_evidence_k{k}"),
            ),
            (
                f"mean_only_k{k}",
                "mean_only",
                feature_partition(mean_x, k, seed + 100 * index + 2, f"mean_only_k{k}"),
            ),
            (
                f"embedding_kmeans_k{k}",
                "embedding_kmeans",
                embedding_kmeans_partition(first.state_space.centroids, k, seed + 100 * index + 3),
            ),
            (
                f"random_k{k}",
                "random",
                random_partition(n_states, k, seed + 100 * index + 4),
            ),
        ])
    out.append(("dense_state", "dense_state", Partition(np.arange(n_states), n_states, "dense_state")))
    return out


def _evaluate_partition(
    name: str,
    family: str,
    partition: Partition,
    runtimes: dict[str, ContextRuntime],
    dense_moments: dict[str, CellMoments],
    state_fit: np.ndarray,
    state_dev: np.ndarray,
    dev_table: EmbeddingTable,
) -> CandidateEvaluation:
    moments = {}
    solutions = {}
    dev_metrics = {}
    projection = {}
    fit_ok = True
    dev_ok = True
    dev_yields = []
    for ctx_name, runtime in runtimes.items():
        ctx = runtime.config
        m = aggregate_cell_moments(dense_moments[ctx_name], partition.labels, state_fit)
        sol = solve_policy(
            m,
            rho=ctx.rho,
            coverage_low=ctx.coverage_low,
            coverage_high=ctx.coverage_high,
            reserve=ctx.reserve,
        )
        cell_dev = partition.labels[state_dev]
        gate_dev = sol.beta[cell_dev]
        metrics = evaluate_one_sided(
            dev_table.embeddings,
            dev_table.identity_ids,
            gate_dev,
            runtime.threshold,
        )
        moments[ctx_name] = m
        solutions[ctx_name] = sol
        dev_metrics[ctx_name] = metrics
        projection[ctx_name] = projection_diagnostics(
            partition,
            runtime.state_space.g,
            runtime.operators,
            runtime.identity_budget,
        )
        fit_ok = fit_ok and sol.feasible
        context_dev_ok = (
            metrics.coverage >= ctx.coverage_low - 1e-8
            and metrics.coverage <= ctx.coverage_high + 1e-8
            and metrics.fmr <= ctx.target_fmr + 1e-12
        )
        dev_ok = dev_ok and context_dev_ok
        dev_yields.append(metrics.genuine_yield)
    score = float(min(dev_yields)) if dev_yields else float("-inf")
    if not fit_ok:
        score -= 10.0
    if not dev_ok:
        score -= 1.0
    return CandidateEvaluation(
        name=name,
        family=family,
        partition=partition,
        moments=moments,
        solutions=solutions,
        dev_metrics=dev_metrics,
        projection=projection,
        fit_feasible=fit_ok,
        dev_feasible=dev_ok,
        selection_score=score,
    )


def _select_candidate(candidates: list[CandidateEvaluation], family: str) -> tuple[CandidateEvaluation, str | None]:
    pool = [c for c in candidates if c.family == family]
    if not pool:
        raise RuntimeError(f"no candidates in family: {family}")
    strict = [c for c in pool if c.fit_feasible and c.dev_feasible]
    if strict:
        return max(strict, key=lambda c: c.selection_score), None
    fit_only = [c for c in pool if c.fit_feasible]
    if fit_only:
        return max(fit_only, key=lambda c: c.selection_score), "NO_DEV_FEASIBLE_CANDIDATE"
    return max(pool, key=lambda c: c.selection_score), "NO_FIT_FEASIBLE_CANDIDATE"


def _evaluate_factorial_partition_arm(
    partition: Partition,
    dense_moments: dict[str, CellMoments],
    runtimes: dict[str, ContextRuntime],
    state_fit: np.ndarray,
    state_dev: np.ndarray,
    dev_table: EmbeddingTable,
    objective_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Evaluate one hard/dense representation under mean or evidence objective."""
    if objective_name not in {"mean_risk", "evidence"}:
        raise ValueError(objective_name)
    state_actions: dict[str, np.ndarray] = {}
    contexts: dict[str, object] = {}
    for name, runtime in runtimes.items():
        context = runtime.config
        moments = aggregate_cell_moments(dense_moments[name], partition.labels, state_fit)
        rho = context.rho if objective_name == "evidence" else 0.0
        solution = solve_policy(
            moments,
            rho=rho,
            coverage_low=context.coverage_low,
            coverage_high=context.coverage_high,
            reserve=context.reserve,
        )
        action_state = solution.beta[partition.labels]
        gate_dev = action_state[state_dev]
        metrics = evaluate_one_sided(
            dev_table.embeddings,
            dev_table.identity_ids,
            gate_dev,
            runtime.threshold,
        )
        dev_feasible = bool(
            solution.feasible
            and metrics.coverage >= context.coverage_low - 1e-8
            and metrics.coverage <= context.coverage_high + 1e-8
            and metrics.fmr <= context.target_fmr + 1e-12
        )
        state_actions[name] = action_state
        contexts[name] = {
            "rho_used": rho,
            "solution": solution,
            "dev_metrics": metrics,
            "dev_feasible": dev_feasible,
        }
    return state_actions, {
        "representation": partition.name,
        "objective": objective_name,
        "contexts": contexts,
        "all_dev_feasible": all(item["dev_feasible"] for item in contexts.values()),
        "min_dev_yield": float(min(item["dev_metrics"].genuine_yield for item in contexts.values())),
    }


def _context_params(runtimes: dict[str, ContextRuntime]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "rho": runtime.config.rho,
            "reserve": runtime.config.reserve,
            "coverage_low": runtime.config.coverage_low,
            "coverage_high": runtime.config.coverage_high,
        }
        for name, runtime in runtimes.items()
    }


def run_experiment(config: ExperimentConfig | str | Path) -> dict:
    """Run the frozen-backbone, shared-router CBP experimental program."""
    config_path = Path(config) if isinstance(config, (str, Path)) else None
    cfg = ExperimentConfig.from_yaml(config_path) if config_path is not None else config
    cfg.validate()
    seed_everything(cfg.seed)
    _reset_cuda_peak_memory()
    total_start = time.perf_counter()
    stage_start = total_start
    stage_times: dict[str, float] = {}
    out_dir = ensure_dir(cfg.output_dir)
    artifact_dir = ensure_dir(out_dir / "artifact")
    baseline_dir = ensure_dir(out_dir / "baselines")

    table = EmbeddingTable.load(cfg.embedding_npz)
    table, split_indices, split_report = _prepare_splits(table, cfg)
    split_tables = {name: table.subset(idx) for name, idx in split_indices.items()}
    repr_table = split_tables["repr"]
    fit_table = split_tables["fit"]
    dev_table = split_tables["dev"]
    cal_table = split_tables["cal"]
    test_table = split_tables["test"]
    stage_times["load_and_split"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()

    # Persist exact identity split membership.
    split_rows = []
    for name, idx in split_indices.items():
        for i in idx:
            split_rows.append({
                "sample_id": table.sample_ids[i],
                "identity_id": table.identity_ids[i],
                "split": name,
            })
    pd.DataFrame(split_rows).to_csv(out_dir / "split_manifest.csv", index=False)

    threshold_info = {}
    thresholds = {}
    for offset, ctx in enumerate(cfg.contexts):
        scores = sample_impostor_scores(
            repr_table.embeddings,
            repr_table.identity_ids,
            max_pairs=ctx.max_impostor_pairs_for_threshold,
            seed=cfg.seed + 500 + offset,
        )
        threshold = float(np.quantile(scores, 1.0 - ctx.target_fmr, method="higher"))
        thresholds[ctx.name] = threshold
        threshold_info[ctx.name] = {
            "threshold": threshold,
            "impostor_pairs": int(len(scores)),
            "empirical_fmr": float(np.mean(scores >= threshold)),
            "target_fmr": ctx.target_fmr,
        }

    stage_times["threshold_design"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()

    first_ctx = cfg.contexts[0]
    base_state = build_state_space(
        repr_table.embeddings,
        repr_table.identity_ids,
        threshold=thresholds[first_ctx.name],
        n_states=cfg.n_states,
        seed=cfg.seed,
    )
    n_cal_ids = len(np.unique(cal_table.identity_ids))
    runtimes: dict[str, ContextRuntime] = {}
    for ctx in cfg.contexts:
        state = _context_state_model(
            base_state,
            repr_table,
            threshold=thresholds[ctx.name],
            alpha=ctx.target_fmr,
        )
        budget = int(ctx.identity_budget or n_cal_ids)
        operators = build_evidence_operators(state.matcher, alpha=ctx.target_fmr, p=state.p)
        info = threshold_info[ctx.name]
        runtimes[ctx.name] = ContextRuntime(
            config=ctx,
            threshold=thresholds[ctx.name],
            threshold_pairs=info["impostor_pairs"],
            threshold_empirical_fmr=info["empirical_fmr"],
            identity_budget=budget,
            state_space=state,
            operators=operators,
        )

    state_fit = base_state.assign(fit_table.embeddings)
    state_dev = base_state.assign(dev_table.embeddings)
    state_cal = base_state.assign(cal_table.embeddings)
    state_test = base_state.assign(test_table.embeddings)

    stage_times["state_construction"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()

    phenomenon = {}
    for offset, (name, runtime) in enumerate(runtimes.items()):
        try:
            phenomenon[name] = _heldout_phenomenon(
                runtime,
                dev_table,
                cfg.phenomenon_permutations,
                cfg.phenomenon_match_quantile,
                cfg.seed + 900 + offset,
            )
        except Exception as exc:
            phenomenon[name] = {"status": "FAILED", "error": repr(exc)}

    stage_times["phenomenon_E1"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()

    dense_moments: dict[str, CellMoments] = {}
    dense_diagnostics = {}
    for offset, (name, runtime) in enumerate(runtimes.items()):
        moments, diagnostics = estimate_identity_cluster_moments(
            fit_table.embeddings,
            fit_table.identity_ids,
            state_fit,
            threshold=runtime.threshold,
            alpha=runtime.config.target_fmr,
            identity_budget=runtime.identity_budget,
            n_cells=len(base_state.centroids),
            max_pairs=cfg.max_fit_identity_pairs,
            seed=cfg.seed + 1200 + offset,
        )
        dense_moments[name] = moments
        dense_diagnostics[name] = {
            "n_identities": diagnostics.n_identities,
            "n_pairs": diagnostics.n_pairs,
            "zeta1_eigenvalues": np.linalg.eigvalsh(diagnostics.zeta1),
            "zeta2_eigenvalues": np.linalg.eigvalsh(diagnostics.zeta2),
        }

    stage_times["identity_moment_fitting"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()

    candidates = []
    for name, family, partition in _partition_candidates(
        runtimes,
        cfg.k_candidates,
        cfg.spectral_dim,
        cfg.seed,
    ):
        candidates.append(
            _evaluate_partition(
                name,
                family,
                partition,
                runtimes,
                dense_moments,
                state_fit,
                state_dev,
                dev_table,
            )
        )

    selected, selection_warning = _select_candidate(candidates, "cbp_evidence")
    candidate_family_certificates = {}
    if cfg.certify_candidate_family:
        adjusted_delta = (0.5 * cfg.certification_delta) / max(
            1, len(candidates) * len(runtimes)
        )
        for candidate in candidates:
            cell_cal = candidate.partition.labels[state_cal]
            candidate_family_certificates[candidate.name] = {}
            for name, runtime in runtimes.items():
                context = runtime.config
                gate = candidate.solutions[name].beta[cell_cal]
                candidate_family_certificates[candidate.name][name] = certify_one_sided(
                    cal_table.embeddings,
                    cal_table.identity_ids,
                    gate,
                    threshold=runtime.threshold,
                    target_fmr=context.target_fmr,
                    coverage_target_low=context.coverage_low,
                    coverage_target_high=context.coverage_high,
                    delta=adjusted_delta,
                )
    stage_times["partition_candidates"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    dense_candidate = next(candidate for candidate in candidates if candidate.family == "dense_state")

    # Decisive factorial controls: {hard CBP, dense state} x
    # {mean-risk objective, evidence-aware objective}. Continuous arms are added
    # below with a shared learned bottleneck.
    factorial_actions: dict[str, dict[str, np.ndarray]] = {}
    factorial_report: dict[str, object] = {}
    for representation_name, partition in (
        ("hard_cbp", selected.partition),
        ("dense_state", dense_candidate.partition),
    ):
        for objective_name in ("mean_risk", "evidence"):
            arm_name = f"{representation_name}__{objective_name}"
            actions, arm_report = _evaluate_factorial_partition_arm(
                partition,
                dense_moments,
                runtimes,
                state_fit,
                state_dev,
                dev_table,
                objective_name,
            )
            factorial_actions[arm_name] = actions
            factorial_report[arm_name] = arm_report

    # Common and context-conditioned scalar-ranking baselines over the same states.
    common_score = np.mean(
        np.column_stack([runtime.state_space.g for runtime in runtimes.values()]),
        axis=1,
    )
    scalar_reports = {"common": {}, "context_conditioned": {}}
    scalar_actions: dict[str, dict[str, np.ndarray]] = {"common": {}, "context_conditioned": {}}
    for name, runtime in runtimes.items():
        ctx = runtime.config
        common = solve_ranked_cell_policy(
            common_score,
            dense_moments[name],
            ctx.rho,
            ctx.coverage_low,
            ctx.coverage_high,
            ctx.reserve,
        )
        conditional = solve_ranked_cell_policy(
            runtime.state_space.g,
            dense_moments[name],
            ctx.rho,
            ctx.coverage_low,
            ctx.coverage_high,
            ctx.reserve,
        )
        scalar_actions["common"][name] = common.action
        scalar_actions["context_conditioned"][name] = conditional.action
        scalar_reports["common"][name] = common
        scalar_reports["context_conditioned"][name] = conditional

    continuous_reports: dict[str, dict] = {}
    continuous_actions_by_objective: dict[str, dict[str, np.ndarray]] = {}
    if cfg.run_continuous_baseline:
        feature_blocks = [base_state.centroids]
        for runtime in runtimes.values():
            feature_blocks.append(
                evidence_features(
                    runtime.state_space.g,
                    runtime.operators,
                    runtime.identity_budget,
                    spectral_dim=cfg.spectral_dim,
                )
            )
        continuous_features = np.column_stack(feature_blocks)
        evidence_params = _context_params(runtimes)
        mean_params = {
            name: {**params, "rho": 0.0}
            for name, params in evidence_params.items()
        }
        for objective_name, params, seed_offset in (
            ("evidence", evidence_params, 1700),
            ("mean_risk", mean_params, 1800),
        ):
            continuous = train_multi_context_continuous_policy(
                continuous_features,
                dense_moments,
                params,
                bottleneck_dim=cfg.continuous_bottleneck_dim,
                epochs=cfg.continuous_epochs,
                seed=cfg.seed + seed_offset,
                device=cfg.device,
            )
            continuous_actions_by_objective[objective_name] = continuous.actions
            dev_metrics = {}
            dev_feasible = {}
            for name, runtime in runtimes.items():
                gate_dev = continuous.actions[name][state_dev]
                metrics = evaluate_one_sided(
                    dev_table.embeddings,
                    dev_table.identity_ids,
                    gate_dev,
                    runtime.threshold,
                )
                context = runtime.config
                dev_metrics[name] = metrics
                dev_feasible[name] = bool(
                    continuous.feasible[name]
                    and metrics.coverage >= context.coverage_low - 1e-8
                    and metrics.coverage <= context.coverage_high + 1e-8
                    and metrics.fmr <= context.target_fmr + 1e-12
                )
            _save_torch_checkpoint(
                continuous.model,
                baseline_dir / f"continuous_{objective_name}.pt",
                {
                    "objective": objective_name,
                    "seed": cfg.seed + seed_offset,
                    "feature_scope": "shared_finite_state_features",
                },
            )
            continuous_reports[objective_name] = {
                "checkpoint": str(baseline_dir / f"continuous_{objective_name}.pt"),
                "objectives": continuous.objectives,
                "coverages": continuous.coverages,
                "risk_values": continuous.risk_values,
                "fit_feasible": continuous.feasible,
                "dev_metrics": dev_metrics,
                "dev_feasible": dev_feasible,
                "final_loss": continuous.loss_history[-1] if continuous.loss_history else None,
            }
            arm_name = f"continuous__{objective_name}"
            factorial_actions[arm_name] = continuous.actions
            factorial_report[arm_name] = {
                "representation": "shared_continuous_bottleneck",
                "objective": objective_name,
                "contexts": {
                    name: {
                        "dev_metrics": dev_metrics[name],
                        "dev_feasible": dev_feasible[name],
                        "fit_feasible": continuous.feasible[name],
                        "fit_objective": continuous.objectives[name],
                        "fit_risk_value": continuous.risk_values[name],
                    }
                    for name in runtimes
                },
                "all_dev_feasible": all(dev_feasible.values()),
                "min_dev_yield": float(
                    min(metric.genuine_yield for metric in dev_metrics.values())
                ),
            }

    # Full-image neural challenge: same raw frozen embeddings, shared trunk and
    # matched mean/evidence objectives. This is the strongest direct comparator
    # to a hard partition and is kept separate from the finite-state continuous
    # baseline above.
    image_neural_reports: dict[str, dict] = {}
    image_neural_models: dict[str, object] = {}
    if cfg.run_image_neural_baseline:
        threshold_map = {name: runtime.threshold for name, runtime in runtimes.items()}
        alpha_map = {name: runtime.config.target_fmr for name, runtime in runtimes.items()}
        budget_map = {name: runtime.identity_budget for name, runtime in runtimes.items()}
        params = _context_params(runtimes)
        for objective_name, seed_offset in (("evidence", 1900), ("mean_risk", 2000)):
            result = train_multi_context_image_policy(
                fit_table.embeddings,
                fit_table.identity_ids,
                thresholds=threshold_map,
                alphas=alpha_map,
                identity_budgets=budget_map,
                context_params=params,
                objective=objective_name,
                hidden=tuple(cfg.image_policy_hidden),
                dropout=cfg.image_policy_dropout,
                epochs=cfg.image_policy_epochs,
                lr=cfg.image_policy_lr,
                max_identity_pairs=cfg.image_policy_max_identity_pairs,
                seed=cfg.seed + seed_offset,
                device=cfg.device,
            )
            image_neural_models[objective_name] = result.model
            dev_metrics = {}
            dev_feasible = {}
            for name, runtime in runtimes.items():
                gate_dev = predict_image_policy(
                    result.model,
                    dev_table.embeddings,
                    name,
                    device=cfg.device,
                )
                metrics = evaluate_one_sided(
                    dev_table.embeddings,
                    dev_table.identity_ids,
                    gate_dev,
                    runtime.threshold,
                )
                context = runtime.config
                dev_metrics[name] = metrics
                dev_feasible[name] = bool(
                    result.feasible[name]
                    and metrics.coverage >= context.coverage_low - 1e-8
                    and metrics.coverage <= context.coverage_high + 1e-8
                    and metrics.fmr <= context.target_fmr + 1e-12
                )
            _save_torch_checkpoint(
                result.model,
                baseline_dir / f"full_image_neural_{objective_name}.pt",
                {
                    "objective": objective_name,
                    "seed": cfg.seed + seed_offset,
                    "feature_scope": "full_frozen_face_embedding",
                },
            )
            image_neural_reports[objective_name] = {
                "checkpoint": str(baseline_dir / f"full_image_neural_{objective_name}.pt"),
                "fit_objectives": result.objectives,
                "fit_coverages": result.coverages,
                "fit_risk_values": result.risk_values,
                "fit_feasible": result.feasible,
                "dev_metrics": dev_metrics,
                "dev_feasible": dev_feasible,
                "final_loss": result.loss_history[-1] if result.loss_history else None,
            }
            factorial_report[f"full_image_neural__{objective_name}"] = {
                "representation": "full_image_embedding_neural_policy",
                "objective": objective_name,
                "contexts": {
                    name: {
                        "dev_metrics": dev_metrics[name],
                        "dev_feasible": dev_feasible[name],
                        "fit_feasible": result.feasible[name],
                        "fit_objective": result.objectives[name],
                        "fit_risk_value": result.risk_values[name],
                    }
                    for name in runtimes
                },
                "all_dev_feasible": all(dev_feasible.values()),
                "min_dev_yield": float(
                    min(metric.genuine_yield for metric in dev_metrics.values())
                ),
            }

    stage_times["strong_baselines"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()

    # Shared inductive router for the selected multi-context teacher partition.
    teacher_repr = selected.partition.labels[base_state.labels]
    teacher_dev = selected.partition.labels[state_dev]
    repr_weights = identity_balanced_sample_weights(repr_table.identity_ids)
    dev_weights = identity_balanced_sample_weights(dev_table.identity_ids)
    router_result = train_router(
        repr_table.embeddings,
        teacher_repr,
        selected.partition.k,
        validation_features=dev_table.embeddings,
        validation_labels=teacher_dev,
        epochs=cfg.router_epochs,
        batch_size=cfg.router_batch_size,
        lr=cfg.router_lr,
        seed=cfg.seed + 2000,
        device=cfg.device,
        sample_weights=repr_weights,
        validation_weights=dev_weights,
        hidden=tuple(cfg.router_hidden),
        dropout=cfg.router_dropout,
        patience=cfg.router_patience,
    )
    router_labels = {
        name: predict_router(router_result.model, split_tables[name].embeddings, device=cfg.device)
        for name in SPLIT_NAMES
    }

    deployed = {
        "router_training": {
            "train_accuracy": router_result.train_accuracy,
            "validation_accuracy": router_result.validation_accuracy,
            "best_epoch": router_result.best_epoch,
            "train_losses": router_result.train_losses,
            "validation_losses": router_result.validation_losses,
        },
        "contexts": {},
    }
    policy_payload = {
        "version": 1,
        "selected_partition": selected.name,
        "contexts": {},
    }
    selected_family_budget = (
        0.5 * cfg.certification_delta
        if cfg.certify_candidate_family
        else cfg.certification_delta
    )
    context_delta = selected_family_budget / max(1, len(runtimes))
    deployed_test_gates: dict[str, np.ndarray] = {}

    for offset, (name, runtime) in enumerate(runtimes.items()):
        ctx = runtime.config
        actual_moments, actual_diag = estimate_identity_cluster_moments(
            fit_table.embeddings,
            fit_table.identity_ids,
            router_labels["fit"],
            threshold=runtime.threshold,
            alpha=ctx.target_fmr,
            identity_budget=runtime.identity_budget,
            n_cells=selected.partition.k,
            max_pairs=cfg.max_fit_identity_pairs,
            seed=cfg.seed + 2300 + offset,
        )
        actual_solution = solve_policy(
            actual_moments,
            rho=ctx.rho,
            coverage_low=ctx.coverage_low,
            coverage_high=ctx.coverage_high,
            reserve=ctx.reserve,
        )
        dev_gate = actual_solution.beta[router_labels["dev"]]
        cal_gate = actual_solution.beta[router_labels["cal"]]
        test_gate = actual_solution.beta[router_labels["test"]]
        deployed_test_gates[name] = test_gate
        dev_metrics = evaluate_one_sided(
            dev_table.embeddings, dev_table.identity_ids, dev_gate, runtime.threshold
        )
        certificate = certify_one_sided(
            cal_table.embeddings,
            cal_table.identity_ids,
            cal_gate,
            threshold=runtime.threshold,
            target_fmr=ctx.target_fmr,
            coverage_target_low=ctx.coverage_low,
            coverage_target_high=ctx.coverage_high,
            delta=context_delta,
        )
        test_metrics = evaluate_one_sided(
            test_table.embeddings, test_table.identity_ids, test_gate, runtime.threshold
        )
        transfer = router_transfer_metrics(
            teacher_dev,
            router_labels["dev"],
            beta=selected.solutions[name].beta,
        )
        profile = decoded_profile(actual_moments, actual_solution)
        quality_score = context_quality_score(actual_moments, actual_solution, ctx.rho)
        deployed["contexts"][name] = {
            "threshold": runtime.threshold,
            "target_fmr": ctx.target_fmr,
            "coverage_low": ctx.coverage_low,
            "coverage_high": ctx.coverage_high,
            "identity_budget": runtime.identity_budget,
            "fit_solution": actual_solution,
            "fit_moments": {
                "p": actual_moments.p,
                "u": actual_moments.u,
                "ell": actual_moments.ell,
                "B_eigenvalues": np.linalg.eigvalsh(actual_moments.B),
                "n_fit_identities": actual_diag.n_identities,
                "n_pairs": actual_diag.n_pairs,
            },
            "router_transfer": transfer,
            "dev_metrics": dev_metrics,
            "certificate": certificate,
            "test_metrics": test_metrics,
        }
        policy_payload["contexts"][name] = {
            "threshold": runtime.threshold,
            "target_fmr": ctx.target_fmr,
            "coverage_low": ctx.coverage_low,
            "coverage_high": ctx.coverage_high,
            "identity_budget": runtime.identity_budget,
            "certificate_status": certificate.status,
            "beta": actual_solution.beta,
            "mu_g": profile["mu_g"],
            "mu_I": profile["mu_I"],
            "evidence_response": profile["evidence_response"],
            "quality_score": quality_score,
        }

    stage_times["router_refit_and_certification"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()

    # Frozen test table for every predeclared partition candidate and strong baseline.
    test_tables = {
        "partitions": {},
        "scalar": {"common": {}, "context_conditioned": {}},
        "factorial": {},
    }
    for candidate in candidates:
        entry = {}
        cell_test = candidate.partition.labels[state_test]
        for name, runtime in runtimes.items():
            gate = candidate.solutions[name].beta[cell_test]
            entry[name] = evaluate_one_sided(
                test_table.embeddings,
                test_table.identity_ids,
                gate,
                runtime.threshold,
            )
        test_tables["partitions"][candidate.name] = entry
    for family in ("common", "context_conditioned"):
        for name, runtime in runtimes.items():
            gate = scalar_actions[family][name][state_test]
            test_tables["scalar"][family][name] = evaluate_one_sided(
                test_table.embeddings, test_table.identity_ids, gate, runtime.threshold
            )
    for arm_name, actions_by_context in factorial_actions.items():
        test_tables["factorial"][arm_name] = {}
        for name, runtime in runtimes.items():
            gate = actions_by_context[name][state_test]
            test_tables["factorial"][arm_name][name] = evaluate_one_sided(
                test_table.embeddings,
                test_table.identity_ids,
                gate,
                runtime.threshold,
            )
    image_neural_test_gates: dict[str, dict[str, np.ndarray]] = {}
    for objective_name, model in image_neural_models.items():
        arm_name = f"full_image_neural__{objective_name}"
        test_tables["factorial"][arm_name] = {}
        image_neural_test_gates[objective_name] = {}
        for name, runtime in runtimes.items():
            gate = predict_image_policy(
                model,
                test_table.embeddings,
                name,
                device=cfg.device,
            )
            image_neural_test_gates[objective_name][name] = gate
            test_tables["factorial"][arm_name][name] = evaluate_one_sided(
                test_table.embeddings,
                test_table.identity_ids,
                gate,
                runtime.threshold,
            )

    paired_test_comparisons = {}
    if cfg.bootstrap_repetitions > 0:
        matched_mean_candidate = next(
            (
                candidate
                for candidate in candidates
                if candidate.family == "mean_only"
                and candidate.partition.k == selected.partition.k
            ),
            None,
        )
        for context_offset, (name, runtime) in enumerate(runtimes.items()):
            primary = identity_components(
                test_table.embeddings,
                test_table.identity_ids,
                deployed_test_gates[name],
                runtime.threshold,
            )
            comparator_gates: dict[str, np.ndarray] = {
                "scalar_common": scalar_actions["common"][name][state_test],
                "scalar_context_conditioned": scalar_actions["context_conditioned"][name][state_test],
            }
            if matched_mean_candidate is not None:
                comparator_gates["mean_only_matched_k"] = matched_mean_candidate.solutions[name].beta[
                    matched_mean_candidate.partition.labels[state_test]
                ]
            for arm_name in ("dense_state__evidence", "continuous__evidence"):
                if arm_name in factorial_actions:
                    comparator_gates[arm_name] = factorial_actions[arm_name][name][state_test]
            if "evidence" in image_neural_test_gates:
                comparator_gates["full_image_neural__evidence"] = image_neural_test_gates["evidence"][name]

            paired_test_comparisons[name] = {}
            for comparator_offset, (comparator_name, gate) in enumerate(comparator_gates.items()):
                comparator = identity_components(
                    test_table.embeddings,
                    test_table.identity_ids,
                    gate,
                    runtime.threshold,
                )
                all_bootstrap = paired_identity_multiplier_bootstrap_all(
                    primary,
                    comparator,
                    repetitions=cfg.bootstrap_repetitions,
                    seed=cfg.seed + 5000 + 100 * context_offset + comparator_offset,
                )
                paired_test_comparisons[name][comparator_name] = {
                    metric: all_bootstrap[metric]
                    for metric in ("genuine_yield", "coverage", "fmr", "retained_tar")
                    if metric in all_bootstrap
                }

    save_router(
        router_result.model,
        artifact_dir / "router.pt",
        metadata={
            "selected_partition": selected.name,
            "input": "l2_normalized_face_embedding",
            "seed": cfg.seed,
        },
    )
    save_json(policy_payload, artifact_dir / "policy.json")
    save_npz(
        artifact_dir / "state_space.npz",
        centroids=base_state.centroids,
        state_to_cell=selected.partition.labels,
        p=base_state.p,
    )

    stage_times["frozen_test_evaluation"] = time.perf_counter() - stage_start
    stage_times["total"] = time.perf_counter() - total_start

    report = {
        "scientific_status": {
            "implementation": "FULL_RESEARCH_SCAFFOLD",
            "empirical_claims_validated": False,
            "warning": (
                "Running this code does not establish the CBP hypotheses. "
                "Use independent identities, inspect all failed certificates, and apply the declared gates."
            ),
        },
        "config": cfg.to_dict(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "compute": {
            "stage_seconds": stage_times,
            "cuda_peak_memory_bytes": _cuda_peak_memory(),
        },
        "input_hashes": {
            "embedding_npz_sha256": sha256_file(cfg.embedding_npz),
            "config_sha256": sha256_file(config_path) if config_path is not None else None,
        },
        "split_report": split_report,
        "thresholds": threshold_info,
        "state_space": {
            "n_states": int(len(base_state.centroids)),
            "embedding_dim": int(base_state.centroids.shape[1]),
            "state_masses": base_state.p,
        },
        "phenomenon_E1": phenomenon,
        "dense_fit_diagnostics": dense_diagnostics,
        "candidates": [c.report() for c in candidates],
        "selection": {
            "selected": selected.name,
            "warning": selection_warning,
        },
        "candidate_family_certificates": candidate_family_certificates,
        "scalar_baselines": scalar_reports,
        "continuous_baselines": continuous_reports,
        "image_neural_baselines": image_neural_reports,
        "factorial_design": factorial_report,
        "deployed_cbp": deployed,
        "test_tables": test_tables,
        "paired_test_comparisons": paired_test_comparisons,
        "gates": _derive_gates(
            phenomenon,
            selected,
            deployed,
            factorial_report,
            candidates,
            cfg,
        ),
    }
    save_json(report, out_dir / "report.json")
    return jsonable(report)


def _derive_gates(
    phenomenon: dict,
    selected: CandidateEvaluation,
    deployed: dict,
    factorial_report: dict[str, object],
    candidates: list[CandidateEvaluation],
    cfg: ExperimentConfig,
) -> dict:
    # Gate A is an explicit effect-size screen, not an automatic significance
    # claim. The permutation null is descriptive because state estimates and
    # mean matching are themselves learned objects.
    phenomenon_alpha = 0.05 / max(1, len(phenomenon))
    phenomenon_contexts = {}
    for name, item in phenomenon.items():
        ok = bool(
            item.get("status") == "OK"
            and item.get("n_pairs", 0) >= cfg.phenomenon_min_pairs
            and item.get("mean_match_distance", float("inf"))
            <= cfg.phenomenon_max_mean_match_distance
            and item.get("evidence_excess_ratio_over_permuted", float("-inf"))
            >= cfg.phenomenon_min_gap_ratio
        )
        phenomenon_contexts[name] = {
            "pass": ok,
            "n_pairs": item.get("n_pairs"),
            "mean_match_distance": item.get("mean_match_distance"),
            "evidence_excess_ratio_over_permuted": item.get(
                "evidence_excess_ratio_over_permuted"
            ),
            "descriptive_permutation_p": item.get("permutation_p_value"),
        }
    phenomenon_ok = all(item["pass"] for item in phenomenon_contexts.values())

    # Gate B uses the actual deployed router after moment refitting and compares
    # its verification value with a matched-K mean-only representation. Merely
    # finding a feasible CBP policy is not evidence that evidence-aware
    # compression matters.
    matched_mean = next(
        (
            candidate
            for candidate in candidates
            if candidate.family == "mean_only"
            and candidate.partition.k == selected.partition.k
        ),
        None,
    )
    decision_contexts = {}
    for name, item in deployed["contexts"].items():
        metrics = item["dev_metrics"]
        deployed_feasible = bool(
            item["fit_solution"].feasible
            and metrics.coverage >= item["coverage_low"] - cfg.gate_numerical_tolerance
            and metrics.coverage <= item["coverage_high"] + cfg.gate_numerical_tolerance
            and metrics.fmr <= item["target_fmr"] + cfg.gate_numerical_tolerance
        )
        matched_metrics = None if matched_mean is None else matched_mean.dev_metrics[name]
        matched_feasible = bool(
            matched_mean is not None
            and matched_mean.fit_feasible
            and matched_mean.dev_feasible
        )
        if matched_metrics is None:
            yield_gain = None
            value_relevance = False
        elif not matched_feasible:
            # Reaching a feasible action where the matched mean-only partition
            # cannot is itself a decision-relevant distinction.
            yield_gain = None
            value_relevance = deployed_feasible
        else:
            yield_gain = float(metrics.genuine_yield - matched_metrics.genuine_yield)
            value_relevance = bool(
                yield_gain
                >= cfg.gate_decision_min_yield_gain - cfg.gate_numerical_tolerance
            )
        ok = bool(deployed_feasible and value_relevance)
        decision_contexts[name] = {
            "pass": ok,
            "deployed_feasible": deployed_feasible,
            "coverage": metrics.coverage,
            "fmr": metrics.fmr,
            "genuine_yield": metrics.genuine_yield,
            "matched_mean_only_candidate": None if matched_mean is None else matched_mean.name,
            "matched_mean_only_feasible": matched_feasible,
            "matched_mean_only_yield": (
                None if matched_metrics is None else matched_metrics.genuine_yield
            ),
            "yield_gain_over_matched_mean_only": yield_gain,
            "required_yield_gain": cfg.gate_decision_min_yield_gain,
        }
    decision_ok = all(item["pass"] for item in decision_contexts.values())

    hard = factorial_report.get("hard_cbp__evidence")
    comparator_names = (
        "dense_state__evidence",
        "continuous__evidence",
        "full_image_neural__evidence",
    )
    comparator_details = {}
    comparator_scores = []
    for name in comparator_names:
        item = factorial_report.get(name)
        if item is None:
            continue
        comparator_details[name] = {
            "all_dev_feasible": item["all_dev_feasible"],
            "min_dev_yield": item["min_dev_yield"],
        }
        if item["all_dev_feasible"]:
            comparator_scores.append(float(item["min_dev_yield"]))
    hard_score = None if hard is None else float(hard["min_dev_yield"])
    best_strong = max(comparator_scores) if comparator_scores else None
    strong_ok = bool(
        hard is not None
        and hard["all_dev_feasible"]
        and best_strong is not None
        and hard_score >= best_strong - cfg.gate_numerical_tolerance
    )
    strong_details = {
        "hard_cbp_all_dev_feasible": None if hard is None else hard["all_dev_feasible"],
        "hard_cbp_min_dev_yield": hard_score,
        "comparators": comparator_details,
        "best_feasible_comparator_yield": best_strong,
        "numerical_tolerance": cfg.gate_numerical_tolerance,
    }

    nonvacuity = all(
        item["certificate"].status == "PASS"
        for item in deployed["contexts"].values()
    )
    return {
        "A_phenomenon": {
            "pass": phenomenon_ok,
            "bonferroni_alpha_per_context_for_descriptive_p": phenomenon_alpha,
            "contexts": phenomenon_contexts,
            "screen_thresholds": {
                "min_pairs": cfg.phenomenon_min_pairs,
                "max_mean_match_distance": cfg.phenomenon_max_mean_match_distance,
                "min_excess_ratio_over_permuted": cfg.phenomenon_min_gap_ratio,
            },
            "note": (
                "Held-out effect-size screen only. The permutation p-value is "
                "descriptive; final support requires nuisance controls and "
                "cross-backbone replication."
            ),
        },
        "B_decision_relevance": {
            "pass": decision_ok,
            "contexts": decision_contexts,
            "teacher_candidate_feasible": selected.fit_feasible and selected.dev_feasible,
            "note": (
                "Uses the deployed router after refitting and a matched-K mean-only "
                "candidate. Feasibility plus the predeclared yield margin (or a "
                "feasibility advantage) is required; reduced variance alone is insufficient."
            ),
        },
        "C_strong_baseline": {
            "pass": strong_ok,
            "details": strong_details,
            "note": (
                "PASS only when hard CBP is dev-feasible and its worst-context "
                "yield is not below every feasible dense, shared-continuous or "
                "full-image neural same-evidence comparator. Failure narrows the "
                "hard-partition claim, not automatically the evidence principle."
            ),
        },
        "D_nonvacuity": {
            "pass": nonvacuity,
            "note": (
                "PASS requires independent calibration certificates in every "
                "primary context; failed certificates remain reported."
            ),
        },
    }
