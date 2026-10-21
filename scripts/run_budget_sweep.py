from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from cbp.config import ExperimentConfig
from cbp.fr.embeddings import EmbeddingTable
from cbp.policy.solver import solve_policy
from cbp.router.train import load_router, predict_router
from cbp.statistics.identity_kernel import estimate_identity_cluster_moments
from cbp.statistics.moments import CellMoments
from cbp.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep the planned independent-identity budget n")
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--run-dir", default="outputs/full_arcface")
    parser.add_argument("--budgets", default="25,50,100,200,500,1000")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    run_dir = Path(args.run_dir)
    table = EmbeddingTable.load(cfg.embedding_npz).normalized()
    split_df = pd.read_csv(run_dir / "split_manifest.csv")
    with open(run_dir / "report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    router, _ = load_router(run_dir / "artifact" / "router.pt")

    fit_ids = set(split_df.loc[split_df["split"] == "fit", "sample_id"].astype(str))
    fit_idx = np.flatnonzero(np.array([x in fit_ids for x in table.sample_ids]))
    if len(fit_idx) == 0:
        raise RuntimeError("could not align fit sample_ids with embedding table")
    fit = table.subset(fit_idx)
    labels = predict_router(router, fit.embeddings)
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]

    output = {}
    for offset, ctx in enumerate(cfg.contexts):
        threshold = float(report["thresholds"][ctx.name]["threshold"])
        rows = []
        for n in budgets:
            moments, diagnostics = estimate_identity_cluster_moments(
                fit.embeddings,
                fit.identity_ids,
                labels,
                threshold=threshold,
                alpha=ctx.target_fmr,
                identity_budget=n,
                n_cells=router.n_cells,
                max_pairs=cfg.max_fit_identity_pairs,
                seed=cfg.seed + offset,
            )
            variants = {
                "full": moments.B,
                "first_order_only": (4.0 / n) * diagnostics.zeta1,
                "degenerate_only": (2.0 / (n * (n - 1))) * diagnostics.zeta2,
                "mean_risk_only": np.zeros_like(moments.B),
            }
            row = {
                "identity_budget": n,
                "fit_identities": diagnostics.n_identities,
                "fit_pairs": diagnostics.n_pairs,
                "variants": {},
            }
            for variant, B in variants.items():
                variant_moments = CellMoments(
                    p=moments.p,
                    u=moments.u,
                    ell=moments.ell,
                    B=0.5 * (B + B.T),
                    mu_g=moments.mu_g,
                    mu_I=moments.mu_I,
                    alpha=moments.alpha,
                    identity_budget=n,
                    counts=moments.counts,
                    n_fit_identities=moments.n_fit_identities,
                    pair_count=moments.pair_count,
                )
                rho = 0.0 if variant == "mean_risk_only" else ctx.rho
                solution = solve_policy(
                    variant_moments,
                    rho=rho,
                    coverage_low=ctx.coverage_low,
                    coverage_high=ctx.coverage_high,
                    reserve=ctx.reserve,
                )
                row["variants"][variant] = {
                    "objective": solution.objective,
                    "coverage": solution.coverage,
                    "risk_value": solution.risk_value,
                    "feasible": solution.feasible,
                    "status": solution.status,
                }
            rows.append(row)
        output[ctx.name] = rows

    save_json(output, run_dir / "budget_sweep.json")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
