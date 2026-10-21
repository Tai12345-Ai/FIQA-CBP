from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from cbp.fr.embeddings import EmbeddingTable
from cbp.metrics.identity import evaluate_one_sided
from cbp.statistics.identity_kernel import estimate_identity_action_statistics


@dataclass
class ExternalScorePolicy:
    threshold: float
    higher_is_better: bool
    fit_statistics: object
    dev_metrics: object
    feasible: bool

    def gate(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        return (
            (scores >= self.threshold).astype(float)
            if self.higher_is_better
            else (scores <= self.threshold).astype(float)
        )


def load_external_scores(
    csv_path: str | Path,
    sample_ids: np.ndarray,
    score_column: str = "score",
) -> np.ndarray:
    frame = pd.read_csv(csv_path)
    required = {"sample_id", score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"external score CSV missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("external score CSV contains duplicated sample_id")
    mapping = dict(zip(frame["sample_id"].astype(str), frame[score_column].astype(float)))
    missing_ids = [str(x) for x in sample_ids if str(x) not in mapping]
    if missing_ids:
        raise ValueError(f"missing external scores for {len(missing_ids)} samples; first={missing_ids[:5]}")
    values = np.asarray([mapping[str(x)] for x in sample_ids], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("external scores must be finite")
    return values


def fit_external_score_policy(
    fit_table: EmbeddingTable,
    fit_scores: np.ndarray,
    dev_table: EmbeddingTable,
    dev_scores: np.ndarray,
    matcher_threshold: float,
    target_fmr: float,
    identity_budget: int,
    rho: float,
    coverage_low: float,
    coverage_high: float,
    reserve: float = 0.0,
    higher_is_better: bool = True,
    max_candidates: int = 200,
) -> ExternalScorePolicy:
    """Select one frozen external-FIQA threshold on dev.

    Candidate thresholds are deterministic fit-score quantiles. Fit feasibility
    uses identity-cluster plug-in moments; final validity still requires an
    independent calibration certificate.
    """
    fit_scores = np.asarray(fit_scores, dtype=float)
    dev_scores = np.asarray(dev_scores, dtype=float)
    quantiles = np.linspace(0.0, 1.0, min(max_candidates, len(fit_scores)))
    thresholds = np.unique(np.quantile(fit_scores, quantiles))
    best = None
    for threshold in thresholds:
        gate_fit = (
            (fit_scores >= threshold).astype(float)
            if higher_is_better
            else (fit_scores <= threshold).astype(float)
        )
        statistics = estimate_identity_action_statistics(
            fit_table.embeddings,
            fit_table.identity_ids,
            gate_fit,
            threshold=matcher_threshold,
            alpha=target_fmr,
            identity_budget=identity_budget,
        )
        risk_value = statistics.excess_risk + rho * statistics.evidence_std + reserve
        fit_feasible = (
            risk_value <= 1e-12
            and statistics.coverage >= coverage_low - 1e-12
            and statistics.coverage <= coverage_high + 1e-12
        )
        if not fit_feasible:
            continue
        gate_dev = (
            (dev_scores >= threshold).astype(float)
            if higher_is_better
            else (dev_scores <= threshold).astype(float)
        )
        metrics = evaluate_one_sided(
            dev_table.embeddings,
            dev_table.identity_ids,
            gate_dev,
            matcher_threshold,
        )
        dev_feasible = (
            metrics.coverage >= coverage_low - 1e-8
            and metrics.coverage <= coverage_high + 1e-8
            and metrics.fmr <= target_fmr + 1e-12
        )
        result = ExternalScorePolicy(
            threshold=float(threshold),
            higher_is_better=higher_is_better,
            fit_statistics=statistics,
            dev_metrics=metrics,
            feasible=bool(dev_feasible),
        )
        if dev_feasible and (best is None or metrics.genuine_yield > best.dev_metrics.genuine_yield):
            best = result
    if best is not None:
        return best
    return ExternalScorePolicy(
        threshold=float("nan"),
        higher_is_better=higher_is_better,
        fit_statistics=None,
        dev_metrics=None,
        feasible=False,
    )
