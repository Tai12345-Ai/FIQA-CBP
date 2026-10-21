from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cbp.statistics.finite_state import EvidenceOperators, weighted_inner
from cbp.statistics.moments import CellMoments


@dataclass
class RankedPolicyResult:
    action: np.ndarray
    threshold: float
    feasible: bool
    objective: float
    coverage: float
    risk_value: float
    rank_index: int


def solve_ranked_threshold_policy(
    score: np.ndarray,
    g: np.ndarray,
    operators: EvidenceOperators,
    identity_budget: int,
    rho: float,
    coverage_low: float,
    coverage_high: float,
    reserve: float = 0.0,
    descending: bool = True,
) -> RankedPolicyResult:
    """Best nested threshold policy induced by one frozen scalar score.

    Ties are kept together. Therefore the returned family is genuinely nested
    and does not silently break ties using another feature.
    """
    score = np.asarray(score, dtype=float)
    g = np.asarray(g, dtype=float)
    if score.shape != g.shape or score.shape != operators.p.shape:
        raise ValueError("score/g/operator state space must align")

    unique = np.unique(score)
    thresholds = unique[::-1] if descending else unique
    candidates: list[np.ndarray] = []
    for t in thresholds:
        action = (score >= t).astype(float) if descending else (score <= t).astype(float)
        candidates.append(action)
    # Include reject-all and keep-all for diagnostics; coverage constraints will
    # decide whether either is admissible.
    candidates.extend([np.zeros_like(score), np.ones_like(score)])

    best = None
    q = operators.q(identity_budget)
    for idx, action in enumerate(candidates):
        coverage = weighted_inner(action, np.ones_like(action), operators.p)
        if coverage < coverage_low - 1e-12 or coverage > coverage_high + 1e-12:
            continue
        utility = weighted_inner(action, g, operators.p)
        excess = weighted_inner(action, operators.r, operators.p)
        std = float(np.sqrt(max(weighted_inner(action, q @ action, operators.p), 0.0)))
        risk = excess + rho * std + reserve
        feasible = risk <= 1e-12
        if not feasible:
            continue
        if best is None or utility > best.objective:
            threshold = float(thresholds[idx]) if idx < len(thresholds) else (
                float("inf") if idx == len(thresholds) else float("-inf")
            )
            best = RankedPolicyResult(
                action=action,
                threshold=threshold,
                feasible=True,
                objective=float(utility),
                coverage=float(coverage),
                risk_value=float(risk),
                rank_index=idx,
            )

    if best is not None:
        return best
    return RankedPolicyResult(
        action=np.zeros_like(score),
        threshold=float("nan"),
        feasible=False,
        objective=0.0,
        coverage=0.0,
        risk_value=float("inf"),
        rank_index=-1,
    )


def solve_ranked_cell_policy(
    score: np.ndarray,
    moments: CellMoments,
    rho: float,
    coverage_low: float,
    coverage_high: float,
    reserve: float = 0.0,
    descending: bool = True,
) -> RankedPolicyResult:
    """Best nested threshold policy under fitted identity-cluster moments."""
    score = np.asarray(score, dtype=float)
    if score.shape != moments.p.shape:
        raise ValueError("score must have one value per moment cell")
    unique = np.unique(score)
    thresholds = unique[::-1] if descending else unique
    best = None
    for idx, t in enumerate(thresholds):
        action = (score >= t).astype(float) if descending else (score <= t).astype(float)
        coverage = float(moments.p @ action)
        if coverage < coverage_low - 1e-12 or coverage > coverage_high + 1e-12:
            continue
        variance = float(max(action @ moments.B @ action, 0.0))
        risk = float(moments.ell @ action + rho * np.sqrt(variance) + reserve)
        if risk > 1e-12:
            continue
        utility = float(moments.u @ action)
        result = RankedPolicyResult(
            action=action,
            threshold=float(t),
            feasible=True,
            objective=utility,
            coverage=coverage,
            risk_value=risk,
            rank_index=idx,
        )
        if best is None or result.objective > best.objective:
            best = result
    if best is not None:
        return best
    return RankedPolicyResult(
        action=np.zeros_like(score),
        threshold=float("nan"),
        feasible=False,
        objective=0.0,
        coverage=0.0,
        risk_value=float("inf"),
        rank_index=-1,
    )
