from __future__ import annotations

import numpy as np
from cbp.statistics.moments import CellMoments
from .solver import PolicySolution


def decoded_profile(moments: CellMoments, solution: PolicySolution) -> dict[str, np.ndarray]:
    return {
        "mu_g": moments.mu_g.copy(),
        "mu_I": moments.mu_I.copy(),
        "evidence_response": moments.evidence_response(solution.beta),
        "beta": solution.beta.copy(),
    }


def context_quality_score(moments: CellMoments, solution: PolicySolution, rho: float) -> np.ndarray:
    if solution.risk_dual is None:
        return np.full_like(moments.p, np.nan, dtype=float)
    lam = solution.risk_dual
    nu_plus = solution.coverage_high_dual or 0.0
    nu_minus = solution.coverage_low_dual or 0.0
    nu = nu_plus - nu_minus
    beta = solution.beta
    s = float(np.sqrt(max(beta @ moments.B @ beta, 0.0)))
    if s <= 1e-12:
        return np.full_like(moments.p, np.nan, dtype=float)
    evidence = (moments.B @ beta) / np.maximum(moments.p * s, 1e-15)
    return moments.mu_g - lam * ((moments.mu_I - moments.alpha) + rho * evidence) - nu
