from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from cbp.statistics.moments import CellMoments


@dataclass
class PolicySolution:
    beta: np.ndarray
    objective: float
    feasible: bool
    risk_value: float
    coverage: float
    status: str
    risk_dual: float | None = None
    coverage_low_dual: float | None = None
    coverage_high_dual: float | None = None


def _sqrt_psd(B: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(0.5 * (B + B.T))
    return (evecs * np.sqrt(np.maximum(evals, 0.0))) @ evecs.T


def solve_policy(
    moments: CellMoments,
    rho: float,
    coverage_low: float,
    coverage_high: float,
    reserve: float = 0.0,
    robust_B_epsilon: float = 0.0,
    prefer_cvxpy: bool = True,
) -> PolicySolution:
    p, u, ell, B = moments.p, moments.u, moments.ell, moments.B
    k = len(p)
    B_eff = B + float(robust_B_epsilon) * np.eye(k)
    unsupported = p <= 0

    if prefer_cvxpy:
        try:
            import cvxpy as cp
            beta = cp.Variable(k)
            S = _sqrt_psd(B_eff)
            risk = ell @ beta + rho * cp.norm(S @ beta, 2) + reserve
            constraints = [
                beta >= 0,
                beta <= 1,
                risk <= 0,
                p @ beta >= coverage_low,
                p @ beta <= coverage_high,
            ]
            if np.any(unsupported):
                constraints.append(beta[unsupported] == 0)
            problem = cp.Problem(cp.Maximize(u @ beta), constraints)
            problem.solve(verbose=False)
            if beta.value is not None:
                b = np.clip(np.asarray(beta.value).reshape(-1), 0, 1)
                risk_value = float(ell @ b + rho * np.sqrt(max(b @ B_eff @ b, 0.0)) + reserve)
                cov = float(p @ b)
                feasible = risk_value <= 1e-6 and cov >= coverage_low - 1e-6 and cov <= coverage_high + 1e-6
                return PolicySolution(
                    beta=b,
                    objective=float(u @ b),
                    feasible=feasible,
                    risk_value=risk_value,
                    coverage=cov,
                    status=str(problem.status),
                    risk_dual=_dual(constraints[2]),
                    coverage_low_dual=_dual(constraints[3]),
                    coverage_high_dual=_dual(constraints[4]),
                )
        except Exception:
            pass

    def std(b):
        return float(np.sqrt(max(b @ B_eff @ b, 0.0)))

    cons = [
        {"type": "ineq", "fun": lambda b: -(ell @ b + rho * std(b) + reserve)},
        {"type": "ineq", "fun": lambda b: p @ b - coverage_low},
        {"type": "ineq", "fun": lambda b: coverage_high - p @ b},
    ]
    x0 = np.full(k, min(coverage_high, max(coverage_low, 0.5)))
    x0[unsupported] = 0.0
    bounds = [(0.0, 0.0) if unsupported[j] else (0.0, 1.0) for j in range(k)]
    res = minimize(
        lambda b: -float(u @ b),
        x0=x0,
        bounds=bounds,
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    b = np.clip(res.x, 0, 1)
    risk_value = float(ell @ b + rho * std(b) + reserve)
    cov = float(p @ b)
    feasible = res.success and risk_value <= 1e-6 and cov >= coverage_low - 1e-6 and cov <= coverage_high + 1e-6
    return PolicySolution(
        beta=b,
        objective=float(u @ b),
        feasible=feasible,
        risk_value=risk_value,
        coverage=cov,
        status=f"SLSQP:{res.message}",
    )


def _dual(constraint) -> float | None:
    try:
        value = constraint.dual_value
        return None if value is None else float(np.asarray(value))
    except Exception:
        return None
