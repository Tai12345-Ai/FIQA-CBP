import numpy as np

from cbp.policy.solver import solve_policy
from cbp.statistics.moments import CellMoments, aggregate_cell_moments


def make_moments():
    p = np.array([0.4, 0.35, 0.25])
    u = np.array([0.36, 0.20, 0.05])
    ell = np.array([-0.04, -0.01, 0.08])
    B = np.diag([1e-4, 2e-4, 3e-4])
    return CellMoments(
        p=p,
        u=u,
        ell=ell,
        B=B,
        mu_g=u / p,
        mu_I=0.01 + ell / p,
        alpha=0.01,
        identity_budget=100,
        counts=np.array([40, 35, 25]),
    )


def test_policy_solver_returns_valid_box_and_constraints():
    moments = make_moments()
    solution = solve_policy(
        moments,
        rho=1.0,
        coverage_low=0.3,
        coverage_high=0.8,
        reserve=0.0,
        prefer_cvxpy=False,
    )
    assert solution.feasible
    assert np.all(solution.beta >= -1e-9)
    assert np.all(solution.beta <= 1 + 1e-9)
    assert 0.3 - 1e-6 <= solution.coverage <= 0.8 + 1e-6
    assert solution.risk_value <= 1e-6


def test_moment_aggregation_preserves_quadratic_form():
    dense = make_moments()
    mapping = np.array([0, 0, 1])
    coarse = aggregate_cell_moments(dense, mapping)
    beta = np.array([0.3, 0.7])
    lifted = beta[mapping]
    assert np.allclose(coarse.p @ beta, dense.p @ lifted)
    assert np.allclose(coarse.u @ beta, dense.u @ lifted)
    assert np.allclose(coarse.ell @ beta, dense.ell @ lifted)
    assert np.allclose(beta @ coarse.B @ beta, lifted @ dense.B @ lifted)
