import numpy as np

from cbp.synthetic.models import cycle_hard_matcher_problem
from cbp.statistics.finite_state import build_evidence_operators
from cbp.statistics.moments import estimate_cell_moments
from cbp.policy.solver import solve_policy


def test_cell_moments_and_solver_shapes():
    problem = cycle_hard_matcher_problem(12)
    ops = build_evidence_operators(problem.matcher, problem.alpha, problem.p)
    labels = np.repeat(np.arange(4), 3)
    moments = estimate_cell_moments(labels, problem.g, ops, identity_budget=30, k=4)
    assert moments.B.shape == (4, 4)
    assert np.all(moments.p > 0)
    sol = solve_policy(
        moments,
        rho=0.1,
        coverage_low=0.1,
        coverage_high=1.0,
        reserve=-0.2,
        prefer_cvxpy=False,
    )
    assert sol.beta.shape == (4,)
    assert np.all((sol.beta >= 0) & (sol.beta <= 1))
