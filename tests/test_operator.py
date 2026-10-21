import numpy as np
from cbp.synthetic.models import cycle_hard_matcher_problem
from cbp.statistics.finite_state import build_evidence_operators, weighted_adjoint


def test_q_is_weighted_self_adjoint_and_psd():
    problem = cycle_hard_matcher_problem(12)
    ops = build_evidence_operators(problem.matcher, problem.alpha, problem.p)
    q = ops.q(40)
    q_star = weighted_adjoint(q, problem.p)
    assert np.allclose(q, q_star, atol=1e-9)
    rng = np.random.default_rng(0)
    for _ in range(100):
        a = rng.normal(size=len(problem.p))
        value = np.sum(problem.p * a * (q @ a))
        assert value >= -1e-9


def test_contrast_variance_nonnegative():
    problem = cycle_hard_matcher_problem(12)
    ops = build_evidence_operators(problem.matcher, problem.alpha, problem.p)
    a = np.linspace(0, 1, 12)
    assert ops.variance(a, 20) >= 0
