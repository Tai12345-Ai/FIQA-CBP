from itertools import product
import numpy as np

from cbp.statistics.finite_state import build_evidence_operators, weighted_adjoint
from cbp.synthetic.simulator import empirical_u_statistic


def test_weighted_adjoint_identity():
    rng = np.random.default_rng(1)
    p = np.array([0.2, 0.3, 0.5])
    M = rng.normal(size=(3, 3))
    u = rng.normal(size=3)
    v = rng.normal(size=3)
    adj = weighted_adjoint(M, p)
    lhs = np.sum(p * u * (M @ v))
    rhs = np.sum(p * (adj @ u) * v)
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_q_variance_matches_exact_enumeration():
    p = np.array([0.2, 0.3, 0.5])
    matcher = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    alpha = 0.15
    action = np.array([0.2, 0.8, 0.4])
    n = 3
    operators = build_evidence_operators(matcher, alpha, p)
    predicted = operators.variance(action, n)

    values = []
    weights = []
    for states in product(range(3), repeat=n):
        states = np.asarray(states, dtype=int)
        values.append(empirical_u_statistic(states, action, operators.H))
        weights.append(float(np.prod(p[states])))
    values = np.asarray(values)
    weights = np.asarray(weights)
    mean = np.sum(weights * values)
    exact = np.sum(weights * (values - mean) ** 2)
    assert np.allclose(predicted, exact, atol=1e-12, rtol=1e-10)


def test_q_is_weighted_psd():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(8, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    matcher = (x @ x.T >= 0.2).astype(float)
    p = rng.random(8)
    p /= p.sum()
    operators = build_evidence_operators(matcher, 0.05, p)
    q = operators.q(20)
    for _ in range(20):
        a = rng.normal(size=8)
        assert np.sum(p * a * (q @ a)) >= -1e-10
