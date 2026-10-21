import numpy as np

from cbp.analysis.phenomenon import mean_matched_evidence_test


def test_mean_matched_evidence_test_runs_and_is_reproducible():
    rng = np.random.default_rng(8)
    n = 30
    g = np.repeat(np.linspace(0.1, 0.9, n // 2), 2)
    r = np.repeat(np.linspace(-0.05, 0.05, n // 2), 2)
    # Adjacent mean-matched states deliberately receive opposite evidence codes.
    evidence = np.column_stack([
        np.tile([0.0, 4.0], n // 2),
        rng.normal(scale=0.01, size=n),
    ])
    a = mean_matched_evidence_test(g, r, evidence, permutations=200, seed=9)
    b = mean_matched_evidence_test(g, r, evidence, permutations=200, seed=9)
    assert a.to_dict() == b.to_dict()
    assert a.n_pairs > 0
    assert 0 <= a.permutation_p_value <= 1
