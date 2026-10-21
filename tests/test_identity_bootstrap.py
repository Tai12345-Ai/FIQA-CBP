import numpy as np

from cbp.metrics.identity import (
    IdentityComponents,
    paired_identity_multiplier_bootstrap_all,
)


def make_components(offset: float = 0.0) -> IdentityComponents:
    identity_ids = np.asarray([f"id{i}" for i in range(8)])
    coverage = np.linspace(0.4, 0.9, 8)
    genuine = np.linspace(0.2, 0.7, 8) + offset
    rng = np.random.default_rng(3)
    H = rng.uniform(0.0, 0.1, size=(8, 8))
    H = 0.5 * (H + H.T)
    np.fill_diagonal(H, 0.0)
    return IdentityComponents(identity_ids, coverage, genuine, H)


def test_paired_multiplier_bootstrap_is_reproducible_and_paired():
    a = make_components(offset=0.05)
    b = make_components(offset=0.00)
    first = paired_identity_multiplier_bootstrap_all(a, b, repetitions=200, seed=9)
    second = paired_identity_multiplier_bootstrap_all(a, b, repetitions=200, seed=9)
    assert first == second
    assert np.isclose(first["genuine_yield"]["point_difference"], 0.05)
    assert first["genuine_yield"]["ci_low"] > 0.049999
    assert first["genuine_yield"]["ci_high"] < 0.050001
    assert first["fmr"]["point_difference"] == 0.0
