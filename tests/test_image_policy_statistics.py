import numpy as np
import torch

from cbp.baselines.image_policy import build_image_context_statistics, _objective_terms
from cbp.statistics.identity_kernel import estimate_identity_action_statistics
from cbp.synthetic.faces import make_synthetic_face_embeddings


def test_sparse_image_statistics_match_direct_action_estimator():
    data = make_synthetic_face_embeddings(12, 3, 12, seed=21)
    z = data.table.embeddings
    ids = data.table.identity_ids
    action = np.linspace(0.1, 0.9, len(z))
    threshold = 0.45
    alpha = 0.10
    budget = 12

    sparse = build_image_context_statistics(
        z,
        ids,
        threshold=threshold,
        alpha=alpha,
        identity_budget=budget,
        device="cpu",
    )
    utility, coverage, theta, variance = _objective_terms(
        torch.as_tensor(action, dtype=torch.float64),
        sparse,
    )
    direct = estimate_identity_action_statistics(
        z,
        ids,
        action,
        threshold=threshold,
        alpha=alpha,
        identity_budget=budget,
    )
    assert np.allclose(float(coverage), direct.coverage, atol=1e-10)
    assert np.allclose(float(utility), direct.genuine_yield, atol=1e-10)
    assert np.allclose(float(theta), direct.excess_risk, atol=1e-10)
    assert np.allclose(float(variance), direct.evidence_variance, atol=1e-10)
