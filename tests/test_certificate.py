import numpy as np

from cbp.certification.certificate import certify_one_sided
from cbp.synthetic.faces import make_synthetic_face_embeddings


def test_certificate_returns_declared_status_and_bounds():
    data = make_synthetic_face_embeddings(30, 3, 16, seed=4)
    gate = np.ones(len(data.table.embeddings), dtype=float)
    result = certify_one_sided(
        data.table.embeddings,
        data.table.identity_ids,
        gate,
        threshold=0.7,
        target_fmr=0.2,
        coverage_target_low=0.1,
        coverage_target_high=1.0,
        delta=0.1,
    )
    assert result.status in {"PASS", "INSUFFICIENT_EVIDENCE"}
    assert 0 <= result.coverage_low <= result.coverage_high <= 1
    assert result.impostor_numerator_up >= result.impostor_numerator_hat
    assert result.genuine_yield_low <= result.genuine_yield_hat


def test_zero_gate_fails_positive_coverage_requirement():
    data = make_synthetic_face_embeddings(20, 2, 8, seed=5)
    gate = np.zeros(len(data.table.embeddings), dtype=float)
    result = certify_one_sided(
        data.table.embeddings,
        data.table.identity_ids,
        gate,
        threshold=0.5,
        target_fmr=0.5,
        coverage_target_low=0.1,
        coverage_target_high=1.0,
        delta=0.1,
    )
    assert result.status == "INSUFFICIENT_EVIDENCE"
