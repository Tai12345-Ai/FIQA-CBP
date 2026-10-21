import numpy as np

from cbp.synthetic.faces import make_synthetic_face_embeddings
from cbp.statistics.identity_kernel import estimate_identity_cluster_moments


def test_identity_cluster_moments_are_psd_and_well_shaped():
    data = make_synthetic_face_embeddings(
        n_identities=20,
        captures_per_identity=3,
        embedding_dim=16,
        seed=11,
    )
    z = data.table.embeddings
    ids = data.table.identity_ids
    labels = np.arange(len(z)) % 4
    moments, diagnostics = estimate_identity_cluster_moments(
        z,
        ids,
        labels,
        threshold=0.4,
        alpha=0.05,
        identity_budget=20,
        n_cells=4,
    )
    assert moments.p.shape == (4,)
    assert moments.B.shape == (4, 4)
    assert np.isclose(moments.p.sum(), 1.0)
    assert np.min(np.linalg.eigvalsh(moments.B)) >= -1e-10
    assert diagnostics.n_identities == 20
    assert diagnostics.n_pairs == 190


def test_pair_subsampling_needs_partner_coverage():
    data = make_synthetic_face_embeddings(10, 2, 8, seed=2)
    labels = np.arange(len(data.table.embeddings)) % 2
    try:
        estimate_identity_cluster_moments(
            data.table.embeddings,
            data.table.identity_ids,
            labels,
            threshold=0.3,
            alpha=0.1,
            identity_budget=10,
            n_cells=2,
            max_pairs=1,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected incomplete-partner diagnostic")
