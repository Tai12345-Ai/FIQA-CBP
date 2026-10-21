import numpy as np

import cbp.statistics.state_space as state_space_module
from cbp.statistics.state_space import build_state_space, state_operators


class _FakeMiniBatchKMeans:
    def __init__(self, n_clusters, **kwargs):
        assert n_clusters == 3
        self.cluster_centers_ = np.asarray(
            [
                [1.0, 0.0],
                [0.7, 0.7],  # deliberately empty cluster
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def fit_predict(self, z):
        assert len(z) == 4
        return np.asarray([0, 0, 2, 2], dtype=np.int64)


def test_build_state_space_removes_empty_kmeans_states(monkeypatch):
    monkeypatch.setattr(
        state_space_module,
        "MiniBatchKMeans",
        _FakeMiniBatchKMeans,
    )

    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    identity_ids = np.asarray(["a", "a", "b", "b"])

    state = build_state_space(
        embeddings,
        identity_ids,
        threshold=0.5,
        n_states=3,
        seed=0,
    )

    assert state.centroids.shape == (2, 2)
    assert np.array_equal(state.labels, np.asarray([0, 0, 1, 1]))
    assert np.all(state.p > 0)
    assert np.isclose(state.p.sum(), 1.0)

    operators = state_operators(state, alpha=0.01)
    assert np.all(operators.p > 0)
