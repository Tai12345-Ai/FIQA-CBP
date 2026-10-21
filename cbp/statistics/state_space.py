from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .finite_state import build_evidence_operators, hard_matcher_from_embeddings
from .targets import estimate_genuine_utility
from cbp.data.pairs import identity_balanced_sample_weights


@dataclass
class StateSpace:
    centroids: np.ndarray
    labels: np.ndarray
    p: np.ndarray
    g: np.ndarray
    matcher: np.ndarray

    def assign(self, embeddings: np.ndarray) -> np.ndarray:
        z = np.asarray(embeddings, dtype=np.float64)
        z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        c = self.centroids / np.maximum(np.linalg.norm(self.centroids, axis=1, keepdims=True), 1e-12)
        return np.argmax(z @ c.T, axis=1)


def build_state_space(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    threshold: float,
    n_states: int = 128,
    seed: int = 0,
) -> StateSpace:
    z = np.asarray(embeddings, dtype=np.float64)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    n_states = min(int(n_states), len(z))
    km = MiniBatchKMeans(
        n_clusters=n_states,
        random_state=seed,
        batch_size=min(4096, max(256, len(z))),
        n_init=10,
    )
    labels = km.fit_predict(z)
    centroids = km.cluster_centers_

    # MiniBatchKMeans can leave one or more cluster centers with no assigned
    # samples. Those states have exactly zero empirical probability and must
    # not enter the finite-state operators, whose weighted adjoint requires
    # strictly positive state mass. Remove empty states and remap labels to a
    # contiguous range before estimating moments.
    occupied = np.unique(labels)
    if len(occupied) < n_states:
        remap = np.full(n_states, -1, dtype=np.int64)
        remap[occupied] = np.arange(len(occupied), dtype=np.int64)
        labels = remap[labels]
        centroids = centroids[occupied]
        n_states = len(occupied)

    weights = identity_balanced_sample_weights(identity_ids)
    p = np.bincount(labels, weights=weights, minlength=n_states).astype(np.float64)
    p = p / p.sum()
    g_sample = estimate_genuine_utility(z, identity_ids, threshold)
    g_num = np.bincount(labels, weights=weights * g_sample, minlength=n_states)
    g = np.divide(g_num, p, out=np.zeros_like(g_num), where=p > 0)
    matcher = hard_matcher_from_embeddings(centroids, threshold)
    return StateSpace(centroids, labels, p, g, matcher)


def state_operators(state_space: StateSpace, alpha: float):
    return build_evidence_operators(state_space.matcher, alpha=alpha, p=state_space.p)
