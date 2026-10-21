"""Compatibility API for shared finite-state context models.

The canonical experiment runner constructs one identity-balanced observable
state space and then derives context-specific matcher operators on that common
space. This module exposes the same operation for external scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from cbp.data.pairs import identity_balanced_sample_weights
from cbp.representation.partitions import Partition, evidence_features, feature_partition
from .finite_state import EvidenceOperators, build_evidence_operators, hard_matcher_from_embeddings
from .targets import estimate_genuine_utility


@dataclass
class ContextStateModel:
    name: str
    threshold: float
    alpha: float
    g: np.ndarray
    operators: EvidenceOperators


@dataclass
class SharedStateSpace:
    centroids: np.ndarray
    sample_state_labels: np.ndarray
    p: np.ndarray
    contexts: dict[str, ContextStateModel]

    def assign(self, embeddings: np.ndarray) -> np.ndarray:
        z = np.asarray(embeddings, dtype=np.float64)
        z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        c = np.asarray(self.centroids, dtype=np.float64)
        c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
        return np.argmax(z @ c.T, axis=1)


def build_shared_state_space(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    context_thresholds: dict[str, float],
    context_alphas: dict[str, float],
    n_states: int = 128,
    seed: int = 0,
) -> SharedStateSpace:
    """Build one shared state space with identity-balanced empirical mass."""
    if set(context_thresholds) != set(context_alphas):
        raise ValueError("context_thresholds and context_alphas must share keys")
    z = np.asarray(embeddings, dtype=np.float64)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    ids = np.asarray(identity_ids).astype(str)
    n_states = min(int(n_states), len(z))
    km = MiniBatchKMeans(
        n_clusters=n_states,
        random_state=seed,
        batch_size=min(4096, max(256, len(z))),
        n_init=10,
    )
    labels = km.fit_predict(z)
    centroids = km.cluster_centers_
    weights = identity_balanced_sample_weights(ids)
    p = np.bincount(labels, weights=weights, minlength=n_states).astype(float)
    p /= p.sum()

    contexts: dict[str, ContextStateModel] = {}
    for name, threshold in context_thresholds.items():
        g_sample = estimate_genuine_utility(z, ids, threshold)
        g_num = np.bincount(labels, weights=weights * g_sample, minlength=n_states)
        g = np.divide(g_num, p, out=np.zeros_like(g_num), where=p > 0)
        matcher = hard_matcher_from_embeddings(centroids, threshold)
        alpha = float(context_alphas[name])
        operators = build_evidence_operators(matcher, alpha=alpha, p=p)
        contexts[name] = ContextStateModel(
            name=name,
            threshold=float(threshold),
            alpha=alpha,
            g=g,
            operators=operators,
        )
    return SharedStateSpace(
        centroids=centroids,
        sample_state_labels=labels,
        p=p,
        contexts=contexts,
    )


def shared_evidence_partition(
    state: SharedStateSpace,
    identity_budgets: dict[str, int],
    k: int,
    seed: int = 0,
    spectral_dim: int = 4,
) -> Partition:
    if set(identity_budgets) != set(state.contexts):
        raise ValueError("identity_budgets must cover every context")
    blocks = []
    for name in sorted(state.contexts):
        context = state.contexts[name]
        blocks.append(
            evidence_features(
                context.g,
                context.operators,
                identity_budgets[name],
                spectral_dim=spectral_dim,
            )
        )
    return feature_partition(
        np.concatenate(blocks, axis=1),
        k=k,
        seed=seed,
        name="cbp_evidence_shared",
    )
