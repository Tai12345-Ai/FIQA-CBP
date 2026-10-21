from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from cbp.statistics.finite_state import EvidenceOperators


@dataclass
class Partition:
    labels: np.ndarray
    k: int
    name: str

    def projection(self, p: np.ndarray) -> np.ndarray:
        labels = np.asarray(self.labels, dtype=int)
        p = np.asarray(p, dtype=float)
        n = len(labels)
        P = np.zeros((n, n), dtype=float)
        for cell in range(self.k):
            idx = np.flatnonzero(labels == cell)
            if len(idx) == 0:
                continue
            mass = p[idx].sum()
            if mass <= 0:
                continue
            P[np.ix_(idx, idx)] = p[idx][None, :] / mass
        return P


def feature_partition(features: np.ndarray, k: int, seed: int, name: str = "feature_kmeans") -> Partition:
    x = StandardScaler().fit_transform(np.asarray(features, dtype=float))
    labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x)
    return Partition(labels=labels, k=k, name=name)


def mean_only_partition(g: np.ndarray, r: np.ndarray, k: int, seed: int = 0) -> Partition:
    return feature_partition(np.column_stack([g, r]), k, seed, "mean_only")


def embedding_kmeans_partition(embeddings: np.ndarray, k: int, seed: int = 0) -> Partition:
    return feature_partition(np.asarray(embeddings, dtype=float), k, seed, "embedding_kmeans")


def random_partition(n_states: int, k: int, seed: int = 0) -> Partition:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n_states)
    for cell in range(k):
        if not np.any(labels == cell):
            labels[cell % n_states] = cell
    return Partition(labels=labels, k=k, name="random")


def evidence_features(
    g: np.ndarray,
    operators: EvidenceOperators,
    identity_budget: int,
    spectral_dim: int = 8,
) -> np.ndarray:
    q = operators.q(identity_budget)
    base = [g, operators.r, q @ g, q @ operators.r]
    p = operators.p
    d = np.sqrt(p)
    sim = d[:, None] * q / d[None, :]
    evals, evecs = np.linalg.eigh(0.5 * (sim + sim.T))
    order = np.argsort(evals)[::-1]
    for j in order[: min(spectral_dim, len(order))]:
        if evals[j] <= 1e-12:
            continue
        base.append(evecs[:, j] / d)
    return np.column_stack(base)


def evidence_partition(
    g: np.ndarray,
    operators: EvidenceOperators,
    identity_budget: int,
    k: int,
    seed: int = 0,
    spectral_dim: int = 8,
) -> Partition:
    x = evidence_features(g, operators, identity_budget, spectral_dim)
    return feature_partition(x, k, seed, "cbp_evidence")


def projection_diagnostics(
    partition: Partition,
    g: np.ndarray,
    operators: EvidenceOperators,
    identity_budget: int,
) -> dict[str, float]:
    P = partition.projection(operators.p)
    I = np.eye(len(g))
    q = operators.q(identity_budget)
    pg_res = P @ g - g
    pr_res = P @ operators.r - operators.r
    leakage = P @ q @ (I - P)
    d = np.sqrt(operators.p)
    leakage_sim = d[:, None] * leakage / d[None, :]
    return {
        "g_l2": float(np.sqrt(np.sum(operators.p * pg_res**2))),
        "r_l2": float(np.sqrt(np.sum(operators.p * pr_res**2))),
        "evidence_cross_op": float(np.linalg.norm(leakage_sim, ord=2)),
    }
