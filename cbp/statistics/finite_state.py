from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _as_probabilities(p: np.ndarray | None, n: int) -> np.ndarray:
    if p is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (n,):
        raise ValueError(f"p must have shape ({n},)")
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("p must be nonnegative and sum to 1")
    if np.any(p == 0):
        raise ValueError("zero-probability states must be removed before building operators")
    return p


def weighted_inner(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    return float(np.sum(p * np.asarray(a) * np.asarray(b)))


def weighted_adjoint(matrix: np.ndarray, p: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return (matrix.T * p[None, :]) / p[:, None]


@dataclass
class EvidenceOperators:
    p: np.ndarray
    alpha: float
    matcher: np.ndarray
    H: np.ndarray
    r: np.ndarray
    s_h2: np.ndarray
    K_h: np.ndarray
    A: np.ndarray
    V: np.ndarray

    def q(self, identity_budget: int) -> np.ndarray:
        n = int(identity_budget)
        if n < 2:
            raise ValueError("identity_budget must be >= 2")
        a_star = weighted_adjoint(self.A, self.p)
        q = (
            (4.0 * (n - 2) / (n * (n - 1))) * (a_star @ self.A)
            + (2.0 / (n * (n - 1))) * self.V
        )
        dp_sqrt = np.sqrt(self.p)
        sim = dp_sqrt[:, None] * q / dp_sqrt[None, :]
        sim = 0.5 * (sim + sim.T)
        evals, evecs = np.linalg.eigh(sim)
        evals = np.maximum(evals, 0.0)
        sim_psd = (evecs * evals) @ evecs.T
        return sim_psd / dp_sqrt[:, None] * dp_sqrt[None, :]

    def variance(self, action: np.ndarray, identity_budget: int) -> float:
        q = self.q(identity_budget)
        a = np.asarray(action, dtype=np.float64)
        return max(0.0, weighted_inner(a, q @ a, self.p))


def build_evidence_operators(
    matcher: np.ndarray,
    alpha: float,
    p: np.ndarray | None = None,
) -> EvidenceOperators:
    m = np.asarray(matcher, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("matcher must be square")
    if not np.allclose(m, m.T, atol=1e-10):
        raise ValueError("matcher must be symmetric for the finite-state core")
    n_states = m.shape[0]
    prob = _as_probabilities(p, n_states)
    H = m - float(alpha)
    K_h = H * prob[None, :]
    r = K_h @ np.ones(n_states)
    s_h2 = (H * H) @ prob
    A = 0.5 * np.diag(r) + 0.5 * K_h - np.ones((n_states, 1)) @ (prob * r)[None, :]
    K_h_sq_kernel = (H * H) * prob[None, :]
    r_outer = np.outer(r, prob * r)
    V = 0.5 * np.diag(s_h2) + 0.5 * K_h_sq_kernel - r_outer
    return EvidenceOperators(prob, float(alpha), m, H, r, s_h2, K_h, A, V)


def hard_matcher_from_embeddings(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    z = np.asarray(embeddings, dtype=np.float64)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    sim = z @ z.T
    return (sim >= threshold).astype(np.float64)
