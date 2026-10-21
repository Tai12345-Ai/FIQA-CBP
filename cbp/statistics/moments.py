from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .finite_state import EvidenceOperators, weighted_inner


@dataclass
class CellMoments:
    p: np.ndarray
    u: np.ndarray
    ell: np.ndarray
    B: np.ndarray
    mu_g: np.ndarray
    mu_I: np.ndarray
    alpha: float
    identity_budget: int
    counts: np.ndarray
    n_fit_identities: int | None = None
    pair_count: int | None = None

    def evidence_response(self, beta: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta, dtype=np.float64)
        num = self.B @ beta
        out = np.full_like(self.p, np.nan, dtype=np.float64)
        mask = self.p > 0
        out[mask] = num[mask] / self.p[mask]
        return out


def indicator_matrix(labels: np.ndarray, k: int | None = None) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if np.any(labels < 0):
        raise ValueError("labels must be nonnegative")
    n_cells = int(labels.max()) + 1 if k is None else int(k)
    z = np.zeros((len(labels), n_cells), dtype=np.float64)
    z[np.arange(len(labels)), labels] = 1.0
    return z


def estimate_cell_moments(
    labels: np.ndarray,
    g: np.ndarray,
    operators: EvidenceOperators,
    identity_budget: int,
    k: int | None = None,
) -> CellMoments:
    labels = np.asarray(labels, dtype=int)
    g = np.asarray(g, dtype=np.float64)
    if len(labels) != len(g) or len(labels) != len(operators.p):
        raise ValueError("labels, g and operator state space must align")
    z = indicator_matrix(labels, k)
    p_state = operators.p
    q = operators.q(identity_budget)
    p_cell = z.T @ p_state
    u = z.T @ (p_state * g)
    ell = z.T @ (p_state * operators.r)
    B = z.T @ (p_state[:, None] * (q @ z))
    B = 0.5 * (B + B.T)
    evals, evecs = np.linalg.eigh(B)
    B = (evecs * np.maximum(evals, 0.0)) @ evecs.T
    mu_g = np.divide(u, p_cell, out=np.full_like(u, np.nan), where=p_cell > 0)
    mu_I = operators.alpha + np.divide(
        ell, p_cell, out=np.full_like(ell, np.nan), where=p_cell > 0
    )
    counts = np.bincount(labels, minlength=z.shape[1])
    return CellMoments(
        p=p_cell,
        u=u,
        ell=ell,
        B=B,
        mu_g=mu_g,
        mu_I=mu_I,
        alpha=operators.alpha,
        identity_budget=int(identity_budget),
        counts=counts,
    )


def aggregate_cell_moments(
    dense: CellMoments,
    state_to_cell: np.ndarray,
    sample_state_labels: np.ndarray | None = None,
) -> CellMoments:
    """Aggregate dense observable-state moments into a hard partition."""
    mapping = np.asarray(state_to_cell, dtype=int)
    if len(mapping) != len(dense.p):
        raise ValueError("state_to_cell must align with dense moments")
    if np.any(mapping < 0):
        raise ValueError("state_to_cell must be nonnegative")
    k = int(mapping.max()) + 1
    Z = indicator_matrix(mapping, k)
    p = Z.T @ dense.p
    u = Z.T @ dense.u
    ell = Z.T @ dense.ell
    B = Z.T @ dense.B @ Z
    B = 0.5 * (B + B.T)
    evals, evecs = np.linalg.eigh(B)
    B = (evecs * np.maximum(evals, 0.0)) @ evecs.T
    mu_g = np.divide(u, p, out=np.full_like(u, np.nan), where=p > 0)
    mu_I = dense.alpha + np.divide(ell, p, out=np.full_like(ell, np.nan), where=p > 0)
    if sample_state_labels is None:
        counts = np.zeros(k, dtype=int)
    else:
        state_labels = np.asarray(sample_state_labels, dtype=int)
        counts = np.bincount(mapping[state_labels], minlength=k)
    return CellMoments(
        p=p,
        u=u,
        ell=ell,
        B=B,
        mu_g=mu_g,
        mu_I=mu_I,
        alpha=dense.alpha,
        identity_budget=dense.identity_budget,
        counts=counts,
        n_fit_identities=dense.n_fit_identities,
        pair_count=dense.pair_count,
    )


def action_metrics(
    action: np.ndarray,
    g: np.ndarray,
    operators: EvidenceOperators,
    identity_budget: int,
) -> dict[str, float]:
    a = np.asarray(action, dtype=np.float64)
    p = operators.p
    g = np.asarray(g, dtype=np.float64)
    coverage = weighted_inner(a, np.ones_like(a), p)
    utility = weighted_inner(a, g, p)
    excess_risk = weighted_inner(a, operators.r, p)
    impostor_num = excess_risk + operators.alpha * coverage
    var = operators.variance(a, identity_budget)
    return {
        "coverage": coverage,
        "genuine_yield": utility,
        "retained_tar": utility / coverage if coverage > 0 else float("nan"),
        "impostor_numerator": impostor_num,
        "fmr": impostor_num / coverage if coverage > 0 else float("nan"),
        "excess_risk": excess_risk,
        "evidence_variance": var,
        "evidence_std": float(np.sqrt(max(var, 0.0))),
    }
