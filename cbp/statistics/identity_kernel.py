from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np

from cbp.fr.cosine import l2_normalize
from .moments import CellMoments


@dataclass
class ActionStatistics:
    coverage: float
    genuine_yield: float
    excess_risk: float
    evidence_variance: float
    evidence_std: float
    n_identities: int
    n_pairs: int


@dataclass
class IdentityKernelDiagnostics:
    n_identities: int
    n_pairs: int
    zeta1: np.ndarray
    zeta2: np.ndarray
    influence: np.ndarray
    pair_mean: np.ndarray


def _identity_groups(identity_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    ids = np.asarray(identity_ids).astype(str)
    unique = np.unique(ids)
    groups = [np.flatnonzero(ids == identity) for identity in unique]
    return unique, groups


def _genuine_yield_by_cell(
    z: np.ndarray,
    groups: list[np.ndarray],
    labels: np.ndarray,
    threshold: float,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identity-balanced p and genuine-yield cell moments.

    p_k is the expected fraction of captures in cell k for a uniformly sampled
    identity. u_k is the expected one-sided genuine yield contributed by that
    cell. Singleton identities contribute coverage but no observed genuine
    success; callers should normally remove them before fitting.
    """
    p_rows = []
    u_rows = []
    counts = np.bincount(labels, minlength=n_cells)
    for idx in groups:
        row_p = np.bincount(labels[idx], minlength=n_cells).astype(float) / len(idx)
        row_u = np.zeros(n_cells, dtype=float)
        if len(idx) >= 2:
            match = (z[idx] @ z[idx].T) >= threshold
            np.fill_diagonal(match, False)
            per_probe = match.sum(axis=1) / (len(idx) - 1)
            for k in range(n_cells):
                mask = labels[idx] == k
                if np.any(mask):
                    row_u[k] = float(np.sum(per_probe[mask]) / len(idx))
        p_rows.append(row_p)
        u_rows.append(row_u)
    return np.mean(p_rows, axis=0), np.mean(u_rows, axis=0), counts


def _pair_kernel_vector(
    z: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    alpha: float,
    n_cells: int,
) -> np.ndarray:
    """Symmetrized directional one-sided excess-risk kernel for cell bases."""
    h = ((z[idx_a] @ z[idx_b].T) >= threshold).astype(float) - float(alpha)
    # Orientation A -> B: gate depends only on the probe capture in A.
    # The bincount form is O(n_a n_b + n_a + n_b), not O(K n_a n_b).
    per_a = np.mean(h, axis=1)
    per_b = np.mean(h, axis=0)
    a_to_b = np.bincount(labels[idx_a], weights=per_a, minlength=n_cells) / len(idx_a)
    b_to_a = np.bincount(labels[idx_b], weights=per_b, minlength=n_cells) / len(idx_b)
    return 0.5 * (a_to_b + b_to_a)


def estimate_identity_action_statistics(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    action: np.ndarray,
    threshold: float,
    alpha: float,
    identity_budget: int,
) -> ActionStatistics:
    """Plug-in identity-cluster moments for one frozen image-level action."""
    z = l2_normalize(embeddings)
    ids = np.asarray(identity_ids).astype(str)
    action = np.clip(np.asarray(action, dtype=float), 0.0, 1.0)
    if len(z) != len(ids) or len(z) != len(action):
        raise ValueError("embeddings, identities and action must align")
    _, groups = _identity_groups(ids)
    m = len(groups)
    if m < 4:
        raise ValueError("at least four identities are required")

    coverage_rows = []
    utility_rows = []
    for idx in groups:
        coverage_rows.append(float(np.mean(action[idx])))
        if len(idx) < 2:
            utility_rows.append(0.0)
            continue
        match = (z[idx] @ z[idx].T) >= threshold
        np.fill_diagonal(match, False)
        per_probe = match.sum(axis=1) / (len(idx) - 1)
        utility_rows.append(float(np.mean(action[idx] * per_probe)))

    pairs = np.asarray(list(combinations(range(m), 2)), dtype=np.int64)
    pair_values = np.empty(len(pairs), dtype=float)
    sums = np.zeros(m, dtype=float)
    for t, (a, b) in enumerate(pairs):
        ia, ib = groups[a], groups[b]
        h = ((z[ia] @ z[ib].T) >= threshold).astype(float) - float(alpha)
        value = 0.5 * (
            float(np.mean(action[ia, None] * h))
            + float(np.mean(action[ib, None] * h.T))
        )
        pair_values[t] = value
        sums[a] += value
        sums[b] += value
    theta = float(np.mean(pair_values))
    influence = sums / (m - 1) - theta
    zeta1 = float(np.mean(influence**2))
    residual = pair_values - theta - influence[pairs[:, 0]] - influence[pairs[:, 1]]
    zeta2 = float(np.mean(residual**2))
    n = int(identity_budget)
    variance = (4.0 / n) * zeta1 + (2.0 / (n * (n - 1))) * zeta2
    return ActionStatistics(
        coverage=float(np.mean(coverage_rows)),
        genuine_yield=float(np.mean(utility_rows)),
        excess_risk=theta,
        evidence_variance=max(0.0, variance),
        evidence_std=float(np.sqrt(max(0.0, variance))),
        n_identities=m,
        n_pairs=len(pairs),
    )


def estimate_identity_cluster_moments(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    cell_labels: np.ndarray,
    threshold: float,
    alpha: float,
    identity_budget: int,
    n_cells: int | None = None,
    max_pairs: int | None = None,
    seed: int = 0,
) -> tuple[CellMoments, IdentityKernelDiagnostics]:
    """Estimate deployed cell moments from independent identity clusters.

    The covariance matrix is a plug-in Hoeffding decomposition of the vector
    order-2 U-statistic. It preserves identity as the independent unit and must
    not be interpreted as an exact finite-sample confidence bound. Independent
    calibration in ``cbp.certification`` remains mandatory.

    If ``max_pairs`` is set, uniformly sampled identity pairs are used. The
    resulting influence estimate is a computational approximation and must be
    reported as such; use all pairs for primary pilot experiments.
    """
    z = l2_normalize(embeddings)
    ids = np.asarray(identity_ids).astype(str)
    labels = np.asarray(cell_labels, dtype=int)
    if len(z) != len(ids) or len(z) != len(labels):
        raise ValueError("embeddings, identities and labels must align")
    if np.any(labels < 0):
        raise ValueError("cell labels must be nonnegative")
    k = int(labels.max()) + 1 if n_cells is None else int(n_cells)
    unique, groups = _identity_groups(ids)
    m = len(unique)
    if m < 4:
        raise ValueError("at least four independent identities are required")

    p, u, counts = _genuine_yield_by_cell(z, groups, labels, threshold, k)

    all_pairs = np.asarray(list(combinations(range(m), 2)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if max_pairs is not None and len(all_pairs) > max_pairs:
        selected = rng.choice(len(all_pairs), size=int(max_pairs), replace=False)
        pairs = all_pairs[selected]
    else:
        pairs = all_pairs

    kernels = np.stack([
        _pair_kernel_vector(z, groups[a], groups[b], labels, threshold, alpha, k)
        for a, b in pairs
    ])
    theta = np.mean(kernels, axis=0)

    # Estimate first-order influence by each identity's observed partner mean.
    sums = np.zeros((m, k), dtype=float)
    degrees = np.zeros(m, dtype=int)
    for value, (a, b) in zip(kernels, pairs):
        sums[a] += value
        sums[b] += value
        degrees[a] += 1
        degrees[b] += 1
    if np.any(degrees == 0):
        raise RuntimeError(
            "incomplete pair sampling left identities without partners; "
            "increase max_pairs or use all pairs"
        )
    influence = sums / degrees[:, None] - theta[None, :]
    influence = influence - influence.mean(axis=0, keepdims=True)
    zeta1 = (influence.T @ influence) / m

    residuals = np.empty_like(kernels)
    for t, (a, b) in enumerate(pairs):
        residuals[t] = kernels[t] - theta - influence[a] - influence[b]
    zeta2 = (residuals.T @ residuals) / len(residuals)

    n = int(identity_budget)
    if n < 2:
        raise ValueError("identity_budget must be >= 2")
    B = (4.0 / n) * zeta1 + (2.0 / (n * (n - 1))) * zeta2
    B = 0.5 * (B + B.T)
    evals, evecs = np.linalg.eigh(B)
    B = (evecs * np.maximum(evals, 0.0)) @ evecs.T

    mu_g = np.divide(u, p, out=np.full_like(u, np.nan), where=p > 0)
    mu_I = float(alpha) + np.divide(theta, p, out=np.full_like(theta, np.nan), where=p > 0)
    moments = CellMoments(
        p=p,
        u=u,
        ell=theta,
        B=B,
        mu_g=mu_g,
        mu_I=mu_I,
        alpha=float(alpha),
        identity_budget=n,
        counts=counts,
        n_fit_identities=m,
        pair_count=len(pairs),
    )
    diagnostics = IdentityKernelDiagnostics(
        n_identities=m,
        n_pairs=len(pairs),
        zeta1=zeta1,
        zeta2=zeta2,
        influence=influence,
        pair_mean=theta,
    )
    return moments, diagnostics
