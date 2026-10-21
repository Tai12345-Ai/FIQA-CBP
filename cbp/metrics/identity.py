from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import numpy as np

from cbp.fr.cosine import l2_normalize


@dataclass
class VerificationMetrics:
    n_identities: int
    n_identity_pairs: int
    coverage: float
    genuine_yield: float
    retained_tar: float
    impostor_numerator: float
    fmr: float
    fnmr: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IdentityComponents:
    identity_ids: np.ndarray
    coverage_by_identity: np.ndarray
    genuine_yield_by_identity: np.ndarray
    impostor_pair_matrix: np.ndarray


def identity_components(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    gate: np.ndarray,
    threshold: float,
) -> IdentityComponents:
    """Construct identity-cluster components for one frozen one-sided gate."""
    z = l2_normalize(embeddings)
    ids = np.asarray(identity_ids).astype(str)
    gate = np.clip(np.asarray(gate, dtype=float), 0.0, 1.0)
    if len(z) != len(ids) or len(z) != len(gate):
        raise ValueError("embeddings, identity_ids and gate must align")
    unique = np.unique(ids)
    groups = [np.flatnonzero(ids == x) for x in unique]
    n = len(groups)
    if n < 2:
        raise ValueError("at least two identities are required")

    coverage = np.array([float(np.mean(gate[idx])) for idx in groups])
    genuine = np.zeros(n, dtype=float)
    for i, idx in enumerate(groups):
        if len(idx) < 2:
            genuine[i] = 0.0
            continue
        match = (z[idx] @ z[idx].T) >= threshold
        np.fill_diagonal(match, False)
        per_probe = match.sum(axis=1) / (len(idx) - 1)
        genuine[i] = float(np.mean(gate[idx] * per_probe))

    pair_matrix = np.zeros((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        ii, jj = groups[i], groups[j]
        match = (z[ii] @ z[jj].T) >= threshold
        i_probe = float(np.mean(gate[ii, None] * match))
        j_probe = float(np.mean(gate[jj, None] * match.T))
        value = 0.5 * (i_probe + j_probe)
        pair_matrix[i, j] = value
        pair_matrix[j, i] = value
    return IdentityComponents(unique, coverage, genuine, pair_matrix)


def metrics_from_components(components: IdentityComponents) -> VerificationMetrics:
    c = float(np.mean(components.coverage_by_identity))
    u = float(np.mean(components.genuine_yield_by_identity))
    n = len(components.identity_ids)
    upper = components.impostor_pair_matrix[np.triu_indices(n, 1)]
    j = float(np.mean(upper)) if len(upper) else float("nan")
    fmr = j / c if c > 0 else float("inf")
    tar = u / c if c > 0 else float("nan")
    return VerificationMetrics(
        n_identities=n,
        n_identity_pairs=len(upper),
        coverage=c,
        genuine_yield=u,
        retained_tar=tar,
        impostor_numerator=j,
        fmr=fmr,
        fnmr=1.0 - tar if np.isfinite(tar) else float("nan"),
    )


def evaluate_one_sided(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    gate: np.ndarray,
    threshold: float,
) -> VerificationMetrics:
    return metrics_from_components(
        identity_components(embeddings, identity_ids, gate, threshold)
    )


def _point_and_influence(components: IdentityComponents) -> dict[str, tuple[float, np.ndarray]]:
    """Point estimates and identity-level influence functions.

    For the pair U-statistic, the first-order influence is
    ``2 * (row_mean_i - theta)``. Ratio influences use the delta method. These
    objects support a scalable paired multiplier bootstrap and remain distinct
    from the independent finite-sample deployment certificate.
    """
    c_i = np.asarray(components.coverage_by_identity, dtype=float)
    u_i = np.asarray(components.genuine_yield_by_identity, dtype=float)
    H = np.asarray(components.impostor_pair_matrix, dtype=float)
    n = len(c_i)
    if H.shape != (n, n):
        raise ValueError("impostor_pair_matrix has invalid shape")
    c = float(c_i.mean())
    u = float(u_i.mean())
    if n < 2:
        raise ValueError("at least two identities are required")
    j = float(H[np.triu_indices(n, 1)].mean())

    if_c = c_i - c
    if_u = u_i - u
    row_mean = H.sum(axis=1) / (n - 1)
    if_j = 2.0 * (row_mean - j)

    out: dict[str, tuple[float, np.ndarray]] = {
        "coverage": (c, if_c),
        "genuine_yield": (u, if_u),
        "impostor_numerator": (j, if_j),
    }
    if c > 0:
        tar = u / c
        fmr = j / c
        if_tar = if_u / c - (u / (c * c)) * if_c
        if_fmr = if_j / c - (j / (c * c)) * if_c
        out["retained_tar"] = (tar, if_tar)
        out["fnmr"] = (1.0 - tar, -if_tar)
        out["fmr"] = (fmr, if_fmr)
    else:
        nan_if = np.full(n, np.nan)
        out["retained_tar"] = (float("nan"), nan_if)
        out["fnmr"] = (float("nan"), nan_if)
        out["fmr"] = (float("nan"), nan_if)
    return out


def paired_identity_multiplier_bootstrap_all(
    a: IdentityComponents,
    b: IdentityComponents,
    repetitions: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    batch_size: int = 1024,
) -> dict[str, dict[str, float | int | str]]:
    """Scalable paired identity-influence multiplier bootstrap for A-B.

    The same centered Gaussian multipliers are applied to both methods, so
    covariance from shared identities is retained. This is an asymptotic
    identity-cluster inference procedure, not a finite-sample certificate.
    Pair observations are never treated as iid samples.
    """
    if not np.array_equal(a.identity_ids, b.identity_ids):
        raise ValueError("paired bootstrap requires identical ordered identities")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not (0 < confidence < 1):
        raise ValueError("confidence must lie in (0,1)")
    n = len(a.identity_ids)
    points_a = _point_and_influence(a)
    points_b = _point_and_influence(b)
    metrics = sorted(set(points_a) & set(points_b))
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in metrics}

    for start in range(0, int(repetitions), int(batch_size)):
        size = min(int(batch_size), int(repetitions) - start)
        multipliers = rng.normal(size=(size, n))
        multipliers -= multipliers.mean(axis=1, keepdims=True)
        for metric in metrics:
            point_a, influence_a = points_a[metric]
            point_b, influence_b = points_b[metric]
            if not (
                np.isfinite(point_a)
                and np.isfinite(point_b)
                and np.all(np.isfinite(influence_a))
                and np.all(np.isfinite(influence_b))
            ):
                continue
            point_difference = point_a - point_b
            perturbation = multipliers @ (influence_a - influence_b) / n
            samples[metric].append(point_difference + perturbation)

    tail = (1.0 - confidence) / 2.0
    out: dict[str, dict[str, float | int | str]] = {}
    for metric in metrics:
        if not samples[metric]:
            continue
        values = np.concatenate(samples[metric])
        point_difference = points_a[metric][0] - points_b[metric][0]
        out[metric] = {
            "method": "paired_identity_influence_gaussian_multiplier",
            "point_difference": float(point_difference),
            "bootstrap_mean_difference": float(values.mean()),
            "ci_low": float(np.quantile(values, tail)),
            "ci_high": float(np.quantile(values, 1.0 - tail)),
            "probability_positive": float(np.mean(values > 0)),
            "repetitions_used": int(len(values)),
            "confidence": float(confidence),
            "n_identities": int(n),
        }
    return out


def paired_identity_bootstrap(
    a: IdentityComponents,
    b: IdentityComponents,
    metric: str = "genuine_yield",
    repetitions: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Compatibility wrapper returning one paired multiplier-bootstrap metric."""
    all_metrics = paired_identity_multiplier_bootstrap_all(
        a,
        b,
        repetitions=repetitions,
        confidence=confidence,
        seed=seed,
    )
    if metric not in all_metrics:
        raise ValueError(f"metric unavailable or nonfinite: {metric}")
    return all_metrics[metric]
