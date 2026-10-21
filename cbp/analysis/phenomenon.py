from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass
class MeanMatchedEvidenceResult:
    observed_evidence_distance: float
    mean_match_distance: float
    null_mean: float
    null_std: float
    permutation_p_value: float
    evidence_excess_ratio_over_permuted: float
    n_pairs: int
    match_quantile: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _weighted_standardize(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    mean = np.sum(p[:, None] * x, axis=0)
    var = np.sum(p[:, None] * (x - mean) ** 2, axis=0)
    return (x - mean) / np.sqrt(np.maximum(var, 1e-12))


def mean_matched_evidence_test(
    g: np.ndarray,
    r: np.ndarray,
    evidence_features: np.ndarray,
    p: np.ndarray | None = None,
    match_quantile: float = 0.25,
    permutations: int = 1000,
    seed: int = 0,
) -> MeanMatchedEvidenceResult:
    """Test whether mean-matched states retain different evidence responses.

    Each state is paired with its nearest neighbor in standardized ``(g, r)``
    space. Only pairs below a prespecified nearest-neighbor distance quantile
    are retained. Evidence vectors are then permuted across states while mean
    matching is fixed. This is an H1/E1 diagnostic, not causal evidence and not
    a substitute for downstream matched-FMR decision experiments.
    """
    g = np.asarray(g, dtype=float)
    r = np.asarray(r, dtype=float)
    e = np.asarray(evidence_features, dtype=float)
    n = len(g)
    if r.shape != (n,) or e.ndim != 2 or e.shape[0] != n:
        raise ValueError("g, r and 2-D evidence_features must align")
    if n < 4:
        raise ValueError("at least four states are required")
    if p is None:
        p = np.full(n, 1.0 / n)
    p = np.asarray(p, dtype=float)
    if p.shape != (n,) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("p must be a nonnegative state-mass vector")
    p = p / p.sum()
    if not (0 < match_quantile <= 1):
        raise ValueError("match_quantile must be in (0,1]")
    if permutations < 1:
        raise ValueError("permutations must be positive")

    mean_x = _weighted_standardize(np.column_stack([g, r]), p)
    e_x = _weighted_standardize(e, p)
    distances = np.linalg.norm(mean_x[:, None, :] - mean_x[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    partner = np.argmin(distances, axis=1)
    nearest = distances[np.arange(n), partner]
    cutoff = float(np.quantile(nearest, match_quantile))
    idx = np.flatnonzero(nearest <= cutoff + 1e-12)
    if len(idx) < 2:
        raise RuntimeError("too few mean-matched pairs; increase match_quantile")
    jdx = partner[idx]
    pair_w = 0.5 * (p[idx] + p[jdx])
    pair_w /= pair_w.sum()

    def statistic(features: np.ndarray) -> float:
        gaps = np.linalg.norm(features[idx] - features[jdx], axis=1)
        return float(np.sum(pair_w * gaps))

    observed = statistic(e_x)
    rng = np.random.default_rng(seed)
    null = np.empty(int(permutations), dtype=float)
    for b in range(int(permutations)):
        null[b] = statistic(e_x[rng.permutation(n)])
    p_value = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    return MeanMatchedEvidenceResult(
        observed_evidence_distance=observed,
        mean_match_distance=float(np.sum(pair_w * nearest[idx])),
        null_mean=float(null.mean()),
        null_std=float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        permutation_p_value=p_value,
        evidence_excess_ratio_over_permuted=float(
            (observed - float(null.mean())) / max(abs(float(null.mean())), 1e-12)
        ),
        n_pairs=int(len(idx)),
        match_quantile=float(match_quantile),
    )
