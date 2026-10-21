from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize the last axis without modifying the input array in place."""
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarities between two embedding matrices."""
    return l2_normalize(a) @ l2_normalize(b).T


def sample_impostor_scores(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    max_pairs: int = 1_000_000,
    seed: int = 0,
) -> np.ndarray:
    """Sample identity-balanced image-level impostor scores.

    Identities are sampled uniformly and then one capture is sampled uniformly
    within each identity. The returned comparisons freeze a matcher threshold;
    their count is not interpreted as the number of independent observations.
    """
    z = l2_normalize(embeddings)
    ids = np.asarray(identity_ids).astype(str)
    n = len(z)
    if n < 2 or len(np.unique(ids)) < 2:
        raise ValueError("at least two identities are required")
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")

    target = int(min(max_pairs, max(10_000, n * 50)))
    rng = np.random.default_rng(seed)
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    sample_prob = 1.0 / (len(counts) * counts[inverse])
    sample_prob /= sample_prob.sum()

    # Preallocation avoids the large Python-object overhead of a float list at
    # low-FMR settings where millions of construction scores may be requested.
    scores = np.empty(target, dtype=np.float64)
    cursor = 0
    attempts = 0
    max_attempts = max(100_000, 50 * target)
    while cursor < target and attempts < max_attempts:
        batch = min(max(4_096, 2 * (target - cursor)), 200_000)
        i = rng.choice(n, size=batch, replace=True, p=sample_prob)
        j = rng.choice(n, size=batch, replace=True, p=sample_prob)
        mask = (i != j) & (ids[i] != ids[j])
        if np.any(mask):
            values = np.sum(z[i[mask]] * z[j[mask]], axis=1)
            take = min(len(values), target - cursor)
            scores[cursor:cursor + take] = values[:take]
            cursor += take
        attempts += batch
    if cursor == 0:
        raise RuntimeError("could not sample any impostor pair")
    if cursor < target:
        scores = scores[:cursor]
    return scores


def calibrate_threshold(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    target_fmr: float,
    max_impostor_pairs: int = 1_000_000,
    seed: int = 0,
) -> float:
    """Freeze a cosine threshold at an empirical impostor quantile.

    This is a design quantity, not an independent certificate. For low FMR,
    report the distinct identity count and the score sample size.
    """
    if not (0.0 < target_fmr < 1.0):
        raise ValueError("target_fmr must be in (0, 1)")
    scores = sample_impostor_scores(
        embeddings,
        identity_ids,
        max_pairs=max_impostor_pairs,
        seed=seed,
    )
    return float(np.quantile(scores, 1.0 - target_fmr, method="higher"))
