from __future__ import annotations

import numpy as np


def estimate_genuine_utility(embeddings: np.ndarray, identity_ids: np.ndarray, threshold: float) -> np.ndarray:
    z = np.asarray(embeddings, dtype=np.float64)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    ids = np.asarray(identity_ids).astype(str)
    g = np.full(len(z), np.nan, dtype=np.float64)
    for identity in np.unique(ids):
        idx = np.flatnonzero(ids == identity)
        if len(idx) < 2:
            continue
        sim = z[idx] @ z[idx].T
        accept = sim >= threshold
        np.fill_diagonal(accept, False)
        g[idx] = accept.sum(axis=1) / (len(idx) - 1)
    observed = g[np.isfinite(g)]
    fill = float(np.mean(observed)) if len(observed) else 0.0
    return np.where(np.isfinite(g), g, fill)


def estimate_impostor_probability(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    threshold: float,
    max_partners: int = 4096,
    seed: int = 0,
) -> np.ndarray:
    z = np.asarray(embeddings, dtype=np.float64)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    ids = np.asarray(identity_ids).astype(str)
    rng = np.random.default_rng(seed)
    out = np.zeros(len(z), dtype=np.float64)
    all_idx = np.arange(len(z))
    for i in range(len(z)):
        candidates = all_idx[ids != ids[i]]
        if len(candidates) == 0:
            raise ValueError("need at least two identities")
        if len(candidates) > max_partners:
            candidates = rng.choice(candidates, size=max_partners, replace=False)
        out[i] = np.mean((z[candidates] @ z[i]) >= threshold)
    return out
