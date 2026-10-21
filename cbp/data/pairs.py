from __future__ import annotations

from itertools import combinations
import numpy as np


def identity_groups(identity_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    ids = np.asarray(identity_ids).astype(str)
    unique = np.unique(ids)
    groups = [np.flatnonzero(ids == x) for x in unique]
    return unique, groups


def all_identity_pairs(n_identities: int) -> np.ndarray:
    return np.asarray(list(combinations(range(n_identities), 2)), dtype=np.int64)


def identity_balanced_sample_weights(identity_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(identity_ids).astype(str)
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return 1.0 / (len(unique) * counts[inverse])
