from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np


SPLIT_NAMES = ("repr", "fit", "dev", "cal", "test")


@dataclass
class IdentitySplits:
    labels: np.ndarray
    split_to_identities: dict[str, list[str]]

    def indices(self, split: str) -> np.ndarray:
        if split not in SPLIT_NAMES:
            raise KeyError(split)
        return np.flatnonzero(self.labels == split)


def eligible_capture_mask(identity_ids: np.ndarray, min_captures: int = 2) -> np.ndarray:
    ids = np.asarray(identity_ids).astype(str)
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    del unique
    return counts[inverse] >= int(min_captures)


def make_identity_splits(
    identity_ids: np.ndarray,
    fractions: Mapping[str, float],
    seed: int,
) -> IdentitySplits:
    ids = np.asarray(identity_ids).astype(str)
    unique = np.unique(ids)
    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(f"fractions keys must be {SPLIT_NAMES}")
    values = np.array([fractions[k] for k in SPLIT_NAMES], dtype=float)
    if np.any(values <= 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("split fractions must be positive and sum to 1")
    if len(unique) < len(SPLIT_NAMES):
        raise ValueError("at least five identities are needed for five nonempty splits")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    raw = values * len(unique)
    counts = np.floor(raw).astype(int)
    remainder = len(unique) - int(counts.sum())
    order = np.argsort(raw - counts)[::-1]
    counts[order[:remainder]] += 1
    # Every split must contain at least one independent identity.
    for i in range(len(counts)):
        if counts[i] == 0:
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                raise ValueError("cannot make all identity splits nonempty")
            counts[donor] -= 1
            counts[i] += 1

    split_to_ids: dict[str, list[str]] = {}
    assignment: dict[str, str] = {}
    cursor = 0
    for name, count in zip(SPLIT_NAMES, counts):
        chosen = perm[cursor:cursor + count]
        cursor += count
        split_to_ids[name] = chosen.tolist()
        for identity in chosen:
            assignment[str(identity)] = name

    labels = np.array([assignment[x] for x in ids], dtype=object)
    assert_identity_disjoint(ids, labels)
    return IdentitySplits(labels=labels, split_to_identities=split_to_ids)


def assert_identity_disjoint(identity_ids: np.ndarray, split_labels: np.ndarray) -> None:
    ids = np.asarray(identity_ids).astype(str)
    splits = np.asarray(split_labels).astype(str)
    mapping: dict[str, set[str]] = {}
    for identity, split in zip(ids, splits):
        mapping.setdefault(identity, set()).add(split)
    bad = {k: v for k, v in mapping.items() if len(v) > 1}
    if bad:
        first = list(bad.items())[:5]
        raise AssertionError(f"identity leakage across splits: {first}")


def cap_identities(
    indices: np.ndarray,
    identity_ids: np.ndarray,
    max_identities: int | None,
    seed: int,
) -> np.ndarray:
    if max_identities is None:
        return np.asarray(indices)
    idx = np.asarray(indices)
    ids = np.asarray(identity_ids).astype(str)
    unique = np.unique(ids[idx])
    if len(unique) <= max_identities:
        return idx
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(unique, size=max_identities, replace=False).tolist())
    return idx[np.array([ids[i] in keep for i in idx])]
