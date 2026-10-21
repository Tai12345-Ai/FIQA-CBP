"""Compatibility wrappers for deployment evaluation.

The canonical implementation lives in :mod:`cbp.metrics.identity` and keeps
identity clusters—not image pairs—as the independent units.
"""

from __future__ import annotations

import numpy as np

from .identity import evaluate_one_sided as _evaluate_one_sided


def evaluate_one_sided(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    gate: np.ndarray,
    threshold: float,
    max_impostor_pairs: int | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Evaluate one-sided gating using complete identity-cluster pairs.

    ``max_impostor_pairs`` and ``seed`` are accepted only for backward
    compatibility and intentionally ignored. Subsampling image pairs would
    change the estimand and would incorrectly treat overlapping pairs as iid.
    """
    del max_impostor_pairs, seed
    return _evaluate_one_sided(
        embeddings=embeddings,
        identity_ids=identity_ids,
        gate=gate,
        threshold=threshold,
    ).to_dict()
