from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from sklearn.cluster import KMeans

from .partitions import Partition


@dataclass
class RefinementStep:
    parent_cell: int
    value_before: float
    value_after: float
    accepted: bool


def greedy_sign_refinement(
    initial: Partition,
    observable: np.ndarray,
    lifted_score: np.ndarray,
    evaluator: Callable[[np.ndarray], float],
    max_cells: int,
    min_cell_size: int = 8,
    seed: int = 0,
) -> tuple[Partition, list[RefinementStep]]:
    labels = np.asarray(initial.labels, dtype=int).copy()
    x = np.asarray(observable, dtype=float)
    score = np.asarray(lifted_score, dtype=float).copy()
    history: list[RefinementStep] = []
    next_label = int(labels.max()) + 1
    current = float(evaluator(labels))
    while next_label < max_cells:
        candidates = []
        for cell in np.unique(labels):
            idx = np.flatnonzero(labels == cell)
            if len(idx) < 2 * min_cell_size:
                continue
            signs = np.sign(score[idx])
            if np.any(signs > 0) and np.any(signs < 0):
                candidates.append((float(np.mean(np.abs(score[idx]))), int(cell), idx))
        if not candidates:
            break
        _, cell, idx = max(candidates, key=lambda t: t[0])
        sub = KMeans(n_clusters=2, n_init=10, random_state=seed + next_label).fit_predict(x[idx])
        if min(np.bincount(sub, minlength=2)) < min_cell_size:
            score[idx] = np.abs(score[idx])
            continue
        proposal = labels.copy()
        proposal[idx[sub == 1]] = next_label
        after = float(evaluator(proposal))
        accepted = after > current + 1e-10
        history.append(RefinementStep(cell, current, after, accepted))
        if accepted:
            labels = proposal
            current = after
            next_label += 1
        else:
            score[idx] = np.abs(score[idx])
    return Partition(labels=labels, k=int(labels.max()) + 1, name="cbp_refined"), history
