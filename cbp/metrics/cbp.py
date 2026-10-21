from __future__ import annotations

import numpy as np

from cbp.representation.partitions import Partition
from cbp.statistics.finite_state import EvidenceOperators


def decision_distortion(partition: Partition, q_score: np.ndarray, p: np.ndarray) -> float:
    P = partition.projection(p)
    q = np.asarray(q_score, dtype=float)
    return float(np.sum(p * np.maximum(q, 0)) - np.sum(p * np.maximum(P @ q, 0)))


def sign_disagreement_mass(partition: Partition, q_score: np.ndarray, p: np.ndarray) -> float:
    P = partition.projection(p)
    q = np.asarray(q_score)
    disagree = (q > 0) != ((P @ q) > 0)
    return float(np.sum(np.asarray(p) * disagree))


def evidence_energy(action: np.ndarray, operators: EvidenceOperators, n: int) -> float:
    return operators.variance(action, n)
