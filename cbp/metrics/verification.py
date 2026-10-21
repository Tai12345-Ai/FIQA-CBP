from __future__ import annotations

import numpy as np


def tar_at_fmr(genuine_scores: np.ndarray, impostor_scores: np.ndarray, target_fmr: float) -> dict[str, float]:
    threshold = float(np.quantile(impostor_scores, 1 - target_fmr, method="higher"))
    fmr = float(np.mean(np.asarray(impostor_scores) >= threshold))
    tar = float(np.mean(np.asarray(genuine_scores) >= threshold))
    return {"threshold": threshold, "tar": tar, "fmr": fmr, "fnmr": 1.0 - tar}


def roc_from_scores(genuine_scores: np.ndarray, impostor_scores: np.ndarray):
    g = np.asarray(genuine_scores, dtype=float)
    i = np.asarray(impostor_scores, dtype=float)
    thresholds = np.unique(np.concatenate([g, i]))[::-1]
    thresholds = np.concatenate([[np.inf], thresholds, [-np.inf]])
    tar = np.array([np.mean(g >= t) for t in thresholds])
    fmr = np.array([np.mean(i >= t) for t in thresholds])
    return thresholds, tar, fmr
