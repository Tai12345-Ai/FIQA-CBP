from __future__ import annotations

import numpy as np


def router_transfer_metrics(
    teacher_labels: np.ndarray,
    deployed_labels: np.ndarray,
    beta: np.ndarray | None = None,
) -> dict[str, float]:
    t = np.asarray(teacher_labels, dtype=int)
    d = np.asarray(deployed_labels, dtype=int)
    if t.shape != d.shape:
        raise ValueError("teacher/deployed labels must align")
    mismatch = t != d
    out = {
        "label_error": float(np.mean(mismatch)),
        "epsilon_1_label": float(np.mean(mismatch)),
        "epsilon_2_label": float(np.mean(mismatch)),
    }
    if beta is not None:
        beta = np.asarray(beta, dtype=float)
        a_t = beta[t]
        a_d = beta[d]
        diff = a_d - a_t
        out.update({
            "policy_l1": float(np.mean(np.abs(diff))),
            "policy_l2_sq": float(np.mean(diff * diff)),
            "policy_disagreement": float(np.mean(np.abs(diff) > 1e-8)),
        })
    return out
