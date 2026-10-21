from __future__ import annotations

import numpy as np


def empirical_u_statistic(
    states: np.ndarray,
    action: np.ndarray,
    H: np.ndarray,
) -> float:
    """Order-2 one-sided symmetrized CBP U-statistic from sampled states."""
    x = np.asarray(states, dtype=int)
    a = np.asarray(action, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("at least two sampled identities are required")
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 0.5 * (a[x[i]] + a[x[j]]) * H[x[i], x[j]]
            pairs += 1
    return float(total / pairs)


def simulate_u_variance(
    p: np.ndarray,
    action: np.ndarray,
    H: np.ndarray,
    identity_budget: int,
    repetitions: int = 20_000,
    seed: int = 0,
) -> dict[str, float]:
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    values = np.empty(int(repetitions), dtype=float)
    for t in range(int(repetitions)):
        states = rng.choice(len(p), size=int(identity_budget), replace=True, p=p)
        values[t] = empirical_u_statistic(states, action, H)
    return {
        "mean": float(values.mean()),
        "variance": float(values.var(ddof=1)),
        "std": float(values.std(ddof=1)),
        "repetitions": int(repetitions),
    }
