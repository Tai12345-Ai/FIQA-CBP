from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cbp.statistics.finite_state import build_evidence_operators


@dataclass
class SyntheticProblem:
    p: np.ndarray
    matcher: np.ndarray
    g: np.ndarray
    alpha: float


def cycle_hard_matcher_problem(n_states: int = 12) -> SyntheticProblem:
    """Hard-cosine-inspired cycle obstruction from the manuscript's P1."""
    if n_states < 8 or n_states % 4 != 0:
        raise ValueError("n_states must be >=8 and divisible by 4")
    angles = 2 * np.pi * np.arange(n_states) / n_states
    z = np.column_stack([np.cos(angles), np.sin(angles)])
    sim = z @ z.T
    lo = np.cos(4 * np.pi / n_states)
    hi = np.cos(2 * np.pi / n_states)
    tau = 0.5 * (lo + hi)
    matcher = (sim >= tau).astype(float)
    p = np.full(n_states, 1.0 / n_states)
    g = np.zeros(n_states)
    g[: n_states // 2] = 1.0
    lam = 1 + 2 * np.cos(2 * np.pi / n_states)
    alpha = (3 + lam) / n_states
    return SyntheticProblem(p=p, matcher=matcher, g=g, alpha=float(alpha))


def exact_variance(problem: SyntheticProblem, action: np.ndarray, n: int) -> float:
    ops = build_evidence_operators(problem.matcher, problem.alpha, problem.p)
    return ops.variance(action, n)


def monte_carlo_u_variance(
    problem: SyntheticProblem,
    action: np.ndarray,
    n: int,
    repetitions: int = 20000,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    a = np.asarray(action, dtype=float)
    H = problem.matcher - problem.alpha
    for rep in range(repetitions):
        x = rng.choice(len(problem.p), size=n, replace=True, p=problem.p)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 0.5 * (a[x[i]] + a[x[j]]) * H[x[i], x[j]]
                count += 1
        values[rep] = total / count
    return float(np.var(values, ddof=1))
