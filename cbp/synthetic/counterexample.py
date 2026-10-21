from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cbp.statistics.finite_state import EvidenceOperators, build_evidence_operators
from cbp.representation.partitions import Partition


@dataclass
class CycleCounterexample:
    embeddings: np.ndarray
    matcher: np.ndarray
    p: np.ndarray
    g: np.ndarray
    action: np.ndarray
    partition: Partition
    alpha: float
    operators: EvidenceOperators

    @property
    def projected_action(self) -> np.ndarray:
        return self.partition.projection(self.p) @ self.action


def make_cycle_counterexample(n_states: int = 16) -> CycleCounterexample:
    """Hard-cosine mean-preserving/evidence-breaking cycle construction.

    Implements the P1 construction in the current CBP proposal for N=4m. The
    chosen threshold makes each state match itself and its two cycle neighbors.
    The two-cell compression preserves the chosen cell-constant genuine profile
    and mean risk, while it generally destroys first-order cancellation of the
    fractional action.
    """
    if n_states < 8 or n_states % 4 != 0:
        raise ValueError("n_states must be a multiple of four and at least eight")
    theta = 2.0 * np.pi * np.arange(n_states) / n_states
    embeddings = np.column_stack([np.cos(theta), np.sin(theta)])
    threshold = 0.5 * (
        np.cos(2.0 * np.pi / n_states)
        + np.cos(4.0 * np.pi / n_states)
    )
    matcher = (embeddings @ embeddings.T >= threshold).astype(float)
    p = np.full(n_states, 1.0 / n_states)

    v = np.cos(theta)
    lam = 1.0 + 2.0 * np.cos(2.0 * np.pi / n_states)
    alpha = (3.0 + lam) / n_states
    action = 0.5 * (1.0 + v)
    g = np.zeros(n_states, dtype=float)
    g[: n_states // 2] = 1.0
    labels = np.zeros(n_states, dtype=int)
    labels[n_states // 2 :] = 1
    partition = Partition(labels=labels, k=2, name="cycle_halves")
    operators = build_evidence_operators(matcher, alpha=alpha, p=p)
    return CycleCounterexample(
        embeddings=embeddings,
        matcher=matcher,
        p=p,
        g=g,
        action=action,
        partition=partition,
        alpha=alpha,
        operators=operators,
    )


def counterexample_diagnostics(example: CycleCounterexample, identity_budget: int) -> dict[str, float]:
    P = example.partition.projection(example.p)
    a = example.action
    pa = P @ a
    op = example.operators
    q = op.q(identity_budget)
    return {
        "coverage_gap": float(abs(example.p @ (a - pa))),
        "utility_gap": float(abs((example.p * example.g) @ (a - pa))),
        "risk_gap": float(abs((example.p * op.r) @ (a - pa))),
        "A_action_norm": float(np.sqrt(np.sum(example.p * (op.A @ a) ** 2))),
        "A_projected_norm": float(np.sqrt(np.sum(example.p * (op.A @ pa) ** 2))),
        "variance_action": float(np.sum(example.p * a * (q @ a))),
        "variance_projected": float(np.sum(example.p * pa * (q @ pa))),
    }
