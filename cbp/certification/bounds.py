from __future__ import annotations

import math


def hoeffding_mean_radius(n: int, delta: float, value_range: float = 1.0) -> float:
    if n <= 0:
        return float("inf")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return value_range * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def hoeffding_u2_radius(n_identities: int, delta: float, value_range: float = 1.0) -> float:
    """Conservative Hoeffding radius for a bounded order-2 U-statistic.

    Uses floor(n/2) independent blocks. This is intentionally conservative and
    is separate from the manuscript's sharper evidence-aware fitting objective.
    """
    blocks = n_identities // 2
    return hoeffding_mean_radius(blocks, delta, value_range)
