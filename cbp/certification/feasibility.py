from __future__ import annotations

import math


def rare_identity_lower_bound(alpha: float, delta: float, beta: float = 0.2) -> float:
    """Manuscript Eq. (34) rare-hub identity lower bound."""
    if not (0 < alpha < 1 and 0 < delta < 1 and 0 <= beta < 1):
        raise ValueError("invalid alpha/delta/beta")
    return 2.0 * math.log((1.0 - beta) / delta) / (-math.log(1.0 - alpha))
