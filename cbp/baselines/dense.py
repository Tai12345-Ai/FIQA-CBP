from __future__ import annotations

import numpy as np

from cbp.representation.partitions import Partition


def dense_state_partition(n_states: int) -> Partition:
    """One independent policy variable per finite observable state.

    This is the matched full-feature comparator inside the same finite state
    space. It is not claimed to be a deployable neural policy by itself.
    """
    return Partition(labels=np.arange(int(n_states), dtype=int), k=int(n_states), name="dense_state")
