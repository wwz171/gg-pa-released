"""Tau schedule helpers."""
from __future__ import annotations

import numpy as np


def linear_schedule(start: float, end: float, n_steps: int) -> np.ndarray:
    """Create a linearly spaced tau schedule.

    Args:
        start: Starting tau value.
        end: Ending tau value.
        n_steps: Number of steps.

    Returns:
        1D array of tau values from start to end.
    """
    return np.linspace(start, end, n_steps)
