"""Diffusion-time schedule helpers."""
from __future__ import annotations

import numpy as np


def linear_schedule(start: float, end: float, n_steps: int) -> np.ndarray:
    """Create a linearly spaced t_diff schedule.

    Args:
        start: Starting t_diff value.
        end: Ending t_diff value.
        n_steps: Number of steps.

    Returns:
        1D array of t_diff values from start to end.
    """
    return np.linspace(start, end, n_steps)
