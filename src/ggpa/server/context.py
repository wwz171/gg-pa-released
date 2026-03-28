"""Context density implementations."""
from __future__ import annotations

from dataclasses import dataclass
from ggpa.server.base import ContextBase
import numpy as np



@dataclass(frozen=True)
class UniformContext(ContextBase):
    """Uniform context density (log_prob = 0).
    
    Provides no constraint on the signal.
    """

    def log_prob(self, s: np.ndarray, tau: float) -> float:
        """Log probability (constant 0, independent of tau)."""
        return 0.0
    
    def grad_log_prob(self, s: np.ndarray, tau: float) -> np.ndarray:
        """Gradient is zero for uniform distribution."""
        s = np.asarray(s)
        return np.zeros_like(s)

