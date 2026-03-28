"""Identity projector."""
from __future__ import annotations

import numpy as np

from ggpa.client.base import ProjectorBase

class IdentityProjector(ProjectorBase):
    """Identity projector: Φ(s) = s.
    
    This is a pass-through projector that returns the signal unchanged.
    The Jacobian is the identity matrix of the same shape as input.
    """
    
    def forward(self, s: np.ndarray) -> np.ndarray:
        """Project signal (identity operation)."""
        return np.asarray(s)
    
    def backprop_gradient(self, s: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """Chain rule gradient for identity: ∇_s = grad_y.
        
        REQUIRED METHOD: For identity projection, gradient passes through unchanged.
        """
        return np.asarray(grad_y)

    def grad_forward(self, s: np.ndarray) -> np.ndarray:
        """Return Jacobian (identity matrix with shape inferred from s).
        
        DEPRECATED: Use backprop_gradient() instead.
        """
        s = np.asarray(s)
        return np.eye(s.size).reshape(s.shape + s.shape)
