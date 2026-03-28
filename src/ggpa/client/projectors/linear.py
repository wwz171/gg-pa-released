"""Linear projector."""
from __future__ import annotations

import numpy as np

from ggpa.client.base import ProjectorBase


class LinearProjector(ProjectorBase):
    """Linear projector: Φ(s) = P @ s.
    
    Args:
        matrix: Projection matrix P with shape (output_dim, input_dim)
    """
    
    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix)

    def forward(self, s: np.ndarray) -> np.ndarray:
        """Apply linear projection.
        
        Supports both single sample (dim_s,) and batch (B, dim_s).
        """
        s = np.asarray(s)
        # Use s @ P.T to support both (dim_s,) and (B, dim_s)
        # For single: (dim_s,) @ (input_dim, output_dim).T = (output_dim,)
        # For batch: (B, dim_s) @ (input_dim, output_dim).T = (B, output_dim)
        return s @ self.matrix.T
    
    def backprop_gradient(self, s: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """Chain rule gradient: ∇_s = P^T @ grad_y.
        
        REQUIRED METHOD: For linear projection y = P @ s, the chain rule gives:
            ∇_s f(P @ s) = P^T @ ∇_y f(y)
        """
        grad_y = np.asarray(grad_y)
        # P.T @ grad_y, supporting batch dimensions
        return grad_y @ self.matrix

    def grad_forward(self, s: np.ndarray) -> np.ndarray:
        """Return Jacobian (constant matrix P).
        
        DEPRECATED: Use backprop_gradient() instead.
        """
        return self.matrix
