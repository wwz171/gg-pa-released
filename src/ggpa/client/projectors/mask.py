"""Mask projector."""
from __future__ import annotations

import numpy as np

from ggpa.core.errors import ShapeError
from ggpa.client.base import ProjectorBase


class MaskProjector(ProjectorBase):
    """Mask projector: Φ(s) = s[indices].
    
    Selects a subset of coordinates from the signal.
    
    Args:
        indices: 1D array of indices to select
    """
    
    def __init__(self, indices: np.ndarray):
        self.indices = np.asarray(indices)
        if self.indices.ndim != 1:
            raise ShapeError("indices must be a 1D array")

    def forward(self, s: np.ndarray) -> np.ndarray:
        """Select subset of signal coordinates.
        
        Supports both single sample (dim_s,) and batch (B, dim_s).
        """
        s = np.asarray(s)
        # Use advanced indexing with ellipsis to support both shapes
        # For single: s[indices] works as before
        # For batch: s[..., indices] selects from last dimension
        return s[..., self.indices]
    
    def backprop_gradient(self, s: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """Chain rule gradient for mask projection.
        
        REQUIRED METHOD: For mask y = s[indices], gradient is placed back
        at selected indices, zeros elsewhere.
        """
        s = np.asarray(s)
        grad_y = np.asarray(grad_y)
        
        # Create zero gradient with same shape as s
        grad_s = np.zeros_like(s)
        
        # Place grad_y values at selected indices
        grad_s[..., self.indices] = grad_y
        
        return grad_s

    def grad_forward(self, s: np.ndarray) -> np.ndarray:
        """Return selection matrix.
        
        DEPRECATED: Use backprop_gradient() instead.
        
        Constructs a matrix where each row is a one-hot vector
        selecting the corresponding index.
        """
        s = np.asarray(s)
        n = s.size  # Total dimension of input signal
        m = self.indices.size  # Number of selected coordinates
        
        # Build selection matrix
        mat = np.zeros((m, n), dtype=float)
        mat[np.arange(m), self.indices] = 1.0
        return mat
