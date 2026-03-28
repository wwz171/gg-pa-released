"""Minimal MBAR reweighting utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MBARResult:
    """Result of MBAR reweighting."""

    free_energies: np.ndarray
    weights: np.ndarray


def mbar_weights(u_kn: np.ndarray, n_k: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> MBARResult:
    """Compute MBAR weights for samples across K states.

    Args:
        u_kn: Reduced potentials with shape (K, N).
        n_k: Sample counts per state with shape (K,).
    """
    u_kn = np.asarray(u_kn, dtype=float)
    n_k = np.asarray(n_k, dtype=float)
    k_states, n_samples = u_kn.shape

    f_k = np.zeros(k_states, dtype=float)
    for _ in range(max_iter):
        f_prev = f_k.copy()
        denom = np.zeros(n_samples, dtype=float)
        for k in range(k_states):
            denom += n_k[k] * np.exp(f_k[k] - u_kn[k])
        for k in range(k_states):
            f_k[k] = -np.log(np.sum(np.exp(-u_kn[k]) / denom))
        if np.max(np.abs(f_k - f_prev)) < tol:
            break

    denom = np.zeros(n_samples, dtype=float)
    for k in range(k_states):
        denom += n_k[k] * np.exp(f_k[k] - u_kn[k])
    weights = 1.0 / denom
    weights /= np.sum(weights)
    return MBARResult(free_energies=f_k, weights=weights)
