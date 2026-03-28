"""Diagnostics and reduced potential structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ReducedPotential:
    """Reduced potential terms for a given tau.

    Attributes:
        tau: Diffusion time.
        log_q_ctx: Context term (possibly unnormalized).
        log_q_fwd: Per-client forward log-likelihoods.
        total_log_q: Sum of all log terms.
        u_tau: Reduced potential = -total_log_q.
    """

    tau: float
    log_q_ctx: float
    log_q_fwd: Dict[str, float]
    total_log_q: float
    u_tau: float


__all__ = ["ReducedPotential"]
