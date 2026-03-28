"""Gibbs state definitions for GG-PA.

IMPORTANT: State is LIGHTWEIGHT - does NOT store xs (per-client samples).
Samples are fetched on-demand via Request/Reply pattern when needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .diagnostics import ReducedPotential


@dataclass
class State:
    """Lightweight Gibbs state for GG-PA.

    ARCHITECTURE: State no longer stores xs (per-client samples).
    Samples are fetched on-demand via ClientRequest/ClientReply pattern.

    Benefits:
    - Lightweight: Smaller state for network transmission
    - Efficient: Better for replica exchange (less data to swap)
    - Flexible: Samples fetched only when needed

    Attributes:
        s: Current signal in the signal domain (any type: numpy, torch, jax, etc.)
        step: Current step index
        rng_state: Optional random state for reproducibility
        cache: Optional extra storage (not used by core, available for extensions)
    """

    s: Any
    step: int = 0
    rng_state: Optional[Any] = None
    cache: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "State":
        """Create a shallow copy for reproducibility.
        
        NOTE: Does NOT clone s itself (assumed immutable or managed externally).
        """
        return State(
            s=self.s,
            step=self.step,
            rng_state=self.rng_state,
            cache={k: v for k, v in self.cache.items()},
        )


@dataclass
class StepDiagnostics:
    """Diagnostics for a single fixed-tau step."""

    step: int
    tau: float
    reduced_potential: Optional[ReducedPotential] = None
    client_diagnostics: Dict[str, Any] = field(default_factory=dict)
    aggregate_diagnostics: Dict[str, Any] = field(default_factory=dict)
    wall_time_s: float | None = None
    signal_norm: float | None = None
