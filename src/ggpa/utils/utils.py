"""Utility helpers for GG-PA."""
from __future__ import annotations

from typing import Optional

import hashlib
import numpy as np


def clamp_tau(tau: float, tau_min: float) -> float:
    """Clamp tau to [tau_min, 1.0]."""
    return float(max(tau_min, min(1.0, tau)))


def seed_for_client(master_seed: Optional[int], step: int, client_id: str) -> Optional[int]:
    """Deterministically derive a per-client seed."""
    if master_seed is None:
        return None
    digest = hashlib.sha256(client_id.encode("utf-8")).digest()
    client_int = int.from_bytes(digest[:4], byteorder="little", signed=False)
    seed_seq = np.random.SeedSequence([int(master_seed), int(step), client_int])
    return int(seed_seq.generate_state(1)[0])


def seed_for_server(master_seed: Optional[int], step: int) -> Optional[int]:
    """Deterministically derive a server seed for aggregation."""
    if master_seed is None:
        return None
    seed_seq = np.random.SeedSequence([int(master_seed), int(step), 0xA5A5A5A5])
    return int(seed_seq.generate_state(1)[0])


def rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    """Create a numpy RNG from an optional seed."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))
