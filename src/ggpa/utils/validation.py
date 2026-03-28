"""Input validation utilities (placeholder)."""
from __future__ import annotations

from typing import Any


def ensure(condition: bool, msg: str) -> None:
    if not condition:
        raise ValueError(msg)
