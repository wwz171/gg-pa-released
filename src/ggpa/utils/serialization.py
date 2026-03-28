"""Serialization helpers (placeholder)."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj
