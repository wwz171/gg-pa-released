"""Core GG-PA components.

Core protocols (used by FixedDiffusionTimeKernel):
- Server: Coordinates GG-PA iterations
- Client: Handles sampling requests
- Transport: Manages client communication

Base classes are now in their respective modules:
- server.base: AggregationBase, ContextBase, ServerBase
- client.base: ClientBase, ProjectorBase, ForwardProcessBase
"""
from ggpa.core.interfaces import (
    Server,
    Client,
    Transport,
)
from ggpa.core.kernel import FixedDiffusionTimeKernel
from ggpa.core.state import State, StepDiagnostics

__all__ = [
    "Server",
    "Client",
    "Transport",
    "FixedDiffusionTimeKernel",
    "State",
    "StepDiagnostics",
]
