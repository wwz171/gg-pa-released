"""Lightweight top-level exports for the GG-PA package."""

from importlib.metadata import PackageNotFoundError, version

from ggpa.client.base import ClientBase, ProjectorBase, ForwardProcessBase
from ggpa.client.forward_processes import GaussianForwardProcess, NoiseSchedule
from ggpa.client.projectors.identity import IdentityProjector
from ggpa.client.projectors.linear import LinearProjector
from ggpa.client.projectors.mask import MaskProjector
from ggpa.core.diagnostics import ReducedPotential
from ggpa.core.kernel import FixedTauKernel
from ggpa.core.state import State, StepDiagnostics
from ggpa.drivers.base import AnnealingDriver, FixedTauSampler, ReplicaExchangeDriver
from ggpa.server.aggregation import GradientMCMCAggregator, RandomWalkMCMCAggregator
from ggpa.server.base import ServerBase, AggregationBase, ContextBase
from ggpa.server.context import UniformContext
from ggpa.server.server import CentralServer
from ggpa.transport.local import LocalTransport
from ggpa.transport.rpc import RPCTransport
from ggpa.utils.mbar import mbar_weights, MBARResult

try:
    __version__ = version("ggpa")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    # Base classes for user customization
    "ServerBase",
    "ClientBase",
    "AggregationBase",
    "ContextBase",
    "ProjectorBase",
    "ForwardProcessBase",
    # Built-in implementations
    "GradientMCMCAggregator",
    "RandomWalkMCMCAggregator",
    "UniformContext",
    "CentralServer",
    "FixedTauKernel",
    "ReducedPotential",
    "StepDiagnostics",
    "State",
    "GaussianForwardProcess",
    "NoiseSchedule",
    "IdentityProjector",
    "LinearProjector",
    "MaskProjector",
    "LocalTransport",
    "RPCTransport",
    "AnnealingDriver",
    "FixedTauSampler",
    "ReplicaExchangeDriver",
    "mbar_weights",
    "MBARResult",
    "__version__",
]
