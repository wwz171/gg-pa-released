"""Central server implementation.

NOTE: ServerBase already provides all functionality via default implementations!
CentralServer exists only for backward compatibility and convenience.
You can use ServerBase directly in most cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ggpa.server.base import AggregationBase, ContextBase, ServerBase


@dataclass
class CentralServer(ServerBase):
    """Central server for GG-PA (simplified implementation).
    
    This is just a thin wrapper around ServerBase for convenience.
    ServerBase already provides all functionality:
    - create_requests(s, t_diff, request_types, step)
    - aggregate(s_current, t_diff, **kwargs)
    - compute_gradient(s, t_diff)
    - reduced_potential(s, t_diff)
    - query_client_properties(property_names)
    - register_clients(clients, transport)
    
    Users only need to:
    1. Provide context and aggregator
    2. Call register_clients() to set up communication
    
    Example:
        >>> from ggpa.server.server import CentralServer
        >>> from ggpa.server.context import UniformContext
        >>> from ggpa.server.aggregation import GradientMCMCAggregator
        >>> 
        >>> server = CentralServer(
        ...     context=UniformContext(),
        ...     aggregator=GradientMCMCAggregator()
        ... )
        >>> 
        >>> # Register clients and transport
        >>> server.register_clients(clients, transport)
        >>> 
        >>> # All methods now work automatically!
        >>> gradient = server.compute_gradient(s, t_diff)
        >>> u_t_diff = server.reduced_potential(s, t_diff)
        >>> s_new, diagnostics = server.aggregate(s, t_diff, seed=42, step=0)
    """
    
    context: ContextBase
    aggregator: AggregationBase
    
    # All functionality is inherited from ServerBase!
    # No additional implementation needed.
