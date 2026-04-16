"""Protocol definitions for GG-PA core components.

This module defines the THREE core protocols that the kernel depends on:
1. Server - coordinates GG-PA iterations
2. Client - handles sampling requests
3. Transport - manages client communication

All other base classes (AggregationBase, ContextBase, ProjectorBase, ForwardProcessBase)
are now in their respective modules:
- server.base: AggregationBase, ContextBase
- client.base: ProjectorBase, ForwardProcessBase
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Union

from .protocol import ClientReply, ClientRequest


class Client(Protocol):
    """Client interface using Request/Reply pattern.
    
    Minimal protocol for clients - all communication via handle_request().
    
    Required attributes:
        client_id: str - Unique identifier for this client
    
    Required methods:
        handle_request(request: ClientRequest) -> ClientReply
    
    Design philosophy:
        - Simple: One method handles all request types
        - Flexible: Batch support via request_types: Union[str, List[str]]
        - Robust: Status codes indicate success/error/partial/unsupported
        - Network-friendly: Easy to serialize and send over network
    """
    
    client_id: str
    
    def handle_request(self, request: ClientRequest) -> ClientReply:
        """Process a client request and return a reply.
        
        Args:
            request: ClientRequest with s, t_diff, request_types
            
        Returns:
            ClientReply with status_code, error (if any), and data dict
        """


class Server(Protocol):
    """Server interface for coordinating GG-PA iterations.
    
    The server orchestrates the Request/Reply cycle and provides
    unified interfaces for computing log-likelihoods and reduced potentials.
    
    Key responsibilities:
    1. create_requests(): Create ClientRequest objects
    2. aggregate(): Combine samples into new s (fetches on-demand)
    3. compute_gradient(): Compute ∇_s log p(Φ(s)) via chain rule
    4. query_client_properties(): Query client metadata
    5. reduced_potential(): Calculate U_t_diff(s, {x_i})
    """

    context: Any  # ContextBase
    aggregator: Any  # AggregationBase
    
    def create_requests(
        self,
        s: Any,
        t_diff: float,
        request_types: Union[str, List[str]],
        step: Optional[int] = None
    ) -> Dict[str, ClientRequest]:
        """Create ClientRequest objects for all clients.
        
        Args:
            s: Current shared state
            t_diff: Diffusion time parameter
            request_types: What to request ('sample', 'gradient', 'log_prob', etc.)
            step: Current iteration number (for seeding)
            
        Returns:
            Dict mapping client_id to ClientRequest
        """

    def aggregate(
        self,
        s_current: Any,
        t_diff: float,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """Aggregate client samples into new shared state.
                
        Args:
            s_current: Current shared state
            t_diff: Diffusion time parameter
            **kwargs: Additional context (seed, step, server, transport, etc.)
            
        Returns:
            Tuple of (s_new, diagnostics)
        """

    def compute_gradient(
        self,
        s: Any,
        t_diff: float
    ) -> Dict[str, Any]:
        """Compute ∇_s log p(Φ_i(s)) for all clients.
        
        Uses chain rule: ∇_s log p(Φ(s)) = (∂Φ/∂s)^T @ ∇_y log p(y)
        
        Args:
            s: Current shared state
            t_diff: Diffusion time parameter
            
        Returns:
            Dictionary mapping client_id to gradient arrays
        """

    def query_client_properties(
        self,
        property_names: Union[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Query client properties/metadata.
        
        Args:
            property_name: Name of property to query
            
        Returns:
            Dictionary mapping client_id to property value
        """

    def reduced_potential(
        self,
        s: Any,
        t_diff: float
    ) -> 'ReducedPotential':
        """Compute reduced potential U_t_diff(s, {x_i}).
        
        The reduced potential combines context and likelihood terms:
        U_t_diff(s, {x_i}) = -log π(s)^β(t_diff) - Σ_i log q_t_diff(y_i | x_i)
        
        Args:
            s: Current shared state
            t_diff: Diffusion time parameter
            
        Returns:
            ReducedPotential object
        """


class Transport(Protocol):
    """Transport interface for client communication."""

    def call(self, client_id: str, request: ClientRequest) -> ClientReply:
        """Call a single client."""

    def call_all(self, requests: Iterable[ClientRequest]) -> Dict[str, ClientReply]:
        """Call all clients.
        
        Args:
            requests: Iterable of ClientRequest objects
            
        Returns:
            Dict mapping client_id to ClientReply
        """
