"""Base classes for server implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ggpa.core.diagnostics import ReducedPotential
from ggpa.core.protocol import ClientRequest, ClientReply
from ggpa.core.interfaces import Server
from ggpa.core.errors import ConfigurationError
from ggpa.core.logging import get_logger

logger = get_logger("server")


class ServerBase(ABC, Server):
    """Abstract base class for Server implementations.
    
    Provides common structure and documentation for server implementations.
    Most methods have default implementations - users rarely need to override.
    
    The server is the central coordinator in GG-PA, managing:
    1. Request creation: s → ClientRequest objects
    2. Aggregation: Fetch samples on-demand, update s
    3. Computation: log-likelihoods, gradients, reduced potentials
    
    KEY DESIGN:
    - Lightweight: State doesn't store xs, fetched on-demand
    - Unified: Request/Reply pattern for all communication
    - Flexible: Batch support, status codes, error handling
    - Minimal user code: Most methods have defaults
    
    REQUIRED ATTRIBUTES:
    ====================
        context (ContextBase): Context density π(s)
        aggregator (AggregationBase): Aggregation strategy
    
    SETUP:
    ======
        Must call register_clients() before using default methods.
    """

    context: Any  # ContextBase
    aggregator: Any  # AggregationBase
    
    # Internal state (set by register_clients)
    _client_registry: Optional[Dict[str, Any]] = None
    _transport: Optional[Any] = None
    master_seed: int = 0  # For deterministic seeding
    
    def register_clients(self, clients: Dict[str, Any], transport: Any) -> None:
        """Register clients and transport for default methods.
        
        Called by FixedDiffusionTimeKernel during initialization.
        Required for default implementations of create_requests, query_client_properties, etc.
        
        Performs validation:
        1. Checks all clients have handle_request method
        2. Tests communication with each client
        3. Verifies basic request/reply functionality
        
        Args:
            clients: Dictionary mapping client_id to Client instances
            transport: Transport instance for communication
            
        Raises:
            ConfigurationError: If clients don't meet requirements
        """
        logger.info(f"Registering {len(clients)} clients with server")
        
        # Validate clients have required interface
        for client_id, client in clients.items():
            if not hasattr(client, 'handle_request'):
                raise ConfigurationError(
                    f"Client '{client_id}' must have handle_request() method"
                )
            
            if not hasattr(client, 'client_id'):
                raise ConfigurationError(
                    f"Client '{client_id}' must have client_id attribute"
                )
            
            logger.debug(f"  - Validated client '{client_id}' interface")
        
        # Store registry
        self._client_registry = clients
        self._transport = transport
        
        # Test communication with each client
        logger.info("Testing communication with clients...")
        test_s = 0.0  # Dummy signal for testing
        test_t_diff = 0.5
        
        for client_id in clients.keys():
            try:
                # Create test request for properties
                test_request = ClientRequest(
                    client_id=client_id,
                    s=test_s,
                    t_diff=test_t_diff,
                    request_types='properties'
                )
                
                # Send via transport
                reply = transport.call(client_id, test_request)
                
                # Check reply
                if reply.status_code == 'error':
                    logger.warning(
                        f"  - Client '{client_id}' returned error: {reply.error}"
                    )
                else:
                    logger.info(
                        f"  ✓ Client '{client_id}' communication OK "
                        f"(status: {reply.status_code})"
                    )
                    
                    # Log available properties
                    if 'properties' in reply.data:
                        props = reply.data['properties']
                        logger.debug(f"    Properties: {list(props.keys())}")
                
            except Exception as e:
                logger.error(
                    f"  ✗ Failed to communicate with client '{client_id}': {e}"
                )
                raise ConfigurationError(
                    f"Cannot communicate with client '{client_id}': {e}"
                )
        
        logger.info(f"Successfully registered {len(clients)} clients")
    
    def _generate_seed(self, step: Optional[int], client_id: str) -> Optional[int]:
        """Generate deterministic seed for client.
        
        Args:
            step: Current step number
            client_id: Client identifier
            
        Returns:
            Deterministic seed or None if step is None
        """
        if step is None:
            return None
        
        from ggpa.utils.utils import seed_for_client
        return seed_for_client(self.master_seed, step, client_id)

    def create_requests(
        self,
        s: Any,
        t_diff: float,
        request_types: Union[str, List[str]],
        step: Optional[int] = None
    ) -> Dict[str, ClientRequest]:
        """Create ClientRequest objects for all clients (DEFAULT IMPLEMENTATION).
        
        Automatically creates requests for all registered clients.
        
        Args:
            s: Current shared state
            t_diff: Diffusion time parameter
            request_types: What to request ('sample', 'gradient', 'log_prob', 'properties')
            step: Current iteration number (for diagnostics/seeding)
            
        Returns:
            Dict mapping client_id to ClientRequest
            
        Raises:
            ConfigurationError: If clients not registered via register_clients()
        """
        from ggpa.core.errors import ConfigurationError
        
        if self._client_registry is None:
            raise ConfigurationError(
                "Server must register clients via register_clients() before calling create_requests()"
            )
        
        requests = {}
        for client_id in self._client_registry.keys():
            requests[client_id] = ClientRequest(
                client_id=client_id,
                s=s,
                t_diff=t_diff,
                request_types=request_types,
                request_id=f"step{step}_{client_id}" if step is not None else None,
                seed=self._generate_seed(step, client_id),
                step=step
            )
        
        return requests

    def aggregate(
        self,
        s_current: Any,
        t_diff: float,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """Aggregate client samples into new shared state (DEFAULT IMPLEMENTATION).
        
        Automatically delegates to self.aggregator with all necessary kwargs.
        
        KEY DESIGN:
        - NO xs parameter! Samples fetched on-demand
        - Auto-injects server, transport, context into kwargs
        - User only implements aggregator.aggregate()
        
        Args:
            s_current: Current shared state
            t_diff: Diffusion time parameter
            **kwargs: Additional context (seed, step, etc.)
            
        Returns:
            Tuple of (s_new, diagnostics)
            - s_new: Updated shared state
            - diagnostics: Dict with aggregation method info
            
        Raises:
            ConfigurationError: If aggregator not set or transport not registered
        """
        from ggpa.core.errors import ConfigurationError
        
        if not hasattr(self, 'aggregator') or self.aggregator is None:
            raise ConfigurationError("Server must have aggregator attribute")
        
        if self._transport is None:
            raise ConfigurationError(
                "Server must register transport via register_clients() before aggregating"
            )
        
        # Auto-inject framework objects into kwargs
        aggregate_kwargs = {
            'server': self,
            'transport': self._transport,
            'context': self.context,
            **kwargs  # User-provided kwargs (seed, step, etc.)
        }
        
        # Delegate to aggregator
        return self.aggregator.aggregate(s_current, t_diff, **aggregate_kwargs)
    def compute_gradient(
        self,
        s: Any,
        t_diff: float
    ) -> Dict[str, Any]:
        """Compute ∇_s log q_t_diff(y_i | x_i) for all clients (DEFAULT IMPLEMENTATION).
        
        Uses chain rule: ∇_s = (∂Φ/∂s)^T @ ∇_y log q_t_diff(y | x)
        Client automatically handles:
        1. Projection: y = Φ(s)
        2. Uses cached x from client._current_x (no re-denoise)
        3. Gradient: ∇_y log q_t_diff(y | x) via forward_process
        4. Chain rule: ∇_s via projector
        
        Args:
            s: Current shared state
            t_diff: Diffusion time parameter
            
        Returns:
            Dictionary mapping client_id to gradient arrays (same shape as s)
            
        Raises:
            ConfigurationError: If transport not registered
        """
        from ggpa.core.errors import ConfigurationError
        
        if self._transport is None:
            raise ConfigurationError(
                "Server must register transport via register_clients() before computing gradients"
            )
        
        # Request ONLY gradient — client uses cached _current_x (no re-denoise)
        requests = self.create_requests(s, t_diff, request_types='gradient')
        replies = self._transport.call_all(requests.values())
        
        # Extract gradients
        gradients = {}
        for client_id, reply in replies.items():
            if reply.status_code in ['success', 'partial']:
                if 'gradient' in reply.data and reply.data['gradient'] is not None:
                    gradients[client_id] = reply.data['gradient']
        
        return gradients

    def query_client_properties(
        self,
        property_names: Union[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Query client properties/metadata (DEFAULT IMPLEMENTATION).
        
        Automatically queries all registered clients via Request/Reply.
        
        Args:
            property_names: Single property name or list of property names to query
            
        Returns:
            Dictionary mapping client_id to dict of {property_name: value}
            If a property is not found, its value will be None.
            
        Example:
            # Query single property
            props = server.query_client_properties('projector_type')
            # Returns: {'client1': {'projector_type': 'Identity'}, 'client2': {...}}
            
            # Query multiple properties
            props = server.query_client_properties(['projector_type', 'forward_process_type'])
            # Returns: {'client1': {'projector_type': 'Identity', 'forward_process_type': 'Gaussian'}, ...}
            
        Raises:
            ConfigurationError: If clients/transport not registered
        """
        from ggpa.core.errors import ConfigurationError
        
        if self._client_registry is None or self._transport is None:
            raise ConfigurationError(
                "Server must register clients via register_clients() before querying properties"
            )
        
        # Standardize property_names to list
        if isinstance(property_names, str):
            property_names = [property_names]
        
        # Create 'properties' requests with property_names in metadata
        requests = {}
        for client_id in self._client_registry.keys():
            requests[client_id] = ClientRequest(
                client_id=client_id,
                s=None,  # Properties don't need s
                t_diff=0.0,  # Arbitrary value
                request_types='properties',
                metadata={'property_names': property_names}
            )
        
        # Call clients
        replies = self._transport.call_all(requests.values())
        
        # Extract properties
        result = {}
        for client_id, reply in replies.items():
            if reply.status_code in ['success', 'partial']:
                result[client_id] = reply.data.get('properties', {})
        
        return result

    def reduced_potential(
        self,
        s: Any,
        t_diff: float
    ) -> ReducedPotential:
        """Compute reduced potential U_t_diff(s, {x_i}) (DEFAULT IMPLEMENTATION).
        
        The reduced potential is the negative log of the joint distribution:
        U_t_diff(s, {x_i}) = -log[π(s)^β(t_diff) * ∏_i q_t_diff(y_i | x_i)]
                      = -β(t_diff) * log π(s) - Σ_i log q_t_diff(y_i | x_i)
        
        Components:
        - Context term: -β(t_diff) * log π(s) from self.context
        - Likelihood terms: -log q_t_diff(y_i | x_i) from clients via Request/Reply
        
        Clients use their cached _current_x (set during the last 'sample'
        request, e.g. from kernel.step) instead of re-denoising:
        1. Projection: y_i = Φ_i(s)
        2. Uses cached x_i from client._current_x
        3. Log prob: log q_t_diff(y_i | x_i)
        
        Args:
            s: Current shared state
            t_diff: Diffusion time parameter
            
        Returns:
            ReducedPotential object with:
            - t_diff: Diffusion time
            - log_q_ctx: Context contribution
            - log_q_fwd: Dict of per-client likelihood contributions
            - total_log_q: Sum of all log probabilities
            - u_t_diff: Reduced potential (negative total_log_q)
            
        Raises:
            ConfigurationError: If context not set or transport not registered
        """
        from ggpa.core.errors import ConfigurationError
        
        if not hasattr(self, 'context') or self.context is None:
            raise ConfigurationError("Server must have context attribute")
        
        if self._transport is None:
            raise ConfigurationError(
                "Server must register transport via register_clients() before computing reduced potential"
            )
        
        # 1. Context term: β(t_diff) * log π(s)
        # Shape: same as what context.log_q_ctx() returns (user-defined)
        log_q_ctx = self.context.log_q_ctx(s, t_diff)
        
        # 2. Likelihood terms: log q_t_diff(y_i | x_i) for each client
        # Request ONLY log_prob — client uses cached _current_x (no re-denoise)
        requests = self.create_requests(s, t_diff, request_types=['log_prob'])
        replies = self._transport.call_all(requests.values())
        
        # 3. Collect results and check shape consistency
        # Shape: preserved from client log_prob returns
        log_q_fwd = {}
        total = log_q_ctx
        
        # Get reference shape from log_q_ctx
        ref_shape = np.asarray(log_q_ctx).shape
        
        for client_id, reply in replies.items():
            if reply.status_code in ['success', 'partial']:
                if 'log_prob' in reply.data and reply.data['log_prob'] is not None:
                    log_q = reply.data['log_prob']
                    log_q_shape = np.asarray(log_q).shape
                    
                    # Check shape compatibility for addition
                    # Allow broadcasting: scalar + (B,) = (B,), but catch incompatible shapes
                    try:
                        _ = total + log_q  # Test if addition is valid
                    except (ValueError, TypeError) as e:
                        from ggpa.core.errors import ShapeError
                        raise ShapeError(
                            f"Shape mismatch in reduced_potential: "
                            f"log_q_ctx shape {ref_shape}, client '{client_id}' log_prob shape {log_q_shape}. "
                            f"Cannot add: {e}"
                        ) from e
                    
                    log_q_fwd[client_id] = log_q
                    total = total + log_q  # Element-wise addition with broadcasting
        
        # 4. Construct ReducedPotential
        u_t_diff = -total
        return ReducedPotential(
            t_diff=t_diff,
            log_q_ctx=log_q_ctx,
            log_q_fwd=log_q_fwd,
            total_log_q=total,
            u_t_diff=u_t_diff
        )


# ========== Server Internal Components ==========
# These are NOT Protocols! They are ABC (Abstract Base Classes).
# Users inherit from these to implement custom aggregators and contexts.


class AggregationBase(ABC):
    """Abstract base class for aggregation implementations.
    
    This is THE CORE of GG-PA - it defines how to sample s from p(s | {x_i}, t_diff).
    
    KEY DESIGN:
    - Signature: aggregate(s_current, t_diff, **kwargs) - NO xs parameter!
    - Helper methods: fetch_samples(), fetch_gradients(), fetch_log_probs()
    - Flexible: Extract only what you need from kwargs
    
    All aggregation strategies must inherit from this class and implement aggregate().
    The framework passes all necessary data via **kwargs for maximum flexibility.
    
    HELPER METHODS (provided by base):
    ===================================
        fetch_samples(s, t_diff, server, transport): Get all client samples
        fetch_gradients(s, t_diff, server, transport): Get all client gradients
        fetch_log_probs(s, t_diff, server, transport): Get all client log_probs
    
    Required method:
        aggregate(s_current, t_diff, **kwargs): Sample new signal from posterior
    
    Example (using helpers):
        class MyMCMCAggregator(AggregationBase):
            def __init__(self, n_steps=100):
                self.n_steps = n_steps
            
            def aggregate(self, s_current, t_diff, **kwargs):
                server = kwargs['server']
                transport = kwargs['transport']
                context = kwargs['context']
                
                # Use helper to fetch samples (easy!)
                xs = self.fetch_samples(s_current, t_diff, server, transport)
                
                # Your aggregation logic
                s_new = my_aggregation_method(s_current, xs, t_diff, context)
                return s_new, {"method": "custom"}
    """
    
    # ========== HELPER METHODS (provided by base) ==========
    
    def fetch_samples(
        self,
        s: Any,
        t_diff: float,
        server: Any,
        transport: Any
    ) -> Dict[str, Any]:
        """Helper: Fetch samples from all clients.
        
        Args:
            s: Current signal
            t_diff: Diffusion time
            server: Server instance (provides create_requests)
            transport: Transport instance (provides call_all)
            
        Returns:
            Dictionary mapping client_id to sample
        """
        requests = server.create_requests(s, t_diff, request_types='sample')
        replies = transport.call_all(requests.values())
        
        xs = {}
        for client_id, reply in replies.items():
            if reply.status_code in ['success', 'partial']:
                if 'sample' in reply.data and reply.data['sample'] is not None:
                    xs[client_id] = reply.data['sample']
        
        return xs
    
    def fetch_gradients(
        self,
        s: Any,
        t_diff: float,
        server: Any,
        transport: Any
    ) -> Dict[str, Any]:
        """Helper: Fetch gradients from all clients.
        
        Uses cached _current_x in each client (set during the last 'sample'
        request). Does NOT trigger re-denoise.
        
        Args:
            s: Current signal
            t_diff: Diffusion time
            server: Server instance
            transport: Transport instance
            
        Returns:
            Dictionary mapping client_id to gradient
        """
        # Request ONLY gradient — client uses cached _current_x (no re-denoise)
        requests = server.create_requests(s, t_diff, request_types='gradient')
        replies = transport.call_all(requests.values())
        
        grads = {}
        for client_id, reply in replies.items():
            if reply.status_code in ['success', 'partial']:
                if 'gradient' in reply.data and reply.data['gradient'] is not None:
                    grads[client_id] = reply.data['gradient']
        
        return grads
    
    def fetch_log_probs(
        self,
        s: Any,
        t_diff: float,
        server: Any,
        transport: Any
    ) -> Dict[str, float]:
        """Helper: Fetch log probabilities from all clients.
        
        Uses cached _current_x in each client (set during the last 'sample'
        request). Does NOT trigger re-denoise.
        
        Args:
            s: Current signal
            t_diff: Diffusion time
            server: Server instance
            transport: Transport instance
            
        Returns:
            Dictionary mapping client_id to log_prob (float)
        """
        # Request ONLY log_prob — client uses cached _current_x (no re-denoise)
        requests = server.create_requests(s, t_diff, request_types='log_prob')
        replies = transport.call_all(requests.values())
        
        log_probs = {}
        for client_id, reply in replies.items():
            if reply.status_code in ['success', 'partial']:
                if 'log_prob' in reply.data and reply.data['log_prob'] is not None:
                    log_probs[client_id] = float(reply.data['log_prob'])
        
        return log_probs
    
    @abstractmethod
    def aggregate(
        self,
        s_current: Any,
        t_diff: float,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """Aggregate client samples into new signal.
        
        This is the CORE method that all aggregation kernels must implement.
        
        KEY DESIGN:
        - NO xs parameter! Fetch samples on-demand via Request/Reply
        - Only 2 required params: s_current, t_diff
        - Everything else via **kwargs
        
        Args:
            s_current: Current signal state (any format)
            t_diff: Diffusion time in [0, 1]
            **kwargs: All data user might need:
                
                Always provided by framework:
                    server: Server instance (provides create_requests, compute_gradient)
                    transport: Transport instance (provides call_all)
                    context: ContextBase - context constraint
                
                Optional (for advanced use):
                    seed: Optional[int] - random seed
                    step: Optional[int] - current step number
        
        Returns:
            (s_new, diagnostics):
                - s_new: New signal (same format as s_current)
                - diagnostics: Dict with aggregation info
        
        Example:
            def aggregate(self, s_current, t_diff, **kwargs):
                server = kwargs['server']
                transport = kwargs['transport']
                
                # Fetch samples
                requests = server.create_requests(s_current, t_diff, 'sample')
                replies = transport.call_all(requests.values())
                xs = {r.client_id: r.data['sample'] for r in replies}
                
                # Your aggregation logic here
                s_new = my_aggregation_method(s_current, xs, t_diff)
                return s_new, {"method": "custom"}
        """
        pass


class ContextBase(ABC):
    """Abstract base class for context density implementations.
    
    Context provides constraints on the global signal s, tempered by t_diff.
    
    Users only need to implement log_prob() for basic functionality.
    Optional: grad_log_prob() for gradient-based aggregation.
    
    The context density π(s) is tempered by f(t_diff) to get π(s)^{f(t_diff)}.
    Default tempering: f(t_diff) = 1 - t_diff (linear annealing from full constraint to none).
    
    Required methods:
        log_prob(s): Unnormalized log probability log π(s)
    
    Optional methods:
        grad_log_prob(s): Gradient ∇_s log π(s) for gradient-based aggregators
        tempering_factor(t_diff): Custom tempering schedule (default: 1 - t_diff)
    """
    
    @abstractmethod
    def log_prob(self, s: Any, t_diff: float) -> Any:
        """Log probability log p_ctx(s) (unnormalized OK).
        
        Args:
            s: Signal (any format)
            t_diff: Diffusion time
            
        Returns:
            Log probability (shape flexible: scalar, (B,), (B,D), etc.)
            
        Note:
            For batch processing, return shape (B,) for per-sample log probs.
            For scalar reduction, return a single float.
            The shape will be preserved through reduced_potential().
        """
        pass

    def tempering_factor(self, t_diff: float) -> float:
        """Tempering schedule f(t_diff) controlling constraint strength.
        
        Default: f(t_diff) = 1 - t_diff (linear annealing)
        - t_diff = 0: f(0) = 1, full constraint
        - t_diff = 1: f(1) = 0, no constraint
        
        Override this method to use custom tempering schedules.
        
        Args:
            t_diff: Diffusion time in [0, 1]
            
        Returns:
            Tempering factor in [0, 1]
        """
        return 1.0 - t_diff

    def log_q_ctx(self, s: Any, t_diff: float) -> Any:
        """Tempered log probability: f(t_diff) * log p_ctx(s).
        
        Usually doesn't need overriding - uses tempering_factor().
        
        Args:
            s: Signal (any format)
            t_diff: Diffusion time
            
        Returns:
            Tempered log probability (shape determined by log_prob implementation)
        """
        return self.tempering_factor(t_diff) * self.log_prob(s, t_diff)

    def grad_log_prob(self, s: Any, t_diff: float) -> Optional[Any]:
        """Gradient ∇_s log p_ctx(s) for gradient-based aggregation.
        
        Optional: Return None if not needed or not computable.
        Required only for gradient-based aggregators (e.g., Langevin dynamics).
        
        Args:
            s: Signal (any format)
            t_diff: Diffusion time
            
        Returns:
            Gradient same shape as s (any format), or None
        """
        return None
