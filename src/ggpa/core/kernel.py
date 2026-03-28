"""Stable fixed-tau kernel implementation."""
from __future__ import annotations

import logging as _logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import time

from ggpa.core.state import StepDiagnostics, State
from ggpa.core.interfaces import Server, Client, Transport
from ggpa.core.protocol import ClientRequest, ClientReply
from ggpa.core.errors import ConfigurationError
from ggpa.core.logging import get_logger
from ggpa.utils.utils import seed_for_client, seed_for_server
from ggpa.server.base import AggregationBase, ContextBase

logger = get_logger("kernel")


def _fmt_u(u):
    """Format u_tau for logging – works for scalar, array, or tensor."""
    try:
        return f"{float(u):.4f}"
    except (TypeError, ValueError):
        import numpy as _np
        a = _np.asarray(u)
        return f"mean={float(a.mean()):.4f} (shape={a.shape})"


@dataclass
class FixedTauKernel:
    """Stable core of GG-PA. DO NOT MODIFY once frozen.
    
    This kernel implements the fixed-tau GG-PA Gibbs iteration:
    1. Create requests with projected signal for all clients
    2. Collect denoised samples from clients via Request/Reply
    3. Aggregate samples into new signal state
    
    Key features:
    - Lightweight State (no xs field stored)
    - Request/Reply pattern for all client communication
    - On-demand sample fetching
    - Batch request support
    - Status codes for error handling
    """

    server: Server  # Uses Server Protocol
    clients: Dict[str, Client]  # Uses Client Protocol
    transport: Transport
    master_seed: Optional[int] = None

    def step(
        self,
        state: State,
        tau: float,
        compute_reduced_potential: bool = True,
    ) -> Tuple[State, StepDiagnostics]:
        """Perform one GG-PA Gibbs iteration at fixed tau.
        
        Process:
        1. Delegate to aggregator to sample new signal
           (aggregator fetches samples/gradients on-demand via helpers)
        2. Optionally compute reduced potential for diagnostics
        3. Return new state and diagnostics
        
        Args:
            state: Current GG-PA state (contains s, step, cache)
            tau: Diffusion time in [0, 1]
            compute_reduced_potential: If True (default), compute reduced
                potential U_τ for diagnostics.  Set to False in tight
                sampling loops (e.g. J scans) for ~2× speedup.
            
        Returns:
            Tuple of (new_state, diagnostics)
        """
        _log_info = logger.isEnabledFor(_logging.INFO)
        _log_debug = logger.isEnabledFor(_logging.DEBUG)
        
        if _log_info:
            logger.info(f"Step {state.step}: Starting GG-PA iteration with tau={tau:.4f}")
        t0 = time.time()
        
        # --- Aggregate: aggregator fetches what it needs on-demand ---
        if _log_debug:
            logger.debug(f"Aggregating with method={self.server.aggregator.__class__.__name__}")
        s_new, agg_diag = self.server.aggregate(
            s_current=state.s,
            tau=tau,
            server=self.server,
            transport=self.transport,
            context=self.server.context,
            seed=seed_for_server(self.master_seed, state.step),
            step=state.step,
        )
        
        # --- Optional reduced potential (skip for speed) ---
        reduced = None
        if compute_reduced_potential:
            reduced = self.server.reduced_potential(s_new, tau)
            if _log_debug:
                logger.debug(f"Reduced potential: u_tau={_fmt_u(reduced.u_tau)}")
        
        # --- New state ---
        new_state = State(s=s_new, step=state.step + 1, cache=dict(state.cache))
        
        # --- Signal norm (cheap, always compute) ---
        signal_norm = None
        try:
            import numpy as np
            if hasattr(s_new, 'norm'):
                signal_norm = float(s_new.norm())
            elif hasattr(s_new, '__array__'):
                signal_norm = float(np.linalg.norm(s_new))
        except Exception:
            pass
        
        wall_time = time.time() - t0
        
        if _log_info:
            u_str = _fmt_u(reduced.u_tau) if reduced is not None else "skip"
            norm_str = f", ||s||={signal_norm:.4f}" if signal_norm is not None else ""
            logger.info(
                f"Step {state.step}: Complete - tau={tau:.4f}{norm_str}, "
                f"u_tau={u_str}, time={wall_time:.3f}s"
            )
        
        diagnostics = StepDiagnostics(
            step=state.step,
            tau=tau,
            reduced_potential=reduced,
            aggregate_diagnostics=agg_diag,
            wall_time_s=wall_time,
            signal_norm=signal_norm,
        )
        return new_state, diagnostics

    def reduced_potential(self, state: State, tau: float):
        """Compute reduced potential at arbitrary tau.
        
        NOTE: Server fetches samples on-demand internally.
        This ensures the method works with lightweight State objects.
        
        Args:
            state: Current GG-PA state
            tau: Diffusion time in [0, 1]
            
        Returns:
            ReducedPotential object with energy components
        """
        logger.debug(f"Computing reduced potential at tau={tau:.4f}")
        reduced = self.server.reduced_potential(state.s, tau)
        logger.debug(f"Reduced potential computed: u_tau={_fmt_u(reduced.u_tau)}")
        return reduced

    @classmethod
    def from_clients(
        cls,
        clients: Dict[str, Any],  # Client instances
        aggregator: AggregationBase,
        context: Optional[ContextBase] = None,
        master_seed: Optional[int] = None,
    ) -> "FixedTauKernel":
        """Create FixedTauKernel from Client instances.
        
        Automatically creates CentralServer and registers clients.
        
        Example:
            >>> clients = {
            ...     "dog": MyClient(dog_model),
            ...     "cat": MyClient(cat_model),
            ... }
            >>> kernel = FixedTauKernel.from_clients(
            ...     clients=clients,
            ...     aggregator=GradientMCMCAggregator(),
            ...     context=UniformContext(),
            ...     master_seed=42
            ... )
        
        Args:
            clients: Dict of Client instances
            aggregator: Aggregation kernel to use
            context: Context density (default: UniformContext)
            master_seed: Random seed for reproducibility
            
        Returns:
            FixedTauKernel ready to use
        """
        from ggpa.server.context import UniformContext
        from ggpa.server.server import CentralServer
        from ggpa.transport.local import LocalTransport
        
        logger.info("Creating FixedTauKernel from client instances")
        
        if context is None:
            context = UniformContext()
            logger.debug("Using default UniformContext")
        
        # Create server
        server = CentralServer(
            context=context,
            aggregator=aggregator,
        )
        logger.debug("Created CentralServer")
        
        # Create transport
        transport = LocalTransport(clients=clients)
        logger.debug("Created LocalTransport")
        
        # Register clients with server
        server.register_clients(clients, transport)
        logger.info(f"Registered {len(clients)} clients with server")
        
        # Summary
        logger.info(
            f"FixedTauKernel initialized: {len(clients)} clients, "
            f"context={type(context).__name__}, aggregator={type(aggregator).__name__}"
        )
        
        return cls(
            server=server,
            clients=clients,
            transport=transport,
            master_seed=master_seed,
        )
