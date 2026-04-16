"""Aggregation kernels for GG-PA.

This module provides built-in aggregation strategies:
- GradientMCMCAggregator: Gradient-based MCMC (recommended for most cases)
- RandomWalkMCMCAggregator: Random-walk MCMC (fallback without gradients)

KEY DESIGN:
- Signature: aggregate(s_current, t_diff, **kwargs) - NO xs parameter
- Samples fetched on-demand via Request/Reply pattern
- Base class AggregationBase is in server.base module
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np

from ggpa.core.errors import NotSupportedError
from ggpa.core.logging import get_logger
from ggpa.server.base import AggregationBase
from ggpa.utils.utils import rng_from_seed

logger = get_logger("aggregation")


@dataclass
class GradientMCMCAggregator(AggregationBase):
    """Gradient-based MCMC aggregator using Langevin dynamics.
    
    Uses unadjusted Langevin algorithm (ULA):
        s_(t+1) = s_t + η ∇ log π(s_t) + √(2η) ε_t
    where ∇ log π(s) = ∇ log_prob(s, t_diff) + Σ_i ∇_s log p(Φ_i(s))
    
    Requirements:
        - context.grad_log_prob() must be available
        - server.compute_gradient() for unified gradient computation
    
    This is the recommended aggregator for most use cases with gradient information.
    """

    step_size: float = 0.01
    n_steps: int = 50

    def aggregate(
        self,
        s_current: Any,
        t_diff: float,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """Gradient-based MCMC aggregation."""
        logger.info(f"GradientMCMC aggregation: t_diff={t_diff:.4f}, n_steps={self.n_steps}, step_size={self.step_size}")
        
        # Extract what we need from kwargs
        server = kwargs['server']
        transport = kwargs['transport']
        context = kwargs['context']
        seed = kwargs.get('seed', None)
        
        rng = rng_from_seed(seed)
        s = np.asarray(s_current).copy()
        
        logger.debug(f"Starting Langevin dynamics")

        def compute_grad_log_target(s_val: np.ndarray) -> np.ndarray:
            """Compute ∇_s log π(s | {x_i}, t_diff)."""
            # Context gradient
            grad_ctx = context.grad_log_prob(s_val, t_diff)
            if grad_ctx is None:
                logger.error("Context does not support grad_log_prob()")
                raise NotSupportedError(
                    "GradientMCMCAggregator requires context.grad_log_prob()"
                )
            grad_total = np.asarray(grad_ctx)
            
            # Client gradients via ServerBase unified interface
            # This automatically:
            # 1. Creates requests ['sample', 'gradient']
            # 2. Clients handle projection, denoising, gradient computation
            # 3. Returns dict {client_id: grad_s}
            client_grads = server.compute_gradient(s_val, t_diff)
            for grad in client_grads.values():
                grad_total += np.asarray(grad)
            
            return grad_total

        # Langevin dynamics
        logger.debug(f"Running Langevin dynamics for {self.n_steps} steps")
        for step_idx in range(self.n_steps):
            grad = compute_grad_log_target(s)
            noise = rng.normal(size=s.shape)
            s = s + self.step_size * grad + np.sqrt(2 * self.step_size) * noise
            
            if step_idx % 10 == 0:
                grad_norm = np.linalg.norm(grad)
                logger.debug(f"  Step {step_idx}/{self.n_steps}: ||∇log π||={grad_norm:.4f}")
        
        logger.info("GradientMCMC aggregation complete")

        diagnostics = {
            "aggregator_type": "gradient_mcmc",
            "step_size": self.step_size,
            "n_steps": self.n_steps,
        }
        return s, diagnostics


@dataclass
class RandomWalkMCMCAggregator(AggregationBase):
    """Random-walk Metropolis-Hastings aggregator.
    
    Uses simple Gaussian random walk proposals with Metropolis acceptance.
    Most general aggregator - works for any combination without gradient information.
    
    Requirements: None (works for any context/client combination)
    """

    step_size: float = 0.1
    n_steps: int = 100

    def aggregate(
        self,
        s_current: Any,
        t_diff: float,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """Random-walk MCMC aggregation."""
        logger.info(f"RandomWalkMCMC aggregation: t_diff={t_diff:.4f}, n_steps={self.n_steps}, step_size={self.step_size}")
        
        # Extract what we need from kwargs
        server = kwargs['server']
        transport = kwargs['transport']
        context = kwargs['context']
        seed = kwargs.get('seed', None)
        
        rng = rng_from_seed(seed)
        s = np.asarray(s_current).copy()

        def log_target(s_val: np.ndarray) -> float:
            """Compute log π(s | {x_i}, t_diff)."""
            # Context term
            log_q_ctx = context.log_q_ctx(s_val, t_diff)
            
            # Client log likelihood terms via AggregationBase helper
            # This automatically:
            # 1. Creates requests ['sample', 'log_prob']
            # 2. Clients handle projection, denoising, log_prob computation
            # 3. Returns dict {client_id: log_prob}
            log_probs = self.fetch_log_probs(s_val, t_diff, server, transport)
            
            # Sum all contributions
            total = log_q_ctx + sum(log_probs.values())
            return float(total)

        accept = 0
        log_target_current = log_target(s)
        logger.debug(f"Running Metropolis-Hastings for {self.n_steps} steps")
        
        for step_idx in range(self.n_steps):
            # Propose
            s_proposal = s + self.step_size * rng.normal(size=s.shape)
            log_target_proposal = log_target(s_proposal)
            
            # Accept/reject
            log_alpha = log_target_proposal - log_target_current
            if np.log(rng.random()) < log_alpha:
                s = s_proposal
                log_target_current = log_target_proposal
                accept += 1
            
            if step_idx % 10 == 0 and step_idx > 0:
                accept_rate = accept / (step_idx + 1)
                logger.debug(f"  Step {step_idx}/{self.n_steps}: accept_rate={accept_rate:.3f}")

        acceptance_rate = accept / self.n_steps
        logger.info(f"RandomWalkMCMC aggregation complete: acceptance_rate={acceptance_rate:.3f}")
        
        if acceptance_rate < 0.1:
            logger.warning(f"Low acceptance rate {acceptance_rate:.3f} - consider reducing step_size")
        elif acceptance_rate > 0.9:
            logger.warning(f"High acceptance rate {acceptance_rate:.3f} - consider increasing step_size")

        diagnostics = {
            "aggregator_type": "random_walk_mcmc",
            "step_size": self.step_size,
            "n_steps": self.n_steps,
            "acceptance_rate": acceptance_rate,
        }
        return s, diagnostics

