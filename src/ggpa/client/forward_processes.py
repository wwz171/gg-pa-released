"""Forward process implementations for GG-PA.

This module contains concrete implementations of forward diffusion processes.
Forward processes are CLIENT-SIDE concepts - they define how clients generate
noisy observations from clean samples.

KEY DESIGN:
- ForwardProcessBase is defined in client/base.py (part of client architecture)
- This module provides concrete implementations (Gaussian, VP, VE schedules)
- Server uses forward_spec only for public clients (Mode 1)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ggpa.client.base import ForwardProcessBase
from ggpa.core.errors import ConfigurationError, ShapeError


@dataclass(frozen=True)
class NoiseSchedule:
    """Noise schedule providing alpha(tau) and sigma(tau).
    
    This is a helper class for Gaussian forward processes.
    """

    alpha_fn: Callable[[float], float]
    sigma_fn: Callable[[float], float]

    def alpha(self, tau: float) -> float:
        """Signal preservation coefficient at time tau."""
        return float(self.alpha_fn(tau))

    def sigma(self, tau: float) -> float:
        """Noise standard deviation at time tau."""
        return float(self.sigma_fn(tau))


@dataclass(frozen=True)
class GaussianForwardProcess(ForwardProcessBase):
    """Gaussian forward process q_tau(y | x) = N(y; alpha(tau) * x, sigma(tau)^2 * I).
    
    This is the most common forward process used in diffusion models.
    
    Provides:
        - log_q_fwd(y, x, tau): Log probability for reduced potential
        - grad_log_q_fwd(y, x, tau): Gradient for MCMC aggregators
        - alpha(tau), sigma(tau): Schedule parameters for closed-form aggregation
    """

    schedule: NoiseSchedule

    def alpha(self, tau: float) -> float:
        """Signal preservation coefficient at time tau."""
        return self.schedule.alpha(tau)

    def sigma(self, tau: float) -> float:
        """Noise standard deviation at time tau."""
        return self.schedule.sigma(tau)

    def log_q_fwd(self, y: np.ndarray, x: np.ndarray, tau: float) -> Any:
        """Compute log q_tau(y | x) for Gaussian forward process.
        
        Args:
            y: Noisy observation (any shape)
            x: Clean sample (same shape as y)
            tau: Diffusion time
            
        Returns:
            Log probability log N(y; alpha*x, sigma^2*I)
            
            Shape behavior:
            - Single sample: y=(D,), x=(D,) → returns scalar
            - Batch: y=(B,D), x=(B,D) → returns scalar (sum over all)
            
            To get per-sample log probs for batch data, reshape and sum
            along the sample dimension manually.
        """
        y = np.asarray(y)
        x = np.asarray(x)
        if y.shape != x.shape:
            raise ShapeError(f"y.shape {y.shape} != x.shape {x.shape}")
        
        alpha = self.alpha(tau)
        sigma = self.sigma(tau)
        if sigma <= 0:
            raise ConfigurationError("sigma must be positive")
        
        resid = y - alpha * x
        dim = y.size  # Total number of elements (B*D for batches)
        log_norm = -0.5 * dim * np.log(2.0 * np.pi * sigma * sigma)
        quad = -0.5 * float(np.dot(resid.ravel(), resid.ravel()) / (sigma * sigma))
        return log_norm + quad

    def grad_log_q_fwd(self, y: np.ndarray, x: np.ndarray, tau: float) -> np.ndarray:
        """Compute gradient ∇_y log q_tau(y | x).
        
        For Gaussian: ∇_y log N(y; μ, σ²I) = -(y - μ) / σ²
        
        Args:
            y: Noisy observation
            x: Clean sample
            tau: Diffusion time
            
        Returns:
            Gradient vector with same shape as y
        """
        y = np.asarray(y)
        x = np.asarray(x)
        alpha = self.alpha(tau)
        sigma = self.sigma(tau)
        if sigma <= 0:
            raise ConfigurationError("sigma must be positive")
        
        # ∇_y log N(y; αx, σ²I) = -(y - αx) / σ²
        return -(y - alpha * x) / (sigma * sigma)


def make_variance_preserving_schedule(
    beta_min: float = 0.1,
    beta_max: float = 20.0
) -> NoiseSchedule:
    """Create a Variance Preserving (VP) noise schedule.
    
    VP schedule ensures α²(τ) + σ²(τ) = 1.
    Commonly used in DDPM-style models.
    
    Args:
        beta_min: Minimum noise level
        beta_max: Maximum noise level
        
    Returns:
        NoiseSchedule with VP properties
    """
    def alpha(tau: float) -> float:
        # Linear interpolation of log SNR
        beta_t = beta_min + tau * (beta_max - beta_min)
        return np.exp(-0.5 * beta_t * tau)
    
    def sigma(tau: float) -> float:
        alpha_t = alpha(tau)
        return np.sqrt(1.0 - alpha_t * alpha_t)
    
    return NoiseSchedule(alpha_fn=alpha, sigma_fn=sigma)


def make_variance_exploding_schedule(
    sigma_min: float = 0.01,
    sigma_max: float = 50.0
) -> NoiseSchedule:
    """Create a Variance Exploding (VE) noise schedule.
    
    VE schedule keeps α(τ) = 1 and varies σ(τ).
    Commonly used in score-based models.
    
    Args:
        sigma_min: Minimum noise standard deviation
        sigma_max: Maximum noise standard deviation
        
    Returns:
        NoiseSchedule with VE properties
    """
    def alpha(tau: float) -> float:
        return 1.0
    
    def sigma(tau: float) -> float:
        # Geometric interpolation
        log_sigma = np.log(sigma_min) + tau * (np.log(sigma_max) - np.log(sigma_min))
        return np.exp(log_sigma)
    
    return NoiseSchedule(alpha_fn=alpha, sigma_fn=sigma)


__all__ = [
    "NoiseSchedule",
    "GaussianForwardProcess",
    "make_variance_preserving_schedule",
    "make_variance_exploding_schedule",
]
