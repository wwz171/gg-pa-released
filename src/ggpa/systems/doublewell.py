r"""Analytic helpers for the coupled double-well quick-run notebook.

This module provides the small amount of reusable physics needed by
``notebooks/example_doublewell.ipynb``:

- the coupled double-well system definition
- exact equilibrium samplers for reference distributions
- the finite-``t`` Gaussian match condition
- a lightweight 1-D VP forward-process wrapper

The GG-PA samplers themselves are implemented directly in the notebook, so
older system-specific client / server / replica-exchange helpers are not kept
here.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ggpa.client.base import ForwardProcessBase


def _as_signal_array(s) -> tuple[np.ndarray, bool]:
    """Return ``(B, 2)`` array and whether the input was a single sample."""
    arr = np.asarray(s, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != 2:
            raise ValueError(f"Expected signal shape (2,), got {arr.shape}")
        return arr[np.newaxis, :], True
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr, False
    raise ValueError(f"Expected signal shape (B, 2), got {arr.shape}")


def _as_column(x) -> tuple[np.ndarray, bool]:
    """Return ``(B, 1)`` array and whether the input was a single scalar."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1, 1), True
    if arr.ndim == 1:
        return arr.reshape(-1, 1), arr.size == 1
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr, False
    raise ValueError(f"Expected shape compatible with (B, 1), got {arr.shape}")


class CoupledDoubleWell:
    """Bare double-well coupled to a harmonic environment."""

    def __init__(
        self,
        a: float = 8.0,
        b: float = 16.0,
        c: float = 0.0,
        k_b: float = 1.0,
        k_c: float = 4.0,
        u_eq: float = 1.0,
    ):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.k_b = float(k_b)
        self.k_c = float(k_c)
        self.u_eq = float(u_eq)

    def base_potential(self, x):
        """Bare 1-D double-well energy ``V0(x) = a x^4 - b x^2 + c x``."""
        x = np.asarray(x, dtype=np.float64)
        return self.a * x**4 - self.b * x**2 + self.c * x

    def coupling_potential(self, s):
        """Quadratic environment/coupling energy ``W(x, u)``."""
        s_arr, single = _as_signal_array(s)
        x = s_arr[:, 0]
        u = s_arr[:, 1]
        w = 0.5 * self.k_b * (u - self.u_eq) ** 2 + 0.5 * self.k_c * (x - u) ** 2
        return float(w[0]) if single else w

    def potential(self, s):
        """Full coupled energy ``V0(x) + W(x, u)``."""
        s_arr, single = _as_signal_array(s)
        total = self.base_potential(s_arr[:, 0]) + self.coupling_potential(s_arr)
        return float(total[0]) if single else total

    def effective_potential(self, x):
        r"""Exact marginal free energy after integrating out ``u``."""
        x = np.asarray(x, dtype=np.float64)
        k_eff = self.k_b * self.k_c / (self.k_b + self.k_c)
        return self.base_potential(x) + 0.5 * k_eff * (x - self.u_eq) ** 2

    def conditional_u_mean(self, x):
        """Mean of ``u | x`` in the exact coupled equilibrium."""
        x = np.asarray(x, dtype=np.float64)
        return (self.k_c * x + self.k_b * self.u_eq) / (self.k_b + self.k_c)

    def conditional_u_var(self, beta: float = 1.0) -> float:
        """Variance of ``u | x`` in the exact coupled equilibrium."""
        return 1.0 / (float(beta) * (self.k_b + self.k_c))

    def base_force(self, x):
        """Force on ``x`` from the bare double-well only."""
        x = np.asarray(x, dtype=np.float64)
        return -(4.0 * self.a * x**3 - 2.0 * self.b * x + self.c)

    def force(self, s):
        """Full force on ``(x, u)`` from ``V0(x) + W(x, u)``."""
        s_arr, single = _as_signal_array(s)
        x = s_arr[:, 0]
        u = s_arr[:, 1]
        dV_dx = 4.0 * self.a * x**3 - 2.0 * self.b * x + self.c + self.k_c * (x - u)
        dV_du = self.k_b * (u - self.u_eq) - self.k_c * (x - u)
        force = -np.stack([dV_dx, dV_du], axis=1)
        return force[0] if single else force


def sample_1d_equilibrium(
    energy_fn,
    n: int,
    *,
    beta: float = 1.0,
    x_range: tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 8193,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample a 1-D equilibrium density via inverse-CDF on a fine grid."""
    if rng is None:
        rng = np.random.default_rng()
    x_grid = np.linspace(x_range[0], x_range[1], n_grid)
    v = np.asarray(energy_fn(x_grid), dtype=np.float64)
    v = v - np.min(v)
    pdf = np.exp(-beta * v)
    cdf = np.zeros_like(pdf)
    cdf[1:] = np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * np.diff(x_grid))
    cdf = cdf / cdf[-1]
    u = rng.random(int(n))
    return np.interp(u, cdf, x_grid)


def sample_base_equilibrium(
    n: int,
    system: CoupledDoubleWell,
    *,
    beta: float = 1.0,
    x_range: tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 8193,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample the bare double-well prior over ``x``."""
    return sample_1d_equilibrium(
        system.base_potential,
        n,
        beta=beta,
        x_range=x_range,
        n_grid=n_grid,
        rng=rng,
    )


def sample_coupled_equilibrium(
    n: int,
    system: CoupledDoubleWell,
    *,
    beta: float = 1.0,
    x_range: tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 8193,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample exact equilibrium pairs ``(x, u)`` from the coupled system."""
    if rng is None:
        rng = np.random.default_rng()
    x = sample_1d_equilibrium(
        system.effective_potential,
        n,
        beta=beta,
        x_range=x_range,
        n_grid=n_grid,
        rng=rng,
    )
    u = system.conditional_u_mean(x) + math.sqrt(system.conditional_u_var(beta)) * rng.standard_normal(n)
    return np.stack([x, u], axis=1)


def finite_t_exact_context_stiffness(
    t: float,
    *,
    system: CoupledDoubleWell,
    forward_process: "DoubleWellVPForwardProcess",
    beta: float = 1.0,
) -> float:
    r"""Annealed harmonic stiffness that makes the finite-``t`` marginal exact.

    For the VP forward kernel

    .. math::

        q_t(y \mid x) = \mathcal{N}(y;\, \alpha_t x,\, \sigma_t^2),

    the exact matched context takes the form

    .. math::

        p_{{\rm ctx},t}(y,u) \propto
        \exp\!\Bigl[-\beta \Bigl(
            \tfrac12 k_b (u-u_{\rm eq})^2
            + \tfrac12 \tilde{k}_c(t)\,(y-\alpha_t u)^2
        \Bigr)\Bigr]

    with

    .. math::

        \tilde{k}_c(t) =
        \frac{k_c}{\alpha_t^2 - \beta k_c \sigma_t^2}.
    """
    alpha_t = float(forward_process.alpha(t))
    sigma_t = float(forward_process.sigma(t))
    denom = alpha_t * alpha_t - float(beta) * float(system.k_c) * sigma_t * sigma_t
    if denom <= 0.0:
        raise ValueError(
            "Finite-t exact Gaussian context is infeasible for this t. "
            f"Need alpha(t)^2 > beta * k_c * sigma(t)^2, got denom={denom:.6e}."
        )
    return float(system.k_c) / denom


class DoubleWellVPForwardProcess(ForwardProcessBase):
    """1-D VP forward process wrapper for the bare double-well diffusion model."""

    def __init__(self, noise_scheduler):
        self._ns = noise_scheduler

    def _params(self, t: float) -> tuple[float, float]:
        idx = max(0, min(self._ns.num_timesteps - 1, int(round(t * (self._ns.num_timesteps - 1)))))
        alpha_bar = float(self._ns.alpha_bars[idx].detach().cpu())
        return math.sqrt(alpha_bar), math.sqrt(max(1.0 - alpha_bar, 0.0))

    def alpha(self, t: float) -> float:
        return self._params(t)[0]

    def sigma(self, t: float) -> float:
        return self._params(t)[1]

    def log_q_fwd(self, y, x, t):
        y_col, y_single = _as_column(y)
        x_col, x_single = _as_column(x)
        if y_col.shape != x_col.shape:
            raise ValueError(f"Shape mismatch: y {y_col.shape}, x {x_col.shape}")

        alpha_t, sigma_t = self._params(t)
        if sigma_t < 1e-12:
            resid = np.abs(y_col - alpha_t * x_col).max(axis=1)
            out = np.where(resid < 1e-10, 0.0, -np.inf)
            return float(out[0]) if y_single and x_single else out

        resid = y_col - alpha_t * x_col
        logp = -0.5 * np.log(2.0 * np.pi * sigma_t * sigma_t) - 0.5 * (resid[:, 0] ** 2) / (sigma_t * sigma_t)
        return float(logp[0]) if y_single and x_single else logp

    def grad_log_q_fwd(self, y, x, t):
        y_col, y_single = _as_column(y)
        x_col, _ = _as_column(x)
        alpha_t, sigma_t = self._params(t)
        if sigma_t < 1e-12:
            grad = np.zeros_like(y_col)
        else:
            grad = -(y_col - alpha_t * x_col) / (sigma_t * sigma_t)
        if y_single:
            return grad.reshape(-1)
        return grad


__all__ = [
    "CoupledDoubleWell",
    "sample_1d_equilibrium",
    "sample_base_equilibrium",
    "sample_coupled_equilibrium",
    "finite_t_exact_context_stiffness",
    "DoubleWellVPForwardProcess",
]
