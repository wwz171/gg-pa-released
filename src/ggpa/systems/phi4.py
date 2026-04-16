r"""2D Ginzburg–Landau :math:`\phi^4` lattice field theory.

.. currentmodule:: ggpa.systems.phi4

Hamiltonian
-----------

.. math::

    H(\phi) = \sum_i (\phi_i^2 - 1)^2
              + J \sum_{\langle i,j\rangle} (\phi_i - \phi_j)^2
              - h \sum_i \phi_i

where :math:`\langle i,j\rangle` runs over each nearest-neighbour edge
**once** and *h* is an optional external field (default 0).

This module provides:

* Energy / force utilities on a 2-D square lattice with periodic boundary
  conditions (PBC).
* Vectorised checkerboard Metropolis–Hastings Monte Carlo.
* A rejection sampler for the single-site double-well
  :math:`V(\phi)=(\phi^2-1)^2`.
* GG-PA adapter classes that couple the lattice system to the
  :class:`~ggpa.core.kernel.FixedDiffusionTimeKernel` framework:
  :class:`LatticeVPForwardProcess`, :class:`LatticeDiffusionClient`,
  :class:`GaussianPBCContext`, and :class:`FFTGaussianAggregator`.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from ggpa.client.base import ClientBase, ForwardProcessBase
from ggpa.client.projectors.identity import IdentityProjector
from ggpa.server.base import AggregationBase, ContextBase


# =====================================================================
# Energy computation
# =====================================================================


def compute_local_energy(phi: np.ndarray, J: float, h: float = 0.0) -> np.ndarray:
    r"""Local energy density at every site (PBC, vectorised).

    Each nearest-neighbour edge appears in the local energies of **both**
    endpoints.  This is intentional: the Metropolis energy difference
    :math:`\Delta E_{\text{local}}(i)` equals the true Hamiltonian change
    :math:`\Delta H` exactly.

    Parameters
    ----------
    phi : (L, L) array
    J : coupling constant
    h : external field strength
    """
    on_site = (phi ** 2 - 1) ** 2
    coupling = J * (
        (phi - np.roll(phi, -1, axis=0)) ** 2
        + (phi - np.roll(phi, 1, axis=0)) ** 2
        + (phi - np.roll(phi, -1, axis=1)) ** 2
        + (phi - np.roll(phi, 1, axis=1)) ** 2
    )
    return on_site + coupling - h * phi


def compute_total_energy(phi: np.ndarray, J: float, h: float = 0.0) -> float:
    """Total Hamiltonian *H*, counting each edge exactly once."""
    on_site = np.sum((phi ** 2 - 1) ** 2)
    coupling = J * (
        np.sum((phi - np.roll(phi, -1, axis=0)) ** 2)
        + np.sum((phi - np.roll(phi, -1, axis=1)) ** 2)
    )
    return float(on_site + coupling - h * np.sum(phi))


# =====================================================================
# Checkerboard Metropolis–Hastings
# =====================================================================


def make_checkerboard_masks(L: int):
    """Return ``[black_mask, white_mask]`` for an L×L lattice."""
    i, j = np.meshgrid(range(L), range(L), indexing="ij")
    black = (i + j) % 2 == 0
    return [black, ~black]


def checkerboard_metropolis_step(
    phi: np.ndarray,
    J: float,
    mask: np.ndarray,
    *,
    neighbors_sum: np.ndarray | None = None,
    proposal_width: float = 0.5,
    T: float = 1.0,
    h: float = 0.0,
) -> tuple[int, int]:
    """One checkerboard half-sweep (in-place).

    Returns ``(n_accepted, n_attempted)``.
    """
    L = phi.shape[0]
    if neighbors_sum is None:
        neighbors_sum = (
            np.roll(phi, -1, axis=0) + np.roll(phi, 1, axis=0)
            + np.roll(phi, -1, axis=1) + np.roll(phi, 1, axis=1)
        )

    old_E = (phi ** 2 - 1) ** 2 + J * (4 * phi ** 2 - 2 * phi * neighbors_sum)
    noise = np.random.uniform(-proposal_width, proposal_width, (L, L))
    phi_new = phi + noise
    new_E = (phi_new ** 2 - 1) ** 2 + J * (4 * phi_new ** 2 - 2 * phi_new * neighbors_sum)

    delta_E = new_E - old_E
    if h != 0.0:
        delta_E = delta_E - h * (phi_new - phi)

    accept = (np.random.random((L, L)) < np.minimum(1.0, np.exp(-delta_E / T))) & mask
    phi[accept] = phi_new[accept]
    return int(np.sum(accept)), int(np.sum(mask))


def run_checkerboard_mc(
    L: int,
    J: float,
    measurement_steps: int,
    equilibration_steps: int = 15_000,
    proposal_width: float = 0.5,
    T: float = 1.0,
    h: float = 0.0,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run a complete checkerboard MC simulation (PBC).

    Returns a dict with keys ``'final_config'``, ``'magnetizations'``
    (absolute value |⟨φ⟩|, one per measurement step), and ``'acceptance_rate'``.
    """
    if seed is not None:
        np.random.seed(seed)
    phi = np.random.uniform(-2, 2, (L, L))
    masks = make_checkerboard_masks(L)
    total_steps = equilibration_steps + measurement_steps
    magnetizations: list[float] = []
    total_acc = total_att = 0

    for step in range(total_steps):
        mask = masks[step % 2]
        ns = (
            np.roll(phi, -1, axis=0) + np.roll(phi, 1, axis=0)
            + np.roll(phi, -1, axis=1) + np.roll(phi, 1, axis=1)
        )
        n_acc, n_att = checkerboard_metropolis_step(
            phi, J, mask, neighbors_sum=ns, proposal_width=proposal_width, T=T, h=h,
        )
        total_acc += n_acc
        total_att += n_att
        if step >= equilibration_steps:
            magnetizations.append(float(np.abs(np.mean(phi))))

    return {
        "final_config": phi,
        "magnetizations": magnetizations,
        "acceptance_rate": total_acc / max(total_att, 1),
        "L": L,
        "J": J,
    }


def J_scan(
    L: int,
    J_values,
    measurement_steps: int = 50_000,
    equilibration_steps: int = 10_000,
    proposal_width: float = 0.5,
    T: float = 1.0,
    h: float = 0.0,
    verbose: bool = True,
) -> dict:
    """Scan |⟨φ⟩| vs J with MC.

    Returns dict with arrays ``'J'``, ``'order_param'``,
    ``'order_err'``, ``'accept_rate'``.
    """
    J_values = np.asarray(J_values)
    ops, errs, rates = [], [], []
    for g in J_values:
        res = run_checkerboard_mc(
            L, g, measurement_steps, equilibration_steps,
            proposal_width=proposal_width, T=T, h=h, verbose=False,
        )
        m = res["magnetizations"]
        ops.append(np.mean(m))
        errs.append(np.std(m) / np.sqrt(len(m)))
        rates.append(res["acceptance_rate"])
        if verbose:
            print(f"  J={g:.4f}  |⟨φ⟩|={ops[-1]:.4f} ± {errs[-1]:.4f}")
    return {
        "J": J_values,
        "order_param": np.array(ops),
        "order_err": np.array(errs),
        "accept_rate": np.array(rates),
    }


# =====================================================================
# Single-site double-well sampler (for diffusion model training data)
# =====================================================================


def sample_double_well(n: int, beta: float = 1.0) -> np.ndarray:
    r"""Rejection-sample from :math:`p(\phi)\propto e^{-\beta(\phi^2-1)^2}`.

    Returns shape ``(n,)`` with symmetrised samples (both + and −).
    """
    samples: list[float] = []
    while len(samples) < n:
        phi = np.random.uniform(-3, 3)
        if np.random.rand() < np.exp(-beta * (phi ** 2 - 1) ** 2):
            samples.append(phi)
    return np.array(samples, dtype=np.float64)


# =====================================================================
# Shared helpers (Fourier-space operations)
# =====================================================================


def _laplacian_spectrum_rfft(L: int) -> np.ndarray:
    """PBC nearest-neighbour Laplacian eigenvalues, shape ``(L, L//2+1)``."""
    kx = np.arange(L).reshape(L, 1)
    ky = np.arange(L // 2 + 1).reshape(1, -1)
    return 4.0 * (np.sin(np.pi * kx / L) ** 2 + np.sin(np.pi * ky / L) ** 2)


def _rfft_last_axis_weights(L: int) -> np.ndarray:
    """Weights that lift an ``rfftn`` half-spectrum back to full-spectrum norms.

    ``np.fft.rfftn`` only removes conjugate redundancy along the final axis.
    Interior columns therefore represent two full-spectrum modes and must be
    counted twice in quadratic forms such as ``psi.T @ Q @ psi``.
    """
    w = np.ones((1, L // 2 + 1), dtype=np.float64)
    if L % 2 == 0:
        if w.shape[1] > 2:
            w[:, 1:-1] = 2.0
    elif w.shape[1] > 1:
        w[:, 1:] = 2.0
    return w


def _vp_params_at_t_diff(noise_scheduler, t_diff: float) -> tuple[float, float]:
    r"""Return :math:`(\bar\alpha, \sigma^2)` at continuous time t_diff ∈ [0, 1]."""
    T = noise_scheduler.num_timesteps
    idx = max(0, min(T - 1, int(t_diff * (T - 1))))
    ab = float(noise_scheduler.alpha_bars[idx].cpu())
    return ab, 1.0 - ab


def _precision_spectrum(
    lam: np.ndarray, J: float, a2: float, s2: float,
) -> np.ndarray:
    r"""Context precision eigenvalues :math:`q_k = 2J\lambda_k / (\alpha^2 - 2J\lambda_k\sigma^2)`."""
    mask = lam > 0
    denom = a2 - 2.0 * J * lam * s2
    if np.any(denom[mask] <= 0):
        raise ValueError(
            f"GG-PA infeasible: min denom = {float(np.min(denom[mask])):.6e}. "
            "Use a smaller t_diff or smaller J."
        )
    q = np.zeros_like(lam)
    q[mask] = (2.0 * J * lam[mask]) / denom[mask]
    return q


def check_max_t_diff(noise_scheduler, J: float, d: int = 2) -> dict:
    r"""Maximum feasible t_diff for GG-PA Gaussian-context construction.

    Condition: :math:`\sigma^2 < \alpha^2 / (2J\lambda_{\max})`.
    """
    lam_max = 4.0 * d
    a_min = (2.0 * J * lam_max) / (1.0 + 2.0 * J * lam_max)
    ab = noise_scheduler.alpha_bars.detach().cpu().numpy()
    ok = ab >= a_min
    if not np.any(ok):
        return {"t_max": None, "note": "No feasible t_diff for this schedule / J"}
    idx = int(np.max(np.where(ok)))
    t_max = max(0.0, idx / (len(ab) - 1) - 1e-6)
    return {
        "t_max": t_max,
        "alpha_min_required": float(a_min),
        "alpha_bar_at_tmax": float(ab[idx]),
    }


# =====================================================================
# GG-PA adapter: Forward Process
# =====================================================================


class LatticeVPForwardProcess(ForwardProcessBase):
    r"""VP forward process for 2-D lattice fields.

    .. math:: q_{t_{\mathrm{diff}}}(\psi \mid \phi) = \mathcal{N}(\psi;\, \alpha\phi,\, \sigma^2 I)
    """

    def __init__(self, noise_scheduler):
        self._ns = noise_scheduler

    def alpha(self, t_diff: float) -> float:
        ab, _ = _vp_params_at_t_diff(self._ns, t_diff)
        return float(np.sqrt(ab))

    def sigma(self, t_diff: float) -> float:
        _, s2 = _vp_params_at_t_diff(self._ns, t_diff)
        return float(np.sqrt(max(s2, 0.0)))

    def log_q_fwd(self, y, x, t_diff):
        y = np.asarray(y, dtype=np.float64).ravel()
        x = np.asarray(x, dtype=np.float64).ravel()
        a, s = self.alpha(t_diff), self.sigma(t_diff)
        D = y.size
        r = y - a * x
        return float(-0.5 * D * np.log(2.0 * np.pi * s * s) - 0.5 * np.dot(r, r) / (s * s))

    def grad_log_q_fwd(self, y, x, t_diff):
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        a, s = self.alpha(t_diff), self.sigma(t_diff)
        return -(y - a * x) / (s * s)


# =====================================================================
# GG-PA adapter: Client
# =====================================================================


class LatticeDiffusionClient(ClientBase):
    """GG-PA client wrapping a 1-D
    :class:`~ggpa.models.SimpleDiffusion` for an L×L lattice.

    Signal *s* = ψ has shape ``(L, L)``.  Denoising flattens to ``(L², 1)``,
    calls the diffusion reverse process, and reshapes back.

    Parameters
    ----------
    client_id : str
    diffusion_model : SimpleDiffusion
    forward_process : LatticeVPForwardProcess
    L : int
    device : str or torch.device
    enforce_symmetry : bool
        Average v(x) and −v(−x) to enforce Z₂ symmetry during reverse.
    """

    def __init__(
        self,
        client_id: str,
        diffusion_model,
        forward_process: LatticeVPForwardProcess,
        L: int,
        device="cpu",
        enforce_symmetry: bool = True,
    ):
        self.client_id = client_id
        self.diffusion_model = diffusion_model
        self.projector = IdentityProjector()
        self.forward_process = forward_process
        self.L = L
        self.device = device
        self.enforce_symmetry = enforce_symmetry

    @torch.no_grad()
    def denoise_sample(self, y, t_diff, seed=None):
        """Denoise ψ (L, L) → φ (L, L)."""
        y_np = np.asarray(y, dtype=np.float32)
        shape = y_np.shape
        y_flat = torch.tensor(
            y_np.ravel()[:, np.newaxis], dtype=torch.float32, device=self.device,
        )

        # Access underlying model if torch.compiled
        model = self.diffusion_model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        phi_flat = model.reverse(
            y_flat, t_start=t_diff, t_end=0.0,
            enforce_symmetry=self.enforce_symmetry,
        )
        return phi_flat.cpu().numpy().ravel().reshape(shape)


# =====================================================================
# GG-PA adapter: Context
# =====================================================================


class GaussianPBCContext(ContextBase):
    r"""Gaussian context with PBC, diagonal in Fourier space.

    .. math::

        p_{\text{ctx}}(\psi) \propto
        \exp\!\bigl(-\tfrac{1}{2}\,\psi^\top Q(t_{\mathrm{diff}})\,\psi + b^\top\psi\bigr)

    where the precision eigenvalues are

    .. math::

        q_k = \frac{2J\lambda_k}{\alpha^2 - 2J\lambda_k\sigma^2}
        \qquad(\lambda_k > 0),\quad q_0 = 0

    and the external-field term contributes
    :math:`\hat b_0 = hL/\alpha` (zero mode only).
    """

    def __init__(self, J: float, L: int, noise_scheduler, h: float = 0.0):
        self.J = float(J)
        self.L = int(L)
        self._ns = noise_scheduler
        self._lam = _laplacian_spectrum_rfft(L)
        self._rfft_weights = _rfft_last_axis_weights(L)
        self._h = float(h)

    def tempering_factor(self, t_diff):
        return 1.0

    def log_prob(self, s, t_diff):
        s = np.asarray(s, dtype=np.float64)
        a2, s2 = _vp_params_at_t_diff(self._ns, t_diff)
        q = _precision_spectrum(self._lam, self.J, a2, s2)
        s_hat = np.fft.rfftn(s, s=(self.L, self.L), norm="ortho")
        result = float(-0.5 * np.sum(self._rfft_weights * q * np.abs(s_hat) ** 2))
        if self._h != 0.0:
            result += float(self._h / np.sqrt(a2) * np.sum(s))
        return result

    def grad_log_prob(self, s, t_diff):
        s = np.asarray(s, dtype=np.float64)
        a2, s2 = _vp_params_at_t_diff(self._ns, t_diff)
        q = _precision_spectrum(self._lam, self.J, a2, s2)
        s_hat = np.fft.rfftn(s, s=(self.L, self.L), norm="ortho")
        grad = np.fft.irfftn(-q * s_hat, s=(self.L, self.L), norm="ortho")
        if self._h != 0.0:
            grad = grad + self._h / np.sqrt(a2)
        return grad

    def check_valid_t_diff(self, t_diff: float) -> bool:
        """Return True if t_diff is within the GG-PA feasibility region."""
        a2, s2 = _vp_params_at_t_diff(self._ns, t_diff)
        mask = self._lam > 0
        denom = a2 - 2.0 * self.J * self._lam * s2
        return bool(np.all(denom[mask] > 0))


# =====================================================================
# GG-PA adapter: Aggregator (closed-form FFT Gaussian)
# =====================================================================


class FFTGaussianAggregator(AggregationBase):
    r"""Closed-form Gaussian aggregation of ψ | φ via FFT.

    The conditional posterior
    :math:`p(\psi \mid \phi) \propto q_{\text{fwd}}(\psi \mid \phi)\,p_{\text{ctx}}(\psi)`
    is Gaussian with per-Fourier-mode parameters:

    .. math::

        A_k = 1/\sigma^2 + q_k, \quad
        v_k = 1/A_k, \quad
        \hat\mu_k = v_k \bigl(\alpha/\sigma^2\,\hat\phi_k + \hat b_k\bigr)

    Sampling: :math:`\hat\psi_k = \hat\mu_k + \sqrt{v_k}\,\hat z_k`,
    then :math:`\psi = \text{IFFT}(\hat\psi)`.

    Parameters
    ----------
    J : float
    L : int
    noise_scheduler : NoiseScheduler
    h : float
        External field strength (default 0).
    client : LatticeDiffusionClient or None
        If provided, enables a GPU fast-path that bypasses the standard
        request/reply transport and keeps denoise + FFT entirely on GPU.
    device : str or torch.device
    """

    def __init__(
        self,
        J: float,
        L: int,
        noise_scheduler,
        *,
        h: float = 0.0,
        client: LatticeDiffusionClient | None = None,
        device: str | torch.device = "cpu",
    ):
        self.J = float(J)
        self.L = int(L)
        self._ns = noise_scheduler
        self._lam = _laplacian_spectrum_rfft(L)
        self._h = float(h)
        self._client = client
        self._device = torch.device(device) if client is not None else None

    # ------------------------------------------------------------------ #
    # GPU fast path (direct client access, no transport overhead)
    # ------------------------------------------------------------------ #

    def _fast_aggregate_gpu(self, s_current, t_diff):
        L = self.L
        cl = self._client

        # Access underlying model if torch.compiled
        model = cl.diffusion_model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        # Denoise on GPU
        s_np = np.asarray(s_current, dtype=np.float32)
        y_flat = torch.tensor(
            s_np.ravel()[:, np.newaxis], dtype=torch.float32, device=self._device,
        )
        phi_flat = model.reverse(
            y_flat, t_start=t_diff, t_end=0.0,
            enforce_symmetry=cl.enforce_symmetry,
        )
        phi_gpu = phi_flat.squeeze(-1).reshape(L, L)
        cl._current_x = phi_gpu.detach().cpu().numpy()

        # VP parameters
        a2, s2 = _vp_params_at_t_diff(self._ns, t_diff)
        a = np.sqrt(a2)
        if s2 < 1e-12:
            return (a * phi_gpu).cpu().numpy(), {"method": "fft_gaussian_gpu", "degenerate": True}

        inv_s2 = 1.0 / s2
        q = _precision_spectrum(self._lam, self.J, a2, s2)
        A_k = inv_s2 + q
        v_k_np = 1.0 / A_k

        v_k = torch.tensor(v_k_np, dtype=torch.float32, device=self._device)
        sv_k = torch.tensor(np.sqrt(v_k_np), dtype=torch.float32, device=self._device)

        phi_hat = torch.fft.rfftn(phi_gpu, s=(L, L), norm="ortho")
        mean_hat = v_k * (a * inv_s2) * phi_hat

        if self._h != 0.0:
            mean_hat[0, 0] = mean_hat[0, 0] + v_k[0, 0] * (self._h * L / a)

        z_hat = torch.fft.rfftn(
            torch.randn(L, L, device=self._device), s=(L, L), norm="ortho",
        )
        psi_hat = mean_hat + sv_k * z_hat
        psi_gpu = torch.fft.irfftn(psi_hat, s=(L, L), norm="ortho")
        return psi_gpu.cpu().numpy(), {"method": "fft_gaussian_gpu", "t_diff": t_diff}

    # ------------------------------------------------------------------ #
    # Standard framework path
    # ------------------------------------------------------------------ #

    def aggregate(self, s_current, t_diff, **kwargs):
        # GPU fast path
        if self._client is not None and self._device is not None and self._device.type != "cpu":
            return self._fast_aggregate_gpu(s_current, t_diff)

        # Standard path via transport
        server = kwargs["server"]
        transport = kwargs["transport"]
        seed = kwargs.get("seed", None)

        xs = self.fetch_samples(s_current, t_diff, server, transport)
        assert len(xs) == 1, f"FFTGaussianAggregator expects 1 client, got {len(xs)}"
        phi = np.asarray(list(xs.values())[0], dtype=np.float64)
        L = self.L

        a2, s2 = _vp_params_at_t_diff(self._ns, t_diff)
        a = np.sqrt(a2)
        if s2 < 1e-12:
            return (a * phi), {"method": "fft_gaussian", "degenerate": True}

        inv_s2 = 1.0 / s2
        q = _precision_spectrum(self._lam, self.J, a2, s2)
        A_k = inv_s2 + q
        v_k = 1.0 / A_k
        sv_k = np.sqrt(v_k)

        phi_hat = np.fft.rfftn(phi, s=(L, L), norm="ortho")
        mean_hat = v_k * (a * inv_s2) * phi_hat

        if self._h != 0.0:
            mean_hat[0, 0] = mean_hat[0, 0] + v_k[0, 0] * (self._h * L / a)

        rng = np.random.default_rng(seed)
        z = rng.standard_normal((L, L))
        z_hat = np.fft.rfftn(z, s=(L, L), norm="ortho")

        psi_hat = mean_hat + sv_k * z_hat
        psi = np.fft.irfftn(psi_hat, s=(L, L), norm="ortho").real
        return psi, {"method": "fft_gaussian", "t_diff": t_diff}


# =====================================================================
# GG-PA adapter: Aggregator – Fixed-Q variant for Replica Exchange
# =====================================================================


class FixedQFFTGaussianAggregator(AggregationBase):
    r"""Like :class:`FFTGaussianAggregator` but the context precision Q is
    computed once at a *fixed* reference t_diff (``t_diff_prod``) and reused for all
    replica t_diff values.  This avoids the feasibility constraint that limits
    the maximum t_diff, allowing wide RE ladders.

    Parameters
    ----------
    J, L, noise_scheduler, h : same as :class:`FFTGaussianAggregator`
    t_diff_prod : float
        Reference diffusion time at which Q is precomputed.
    client : LatticeDiffusionClient or None
    device : str or torch.device
    """

    def __init__(
        self,
        J: float,
        L: int,
        noise_scheduler,
        *,
        t_diff_prod: float,
        h: float = 0.0,
        client: LatticeDiffusionClient | None = None,
        device: str | torch.device = "cpu",
    ):
        self.J = float(J)
        self.L = int(L)
        self._ns = noise_scheduler
        self._lam = _laplacian_spectrum_rfft(L)
        self._h = float(h)
        self._client = client
        self.t_diff_prod = float(t_diff_prod)
        self._device = torch.device(device) if client is not None else None

        # Precompute Q at t_diff_prod
        a2_prod, s2_prod = _vp_params_at_t_diff(noise_scheduler, t_diff_prod)
        self._q_prod = _precision_spectrum(self._lam, self.J, a2_prod, s2_prod)

    def _fast_aggregate_gpu(self, s_current, t_diff):
        L = self.L
        cl = self._client

        model = cl.diffusion_model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        # Denoise at the replica's actual t_diff
        s_np = np.asarray(s_current, dtype=np.float32)
        y_flat = torch.tensor(
            s_np.ravel()[:, np.newaxis], dtype=torch.float32, device=self._device,
        )
        phi_flat = model.reverse(
            y_flat, t_start=t_diff, t_end=0.0,
            enforce_symmetry=cl.enforce_symmetry,
        )
        phi_gpu = phi_flat.squeeze(-1).reshape(L, L)
        cl._current_x = phi_gpu.detach().cpu().numpy()

        # VP params at the replica's t_diff (for the forward-process likelihood)
        a2, s2 = _vp_params_at_t_diff(self._ns, t_diff)
        a = np.sqrt(a2)
        if s2 < 1e-12:
            return (a * phi_gpu).cpu().numpy(), {"method": "fixedq_fft_gpu", "degenerate": True}

        inv_s2 = 1.0 / s2
        # Use fixed Q from t_diff_prod
        A_k = inv_s2 + self._q_prod
        v_k_np = 1.0 / A_k

        v_k = torch.tensor(v_k_np, dtype=torch.float32, device=self._device)
        sv_k = torch.tensor(np.sqrt(v_k_np), dtype=torch.float32, device=self._device)

        phi_hat = torch.fft.rfftn(phi_gpu, s=(L, L), norm="ortho")
        mean_hat = v_k * (a * inv_s2) * phi_hat

        if self._h != 0.0:
            # External field with fixed Q's zero-mode contribution
            a_prod = np.sqrt(_vp_params_at_t_diff(self._ns, self.t_diff_prod)[0])
            mean_hat[0, 0] = mean_hat[0, 0] + v_k[0, 0] * (self._h * L / a_prod)

        z_hat = torch.fft.rfftn(
            torch.randn(L, L, device=self._device), s=(L, L), norm="ortho",
        )
        psi_hat = mean_hat + sv_k * z_hat
        psi_gpu = torch.fft.irfftn(psi_hat, s=(L, L), norm="ortho")
        return psi_gpu.cpu().numpy(), {"method": "fixedq_fft_gpu", "t_diff": t_diff, "t_diff_prod": self.t_diff_prod}

    def aggregate(self, s_current, t_diff, **kwargs):
        if self._client is not None and self._device is not None and self._device.type != "cpu":
            return self._fast_aggregate_gpu(s_current, t_diff)

        server = kwargs["server"]
        transport = kwargs["transport"]
        seed = kwargs.get("seed", None)

        xs = self.fetch_samples(s_current, t_diff, server, transport)
        assert len(xs) == 1
        phi = np.asarray(list(xs.values())[0], dtype=np.float64)
        L = self.L

        a2, s2 = _vp_params_at_t_diff(self._ns, t_diff)
        a = np.sqrt(a2)
        if s2 < 1e-12:
            return (a * phi), {"method": "fixedq_fft", "degenerate": True}

        inv_s2 = 1.0 / s2
        A_k = inv_s2 + self._q_prod
        v_k = 1.0 / A_k
        sv_k = np.sqrt(v_k)

        phi_hat = np.fft.rfftn(phi, s=(L, L), norm="ortho")
        mean_hat = v_k * (a * inv_s2) * phi_hat

        if self._h != 0.0:
            a_prod = np.sqrt(_vp_params_at_t_diff(self._ns, self.t_diff_prod)[0])
            mean_hat[0, 0] = mean_hat[0, 0] + v_k[0, 0] * (self._h * L / a_prod)

        rng = np.random.default_rng(seed)
        z_hat = np.fft.rfftn(rng.standard_normal((L, L)), s=(L, L), norm="ortho")
        psi_hat = mean_hat + sv_k * z_hat
        psi = np.fft.irfftn(psi_hat, s=(L, L), norm="ortho").real
        return psi, {"method": "fixedq_fft", "t_diff": t_diff, "t_diff_prod": self.t_diff_prod}


class FixedQGaussianPBCContext(ContextBase):
    r"""Gaussian PBC context with Q fixed at a reference t_diff.

    Used with :class:`FixedQFFTGaussianAggregator` for Replica Exchange,
    where replicas at high t_diff would violate the feasibility condition with
    the standard t_diff-dependent Q.
    """

    def __init__(self, J: float, L: int, noise_scheduler, *, t_diff_prod: float, h: float = 0.0):
        self.J = float(J)
        self.L = int(L)
        self._ns = noise_scheduler
        self._lam = _laplacian_spectrum_rfft(L)
        self._rfft_weights = _rfft_last_axis_weights(L)
        self._h = float(h)
        self.t_diff_prod = float(t_diff_prod)
        a2, s2 = _vp_params_at_t_diff(noise_scheduler, t_diff_prod)
        self._q_prod = _precision_spectrum(self._lam, self.J, a2, s2)

    def tempering_factor(self, t_diff):
        return 1.0
    
    def log_prob(self, s, t_diff):
        s = np.asarray(s, dtype=np.float64)
        s_hat = np.fft.rfftn(s, s=(self.L, self.L), norm="ortho")
        result = float(-0.5 * np.sum(self._rfft_weights * self._q_prod * np.abs(s_hat) ** 2))
        if self._h != 0.0:
            a_prod = np.sqrt(_vp_params_at_t_diff(self._ns, self.t_diff_prod)[0])
            result += float(self._h / a_prod * np.sum(s))
        return result

    def grad_log_prob(self, s, t_diff):
        s = np.asarray(s, dtype=np.float64)
        s_hat = np.fft.rfftn(s, s=(self.L, self.L), norm="ortho")
        grad = np.fft.irfftn(-self._q_prod * s_hat, s=(self.L, self.L), norm="ortho")
        if self._h != 0.0:
            a_prod = np.sqrt(_vp_params_at_t_diff(self._ns, self.t_diff_prod)[0])
            grad = grad + self._h / a_prod
        return grad


# =====================================================================
# Replica Exchange runner
# =====================================================================


class LatticeRERunner:
    r"""Replica Exchange GG-PA runner for the lattice :math:`\phi^4` system.

    Runs *R* replicas at a geometric ladder of diffusion times t_diff.
    Uses **ragged-batch** GPU denoising (all active replicas in one GPU call
    per timestep) and batched FFT aggregation with **fixed** context
    precision *Q* precomputed at ``t_diff_prod = t_diff_ladder[0]``.

    Parameters
    ----------
    diffusion_model : SimpleDiffusion
        Trained 1-D diffusion model.
    J : float
        Coupling constant.
    L : int
        Lattice side length.
    t_diff_ladder : sequence of float
        Ascending diffusion-time values; ``t_diff_ladder[0]`` is the production replica.
    noise_scheduler : NoiseScheduler
        From ``diffusion_model.noise_scheduler``.
    h : float
        External field (default 0).
    device : str or torch.device
    """

    def __init__(
        self,
        diffusion_model,
        J: float,
        L: int,
        t_diff_ladder,
        noise_scheduler,
        *,
        h: float = 0.0,
        device: str | torch.device = "cpu",
        init_mode: str = "uniform",
    ):
        self.J = float(J)
        self.L = int(L)
        self.L2 = L * L
        self.t_diff_ladder = list(t_diff_ladder)
        self.n_rep = len(self.t_diff_ladder)
        self._ns = noise_scheduler
        self._h = float(h)
        self.device = device
        self.init_mode = str(init_mode)

        T_diff = noise_scheduler.num_timesteps
        t_diff_prod = self.t_diff_ladder[0]

        # Context precision Q fixed at t_diff_prod
        lam = _laplacian_spectrum_rfft(L)
        a2_p, s2_p = _vp_params_at_t_diff(noise_scheduler, t_diff_prod)
        self._alpha_prod = float(np.sqrt(a2_p))
        self._q_prod = _precision_spectrum(lam, self.J, a2_p, s2_p)

        # VP forward-kernel parameters per replica (for swap criterion)
        self._alpha = np.array(
            [np.sqrt(_vp_params_at_t_diff(noise_scheduler, t)[0]) for t in self.t_diff_ladder]
        )
        self._inv_s2 = np.array(
            [1.0 / _vp_params_at_t_diff(noise_scheduler, t)[1] for t in self.t_diff_ladder]
        )

        # Ragged-batch structure: which replicas are active at each t_idx
        start_indices = [
            max(0, min(T_diff - 1, int(round(t * (T_diff - 1)))))
            for t in self.t_diff_ladder
        ]
        self._max_start = max(start_indices)
        self._active_at = {
            t: torch.tensor(
                [i for i, si in enumerate(start_indices) if si >= t],
                dtype=torch.long, device=device,
            )
            for t in range(self._max_start + 1)
        }

        # Un-compiled denoiser / velocity model for variable-batch ragged reverse
        self._velocity_model = getattr(
            diffusion_model.velocity_model, "_orig_mod", diffusion_model.velocity_model
        )

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _batched_denoise(self, psi_gpu: torch.Tensor) -> torch.Tensor:
        """Ragged-batch reverse diffusion for all replicas."""
        ns = self._ns
        T_diff = ns.num_timesteps
        x = psi_gpu.clone()
        for t_idx in range(self._max_start, -1, -1):
            ai = self._active_at[t_idx]
            xa = x[ai].reshape(-1, 1)
            B = xa.shape[0]
            t_cont = torch.full((B,), t_idx / (T_diff - 1), device=self.device)
            # Z₂ symmetry: denoise [xa, -xa] in one call
            v_both = self._velocity_model(torch.cat([xa, -xa]), t_cont.repeat(2))
            v = 0.5 * (v_both[:B] - v_both[B:])
            abar = ns.alpha_bars[t_idx]
            sigma = torch.sqrt(1.0 - abar)
            eps = torch.sqrt(abar) * v + sigma * xa
            mean = (xa - (ns.betas[t_idx] / sigma) * eps) / torch.sqrt(ns.alphas[t_idx])
            if t_idx > 0:
                tilde_beta = (
                    (1.0 - ns.alpha_bars[t_idx - 1]) / (1.0 - abar) * ns.betas[t_idx]
                )
                xa = mean + torch.sqrt(tilde_beta) * torch.randn_like(xa)
            else:
                xa = mean
            x[ai] = xa.reshape(-1, self.L2)
        return x

    def _fft_aggregate_all(
        self,
        phi_gpu: torch.Tensor,
        psi_np: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """FFT aggregate every replica with fixed Q; update *psi_np* in place.

        Returns ``phi_np`` (denoised fields for swap use).
        """
        L = self.L
        phi_np = phi_gpu.reshape(self.n_rep, L, L).cpu().numpy()
        phi_out = phi_np.copy()
        q_prod = self._q_prod[np.newaxis, :, :]
        alpha = self._alpha[:, np.newaxis, np.newaxis]
        inv_s2 = self._inv_s2[:, np.newaxis, np.newaxis]
        v_k = 1.0 / (inv_s2 + q_prod)

        phi_hat = np.fft.rfftn(phi_np, axes=(1, 2), norm="ortho")
        mean_hat = v_k * (alpha * inv_s2) * phi_hat
        if self._h != 0.0:
            mean_hat[:, 0, 0] = mean_hat[:, 0, 0] + v_k[:, 0, 0] * (self._h * L / self._alpha_prod)

        z_hat = np.fft.rfftn(
            rng.standard_normal((self.n_rep, L, L)),
            axes=(1, 2),
            norm="ortho",
        )
        psi_np[:] = np.fft.irfftn(
            mean_hat + np.sqrt(v_k) * z_hat,
            s=(L, L),
            axes=(1, 2),
            norm="ortho",
        ).real
        return phi_out

    def _make_initial_psi(self, seed: int) -> np.ndarray:
        """Sample the initial signal state for all replicas."""
        rng = np.random.default_rng(seed)
        if self.init_mode == "uniform":
            return rng.uniform(-2.0, 2.0, size=(self.n_rep, self.L, self.L)).astype(np.float64)
        if self.init_mode in {"ising", "pm1"}:
            return rng.choice([-1.0, 1.0], size=(self.n_rep, self.L, self.L)).astype(np.float64)
        raise ValueError(
            f"Unsupported init_mode={self.init_mode!r}. Expected 'uniform' or 'ising'."
        )

    def _attempt_swaps(
        self,
        psi_np: np.ndarray,
        phi_np: np.ndarray,
        rng,
        *,
        parity: int,
        swap_attempts_by_pair: np.ndarray | None = None,
        swap_accepted_by_pair: np.ndarray | None = None,
    ) -> tuple[int, int]:
        r"""Attempt one odd/even swap round via VP forward-kernel log-density.

        .. math::

            \log\alpha = -\tfrac{1}{2}\bigl[
              s_i(\|\psi_j - \alpha_i\phi_j\|^2 - \|\psi_i - \alpha_i\phi_i\|^2)
            + s_j(\|\psi_i - \alpha_j\phi_i\|^2 - \|\psi_j - \alpha_j\phi_j\|^2)
            \bigr]
        """
        n_rep = self.n_rep
        ii = np.arange(parity, n_rep - 1, 2)
        if len(ii) == 0:
            return 0, 0

        jj = ii + 1
        pi = psi_np[ii].reshape(len(ii), -1).astype(np.float64)
        pj = psi_np[jj].reshape(len(ii), -1).astype(np.float64)
        fi = phi_np[ii].reshape(len(ii), -1).astype(np.float64)
        fj = phi_np[jj].reshape(len(ii), -1).astype(np.float64)
        ai = self._alpha[ii, None]
        aj = self._alpha[jj, None]
        si = self._inv_s2[ii]
        sj = self._inv_s2[jj]
        sq = lambda r: (r * r).sum(axis=1)  # noqa: E731
        log_a = -0.5 * (
            si * (sq(pj - ai * fj) - sq(pi - ai * fi))
            + sj * (sq(pi - aj * fi) - sq(pj - aj * fj))
        )
        u = np.log(rng.uniform(size=len(ii)))
        acc = (log_a >= 0) | (u < log_a)

        if swap_attempts_by_pair is not None:
            swap_attempts_by_pair[ii] += 1
        if swap_accepted_by_pair is not None:
            swap_accepted_by_pair[ii] += acc.astype(int)

        for idx in np.where(acc)[0]:
            i, j = int(ii[idx]), int(jj[idx])
            psi_np[i], psi_np[j] = psi_np[j].copy(), psi_np[i].copy()
            phi_np[i], phi_np[j] = phi_np[j].copy(), phi_np[i].copy()
        return int(acc.sum()), int(len(ii))

    def run_sweeps(
        self,
        n_sweeps: int,
        record_interval: int = 1,
        *,
        seed: int = 456,
        verbose: bool = True,
        log_every: int | None = None,
        record_all_replicas: bool = False,
    ) -> dict:
        """Execute the replica-exchange chain for a fixed number of sweeps.

        One sweep consists of denoise -> record (optional) -> aggregate -> swap.
        The swap parity alternates each sweep: (0,1),(2,3),... then
        (1,2),(3,4),...
        """
        if n_sweeps <= 0:
            raise ValueError("n_sweeps must be positive.")
        if record_interval <= 0:
            raise ValueError("record_interval must be positive.")

        L, L2, n_rep = self.L, self.L2, self.n_rep
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        psi_all = self._make_initial_psi(seed)
        phi_all = np.zeros_like(psi_all)
        mag_records: list[np.ndarray] = []
        swap_attempts_by_pair = np.zeros(n_rep - 1, dtype=int)
        swap_accepted_by_pair = np.zeros(n_rep - 1, dtype=int)
        swap_acc = 0
        swap_att = 0
        t0 = time.time()

        if log_every is None:
            log_every = max(1, n_sweeps // 10)

        for sweep in range(n_sweeps):
            psi_flat = torch.tensor(
                psi_all.reshape(n_rep, L2), dtype=torch.float32, device=self.device,
            )
            phi_flat = self._batched_denoise(psi_flat)
            phi_all = self._fft_aggregate_all(phi_flat, psi_all, rng)

            if sweep % record_interval == 0:
                mag_records.append(np.mean(phi_all, axis=(1, 2), dtype=np.float64))

            n_acc, n_att = self._attempt_swaps(
                psi_all,
                phi_all,
                rng,
                parity=sweep % 2,
                swap_attempts_by_pair=swap_attempts_by_pair,
                swap_accepted_by_pair=swap_accepted_by_pair,
            )
            swap_acc += n_acc
            swap_att += n_att

            if verbose and (sweep + 1) % log_every == 0:
                rate = swap_acc / max(swap_att, 1)
                print(
                    f"  Sweep {sweep+1}/{n_sweeps}  "
                    f"m_prod={float(np.mean(phi_all[0])):+.4f}  "
                    f"swap_rate={rate:.3f}  "
                    f"[{time.time()-t0:.1f}s]"
                )

        wall = time.time() - t0
        rate = swap_acc / max(swap_att, 1)
        mag_arr = np.asarray(mag_records, dtype=np.float64)
        if mag_arr.ndim == 1:
            mag_arr = mag_arr[:, np.newaxis]

        if verbose:
            print(f"RE done: {mag_arr.shape[0]} records, {wall:.1f}s")
            print(f"Swap acceptance: {swap_acc}/{swap_att} = {rate:.3f}")

        result = {
            "mags": mag_arr[:, 0].copy(),
            "phi_final": phi_all[0].copy(),
            "psi_final": psi_all[0].copy(),
            "phi_final_replicas": phi_all.copy(),
            "psi_final_replicas": psi_all.copy(),
            "swap_rate": rate,
            "swap_attempts": int(swap_att),
            "swap_accepted": int(swap_acc),
            "swap_attempts_by_pair": swap_attempts_by_pair.copy(),
            "swap_accepted_by_pair": swap_accepted_by_pair.copy(),
            "wall_time": wall,
        }
        if record_all_replicas:
            result["magnetizations"] = mag_arr.copy()
        return result

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        n_blocks: int,
        inner_steps: int,
        *,
        seed: int = 456,
        verbose: bool = True,
    ) -> dict:
        """Backward-compatible wrapper around :meth:`run_sweeps`.

        Returns
        -------
        dict
            ``mags`` : (n_blocks,) signed ⟨φ⟩ at the production replica,
            ``phi_final`` : (L, L) last denoised field of the production replica,
            ``psi_final`` : (L, L) last signal field of the production replica,
            ``swap_rate`` : overall acceptance rate,
            ``wall_time`` : elapsed seconds.
        """
        result = self.run_sweeps(
            n_sweeps=int(n_blocks) * int(inner_steps),
            record_interval=int(inner_steps),
            seed=seed,
            verbose=verbose,
            record_all_replicas=False,
        )
        return {
            "mags": result["mags"],
            "phi_final": result["phi_final"],
            "psi_final": result["psi_final"],
            "swap_rate": result["swap_rate"],
            "wall_time": result["wall_time"],
        }


# =====================================================================
# Observable helpers
# =====================================================================


def autocorrelation_function(x, max_lag: int | None = None) -> np.ndarray:
    """Normalised autocorrelation via FFT."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if max_lag is None:
        max_lag = N // 4
    x = x - np.mean(x)
    fft_x = np.fft.fft(x, n=2 * N)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:N]
    acf = acf / np.arange(N, 0, -1)
    if acf[0] > 0:
        acf = acf / acf[0]
    return acf[: max_lag + 1]


def integrated_autocorrelation_time(x, c: float = 5.0):
    """Integrated autocorrelation time with Sokal automatic windowing.

    Returns ``(tau_int, window_size, acf)``.
    """
    x = np.asarray(x, dtype=np.float64)
    max_lag = len(x) // 4
    acf = autocorrelation_function(x, max_lag)
    tau_int = 0.5
    for t in range(1, max_lag):
        tau_int += acf[t]
        if t >= c * tau_int:
            return float(tau_int), t, acf
    return float(tau_int), max_lag, acf
