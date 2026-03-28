"""DDPM diffusion model with v-prediction parameterization.

Provides a reusable diffusion model stack shared across different physical
systems (2D rings, Ginzburg-Landau lattice, etc.):

* :class:`NoiseScheduler` – cosine or linear VP noise schedule.
* :class:`ResidualModel` – residual MLP denoiser / velocity network with
  sinusoidal time embeddings.
* :class:`SimpleDiffusion` – DDPM training (v-prediction + optional SNR
  weighting) and reverse sampling.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------


class NoiseScheduler(nn.Module):
    """Variance-preserving noise schedule (cosine or linear).

    Parameters
    ----------
    num_timesteps : int
        Number of discrete diffusion steps.
    style : ``'cosine'`` | ``'linear'``
        Schedule type.  Cosine follows Nichol & Dhariwal (2021).
    device : str or torch.device
        Device on which to store the schedule buffers.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        style: Literal["cosine", "linear"] = "cosine",
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.style = style

        if style == "cosine":
            s = 0.008
            t = torch.arange(num_timesteps + 1, device=device, dtype=torch.float32)
            f = torch.cos(((t / num_timesteps + s) / (1 + s)) * math.pi / 2) ** 2
            alpha_bars = f[1:] / f[0]
            alphas = torch.empty_like(alpha_bars)
            alphas[0] = alpha_bars[0]
            alphas[1:] = alpha_bars[1:] / alpha_bars[:-1]
            betas = 1.0 - alphas
        else:
            betas = torch.linspace(0.0001, 0.02, num_timesteps, device=device)
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Return $\\bar\\alpha$ at continuous time *t* ∈ [0, 1]."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.alpha_bars.device)
        t_flat = t.flatten() if t.dim() > 0 else t.unsqueeze(0)
        idx = (t_flat * (self.num_timesteps - 1)).long().clamp(0, self.num_timesteps - 1)
        result = self.alpha_bars[idx]
        return result.squeeze(0) if t.dim() == 0 else result.reshape(t.shape)


# ---------------------------------------------------------------------------
# Denoiser / velocity network
# ---------------------------------------------------------------------------


class TimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the diffusion timestep.

    Produces a vector of dimension ``embed_dim`` from a scalar *t* ∈ [0, 1].
    Frequencies are pre-computed and stored as a non-persistent buffer to
    avoid per-call allocation.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000) / (half - 1)))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        arg = t.unsqueeze(1) * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=1)


class ResidualBlock(nn.Module):
    """Single residual block: ``x + fc2(SiLU(fc1(x) + time_proj(t)))``."""

    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x) + self.time_proj(t_emb))
        return x + self.fc2(h)


class ResidualModel(nn.Module):
    """Residual MLP denoiser / velocity network with sinusoidal time embeddings.

    Parameters
    ----------
    input_dim, output_dim : int
        Data dimensionality (typically equal).
    time_embed_dim : int
        Dimension of the sinusoidal time embedding.
    hidden_dim : int
        Width of each residual block.
    n_blocks : int
        Number of residual blocks.
    use_abs : bool
        If ``True``, the model operates on ``|x|`` and restores the original
        sign on the output. Useful for enforcing Z₂ symmetry (e.g.
        Ginzburg-Landau lattice models).
    """

    def __init__(
        self,
        input_dim: int = 2,
        time_embed_dim: int = 32,
        hidden_dim: int = 32,
        n_blocks: int = 3,
        output_dim: int = 2,
        use_abs: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_abs = use_abs

        self.time_embedding = TimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, time_embed_dim) for _ in range(n_blocks)]
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_embedding(t))

        if self.use_abs:
            sign = torch.sign(x)
            x = torch.abs(x)

        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        out = self.out_proj(h)

        if self.use_abs:
            out = out * sign
        return out


# Backward-compatible semantic alias.
VelocityModel = ResidualModel


# ---------------------------------------------------------------------------
# Diffusion model
# ---------------------------------------------------------------------------


def _snr_weight(alpha_bar: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    """SNR-based per-sample loss weight: ``snr / (snr + gamma)``."""
    snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
    return snr / (snr + gamma)


class SimpleDiffusion(nn.Module):
    """DDPM with v-prediction parameterisation and cosine noise schedule.

    Parameters
    ----------
    score_model : ResidualModel, optional
        Legacy name for the denoiser / velocity network.
    velocity_model : ResidualModel, optional
        Preferred name for the v-prediction network used by the reverse
        process. Exactly one of ``score_model`` and ``velocity_model`` must
        be provided.
    noise_scheduler : NoiseScheduler
        VP noise schedule.
    snr_gamma : float
        Gamma parameter for SNR-weighted loss.  Larger values down-weight
        high-SNR (low-noise) timesteps less aggressively.
    time_beta_a, time_beta_b : float
        Parameters of the ``Beta(a, b)`` distribution used to sample
        training timesteps.  ``Beta(0.5, 1.0)`` biases toward lower noise
        levels; ``Beta(0.2, 1.0)`` biases even more.
    """

    def __init__(
        self,
        score_model: ResidualModel | None = None,
        noise_scheduler: NoiseScheduler | None = None,
        snr_gamma: float = 5.0,
        time_beta_a: float = 0.5,
        time_beta_b: float = 1.0,
        *,
        velocity_model: ResidualModel | None = None,
    ):
        super().__init__()
        if velocity_model is not None:
            if score_model is not None and velocity_model is not score_model:
                raise ValueError(
                    "Received both score_model and velocity_model with different objects. "
                    "Use only one name for the denoiser network."
                )
            score_model = velocity_model
        if score_model is None:
            raise ValueError("A denoiser / velocity network must be provided.")
        if noise_scheduler is None:
            raise ValueError("noise_scheduler must be provided.")
        self.score_model = score_model
        self.noise_scheduler = noise_scheduler
        self.num_timesteps = noise_scheduler.num_timesteps
        self.snr_gamma = snr_gamma
        self.time_beta_a = time_beta_a
        self.time_beta_b = time_beta_b

    @property
    def velocity_model(self) -> ResidualModel:
        """Alias for the v-prediction network."""
        return self.score_model

    @property
    def denoiser(self) -> ResidualModel:
        """Alias for the v-prediction network."""
        return self.score_model

    @property
    def backbone(self) -> ResidualModel:
        """Alias for the denoiser network used by the diffusion model."""
        return self.score_model

    # ---- persistence ----

    def save(self, path: str) -> None:
        """Save model weights and config to *path*."""
        torch.save(
            {
                "score_model_state_dict": self.score_model.state_dict(),
                "velocity_model_state_dict": self.score_model.state_dict(),
                "noise_scheduler_state_dict": self.noise_scheduler.state_dict(),
                "scheduler_config": {
                    "num_timesteps": self.noise_scheduler.num_timesteps,
                    "style": self.noise_scheduler.style,
                },
                "score_model_config": {
                    "input_dim": self.score_model.input_dim,
                    "output_dim": self.score_model.output_dim,
                    "use_abs": self.score_model.use_abs,
                },
                "velocity_model_config": {
                    "input_dim": self.score_model.input_dim,
                    "output_dim": self.score_model.output_dim,
                    "use_abs": self.score_model.use_abs,
                },
                "diffusion_config": {
                    "snr_gamma": self.snr_gamma,
                    "time_beta_a": self.time_beta_a,
                    "time_beta_b": self.time_beta_b,
                },
            },
            path,
        )

    @classmethod
    def load_from_checkpoint(cls, path: str, device: str | torch.device = "cpu") -> "SimpleDiffusion":
        """Reconstruct a :class:`SimpleDiffusion` from a saved checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)

        ns = NoiseScheduler(
            num_timesteps=ckpt["scheduler_config"]["num_timesteps"],
            style=ckpt["scheduler_config"]["style"],
            device=device,
        )

        sd = ckpt.get("velocity_model_state_dict", ckpt["score_model_state_dict"])
        hidden_dim = sd["in_proj.weight"].shape[0]
        n_blocks = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".fc1.weight"))
        time_embed_dim = sd["time_mlp.0.weight"].shape[1]
        mc = ckpt.get("velocity_model_config", ckpt["score_model_config"])

        model = ResidualModel(
            input_dim=mc["input_dim"],
            time_embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            output_dim=mc["output_dim"],
            use_abs=mc.get("use_abs", False),
        )

        diff_cfg = ckpt.get("diffusion_config", {})
        diff = cls(
            velocity_model=model,
            noise_scheduler=ns,
            snr_gamma=diff_cfg.get("snr_gamma", 5.0),
            time_beta_a=diff_cfg.get("time_beta_a", 0.5),
            time_beta_b=diff_cfg.get("time_beta_b", 1.0),
        )
        diff.velocity_model.load_state_dict(sd)
        diff.noise_scheduler.load_state_dict(ckpt["noise_scheduler_state_dict"])
        return diff.to(device)

    # ---- training ----

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute v-prediction training loss with SNR weighting."""
        B = x0.shape[0]
        t = torch.distributions.Beta(self.time_beta_a, self.time_beta_b).sample((B,)).to(x0.device)
        abar = self.noise_scheduler.get_alpha_bar(t).unsqueeze(1)
        alpha = torch.sqrt(abar)
        sigma = torch.sqrt(1.0 - abar)
        eps = torch.randn_like(x0)
        xt = alpha * x0 + sigma * eps
        v_target = alpha * eps - sigma * x0
        v_pred = self.velocity_model(xt, t)
        per_sample = (v_pred - v_target).pow(2).mean(dim=1)
        w = _snr_weight(abar, gamma=self.snr_gamma).squeeze(1)
        return (w * per_sample).mean()

    # ---- sampling ----

    @torch.no_grad()
    def reverse(
        self,
        xt: torch.Tensor,
        t_start: float = 1.0,
        t_end: float = 0.0,
        enforce_symmetry: bool = False,
    ) -> torch.Tensor:
        """DDPM reverse process from *t_start* to *t_end*.

        Parameters
        ----------
        enforce_symmetry : bool
            If ``True``, average ``v(x)`` and ``-v(-x)`` at each step to
            enforce Z₂ symmetry (useful for lattice phi⁴ models).
        """
        x = xt
        B = x.shape[0]
        T = self.num_timesteps
        start_idx = max(0, min(T - 1, int(round(t_start * (T - 1)))))
        end_idx = max(0, min(T - 1, int(round(t_end * (T - 1)))))

        for t_idx in range(start_idx, end_idx - 1, -1):
            t_cont = torch.full((B,), t_idx / (T - 1), device=x.device)
            abar = self.noise_scheduler.alpha_bars[t_idx]
            alpha_sq = torch.sqrt(abar)
            sigma = torch.sqrt(1.0 - abar)

            if enforce_symmetry:
                # Batch [x, -x] into single forward pass (halves NN calls)
                x_both = torch.cat([x, -x], dim=0)
                t_both = t_cont.repeat(2)
                v_both = self.velocity_model(x_both, t_both)
                v = 0.5 * (v_both[:B] - v_both[B:])
            else:
                v = self.velocity_model(x, t_cont)

            eps = alpha_sq * v + sigma * x
            beta = self.noise_scheduler.betas[t_idx]
            alpha_t = self.noise_scheduler.alphas[t_idx]
            mean = (x - (beta / sigma) * eps) / torch.sqrt(alpha_t)

            if t_idx > end_idx:
                abar_prev = (
                    self.noise_scheduler.alpha_bars[t_idx - 1]
                    if t_idx > 0
                    else x.new_tensor(1.0)
                )
                tilde_beta = (1.0 - abar_prev) / (1.0 - abar) * beta
                x = mean + torch.sqrt(tilde_beta) * torch.randn_like(x)
            else:
                x = mean
        return x

    @torch.no_grad()
    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Generate samples by running the full reverse process from pure noise."""
        xT = torch.randn(batch_size, self.velocity_model.input_dim, device=device)
        return self.reverse(xT)
