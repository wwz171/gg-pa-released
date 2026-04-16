r"""Alanine dipeptide torsion-space systems for GG-PA.

.. currentmodule:: ggpa.systems.alanine_dipeptide

This module provides GG-PA components for sampling alanine dipeptide systems
with a shared torsion-space diffusion prior coupled to OpenMM molecular
dynamics. The public workflows currently supported here are:

- alanine dipeptide + Na+ (single monomer, single-``t_diff`` GG-PA)
- alanine dipeptide dimer (two monomers, replica-exchange GG-PA)

Signal space: full-atom Cartesian coordinates  s ∈ R^{N_atoms × 3}  (nm)
Each client projects s → (φ, ψ) dihedral angles for one monomer.

Components
----------
Geometry & helpers:
    :func:`compute_dihedral_angle`, :func:`compute_dihedrals`,
    :func:`extract_monomer_torsion_indices`,
    :func:`extract_dimer_torsion_indices`, :func:`wrap_angle`

Torsion diffusion model:
    :class:`TorsionSimpleMLP`, :class:`WrappedGaussianForward`,
    :class:`TorsionDiffusion`

GG-PA adapters:
    :class:`TorsionProjector` (ProjectorBase),
    :class:`WrappedGaussianForwardProcess` (ForwardProcessBase),
    :class:`TorsionDiffusionClient` (ClientBase),
    :class:`OpenMMMDAggregator` (AggregationBase),
    :class:`OpenMMIonAggregator` (AggregationBase)

Pipeline:
    :func:`build_monomer_sodium_pipeline`,
    :func:`build_dimer_pipeline`,
    :class:`AlanineReplicaExchange`
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ggpa.client.base import ClientBase, ForwardProcessBase, ProjectorBase
from ggpa.server.base import AggregationBase


# ══════════════════════════════════════════════════════════════════════════════
# §0  Utility
# ══════════════════════════════════════════════════════════════════════════════

def wrap_angle(x):
    """Wrap angles to [-π, π).  Works with torch.Tensor *and* numpy arrays."""
    if isinstance(x, torch.Tensor):
        return (x + math.pi) % (2 * math.pi) - math.pi
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


# ══════════════════════════════════════════════════════════════════════════════
# §1  Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_dihedral_angle(positions: np.ndarray, indices: np.ndarray) -> float:
    """Compute a single dihedral angle defined by 4 atom indices.

    Args:
        positions: (N, 3) atom coordinates in arbitrary units.
        indices:   (4,) 0-based atom indices [i, j, k, l].

    Returns:
        Dihedral angle in radians ∈ (-π, π].
    """
    p = positions[indices]
    b0 = -(p[1] - p[0])
    b1 = p[2] - p[1]
    b2 = p[3] - p[2]

    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1e-12:
        return 0.0
    b1 = b1 / b1_norm

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def compute_dihedrals(positions: np.ndarray, indices_array: np.ndarray) -> np.ndarray:
    """Compute multiple dihedral angles.

    Args:
        positions:     (N, 3) atom coordinates.
        indices_array: (K, 4) atom index array — each row defines one dihedral.

    Returns:
        (K,) array of dihedral angles in radians.
    """
    return np.array(
        [compute_dihedral_angle(positions, idx) for idx in indices_array],
        dtype=np.float64,
    )


def extract_dimer_torsion_indices(pdb_path: str) -> Dict:
    """Extract φ/ψ atom indices for each chain via mdtraj.

    Returns:
        {
          'chain_A': {'phi': (4,), 'psi': (4,), 'all': (2,4)},
          'chain_B': {'phi': (4,), 'psi': (4,), 'all': (2,4)},
          'all':     (4, 4),          # stacked [φ_A, ψ_A, φ_B, ψ_B]
          'n_atoms': int,
          'chain_atom_lists': [[chain_A atoms], [chain_B atoms]],
        }
    """
    import mdtraj as md

    traj = md.load(pdb_path)
    phi_idx, _ = md.compute_phi(traj)
    psi_idx, _ = md.compute_psi(traj)

    assert phi_idx.shape[0] == 2, f"Expected 2 φ, got {phi_idx.shape[0]}"
    assert psi_idx.shape[0] == 2, f"Expected 2 ψ, got {psi_idx.shape[0]}"

    chain_a = {
        "phi": phi_idx[0],
        "psi": psi_idx[0],
        "all": np.stack([phi_idx[0], psi_idx[0]]),
    }
    chain_b = {
        "phi": phi_idx[1],
        "psi": psi_idx[1],
        "all": np.stack([phi_idx[1], psi_idx[1]]),
    }
    all_indices = np.stack([phi_idx[0], psi_idx[0], phi_idx[1], psi_idx[1]])

    return {
        "chain_A": chain_a,
        "chain_B": chain_b,
        "all": all_indices,
        "n_atoms": traj.n_atoms,
        "chain_atom_lists": [
            [a.index for a in list(traj.topology.chains)[0].atoms],
            [a.index for a in list(traj.topology.chains)[1].atoms],
        ],
    }


def extract_monomer_torsion_indices(pdb_path: str) -> Dict[str, Any]:
    """Extract φ/ψ atom indices for a single alanine-dipeptide monomer.

    Returns:
        {
          'phi': (4,),
          'psi': (4,),
          'all': (2, 4),      # stacked [φ, ψ]
          'n_atoms': int,
          'atom_list': [atom ids in the monomer],
          'torsion_atom_list': [unique atoms that appear in φ/ψ],
        }
    """
    import mdtraj as md

    traj = md.load(pdb_path)
    phi_idx, _ = md.compute_phi(traj)
    psi_idx, _ = md.compute_psi(traj)

    assert phi_idx.shape[0] == 1, f"Expected 1 φ, got {phi_idx.shape[0]}"
    assert psi_idx.shape[0] == 1, f"Expected 1 ψ, got {psi_idx.shape[0]}"

    all_indices = np.stack([phi_idx[0], psi_idx[0]])
    torsion_atom_list = sorted(set(int(a) for row in all_indices for a in row))

    return {
        "phi": phi_idx[0],
        "psi": psi_idx[0],
        "all": all_indices,
        "n_atoms": traj.n_atoms,
        "atom_list": [a.index for a in traj.topology.atoms],
        "torsion_atom_list": torsion_atom_list,
    }


def extract_monomer_oxygen_indices(pdb_path: str) -> Dict[str, Any]:
    """Return the two carbonyl oxygen atom indices for one alanine-dipeptide monomer.

    The public AD+Na analysis uses the O-O distance between the two backbone
    carbonyl oxygens. We identify them directly from topology rather than
    hard-coding atom numbers.
    """
    import mdtraj as md

    traj = md.load(pdb_path)
    oxygen_atoms = [
        atom for atom in traj.topology.atoms
        if atom.name == "O"
    ]
    assert len(oxygen_atoms) == 2, (
        f"Expected exactly 2 monomer oxygen atoms named 'O', got {len(oxygen_atoms)}"
    )
    return {
        "pair": np.array([oxygen_atoms[0].index, oxygen_atoms[1].index], dtype=np.int64),
        "atom_names": [oxygen_atoms[0].name, oxygen_atoms[1].name],
        "residue_names": [oxygen_atoms[0].residue.name, oxygen_atoms[1].residue.name],
    }


# ══════════════════════════════════════════════════════════════════════════════
# §2  Torsion diffusion model (score-based VE on the torus)
# ══════════════════════════════════════════════════════════════════════════════

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        arg = t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=1)


class TorsionSimpleMLP(nn.Module):
    """Simple MLP score network for small torsion systems.

    Input  (B, L, K) → Output (B, L, K)   [predicts σ · score]
    """

    def __init__(self, n_angles: int = 2, hidden_dim: int = 64,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_angles = n_angles

        self.angle_embed = nn.Sequential(
            nn.Linear(2 * n_angles, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.merge_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.time_embed = TimeEmbedding(hidden_dim)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_angles),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, K = x.shape
        x_flat = x.reshape(B * L, K)

        feat = torch.cat([torch.sin(x_flat), torch.cos(x_flat)], dim=-1)
        h = self.angle_embed(feat)

        t_exp = t.unsqueeze(1).expand(B, L).reshape(B * L)
        t_emb = self.time_embed(t_exp)
        h = self.merge_layer(torch.cat([h, t_emb], dim=-1))

        for layer in self.layers:
            h = h + layer(h)
            h = self.norm(h)

        out = self.out(h)
        return out.reshape(B, L, K)


class WrappedGaussianForward:
    """VE-style forward diffusion on the torus.

    Shape convention: (B, L, K)
    t ∈ [0, 1]  (0 = clean, 1 = maximum noise)
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._device = torch.device('cpu')
        self._log_sigma_min = np.log(sigma_min)
        self._log_sigma_max = np.log(sigma_max)

    def to(self, device):
        self._device = device
        return self

    @property
    def device(self):
        return self._device

    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        log_sigma = self._log_sigma_min + t * (self._log_sigma_max - self._log_sigma_min)
        return torch.exp(log_sigma)

    def forward(self, x0: torch.Tensor, t: torch.Tensor,
                noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sigma = self.get_sigma(t).view(-1, 1, 1)
        x_t = wrap_angle(x0 + sigma * noise)
        return x_t, noise

    def score(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor,
              k_max: int = 5) -> torch.Tensor:
        sigma = self.get_sigma(t).view(-1, 1, 1, 1)
        ks = torch.arange(-k_max, k_max + 1, device=self.device).view(1, 1, 1, -1)
        diff = wrap_angle(xt - x0)
        diff_k = diff.unsqueeze(-1) + ks * (2 * torch.pi)
        log_prob_k = -0.5 * (diff_k / sigma) ** 2
        weights = F.softmax(log_prob_k, dim=-1)
        effective_diff = torch.sum(weights * diff_k, dim=-1)
        sigma_sq = sigma.squeeze(-1) ** 2
        return -effective_diff / (sigma_sq + 1e-8)


class TorsionDiffusion:
    """Full torsion diffusion pipeline: training loss + ancestral sampling."""

    def __init__(self, model: nn.Module, forward_process: WrappedGaussianForward,
                 geo_weight: float = 0.0):
        self.model = model
        self.forward_process = forward_process
        self.geo_weight = geo_weight

    @torch.no_grad()
    def sample(self, n_samples: int, length: int, n_angles: int,
               device: torch.device, n_steps: int = 100,
               masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.model.eval()
        x = torch.rand(n_samples, length, n_angles, device=device) * 2 * np.pi - np.pi
        if masks is not None:
            if masks.shape[0] != n_samples and masks.shape[0] == 1:
                masks = masks.expand(n_samples, -1, -1)
            x = x * masks
        timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        for i in range(n_steps):
            x = self._reverse_step(x, timesteps[i], timesteps[i + 1], device, masks)
        return x

    @torch.no_grad()
    def _reverse_step(self, x: torch.Tensor, t_curr: float, t_next: float,
                      device: torch.device,
                      masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        t_batch = torch.full((B,), t_curr, device=device, dtype=torch.float)
        sigma_t = self.forward_process.get_sigma(t_batch)

        model_output = self.model(x, t_batch)
        score_pred = model_output / (sigma_t.view(-1, 1, 1) + 1e-8)
        x0_pred = wrap_angle(x + (sigma_t.view(-1, 1, 1) ** 2) * score_pred)

        if t_next > 0:
            t_next_batch = torch.full((B,), t_next, device=device, dtype=torch.float)
            sigma_next = self.forward_process.get_sigma(t_next_batch)
            x = wrap_angle(x0_pred + sigma_next.view(-1, 1, 1) * torch.randn_like(x))
        else:
            x = x0_pred

        if masks is not None:
            x = x * masks
        return x

    @classmethod
    def load_from_file(cls, path: str, device: str = 'cpu') -> 'TorsionDiffusion':
        """Load a torsion diffusion prior from a checkpoint file.

        Args:
            path: Path to checkpoint (.pt).
            device: torch device string.

        Returns:
            TorsionDiffusion instance with model in eval mode.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = ckpt['model_config']
        model_type = config.get('type', 'TorsionSimpleMLP')
        if model_type == 'TorsionSimpleMLP':
            model = TorsionSimpleMLP(
                n_angles=config['n_angles'],
                hidden_dim=config['hidden_dim'],
                n_layers=config['n_layers'],
                dropout=config.get('dropout', 0.1),
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        model.load_state_dict(ckpt['state_dict'])
        fp_cfg = ckpt['forward_process']
        forward_process = WrappedGaussianForward(
            sigma_min=fp_cfg['sigma_min'],
            sigma_max=fp_cfg['sigma_max'],
        ).to(device)
        init_args = ckpt.get('init_args', {})
        diffusion = cls(model=model, forward_process=forward_process,
                        geo_weight=init_args.get('geo_weight', 0.0))
        model.eval()
        return diffusion

    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = 'cpu') -> 'TorsionDiffusion':
        """Compatibility alias matching the naming used by other GG-PA priors."""
        return cls.load_from_file(path, device=device)


# ══════════════════════════════════════════════════════════════════════════════
# §3  GG-PA adapter: TorsionProjector
# ══════════════════════════════════════════════════════════════════════════════

class TorsionProjector(ProjectorBase):
    """Projects full-atom positions → (φ, ψ) dihedral angles for one chain.

    forward(s):  s ∈ R^{N×3}  →  y ∈ R^{2}  (radians)
    """

    def __init__(self, torsion_indices: np.ndarray):
        self.torsion_indices = np.asarray(torsion_indices, dtype=int)
        assert self.torsion_indices.shape == (2, 4)

    def forward(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        return compute_dihedrals(s, self.torsion_indices)

    def backprop_gradient(self, s: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """Chain-rule gradient via finite differences (O(3·N_atoms))."""
        s = np.asarray(s, dtype=np.float64)
        grad_y = np.asarray(grad_y, dtype=np.float64)

        eps = 1e-5
        grad_s = np.zeros_like(s)
        s_flat = s.ravel()

        for i in range(len(s_flat)):
            s_p = s_flat.copy(); s_p[i] += eps
            s_m = s_flat.copy(); s_m[i] -= eps
            y_p = self.forward(s_p.reshape(s.shape))
            y_m = self.forward(s_m.reshape(s.shape))
            dy = wrap_angle(y_p - y_m) / (2.0 * eps)
            grad_s.ravel()[i] = np.dot(dy.ravel(), grad_y.ravel())

        return grad_s


# ══════════════════════════════════════════════════════════════════════════════
# §4  GG-PA adapter: WrappedGaussianForwardProcess (VE on the torus)
# ══════════════════════════════════════════════════════════════════════════════

class WrappedGaussianForwardProcess(ForwardProcessBase):
    r"""VE-style wrapped Gaussian forward process for torsion angles.

    q_t_diff(y | x) = WG(y; x, σ(t_diff)²)   with  σ(t_diff) = σ_min · (σ_max/σ_min)^t_diff
    α(t_diff) = 1.0  (VE schedule keeps the mean at x).
    """

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 3.0,
                 k_max: int = 5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.k_max = k_max
        self._log_ratio = np.log(sigma_max / sigma_min)

    def alpha(self, t_diff: float) -> float:
        return 1.0

    def sigma(self, t_diff: float) -> float:
        return self.sigma_min * np.exp(self._log_ratio * t_diff)

    def log_q_fwd(self, y: np.ndarray, x: np.ndarray, t_diff: float) -> float:
        """log q_t_diff(y | x) — wrapped Gaussian log-density."""
        y = np.asarray(y, dtype=np.float64).ravel()
        x = np.asarray(x, dtype=np.float64).ravel()
        sig = self.sigma(t_diff)
        sig2 = sig * sig
        K = len(y)

        ks = np.arange(-self.k_max, self.k_max + 1)
        diff = wrap_angle(y - x)
        diff_k = diff[:, None] + ks[None, :] * 2.0 * np.pi
        log_terms = -0.5 * diff_k ** 2 / sig2

        log_max = log_terms.max(axis=1, keepdims=True)
        lse = log_max.squeeze(-1) + np.log(np.sum(np.exp(log_terms - log_max), axis=1))
        log_norm = -0.5 * np.log(2.0 * np.pi * sig2)
        return float(np.sum(log_norm + lse))

    def grad_log_q_fwd(self, y: np.ndarray, x: np.ndarray, t_diff: float) -> np.ndarray:
        """∇_y log q_t_diff(y | x) via truncated winding-number sum."""
        y = np.asarray(y, dtype=np.float64).ravel()
        x = np.asarray(x, dtype=np.float64).ravel()
        sig = self.sigma(t_diff)
        sig2 = sig * sig

        ks = np.arange(-self.k_max, self.k_max + 1)
        diff = wrap_angle(y - x)
        diff_k = diff[:, None] + ks[None, :] * 2.0 * np.pi
        log_w = -0.5 * diff_k ** 2 / sig2

        log_max = log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w - log_max)
        w /= w.sum(axis=1, keepdims=True)

        eff_diff = np.sum(w * diff_k, axis=1)
        return -eff_diff / (sig2 + 1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# §5  Partial reverse diffusion helper
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def reverse_from_t_diff(
    diffusion: TorsionDiffusion,
    y_np: np.ndarray,
    t_diff: float,
    n_steps: int = 50,
    device: str = "cpu",
) -> np.ndarray:
    """Run reverse SDE from t=t_diff to t≈0.

    Args:
        diffusion:  TorsionDiffusion loaded from checkpoint.
        y_np:       (K,) noisy dihedral angles (radians).
        t_diff:        diffusion time at which y was observed.
        n_steps:    total steps (actual steps = max(t_diff * n_steps, 2)).
        device:     torch device.

    Returns:
        (K,) denoised angles x₀ (radians).
    """
    model = diffusion.model.to(device)
    model.eval()

    K = y_np.shape[-1] if y_np.ndim > 0 else 1
    x = torch.tensor(y_np, dtype=torch.float32, device=device)
    x = x.reshape(1, 1, K)

    n_eff = max(int(t_diff * n_steps), 2)
    ts = torch.linspace(float(t_diff), 0.0, n_eff + 1, device=device)

    for i in range(n_eff):
        x = diffusion._reverse_step(x, float(ts[i]), float(ts[i + 1]), device)

    return wrap_angle(x.squeeze().cpu().numpy())


# ══════════════════════════════════════════════════════════════════════════════
# §6  GG-PA adapter: TorsionDiffusionClient
# ══════════════════════════════════════════════════════════════════════════════

class TorsionDiffusionClient(ClientBase):
    """Client wrapping a trained TorsionDiffusion model for one chain.

    Given noisy dihedrals y = (φ, ψ) at noise level t_diff, produces a
    denoised estimate x₀ by running partial reverse diffusion.
    """

    def __init__(
        self,
        client_id: str,
        diffusion_model: TorsionDiffusion,
        torsion_indices: np.ndarray,
        sigma_min: float = 0.1,
        sigma_max: float = 3.0,
        n_reverse_steps: int = 50,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.projector = TorsionProjector(torsion_indices)
        self.forward_process = WrappedGaussianForwardProcess(
            sigma_min=sigma_min, sigma_max=sigma_max,
        )
        self._diffusion = diffusion_model
        self._n_reverse_steps = n_reverse_steps
        self._device = device
        self._current_x = None

    def handle_request(self, request):
        """Override to handle 'properties' requests without projecting s."""
        from ggpa.core.protocol import ClientReply
        types = request.request_types
        if isinstance(types, str):
            types = [types]
        if types == ["properties"]:
            props = self.get_properties()
            return ClientReply(
                client_id=self.client_id,
                request_id=getattr(request, "request_id", None),
                status_code="success",
                data={"properties": props},
            )
        return super().handle_request(request)

    def denoise_sample(self, y: np.ndarray, t_diff: float,
                       seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
        if t_diff < 1e-6:
            return np.asarray(y, dtype=np.float64)
        return reverse_from_t_diff(
            self._diffusion, np.asarray(y), t_diff,
            n_steps=self._n_reverse_steps,
            device=self._device,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §7  GG-PA adapter: OpenMMMDAggregator
# ══════════════════════════════════════════════════════════════════════════════

class OpenMMMDAggregator(AggregationBase):
    """GG-PA aggregator: ONE OpenMM simulation per dimer system.

    Force-field modifications:
      (a) Segment-based nonbonded removal (intra-segment LJ+Coulomb zeroed).
      (b) Bonded force scaling (PeriodicTorsionForce zeroed for torsion atoms).
      (c) Wrapped-Gaussian CustomTorsionForce restraints.
      (d) Optional flat-bottom centering force between chain centroids.

    Velocity management:
      First call initialises velocities from Maxwell-Boltzmann; subsequent
      calls preserve velocities.
    """

    def __init__(
        self,
        pdb_path: str,
        forcefield_files: List[str],
        torsion_indices: np.ndarray,
        chain_atom_lists: List[List[int]],
        sigma_min: float = 0.1,
        sigma_max: float = 3.0,
        temperature: float = 300.0,
        friction: float = 1.0,
        timestep: float = 0.002,
        md_steps: int = 500,
        kappa: float = 1.0,
        k_max: int = 1,
        platform_name: str = "CUDA",
        minimize_before_md: bool = False,
        minimize_max_iter: int = 100,
        nonbonded_mode: str = "none",
        internal_strength_scaling: Optional[Dict[str, float]] = None,
        centering_force_k: float = 100.0,
        centering_d0: Optional[float] = None,
        centering_warmup_steps: int = 50,
        centering_schedule: str = "warmup",
        client_order: Optional[List[str]] = None,
    ):
        self.pdb_path = pdb_path
        self.forcefield_files = forcefield_files
        self.torsion_indices = np.asarray(torsion_indices, dtype=int)
        self.chain_atom_lists = chain_atom_lists
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._log_ratio = np.log(sigma_max / sigma_min)
        self.temperature = temperature
        self.friction = friction
        self.timestep = timestep
        self.md_steps = md_steps
        self.kappa = kappa
        self.k_max = k_max
        self.platform_name = platform_name
        self.minimize_before_md = minimize_before_md
        self.minimize_max_iter = minimize_max_iter
        self.nonbonded_mode = nonbonded_mode
        self.internal_strength_scaling = (
            internal_strength_scaling if internal_strength_scaling is not None
            else {"dihedral": 0.0}
        )
        self.centering_force_k = centering_force_k
        self.centering_d0 = centering_d0
        self.centering_warmup_steps = centering_warmup_steps
        self.centering_schedule = str(centering_schedule).lower()
        if self.centering_schedule not in {"off", "warmup", "always"}:
            raise ValueError(
                "centering_schedule must be one of {'off', 'warmup', 'always'}"
            )
        self.client_order = list(client_order) if client_order is not None else None

        self._torsion_segments, self._atom_segments = self._derive_segments()
        self._n_torsions = self.torsion_indices.shape[0]

        self._simulation = None
        self._topology = None
        self._restraint_force = None
        self._centering_force = None
        self._aggregate_count = 0
        self._velocity_initialized = False

    def sigma_for_t_diff(self, t_diff: float) -> float:
        return self.sigma_min * np.exp(self._log_ratio * t_diff)

    def _derive_segments(self):
        chain_sets = [set(atoms) for atoms in self.chain_atom_lists]
        seg_dict: Dict[int, list] = {}

        for row in self.torsion_indices:
            atoms = set(int(a) for a in row)
            for ci, cs in enumerate(chain_sets):
                if atoms.issubset(cs):
                    seg_dict.setdefault(ci, []).append(row)
                    break
            else:
                seg_dict.setdefault(-1, []).append(row)

        torsion_segments = []
        atom_segments = []
        for ci in sorted(seg_dict.keys()):
            rows = np.array(seg_dict[ci], dtype=int)
            torsion_segments.append(rows)
            unique_atoms = sorted(set(int(a) for row in rows for a in row))
            atom_segments.append(unique_atoms)

        return torsion_segments, atom_segments

    def _ensure_openmm(self):
        if self._simulation is not None:
            return

        import openmm
        import openmm.app as app
        import openmm.unit as unit

        pdb = app.PDBFile(self.pdb_path)
        ff = app.ForceField(*self.forcefield_files)
        system = ff.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
        )

        # (a) Segment-based nonbonded modifications
        if self.nonbonded_mode != "full":
            self._modify_nonbonded_segments(system)

        # (b) Bonded-force scaling
        if self.internal_strength_scaling:
            self._scale_bonded_forces(system)

        # (c) Wrapped-Gaussian torsion restraint
        energy_terms = []
        for k in range(-self.k_max, self.k_max + 1):
            if k == 0:
                energy_terms.append("exp(-0.5*((theta-theta0)/sigma)^2)")
            else:
                shift = 2 * k
                energy_terms.append(
                    f"exp(-0.5*((theta-theta0+{shift}*pi)/sigma)^2)"
                )
        sum_expr = " + ".join(energy_terms)
        energy_expr = f"-kappa*log(max({sum_expr}, 1e-10))"

        restraint = openmm.CustomTorsionForce(energy_expr)
        restraint.addGlobalParameter("kappa", self.kappa)
        restraint.addGlobalParameter("sigma", self.sigma_for_t_diff(1.0))
        restraint.addGlobalParameter("pi", np.pi)
        restraint.addPerTorsionParameter("theta0")
        restraint.setForceGroup(31)

        for row in self.torsion_indices:
            restraint.addTorsion(int(row[0]), int(row[1]),
                                 int(row[2]), int(row[3]), [0.0])
        system.addForce(restraint)
        self._restraint_force = restraint

        # (d) Flat-bottom centering force
        use_centering_force = (
            self.centering_force_k > 0
            and (
                self.centering_schedule == "always"
                or (
                    self.centering_schedule == "warmup"
                    and self.centering_warmup_steps > 0
                )
            )
        )
        if use_centering_force:
            centroid_force = openmm.CustomCentroidBondForce(
                2,
                "0.5*centering_k*step(d-d0)*(d-d0)^2; d=distance(g1,g2)",
            )
            centroid_force.addGlobalParameter("centering_k", self.centering_force_k)

            if self.centering_d0 is None:
                pos = np.array(pdb.positions.value_in_unit(unit.nanometer))
                com_A = pos[self.chain_atom_lists[0]].mean(axis=0)
                com_B = pos[self.chain_atom_lists[1]].mean(axis=0)
                d_init = float(np.linalg.norm(com_A - com_B))
                d0_val = d_init + 0.5
            else:
                d0_val = self.centering_d0

            centroid_force.addGlobalParameter("d0", d0_val)
            centroid_force.setForceGroup(30)

            g1 = centroid_force.addGroup(self.chain_atom_lists[0])
            g2 = centroid_force.addGroup(self.chain_atom_lists[1])
            centroid_force.addBond([g1, g2])
            system.addForce(centroid_force)
            self._centering_force = centroid_force

        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature * unit.kelvin,
            self.friction / unit.picosecond,
            self.timestep * unit.picosecond,
        )

        try:
            platform = openmm.Platform.getPlatformByName(self.platform_name)
        except Exception:
            warnings.warn(f"Platform '{self.platform_name}' not available, "
                          f"falling back to CPU.")
            platform = openmm.Platform.getPlatformByName("CPU")

        self._simulation = app.Simulation(
            pdb.topology, system, integrator, platform,
        )
        self._topology = pdb.topology
        self._simulation.context.setPositions(pdb.positions)
        self._simulation.minimizeEnergy(maxIterations=1_000_000)

    def _modify_nonbonded_segments(self, system):
        import openmm

        nb_force = None
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                nb_force = force
                break
        if nb_force is None:
            return

        mode = self.nonbonded_mode

        for seg_atoms in self._atom_segments:
            if not seg_atoms:
                continue
            indices = list(seg_atoms)

            if mode == "none":
                for ii in range(len(indices)):
                    for jj in range(ii + 1, len(indices)):
                        p1, p2 = indices[ii], indices[jj]
                        nb_force.addException(p1, p2, 0.0, 1.0, 0.0,
                                              replace=True)

            elif mode == "standard":
                import openmm.unit as unit
                atom_params = {
                    idx: nb_force.getParticleParameters(idx)
                    for idx in indices
                }
                for ii in range(len(indices)):
                    for jj in range(ii + 1, len(indices)):
                        p1, p2 = indices[ii], indices[jj]
                        q1, s1, e1 = atom_params[p1]
                        q2, s2, e2 = atom_params[p2]
                        sig = 0.5 * (s1 + s2)
                        eps = (e1 * e2) ** 0.5
                        nb_force.addException(p1, p2, 0.0, sig, eps,
                                              replace=True)

    def _scale_bonded_forces(self, system):
        import openmm

        all_torsion_atoms = set()
        for seg in self._atom_segments:
            all_torsion_atoms.update(seg)

        strength_map = self.internal_strength_scaling

        for force in system.getForces():
            if isinstance(force, openmm.HarmonicBondForce) and "bond" in strength_map:
                scale = strength_map["bond"]
                for i in range(force.getNumBonds()):
                    p1, p2, r0, k = force.getBondParameters(i)
                    if p1 in all_torsion_atoms and p2 in all_torsion_atoms:
                        force.setBondParameters(i, p1, p2, r0, k * scale)

            elif isinstance(force, openmm.HarmonicAngleForce) and "angle" in strength_map:
                scale = strength_map["angle"]
                for i in range(force.getNumAngles()):
                    p1, p2, p3, theta0, k = force.getAngleParameters(i)
                    if {p1, p2, p3}.issubset(all_torsion_atoms):
                        force.setAngleParameters(
                            i, p1, p2, p3, theta0, k * scale,
                        )

            elif isinstance(force, openmm.PeriodicTorsionForce) and "dihedral" in strength_map:
                scale = strength_map["dihedral"]
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, k = (
                        force.getTorsionParameters(i)
                    )
                    if {p1, p2, p3, p4}.issubset(all_torsion_atoms):
                        force.setTorsionParameters(
                            i, p1, p2, p3, p4,
                            periodicity, phase, k * scale,
                        )

    def _set_positions(self, positions_nm: np.ndarray):
        import openmm.unit as unit
        self._simulation.context.setPositions(
            positions_nm * unit.nanometer,
        )

    def _get_positions(self) -> np.ndarray:
        import openmm.unit as unit
        state = self._simulation.context.getState(getPositions=True)
        return state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    def _update_restraint_anchors(self, angles: np.ndarray):
        for i in range(self._n_torsions):
            p1, p2, p3, p4 = [int(x) for x in self.torsion_indices[i]]
            self._restraint_force.setTorsionParameters(
                i, p1, p2, p3, p4, [float(angles[i])],
            )
        self._restraint_force.updateParametersInContext(
            self._simulation.context,
        )

    def _update_restraint_sigma(self, sigma: float):
        self._simulation.context.setParameter("sigma", sigma)

    def aggregate(
        self,
        s_current: np.ndarray,
        t_diff: float,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Aggregate step: MD simulation with diffusion-guided torsion restraints."""
        import openmm.unit as unit
        self._ensure_openmm()

        server = kwargs["server"]
        transport = kwargs["transport"]
        xs_override = kwargs.get("xs_override", None)
        on_the_fly_minimize = kwargs.get("on_the_fly_minimize", None)
        on_the_fly_reset_velocity = bool(
            kwargs.get("on_the_fly_reset_velocity", False)
        )

        # 1) Fetch denoised angles from every client
        if xs_override is None:
            xs = self.fetch_samples(s_current, t_diff, server, transport)
        else:
            xs = xs_override

        # 2) Build anchor array in the requested client order
        if self.client_order is None:
            client_ids = sorted(xs.keys())
        else:
            missing = [cid for cid in self.client_order if cid not in xs]
            extra = [cid for cid in xs if cid not in self.client_order]
            if missing or extra:
                raise RuntimeError(
                    "Client mismatch in OpenMMMDAggregator.aggregate: "
                    f"missing={missing}, extra={extra}"
                )
            client_ids = list(self.client_order)

        anchors = np.concatenate([np.asarray(xs[cid], dtype=np.float64) for cid in client_ids])
        if anchors.shape[0] != self._n_torsions:
            raise RuntimeError(
                f"Expected {self._n_torsions} torsion anchors, got {anchors.shape[0]}"
            )

        # 3) Update restraint anchors & strength
        self._update_restraint_anchors(anchors)
        sigma = self.sigma_for_t_diff(t_diff)
        self._update_restraint_sigma(sigma)

        # 4) Set positions
        self._set_positions(np.asarray(s_current, dtype=np.float64))

        # 5) Velocity management
        if not self._velocity_initialized:
            self._simulation.context.setVelocitiesToTemperature(
                self.temperature * unit.kelvin,
            )
            self._velocity_initialized = True

        # 6) Optional minimisation
        do_minimize = (
            self.minimize_before_md
            if on_the_fly_minimize is None
            else bool(on_the_fly_minimize)
        )
        if do_minimize:
            self._simulation.minimizeEnergy(maxIterations=self.minimize_max_iter)
            if on_the_fly_reset_velocity:
                self._simulation.context.setVelocitiesToTemperature(
                    self.temperature * unit.kelvin,
                )

        # 7) Run MD
        self._simulation.step(self.md_steps)

        # 8) Extract new positions
        s_new = self._get_positions()

        # 9) Manage centering force warmup
        self._aggregate_count += 1
        centering_active = False
        if self._centering_force is not None:
            if self.centering_schedule == "always":
                self._simulation.context.setParameter(
                    "centering_k", self.centering_force_k,
                )
                centering_active = True
            elif self._aggregate_count >= self.centering_warmup_steps:
                self._simulation.context.setParameter("centering_k", 0.0)
            else:
                self._simulation.context.setParameter(
                    "centering_k", self.centering_force_k,
                )
                centering_active = True

        diag = {
            "t_diff": t_diff,
            "sigma": sigma,
            "md_steps": self.md_steps,
            "anchor_angles_deg": np.degrees(anchors).tolist(),
            "client_ids": client_ids,
            "centering_active": centering_active,
            "centering_schedule": self.centering_schedule,
            "aggregate_count": self._aggregate_count,
        }
        return s_new, diag

    def add_dcd_reporter(self, dcd_path: str,
                         report_interval: Optional[int] = None):
        """Attach an OpenMM DCDReporter to this aggregator's simulation."""
        from openmm.app import DCDReporter
        self._ensure_openmm()
        if report_interval is None:
            report_interval = self.md_steps
        self._simulation.reporters.append(
            DCDReporter(dcd_path, report_interval)
        )

    def write_current_pdb(self, pdb_path: str):
        """Write the current OpenMM state to a PDB file."""
        from openmm.app import PDBFile

        self._ensure_openmm()
        state = self._simulation.context.getState(getPositions=True)
        with open(pdb_path, "w", encoding="utf-8") as handle:
            PDBFile.writeFile(self._topology, state.getPositions(), handle)


# ══════════════════════════════════════════════════════════════════════════════
# §8  Ion aggregator and pipeline builders
# ══════════════════════════════════════════════════════════════════════════════

class OpenMMIonAggregator(OpenMMMDAggregator):
    """Single-monomer alanine-dipeptide + Na+ OpenMM aggregator."""

    def __init__(
        self,
        pdb_path: str,
        forcefield_files: List[str],
        torsion_indices: np.ndarray,
        sigma_min: float = 0.1,
        sigma_max: float = 3.0,
        temperature: float = 300.0,
        friction: float = 1.0,
        timestep: float = 0.001,
        md_steps: int = 100,
        kappa: float = 1.0,
        k_max: int = 1,
        platform_name: str = "CUDA",
        minimize_before_md: bool = False,
        minimize_max_iter: int = 100,
        nonbonded_mode: str = "none",
        internal_strength_scaling: Optional[Dict[str, float]] = None,
        ion_element_symbol: str = "Na",
        ion_resname: str = "NA",
        ion_atom_name: str = "Na+",
        ion_offset_nm: float = 0.5,
        leash_r0_nm: float = 0.4,
        leash_k_kj_mol_nm2: float = 50.0,
    ):
        torsion_indices = np.asarray(torsion_indices, dtype=int)
        torsion_atom_list = sorted(set(int(a) for row in torsion_indices for a in row))
        super().__init__(
            pdb_path=pdb_path,
            forcefield_files=forcefield_files,
            torsion_indices=torsion_indices,
            chain_atom_lists=[torsion_atom_list],
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            temperature=temperature,
            friction=friction,
            timestep=timestep,
            md_steps=md_steps,
            kappa=kappa,
            k_max=k_max,
            platform_name=platform_name,
            minimize_before_md=minimize_before_md,
            minimize_max_iter=minimize_max_iter,
            nonbonded_mode=nonbonded_mode,
            internal_strength_scaling=internal_strength_scaling,
            centering_force_k=0.0,
            centering_d0=None,
            centering_warmup_steps=0,
            client_order=["monomer"],
        )
        self.ion_element_symbol = ion_element_symbol
        self.ion_resname = ion_resname
        self.ion_atom_name = ion_atom_name
        self.ion_offset_nm = ion_offset_nm
        self.leash_r0_nm = leash_r0_nm
        self.leash_k_kj_mol_nm2 = leash_k_kj_mol_nm2

    def _add_leash_force(self, topology, system):
        import openmm
        import openmm.unit as unit

        try:
            ca_index = next(atom.index for atom in topology.atoms() if atom.name == "CA")
        except StopIteration as exc:
            raise ValueError("Could not find a CA atom in the monomer topology.") from exc

        try:
            ion_index = next(atom.index for atom in topology.atoms() if atom.name == self.ion_atom_name)
        except StopIteration as exc:
            raise ValueError(f"Could not find ion atom '{self.ion_atom_name}'.") from exc

        bond_force = openmm.HarmonicBondForce()
        bond_force.setName("IonLeashBond")
        bond_force.addBond(
            ca_index,
            ion_index,
            self.leash_r0_nm * unit.nanometer,
            self.leash_k_kj_mol_nm2 * unit.kilojoules_per_mole / unit.nanometer**2,
        )
        system.addForce(bond_force)

    def _ensure_openmm(self):
        if self._simulation is not None:
            return

        import openmm
        import openmm.app as app
        import openmm.unit as unit

        pdb = app.PDBFile(self.pdb_path)
        modeller = app.Modeller(pdb.topology, pdb.positions)

        positions_nm = np.array(modeller.positions.value_in_unit(unit.nanometer))
        center = positions_nm.mean(axis=0)

        ion_top = app.Topology()
        ion_chain = ion_top.addChain()
        ion_residue = ion_top.addResidue(self.ion_resname, ion_chain)
        ion_element = app.element.Element.getBySymbol(self.ion_element_symbol)
        ion_top.addAtom(self.ion_atom_name, ion_element, ion_residue)

        ion_position = [
            openmm.Vec3(
                float(center[0] + self.ion_offset_nm),
                float(center[1]),
                float(center[2]),
            ) * unit.nanometer
        ]
        modeller.add(ion_top, ion_position)

        ff = app.ForceField(*self.forcefield_files)
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            rigidWater=False,
        )
        system.addForce(openmm.CMMotionRemover(100))
        self._add_leash_force(modeller.topology, system)

        if self.nonbonded_mode != "full":
            self._modify_nonbonded_segments(system)
        if self.internal_strength_scaling:
            self._scale_bonded_forces(system)

        energy_terms = []
        for k in range(-self.k_max, self.k_max + 1):
            if k == 0:
                energy_terms.append("exp(-0.5*((theta-theta0)/sigma)^2)")
            else:
                shift = 2 * k
                energy_terms.append(
                    f"exp(-0.5*((theta-theta0+{shift}*pi)/sigma)^2)"
                )
        sum_expr = " + ".join(energy_terms)
        energy_expr = f"-kappa*log(max({sum_expr}, 1e-10))"

        restraint = openmm.CustomTorsionForce(energy_expr)
        restraint.addGlobalParameter("kappa", self.kappa)
        restraint.addGlobalParameter("sigma", self.sigma_for_t_diff(1.0))
        restraint.addGlobalParameter("pi", np.pi)
        restraint.addPerTorsionParameter("theta0")
        restraint.setForceGroup(31)

        for row in self.torsion_indices:
            restraint.addTorsion(int(row[0]), int(row[1]), int(row[2]), int(row[3]), [0.0])
        system.addForce(restraint)
        self._restraint_force = restraint

        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature * unit.kelvin,
            self.friction / unit.picosecond,
            self.timestep * unit.picosecond,
        )

        try:
            platform = openmm.Platform.getPlatformByName(self.platform_name)
        except Exception:
            warnings.warn(
                f"Platform '{self.platform_name}' not available, falling back to CPU."
            )
            platform = openmm.Platform.getPlatformByName("CPU")

        self._simulation = app.Simulation(
            modeller.topology, system, integrator, platform,
        )
        self._topology = modeller.topology
        self._simulation.context.setPositions(modeller.positions)
        self._simulation.minimizeEnergy(maxIterations=1_000_000)


def build_monomer_sodium_pipeline(
    pdb_path: str,
    checkpoint_path: str,
    forcefield_files: Optional[List[str]] = None,
    temperature: float = 300.0,
    friction: float = 1.0,
    md_steps: int = 100,
    platform_name: str = "CUDA",
    device: str = "cpu",
    kappa: float = 1.0,
    k_max: int = 1,
    timestep: float = 0.001,
    n_reverse_steps: int = 50,
    minimize_before_md: bool = False,
    nonbonded_mode: str = "none",
    internal_strength_scaling: Optional[Dict[str, float]] = None,
    ion_element_symbol: str = "Na",
    ion_resname: str = "NA",
    ion_atom_name: str = "Na+",
    ion_offset_nm: float = 0.5,
    leash_r0_nm: float = 0.4,
    leash_k_kj_mol_nm2: float = 50.0,
    master_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble GG-PA components for alanine dipeptide + Na+."""
    from ggpa.core.kernel import FixedDiffusionTimeKernel
    from ggpa.core.state import State
    from ggpa.server.context import UniformContext

    if forcefield_files is None:
        forcefield_files = ["amber99sbildn.xml", "tip3p.xml"]

    torsion_info = extract_monomer_torsion_indices(pdb_path)

    diffusion = TorsionDiffusion.load_from_checkpoint(checkpoint_path, device=device)
    sigma_min = diffusion.forward_process.sigma_min
    sigma_max = diffusion.forward_process.sigma_max

    clients = {
        "monomer": TorsionDiffusionClient(
            client_id="monomer",
            diffusion_model=diffusion,
            torsion_indices=torsion_info["all"],
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            n_reverse_steps=n_reverse_steps,
            device=device,
        ),
    }

    aggregator = OpenMMIonAggregator(
        pdb_path=pdb_path,
        forcefield_files=forcefield_files,
        torsion_indices=torsion_info["all"],
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        temperature=temperature,
        friction=friction,
        timestep=timestep,
        md_steps=md_steps,
        kappa=kappa,
        k_max=k_max,
        platform_name=platform_name,
        minimize_before_md=minimize_before_md,
        nonbonded_mode=nonbonded_mode,
        internal_strength_scaling=internal_strength_scaling,
        ion_element_symbol=ion_element_symbol,
        ion_resname=ion_resname,
        ion_atom_name=ion_atom_name,
        ion_offset_nm=ion_offset_nm,
        leash_r0_nm=leash_r0_nm,
        leash_k_kj_mol_nm2=leash_k_kj_mol_nm2,
    )

    context = UniformContext()
    kernel = FixedDiffusionTimeKernel.from_clients(
        clients=clients,
        aggregator=aggregator,
        context=context,
        master_seed=master_seed,
    )

    aggregator._ensure_openmm()
    init_pos = np.array(aggregator._get_positions(), dtype=np.float64)
    init_state = State(s=init_pos, step=0)

    return {
        "kernel": kernel,
        "clients": clients,
        "aggregator": aggregator,
        "context": context,
        "torsion_info": torsion_info,
        "init_state": init_state,
        "diffusion": diffusion,
    }

def build_dimer_pipeline(
    pdb_path: str,
    checkpoint_path: str,
    forcefield_files: Optional[List[str]] = None,
    temperature: float = 300.0,
    friction: float = 1.0,
    md_steps: int = 500,
    platform_name: str = "CUDA",
    device: str = "cpu",
    kappa: float = 1.0,
    k_max: int = 1,
    timestep: float = 0.002,
    n_reverse_steps: int = 50,
    minimize_before_md: bool = False,
    nonbonded_mode: str = "none",
    internal_strength_scaling: Optional[Dict[str, float]] = None,
    centering_force_k: float = 100.0,
    centering_d0: Optional[float] = None,
    centering_warmup_steps: int = 50,
    centering_schedule: str = "warmup",
    master_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble all GG-PA components for one alanine dipeptide dimer system.

    Each call creates ONE independent pipeline (Aggregator + Clients + Kernel)
    with its own OpenMM simulation context.

    Returns a dictionary with:
        kernel, clients, aggregator, context, torsion_info, init_state, diffusion
    """
    from ggpa.core.kernel import FixedDiffusionTimeKernel
    from ggpa.core.state import State
    from ggpa.server.context import UniformContext

    if forcefield_files is None:
        forcefield_files = ["amber99sbildn.xml", "tip3p.xml"]

    torsion_info = extract_dimer_torsion_indices(pdb_path)

    diffusion = TorsionDiffusion.load_from_checkpoint(checkpoint_path, device=device)
    sigma_min = diffusion.forward_process.sigma_min
    sigma_max = diffusion.forward_process.sigma_max

    clients = {
        "chain_A": TorsionDiffusionClient(
            client_id="chain_A",
            diffusion_model=diffusion,
            torsion_indices=torsion_info["chain_A"]["all"],
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            n_reverse_steps=n_reverse_steps,
            device=device,
        ),
        "chain_B": TorsionDiffusionClient(
            client_id="chain_B",
            diffusion_model=diffusion,
            torsion_indices=torsion_info["chain_B"]["all"],
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            n_reverse_steps=n_reverse_steps,
            device=device,
        ),
    }

    aggregator = OpenMMMDAggregator(
        pdb_path=pdb_path,
        forcefield_files=forcefield_files,
        torsion_indices=torsion_info["all"],
        chain_atom_lists=torsion_info["chain_atom_lists"],
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        temperature=temperature,
        friction=friction,
        timestep=timestep,
        md_steps=md_steps,
        kappa=kappa,
        k_max=k_max,
        platform_name=platform_name,
        minimize_before_md=minimize_before_md,
        nonbonded_mode=nonbonded_mode,
        internal_strength_scaling=internal_strength_scaling,
        centering_force_k=centering_force_k,
        centering_d0=centering_d0,
        centering_warmup_steps=centering_warmup_steps,
        centering_schedule=centering_schedule,
        client_order=["chain_A", "chain_B"],
    )

    context = UniformContext()

    kernel = FixedDiffusionTimeKernel.from_clients(
        clients=clients,
        aggregator=aggregator,
        context=context,
        master_seed=master_seed,
    )

    import openmm.app as app
    import openmm.unit as unit
    pdb = app.PDBFile(pdb_path)
    init_pos = np.array(
        pdb.positions.value_in_unit(unit.nanometer), dtype=np.float64,
    )
    init_state = State(s=init_pos, step=0)

    return {
        "kernel": kernel,
        "clients": clients,
        "aggregator": aggregator,
        "context": context,
        "torsion_info": torsion_info,
        "init_state": init_state,
        "diffusion": diffusion,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §9  Replica exchange
# ══════════════════════════════════════════════════════════════════════════════

class AlanineReplicaExchange:
    """Replica-exchange driver managing independent alanine torsion pipelines.

    Each replica is a separate pipeline (own Aggregator, Clients, Kernel),
    each with its own OpenMM simulation context and persistent velocities.
    Odd/even alternating swap pairs (standard RE protocol).

    Usage::

        pipes = [build_dimer_pipeline(...) for _ in range(4)]
        re = AlanineReplicaExchange(pipes, t_diffs=[0.05, 0.1, 0.2, 0.4])
        results = re.run(n_blocks=500, inner_steps=1, init_states=...)
    """

    def __init__(
        self,
        pipes: List[Dict[str, Any]],
        t_diffs: List[float],
        rng_seed: Optional[int] = None,
    ):
        assert len(pipes) == len(t_diffs)
        self.pipes = pipes
        self.t_diffs = list(t_diffs)
        self.n_replicas = len(t_diffs)
        self.rng = np.random.default_rng(rng_seed)
        self.torsion_info = pipes[0]["torsion_info"]
        self.client_ids = sorted(list(pipes[0]["clients"].keys()))

        self._replica_xs: List[Optional[Dict[str, np.ndarray]]] = [
            None
        ] * self.n_replicas
        self.states: Optional[List] = None

    def _inject_client_xs(self, pipe_idx: int, xs: Dict[str, np.ndarray]):
        clients = self.pipes[pipe_idx]["clients"]
        for cid, x in xs.items():
            if cid in clients:
                clients[cid].current_x = np.asarray(x, dtype=np.float64).copy()

    def _concat_xs(self, xs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(xs[cid], dtype=np.float64).ravel()
                for cid in self.client_ids
            ]
        )

    def _denoise_replica(self, rep_idx: int, state, t_diff: float) -> Dict[str, np.ndarray]:
        kernel = self.pipes[rep_idx]["kernel"]
        server = kernel.server
        transport = kernel.transport

        requests = server.create_requests(
            s=state.s, t_diff=t_diff, request_types="sample", step=state.step,
        )
        replies = transport.call_all(requests.values())

        xs: Dict[str, np.ndarray] = {}
        for client_id, reply in replies.items():
            if reply.status_code in ["success", "partial"]:
                sample = reply.data.get("sample", None)
                if sample is not None:
                    xs[client_id] = np.asarray(sample, dtype=np.float64).copy()

        self._inject_client_xs(rep_idx, xs)
        self._replica_xs[rep_idx] = xs
        return xs

    def _inner_loop(self, states: List, inner_steps: int):
        from ggpa.core.state import State
        from ggpa.utils.utils import seed_for_server

        for _ in range(inner_steps):
            for i, t_diff in enumerate(self.t_diffs):
                state_i = states[i]
                xs = self._denoise_replica(i, state_i, t_diff)

                aggregator = self.pipes[i]["aggregator"]
                kernel = self.pipes[i]["kernel"]

                s_new, _ = aggregator.aggregate(
                    s_current=state_i.s,
                    t_diff=t_diff,
                    server=kernel.server,
                    transport=kernel.transport,
                    context=kernel.server.context,
                    seed=seed_for_server(kernel.master_seed, state_i.step),
                    step=state_i.step,
                    xs_override=xs,
                )

                states[i] = State(
                    s=s_new,
                    step=state_i.step + 1,
                    cache=dict(state_i.cache),
                )

        return states

    def _swap_openmm_states(self, i: int, j: int):
        sim_i = self.pipes[i]["aggregator"]._simulation
        sim_j = self.pipes[j]["aggregator"]._simulation

        s_i = sim_i.context.getState(getPositions=True, getVelocities=True)
        s_j = sim_j.context.getState(getPositions=True, getVelocities=True)

        sim_i.context.setState(s_j)
        sim_j.context.setState(s_i)

    def _attempt_swaps(self, states: List, block_idx: int) -> List[Dict]:
        swap_info = []
        start = 1 if (block_idx % 2 == 1) else 0
        for idx in range(start, self.n_replicas - 1, 2):
            i, j = idx, idx + 1
            t_diff_i, t_diff_j = self.t_diffs[i], self.t_diffs[j]
            state_i, state_j = states[i], states[j]

            self._inject_client_xs(i, self._replica_xs[i])
            kernel_i = self.pipes[i]["kernel"]
            u_i_i = float(kernel_i.reduced_potential(state_i, t_diff_i).u_t_diff)
            u_j_i = float(kernel_i.reduced_potential(state_i, t_diff_j).u_t_diff)

            self._inject_client_xs(j, self._replica_xs[j])
            kernel_j = self.pipes[j]["kernel"]
            u_j_j = float(kernel_j.reduced_potential(state_j, t_diff_j).u_t_diff)
            u_i_j = float(kernel_j.reduced_potential(state_j, t_diff_i).u_t_diff)

            log_alpha = -(u_i_j + u_j_i) + (u_i_i + u_j_j)
            accept = log_alpha >= 0.0
            if not accept:
                accept = bool(np.log(self.rng.uniform()) < log_alpha)

            if accept:
                states[i], states[j] = states[j], states[i]
                self._replica_xs[i], self._replica_xs[j] = (
                    self._replica_xs[j], self._replica_xs[i],
                )
                self._swap_openmm_states(i, j)

            swap_info.append(
                {"pair": (i, j), "accepted": accept, "log_alpha": log_alpha}
            )

        return swap_info

    def run(
        self,
        n_blocks: int = 500,
        inner_steps: int = 1,
        record_interval: int = 1,
        init_states: Optional[List] = None,
        save_positions: bool = False,
        print_every: int = 50,
    ) -> Dict[str, Any]:
        """Run replica exchange and return trajectory data.

        Args:
            n_blocks:        Number of RE blocks (sweeps).
            inner_steps:     Number of local denoise→aggregate updates per sweep.
            record_interval: Record dihedrals every *record_interval* blocks.
            init_states:     Starting states (one per replica).
            save_positions:  If True, store full-atom positions.
            print_every:     Print progress every N blocks (0 = silent).

        Returns:
            Dictionary with dihedrals, x_dihedrals, positions, swap_log,
            t_diffs, acceptance_rates, wall_time_s, states.
        """
        import time as _time

        if init_states is None:
            raise ValueError("init_states must be provided")
        self.states = init_states
        self._replica_xs = [None] * self.n_replicas

        for i in range(self.n_replicas):
            self.pipes[i]["aggregator"]._ensure_openmm()

        dihedrals = {i: [] for i in range(self.n_replicas)}
        dihedrals_rad = {i: [] for i in range(self.n_replicas)}
        x_dihedrals = {i: [] for i in range(self.n_replicas)}
        x_dihedrals_rad = {i: [] for i in range(self.n_replicas)}
        positions = {i: [] for i in range(self.n_replicas)} if save_positions else None
        swap_log: List[List[Dict]] = []

        for rep, t_diff in enumerate(self.t_diffs):
            self._denoise_replica(rep, self.states[rep], t_diff)

        t0 = _time.time()
        for block in range(n_blocks):
            swaps = self._attempt_swaps(self.states, block)
            swap_log.append(swaps)

            self.states = self._inner_loop(self.states, inner_steps)

            if (block + 1) % record_interval == 0:
                for rep in range(self.n_replicas):
                    s_angles = compute_dihedrals(
                        self.states[rep].s, self.torsion_info["all"]
                    )
                    dihedrals_rad[rep].append(s_angles)
                    dihedrals[rep].append(np.degrees(s_angles))

                    xs_rep = self._replica_xs[rep]
                    if xs_rep is None:
                        xs_rep = self._denoise_replica(
                            rep, self.states[rep], self.t_diffs[rep]
                        )
                    x_concat = self._concat_xs(xs_rep)
                    x_dihedrals_rad[rep].append(x_concat)
                    x_dihedrals[rep].append(np.degrees(x_concat))

                    if save_positions:
                        positions[rep].append(np.copy(self.states[rep].s))

            if print_every > 0 and (block + 1) % print_every == 0:
                elapsed = _time.time() - t0
                rate = elapsed / (block + 1)
                eta = rate * (n_blocks - block - 1)
                acc_str = self._format_acceptance(swap_log)
                print(
                    f"  Block {block+1}/{n_blocks}  "
                    f"({elapsed:.0f}s, ETA {eta:.0f}s)  "
                    f"acc=[{acc_str}]"
                )

        wall_time = _time.time() - t0

        for rep in range(self.n_replicas):
            dihedrals[rep] = np.array(dihedrals[rep])
            dihedrals_rad[rep] = np.array(dihedrals_rad[rep])
            x_dihedrals[rep] = np.array(x_dihedrals[rep])
            x_dihedrals_rad[rep] = np.array(x_dihedrals_rad[rep])
            if save_positions:
                positions[rep] = np.array(positions[rep])

        acceptance = self._compute_acceptance(swap_log)

        return {
            "dihedrals": dihedrals,
            "dihedrals_rad": dihedrals_rad,
            "x_dihedrals": x_dihedrals,
            "x_dihedrals_rad": x_dihedrals_rad,
            "positions": positions,
            "swap_log": swap_log,
            "t_diffs": np.array(self.t_diffs),
            "acceptance_rates": acceptance,
            "wall_time_s": wall_time,
            "states": self.states,
        }

    def _compute_acceptance(self, swap_log):
        counts: Dict[Tuple, List[bool]] = {}
        for block_swaps in swap_log:
            for s in block_swaps:
                p = tuple(s["pair"])
                counts.setdefault(p, []).append(s["accepted"])
        return {p: np.mean(v) for p, v in counts.items()}

    def _format_acceptance(self, swap_log):
        acc = self._compute_acceptance(swap_log)
        parts = [f"({p[0]},{p[1]}):{r:.1%}" for p, r in sorted(acc.items())]
        return "  ".join(parts) if parts else "N/A"


# Backward-compatible alias for older notebooks/scripts.
DimerReplicaExchange = AlanineReplicaExchange


# ══════════════════════════════════════════════════════════════════════════════
# §10  Analysis utilities
# ══════════════════════════════════════════════════════════════════════════════

def _orientation_vector_single(positions: np.ndarray,
                               atom_indices: List[int],
                               pdb_path: str) -> np.ndarray:
    """Orientation vector (first C → last N) for one chain from raw positions.

    Args:
        positions: (N_atoms, 3) full dimer positions.
        atom_indices: Atom indices belonging to this chain.
        pdb_path: PDB file for topology lookup.

    Returns:
        (3,) orientation vector.
    """
    import mdtraj as md
    top = md.load_topology(pdb_path)

    sub_top = top.subset(atom_indices)
    c_atoms = sub_top.select("name C")
    n_atoms = sub_top.select("name N")

    if len(c_atoms) > 0 and len(n_atoms) > 0:
        # Map sub-topology indices back to full topology
        idx_start = atom_indices[c_atoms[0]]
        idx_end = atom_indices[n_atoms[-1]]
    else:
        idx_start = atom_indices[0]
        idx_end = atom_indices[-1]

    return positions[idx_end] - positions[idx_start]


def compute_cosine_similarity(positions: np.ndarray,
                              chain_atom_lists: List[List[int]],
                              pdb_path: Optional[str] = None) -> float:
    """Cosine similarity between chain backbone orientation vectors.

    Orientation vector: first "name C" → last "name N" in each chain.
    +1 = parallel, −1 = antiparallel.

    Args:
        positions:       (N_atoms, 3) atom coordinates.
        chain_atom_lists: [[chain_A atoms], [chain_B atoms]].
        pdb_path:        PDB file path (needed for topology).  If None, falls
                         back to first/last atom heuristic.
    """
    if pdb_path is not None:
        v0 = _orientation_vector_single(positions, chain_atom_lists[0], pdb_path)
        v1 = _orientation_vector_single(positions, chain_atom_lists[1], pdb_path)
    else:
        c0 = chain_atom_lists[0]
        c1 = chain_atom_lists[1]
        v0 = positions[c0[-1]] - positions[c0[0]]
        v1 = positions[c1[-1]] - positions[c1[0]]
    dot = np.dot(v0, v1)
    return float(dot / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-12))


def compute_com_distance(positions: np.ndarray,
                         chain_atom_lists: List[List[int]]) -> float:
    """Center-of-mass distance between chains (nm)."""
    com0 = positions[chain_atom_lists[0]].mean(axis=0)
    com1 = positions[chain_atom_lists[1]].mean(axis=0)
    return float(np.linalg.norm(com0 - com1))


def analyze_dimer_trajectory(
    positions_list: List[np.ndarray],
    torsion_info: Dict,
    pdb_path: str,
) -> Dict[str, Any]:
    """Compute per-frame observables from a trajectory of dimer positions.

    Args:
        positions_list: List of (N_atoms, 3) arrays in nm.
        torsion_info:   Output of :func:`extract_dimer_torsion_indices`.
        pdb_path:       PDB file path (used for topology in H-bond / cosine).

    Returns:
        Dictionary with keys:
            com_distances, cosine_similarities, hbond_counts,
            reciprocal_counts, dihedrals_seg1 (n,2), dihedrals_seg2 (n,2)
    """
    import mdtraj as md

    n_frames = len(positions_list)
    all_idx = torsion_info["all"]
    chain_atoms = torsion_info["chain_atom_lists"]
    n_atoms = torsion_info["n_atoms"]
    mid = n_atoms // 2

    # Build mdtraj trajectory for H-bond analysis
    top = md.load_topology(pdb_path)
    xyz = np.array(positions_list)  # (n_frames, n_atoms, 3)
    traj = md.Trajectory(xyz, top)
    seg1 = traj.atom_slice(range(mid))
    seg2 = traj.atom_slice(range(mid, n_atoms))

    results: Dict[str, Any] = {}

    # COM distance
    com1 = md.compute_center_of_mass(seg1)
    com2 = md.compute_center_of_mass(seg2)
    results["com_distances"] = np.linalg.norm(com1 - com2, axis=1)

    # Cosine similarity (per-frame)
    cos_sims = np.zeros(n_frames)
    for i in range(n_frames):
        cos_sims[i] = compute_cosine_similarity(
            positions_list[i], chain_atoms, pdb_path,
        )
    results["cosine_similarities"] = cos_sims

    # Dihedrals per segment → (n_frames, 2) = [φ, ψ]
    phi1 = md.compute_phi(seg1)[1]
    psi1 = md.compute_psi(seg1)[1]
    phi2 = md.compute_phi(seg2)[1]
    psi2 = md.compute_psi(seg2)[1]
    results["dihedrals_seg1"] = np.hstack([phi1, psi1])
    results["dihedrals_seg2"] = np.hstack([phi2, psi2])

    # H-bonds (inter-molecular)
    all_hb = md.baker_hubbard(traj, periodic=False)
    inter = np.array([b for b in all_hb if (b[0] < mid) != (b[2] < mid)])
    hbond_counts = np.zeros(n_frames)
    reciprocal_counts = np.zeros(n_frames)
    if len(inter) > 0:
        dists = md.compute_distances(traj, inter[:, [0, 2]])
        angles = md.compute_angles(traj, inter)
        is_hb = (dists < 0.35) & (angles > np.radians(120))
        hbond_counts = np.sum(is_hb, axis=1).astype(float)
        for f in range(n_frames):
            active = inter[is_hb[f]]
            if len(active) >= 2:
                donors = active[:, 0]
                if np.any(donors < mid) and np.any(donors >= mid):
                    reciprocal_counts[f] = 1
    results["hbond_counts"] = hbond_counts
    results["reciprocal_counts"] = reciprocal_counts

    return results


def extract_valid_strict(
    res: Dict[str, Any],
    cos_thresh: float = 0.95,
    hb_min: int = 2,
) -> Dict[str, Any]:
    """Filter frames into antiparallel / parallel using strict criteria.

    Criteria: |cos_sim| >= cos_thresh AND hbond_count >= hb_min AND reciprocal == 1.

    Returns:
        Dictionary with filtered dihedrals and counts.
    """
    cos = res["cosine_similarities"]
    hb = res["hbond_counts"]
    rec = res["reciprocal_counts"]
    d1 = res["dihedrals_seg1"]
    d2 = res["dihedrals_seg2"]

    bound = (np.abs(cos) >= cos_thresh) & (hb >= hb_min) & (rec == 1)
    anti = bound & (cos < 0)
    para = bound & (cos > 0)

    return {
        "dihedrals_antiparallel_seg1": d1[anti],
        "dihedrals_antiparallel_seg2": d2[anti],
        "dihedrals_parallel_seg1": d1[para],
        "dihedrals_parallel_seg2": d2[para],
        "n_anti": int(anti.sum()),
        "n_para": int(para.sum()),
        "n_total": len(cos),
        "mask_anti": anti,
        "mask_para": para,
    }


def wrap_abs_diff(d1: np.ndarray, d2: np.ndarray, idx: int = 0) -> np.ndarray:
    """Wrapped absolute dihedral difference, folded into [0, π].

    Args:
        d1: (N, K) dihedrals for segment 1 (radians).
        d2: (N, K) dihedrals for segment 2.
        idx: Column index (0=φ, 1=ψ).

    Returns:
        (N,) array in [0, π].
    """
    diff = np.abs(d1[:, idx] - d2[:, idx])
    return np.minimum(diff, 2 * np.pi - diff)


_DIMER_MONOMER_BASIN_MODEL = {
    "left_center": np.array([-2.4234, 2.6856], dtype=np.float64),
    "right_center": np.array([-1.4261, 1.1006], dtype=np.float64),
    "left_radius": 0.5938,
    "right_radius": 0.7878,
}


def _wrap_angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def classify_dimer_monomer_basin(dihedrals_rad: np.ndarray) -> np.ndarray:
    """Assign dimer monomer torsions to reference-driven L/R/U basins.

    The public dimer diagnostics use two coarse monomer basins derived from the
    bundled MD reference. Frames inside the left/right basin radii are labeled
    ``L``/``R`` and the remaining frames are labeled ``U``.
    """
    dihedrals_rad = np.asarray(dihedrals_rad, dtype=np.float64)
    left_center = _DIMER_MONOMER_BASIN_MODEL["left_center"]
    right_center = _DIMER_MONOMER_BASIN_MODEL["right_center"]
    left_radius = float(_DIMER_MONOMER_BASIN_MODEL["left_radius"])
    right_radius = float(_DIMER_MONOMER_BASIN_MODEL["right_radius"])

    d_left = np.sqrt(
        _wrap_angle_delta(dihedrals_rad[:, 0], left_center[0]) ** 2
        + _wrap_angle_delta(dihedrals_rad[:, 1], left_center[1]) ** 2
    )
    d_right = np.sqrt(
        _wrap_angle_delta(dihedrals_rad[:, 0], right_center[0]) ** 2
        + _wrap_angle_delta(dihedrals_rad[:, 1], right_center[1]) ** 2
    )

    basin = np.full(dihedrals_rad.shape[0], "U", dtype=object)
    left_mask = (d_left <= left_radius) & (d_left < d_right)
    right_mask = (d_right <= right_radius) & (d_right < d_left)
    basin[left_mask] = "L"
    basin[right_mask] = "R"
    return basin


def classify_states(res: Dict[str, Any]) -> np.ndarray:
    """Classify each frame into coarse dimer states using the public basin model.

    Monomers are first assigned to reference-driven ``L/R/U`` basins in
    ``(phi, psi)`` space. Bound antiparallel ``LL`` frames are labeled
    ``Anti-LL``; bound parallel ``LR``/``RL`` frames are labeled
    ``Para-LR``/``Para-RL``. Other bound structured frames become
    ``Bound-Other`` and the rest are labeled ``Transition``.
    """
    n = len(res["cosine_similarities"])
    cos = res["cosine_similarities"]
    hb = res["hbond_counts"]
    rec = res["reciprocal_counts"]
    d1 = np.asarray(res["dihedrals_seg1"], dtype=np.float64)
    d2 = np.asarray(res["dihedrals_seg2"], dtype=np.float64)

    m1 = classify_dimer_monomer_basin(d1)
    m2 = classify_dimer_monomer_basin(d2)

    bound = (np.abs(cos) >= 0.9) & (hb >= 2) & (rec == 1)
    anti = bound & (cos < -0.5)
    para = bound & (cos > 0.5)

    labels = np.full(n, "Transition", dtype=object)
    for i in range(n):
        if anti[i] and m1[i] == "L" and m2[i] == "L":
            labels[i] = "Anti-LL"
        elif para[i] and m1[i] == "L" and m2[i] == "R":
            labels[i] = "Para-LR"
        elif para[i] and m1[i] == "R" and m2[i] == "L":
            labels[i] = "Para-RL"
        elif anti[i] or para[i]:
            labels[i] = "Bound-Other"
    return labels


def autocorrelation_function(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Normalised autocorrelation function of a 1-D signal."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = n // 2
    acf = np.correlate(x, x, mode='full')
    acf = acf[n - 1:]  # keep positive lags only
    acf = acf[:max_lag + 1]
    if acf[0] != 0:
        acf /= acf[0]
    return acf


def integrated_autocorrelation_time(acf: np.ndarray) -> float:
    """Integrated autocorrelation time from ACF."""
    # Sum until first negative value
    total = 0.0
    for i in range(1, len(acf)):
        if acf[i] < 0:
            break
        total += acf[i]
    return 1.0 + 2.0 * total
