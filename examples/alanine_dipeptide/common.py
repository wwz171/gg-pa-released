#!/usr/bin/env python3
"""Shared helpers for the released alanine-dipeptide examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


def configure_unbuffered_stdio() -> None:
    """Prefer line-buffered stdout/stderr for long runs."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)


def resolve_project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    required = [root / "src", root / "configs", root / "checkpoints", root / "examples"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Could not resolve the GG-PA project root from examples/alanine_dipeptide. "
            f"Missing: {', '.join(str(p) for p in missing)}"
        )
    return root


ROOT = resolve_project_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def resolve_path(value: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def choose_device(requested: str | None) -> str:
    if requested is None:
        requested = "cuda"
    requested = str(requested)
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the alanine example config and resolve repo-relative paths."""
    path = (
        resolve_path(config_path, root=ROOT)
        if config_path is not None
        else ROOT / "configs" / "alanine_dipeptide_example.yaml"
    )
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    shared = cfg.setdefault("shared", {})
    sodium = cfg.setdefault("ad_sodium", {})
    sodium_ensemble = cfg.setdefault("ad_sodium_ensemble", {})
    dimer = cfg.setdefault("ad_dimer", {})

    shared.setdefault("checkpoint", "checkpoints/ad_torsion_prior.pt")
    shared.setdefault("output_root", "examples/alanine_dipeptide/results")
    shared.setdefault("forcefield_files", ["amber99sbildn.xml", "tip3p.xml"])
    shared.setdefault("device", "cuda")
    shared.setdefault("platform_name", "CUDA")
    shared.setdefault("master_seed", 42)
    shared.setdefault("n_reverse_steps", 50)
    shared.setdefault("kappa", 1.0)
    shared.setdefault("k_max", 1)
    shared.setdefault("nonbonded_mode", "none")
    shared.setdefault("internal_strength_scaling", {"dihedral": 0.0})

    shared["checkpoint"] = resolve_path(shared["checkpoint"], root=ROOT)
    if not shared["checkpoint"].exists():
        raise FileNotFoundError(f"Could not find torsion checkpoint: {shared['checkpoint']}")
    shared["output_root"] = resolve_path(shared["output_root"], root=ROOT)
    shared["device"] = choose_device(shared["device"])

    sodium.setdefault("pdb", "data/adp_monomer_vacuum.pdb")
    sodium.setdefault("temperature", 300.0)
    sodium.setdefault("friction", 1.0)
    sodium.setdefault("timestep", 0.001)
    sodium.setdefault("t_diff", 0.1)
    sodium.setdefault("md_steps", 100)
    sodium.setdefault("n_steps", 1000)
    sodium.setdefault("record_interval", 1)
    sodium.setdefault("print_every", 100)
    sodium.setdefault("compute_reduced_potential", False)
    sodium.setdefault("save_dcd", False)
    sodium.setdefault("dcd_interval", 100)
    sodium.setdefault("minimize_before_md", False)
    sodium.setdefault("ion_element_symbol", "Na")
    sodium.setdefault("ion_resname", "NA")
    sodium.setdefault("ion_atom_name", "Na+")
    sodium.setdefault("ion_offset_nm", 0.5)
    sodium.setdefault("leash_r0_nm", 0.4)
    sodium.setdefault("leash_k_kj_mol_nm2", 50.0)
    sodium.setdefault("output_dir", shared["output_root"] / "ad_sodium_single")
    sodium["pdb"] = resolve_path(sodium["pdb"], root=ROOT)
    sodium["output_dir"] = resolve_path(sodium["output_dir"], root=ROOT)

    sodium_ensemble.setdefault("output_dir", shared["output_root"] / "ad_sodium")
    sodium_ensemble.setdefault("n_trajectories", 5)
    sodium_ensemble.setdefault("seeds", [42, 43, 44, 45, 46])
    sodium_ensemble.setdefault("burnin_fraction", 0.2)
    sodium_ensemble.setdefault("max_plot_points", 16000)
    sodium_ensemble["output_dir"] = resolve_path(sodium_ensemble["output_dir"], root=ROOT)

    dimer.setdefault("pdb", "data/adp_dimer_vacuum.pdb")
    dimer.setdefault("temperature", 300.0)
    dimer.setdefault("friction", 1.0)
    dimer.setdefault("timestep", 0.002)
    dimer.setdefault("t_diff_list", [0.1, 0.15, 0.25, 0.4])
    dimer.setdefault("md_steps", 100)
    dimer.setdefault("n_blocks", 1000)
    dimer.setdefault("inner_steps", 1)
    dimer.setdefault("record_interval", 1)
    dimer.setdefault("print_every", 100)
    dimer.setdefault("save_positions", True)
    dimer.setdefault("save_dcd", True)
    dimer.setdefault("dcd_interval", 100)
    dimer.setdefault("minimize_before_md", False)
    dimer.setdefault("centering_force_k", 50.0)
    dimer.setdefault("centering_d0", 0.9)
    dimer.setdefault("centering_warmup_steps", 0)
    dimer.setdefault("centering_schedule", "always")
    dimer.setdefault("burnin_fraction", 0.2)
    dimer.setdefault("output_dir", shared["output_root"] / "ad_dimer")
    dimer["pdb"] = resolve_path(dimer["pdb"], root=ROOT)
    dimer["output_dir"] = resolve_path(dimer["output_dir"], root=ROOT)

    return cfg
