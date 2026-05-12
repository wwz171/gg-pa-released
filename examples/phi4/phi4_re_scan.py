#!/usr/bin/env python3
"""Reproducible Ginzburg-Landau phi^4 GG-PA replica-exchange scans.

This script reproduces two paper-facing scans:

1. `phase-transition`
   Zero-field (`h=0`) replica-exchange scan across `J`.
2. `critical-scaling`
   Nonzero-field scaling-window scans that save one `.npz` file per `(h, J)`
   point for later data-collapse analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def configure_unbuffered_stdio() -> None:
    """Prefer line-buffered stdout/stderr for long background runs."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)


def resolve_project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    required = [root / "src", root / "checkpoints", root / "configs", root / "examples"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Could not resolve the GG-PA project root from examples/phi4. "
            f"Missing: {', '.join(str(p) for p in missing)}"
        )
    return root


ROOT = resolve_project_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ggpa.models.diffusion import SimpleDiffusion
from ggpa.systems.phi4 import LatticeRERunner, integrated_autocorrelation_time


def _resolve_path(value: str | Path, *, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def _as_bool_override(value: bool | None, current: bool) -> bool:
    return current if value is None else bool(value)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the phi4 example config and resolve repo-relative paths."""
    path = (
        _resolve_path(config_path, root=ROOT)
        if config_path is not None
        else ROOT / "configs" / "phi4_example.yaml"
    )
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    runtime = cfg.setdefault("runtime", {})
    model = cfg.setdefault("model", {})
    diffusion = cfg.setdefault("diffusion", {})
    phase = cfg.setdefault("phase_transition", {})
    critical = cfg.setdefault("critical_scaling", {})
    plotting = cfg.setdefault("plotting", {})

    runtime.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    runtime.setdefault("compile_model", False)
    runtime.setdefault("quiet", False)
    runtime.setdefault("output_root", "examples/phi4/results")
    runtime["output_root"] = _resolve_path(runtime["output_root"], root=ROOT)

    model.setdefault("checkpoint", "checkpoints/phi4_prior.pt")
    model["checkpoint"] = _resolve_path(model["checkpoint"], root=ROOT)

    diffusion.setdefault("t_prod", 0.1)
    diffusion.setdefault("t_max", 0.6)
    diffusion.setdefault("n_replicas", 48)

    phase.setdefault("L", 32)
    phase.setdefault("h", 0.0)
    phase.setdefault("seed", 42)
    phase.setdefault("n_sweeps", 10_000)
    phase.setdefault("record_interval", 1)
    phase.setdefault("burnin_fraction", 0.3)
    phase.setdefault("save_all_replicas", True)
    phase.setdefault("output_dir", runtime["output_root"] / "phase_transition")
    phase["output_dir"] = _resolve_path(phase["output_dir"], root=ROOT)

    critical.setdefault("L", 32)
    critical.setdefault("seed", 42)
    critical.setdefault("n_sweeps", 10_000)
    critical.setdefault("record_interval", 1)
    critical.setdefault("burnin_fraction", 0.3)
    critical.setdefault("save_all_replicas", True)
    critical.setdefault("resume", False)
    critical.setdefault("output_dir", runtime["output_root"] / "critical_scaling")
    critical["output_dir"] = _resolve_path(critical["output_dir"], root=ROOT)

    plotting.setdefault("output_dir", runtime["output_root"] / "figures")
    plotting["output_dir"] = _resolve_path(plotting["output_dir"], root=ROOT)
    plotting.setdefault("gamma_c", float(critical.get("gamma_c", 0.436)))
    plotting.setdefault("beta", 0.125)
    plotting.setdefault("delta", 15.0)

    cfg["_config_path"] = path
    cfg["_root"] = ROOT
    return cfg


def build_phase_transition_gamma_grid(section: dict[str, Any]) -> np.ndarray:
    """Build the zero-field `J` grid from a list of linspace segments."""
    segments = section.get("gamma_grid_segments", [])
    if not segments:
        raise ValueError("phase_transition.gamma_grid_segments must be provided.")
    parts = [
        np.linspace(seg["start"], seg["stop"], int(seg["num"]), dtype=np.float64)
        for seg in segments
    ]
    return np.sort(np.unique(np.concatenate(parts)))


def build_scaling_window_gamma_grid(
    h: float,
    *,
    gamma_c: float = 0.436,
    window_half: float = 2.0,
    n_gamma: int = 10,
    gamma_min: float = 0.10,
    gamma_max: float = 0.80,
) -> np.ndarray:
    """Return the critical-scaling window `J = J_c +- c h^(8/15)` grid."""
    width = float(h) ** (8.0 / 15.0)
    gamma_lo = max(gamma_c - window_half * width, gamma_min)
    gamma_hi = min(gamma_c + window_half * width, gamma_max)
    return np.linspace(gamma_lo, gamma_hi, int(n_gamma), dtype=np.float64)


def build_t_diff_ladder(t_prod: float, t_max: float, n_replicas: int) -> np.ndarray:
    return np.geomspace(float(t_prod), float(t_max), int(n_replicas), dtype=np.float64)


def filter_records_by_h_values(
    records: list[dict[str, Any]],
    h_values: list[float] | tuple[float, ...] | None,
) -> list[dict[str, Any]]:
    if not h_values:
        return records
    allowed = {round(float(h), 12) for h in h_values}
    return [rec for rec in records if round(float(rec["h"]), 12) in allowed]


def phase_point_n_sweeps(section: dict[str, Any], J: float) -> int:
    """Return the sweep budget for a zero-field point, with critical-window overrides."""
    base = int(section["n_sweeps"])
    window = section.get("critical_window")
    if not window:
        return base
    start = float(window["start"])
    stop = float(window["stop"])
    if start <= float(J) <= stop:
        return int(window["n_sweeps"])
    return base


def compute_observables_from_history(
    magnetizations: np.ndarray,
    *,
    L: int,
    h: float,
) -> dict[str, Any]:
    """Compute observables from the production-replica magnetization history."""
    m = np.asarray(magnetizations, dtype=np.float64)
    if m.ndim != 1:
        raise ValueError("magnetizations must be a one-dimensional array")
    if len(m) == 0:
        raise ValueError("magnetizations must contain at least one sample")

    mean_m = float(np.mean(m))
    mean_abs_m = float(np.mean(np.abs(m)))
    m2 = float(np.mean(m**2))
    m4 = float(np.mean(m**4))

    zero_field = abs(float(h)) < 1e-15
    if zero_field:
        primary_series = np.abs(m)
        primary_value = mean_abs_m
        primary_name = "order_param"
        primary_err_name = "order_err"
    else:
        primary_series = m
        primary_value = mean_m
        primary_name = "magnetization"
        primary_err_name = "mag_err"

    chi = float(L**2 * (m2 - primary_value**2))
    binder = float(1.0 - m4 / (3.0 * m2**2)) if m2 > 1e-12 else 0.0
    tau_int, window, acf = integrated_autocorrelation_time(primary_series)
    n_samples = len(primary_series)
    naive_err = float(np.std(primary_series) / math.sqrt(n_samples))
    corrected_err = float(naive_err * math.sqrt(max(2.0 * tau_int, 1.0)))
    chi_err = float(L**2 * 2.0 * abs(primary_value) * corrected_err)
    n_eff = float(n_samples / max(2.0 * tau_int, 1.0))

    result = {
        "mean_m": mean_m,
        "mean_abs_m": mean_abs_m,
        "m2": m2,
        "m4": m4,
        "susceptibility": chi,
        "susceptibility_err": chi_err,
        "binder": binder,
        "tau_int": float(tau_int),
        "tau_window": int(window),
        "n_samples": int(n_samples),
        "n_eff": n_eff,
        "acf": np.asarray(acf, dtype=np.float64),
        primary_name: float(primary_value),
        primary_err_name: corrected_err,
    }
    if zero_field:
        result["magnetization"] = mean_m
        result["mag_err"] = corrected_err
    else:
        result["order_param"] = primary_value
        result["order_err"] = corrected_err
    return result


def maybe_compile_diffusion(diffusion: SimpleDiffusion, enabled: bool) -> SimpleDiffusion:
    """Optionally torch.compile the diffusion model."""
    if not enabled or not hasattr(torch, "compile"):
        return diffusion
    try:
        return torch.compile(diffusion, mode="reduce-overhead")
    except Exception as exc:  # pragma: no cover - best-effort optimization
        print(f"[warn] torch.compile failed; using eager mode instead: {exc}")
        return diffusion


def load_diffusion_model(
    checkpoint: Path,
    *,
    device: torch.device,
    compile_model: bool,
) -> tuple[SimpleDiffusion, int]:
    diffusion = SimpleDiffusion.load_from_checkpoint(str(checkpoint), device=device)
    diffusion.eval()
    diffusion = maybe_compile_diffusion(diffusion, enabled=compile_model)
    n_params = sum(p.numel() for p in diffusion.parameters())
    return diffusion, int(n_params)


def make_runner(
    diffusion: SimpleDiffusion,
    *,
    J: float,
    h: float,
    L: int,
    t_diff_ladder: np.ndarray,
    device: torch.device,
) -> LatticeRERunner:
    return LatticeRERunner(
        diffusion_model=diffusion,
        J=float(J),
        L=int(L),
        t_diff_ladder=t_diff_ladder,
        noise_scheduler=diffusion.noise_scheduler,
        h=float(h),
        device=device,
        init_mode="uniform",
    )


def run_single_point(
    diffusion: SimpleDiffusion,
    *,
    J: float,
    h: float,
    L: int,
    t_diff_ladder: np.ndarray,
    n_sweeps: int,
    record_interval: int,
    burnin_fraction: float,
    seed: int,
    device: torch.device,
    verbose: bool,
    log_every: int | None,
    save_all_replicas: bool,
) -> dict[str, Any]:
    runner = make_runner(
        diffusion,
        J=J,
        h=h,
        L=L,
        t_diff_ladder=t_diff_ladder,
        device=device,
    )
    result = runner.run_sweeps(
        n_sweeps=int(n_sweeps),
        record_interval=int(record_interval),
        seed=int(seed),
        verbose=verbose,
        log_every=log_every,
        record_all_replicas=save_all_replicas,
    )

    if save_all_replicas:
        mag_all = np.asarray(result["magnetizations"], dtype=np.float64)
    else:
        mag_all = np.asarray(result["mags"], dtype=np.float64)[:, np.newaxis]

    prod_series = mag_all[:, 0]
    burn_idx = int(len(prod_series) * burnin_fraction)
    prod_meas = prod_series[burn_idx:]
    obs = compute_observables_from_history(prod_meas, L=L, h=h)

    swap_attempts_by_pair = np.asarray(result["swap_attempts_by_pair"], dtype=np.int64)
    swap_accepted_by_pair = np.asarray(result["swap_accepted_by_pair"], dtype=np.int64)
    pair_rates = swap_accepted_by_pair / np.maximum(swap_attempts_by_pair, 1)
    positive_rates = pair_rates[swap_attempts_by_pair > 0]
    mean_swap_rate = float(np.mean(positive_rates)) if len(positive_rates) else 0.0

    return {
        "J": float(J),
        "h": float(h),
        "L": int(L),
        "n_replicas": int(len(t_diff_ladder)),
        "n_sweeps": int(n_sweeps),
        "record_interval": int(record_interval),
        "burnin_fraction": float(burnin_fraction),
        "n_records": int(len(prod_series)),
        "n_burnin": int(burn_idx),
        "n_meas": int(len(prod_meas)),
        "seed": int(seed),
        "t_diff_ladder": np.asarray(t_diff_ladder, dtype=np.float64),
        "mean_swap_rate": mean_swap_rate,
        "swap_rates": pair_rates.astype(np.float64),
        "wall_time": float(result["wall_time"]),
        "phi_final": np.asarray(result["phi_final"], dtype=np.float32),
        "psi_final": np.asarray(result["psi_final"], dtype=np.float32),
        "phi_final_replicas": np.asarray(result["phi_final_replicas"], dtype=np.float32),
        "psi_final_replicas": np.asarray(result["psi_final_replicas"], dtype=np.float32),
        "mag_replica_0": prod_series.astype(np.float32),
        "magnetizations": mag_all.astype(np.float32),
        **obs,
    }


def save_critical_point(result: dict[str, Any], output_dir: Path, *, prefix: str = "cs") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    J = result["J"]
    h = result["h"]
    path = output_dir / f"{prefix}_h{h:.6f}_g{J:.6f}.npz"

    metadata = {
        "h": result["h"],
        "gamma": result["J"],
        "L": result["L"],
        "n_replicas": result["n_replicas"],
        "n_sweeps": result["n_sweeps"],
        "record_interval": result["record_interval"],
        "burnin_fraction": result["burnin_fraction"],
        "n_records": result["n_records"],
        "n_burnin": result["n_burnin"],
        "n_meas": result["n_meas"],
        "seed": result["seed"],
        "magnetization": result["magnetization"],
        "mag_err": result["mag_err"],
        "m2": result["m2"],
        "m4": result["m4"],
        "susceptibility": result["susceptibility"],
        "susceptibility_err": result["susceptibility_err"],
        "binder": result["binder"],
        "tau_int": result["tau_int"],
        "tau_window": result["tau_window"],
        "n_samples": result["n_samples"],
        "n_eff": result["n_eff"],
        "mean_swap_rate": result["mean_swap_rate"],
        "wall_time": result["wall_time"],
    }

    save_dict: dict[str, np.ndarray] = {
        "metadata": np.array(json.dumps(metadata)),
        "t_diff_ladder": np.asarray(result["t_diff_ladder"], dtype=np.float64),
        "swap_rates": np.asarray(result["swap_rates"], dtype=np.float64),
        "phi_final": np.asarray(result["phi_final"], dtype=np.float32),
        "psi_final": np.asarray(result["psi_final"], dtype=np.float32),
        "phi_final_replicas": np.asarray(result["phi_final_replicas"], dtype=np.float32),
        "psi_final_replicas": np.asarray(result["psi_final_replicas"], dtype=np.float32),
        "magnetizations": np.asarray(result["magnetizations"], dtype=np.float32),
    }
    mag_all = np.asarray(result["magnetizations"], dtype=np.float32)
    for rep in range(mag_all.shape[1]):
        save_dict[f"mag_replica_{rep}"] = mag_all[:, rep]

    np.savez_compressed(path, **save_dict)
    return path


def save_phase_transition_scan(
    results: list[dict[str, Any]],
    output_dir: Path,
    *,
    checkpoint: Path,
    t_prod: float,
    t_max: float,
    n_params: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    def _pad_first_axis(
        arrays: list[np.ndarray],
        *,
        dtype: np.dtype,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        if not arrays:
            raise ValueError("Expected at least one array to pad")
        max_len = max(int(arr.shape[0]) for arr in arrays)
        tail_shape = arrays[0].shape[1:]
        out = np.full((len(arrays), max_len) + tail_shape, fill_value, dtype=dtype)
        for idx, arr in enumerate(arrays):
            out[idx, : arr.shape[0], ...] = arr.astype(dtype, copy=False)
        return out

    gamma = np.array([r["J"] for r in results], dtype=np.float64)
    order_param = np.array([r["order_param"] for r in results], dtype=np.float64)
    order_err = np.array([r["order_err"] for r in results], dtype=np.float64)
    susceptibility = np.array([r["susceptibility"] for r in results], dtype=np.float64)
    susceptibility_err = np.array([r["susceptibility_err"] for r in results], dtype=np.float64)
    binder = np.array([r["binder"] for r in results], dtype=np.float64)
    tau_int = np.array([r["tau_int"] for r in results], dtype=np.float64)
    m2 = np.array([r["m2"] for r in results], dtype=np.float64)
    m4 = np.array([r["m4"] for r in results], dtype=np.float64)
    elapsed = np.array([r["wall_time"] for r in results], dtype=np.float64)
    mean_swap_rate = np.array([r["mean_swap_rate"] for r in results], dtype=np.float64)
    sweep_counts = np.array([r["n_sweeps"] for r in results], dtype=np.int64)
    record_intervals = np.array([r["record_interval"] for r in results], dtype=np.int64)
    burnin_fractions = np.array([r["burnin_fraction"] for r in results], dtype=np.float64)
    n_records_by_gamma = np.array([r["n_records"] for r in results], dtype=np.int64)
    n_burnin_by_gamma = np.array([r["n_burnin"] for r in results], dtype=np.int64)
    n_meas_by_gamma = np.array([r["n_meas"] for r in results], dtype=np.int64)
    n_eff = np.array([r["n_eff"] for r in results], dtype=np.float64)
    t_diff_ladder = np.asarray(results[0]["t_diff_ladder"], dtype=np.float64)
    magnetization_series = _pad_first_axis(
        [np.asarray(r["mag_replica_0"], dtype=np.float32) for r in results],
        dtype=np.float32,
    )
    all_replica_magnetizations = _pad_first_axis(
        [np.asarray(r["magnetizations"], dtype=np.float32) for r in results],
        dtype=np.float32,
    )
    acf_series = _pad_first_axis(
        [np.asarray(r["acf"][:200], dtype=np.float64) for r in results],
        dtype=np.float64,
    )
    phi_final_replicas = np.stack(
        [np.asarray(r["phi_final_replicas"], dtype=np.float32) for r in results],
        axis=0,
    )
    psi_final_replicas = np.stack(
        [np.asarray(r["psi_final_replicas"], dtype=np.float32) for r in results],
        axis=0,
    )
    swap_rates_by_pair = np.stack(
        [np.asarray(r["swap_rates"], dtype=np.float64) for r in results],
        axis=0,
    )

    npz_path = output_dir / "re_scan.npz"
    np.savez_compressed(
        npz_path,
        gamma=gamma,
        order_param=order_param,
        order_err=order_err,
        susceptibility=susceptibility,
        susceptibility_err=susceptibility_err,
        binder=binder,
        tau_int=tau_int,
        m2=m2,
        m4=m4,
        L=np.array(results[0]["L"]),
        t_prod=np.array(float(t_prod)),
        t_max=np.array(float(t_max)),
        n_replicas=np.array(results[0]["n_replicas"]),
        re_sweeps=np.array(int(results[0]["n_sweeps"])),
        re_sweeps_by_gamma=sweep_counts,
        re_record=np.array(int(results[0]["record_interval"])),
        re_record_by_gamma=record_intervals,
        re_burnin=np.array(float(results[0]["burnin_fraction"])),
        re_burnin_by_gamma=burnin_fractions,
        n_records_by_gamma=n_records_by_gamma,
        n_burnin_by_gamma=n_burnin_by_gamma,
        n_meas_by_gamma=n_meas_by_gamma,
        n_eff=n_eff,
        t_diff_ladder=t_diff_ladder,
        model_tag=np.array(checkpoint.stem),
        n_params=np.array(int(n_params)),
        magnetization_series=magnetization_series,
        magnetizations_all=all_replica_magnetizations,
        acf_series=acf_series,
        phi_final_replicas=phi_final_replicas,
        psi_final_replicas=psi_final_replicas,
        swap_rates_by_pair=swap_rates_by_pair,
        elapsed=elapsed,
        mean_swap_rate=mean_swap_rate,
    )

    json_path = output_dir / "summary.json"
    summary = {
        "model": {
            "tag": checkpoint.stem,
            "checkpoint": str(checkpoint),
            "n_params": int(n_params),
        },
        "lattice": {
            "L": int(results[0]["L"]),
            "n_sites": int(results[0]["L"]) ** 2,
            "boundary": "periodic",
            "h": 0.0,
        },
        "re_config": {
            "t_prod": float(t_prod),
            "t_max": float(t_max),
            "n_replicas": int(results[0]["n_replicas"]),
            "sweeps_default": int(results[0]["n_sweeps"]),
            "sweeps_by_gamma": sweep_counts.tolist(),
            "record_every": int(results[0]["record_interval"]),
            "burnin_frac": float(results[0]["burnin_fraction"]),
        },
        "gamma_grid": gamma.tolist(),
        "n_gamma": int(len(gamma)),
        "results_re": {
            "gamma": gamma.tolist(),
            "order_param": order_param.tolist(),
            "order_err": order_err.tolist(),
            "susceptibility": susceptibility.tolist(),
            "susceptibility_err": susceptibility_err.tolist(),
            "binder": binder.tolist(),
            "tau_int": tau_int.tolist(),
            "n_eff": n_eff.tolist(),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return npz_path, json_path


def save_phase_transition_point(
    result: dict[str, Any],
    output_dir: Path,
    *,
    prefix: str = "pt",
) -> Path:
    """Persist one zero-field coupling point so partial runs are inspectable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_g{float(result['J']):.6f}.npz"

    metadata = {
        "gamma": result["J"],
        "L": result["L"],
        "n_replicas": result["n_replicas"],
        "n_sweeps": result["n_sweeps"],
        "record_interval": result["record_interval"],
        "burnin_fraction": result["burnin_fraction"],
        "n_records": result["n_records"],
        "n_burnin": result["n_burnin"],
        "n_meas": result["n_meas"],
        "seed": result["seed"],
        "order_param": result["order_param"],
        "order_err": result["order_err"],
        "magnetization": result["magnetization"],
        "mag_err": result["mag_err"],
        "m2": result["m2"],
        "m4": result["m4"],
        "susceptibility": result["susceptibility"],
        "susceptibility_err": result["susceptibility_err"],
        "binder": result["binder"],
        "tau_int": result["tau_int"],
        "tau_window": result["tau_window"],
        "n_samples": result["n_samples"],
        "n_eff": result["n_eff"],
        "mean_swap_rate": result["mean_swap_rate"],
        "wall_time": result["wall_time"],
    }

    save_dict: dict[str, np.ndarray] = {
        "metadata": np.array(json.dumps(metadata)),
        "t_diff_ladder": np.asarray(result["t_diff_ladder"], dtype=np.float64),
        "swap_rates": np.asarray(result["swap_rates"], dtype=np.float64),
        "acf": np.asarray(result["acf"], dtype=np.float64),
        "phi_final": np.asarray(result["phi_final"], dtype=np.float32),
        "psi_final": np.asarray(result["psi_final"], dtype=np.float32),
        "phi_final_replicas": np.asarray(result["phi_final_replicas"], dtype=np.float32),
        "psi_final_replicas": np.asarray(result["psi_final_replicas"], dtype=np.float32),
        "mag_replica_0": np.asarray(result["mag_replica_0"], dtype=np.float32),
        "magnetizations": np.asarray(result["magnetizations"], dtype=np.float32),
    }
    np.savez_compressed(path, **save_dict)
    return path


def write_phase_transition_progress(
    results: list[dict[str, Any]],
    output_dir: Path,
    *,
    total_points: int,
    checkpoint: Path,
    t_prod: float,
    t_max: float,
    n_params: int,
) -> Path:
    """Write a lightweight progress snapshot after each zero-field point."""
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = len(results)
    latest = results[-1]
    summary = {
        "model": {
            "tag": checkpoint.stem,
            "checkpoint": str(checkpoint),
            "n_params": int(n_params),
        },
        "re_config": {
            "t_prod": float(t_prod),
            "t_max": float(t_max),
            "n_replicas": int(latest["n_replicas"]),
            "sweeps": int(latest["n_sweeps"]),
            "record_every": int(latest["record_interval"]),
            "burnin_frac": float(latest["burnin_fraction"]),
        },
        "progress": {
            "completed_points": int(completed),
            "total_points": int(total_points),
            "remaining_points": int(total_points - completed),
            "latest_gamma": float(latest["J"]),
            "latest_order_param": float(latest["order_param"]),
            "latest_order_err": float(latest["order_err"]),
            "latest_susceptibility": float(latest["susceptibility"]),
            "latest_tau_int": float(latest["tau_int"]),
            "latest_swap_rate": float(latest["mean_swap_rate"]),
            "latest_elapsed_s": float(latest["wall_time"]),
        },
        "results_so_far": {
            "gamma": [float(r["J"]) for r in results],
            "order_param": [float(r["order_param"]) for r in results],
            "order_err": [float(r["order_err"]) for r in results],
            "susceptibility": [float(r["susceptibility"]) for r in results],
            "susceptibility_err": [float(r["susceptibility_err"]) for r in results],
            "binder": [float(r["binder"]) for r in results],
            "tau_int": [float(r["tau_int"]) for r in results],
            "mean_swap_rate": [float(r["mean_swap_rate"]) for r in results],
        },
    }
    path = output_dir / "progress.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def save_critical_summary(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    checkpoint: Path,
    gamma_c: float,
    n_params: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    h = np.array([r["h"] for r in records], dtype=np.float64)
    gamma = np.array([r["J"] for r in records], dtype=np.float64)
    magnetization = np.array([r["magnetization"] for r in records], dtype=np.float64)
    mag_err = np.array([r["mag_err"] for r in records], dtype=np.float64)
    susceptibility = np.array([r["susceptibility"] for r in records], dtype=np.float64)
    susceptibility_err = np.array([r["susceptibility_err"] for r in records], dtype=np.float64)
    tau_int = np.array([r["tau_int"] for r in records], dtype=np.float64)
    mean_swap_rate = np.array([r["mean_swap_rate"] for r in records], dtype=np.float64)

    npz_path = output_dir / "critical_scaling_summary.npz"
    np.savez(
        npz_path,
        h=h,
        gamma=gamma,
        magnetization=magnetization,
        mag_err=mag_err,
        susceptibility=susceptibility,
        susceptibility_err=susceptibility_err,
        tau_int=tau_int,
        mean_swap_rate=mean_swap_rate,
        gamma_c=np.array(float(gamma_c)),
    )

    by_h: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        key = f"{rec['h']:.6f}"
        by_h.setdefault(key, {"gamma": [], "magnetization": [], "susceptibility": [], "tau_int": []})
        by_h[key]["gamma"].append(float(rec["J"]))
        by_h[key]["magnetization"].append(float(rec["magnetization"]))
        by_h[key]["susceptibility"].append(float(rec["susceptibility"]))
        by_h[key]["tau_int"].append(float(rec["tau_int"]))

    json_path = output_dir / "scan_summary.json"
    summary = {
        "model": {
            "tag": checkpoint.stem,
            "checkpoint": str(checkpoint),
            "n_params": int(n_params),
        },
        "gamma_c": float(gamma_c),
        "h_values": sorted({float(rec["h"]) for rec in records}),
        "results": by_h,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return npz_path, json_path


def _device_from_string(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _phase_settings(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    section = dict(cfg["phase_transition"])
    if "critical_window" in section and section["critical_window"] is not None:
        section["critical_window"] = dict(section["critical_window"])
    if args.output_dir is not None:
        section["output_dir"] = _resolve_path(args.output_dir, root=ROOT)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.compile_model:
        cfg["runtime"]["compile_model"] = True
    if args.seed is not None:
        section["seed"] = int(args.seed)
    if args.L is not None:
        section["L"] = int(args.L)
    if args.n_sweeps is not None:
        section["n_sweeps"] = int(args.n_sweeps)
    if args.record_interval is not None:
        section["record_interval"] = int(args.record_interval)
    if args.burnin_fraction is not None:
        section["burnin_fraction"] = float(args.burnin_fraction)
    section["save_all_replicas"] = _as_bool_override(args.save_all_replicas, bool(section["save_all_replicas"]))
    return section


def _critical_settings(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    section = dict(cfg["critical_scaling"])
    if args.output_dir is not None:
        section["output_dir"] = _resolve_path(args.output_dir, root=ROOT)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.compile_model:
        cfg["runtime"]["compile_model"] = True
    if args.seed is not None:
        section["seed"] = int(args.seed)
    if args.L is not None:
        section["L"] = int(args.L)
    if args.n_sweeps is not None:
        section["n_sweeps"] = int(args.n_sweeps)
    if args.record_interval is not None:
        section["record_interval"] = int(args.record_interval)
    if args.burnin_fraction is not None:
        section["burnin_fraction"] = float(args.burnin_fraction)
    if args.resume:
        section["resume"] = True
    section["save_all_replicas"] = _as_bool_override(args.save_all_replicas, bool(section["save_all_replicas"]))
    return section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=ROOT / "configs" / "phi4_example.yaml")
    common.add_argument("--device", type=str, default=None)
    common.add_argument("--compile-model", action="store_true")
    common.add_argument("--seed", type=int, default=None)
    common.add_argument("--L", type=int, default=None)
    common.add_argument("--n-sweeps", type=int, default=None)
    common.add_argument("--record-interval", type=int, default=None)
    common.add_argument("--burnin-fraction", type=float, default=None)
    common.add_argument("--output-dir", type=Path, default=None)
    common.add_argument("--dry-run", action="store_true")
    common.add_argument("--quiet", action="store_true")
    common.add_argument("--save-all-replicas", action=argparse.BooleanOptionalAction, default=None)

    phase = subparsers.add_parser(
        "phase-transition",
        parents=[common],
        help="Run the zero-field RE scan with config-driven parameters.",
    )
    critical = subparsers.add_parser(
        "critical-scaling",
        parents=[common],
        help="Run the nonzero-field critical-scaling scan with config-driven parameters.",
    )
    critical.add_argument("--resume", action="store_true")

    _ = phase
    return parser.parse_args()


def run_phase_transition(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    runtime = cfg["runtime"]
    diffusion_cfg = cfg["diffusion"]
    phase_cfg = _phase_settings(cfg, args)
    if args.quiet:
        runtime["quiet"] = True

    output_dir = Path(phase_cfg["output_dir"])
    checkpoint = Path(cfg["model"]["checkpoint"])
    device = _device_from_string(str(runtime["device"]))
    t_diff_ladder = build_t_diff_ladder(
        diffusion_cfg["t_prod"],
        diffusion_cfg["t_max"],
        diffusion_cfg["n_replicas"],
    )
    gamma_grid = build_phase_transition_gamma_grid(phase_cfg)

    if args.dry_run:
        payload = {
            "config": str(cfg["_config_path"]),
            "checkpoint": str(checkpoint),
            "device": str(device),
            "output_dir": str(output_dir),
            "n_gamma": int(len(gamma_grid)),
            "n_sweeps_default": int(phase_cfg["n_sweeps"]),
            "critical_window": phase_cfg.get("critical_window"),
            "record_interval": int(phase_cfg["record_interval"]),
            "t_diff_ladder": t_diff_ladder.tolist(),
        }
        print(json.dumps(payload, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    diffusion, n_params = load_diffusion_model(
        checkpoint,
        device=device,
        compile_model=bool(runtime["compile_model"]),
    )

    print(
        f"Running zero-field phase-transition scan with {len(gamma_grid)} couplings; "
        f"L={phase_cfg['L']}, base_sweeps={phase_cfg['n_sweeps']}, "
        f"record_every={phase_cfg['record_interval']}, "
        f"t=[{diffusion_cfg['t_prod']}, {diffusion_cfg['t_max']}] "
        f"({diffusion_cfg['n_replicas']} replicas)."
    )
    if phase_cfg.get("critical_window") is not None:
        window = phase_cfg["critical_window"]
        print(
            "Critical-window override: "
            f"J in [{float(window['start']):.3f}, {float(window['stop']):.3f}] "
            f"uses {int(window['n_sweeps'])} sweeps."
        )

    results: list[dict[str, Any]] = []
    for idx, J in enumerate(gamma_grid):
        point_seed = int(phase_cfg["seed"] + idx)
        point_sweeps = phase_point_n_sweeps(phase_cfg, float(J))
        print(
            f"[{idx+1:02d}/{len(gamma_grid):02d}] J={J:.6f} seed={point_seed} "
            f"sweeps={point_sweeps}"
        )
        rec = run_single_point(
            diffusion,
            J=float(J),
            h=float(phase_cfg["h"]),
            L=int(phase_cfg["L"]),
            t_diff_ladder=t_diff_ladder,
            n_sweeps=point_sweeps,
            record_interval=int(phase_cfg["record_interval"]),
            burnin_fraction=float(phase_cfg["burnin_fraction"]),
            seed=point_seed,
            device=device,
            verbose=not runtime["quiet"],
            log_every=max(1, point_sweeps // 10),
            save_all_replicas=bool(phase_cfg["save_all_replicas"]),
        )
        results.append(rec)
        point_path = save_phase_transition_point(rec, output_dir, prefix="pt")
        progress_path = write_phase_transition_progress(
            results,
            output_dir,
            total_points=len(gamma_grid),
            checkpoint=checkpoint,
            t_prod=float(diffusion_cfg["t_prod"]),
            t_max=float(diffusion_cfg["t_max"]),
            n_params=n_params,
        )
        print(
            f"  <|m|>={rec['order_param']:.4f} +- {rec['order_err']:.4f}  "
            f"chi={rec['susceptibility']:.3f}  tau_int={rec['tau_int']:.2f}  "
            f"swap={rec['mean_swap_rate']:.3f}  elapsed={rec['wall_time']:.1f}s"
        )
        print(f"  wrote: {point_path.name}, {progress_path.name}")

    npz_path, json_path = save_phase_transition_scan(
        results,
        output_dir,
        checkpoint=checkpoint,
        t_prod=float(diffusion_cfg["t_prod"]),
        t_max=float(diffusion_cfg["t_max"]),
        n_params=n_params,
    )

    print("\nSaved files:")
    print(f"  raw scan: {npz_path}")
    print(f"  summary : {json_path}")
    return 0


def run_critical_scaling(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    runtime = cfg["runtime"]
    diffusion_cfg = cfg["diffusion"]
    critical_cfg = _critical_settings(cfg, args)
    if args.quiet:
        runtime["quiet"] = True

    output_dir = Path(critical_cfg["output_dir"])
    checkpoint = Path(cfg["model"]["checkpoint"])
    device = _device_from_string(str(runtime["device"]))
    t_diff_ladder = build_t_diff_ladder(
        diffusion_cfg["t_prod"],
        diffusion_cfg["t_max"],
        diffusion_cfg["n_replicas"],
    )

    if args.dry_run:
        payload = {
            "config": str(cfg["_config_path"]),
            "checkpoint": str(checkpoint),
            "device": str(device),
            "output_dir": str(output_dir),
            "h_values": list(critical_cfg["h_values"]),
            "n_sweeps": int(critical_cfg["n_sweeps"]),
            "record_interval": int(critical_cfg["record_interval"]),
            "t_diff_ladder": t_diff_ladder.tolist(),
        }
        print(json.dumps(payload, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    diffusion, n_params = load_diffusion_model(
        checkpoint,
        device=device,
        compile_model=bool(runtime["compile_model"]),
    )

    records: list[dict[str, Any]] = []
    point_index = 0
    for h in critical_cfg["h_values"]:
        gamma_values = build_scaling_window_gamma_grid(
            float(h),
            gamma_c=float(critical_cfg["gamma_c"]),
            window_half=float(critical_cfg["window_half"]),
            n_gamma=int(critical_cfg["n_gamma"]),
            gamma_min=float(critical_cfg["gamma_min"]),
            gamma_max=float(critical_cfg["gamma_max"]),
        )
        print(
            f"h={float(h):.6f}: scanning {len(gamma_values)} couplings "
            f"in [{gamma_values[0]:.6f}, {gamma_values[-1]:.6f}]"
        )

        for J in gamma_values:
            point_seed = int(critical_cfg["seed"] + point_index)
            point_index += 1
            point_path = output_dir / f"cs_h{float(h):.6f}_g{float(J):.6f}.npz"
            if critical_cfg["resume"] and point_path.exists():
                print(f"  skipping existing point {point_path.name}")
                continue

            print(f"  J={float(J):.6f} seed={point_seed}")
            rec = run_single_point(
                diffusion,
                J=float(J),
                h=float(h),
                L=int(critical_cfg["L"]),
                t_diff_ladder=t_diff_ladder,
                n_sweeps=int(critical_cfg["n_sweeps"]),
                record_interval=int(critical_cfg["record_interval"]),
                burnin_fraction=float(critical_cfg["burnin_fraction"]),
                seed=point_seed,
                device=device,
                verbose=not runtime["quiet"],
                log_every=max(1, int(critical_cfg["n_sweeps"]) // 10),
                save_all_replicas=bool(critical_cfg["save_all_replicas"]),
            )
            save_critical_point(rec, output_dir, prefix="cs")
            records.append(rec)
            print(
                f"    <m>={rec['magnetization']:+.4f} +- {rec['mag_err']:.4f}  "
                f"chi={rec['susceptibility']:.3f}  tau_int={rec['tau_int']:.2f}  "
                f"swap={rec['mean_swap_rate']:.3f}  elapsed={rec['wall_time']:.1f}s"
            )

    if critical_cfg["resume"]:
        seen_points = {(round(float(rec["h"]), 12), round(float(rec["J"]), 12)) for rec in records}
        for path in sorted(output_dir.glob("cs_*.npz")):
            parts = path.stem.split("_")
            h_val = round(float(parts[1][1:]), 12)
            if critical_cfg.get("h_values") and h_val not in {
                round(float(h), 12) for h in critical_cfg["h_values"]
            }:
                continue
            j_val = round(float(parts[2][1:]), 12)
            if (h_val, j_val) in seen_points:
                continue
            data = np.load(path, allow_pickle=True)
            meta = json.loads(str(data["metadata"]))
            records.append(
                {
                    "h": float(meta["h"]),
                    "J": float(meta["gamma"]),
                    "magnetization": float(meta["magnetization"]),
                    "mag_err": float(meta["mag_err"]),
                    "susceptibility": float(meta["susceptibility"]),
                    "susceptibility_err": float(meta["susceptibility_err"]),
                    "tau_int": float(meta["tau_int"]),
                    "mean_swap_rate": float(meta["mean_swap_rate"]),
                }
            )

    records = filter_records_by_h_values(records, critical_cfg.get("h_values"))
    records = sorted(records, key=lambda rec: (float(rec["h"]), float(rec["J"])))
    npz_path, json_path = save_critical_summary(
        records,
        output_dir,
        checkpoint=checkpoint,
        gamma_c=float(critical_cfg["gamma_c"]),
        n_params=n_params,
    )

    print("\nSaved files:")
    print(f"  summary : {npz_path}")
    print(f"  summary : {json_path}")
    return 0


def main() -> int:
    configure_unbuffered_stdio()
    args = parse_args()
    if args.command == "phase-transition":
        return run_phase_transition(args)
    if args.command == "critical-scaling":
        return run_critical_scaling(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
