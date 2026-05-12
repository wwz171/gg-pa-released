#!/usr/bin/env python3
"""Released GG-PA example for AD-Na+."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch

from analysis_utils import burnin_slice, load_ad_sodium_result, load_curated_ad_sodium_reference
from common import (
    ROOT,
    choose_device,
    configure_unbuffered_stdio,
    ensure_dir,
    load_config,
    resolve_path,
    save_json,
)

from ggpa.core.logging import setup_logging
from ggpa.systems.alanine_dipeptide import (
    build_monomer_sodium_pipeline,
    compute_dihedrals,
    extract_monomer_oxygen_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--device", type=str, default=None, help="Override torch device.")
    parser.add_argument("--platform", type=str, default=None, help="Override OpenMM platform.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override torsion checkpoint.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    parser.add_argument("--n-steps", type=int, default=None, help="Override number of GG-PA steps.")
    parser.add_argument("--n-trajectories", type=int, default=None, help="Override trajectory count for the ensemble.")
    parser.add_argument("--burnin-fraction", type=float, default=None, help="Override burn-in fraction for ensemble analysis.")
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run one standalone trajectory instead of the default released ensemble.",
    )
    parser.add_argument(
        "--save-dcd",
        dest="save_dcd",
        action="store_true",
        help="Force DCD saving on.",
    )
    parser.add_argument(
        "--no-save-dcd",
        dest="save_dcd",
        action="store_false",
        help="Force DCD saving off.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the summary plot generated at the end of the run.",
    )
    parser.set_defaults(save_dcd=None)
    return parser.parse_args()


def run_ad_sodium_once(
    cfg: dict,
    *,
    device_override: str | None = None,
    platform_override: str | None = None,
    checkpoint_override: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    seed_override: int | None = None,
    n_steps_override: int | None = None,
    save_dcd_override: bool | None = None,
    make_plots: bool = True,
) -> dict[str, object]:
    shared = cfg["shared"]
    section = cfg["ad_sodium"]

    if device_override is not None:
        shared["device"] = choose_device(device_override)
    if platform_override is not None:
        shared["platform_name"] = platform_override
    if checkpoint_override is not None:
        shared["checkpoint"] = Path(checkpoint_override).resolve()
    if output_dir_override is not None:
        section["output_dir"] = Path(output_dir_override).resolve()
    if seed_override is not None:
        shared["master_seed"] = int(seed_override)
    if n_steps_override is not None:
        section["n_steps"] = int(n_steps_override)
    if save_dcd_override is not None:
        section["save_dcd"] = bool(save_dcd_override)

    out_dir = ensure_dir(section["output_dir"])
    fig_dir = ensure_dir(shared["output_root"] / "figures")

    seed = int(shared["master_seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(
        "Running AD-Na+ GG-PA example "
        f"(t_diff={section['t_diff']}, steps={section['n_steps']}, device={shared['device']}, "
        f"platform={shared['platform_name']})"
    )

    pipe = build_monomer_sodium_pipeline(
        pdb_path=str(section["pdb"]),
        checkpoint_path=str(shared["checkpoint"]),
        forcefield_files=list(shared["forcefield_files"]),
        temperature=float(section["temperature"]),
        friction=float(section["friction"]),
        md_steps=int(section["md_steps"]),
        platform_name=str(shared["platform_name"]),
        device=str(shared["device"]),
        kappa=float(shared["kappa"]),
        k_max=int(shared["k_max"]),
        timestep=float(section["timestep"]),
        n_reverse_steps=int(shared["n_reverse_steps"]),
        minimize_before_md=bool(section["minimize_before_md"]),
        nonbonded_mode=str(shared["nonbonded_mode"]),
        internal_strength_scaling=dict(shared["internal_strength_scaling"]),
        ion_element_symbol=str(section["ion_element_symbol"]),
        ion_resname=str(section["ion_resname"]),
        ion_atom_name=str(section["ion_atom_name"]),
        ion_offset_nm=float(section["ion_offset_nm"]),
        leash_r0_nm=float(section["leash_r0_nm"]),
        leash_k_kj_mol_nm2=float(section["leash_k_kj_mol_nm2"]),
        master_seed=seed,
    )

    aggregator = pipe["aggregator"]
    clients = pipe["clients"]
    kernel = pipe["kernel"]
    torsion_info = pipe["torsion_info"]
    state = pipe["init_state"].clone()

    if bool(section["save_dcd"]):
        aggregator.add_dcd_reporter(
            str(out_dir / "ad_sodium_trajectory.dcd"),
            report_interval=int(section["dcd_interval"]),
        )

    aggregator.write_current_pdb(out_dir / "ad_sodium_initial.pdb")

    t_diff = float(section["t_diff"])
    n_steps = int(section["n_steps"])
    record_interval = int(section["record_interval"])
    print_every = int(section["print_every"])
    compute_reduced_potential = bool(section["compute_reduced_potential"])
    oxygen_pair = extract_monomer_oxygen_indices(str(section["pdb"]))["pair"]

    recorded_steps: list[int] = []
    dihedrals_rad: list[np.ndarray] = []
    x_dihedrals_rad: list[np.ndarray] = []
    oo_distance_nm: list[float] = []

    t0 = time.time()
    for step_idx in range(n_steps):
        state, _ = kernel.step(
            state,
            t_diff,
            compute_reduced_potential=compute_reduced_potential,
        )

        if (step_idx + 1) % record_interval == 0:
            current_y = compute_dihedrals(state.s, torsion_info["all"])
            current_x = np.asarray(clients["monomer"].current_x, dtype=np.float64).reshape(-1)
            current_oo = float(np.linalg.norm(state.s[oxygen_pair[0]] - state.s[oxygen_pair[1]]))
            recorded_steps.append(step_idx + 1)
            dihedrals_rad.append(current_y)
            x_dihedrals_rad.append(current_x)
            oo_distance_nm.append(current_oo)

        if print_every > 0 and (step_idx + 1) % print_every == 0:
            elapsed = time.time() - t0
            rate = elapsed / (step_idx + 1)
            eta = rate * (n_steps - step_idx - 1)
            print(
                f"  step {step_idx + 1}/{n_steps}  "
                f"elapsed={elapsed:.1f}s  eta={eta:.1f}s"
            )

    wall_time = time.time() - t0
    aggregator.write_current_pdb(out_dir / "ad_sodium_final.pdb")

    dihedrals_rad_arr = np.asarray(dihedrals_rad, dtype=np.float64)
    x_dihedrals_rad_arr = np.asarray(x_dihedrals_rad, dtype=np.float64)
    recorded_steps_arr = np.asarray(recorded_steps, dtype=np.int64)
    oo_distance_nm_arr = np.asarray(oo_distance_nm, dtype=np.float64)

    np.savez_compressed(
        out_dir / "ad_sodium_results.npz",
        steps=recorded_steps_arr,
        t_diff=np.array(t_diff, dtype=np.float64),
        wall_time_s=np.array(wall_time, dtype=np.float64),
        dihedrals_rad=dihedrals_rad_arr,
        dihedrals_deg=np.degrees(dihedrals_rad_arr),
        x_dihedrals_rad=x_dihedrals_rad_arr,
        x_dihedrals_deg=np.degrees(x_dihedrals_rad_arr),
        oo_distance_nm=oo_distance_nm_arr,
        oxygen_pair=oxygen_pair,
        final_positions_nm=np.asarray(state.s, dtype=np.float64),
    )

    save_json(
        out_dir / "summary.json",
        {
            "mode": "ad_sodium",
            "checkpoint": str(shared["checkpoint"]),
            "pdb": str(section["pdb"]),
            "t_diff": t_diff,
            "n_steps": n_steps,
            "record_interval": record_interval,
            "md_steps": int(section["md_steps"]),
            "temperature": float(section["temperature"]),
            "friction": float(section["friction"]),
            "timestep": float(section["timestep"]),
            "wall_time_s": wall_time,
            "final_phi_deg": float(np.degrees(dihedrals_rad_arr[-1, 0])),
            "final_psi_deg": float(np.degrees(dihedrals_rad_arr[-1, 1])),
            "mean_phi_deg": float(np.degrees(dihedrals_rad_arr[:, 0]).mean()),
            "mean_psi_deg": float(np.degrees(dihedrals_rad_arr[:, 1]).mean()),
            "mean_oo_distance_nm": float(oo_distance_nm_arr.mean()),
            "final_oo_distance_nm": float(oo_distance_nm_arr[-1]),
        },
    )

    if make_plots:
        from plot_alanine_results import plot_ad_sodium

        plot_ad_sodium(out_dir / "ad_sodium_results.npz", fig_dir)
    if make_plots:
        print(
            f"Finished in {wall_time:.1f}s. Results: {out_dir / 'ad_sodium_results.npz'}; "
            f"figures: {fig_dir / 'ad_sodium_overview.png'}"
        )
    else:
        print(f"Finished in {wall_time:.1f}s. Results: {out_dir / 'ad_sodium_results.npz'}")
    return {
        "out_dir": out_dir,
        "fig_dir": fig_dir,
        "result_path": out_dir / "ad_sodium_results.npz",
        "summary_path": out_dir / "summary.json",
        "wall_time_s": wall_time,
        "seed": seed,
        "t_diff": t_diff,
        "n_steps": n_steps,
        "checkpoint": shared["checkpoint"],
    }


def _fraction_positive(col: np.ndarray) -> float:
    return float(np.mean(np.asarray(col, dtype=np.float64) > 0.0))


def _fraction_psi_gt_90(col: np.ndarray) -> float:
    return float(np.mean(np.asarray(col, dtype=np.float64) > 90.0))


def _summarize_angles(label: str, y_deg: np.ndarray, oo_nm: np.ndarray) -> dict[str, object]:
    return {
        "label": label,
        "n_frames": int(len(y_deg)),
        "phi_mean_deg": float(np.mean(y_deg[:, 0])),
        "psi_mean_deg": float(np.mean(y_deg[:, 1])),
        "phi_positive_fraction": _fraction_positive(y_deg[:, 0]),
        "psi_gt_90_fraction": _fraction_psi_gt_90(y_deg[:, 1]),
        "oo_mean_nm": float(np.mean(oo_nm)),
        "oo_std_nm": float(np.std(oo_nm)),
    }


def run_ad_sodium_ensemble(
    cfg: dict,
    *,
    device_override: str | None = None,
    platform_override: str | None = None,
    checkpoint_override: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    n_steps_override: int | None = None,
    n_trajectories_override: int | None = None,
    burnin_fraction_override: float | None = None,
    save_dcd_override: bool | None = None,
    make_plots: bool = True,
) -> dict[str, object]:
    shared = cfg["shared"]
    sodium = cfg["ad_sodium"]
    ensemble = cfg["ad_sodium_ensemble"]

    output_root = (
        resolve_path(output_dir_override, root=ROOT)
        if output_dir_override is not None
        else Path(ensemble["output_dir"])
    )
    output_root = ensure_dir(output_root)
    run_dir = ensure_dir(output_root / "runs")
    figure_root = shared["output_root"] if output_dir_override is None else output_root
    fig_dir = ensure_dir(Path(figure_root) / "figures")

    if checkpoint_override is not None:
        checkpoint_path = resolve_path(checkpoint_override, root=ROOT)
    else:
        checkpoint_path = Path(shared["checkpoint"])

    n_steps = int(n_steps_override if n_steps_override is not None else sodium["n_steps"])
    burnin_fraction = float(
        burnin_fraction_override if burnin_fraction_override is not None else ensemble["burnin_fraction"]
    )
    seeds = list(int(x) for x in ensemble["seeds"])
    if n_trajectories_override is not None:
        seeds = seeds[: int(n_trajectories_override)]
    n_trajectories = len(seeds)

    print(
        "Running AD-Na+ zero-shot GG-PA ensemble "
        f"({n_trajectories} trajectories, {n_steps} GG-PA steps each, burn-in={burnin_fraction:.2f})"
    )

    run_summaries: list[dict[str, object]] = []
    y_deg_all: list[np.ndarray] = []
    x_deg_all: list[np.ndarray] = []
    oo_all: list[np.ndarray] = []

    for traj_idx, seed in enumerate(seeds, start=1):
        traj_dir = ensure_dir(run_dir / f"traj_{traj_idx:02d}")
        print(f"  trajectory {traj_idx}/{n_trajectories}  seed={seed}")
        result = run_ad_sodium_once(
            cfg,
            device_override=device_override,
            platform_override=platform_override,
            checkpoint_override=checkpoint_path,
            output_dir_override=traj_dir,
            seed_override=seed,
            n_steps_override=n_steps,
            save_dcd_override=True if save_dcd_override is None else save_dcd_override,
            make_plots=False,
        )

        payload = load_ad_sodium_result(result["result_path"])
        burn = burnin_slice(len(payload["steps"]), burnin_fraction)

        y_post = payload["y_deg"][burn]
        x_post = payload["x_deg"][burn]
        oo_post = payload["oo_distance_nm"][burn]

        y_deg_all.append(y_post)
        x_deg_all.append(x_post)
        oo_all.append(oo_post)

        postburn_summary = {
            "traj_idx": traj_idx,
            "seed": seed,
            "n_frames_total": int(len(payload["steps"])),
            "n_frames_postburn": int(len(y_post)),
            "wall_time_s": float(result["wall_time_s"]),
            "result_path": str(result["result_path"]),
            "summary_path": str(result["summary_path"]),
            "dcd_path": str(traj_dir / "ad_sodium_trajectory.dcd"),
            "phi_positive_fraction_noisy": _fraction_positive(y_post[:, 0]),
            "phi_positive_fraction_clean": _fraction_positive(x_post[:, 0]),
            "psi_gt_90_fraction_noisy": _fraction_psi_gt_90(y_post[:, 1]),
            "psi_gt_90_fraction_clean": _fraction_psi_gt_90(x_post[:, 1]),
            "oo_mean_nm": float(np.mean(oo_post)),
            "oo_std_nm": float(np.std(oo_post)),
        }
        save_json(traj_dir / "postburn_summary.json", postburn_summary)
        run_summaries.append(postburn_summary)

    y_deg = np.concatenate(y_deg_all, axis=0)
    x_deg = np.concatenate(x_deg_all, axis=0)
    oo_nm = np.concatenate(oo_all, axis=0)

    aggregate_path = output_root / "aggregate_samples.npz"
    np.savez_compressed(
        aggregate_path,
        y_deg=np.asarray(y_deg, dtype=np.float64),
        x_deg=np.asarray(x_deg, dtype=np.float64),
        y_rad=np.deg2rad(np.asarray(y_deg, dtype=np.float64)),
        x_rad=np.deg2rad(np.asarray(x_deg, dtype=np.float64)),
        oo_distance_nm=np.asarray(oo_nm, dtype=np.float64),
        burnin_fraction=np.array(burnin_fraction, dtype=np.float64),
        n_trajectories=np.array(n_trajectories, dtype=np.int64),
        n_steps=np.array(n_steps, dtype=np.int64),
    )

    ref_dir = ROOT / "data" / "ad_sodium_ref"
    vacuum_ref = load_curated_ad_sodium_reference(ref_dir / "monomer_vacuum_300k_ref.npz")
    ion_ref = load_curated_ad_sodium_reference(ref_dir / "ad_sodium_md_300k_ref.npz")

    summary = {
        "mode": "ad_sodium_ensemble",
        "checkpoint": str(checkpoint_path),
        "config_path": str(ROOT / "configs" / "alanine_dipeptide_example.yaml"),
        "output_dir": str(output_root),
        "n_trajectories": n_trajectories,
        "seeds": seeds,
        "n_steps": n_steps,
        "burnin_fraction": burnin_fraction,
        "aggregate_path": str(aggregate_path),
        "device": str(device_override or shared["device"]),
        "platform": str(platform_override or shared["platform_name"]),
        "trajectory_summary": run_summaries,
        "aggregate_noisy": _summarize_angles("ggpa_noisy", y_deg, oo_nm),
        "aggregate_clean": _summarize_angles("ggpa_clean", x_deg, oo_nm),
        "reference_counts": {
            "vacuum_frames": int(vacuum_ref["n_frames"]),
            "ion_coupled_frames": int(ion_ref["n_frames"]),
        },
    }
    save_json(output_root / "summary.json", summary)
    shutil.copy2(ROOT / "configs" / "alanine_dipeptide_example.yaml", output_root / "config_used.yaml")

    if make_plots:
        from plot_alanine_results import plot_ad_sodium_ensemble

        plot_ad_sodium_ensemble(output_root, fig_dir)
        print(
            "Finished AD-Na+ zero-shot ensemble. "
            f"aggregate: {aggregate_path}; figures: {fig_dir / 'ad_sodium_rama_triptych.png'}"
        )
    else:
        print(f"Finished AD-Na+ zero-shot ensemble. aggregate: {aggregate_path}")

    return {
        "output_dir": output_root,
        "aggregate_path": aggregate_path,
        "summary_path": output_root / "summary.json",
        "figure_dir": fig_dir,
    }


def main() -> None:
    configure_unbuffered_stdio()
    setup_logging(level=logging.WARNING)

    args = parse_args()
    cfg = load_config(args.config)
    if args.single_run:
        run_ad_sodium_once(
            cfg,
            device_override=args.device,
            platform_override=args.platform,
            checkpoint_override=args.checkpoint,
            output_dir_override=args.output_dir,
            seed_override=args.seed,
            n_steps_override=args.n_steps,
            save_dcd_override=args.save_dcd,
            make_plots=not args.no_plot,
        )
    else:
        run_ad_sodium_ensemble(
            cfg,
            device_override=args.device,
            platform_override=args.platform,
            checkpoint_override=args.checkpoint,
            output_dir_override=args.output_dir,
            n_steps_override=args.n_steps,
            n_trajectories_override=args.n_trajectories,
            burnin_fraction_override=args.burnin_fraction,
            save_dcd_override=args.save_dcd,
            make_plots=not args.no_plot,
        )


if __name__ == "__main__":
    main()
