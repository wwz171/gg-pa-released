#!/usr/bin/env python3
"""Public GG-PA replica-exchange example for the alanine dipeptide dimer."""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch

from common import choose_device, configure_unbuffered_stdio, ensure_dir, load_config, save_json

from ggpa.core.logging import setup_logging
from ggpa.systems.alanine_dipeptide import (
    AlanineReplicaExchange,
    analyze_dimer_trajectory,
    build_dimer_pipeline,
    classify_states,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--device", type=str, default=None, help="Override torch device.")
    parser.add_argument("--platform", type=str, default=None, help="Override OpenMM platform.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override torsion checkpoint.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    parser.add_argument("--n-blocks", type=int, default=None, help="Override number of RE blocks.")
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
        help="Skip summary plotting at the end of the run.",
    )
    parser.set_defaults(save_dcd=None)
    return parser.parse_args()


def run_ad_dimer_once(
    cfg: dict,
    *,
    device_override: str | None = None,
    platform_override: str | None = None,
    checkpoint_override: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    seed_override: int | None = None,
    n_blocks_override: int | None = None,
    save_dcd_override: bool | None = None,
    make_plots: bool = True,
) -> dict[str, object]:
    cfg = copy.deepcopy(cfg)
    shared = cfg["shared"]
    section = cfg["ad_dimer"]

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
    if n_blocks_override is not None:
        section["n_blocks"] = int(n_blocks_override)
    if save_dcd_override is not None:
        section["save_dcd"] = bool(save_dcd_override)

    out_dir = ensure_dir(section["output_dir"])
    fig_root = shared["output_root"] if output_dir_override is None else out_dir.parent
    fig_dir = ensure_dir(Path(fig_root) / "figures")

    seed = int(shared["master_seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    tau_list = [float(tau) for tau in section["tau_list"]]
    n_replicas = len(tau_list)
    print(
        "Running alanine dipeptide dimer RE example "
        f"(replicas={n_replicas}, taus={tau_list}, blocks={section['n_blocks']}, "
        f"device={shared['device']}, platform={shared['platform_name']})"
    )

    pipes = []
    for replica_idx in range(n_replicas):
        pipe = build_dimer_pipeline(
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
            centering_force_k=float(section["centering_force_k"]),
            centering_d0=section["centering_d0"],
            centering_warmup_steps=int(section["centering_warmup_steps"]),
            centering_schedule=str(section["centering_schedule"]),
            master_seed=seed + 1000 * replica_idx,
        )
        if bool(section["save_dcd"]):
            pipe["aggregator"].add_dcd_reporter(
                str(out_dir / f"ad_dimer_replica_{replica_idx}.dcd"),
                report_interval=int(section["dcd_interval"]),
            )
        pipe["aggregator"].write_current_pdb(out_dir / f"ad_dimer_replica_{replica_idx}_initial.pdb")
        pipes.append(pipe)

    init_states = [pipe["init_state"].clone() for pipe in pipes]
    runner = AlanineReplicaExchange(pipes, taus=tau_list, rng_seed=seed)
    result = runner.run(
        n_blocks=int(section["n_blocks"]),
        inner_steps=int(section["inner_steps"]),
        record_interval=int(section["record_interval"]),
        init_states=init_states,
        save_positions=bool(section["save_positions"]),
        print_every=int(section["print_every"]),
    )

    for replica_idx, pipe in enumerate(pipes):
        pipe["aggregator"].write_current_pdb(out_dir / f"ad_dimer_replica_{replica_idx}_final.pdb")

    production_replica = 0
    all_dihedrals_rad = np.stack(
        [np.asarray(result["dihedrals_rad"][rep], dtype=np.float64) for rep in range(n_replicas)],
        axis=0,
    )
    all_x_rad = np.stack(
        [np.asarray(result["x_dihedrals_rad"][rep], dtype=np.float64) for rep in range(n_replicas)],
        axis=0,
    )
    production_dihedrals_rad = all_dihedrals_rad[production_replica]
    production_x_rad = all_x_rad[production_replica]
    production_steps = (
        np.arange(len(production_dihedrals_rad), dtype=np.int64) + 1
    ) * int(section["record_interval"])

    summary = {
        "mode": "ad_dimer",
        "checkpoint": str(shared["checkpoint"]),
        "pdb": str(section["pdb"]),
        "tau_list": tau_list,
        "n_blocks": int(section["n_blocks"]),
        "inner_steps": int(section["inner_steps"]),
        "record_interval": int(section["record_interval"]),
        "md_steps": int(section["md_steps"]),
        "burnin_fraction": float(section.get("burnin_fraction", 0.2)),
        "temperature": float(section["temperature"]),
        "friction": float(section["friction"]),
        "timestep": float(section["timestep"]),
        "wall_time_s": float(result["wall_time_s"]),
        "acceptance_rates": {
            f"{pair[0]}-{pair[1]}": float(rate)
            for pair, rate in result["acceptance_rates"].items()
        },
    }

    savez_payload = {
        "taus": np.asarray(result["taus"], dtype=np.float64),
        "production_steps": production_steps,
        "dihedrals_by_replica_rad": all_dihedrals_rad,
        "dihedrals_by_replica_deg": np.degrees(all_dihedrals_rad),
        "x_dihedrals_by_replica_rad": all_x_rad,
        "x_dihedrals_by_replica_deg": np.degrees(all_x_rad),
        "production_dihedrals_rad": production_dihedrals_rad,
        "production_dihedrals_deg": np.degrees(production_dihedrals_rad),
        "production_x_dihedrals_rad": production_x_rad,
        "production_x_dihedrals_deg": np.degrees(production_x_rad),
        "wall_time_s": np.array(result["wall_time_s"], dtype=np.float64),
        "final_dihedrals_by_replica_deg": np.vstack(
            [np.asarray(result["dihedrals"][rep])[-1] for rep in range(n_replicas)]
        ),
    }

    if bool(section["save_positions"]):
        all_positions = np.stack(
            [np.asarray(result["positions"][rep], dtype=np.float64) for rep in range(n_replicas)],
            axis=0,
        )
        analyses = []
        replica_summaries = {}
        for rep in range(n_replicas):
            analysis_rep = analyze_dimer_trajectory(
                positions_list=list(all_positions[rep]),
                torsion_info=pipes[rep]["torsion_info"],
                pdb_path=str(section["pdb"]),
            )
            labels_rep = classify_states(analysis_rep)
            analyses.append(analysis_rep)
            replica_summaries[f"replica_{rep}"] = {
                "tau": tau_list[rep],
                "mean_com_distance_nm": float(np.mean(analysis_rep["com_distances"])),
                "mean_cosine_similarity": float(np.mean(analysis_rep["cosine_similarities"])),
                "mean_hbond_count": float(np.mean(analysis_rep["hbond_counts"])),
                "state_counts": {
                    label: int(np.sum(labels_rep == label))
                    for label in np.unique(labels_rep)
                },
            }

        production_positions = all_positions[production_replica]
        analysis = analyses[production_replica]
        labels = classify_states(analysis)
        summary.update(
            {
                "mean_com_distance_nm": float(np.mean(analysis["com_distances"])),
                "mean_cosine_similarity": float(np.mean(analysis["cosine_similarities"])),
                "mean_hbond_count": float(np.mean(analysis["hbond_counts"])),
                "state_counts": {
                    label: int(np.sum(labels == label))
                    for label in np.unique(labels)
                },
                "replica_summaries": replica_summaries,
            }
        )
        analysis_com = np.stack(
            [np.asarray(a["com_distances"], dtype=np.float64) for a in analyses], axis=0,
        )
        analysis_cos = np.stack(
            [np.asarray(a["cosine_similarities"], dtype=np.float64) for a in analyses], axis=0,
        )
        analysis_hb = np.stack(
            [np.asarray(a["hbond_counts"], dtype=np.float64) for a in analyses], axis=0,
        )
        analysis_rec = np.stack(
            [np.asarray(a["reciprocal_counts"], dtype=np.float64) for a in analyses], axis=0,
        )
        analysis_seg1 = np.stack(
            [np.asarray(a["dihedrals_seg1"], dtype=np.float64) for a in analyses], axis=0,
        )
        analysis_seg2 = np.stack(
            [np.asarray(a["dihedrals_seg2"], dtype=np.float64) for a in analyses], axis=0,
        )
        savez_payload.update(
            {
                "positions_by_replica_nm": all_positions,
                "production_positions_nm": production_positions,
                "analysis_com_distances_by_replica": analysis_com,
                "analysis_cosine_similarities_by_replica": analysis_cos,
                "analysis_hbond_counts_by_replica": analysis_hb,
                "analysis_reciprocal_counts_by_replica": analysis_rec,
                "analysis_dihedrals_seg1_by_replica": analysis_seg1,
                "analysis_dihedrals_seg2_by_replica": analysis_seg2,
                "analysis_com_distances": np.asarray(analysis["com_distances"], dtype=np.float64),
                "analysis_cosine_similarities": np.asarray(analysis["cosine_similarities"], dtype=np.float64),
                "analysis_hbond_counts": np.asarray(analysis["hbond_counts"], dtype=np.float64),
                "analysis_reciprocal_counts": np.asarray(analysis["reciprocal_counts"], dtype=np.float64),
                "analysis_dihedrals_seg1": np.asarray(analysis["dihedrals_seg1"], dtype=np.float64),
                "analysis_dihedrals_seg2": np.asarray(analysis["dihedrals_seg2"], dtype=np.float64),
            }
        )

    np.savez_compressed(out_dir / "ad_dimer_results.npz", **savez_payload)
    save_json(out_dir / "summary.json", summary)

    if bool(section["save_positions"]) and make_plots:
        from plot_alanine_results import plot_ad_dimer

        plot_info = plot_ad_dimer(
            out_dir / "ad_dimer_results.npz",
            fig_dir,
            burnin_fraction=float(section.get("burnin_fraction", 0.2)),
        )
        plot_target = plot_info["abs_psi_figure"]
    else:
        plot_target = "(positions disabled; no plots generated)"

    print(
        f"Finished in {float(result['wall_time_s']):.1f}s. Results: {out_dir / 'ad_dimer_results.npz'}; "
        f"figures: {plot_target}"
    )
    return {
        "out_dir": out_dir,
        "fig_dir": fig_dir,
        "result_path": out_dir / "ad_dimer_results.npz",
        "summary_path": out_dir / "summary.json",
        "wall_time_s": float(result["wall_time_s"]),
        "seed": seed,
        "tau_list": tau_list,
        "n_blocks": int(section["n_blocks"]),
        "checkpoint": shared["checkpoint"],
    }


def main() -> None:
    configure_unbuffered_stdio()
    setup_logging(level=logging.WARNING)

    args = parse_args()
    cfg = load_config(args.config)
    run_ad_dimer_once(
        cfg,
        device_override=args.device,
        platform_override=args.platform,
        checkpoint_override=args.checkpoint,
        output_dir_override=args.output_dir,
        seed_override=args.seed,
        n_blocks_override=args.n_blocks,
        save_dcd_override=args.save_dcd,
        make_plots=not args.no_plot,
    )


if __name__ == "__main__":
    main()
