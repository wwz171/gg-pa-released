#!/usr/bin/env python3
"""Plot saved results from the public alanine-dipeptide examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from common import ROOT, load_config, resolve_path
from analysis_utils import (
    burnin_slice,
    load_ad_dimer_result,
    load_curated_ad_dimer_reference,
    load_curated_ad_sodium_reference,
)
from ggpa.systems.alanine_dipeptide import classify_dimer_monomer_basin


def _deg(x: np.ndarray) -> np.ndarray:
    return np.degrees(np.asarray(x, dtype=np.float64))


def _wrap_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _extract_abs_psi(obs_dict: dict[str, np.ndarray], sector: str) -> np.ndarray:
    cosine = np.asarray(obs_dict["cosine_sim"], dtype=np.float64)
    hbonds = np.asarray(obs_dict["hbond_counts"], dtype=np.float64)
    reciprocal = np.asarray(obs_dict["reciprocal_counts"], dtype=np.float64)
    dihedrals_1 = np.asarray(obs_dict["dihedrals_seg1"], dtype=np.float64)
    dihedrals_2 = np.asarray(obs_dict["dihedrals_seg2"], dtype=np.float64)
    basin1 = classify_dimer_monomer_basin(dihedrals_1)
    basin2 = classify_dimer_monomer_basin(dihedrals_2)

    valid = (reciprocal == 1.0) & (np.abs(cosine) >= 0.9) & (hbonds >= 2.0)
    if sector == "A":
        valid &= cosine <= -0.5
    elif sector == "P":
        valid &= cosine >= 0.5
    else:
        raise ValueError(f"Unknown sector: {sector}")
    valid &= (basin1 != "U") & (basin2 != "U")
    return np.abs(_wrap_delta(dihedrals_1[valid, 1], dihedrals_2[valid, 1]))


def _extract_abs_phi(obs_dict: dict[str, np.ndarray], sector: str) -> np.ndarray:
    cosine = np.asarray(obs_dict["cosine_sim"], dtype=np.float64)
    hbonds = np.asarray(obs_dict["hbond_counts"], dtype=np.float64)
    reciprocal = np.asarray(obs_dict["reciprocal_counts"], dtype=np.float64)
    dihedrals_1 = np.asarray(obs_dict["dihedrals_seg1"], dtype=np.float64)
    dihedrals_2 = np.asarray(obs_dict["dihedrals_seg2"], dtype=np.float64)
    basin1 = classify_dimer_monomer_basin(dihedrals_1)
    basin2 = classify_dimer_monomer_basin(dihedrals_2)

    valid = (reciprocal == 1.0) & (np.abs(cosine) >= 0.9) & (hbonds >= 2.0)
    if sector == "A":
        valid &= cosine <= -0.5
    elif sector == "P":
        valid &= cosine >= 0.5
    else:
        raise ValueError(f"Unknown sector: {sector}")
    valid &= (basin1 != "U") & (basin2 != "U")
    return np.abs(_wrap_delta(dihedrals_1[valid, 0], dihedrals_2[valid, 0]))


def plot_ad_sodium(result_path: str | Path, output_dir: str | Path) -> None:
    payload = np.load(result_path, allow_pickle=True)
    steps = payload["steps"]
    y_deg = payload["dihedrals_deg"]
    x_deg = payload["x_dihedrals_deg"]
    oo_distance_nm = payload["oo_distance_nm"] if "oo_distance_nm" in payload.files else None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)

    axes[0, 0].plot(steps, y_deg[:, 0], label=r"$\phi$", lw=1.2)
    axes[0, 0].plot(steps, y_deg[:, 1], label=r"$\psi$", lw=1.2)
    axes[0, 0].set_title("Sampled Torsions")
    axes[0, 0].set_xlabel("GG-PA step")
    axes[0, 0].set_ylabel("degrees")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(steps, x_deg[:, 0], label=r"$x_\phi$", lw=1.2)
    axes[0, 1].plot(steps, x_deg[:, 1], label=r"$x_\psi$", lw=1.2)
    axes[0, 1].set_title("Denoised Torsion Anchors")
    axes[0, 1].set_xlabel("GG-PA step")
    axes[0, 1].set_ylabel("degrees")
    axes[0, 1].legend(frameon=False)

    if oo_distance_nm is not None:
        axes[0, 2].plot(steps, oo_distance_nm, lw=1.2, color="#2ca02c")
        axes[0, 2].set_title("O-O Distance")
        axes[0, 2].set_xlabel("GG-PA step")
        axes[0, 2].set_ylabel("nm")
    else:
        axes[0, 2].axis("off")

    axes[1, 0].scatter(y_deg[:, 0], y_deg[:, 1], s=6, alpha=0.35, color="#1f77b4")
    axes[1, 0].set_title("Sampled Ramachandran")
    axes[1, 0].set_xlabel(r"$\phi$ (deg)")
    axes[1, 0].set_ylabel(r"$\psi$ (deg)")
    axes[1, 0].set_xlim(-180, 180)
    axes[1, 0].set_ylim(-180, 180)

    axes[1, 1].scatter(x_deg[:, 0], x_deg[:, 1], s=6, alpha=0.35, color="#d62728")
    axes[1, 1].set_title("Denoised Anchor Ramachandran")
    axes[1, 1].set_xlabel(r"$\phi$ (deg)")
    axes[1, 1].set_ylabel(r"$\psi$ (deg)")
    axes[1, 1].set_xlim(-180, 180)
    axes[1, 1].set_ylim(-180, 180)

    if oo_distance_nm is not None:
        axes[1, 2].hist(oo_distance_nm, bins=40, density=True, color="#2ca02c", alpha=0.75)
        axes[1, 2].set_title("O-O Distance Distribution")
        axes[1, 2].set_xlabel("nm")
        axes[1, 2].set_ylabel("density")
    else:
        axes[1, 2].axis("off")

    fig.suptitle("Alanine Dipeptide + Na+", fontsize=13)
    fig.savefig(output_dir / "ad_sodium_overview.png", dpi=200)
    fig.savefig(output_dir / "ad_sodium_overview.pdf")
    plt.close(fig)


def plot_ad_sodium_ensemble(result_dir: str | Path, output_dir: str | Path) -> None:
    result_dir = Path(result_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(result_dir / "aggregate_samples.npz", allow_pickle=True)
    y_deg = np.asarray(payload["y_deg"], dtype=np.float64)
    oo_nm = np.asarray(payload["oo_distance_nm"], dtype=np.float64)

    vacuum_ref = load_curated_ad_sodium_reference(
        ROOT / "data" / "ad_sodium_ref" / "monomer_vacuum_300k_ref.npz"
    )
    ion_ref = load_curated_ad_sodium_reference(
        ROOT / "data" / "ad_sodium_ref" / "ad_sodium_md_300k_ref.npz"
    )

    num_shown = min(16000, len(y_deg))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_deg), size=num_shown, replace=False) if len(y_deg) > num_shown else slice(None)

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.0), sharex=True, sharey=True, constrained_layout=True)
    panels = [
        ("Vacuum MD", vacuum_ref["dihedrals_deg"], "#CCBB44"),
        ("GG-PA (0%)", y_deg[idx], "#AA3377"),
        ("Ion-coupled MD", ion_ref["dihedrals_deg"], "#EE6677"),
    ]
    for ax, (title, arr, color) in zip(axes, panels):
        ax.scatter(arr[:, 0], arr[:, 1], s=2, alpha=0.28, linewidths=0, color=color, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel(r"$\phi$ (deg)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$\psi$ (deg)")
    fig.savefig(output_dir / "ad_sodium_rama_triptych.png", dpi=220)
    fig.savefig(output_dir / "ad_sodium_rama_triptych.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    sns.kdeplot(vacuum_ref["oo_distance_nm"] * 10.0, ax=ax, color="#CCBB44", linewidth=1.6, bw_adjust=1.5, fill=False, label="Vacuum MD")
    sns.kdeplot(oo_nm * 10.0, ax=ax, color="#AA3377", linewidth=1.8, bw_adjust=1.5, fill=False, label="GG-PA (0%)")
    sns.kdeplot(ion_ref["oo_distance_nm"] * 10.0, ax=ax, color="#EE6677", linewidth=1.6, bw_adjust=1.5, fill=False, linestyle="--", label="Ion-coupled MD")
    ax.set_xlabel(r"$d_\mathrm{OO}$ ($\AA$)")
    ax.set_ylabel("Density")
    ax.set_xlim(1.5, 6.5)
    ax.set_xticks(np.arange(2.0, 7.0, 1.0))
    ax.legend(loc="upper right")
    fig.savefig(output_dir / "ad_sodium_oo_kde.png", dpi=220)
    fig.savefig(output_dir / "ad_sodium_oo_kde.pdf")
    plt.close(fig)


def plot_ad_dimer(
    result_path: str | Path,
    output_dir: str | Path,
    *,
    burnin_fraction: float = 0.2,
) -> dict[str, Path]:
    payload = load_ad_dimer_result(result_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref = load_curated_ad_dimer_reference(ROOT / "data" / "ad_dimer_ref" / "ad_dimer_md_100ns_ref.npz")
    gg_slice = burnin_slice(len(payload["analysis_cosine_similarities"]), burnin_fraction)

    gg_obs = {
        "cosine_sim": np.asarray(payload["analysis_cosine_similarities"], dtype=np.float64)[gg_slice],
        "hbond_counts": np.asarray(payload["analysis_hbond_counts"], dtype=np.float64)[gg_slice],
        "reciprocal_counts": np.asarray(payload["analysis_reciprocal_counts"], dtype=np.float64)[gg_slice],
        "dihedrals_seg1": np.asarray(payload["analysis_dihedrals_seg1"], dtype=np.float64)[gg_slice],
        "dihedrals_seg2": np.asarray(payload["analysis_dihedrals_seg2"], dtype=np.float64)[gg_slice],
    }
    md_obs = {
        "cosine_sim": np.asarray(ref["cosine_sim"], dtype=np.float64),
        "hbond_counts": np.asarray(ref["hbond_counts"], dtype=np.float64),
        "reciprocal_counts": np.asarray(ref["reciprocal_counts"], dtype=np.float64),
        "dihedrals_seg1": np.asarray(ref["dihedrals_seg1"], dtype=np.float64),
        "dihedrals_seg2": np.asarray(ref["dihedrals_seg2"], dtype=np.float64),
    }

    gg_anti_abs = _extract_abs_psi(gg_obs, "A")
    gg_para_abs = _extract_abs_psi(gg_obs, "P")
    md_anti_abs = _extract_abs_psi(md_obs, "A")
    md_para_abs = _extract_abs_psi(md_obs, "P")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, constrained_layout=True)
    panel_specs = [
        (axes[0], "Anti-Parallel", gg_anti_abs, md_anti_abs, "#4477AA"),
        (axes[1], "Parallel", gg_para_abs, md_para_abs, "#AA3377"),
    ]
    for ax, title, gg_arr, md_arr, color in panel_specs:
        if md_arr.size >= 3:
            sns.kdeplot(md_arr, ax=ax, color=color, linewidth=1.6, linestyle="--", fill=False, label=f"Reference MD (n={md_arr.size})")
        if gg_arr.size >= 3:
            sns.kdeplot(gg_arr, ax=ax, color=color, linewidth=2.0, fill=False, label=f"GG-PA (n={gg_arr.size})")
        ax.set_title(title)
        ax.set_xlim(0.0, np.pi)
        ax.set_xticks([0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0, np.pi])
        ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
        ax.set_xlabel(r"$|\psi_1 - \psi_2|$ (rad)")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("Density")
    fig.savefig(output_dir / "ad_dimer_abs_psi_diff.png", dpi=220)
    fig.savefig(output_dir / "ad_dimer_abs_psi_diff.pdf")
    plt.close(fig)

    gg_anti_abs_phi = _extract_abs_phi(gg_obs, "A")
    gg_para_abs_phi = _extract_abs_phi(gg_obs, "P")
    md_anti_abs_phi = _extract_abs_phi(md_obs, "A")
    md_para_abs_phi = _extract_abs_phi(md_obs, "P")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, constrained_layout=True)
    phi_specs = [
        (axes[0], "Anti-Parallel", gg_anti_abs_phi, md_anti_abs_phi, "#4477AA"),
        (axes[1], "Parallel", gg_para_abs_phi, md_para_abs_phi, "#AA3377"),
    ]
    for ax, title, gg_arr, md_arr, color in phi_specs:
        if md_arr.size >= 3:
            sns.kdeplot(md_arr, ax=ax, color=color, linewidth=1.6, linestyle="--", fill=False, label=f"Reference MD (n={md_arr.size})")
        if gg_arr.size >= 3:
            sns.kdeplot(gg_arr, ax=ax, color=color, linewidth=2.0, fill=False, label=f"GG-PA (n={gg_arr.size})")
        ax.set_title(title)
        ax.set_xlim(0.0, np.pi)
        ax.set_xticks([0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0, np.pi])
        ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
        ax.set_xlabel(r"$|\phi_1 - \phi_2|$ (rad)")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("Density")
    fig.savefig(output_dir / "ad_dimer_abs_phi_diff.png", dpi=220)
    fig.savefig(output_dir / "ad_dimer_abs_phi_diff.pdf")
    plt.close(fig)
    return {
        "abs_phi_figure": output_dir / "ad_dimer_abs_phi_diff.png",
        "abs_psi_figure": output_dir / "ad_dimer_abs_psi_diff.png",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["ad-sodium", "ad-sodium-single", "ad-dimer", "all"],
        help="Which public alanine example to plot.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--result-dir", type=str, default=None, help="Override the result directory for the selected mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_root = resolve_path(cfg["shared"]["output_root"], root=ROOT)

    if args.mode in {"ad-sodium-single"}:
        base_dir = resolve_path(args.result_dir, root=ROOT) if args.result_dir else (output_root / "ad_sodium_single")
        result_path = base_dir / "ad_sodium_results.npz"
        if result_path.exists():
            plot_ad_sodium(result_path, output_root / "figures")

    if args.mode in {"ad-sodium", "all"}:
        result_dir = resolve_path(args.result_dir, root=ROOT) if args.result_dir else (output_root / "ad_sodium")
        if (result_dir / "aggregate_samples.npz").exists():
            plot_ad_sodium_ensemble(result_dir, output_root / "figures")

    if args.mode in {"ad-dimer", "all"}:
        base_dir = resolve_path(args.result_dir, root=ROOT) if args.result_dir else (output_root / "ad_dimer")
        result_path = base_dir / "ad_dimer_results.npz"
        if result_path.exists():
            plot_ad_dimer(
                result_path,
                output_root / "figures",
                burnin_fraction=float(cfg["ad_dimer"].get("burnin_fraction", 0.2)),
            )


if __name__ == "__main__":
    main()
