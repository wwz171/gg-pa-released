#!/usr/bin/env python3
"""Plot phi4 paper figures from raw results generated under examples/phi4."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from phi4_re_scan import ROOT, load_config


def load_phase_transition_scan(phase_dir: Path) -> dict[str, np.ndarray]:
    path = phase_dir / "re_scan.npz"
    if not path.exists():
        raise FileNotFoundError(f"Phase-transition raw file not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def load_critical_scaling_records(critical_dir: Path) -> list[dict[str, float]]:
    summary_path = critical_dir / "critical_scaling_summary.npz"
    if summary_path.exists():
        data = np.load(summary_path, allow_pickle=True)
        gamma_c = float(data["gamma_c"])
        records: list[dict[str, float]] = []
        for h, gamma, m, m_err, chi, chi_err, tau_int, swap in zip(
            np.asarray(data["h"], dtype=np.float64),
            np.asarray(data["gamma"], dtype=np.float64),
            np.asarray(data["magnetization"], dtype=np.float64),
            np.asarray(data["mag_err"], dtype=np.float64),
            np.asarray(data["susceptibility"], dtype=np.float64),
            np.asarray(data["susceptibility_err"], dtype=np.float64),
            np.asarray(data["tau_int"], dtype=np.float64),
            np.asarray(data["mean_swap_rate"], dtype=np.float64),
        ):
            records.append(
                {
                    "h": float(h),
                    "J": float(gamma),
                    "magnetization": float(m),
                    "mag_err": float(m_err),
                    "susceptibility": float(chi),
                    "susceptibility_err": float(chi_err),
                    "tau_int": float(tau_int),
                    "mean_swap_rate": float(swap),
                    "gamma_c": gamma_c,
                }
            )
        return records

    records = []
    for path in sorted(critical_dir.glob("cs_*.npz")):
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
    if not records:
        raise FileNotFoundError(f"No critical-scaling point files found under {critical_dir}")
    return records


def filter_critical_scaling_records(
    records: list[dict[str, float]],
    *,
    h_values: list[float] | tuple[float, ...] | None,
) -> list[dict[str, float]]:
    if not h_values:
        return records
    allowed = {round(float(h), 12) for h in h_values}
    filtered = [rec for rec in records if round(float(rec["h"]), 12) in allowed]
    if not filtered:
        raise ValueError(
            "No critical-scaling records matched the configured h_values: "
            + ", ".join(f"{float(h):.6f}" for h in h_values)
        )
    return filtered


def make_phase_transition_plot(data_path: Path, output_dir: Path) -> tuple[Path, Path]:
    data = np.load(data_path, allow_pickle=True)
    gamma = np.asarray(data["gamma"], dtype=np.float64)
    order_param = np.asarray(data["order_param"], dtype=np.float64)
    order_err = np.asarray(data["order_err"], dtype=np.float64)
    chi = np.asarray(data["susceptibility"], dtype=np.float64)
    chi_err = np.asarray(data["susceptibility_err"], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].errorbar(gamma, order_param, yerr=order_err, fmt="o-", lw=1.2, ms=3.5)
    axes[0].set_xlabel(r"$J$")
    axes[0].set_ylabel(r"$\langle |m| \rangle$")
    axes[0].set_title("Zero-field phase transition")
    axes[0].grid(alpha=0.2)

    axes[1].errorbar(gamma, chi, yerr=chi_err, fmt="o-", lw=1.2, ms=3.5, color="C3")
    axes[1].set_xlabel(r"$J$")
    axes[1].set_ylabel(r"$\chi$")
    axes[1].set_title("Zero-field susceptibility")
    axes[1].grid(alpha=0.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "phase_transition_curve.png"
    pdf_path = output_dir / "phase_transition_curve.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _save_single_curve_figure(
    x: np.ndarray,
    y: np.ndarray,
    *,
    yerr: np.ndarray | None,
    xlabel: str,
    ylabel: str,
    title: str,
    basename: str,
    output_dir: Path,
    color: str = "C0",
    gamma_c: float | None = None,
    peak_x: float | None = None,
    peak_label: str | None = None,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    if yerr is None:
        ax.plot(x, y, "o-", lw=1.4, ms=4.0, color=color)
    else:
        ax.errorbar(x, y, yerr=yerr, fmt="o-", lw=1.4, ms=4.0, color=color)

    if gamma_c is not None:
        ax.axvline(float(gamma_c), color="0.35", ls="--", lw=1.0, label=fr"$J_c={gamma_c:.3f}$")
    if peak_x is not None:
        label = peak_label or fr"peak={peak_x:.3f}"
        ax.axvline(float(peak_x), color=color, ls=":", lw=1.2, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    if gamma_c is not None or peak_x is not None:
        ax.legend(frameon=False, fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def make_phase_transition_diagnostic_plots(
    data_path: Path,
    output_dir: Path,
    *,
    gamma_c: float | None = None,
) -> list[tuple[Path, Path]]:
    data = np.load(data_path, allow_pickle=True)
    gamma = np.asarray(data["gamma"], dtype=np.float64)
    order_param = np.asarray(data["order_param"], dtype=np.float64)
    order_err = np.asarray(data["order_err"], dtype=np.float64)
    chi = np.asarray(data["susceptibility"], dtype=np.float64)
    chi_err = np.asarray(data["susceptibility_err"], dtype=np.float64)
    tau_int = np.asarray(data["tau_int"], dtype=np.float64)

    chi_peak = float(gamma[np.argmax(chi)])
    tau_peak = float(gamma[np.argmax(tau_int)])

    outputs = [
        _save_single_curve_figure(
            gamma,
            order_param,
            yerr=order_err,
            xlabel=r"$J$",
            ylabel=r"$\langle |m| \rangle$",
            title="Zero-field order parameter",
            basename="phase_transition_order_parameter",
            output_dir=output_dir,
            color="C0",
            gamma_c=gamma_c,
            peak_x=chi_peak,
            peak_label=fr"$J_{{\chi,\max}}={chi_peak:.3f}$",
        ),
        _save_single_curve_figure(
            gamma,
            chi,
            yerr=chi_err,
            xlabel=r"$J$",
            ylabel=r"$\chi$",
            title="Zero-field susceptibility",
            basename="phase_transition_susceptibility",
            output_dir=output_dir,
            color="C3",
            gamma_c=gamma_c,
            peak_x=chi_peak,
            peak_label=fr"$J_{{\chi,\max}}={chi_peak:.3f}$",
        ),
        _save_single_curve_figure(
            gamma,
            tau_int,
            yerr=None,
            xlabel=r"$J$",
            ylabel=r"$\tau_{\mathrm{int}}$",
            title="Integrated autocorrelation time",
            basename="phase_transition_iat",
            output_dir=output_dir,
            color="C2",
            gamma_c=gamma_c,
            peak_x=tau_peak,
            peak_label=fr"$J_{{\tau,\max}}={tau_peak:.3f}$",
        ),
    ]
    return outputs


def make_data_collapse_plot(
    records: list[dict[str, float]],
    output_dir: Path,
    *,
    gamma_c: float,
    beta: float,
    delta: float,
) -> tuple[Path, Path]:
    grouped: dict[float, list[dict[str, float]]] = {}
    for rec in records:
        grouped.setdefault(float(rec["h"]), []).append(rec)

    inv_beta_delta = 1.0 / (beta * delta)
    one_over_delta = 1.0 / delta
    chi_exp = (delta - 1.0) / delta

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for h, recs in sorted(grouped.items()):
        recs = sorted(recs, key=lambda item: item["J"])
        x = np.array([(r["J"] - gamma_c) / (h ** inv_beta_delta) for r in recs], dtype=np.float64)
        scaled_m = np.array([r["magnetization"] / (h ** one_over_delta) for r in recs], dtype=np.float64)
        scaled_m_err = np.array([r["mag_err"] / (h ** one_over_delta) for r in recs], dtype=np.float64)
        scaled_chi = np.array([r["susceptibility"] * (h ** chi_exp) for r in recs], dtype=np.float64)
        scaled_chi_err = np.array(
            [r["susceptibility_err"] * (h ** chi_exp) for r in recs],
            dtype=np.float64,
        )

        axes[0].errorbar(x, scaled_m, yerr=scaled_m_err, fmt="o-", ms=3.0, lw=1.1, label=fr"$h={h:.3f}$")
        axes[1].errorbar(
            x,
            scaled_chi,
            yerr=scaled_chi_err,
            fmt="o-",
            ms=3.0,
            lw=1.1,
            label=fr"$h={h:.3f}$",
        )

    axes[0].set_xlabel(r"$(J - J_c) / h^{1/(\beta \delta)}$")
    axes[0].set_ylabel(r"$m / h^{1/\delta}$")
    axes[0].set_title("Magnetization data collapse")
    axes[0].grid(alpha=0.2)

    axes[1].set_xlabel(r"$(J - J_c) / h^{1/(\beta \delta)}$")
    axes[1].set_ylabel(r"$\chi \, h^{(\delta - 1)/\delta}$")
    axes[1].set_title("Susceptibility data collapse")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "data_collapse.png"
    pdf_path = output_dir / "data_collapse.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _critical_publication_metadata(
    records: list[dict[str, float]],
    *,
    exclude_smallest_h: bool = False,
) -> tuple[list[float], dict[float, str], dict[float, str], dict[str, Any]]:
    plot_h = sorted({float(rec["h"]) for rec in records})
    if exclude_smallest_h and len(plot_h) >= 4:
        plot_h = plot_h[1:]

    colors = ["#4477AA", "#228833", "#AA3377", "#EE6677", "#CCBB44"]
    markers = ["o", "s", "^", "D", "v"]
    palette = {h: colors[idx % len(colors)] for idx, h in enumerate(plot_h)}
    marker_map = {h: markers[idx % len(markers)] for idx, h in enumerate(plot_h)}
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 6.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.default": "regular",
    }
    return plot_h, palette, marker_map, style


def make_critical_publication_plots(
    records: list[dict[str, float]],
    output_dir: Path,
    *,
    gamma_c: float,
    beta: float,
    delta: float,
    exclude_smallest_h: bool = False,
) -> list[tuple[Path, Path]]:
    plot_h, palette, marker_map, style = _critical_publication_metadata(
        records,
        exclude_smallest_h=exclude_smallest_h,
    )
    grouped: dict[float, list[dict[str, float]]] = {}
    for rec in records:
        grouped.setdefault(float(rec["h"]), []).append(rec)

    common_kw = {
        "capsize": 1.5,
        "ms": 4,
        "lw": 0.9,
        "markeredgewidth": 0.4,
        "markeredgecolor": "k",
        "elinewidth": 0.5,
    }
    fig_size = (3.6, 2.8)
    beta_delta = beta * delta
    chi_scale_exp = (delta - 1.0) / delta

    outputs: list[tuple[Path, Path]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=fig_size)
        for hv in plot_h:
            recs = sorted(grouped[hv], key=lambda item: item["J"])
            gamma = np.array([r["J"] for r in recs], dtype=np.float64)
            mag = np.array([r["magnetization"] for r in recs], dtype=np.float64)
            mag_err = np.array([r["mag_err"] for r in recs], dtype=np.float64)
            ax.errorbar(
                gamma,
                mag,
                yerr=mag_err,
                fmt=marker_map[hv] + "-",
                color=palette[hv],
                label=fr"$h = {hv:.3f}$",
                **common_kw,
            )
        ax.axvline(gamma_c, ls="--", color="0.4", lw=0.5, zorder=0)
        ax.set_xlabel(r"Coupling $\gamma$")
        ax.set_ylabel(r"Magnetization $\langle M \rangle$")
        ax.legend(loc="upper left", handlelength=1.8, columnspacing=1.0)
        ax.tick_params(which="both", top=True, right=True)
        fig.tight_layout(pad=0.3)
        png_path = output_dir / "critical_raw_M.png"
        pdf_path = output_dir / "critical_raw_M.pdf"
        fig.savefig(png_path)
        fig.savefig(pdf_path)
        plt.close(fig)
        outputs.append((png_path, pdf_path))

        fig, ax = plt.subplots(figsize=fig_size)
        for hv in plot_h:
            recs = sorted(grouped[hv], key=lambda item: item["J"])
            gamma = np.array([r["J"] for r in recs], dtype=np.float64)
            mag = np.array([r["magnetization"] for r in recs], dtype=np.float64)
            mag_err = np.array([r["mag_err"] for r in recs], dtype=np.float64)
            x_sc = (gamma - gamma_c) / (hv ** (1.0 / beta_delta))
            y_sc = mag / (hv ** (1.0 / delta))
            y_err = mag_err / (hv ** (1.0 / delta))
            idx = np.argsort(x_sc)
            ax.errorbar(
                x_sc[idx],
                y_sc[idx],
                yerr=y_err[idx],
                fmt=marker_map[hv] + "-",
                color=palette[hv],
                label=fr"$h = {hv:.3f}$",
                **common_kw,
            )
        ax.axvline(0.0, ls="--", color="0.5", lw=0.4, zorder=0)
        ax.set_xlabel(r"$(\gamma - \gamma_c)\, /\, h^{8/15}$")
        ax.set_ylabel(r"$M \, / \, h^{1/15}$")
        ax.legend(loc="upper left", handlelength=1.8)
        ax.tick_params(which="both", top=True, right=True)
        fig.tight_layout(pad=0.3)
        png_path = output_dir / "critical_M_collapse.png"
        pdf_path = output_dir / "critical_M_collapse.pdf"
        fig.savefig(png_path)
        fig.savefig(pdf_path)
        plt.close(fig)
        outputs.append((png_path, pdf_path))

        fig, ax = plt.subplots(figsize=fig_size)
        for hv in plot_h:
            recs = sorted(grouped[hv], key=lambda item: item["J"])
            gamma = np.array([r["J"] for r in recs], dtype=np.float64)
            chi = np.array([r["susceptibility"] for r in recs], dtype=np.float64)
            chi_err = np.array([r["susceptibility_err"] for r in recs], dtype=np.float64)
            x_sc = (gamma - gamma_c) / (hv ** (1.0 / beta_delta))
            y_sc = chi * (hv ** chi_scale_exp)
            y_err = chi_err * (hv ** chi_scale_exp)
            idx = np.argsort(x_sc)
            ax.errorbar(
                x_sc[idx],
                y_sc[idx],
                yerr=y_err[idx],
                fmt=marker_map[hv] + "-",
                color=palette[hv],
                label=fr"$h = {hv:.3f}$",
                **common_kw,
            )
        ax.axvline(0.0, ls="--", color="0.5", lw=0.4, zorder=0)
        ax.set_xlabel(r"$(\gamma - \gamma_c)\, /\, h^{8/15}$")
        ax.set_ylabel(r"$\chi \cdot h^{14/15}$")
        ax.legend(loc="upper left", handlelength=1.8)
        ax.tick_params(which="both", top=True, right=True)
        fig.tight_layout(pad=0.3)
        png_path = output_dir / "critical_chi_collapse.png"
        pdf_path = output_dir / "critical_chi_collapse.pdf"
        fig.savefig(png_path)
        fig.savefig(pdf_path)
        plt.close(fig)
        outputs.append((png_path, pdf_path))

    return outputs


def make_critical_publication_combined_plot(
    records: list[dict[str, float]],
    output_dir: Path,
    *,
    gamma_c: float,
    beta: float,
    delta: float,
    exclude_smallest_h: bool = False,
) -> tuple[Path, Path]:
    plot_h, palette, marker_map, style = _critical_publication_metadata(
        records,
        exclude_smallest_h=exclude_smallest_h,
    )
    grouped: dict[float, list[dict[str, float]]] = {}
    for rec in records:
        grouped.setdefault(float(rec["h"]), []).append(rec)

    common_kw = {
        "capsize": 1.5,
        "ms": 3.8,
        "lw": 0.9,
        "markeredgewidth": 0.35,
        "markeredgecolor": "k",
        "elinewidth": 0.45,
    }
    beta_delta = beta * delta
    chi_scale_exp = (delta - 1.0) / delta
    output_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.8), constrained_layout=True)

        for hv in plot_h:
            recs = sorted(grouped[hv], key=lambda item: item["J"])
            gamma = np.array([r["J"] for r in recs], dtype=np.float64)
            mag = np.array([r["magnetization"] for r in recs], dtype=np.float64)
            mag_err = np.array([r["mag_err"] for r in recs], dtype=np.float64)
            chi = np.array([r["susceptibility"] for r in recs], dtype=np.float64)
            chi_err = np.array([r["susceptibility_err"] for r in recs], dtype=np.float64)
            x_sc = (gamma - gamma_c) / (hv ** (1.0 / beta_delta))

            axes[0].errorbar(
                gamma,
                mag,
                yerr=mag_err,
                fmt=marker_map[hv] + "-",
                color=palette[hv],
                label=fr"$h = {hv:.3f}$",
                **common_kw,
            )
            axes[1].errorbar(
                np.sort(x_sc),
                (mag / (hv ** (1.0 / delta)))[np.argsort(x_sc)],
                yerr=(mag_err / (hv ** (1.0 / delta)))[np.argsort(x_sc)],
                fmt=marker_map[hv] + "-",
                color=palette[hv],
                **common_kw,
            )
            axes[2].errorbar(
                np.sort(x_sc),
                (chi * (hv ** chi_scale_exp))[np.argsort(x_sc)],
                yerr=(chi_err * (hv ** chi_scale_exp))[np.argsort(x_sc)],
                fmt=marker_map[hv] + "-",
                color=palette[hv],
                **common_kw,
            )

        axes[0].axvline(gamma_c, ls="--", color="0.4", lw=0.5, zorder=0)
        axes[0].set_xlabel(r"Coupling $\gamma$")
        axes[0].set_ylabel(r"Magnetization $\langle M \rangle$")
        axes[0].tick_params(which="both", top=True, right=True)
        axes[0].legend(loc="upper left", handlelength=1.6, columnspacing=0.8)

        axes[1].axvline(0.0, ls="--", color="0.5", lw=0.4, zorder=0)
        axes[1].set_xlabel(r"$(\gamma - \gamma_c)\, /\, h^{8/15}$")
        axes[1].set_ylabel(r"$M \, / \, h^{1/15}$")
        axes[1].tick_params(which="both", top=True, right=True)

        axes[2].axvline(0.0, ls="--", color="0.5", lw=0.4, zorder=0)
        axes[2].set_xlabel(r"$(\gamma - \gamma_c)\, /\, h^{8/15}$")
        axes[2].set_ylabel(r"$\chi \cdot h^{14/15}$")
        axes[2].tick_params(which="both", top=True, right=True)

        for ax in axes:
            ax.grid(alpha=0.15)

        png_path = output_dir / "critical_combined_1x3.png"
        pdf_path = output_dir / "critical_combined_1x3.pdf"
        fig.savefig(png_path)
        fig.savefig(pdf_path)
        plt.close(fig)
    return png_path, pdf_path


def make_critical_tau_vs_gamma_plot(
    records: list[dict[str, float]],
    output_dir: Path,
    *,
    gamma_c: float,
    exclude_smallest_h: bool = False,
) -> tuple[Path, Path]:
    plot_h, palette, marker_map, style = _critical_publication_metadata(
        records,
        exclude_smallest_h=exclude_smallest_h,
    )
    grouped: dict[float, list[dict[str, float]]] = {}
    for rec in records:
        grouped.setdefault(float(rec["h"]), []).append(rec)

    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.8, 2.9), constrained_layout=True)
        for hv in plot_h:
            recs = sorted(grouped[hv], key=lambda item: item["J"])
            gamma = np.array([r["J"] for r in recs], dtype=np.float64)
            tau = np.array([r["tau_int"] for r in recs], dtype=np.float64)
            ax.plot(
                gamma,
                tau,
                marker_map[hv] + "-",
                color=palette[hv],
                lw=0.9,
                ms=3.8,
                markeredgewidth=0.35,
                markeredgecolor="k",
                label=fr"$h = {hv:.3f}$",
            )
        ax.axvline(gamma_c, ls="--", color="0.4", lw=0.5, zorder=0)
        ax.set_xlabel(r"Coupling $\gamma$")
        ax.set_ylabel(r"$\tau_{\mathrm{int}}$")
        ax.set_title("IAT across the scaling window")
        ax.legend(loc="upper left", handlelength=1.6, columnspacing=0.8)
        ax.tick_params(which="both", top=True, right=True)
        ax.grid(alpha=0.15)
        png_path = output_dir / "critical_tau_vs_gamma.png"
        pdf_path = output_dir / "critical_tau_vs_gamma.pdf"
        fig.savefig(png_path)
        fig.savefig(pdf_path)
        plt.close(fig)
    return png_path, pdf_path


def make_critical_iat_plot(
    records: list[dict[str, float]],
    output_dir: Path,
    *,
    gamma_c: float,
    beta: float,
    delta: float,
) -> tuple[Path, Path]:
    grouped: dict[float, list[dict[str, float]]] = {}
    for rec in records:
        grouped.setdefault(float(rec["h"]), []).append(rec)

    inv_beta_delta = 1.0 / (beta * delta)

    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    for h, recs in sorted(grouped.items()):
        recs = sorted(recs, key=lambda item: item["J"])
        x = np.array([(r["J"] - gamma_c) / (h ** inv_beta_delta) for r in recs], dtype=np.float64)
        tau = np.array([r["tau_int"] for r in recs], dtype=np.float64)
        ax.plot(x, tau, "o-", ms=3.5, lw=1.2, label=fr"$h={h:.3f}$")

    ax.axvline(0.0, color="0.35", ls="--", lw=1.0, label=r"$x=0$")
    ax.set_xlabel(r"$(J - J_c) / h^{1/(\beta \delta)}$")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$")
    ax.set_title("Integrated autocorrelation across scaling window")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "critical_scaling_iat.png"
    pdf_path = output_dir / "critical_scaling_iat.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phi4_example.yaml")
    parser.add_argument("--phase-dir", type=Path, default=None)
    parser.add_argument("--critical-dir", type=Path, default=None)
    parser.add_argument("--fig-dir", type=Path, default=None)
    parser.add_argument("--gamma-c", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument(
        "command",
        choices=["phase-transition", "critical-scaling", "all"],
        default="all",
        nargs="?",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    phase_dir = args.phase_dir or cfg["phase_transition"]["output_dir"]
    critical_dir = args.critical_dir or cfg["critical_scaling"]["output_dir"]
    fig_dir = args.fig_dir or cfg["plotting"]["output_dir"]
    gamma_c = float(args.gamma_c if args.gamma_c is not None else cfg["plotting"]["gamma_c"])
    beta = float(args.beta if args.beta is not None else cfg["plotting"]["beta"])
    delta = float(args.delta if args.delta is not None else cfg["plotting"]["delta"])

    if args.command in {"phase-transition", "all"}:
        phase_png, phase_pdf = make_phase_transition_plot(Path(phase_dir) / "re_scan.npz", Path(fig_dir))
        print(f"Saved phase-transition figure: {phase_png}")
        print(f"Saved phase-transition figure: {phase_pdf}")
        diagnostic_paths = make_phase_transition_diagnostic_plots(
            Path(phase_dir) / "re_scan.npz",
            Path(fig_dir),
            gamma_c=gamma_c,
        )
        for png_path, pdf_path in diagnostic_paths:
            print(f"Saved phase-transition diagnostic: {png_path}")
            print(f"Saved phase-transition diagnostic: {pdf_path}")

    if args.command in {"critical-scaling", "all"}:
        records = load_critical_scaling_records(Path(critical_dir))
        records = filter_critical_scaling_records(
            records,
            h_values=cfg["critical_scaling"].get("h_values"),
        )
        publication_paths = make_critical_publication_plots(
            records,
            Path(fig_dir),
            gamma_c=gamma_c,
            beta=beta,
            delta=delta,
        )
        for png_path, pdf_path in publication_paths:
            print(f"Saved critical publication figure: {png_path}")
            print(f"Saved critical publication figure: {pdf_path}")
        combined_png, combined_pdf = make_critical_publication_combined_plot(
            records,
            Path(fig_dir),
            gamma_c=gamma_c,
            beta=beta,
            delta=delta,
        )
        print(f"Saved critical combined figure: {combined_png}")
        print(f"Saved critical combined figure: {combined_pdf}")
        collapse_png, collapse_pdf = make_data_collapse_plot(
            records,
            Path(fig_dir),
            gamma_c=gamma_c,
            beta=beta,
            delta=delta,
        )
        print(f"Saved data-collapse figure: {collapse_png}")
        print(f"Saved data-collapse figure: {collapse_pdf}")
        tau_png, tau_pdf = make_critical_tau_vs_gamma_plot(
            records,
            Path(fig_dir),
            gamma_c=gamma_c,
        )
        print(f"Saved critical tau-vs-gamma figure: {tau_png}")
        print(f"Saved critical tau-vs-gamma figure: {tau_pdf}")
        iat_png, iat_pdf = make_critical_iat_plot(
            records,
            Path(fig_dir),
            gamma_c=gamma_c,
            beta=beta,
            delta=delta,
        )
        print(f"Saved critical-scaling IAT figure: {iat_png}")
        print(f"Saved critical-scaling IAT figure: {iat_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
