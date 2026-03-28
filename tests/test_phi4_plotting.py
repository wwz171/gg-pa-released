import importlib.util
from pathlib import Path

import numpy as np


def load_phi4_plot_module():
    path = Path("examples/phi4/plot_phi4_results.py").resolve()
    spec = importlib.util.spec_from_file_location("plot_phi4_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_plotting_smoke(tmp_path):
    module = load_phi4_plot_module()

    phase_path = tmp_path / "re_scan.npz"
    np.savez(
        phase_path,
        gamma=np.array([0.1, 0.2], dtype=np.float64),
        order_param=np.array([0.2, 0.8], dtype=np.float64),
        order_err=np.array([0.01, 0.02], dtype=np.float64),
        susceptibility=np.array([1.0, 2.0], dtype=np.float64),
        susceptibility_err=np.array([0.1, 0.2], dtype=np.float64),
        tau_int=np.array([2.0, 5.0], dtype=np.float64),
    )

    phase_png, phase_pdf = module.make_phase_transition_plot(phase_path, tmp_path)
    assert phase_png.exists()
    assert phase_pdf.exists()
    diag_paths = module.make_phase_transition_diagnostic_plots(
        phase_path,
        tmp_path,
        gamma_c=0.436,
    )
    assert len(diag_paths) == 3
    for png_path, pdf_path in diag_paths:
        assert png_path.exists()
        assert pdf_path.exists()

    records = [
        {
            "h": 0.005,
            "J": 0.40,
            "magnetization": 0.10,
            "mag_err": 0.01,
            "susceptibility": 8.0,
            "susceptibility_err": 0.4,
            "tau_int": 12.0,
        },
        {
            "h": 0.005,
            "J": 0.45,
            "magnetization": 0.20,
            "mag_err": 0.01,
            "susceptibility": 10.0,
            "susceptibility_err": 0.5,
            "tau_int": 18.0,
        },
        {
            "h": 0.02,
            "J": 0.36,
            "magnetization": 0.15,
            "mag_err": 0.02,
            "susceptibility": 6.0,
            "susceptibility_err": 0.3,
            "tau_int": 9.0,
        },
        {
            "h": 0.02,
            "J": 0.44,
            "magnetization": 0.25,
            "mag_err": 0.02,
            "susceptibility": 9.0,
            "susceptibility_err": 0.4,
            "tau_int": 14.0,
        },
    ]

    collapse_png, collapse_pdf = module.make_data_collapse_plot(
        records,
        tmp_path,
        gamma_c=0.436,
        beta=0.125,
        delta=15.0,
    )
    assert collapse_png.exists()
    assert collapse_pdf.exists()

    pub_paths = module.make_critical_publication_plots(
        records,
        tmp_path,
        gamma_c=0.436,
        beta=0.125,
        delta=15.0,
        exclude_smallest_h=False,
    )
    assert len(pub_paths) == 3
    for png_path, pdf_path in pub_paths:
        assert png_path.exists()
        assert pdf_path.exists()

    combined_png, combined_pdf = module.make_critical_publication_combined_plot(
        records,
        tmp_path,
        gamma_c=0.436,
        beta=0.125,
        delta=15.0,
        exclude_smallest_h=False,
    )
    assert combined_png.exists()
    assert combined_pdf.exists()

    tau_png, tau_pdf = module.make_critical_tau_vs_gamma_plot(
        records,
        tmp_path,
        gamma_c=0.436,
        exclude_smallest_h=False,
    )
    assert tau_png.exists()
    assert tau_pdf.exists()

    iat_png, iat_pdf = module.make_critical_iat_plot(
        records,
        tmp_path,
        gamma_c=0.436,
        beta=0.125,
        delta=15.0,
    )
    assert iat_png.exists()
    assert iat_pdf.exists()
