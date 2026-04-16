import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_phi4_example_module():
    path = Path("examples/phi4/phi4_re_scan.py").resolve()
    spec = importlib.util.spec_from_file_location("phi4_re_scan", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase_transition_grid_matches_legacy_layout():
    torch = pytest.importorskip("torch")
    _ = torch  # imported for module availability
    module = load_phi4_example_module()

    cfg = module.load_config("configs/phi4_example.yaml")
    grid = module.build_phase_transition_gamma_grid(cfg["phase_transition"])
    assert len(grid) == 30
    assert np.isclose(grid[0], 0.05)
    assert np.isclose(grid[-1], 0.95)
    assert np.isclose(grid[14], 0.43571428571428567)


def test_scaling_window_grid_matches_formula():
    torch = pytest.importorskip("torch")
    _ = torch
    module = load_phi4_example_module()

    grid = module.build_scaling_window_gamma_grid(0.005)
    assert len(grid) == 10
    assert np.isclose(grid[0], 0.3174743651988184)
    assert np.isclose(grid[-1], 0.5545256348011816)


def test_phase_point_n_sweeps_uses_critical_window_override():
    torch = pytest.importorskip("torch")
    _ = torch
    module = load_phi4_example_module()

    section = {
        "n_sweeps": 10_000,
        "critical_window": {
            "start": 0.40,
            "stop": 0.48,
            "n_sweeps": 30_000,
        },
    }
    assert module.phase_point_n_sweeps(section, 0.39) == 10_000
    assert module.phase_point_n_sweeps(section, 0.40) == 30_000
    assert module.phase_point_n_sweeps(section, 0.4357142857) == 30_000
    assert module.phase_point_n_sweeps(section, 0.48) == 30_000
    assert module.phase_point_n_sweeps(section, 0.49) == 10_000


def test_observables_switch_between_zero_and_nonzero_field_conventions():
    torch = pytest.importorskip("torch")
    _ = torch
    module = load_phi4_example_module()

    series = np.array([-1.0, 0.0, 1.0, 1.0], dtype=np.float64)
    h0 = module.compute_observables_from_history(series, L=4, h=0.0)
    hp = module.compute_observables_from_history(series, L=4, h=0.02)

    assert np.isclose(h0["order_param"], 0.75)
    assert np.isclose(h0["magnetization"], 0.25)
    assert np.isclose(hp["magnetization"], 0.25)
    assert np.isclose(hp["order_param"], 0.25)
    assert h0["susceptibility"] != hp["susceptibility"]


def test_run_single_point_smoke():
    torch = pytest.importorskip("torch")
    module = load_phi4_example_module()

    diffusion, n_params = module.load_diffusion_model(
        Path("checkpoints/phi4_prior.pt"),
        device=torch.device("cpu"),
        compile_model=False,
    )
    record = module.run_single_point(
        diffusion,
        J=0.2,
        h=0.0,
        L=4,
        t_diff_ladder=module.build_t_diff_ladder(0.1, 0.2, 3),
        n_sweeps=2,
        record_interval=1,
        burnin_fraction=0.5,
        seed=7,
        device=torch.device("cpu"),
        verbose=False,
        log_every=1,
        save_all_replicas=True,
    )

    assert n_params == 3569
    assert record["mag_replica_0"].shape == (2,)
    assert record["magnetizations"].shape == (2, 3)
    assert record["phi_final_replicas"].shape == (3, 4, 4)
    assert record["psi_final_replicas"].shape == (3, 4, 4)
    assert np.isfinite(record["order_param"])
    assert np.isfinite(record["susceptibility"])


def test_run_single_point_is_seed_reproducible_for_nonzero_field():
    torch = pytest.importorskip("torch")
    module = load_phi4_example_module()

    diffusion, _ = module.load_diffusion_model(
        Path("checkpoints/phi4_prior.pt"),
        device=torch.device("cpu"),
        compile_model=False,
    )
    kwargs = dict(
        diffusion=diffusion,
        J=0.45,
        h=0.01,
        L=4,
        t_diff_ladder=module.build_t_diff_ladder(0.1, 0.2, 3),
        n_sweeps=2,
        record_interval=1,
        burnin_fraction=0.5,
        seed=11,
        device=torch.device("cpu"),
        verbose=False,
        log_every=1,
        save_all_replicas=True,
    )
    a = module.run_single_point(**kwargs)
    b = module.run_single_point(**kwargs)

    assert np.allclose(a["magnetizations"], b["magnetizations"])
    assert np.allclose(a["phi_final_replicas"], b["phi_final_replicas"])
    assert np.allclose(a["psi_final_replicas"], b["psi_final_replicas"])


def test_phase_transition_progress_files_are_written(tmp_path):
    torch = pytest.importorskip("torch")
    module = load_phi4_example_module()

    diffusion, _ = module.load_diffusion_model(
        Path("checkpoints/phi4_prior.pt"),
        device=torch.device("cpu"),
        compile_model=False,
    )
    record = module.run_single_point(
        diffusion,
        J=0.2,
        h=0.0,
        L=4,
        t_diff_ladder=module.build_t_diff_ladder(0.1, 0.2, 3),
        n_sweeps=2,
        record_interval=1,
        burnin_fraction=0.5,
        seed=7,
        device=torch.device("cpu"),
        verbose=False,
        log_every=1,
        save_all_replicas=True,
    )

    point_path = module.save_phase_transition_point(record, tmp_path)
    progress_path = module.write_phase_transition_progress(
        [record],
        tmp_path,
        total_points=3,
        checkpoint=Path("checkpoints/phi4_prior.pt"),
        t_prod=0.1,
        t_max=0.2,
        n_params=3569,
    )

    assert point_path.exists()
    assert progress_path.exists()
    progress = progress_path.read_text(encoding="utf-8")
    assert '"completed_points": 1' in progress
    assert '"remaining_points": 2' in progress


def test_save_phase_transition_scan_handles_variable_sweeps(tmp_path):
    torch = pytest.importorskip("torch")
    module = load_phi4_example_module()

    diffusion, _ = module.load_diffusion_model(
        Path("checkpoints/phi4_prior.pt"),
        device=torch.device("cpu"),
        compile_model=False,
    )
    rec_a = module.run_single_point(
        diffusion,
        J=0.20,
        h=0.0,
        L=4,
        t_diff_ladder=module.build_t_diff_ladder(0.1, 0.2, 3),
        n_sweeps=2,
        record_interval=1,
        burnin_fraction=0.5,
        seed=7,
        device=torch.device("cpu"),
        verbose=False,
        log_every=1,
        save_all_replicas=True,
    )
    rec_b = module.run_single_point(
        diffusion,
        J=0.22,
        h=0.0,
        L=4,
        t_diff_ladder=module.build_t_diff_ladder(0.1, 0.2, 3),
        n_sweeps=4,
        record_interval=1,
        burnin_fraction=0.5,
        seed=8,
        device=torch.device("cpu"),
        verbose=False,
        log_every=1,
        save_all_replicas=True,
    )

    npz_path, json_path = module.save_phase_transition_scan(
        [rec_a, rec_b],
        tmp_path,
        checkpoint=Path("checkpoints/phi4_prior.pt"),
        t_prod=0.1,
        t_max=0.2,
        n_params=3569,
    )

    assert npz_path.exists()
    assert json_path.exists()
    data = np.load(npz_path)
    assert data["re_sweeps_by_gamma"].tolist() == [2, 4]
    assert data["magnetization_series"].shape == (2, 4)
    assert data["magnetizations_all"].shape == (2, 4, 3)
