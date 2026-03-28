import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest


def _load_example_module(relpath: str, name: str):
    path = Path(relpath).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_torsion_checkpoint_alias_loads():
    torch = pytest.importorskip("torch")
    _ = torch

    from ggpa.systems.alanine_dipeptide import TorsionDiffusion

    diffusion = TorsionDiffusion.load_from_checkpoint("checkpoints/ad_torsion_prior.pt", device="cpu")
    assert diffusion.forward_process.sigma_min == pytest.approx(0.1)
    assert diffusion.forward_process.sigma_max == pytest.approx(3.0)


def test_monomer_and_dimer_torsion_extractors():
    pytest.importorskip("mdtraj")

    from ggpa.systems.alanine_dipeptide import (
        extract_dimer_torsion_indices,
        extract_monomer_torsion_indices,
    )

    mono = extract_monomer_torsion_indices("data/adp_monomer_vacuum.pdb")
    dimer = extract_dimer_torsion_indices("data/adp_dimer_vacuum.pdb")

    assert mono["all"].shape == (2, 4)
    assert len(mono["torsion_atom_list"]) >= 4
    assert dimer["all"].shape == (4, 4)
    assert len(dimer["chain_atom_lists"]) == 2


def test_alanine_public_config_defaults():
    yaml = pytest.importorskip("yaml")
    _ = yaml

    import yaml as _yaml

    cfg = _yaml.safe_load(Path("configs/alanine_dipeptide_example.yaml").read_text())
    assert cfg["shared"]["checkpoint"] == "checkpoints/ad_torsion_prior.pt"
    assert cfg["ad_sodium"]["tau"] == pytest.approx(0.1)
    assert cfg["ad_sodium"]["n_steps"] == 5000
    assert cfg["ad_sodium"]["save_dcd"] is True
    assert cfg["ad_sodium_ensemble"]["n_trajectories"] == 5
    assert cfg["ad_sodium_ensemble"]["burnin_fraction"] == pytest.approx(0.2)
    assert cfg["ad_dimer"]["tau_list"] == pytest.approx([0.1, 0.15, 0.25, 0.4])
    assert cfg["ad_dimer"]["n_blocks"] == 1000
    assert cfg["ad_dimer"]["burnin_fraction"] == pytest.approx(0.2)
    assert cfg["ad_dimer"]["save_dcd"] is True
    assert cfg["ad_dimer"]["centering_schedule"] == "always"
    assert cfg["ad_dimer"]["centering_force_k"] == pytest.approx(50.0)
    assert cfg["ad_dimer"]["centering_d0"] == pytest.approx(0.9)


def test_dimer_state_classification_uses_reference_driven_lr_basins():
    from ggpa.systems.alanine_dipeptide import classify_dimer_monomer_basin, classify_states

    left = np.array([-2.4234, 2.6856], dtype=np.float64)
    right = np.array([-1.4261, 1.1006], dtype=np.float64)
    unknown = np.array([2.4, -2.4], dtype=np.float64)
    dihedrals = np.vstack([left, right, unknown])

    basin = classify_dimer_monomer_basin(dihedrals)
    assert basin.tolist() == ["L", "R", "U"]

    res = {
        "cosine_similarities": np.array([-0.96, 0.96, 0.97, 0.98], dtype=np.float64),
        "hbond_counts": np.array([2, 2, 2, 2], dtype=np.float64),
        "reciprocal_counts": np.array([1, 1, 1, 1], dtype=np.float64),
        "dihedrals_seg1": np.vstack([left, left, right, unknown]),
        "dihedrals_seg2": np.vstack([left, right, left, left]),
    }
    labels = classify_states(res)
    assert labels.tolist() == ["Anti-LL", "Para-LR", "Para-RL", "Bound-Other"]


@pytest.mark.parametrize(
    "script_path",
    [
        "examples/alanine_dipeptide/run_ad_sodium.sh",
        "examples/alanine_dipeptide/run_ad_dimer.sh",
    ],
)
def test_alanine_shell_launchers_have_valid_bash_syntax(script_path):
    subprocess.run(["bash", "-n", script_path], check=True)


def test_ad_sodium_ensemble_smoke(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("openmm")
    pytest.importorskip("mdtraj")

    run_mod = _load_example_module(
        "examples/alanine_dipeptide/run_ad_sodium.py",
        "run_ad_sodium_example",
    )

    cfg = run_mod.load_config()
    cfg["shared"]["device"] = "cpu"
    cfg["shared"]["platform_name"] = "CPU"
    cfg["shared"]["output_root"] = tmp_path
    cfg["ad_sodium"]["md_steps"] = 1
    cfg["ad_sodium"]["n_steps"] = 2
    cfg["ad_sodium"]["save_dcd"] = False
    cfg["ad_sodium_ensemble"]["output_dir"] = tmp_path / "ad_sodium"
    cfg["ad_sodium_ensemble"]["seeds"] = [42]

    result = run_mod.run_ad_sodium_ensemble(
        cfg,
        n_steps_override=2,
        n_trajectories_override=1,
        save_dcd_override=False,
        make_plots=True,
    )

    assert result["aggregate_path"].exists()
    assert result["summary_path"].exists()
    payload = np.load(result["aggregate_path"])
    assert payload["n_trajectories"].item() == 1
    assert payload["n_steps"].item() == 2
    assert (result["figure_dir"] / "ad_sodium_rama_triptych.png").exists()
    assert (result["figure_dir"] / "ad_sodium_oo_kde.png").exists()


def test_ad_dimer_smoke(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("openmm")
    pytest.importorskip("mdtraj")

    run_mod = _load_example_module(
        "examples/alanine_dipeptide/run_ad_dimer.py",
        "run_ad_dimer_example",
    )

    cfg = run_mod.load_config()
    cfg["shared"]["device"] = "cpu"
    cfg["shared"]["platform_name"] = "CPU"
    cfg["shared"]["output_root"] = tmp_path
    cfg["ad_dimer"]["output_dir"] = tmp_path / "ad_dimer"
    cfg["ad_dimer"]["md_steps"] = 1
    cfg["ad_dimer"]["n_blocks"] = 2
    cfg["ad_dimer"]["print_every"] = 1
    cfg["ad_dimer"]["save_dcd"] = False
    cfg["ad_dimer"]["dcd_interval"] = 1

    result = run_mod.run_ad_dimer_once(
        cfg,
        n_blocks_override=2,
        save_dcd_override=False,
        make_plots=True,
    )

    assert result["result_path"].exists()
    assert result["summary_path"].exists()
    payload = np.load(result["result_path"])
    assert payload["dihedrals_by_replica_deg"].shape[0] == 4
    assert payload["x_dihedrals_by_replica_deg"].shape[0] == 4
    assert (result["fig_dir"] / "ad_dimer_abs_psi_diff.png").exists()
    assert (result["fig_dir"] / "ad_dimer_abs_phi_diff.png").exists()
