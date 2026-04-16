from importlib import import_module
from pathlib import Path

import pytest


def test_top_level_import_is_lightweight():
    import ggpa

    assert ggpa.__version__
    assert ggpa.FixedDiffusionTimeKernel.__name__ == "FixedDiffusionTimeKernel"
    assert ggpa.State.__name__ == "State"


def test_systems_namespace_is_curated():
    import ggpa.systems as systems

    assert systems.__all__ == ["phi4", "alanine_dipeptide"]


def test_public_repo_assets_exist():
    required = [
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        "configs/phi4_example.yaml",
        "configs/alanine_dipeptide_example.yaml",
        "examples/phi4/README.md",
        "examples/phi4/phi4_re_scan.py",
        "examples/phi4/plot_phi4_results.py",
        "examples/phi4/run_phase_transition.sh",
        "examples/phi4/run_critical_scaling.sh",
        "examples/alanine_dipeptide/README.md",
        "examples/alanine_dipeptide/common.py",
        "examples/alanine_dipeptide/run_ad_sodium.sh",
        "examples/alanine_dipeptide/run_ad_sodium.py",
        "examples/alanine_dipeptide/run_ad_dimer.sh",
        "examples/alanine_dipeptide/run_ad_dimer.py",
        "examples/alanine_dipeptide/plot_alanine_results.py",
        "notebooks/example_doublewell.ipynb",
        "notebooks/example_phi4.ipynb",
        "data/adp_monomer_vacuum.pdb",
        "data/adp_dimer_vacuum.pdb",
        "data/ad_sodium_ref/monomer_vacuum_300k_ref.npz",
        "data/ad_sodium_ref/ad_sodium_md_300k_ref.npz",
        "data/ad_dimer_ref/ad_dimer_md_100ns_ref.npz",
        "data/ad_dimer_ref/manifest.json",
        "checkpoints/ad_torsion_prior.pt",
        "checkpoints/doublewell_prior.pt",
        "checkpoints/phi4_prior.pt",
    ]
    for relpath in required:
        assert Path(relpath).exists(), relpath


def test_quick_run_modules_import():
    pytest.importorskip("torch")

    import_module("ggpa.models.diffusion")
    import_module("ggpa.systems.doublewell")
    import_module("ggpa.systems.phi4")
    import_module("ggpa.systems.alanine_dipeptide")


def test_phi4_and_doublewell_checkpoints_load_expected_backbones():
    torch = pytest.importorskip("torch")
    _ = torch

    from ggpa.models import ResidualModel, SimpleDiffusion, VelocityModel

    phi4 = SimpleDiffusion.load_from_checkpoint("checkpoints/phi4_prior.pt", device="cpu")
    doublewell = SimpleDiffusion.load_from_checkpoint("checkpoints/doublewell_prior.pt", device="cpu")

    assert isinstance(phi4, SimpleDiffusion)
    assert isinstance(doublewell, SimpleDiffusion)
    assert isinstance(phi4.velocity_model, ResidualModel)
    assert isinstance(doublewell.velocity_model, VelocityModel)
