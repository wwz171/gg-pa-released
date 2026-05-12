# GG-PA: Generative Gibbs for Physics-Aware Sampling

This repository contains the reference research code for **GG-PA**.

GG-PA is a training-free generative Gibbs framework for composing pretrained diffusion priors with explicit physical context in an augmented state space. The codebase is organized as a research prototype: the package under `src/ggpa` contains the reusable implementation, while the root `notebooks/` directory provides quick runs that reproduce small paper-facing demonstrations with bundled checkpoints.

## What Is Included

- `notebooks/`: quick-run notebooks intended for direct use
- `checkpoints/`: pretrained model weights used by the released notebooks
- `data/`: packaged topologies and compact processed references used by the alanine examples
- `src/ggpa/`: package code for kernels, clients, server components, and system-specific samplers
- `configs/`: lightweight configuration files
- `examples/`: larger end-to-end scripts that generate raw data and paper-facing figures

The exploratory folders used during development are not part of the intended public surface of the project.

Within `src/ggpa`, the maintained reusable system modules are currently `phi4` and `alanine_dipeptide`. The double-well code is kept to support the released quick-run notebook.

## Installation

From the repository root:

```bash
python -m pip install -e .
```

This installs the dependency set used by the released notebooks and examples.

If you prefer conda, use:

```bash
conda env create -f environment.yml
conda activate ggpa
```

If you need a specific CUDA-enabled PyTorch build, install `torch` using the official PyTorch selector for your platform.

## Quick Start

### Double-Well Notebook

`notebooks/example_doublewell.ipynb` is the smallest quick run in the repository. It loads a pretrained one-dimensional diffusion prior, constructs the annealed GG-PA context, and compares direct diffusion, single-`t` GG-PA, and replica-exchange GG-PA on the coupled double-well system.

### $\phi^4$ Notebook

`notebooks/example_phi4.ipynb` provides a small two-dimensional lattice quick run. It includes:

- an `L=32` snapshot demonstration at several couplings `J`
- a fast `L=8` sweep showing how `<|m|>` and `\chi` vary with `J`

Both notebooks are meant to be runnable directly from the repository with the checkpoints already stored under `checkpoints/`.

### $\phi^4$ Replica-Exchange Script

For the heavier `L=32` paper-facing Ginzburg-Landau phi^4 scans, use the dedicated folder in `examples/` instead of the notebook:

```bash
bash examples/phi4/run_phase_transition.sh
bash examples/phi4/run_critical_scaling.sh
python examples/phi4/plot_phi4_results.py all
```

The two shell scripts launch the zero-field phase-transition scan and the nonzero-field critical-scaling scan in the background. Both workflows use the shared YAML config at `configs/phi4_example.yaml`, and raw outputs are written under `examples/phi4/results/`. The plotting script then loads those raw files and produces the phase-transition and data-collapse figures. The shell launchers use the current shell environment by default; set `GGPA_CONDA_ENV=<env>` to force `conda run -n <env>`.

The script defaults match the SI description for the paper-facing Ginzburg-Landau phi^4 runs:
`10_000` sweeps per point, magnetization recorded every sweep, `30%` burn-in,
and a `48`-replica ladder between `t=0.1` and `t=0.6`. The raw outputs also
store the per-replica magnetization histories together with the final
per-replica `(phi, psi)` fields.

### Alanine Dipeptide Scripts

The alanine-dipeptide examples in `examples/alanine_dipeptide/` expose the
shared torsion-diffusion prior used for two explicit MD systems:

```bash
python examples/alanine_dipeptide/run_ad_sodium.py
python examples/alanine_dipeptide/plot_alanine_results.py ad-sodium
python examples/alanine_dipeptide/run_ad_dimer.py
python examples/alanine_dipeptide/plot_alanine_results.py ad-dimer
```

The same workflows can be launched in the background with:

```bash
bash examples/alanine_dipeptide/run_ad_sodium.sh
bash examples/alanine_dipeptide/run_ad_dimer.sh
```

Both workflows read from `configs/alanine_dipeptide_example.yaml`. The
AD-Na+ example is a zero-shot transfer run built around the bundled
`0%` torsion prior: `5` trajectories, `5000` GG-PA steps each, `20%` burn-in,
and saved DCD files with one frame per GG-PA step. The dimer example remains a
four-replica GG-PA-RE ladder with `t = [0.1, 0.15, 0.25, 0.4]`, `1000` outer
blocks, saved replica DCD files, and two comparison figures
(`|Δ\psi|` and `|Δ\phi|`). Outputs are written under
`examples/alanine_dipeptide/results/`. The released alanine config defaults to
`device: cpu` and `platform_name: CUDA`, which is the most robust setting on
machines where OpenMM can use the GPU but the local PyTorch build is not yet
matched to the installed accelerator. The shell launchers use
the current shell environment by default; set `GGPA_CONDA_ENV=<env>` to force
`conda run -n <env>`. On CPU-only machines, add `--platform CPU`.

## Repository Layout

```text
.
|-- checkpoints/          pretrained weights for the released notebooks
|-- configs/              lightweight configuration files
|-- data/                 compact topologies and processed references
|-- notebooks/            quick-run demonstrations
|-- examples/             heavier scripts for raw-data generation and figures
`-- src/ggpa/             reusable GG-PA package code
```

## Citation

If this repository is useful in your work, please cite the paper:

```bibtex
@misc{wang2026composingdiffusionpriorsexplicit,
      title={Composing diffusion priors with explicit physical context via generative Gibbs sampling},
      author={Weizhou Wang and Jonathan Weare and Aaron R. Dinner},
      year={2026},
      eprint={2605.10642},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.10642},
}
```

The repository also includes software citation metadata in [CITATION.cff](CITATION.cff), with the arXiv paper set as the preferred citation.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
