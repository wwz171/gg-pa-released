# GG-PA: Generative Gibbs for Physics-Aware Sampling

This repository contains the reference research code for **GG-PA**.

GG-PA is a generative Gibbs framework for coupling pretrained diffusion priors with explicit physical contexts in an augmented space. The codebase is organized as a research prototype: the package under `src/ggpa` contains the reusable implementation, while the root `notebooks/` directory provides quick runs that reproduce small paper-facing demonstrations with bundled checkpoints.

## What Is Included

- `notebooks/`: quick-run notebooks intended for direct use
- `checkpoints/`: pretrained model weights used by the public notebooks
- `src/ggpa/`: package code for kernels, clients, server components, and system-specific samplers
- `configs/`: lightweight configuration files
- `examples/`: larger end-to-end scripts that generate raw data and paper-facing figures

The exploratory folders used during development are not part of the intended public surface of the project.

Within `src/ggpa`, the maintained reusable system modules are currently `phi4` and `alanine_dipeptide`. The double-well code is kept to support the public quick-run notebook.

## Installation

From the repository root:

```bash
python -m pip install -e .
```

This installs the pinned public environment used by the released notebooks and examples.

If you prefer conda, use:

```bash
conda env create -f environment.yml
conda activate ggpa
```

The default `torch` pin is CPU-oriented. If you need CUDA, replace the torch installation afterward using the official PyTorch selector.

## Quick Start

### Double-Well Notebook

`notebooks/example_doublewell.ipynb` is the smallest quick run in the repository. It loads a pretrained one-dimensional diffusion prior, constructs the annealed GG-PA context, and compares direct diffusion, single-`t` GG-PA, and replica-exchange GG-PA on the coupled double-well system.

### $\phi^4$ Notebook

`notebooks/example_phi4.ipynb` provides a small two-dimensional lattice quick run. It includes:

- an `L=32` snapshot demonstration at several couplings `J`
- a fast `L=8` sweep showing how `<|m|>` and `\chi` vary with `J`

Both notebooks are meant to be runnable directly from the repository with the checkpoints already stored under `checkpoints/`.

### $\phi^4$ Replica-Exchange Script

For the heavier `L=32` paper-facing phi4 scans, use the dedicated folder in `examples/` instead of the notebook:

```bash
bash examples/phi4/run_phase_transition.sh
bash examples/phi4/run_critical_scaling.sh
python examples/phi4/plot_phi4_results.py all
```

The two shell scripts launch the zero-field phase-transition scan and the nonzero-field critical-scaling scan in the background. Both workflows use the shared YAML config at `configs/phi4_example.yaml`, and raw outputs are written under `examples/phi4/results/`. The plotting script then loads those raw files and produces the phase-transition and data-collapse figures. The shell launchers use the current shell environment by default; set `GGPA_CONDA_ENV=<env>` to force `conda run -n <env>`.

The script defaults match the SI description for the paper-facing phi4 runs:
`10_000` sweeps per point, magnetization recorded every sweep, `30%` burn-in,
and a `48`-replica ladder between `t=0.1` and `t=0.6`. The raw outputs also
store the per-replica magnetization histories together with the final
per-replica `(phi, psi)` fields.

### Alanine Dipeptide Scripts

The alanine-dipeptide examples in `examples/alanine_dipeptide/` expose the
shared torsion-diffusion prior used for two explicit MD systems:

```bash
bash examples/alanine_dipeptide/run_ad_sodium.sh
python examples/alanine_dipeptide/run_ad_sodium.py
bash examples/alanine_dipeptide/run_ad_dimer.sh
python examples/alanine_dipeptide/run_ad_dimer.py
python examples/alanine_dipeptide/plot_alanine_results.py ad-sodium
python examples/alanine_dipeptide/plot_alanine_results.py ad-dimer
```

Both workflows read from `configs/alanine_dipeptide_example.yaml`. The
monomer+Na public example is a zero-shot transfer run built around the bundled
`0%` torsion prior: `5` trajectories, `5000` GG-PA steps each, `20%` burn-in,
and saved DCD files with one frame per GG-PA step. The dimer example remains a
four-replica GG-PA-RE ladder with `t = [0.1, 0.15, 0.25, 0.4]`, `1000` outer
blocks, saved replica DCD files, and two public comparison figures
(`|Δ\psi|` and `|Δ\phi|`). Outputs are written under
`examples/alanine_dipeptide/results/`. The public alanine config defaults to
`device: cpu` and `platform_name: CUDA`, which is the most robust setting on
machines where OpenMM can use the GPU but the local PyTorch build is not yet
matched to the installed accelerator. The shell launcher uses
the current shell environment by default; set `GGPA_CONDA_ENV=<env>` to force
`conda run -n <env>`.

## Repository Layout

```text
.
|-- checkpoints/          pretrained weights for the public notebooks
|-- configs/              lightweight configuration files
|-- notebooks/            quick-run demonstrations
|-- examples/             heavier scripts for raw-data generation and figures
`-- src/ggpa/             reusable GG-PA package code
```

## Citation

If this repository is useful in your work, please cite the software record in [CITATION.cff](CITATION.cff).

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
