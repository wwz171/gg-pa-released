# Alanine Dipeptide Examples

This folder contains two public GG-PA examples driven by the shared torsion
checkpoint `checkpoints/ad_torsion_prior.pt`.

## Included workflows

- `run_ad_sodium.py`
  - public zero-shot `AD + Na+` example built around the bundled `0%` torsion prior
  - by default it runs a small ensemble of `5` trajectories, each with `5000` GG-PA steps
  - applies `20%` burn-in during aggregation and writes one aggregated raw-data file
  - saves per-trajectory DCD files together with an aggregated Ramachandran / O-O comparison
  - `--single-run` switches back to a single standalone debugging trajectory

- `run_ad_dimer.py`
  - public alanine-dipeptide dimer example using replica exchange
  - uses the root `data/adp_dimer_vacuum.pdb`
  - keeps a flat-bottom COM centering force active to reduce dimer dissociation
  - uses the tested `t = [0.1, 0.15, 0.25, 0.4]` ladder
  - by default runs a single `1000`-block RE trajectory and applies `20%` burn-in in plotting
  - saves full noisy observables for every replica plus two public comparison figures:
    an `|Δψ|` comparison and an `|Δφ|` comparison against the packaged MD reference
  - a single run may spontaneously select one of the two symmetry-related parallel branches,
    so `P_LR` vs `P_RL` asymmetry in the public figure is expected rather than a bug

- `run_ad_dimer.sh`
  - one-command background launcher for the public AD dimer RE example
  - runs the simulation and then regenerates the public figures
  - uses the current shell environment by default; set `GGPA_CONDA_ENV=<env>` to force `conda run`

- `plot_alanine_results.py`
  - replots saved raw files without rerunning the simulations

- `run_ad_sodium.sh`
  - one-command background launcher for the public AD+Na zero-shot ensemble
  - runs the ensemble and then automatically generates the comparison figures
  - uses the current shell environment by default; set `GGPA_CONDA_ENV=<env>` to force `conda run`

## Config

Both examples read their defaults from:

- `configs/alanine_dipeptide_example.yaml`

The YAML file controls the shared checkpoint, MD parameters, diffusion settings,
and output locations.

The default public config uses `device: cpu` together with `platform_name: CUDA`.
This keeps the torsion denoiser on CPU while allowing OpenMM to use the GPU when
available, which is the most robust choice on machines where the local PyTorch
build is not yet matched to the installed GPU.

For the public AD+Na example, the default config now reflects the more stable
zero-shot setting:

- one bundled checkpoint: `checkpoints/ad_torsion_prior.pt`
- `5` trajectories
- `5000` GG-PA steps per trajectory
- `20%` burn-in during aggregation
- one saved DCD frame per GG-PA step

For the public AD dimer example, the default config reflects the tested RE setup:

- `4` replicas with `t = [0.1, 0.15, 0.25, 0.4]`
- `1000` RE blocks
- `100` MD steps per block
- `20%` burn-in used only in the plotting / post-processing summaries
- `save_dcd = true`, with one DCD frame per outer sweep
- `centering_schedule = always`, `centering_force_k = 50.0`, `centering_d0 = 0.9`
- on the current machine this public run takes about `2 min`

If you are on a CPU-only machine, override the OpenMM platform explicitly:

```bash
python examples/alanine_dipeptide/run_ad_sodium.py --platform CPU
python examples/alanine_dipeptide/run_ad_dimer.py --platform CPU
```

## Commands

Run the public AD+Na zero-shot ensemble in the foreground:

```bash
python examples/alanine_dipeptide/run_ad_sodium.py
python examples/alanine_dipeptide/plot_alanine_results.py ad-sodium
```

Run the same AD+Na workflow in the background:

```bash
bash examples/alanine_dipeptide/run_ad_sodium.sh
```

Run the single-trajectory monomer + sodium debug entry:

```bash
python examples/alanine_dipeptide/run_ad_sodium.py --single-run
```

Run the dimer replica-exchange example:

```bash
python examples/alanine_dipeptide/run_ad_dimer.py
```

Run the same dimer workflow in the background:

```bash
bash examples/alanine_dipeptide/run_ad_dimer.sh
```

Replot saved results:

```bash
python examples/alanine_dipeptide/plot_alanine_results.py ad-sodium
python examples/alanine_dipeptide/plot_alanine_results.py ad-sodium-single
python examples/alanine_dipeptide/plot_alanine_results.py ad-dimer
```
