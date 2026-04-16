# Examples

This directory contains heavier end-to-end GG-PA examples that go beyond the quick-run notebooks in `notebooks/`.

Current public entry points:

- `phi4/`
  - `phi4_re_scan.py`: reproducible replica-exchange GG-PA scans for the 2D phi4 lattice.
  - `plot_phi4_results.py`: plotting utilities that load the saved raw files and produce the phase-transition and data-collapse figures.
  - `run_phase_transition.sh` / `run_critical_scaling.sh`: background launchers that use the current shell environment by default, or `GGPA_CONDA_ENV=<env>` when you want `conda run`.
  - `README.md`: folder-local usage notes.

- `alanine_dipeptide/`
  - `run_ad_sodium.sh`: background launcher for the AD+Na ensemble plus automatic plotting.
  - `run_ad_sodium.py`: public zero-shot AD+Na runner; defaults to the `5 x 5000` ensemble and supports `--single-run` for debugging.
  - `run_ad_dimer.py`: replica-exchange GG-PA example for the alanine dipeptide dimer.
  - `run_ad_dimer.sh`: background launcher for the public dimer RE example plus automatic plotting.
  - `plot_alanine_results.py`: plotting utilities for the saved alanine raw files.
  - `README.md`: folder-local usage notes.

The root `notebooks/` directory remains the quickest way to try the project interactively. The scripts in `examples/` are meant for longer paper-facing runs that generate raw data and figures from the bundled checkpoints.
