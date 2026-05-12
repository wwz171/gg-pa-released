# Ginzburg-Landau Phi^4 Example

This directory contains the heavier paper-facing Ginzburg-Landau phi^4 reproduction workflow.

Main entry points:

- `phi4_re_scan.py`
  - `phase-transition`: zero-field replica-exchange scan across `J`
  - `critical-scaling`: nonzero-field scaling-window scan across `(h, J)`
- `plot_phi4_results.py`
  - `phase-transition`: plot the zero-field phase-transition figure from `re_scan.npz`
  - `critical-scaling`: plot the data-collapse figure from `cs_h..._g....npz` or `critical_scaling_summary.npz`
  - `all`: produce both figures
- `run_phase_transition.sh`
  - launch the zero-field scan in the background
- `run_critical_scaling.sh`
  - launch the critical-scaling scan in the background
  - both shell launchers forward any extra CLI flags to `phi4_re_scan.py`, e.g. `--compile-model`
  - both shell launchers use the current shell environment by default; set `GGPA_CONDA_ENV=<env>` to force `conda run -n <env>`

Default parameters live in `configs/phi4_example.yaml`.

The paper denotes the lattice coupling by `J`. Some saved arrays, filenames,
and config keys retain the legacy name `gamma` for compatibility with earlier
run outputs; in this example, read `gamma` as `J`.

By default, raw outputs are written under:

- `examples/phi4/results/phase_transition`
- `examples/phi4/results/critical_scaling`
- `examples/phi4/results/figures`
