# Phi4 Example

This directory contains the heavier paper-facing phi4 reproduction workflow.

Main entry points:

- `phi4_re_scan.py`
  - `phase-transition`: zero-field replica-exchange scan across `J`
  - `critical-scaling`: nonzero-field scaling-window scan across `(h, J)`
- `plot_phi4_results.py`
  - `phase-transition`: plot the zero-field phase-transition figure from `re_scan.npz`
  - `critical-scaling`: plot the data-collapse figure from `cs_h..._g....npz` or `critical_scaling_summary.npz`
  - `all`: produce both figures
- `run_phase_transition.sh`
  - launch the zero-field scan in the background with `conda run -n pyg`
- `run_critical_scaling.sh`
  - launch the critical-scaling scan in the background with `conda run -n pyg`
  - both shell launchers forward any extra CLI flags to `phi4_re_scan.py`, e.g. `--compile-model`

Default parameters live in `configs/phi4_example.yaml`.

By default, raw outputs are written under:

- `examples/phi4/results/phase_transition`
- `examples/phi4/results/critical_scaling`
- `examples/phi4/results/figures`
