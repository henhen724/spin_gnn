# LOG

Dated history for the DD Sudoku project (`sudoku-dd` branch). Do not read
this file in full from the start of a session -- search it for whatever
you need. At most 3 lines appended per session.

- 2026-10-15: Picked up from a previous (killed) agent session that had
  committed the model/code/handoff doc (`431aee3`) but never run anything
  on Sherlock. Fixed a broken `pip install numba` (used Sherlock's
  `py-numba/0.60.0_py312` module instead; 88 tests pass) and ran the
  first real greedy hint-removal sweep (Slurm `47921119`, 840 runs).
- 2026-10-15: Analysis of that sweep recovers the claimed easy-to-hard
  transition for `d=7` (gamma(rho) jumps right at rho_c=0.27) but not yet
  for `d=5` (greedy stuck above its rho_c=0.46) -- see
  `notebooks/results_transition.ipynb`.
