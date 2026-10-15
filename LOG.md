# LOG

Dated history for the DD Sudoku project (`sudoku-dd` branch). Do not read
this file in full from the start of a session -- search it for whatever
you need. At most 3 lines appended per session.

- 2026-10-15: Picked up from a previous (killed) agent session that had
  committed the model/code/handoff doc (`431aee3`) but never run anything
  on Sherlock. Fixed a broken `pip install numba` (used Sherlock's
  `py-numba/0.60.0_py312` module instead; 88 tests pass) and launched the
  first real greedy hint-removal sweep (Slurm `47921119`, q=4, d in
  4..8, 24 seeds) -- queued/running as of session end, no results yet.
