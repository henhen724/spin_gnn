# STATUS

Project: k-SAT reasoning-trace transformer (v2 event-based encoding,
`spin_gnn/ksat2*.py`, `sat_transformer2.py`). Separate from this repo's
kinetic-Ising/AMP work and the `sudoku-dd` branch -- those are other
agents' threads, not touched here.

## Current best result
`solve_rate=12.3%` (free-running generation, verified correct), config
`d_model=128, nhead=2, num_layers=16, lr=3e-4, norm_first=True`,
`n_vars=16, alpha=1.25 (m_clauses=20), n_hints=0`. Still the peak of 4
samples at this exact config now: 12.3%, 7.0%, 4.3%, 3.3% (mean ~6.7%) --
see "This session" below. Goal is `solve_rate>=50%`; not close. Commit
`0ec522d`. Results notebooks: `notebooks/results_prenorm_scan.ipynb`,
`results_clause_status.ipynb` (negative), `results_seedfix_validation.ipynb`.

## Running jobs
None.

## This session (2026-10-15)
Hypothesis: unseeded model init (`torch.manual_seed` was never called) is
the dominant cause of the 3-25x run-to-run `solve_rate` noise seen in
every sweep. Fix implemented (`train_sat_transformer2.py`, seeds from
`sat_cfg.seed`) and tested: **CPU runs are now bit-identical for the same
seed** (new regression test). **GPU runs (the actual training path) are
only partially helped**: two seed=0 GPU runs at the current-best config
gave 7.0%/3.3% (ratio 2.1x) vs. the pre-fix 12.3%/4.3% (ratio 2.9x) --
smaller spread, not eliminated, because cuDNN's own algorithm/reduction
nondeterminism isn't pinned by `manual_seed` alone. **Not a solve_rate
improvement** -- purely a reproducibility fix, honestly only partial on
GPU. Slurm `47916082`. Output:
`$SCRATCH/spin_gnn/runs/2026-10-15_seed-fix-validation/task{0,1}/`.

## Open problems
1. Per-token diagnostic found the real bottleneck is choosing *which*
   variable to act on next (35-42% accuracy, near chance), not value
   prediction (96-100%) -- still untouched by any fix so far.
2. GPU run-to-run noise is reduced but not eliminated (see above) --
   still treat any single GPU run as unreliable; full determinism would
   need `torch.use_deterministic_algorithms(True)` (untried, unknown
   speed cost).
3. Still all-SAT in the sampled regime (`alpha=1.25`) -- no real UNSAT
   discrimination has been tested yet.

## Next steps
1. Attack the variable-selection bottleneck directly (open problem 1) --
   e.g. an auxiliary loss/signal for "which variable", not more
   bookkeeping tokens (already tried and rejected, see LOG.md).
2. If GPU determinism matters more going forward, try
   `torch.use_deterministic_algorithms(True)` and measure the speed cost
   before deciding whether to adopt it.
3. Try a higher `alpha` once `solve_rate` improves at the easy end, to
   test real SAT/UNSAT discrimination (open problem 3).
