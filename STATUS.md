# STATUS

Project: k-SAT reasoning-trace transformer (v2 event-based encoding,
`spin_gnn/ksat2*.py`, `sat_transformer2.py`). Separate from this repo's
kinetic-Ising/AMP work and the `sudoku-dd` branch -- those are other
agents' threads, not touched here.

## Current best result
`solve_rate=12.3%` (free-running generation, verified correct), config
`d_model=128, nhead=2, num_layers=16, lr=3e-4, norm_first=True`,
`n_vars=16, alpha=1.25 (m_clauses=20), n_hints=0`. Noisy: a repeat run at
the identical config landed at 4.3% (model init is unseeded -- data seed
is fixed at 0 throughout). Checkpoint:
`$SCRATCH/spin_gnn/sat2_hparam_sweep/m020_d0128_h02_L16_ff00512_lr3e-04_pre`.
Goal is `solve_rate>=50%`; not close yet. Commit `12d5d71`.
Results notebook: `notebooks/results_prenorm_scan.ipynb`.

## Running jobs
- Slurm `47896056` (array `0-1%1`, `sat2_clause_status_test.sbatch`):
  testing hypothesis that `[SATISFIED]`/`[REACTIVATED]` clause-status
  tokens (`use_clause_status=True`, same config as current best) raise
  `solve_rate` above 12.3%/4.3%. Submitted while the output dir was
  misnamed with tomorrow's date (caught before it started, fixed on disk
  for next time, but sbatch had already snapshotted the old path) -- it
  will write to `$SCRATCH/spin_gnn/runs/2026-10-15_clause-status-tokens/
  task{0,1}/`; **rename that to `2026-10-14_...` once it finishes** (plain
  `mv`, no new job needed). Check `sacct -j 47896056`; results notebook
  not yet made -- do that once both tasks finish (not done as of writing).

## Open problems
1. Per-token diagnostic (`scripts/_sat2_step_accuracy_probe.py`, on the
   12.3% checkpoint) found the real bottleneck is choosing *which*
   variable to act on next (35-42% accuracy, near chance), not value
   prediction (96-100%) -- the clause-status token experiment targets the
   bookkeeping half of this, not the variable-selection half directly.
2. Run-to-run noise is large (~3x) because model init isn't seeded --
   single-run comparisons below ~2x should not be trusted.
3. Still all-SAT in the sampled regime (`alpha=1.25`) -- no real UNSAT
   discrimination has been tested yet.

## Next steps
1. Once job `47896056` finishes: build a results notebook for it
   (`notebooks/results_clause_status.ipynb`), comparing both task
   solve_rates against the 12.3%/4.3% baseline, same eval.
2. If it helps: make `use_clause_status=True` the default and re-run the
   nhead scan with it. If it doesn't: document as a negative result here
   and in LOG.md, move on to the variable-selection bottleneck instead
   (e.g. an explicit "candidate variable" auxiliary signal).
3. Consider seeding model init (`torch.manual_seed`) so future comparisons
   aren't this noisy -- currently un-decided, not yet done.
