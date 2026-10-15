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
Goal is `solve_rate>=50%`; not close yet. Commit `c8da12b`.
Results notebook: `notebooks/results_prenorm_scan.ipynb`.

## Running jobs
None.

## Negative result this session
Hypothesis: explicit `[SATISFIED]`/`[REACTIVATED]` clause-status tokens
(`use_clause_status=True`, implemented+tested, opt-in, default False --
every old checkpoint still loads unaffected) would raise `solve_rate`
above the 12.3%/4.3% baseline. **Not supported**: two runs, identical
config otherwise, got 0.0% and 0.67% -- both well below *both* baseline
runs (Slurm `47896056`, see `notebooks/results_clause_status.ipynb`).
Plausible cause: the extra tags lengthen every sequence a lot, which may
dilute training signal rather than reduce bookkeeping burden as
hypothesized. Feature stays in the codebase (opt-in) but is not adopted.
Output: `$SCRATCH/spin_gnn/runs/2026-10-14_clause-status-tokens/task{0,1}/`
(the per-row `checkpoint_dir` string inside those CSVs still says
`2026-10-15` -- a harmless stale path string from before a post-hoc `mv`,
not a data error; the actual checkpoint files are under `2026-10-14`).

## Open problems
1. Per-token diagnostic (`scripts/_sat2_step_accuracy_probe.py`, on the
   12.3% checkpoint) found the real bottleneck is choosing *which*
   variable to act on next (35-42% accuracy, near chance), not value
   prediction (96-100%) -- untouched by this session's experiment.
2. Run-to-run noise is large (~3-25x seen so far) because model init isn't
   seeded -- single/double-run comparisons should not be trusted much.
3. Still all-SAT in the sampled regime (`alpha=1.25`) -- no real UNSAT
   discrimination has been tested yet.

## Next steps
1. Attack the variable-selection bottleneck directly (open problem 1) --
   e.g. an auxiliary loss/signal for "which variable", not more
   bookkeeping tokens.
2. Consider seeding model init (`torch.manual_seed`) before trusting any
   future single-digit-percent comparison -- not yet done, undecided
   where in the training loop to add it without changing data seeding.
3. Still all-SAT regime (open problem 3): try a higher `alpha` once
   `solve_rate` improves at the easy end, to test real SAT/UNSAT
   discrimination.
