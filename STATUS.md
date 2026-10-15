# STATUS

Project: DD Sudoku (direct-deduction coloring) on random regular graphs --
reproducing the reasoning-depth phase transition from Behrens et al.-style
theory. Branch `sudoku-dd`. Picked up from a previous agent session that
was killed before running anything on Sherlock (no prior Slurm history,
no output files existed) -- see `SUDOKU_DD.md` for the full model/code
handoff, still accurate. Separate from this repo's main k-SAT reasoning-
trace project (`master` branch) and the AMP work -- not touched here.

## Current best result
None yet -- this is the first real run of this exploration on Sherlock.
`SUDOKU_DD.md`'s own "Observations so far" section (preliminary, small/
partial data from the previous agent's own laptop, explicitly to be
ignored/re-derived) reported: `q=4,d=5` greedy hint-removal gets stuck at
`rho≈0.477` (just above the slides' claimed `rho_c≈0.46`); `q=4,d=7`
reaches a clear slow phase (`T ~ N^0.74`) below its `rho_c≈0.27`. This
session's sweep (below) will give the first real Sherlock-derived numbers
to check that pattern against.

## Running jobs
- Slurm `47921119` (`sudoku_dd_sweep.sbatch`, single job, 16 CPUs/32GB/4h,
  internally parallel over a process pool): greedy hint-removal paths for
  `q=4`, `d in {4,5,6,7,8}`, all 7 slide `N` values, 24 seeds each = 840
  `(d,N,seed)` jobs. Writes one `.npz` per job to
  `$SCRATCH/spin_gnn/sudoku_dd_results/`, safe to resubmit after a timeout
  (finished files are skipped). Check `sacct -j 47921119`; was still
  PENDING (queued for resources, reason `None`) as of last check -- not
  stuck, just waiting. Commit `2b6e792`.

## Fixed this session
`pip install numba` (both sbatch scripts) had no compatible wheel and
fell back to a source build that also tried building numpy from source
and failed on an assembler incompatibility (`endbr64`) -- an HPC toolchain
issue, not a code bug. Fixed by using Sherlock's own `py-numba/0.60.0_py312`
module instead (verified compatible with the rest of this repo's module
stack). All 88 tests pass (32 for `sudoku_dd`, rest of the repo's suite).

## Open problems
1. (From the handoff doc) Random-order greedy removal may not be able to
   reach the slow phase for `d=5` at all (got stuck above `rho_c` even on
   the previous agent's own small-scale laptop data) -- may need
   `sudoku_mcmc.anneal` (implemented, tested, never yet used for results).
2. No results exist yet to actually check against the slides' claimed
   `rho_c`/`<m>` continuity-vs-discontinuity pattern.

## Next steps
1. Once Slurm `47921119` finishes (or makes meaningful partial progress):
   build a results notebook -- `T/log N` and `<m>` vs `rho` per `d`, locate
   `rho_c` from the data (curve collapse / `gamma(rho)` fit / variance
   peak), compare to slides' 0.46 (d=5, continuous) and 0.27 (d=7,
   discontinuous). This is handoff-doc next-step #2-3.
2. If `d=5` greedy still can't reach below `rho_c` (per open problem 1):
   try `sudoku_mcmc.anneal` at a smaller `N` (~4k-17k per the handoff doc).
3. Per-round deduction profile (handoff doc next-step #6), once a results
   notebook exists to build on.
