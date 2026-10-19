# STATUS

Project: DD Sudoku (direct-deduction coloring) on random regular graphs --
reproducing the reasoning-depth phase transition, plus a side task
pretraining a transformer on DD reasoning traces. Branch `sudoku-dd`. See
`SUDOKU_DD.md` for model/code. Separate from `master` (k-SAT) and AMP work.

## Current best result
**Phase transition (main task):** `d=7` recovers the claimed transition
(`gamma(rho)` jumps at `rho_c=0.27`); `d=5` does not yet (greedy floor
`rho~0.478`, above claimed `rho_c=0.46`). See LOG for the sweep (`47921119`).

**Transformer side task:** best checkpoint is `d_model=64, nhead=4,
num_layers=12, dim_feedforward=1024, epochs=300`
(`d0064_h04_L12_ff01024_lr1e-03_pre`, commit `48162371`'s sweep):
`mean_o1_solve_rate=0.980`, `agg_solve_rate=0.72`. An attention-distance
analysis (`48267874`) found the remaining gap is content-*targeting*, not
reach (attention doesn't decay with distance, but only weakly prefers the
right token, ~1.0-1.7x baseline). Two follow-up scaling sweeps both came
back **negative** -- neither more heads (`48296020`) nor wider heads via
bigger `d_model` at fixed `nhead=4` (`48446043`) raises targeting
enrichment or `mean_o1_solve_rate`; `nhead=4/head_dim=16` is already a
local optimum on both axes. Full comparison + metrics table + plots:
`notebooks/sudoku_head_scaling_results.ipynb` (commit `ad6c5bb`, same seeds
as the hparam sweep). Details: search LOG.md for `2026-10-17`/`2026-10-18`.

## Running jobs
None. Checked `squeue -u hshunt` and `ListAgents` this session (2026-10-19)
-- no other session is working on `spin_gnn`, nothing running or pending.

## Open problems
1. Phase transition: `d=5`'s transition still unconfirmed (needs
   `sudoku_mcmc.anneal`, implemented/tested, never run for results); `<m>`
   order-parameter analysis not done; `d=4,6,8` data exists, unanalyzed.
2. Transformer: content-addressing ceiling (~1.25x enrichment) is stuck
   across both head-count and head-width scaling. The `d_model` sweep's
   `agg_solve_rate` UP (0.72->0.85) while `mean_o1_solve_rate`/`k0` DOWN
   (0.980/0.98->0.842/0.24) and `val_nll` worse (1.2->2.41) is unexplained
   -- overfitting to harder/longer chains at larger width, vs. a genuine
   head_dim effect, not distinguished yet.

## Next steps
1. Phase transition: run `sudoku_mcmc.anneal` for `d=5` (open problem 1);
   analyze `<m>` vs `rho` for `d=5,7`; extend `gamma(rho)` to `d=4,6,8`.
2. Transformer: since scaling nhead/d_model alone didn't fix targeting,
   next idea is a non-parameter-count architecture change aimed at
   content-addressing directly (e.g. relative/rotary position, or an
   explicit copy/pointer mechanism) rather than another head sweep point --
   flagged to the user 2026-10-18 as a fork worth their input on before
   spending more GPU time; awaiting direction.
3. Decide whether to push the accumulated unpushed commits (`79ede2e`
   through `ad6c5bb`) to `origin/sudoku-dd`.
