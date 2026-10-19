# STATUS

Project: DD Sudoku (direct-deduction coloring) on random regular graphs --
reproducing the reasoning-depth phase transition, plus a side task
pretraining a transformer on DD reasoning traces. Branch `sudoku-dd`. See
`SUDOKU_DD.md` for model/code. Separate from `master` (k-SAT) and AMP work.

## Current best result
**Phase transition (main task):** `d=7` recovers the claimed transition
(`gamma(rho)` jumps at `rho_c=0.27`); `d=5` does not yet (greedy floor
`rho~0.478`, above claimed `rho_c=0.46`). See LOG for the sweep (`47921119`).

**Transformer side task:** new best is `d_model=128, nhead=4, num_layers=12,
dim_feedforward=1024, epochs=900` (Slurm `48513926_1`, dir
`sudoku_hparam_sweep_2026-10-19_ep900/d0128_h04_L12_ff01024_lr1e-03_pre`):
`agg_solve_rate=0.767` (+0.047 over the old best's 0.720) at
`mean_o1_solve_rate=0.968`. Old best `d_model=64 @ 300ep` (`48162371`) still
leads the near-trivial metric (`0.980`, or `0.984` at 900ep). An attention-
distance analysis (`48267874`) showed the remaining gap is content-
*targeting*, not reach -- attention reaches the needed distances without
decay but prefers the right token only ~1.1-1.7x over a distance-matched
baseline. **Three levers are now all negative on that targeting metric:**
more heads (`48296020`), wider heads (`48446043`), and 3x training budget
(`48513926`) -- enrichment stays 1.13-1.26x everywhere. Metrics tables +
plots: `notebooks/sudoku_long_train_2026-10-19_results.ipynb` and
`notebooks/sudoku_head_scaling_results.ipynb`. Search LOG.md for
`2026-10-17`..`2026-10-19`.

## Running jobs
None. Slurm `48513926` (900-epoch retrain) and `48522086/48522087`
(enrichment) all COMPLETED 2026-10-19 and are written up. No other session is
working on `spin_gnn` (checked `ListAgents`/`squeue`).

## Open problems
1. Phase transition: `d=5`'s transition still unconfirmed (needs
   `sudoku_mcmc.anneal`, implemented/tested, never run for results); `<m>`
   order-parameter analysis not done; `d=4,6,8` data exists, unanalyzed.
2. Transformer: content-addressing ceiling (~1.13-1.26x enrichment) survives
   head count, head width, and 3x budget. The 300-epoch "wider is worse on
   easy cases" claim is **retracted** (it was undertraining -- see H2 in the
   2026-10-19 notebook). Still open: `agg_solve_rate` is pinned ~0.72-0.77
   while the near-trivial end is ~0.97-0.98, and longer training now just
   raises `val_nll` (overfitting) without moving either solve metric.
3. `d_model=256` was only ever run at 300 epochs, where it is budget-confounded
   (`mean_o1=0.842`, `k0=0.24`); its true quality is unknown.

## Next steps
1. Phase transition: run `sudoku_mcmc.anneal` for `d=5` (open problem 1);
   analyze `<m>` vs `rho` for `d=5,7`; extend `gamma(rho)` to `d=4,6,8`.
2. Transformer: parameters and steps are both exhausted as levers, so the
   next test should target content-addressing itself -- an explicit
   copy/pointer over node ids (a relative/rotary scheme is the weaker bet:
   `48267874` showed attention already reaches far without decay). Cheaper,
   more decisive diagnostic first: restate the needed neighbour rows next to
   each deduction in the token layout, testing whether long-range *retrieval*
   binds at all before building a mechanism for it.
3. Optionally re-run `d_model=256` at 900 epochs (open problem 3), and push
   the unpushed commits (`79ede2e`..`e4eba53`) to `origin/sudoku-dd`.
