# STATUS

Project: DD Sudoku (direct-deduction coloring) on random regular graphs --
reproducing the reasoning-depth phase transition from Behrens et al.-style
theory. Branch `sudoku-dd`. See `SUDOKU_DD.md` for the full model/code.
Separate from this repo's main k-SAT reasoning-trace project (`master`
branch) and the AMP work -- not touched here.

## Current best result
First real sweep on Sherlock (840 runs, `q=4, d in {4..8}`, 7 slide `N`
values, 24 seeds, Slurm `47921119`, completed). Analysis
(`notebooks/results_transition.ipynb`): **`d=7` recovers the claimed
transition** -- the growth exponent `gamma(rho)` (`log T ~ gamma*log N`)
jumps from a flat ~0.1 to 0.46-0.64 right at the slides' `rho_c=0.27`
(between `rho=0.28` and `0.26`). **`d=5` does not yet** -- greedy removal
hits a hard, very tight floor at `rho_min~0.4778` across every `N`/seed,
*above* the claimed `rho_c=0.46`, matching the prior agent's own smaller-
scale number almost exactly. Commit `bffce3a`.

## Running jobs
None.

## Open problems
1. `d=5`'s transition is still unconfirmed -- random-order greedy removal
   cannot get below `rho~0.478`. Needs `spin_gnn.sudoku_mcmc.anneal`
   (implemented, tested, never yet used for results) to push below that
   floor and check for a slow phase / the claimed continuous `<m>`.
2. `<m>` (the order parameter itself, not just `T`/`gamma`) hasn't been
   analyzed yet -- only `T`/`gamma(rho)` so far. Needed to check the
   slides' continuous-(`d=5`)-vs-discontinuous-(`d=7`) claim directly.
3. Only `d=5,7` analyzed in depth; `d=4,6,8` data exists (same sweep) but
   unexamined -- useful for "order of the transition vs d" (handoff doc
   next-step #5).

## Next steps
1. Analyze `<m>` vs `rho` for `d=5,7` (continuous vs. discontinuous jump)
   -- the other half of the slides' claim, not yet checked.
2. Run `sudoku_mcmc.anneal` for `d=5` at `N~4k-17k` to try to push below
   `rho~0.46` and see if a slow phase appears there too (open problem 1).
3. Extend the `gamma(rho)` analysis to `d=4,6,8` to locate `rho_c` per
   degree and see where the continuous-to-discontinuous switch happens
   (open problem 3 / handoff doc next-step #5).
