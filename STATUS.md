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

## Side task: transformer pretraining on DD-Sudoku reasoning traces

Separate from the phase-transition sweep above. Mirrors the `master`
branch's k-SAT v2 tokenization lessons (event-based trace, explicit
structural tags, pre-norm, seeded init, solve_rate-style end-to-end eval)
applied to a new problem: `sudoku_trace.py` (DD solver instrumented to
emit a round-by-round event trace), `sudoku_tokens.py` (vocab/encode/
decode), `train_sudoku_transformer.py` (training loop, reuses
`SATReasoningTransformer`/`sat_dataset2` unchanged). Tests:
`test_sudoku_trace.py`, `test_sudoku_tokens.py`,
`test_train_sudoku_transformer.py` (27 passing).

**Correction (superseding the first attempt below):** the first version of
`build_instances` sampled hints by independent uniform random draw, which
is *not* the slides' protocol and regularly produces unresolvable
instances (not a finite-size bug, just the wrong sampling scheme -- see
LOG). The slides (and this branch's own `sudoku_dd.removal_path`/
`remove_hint`) instead start fully hinted and remove hints one at a time
in random order, **only accepting a removal that still leaves the
coloring fully DD-recoverable** -- every instance generated this way is
resolvable by construction, with no UNRESOLVED case at all.
`build_instances` now does this (pick a target density via
`removal_path(..., snapshot_rhos=(target_rho,))`, falling back to that
path's own minimal density if the target is unreachably sparse), and
asserts `trace["resolved"]` as a correctness check.
`SudokuReasoningConfig.rho/rho_max` now default to `0.3`/`0.95`, spanning
shallow to deep reasoning chains (previously 0.9/0.96, which was tuned to
patch over the wrong protocol's resolved/unresolved mix -- no longer
needed or correct).

First training run under the **corrected** protocol
(`scripts/slurm/sudoku2_train_transformer.sbatch`, Slurm `48032796`,
N=24/d=5/q=4, d_model=64/2 layers, 2000 train + 200 val instances, 50
epochs, ~1 min wall-clock): final val_accuracy=0.332, outcome_accuracy=
0.995, well_formed_rate=0.995, but **`solve_rate=0.005`** -- i.e. outcome
prediction is now a near-freebie (virtually every instance is SOLVED by
construction) while true end-to-end exact-solve collapsed from the
uncorrected run's 0.405. This is the same "outcome accuracy deceptively
high, true solve rate near zero" pattern the k-SAT v2 work's
copy_vs_delta_breakdown finding flagged -- and makes sense here: the
corrected `rho` range reaches reasoning chains of 11+ rounds (vs. the old
protocol's near-fixed 1-2 rounds), and any single wrong color early in a
long autoregressive chain cascades through later rounds. Current
d_model=64/2-layer/2000-instance/50-epoch budget is nowhere near enough
for this harder depth distribution. Scaling up model size/width/
instances/epochs, or restricting `rho` to a narrower/shallower band as an
easier curriculum stage, is open for a future session.

**Checkpoint-vs-difficulty transition sweep.** To see whether that aggregate
`solve_rate=0.005` means "uniformly bad" or "learned the easy end, still
failing the hard end," added checkpoint snapshotting to `train_sudoku_model`
(`checkpoint_epochs=` kwarg) and a pinned-density evaluator,
`eval_solve_by_rho` (each bin's instances all target one fixed `rho`, not
the training `[rho, rho_max]` spread) -- both in `train_sudoku_transformer.
py`. Driven by the new `spin_gnn.sudoku_transition_sweep` script /
`sudoku2_transition_sweep.sbatch`, plotted in
`notebooks/sudoku_transition.ipynb` (Slurm `48144619` train+eval,
`48145554` notebook render). **Confirms the second hypothesis**: epochs
0-20 are flat at `solve_rate=0.00` for every `rho` in `{0.3..0.95}`; epochs
30-50 rise *only* at the high-`rho` (easy/shallow) end while staying
`0.00` for `rho<=0.8` -- but the rise is modest (epoch 50 reaches only
`0.16` at `rho=0.95`), nowhere near a full ~100% easy-end plateau.

Extended the grid to `rho=0.97/0.99/1.0` (Slurm `48146991` train+eval,
`48150261` notebook render) -- `1.0` is the fully trivial, zero-reasoning-
round, all-hints-given case. **Result: not a difficulty-curve-positioning
problem.** `solve_rate` plateaus around `0.28-0.34` even at `rho=1.0`,
never approaching ~1.0, while `outcome_accuracy`/`well_formed_rate` are
already saturated at `1.0` across the whole `rho>=0.9` range by epoch 50.
The model reliably produces well-formed output and calls SOLVED/
UNRESOLVED correctly, but still gets >=1 of the 24 colors wrong in ~2/3 of
even-trivial instances -- a model-capacity/precision ceiling on exact
24-node copying, not a positioning issue. Scaling model size/instances/
epochs (already flagged above) is the direct next step.

**Model-size hyperparameter scan.** To find a config that actually reaches
~100% on the near-trivial end: defined "O(1) uncolored nodes" as a fixed
small-`k` grid (`k=0..4`, i.e. `rho=(24-k)/24`) rather than a density
range, since `k` should stay a small constant independent of `n_nodes`
rather than scaling with it. New `spin_gnn.sudoku_hparam_sweep` trains one
`(d_model, num_layers, nhead, dim_feedforward, lr)` config, evals
`solve_rate` at each `k` via the existing `eval_solve_by_rho`, and reports
`mean_o1_solve_rate` (averaged over `k=0..4`) plus a per-config CSV row +
checkpoint -- mirrors `sat2_hparam_sweep.py`'s pattern from the k-SAT v2
work.

**Round 1** (`scripts/slurm/sudoku2_mlp_sweep.sbatch`, Slurm `48157903`,
array `0-15`): `num_layers in {2,4,6,8} x dim_feedforward in
{128,256,512,1024}` at fixed `d_model=64, nhead=4`, `epochs=150`,
`n_train_instances=5000`. **Both depth and width matter, and neither
alone is enough**: at `num_layers=8`, `mean_o1_solve_rate` rises
`0.698 (ff=128) -> 0.798 (ff=256) -> 0.804 (ff=512) -> 0.904 (ff=1024)`,
while at fixed small `ff`, more layers alone plateaus around `0.45-0.55`
until `num_layers>=6`. Best: `num_layers=8, dim_feedforward=1024`
(1.21M params) reaches `mean_o1_solve_rate=0.904` (`k=0..4:
0.82/1.00/0.96/0.93/0.81`), `agg_solve_rate=0.487` (vs. the original
`d_model=64/L=2` baseline's `0.005`-`0.16`) -- a large jump, confirming
scaling model capacity (not difficulty-positioning) was the right lever,
but still short of ~100%. All 16 runs finished in 4-6 min against a 3h
budget, and the best run's `val_accuracy` was still rising at the final
epoch (`0.521 -> 0.561 -> 0.571` at epochs 50/100/149) -- undertrained,
not capacity-saturated.

**Round 2** (`scripts/slurm/sudoku2_mlp_sweep2.sbatch`, Slurm `48162371`,
array `0-5`): pushed `num_layers in {8,10,12} x dim_feedforward in
{1024,2048}` at the same `d_model=64, nhead=4`, doubling `epochs`
`150->300`. **Goal reached**: `num_layers=12, dim_feedforward=1024`
(1.81M params) hits `mean_o1_solve_rate=0.980` (`k=0..4:
0.99/0.97/1.00/0.98/0.96`), i.e. ~100% across the whole near-trivial
O(1)-uncolored-nodes range, up from round 1's `0.904` at 150 epochs.
`num_layers=8, dim_feedforward=1024` (1.21M params, cheaper) is close
behind at `0.968` -- the practical pick if the extra 2 layers aren't worth
it. `agg_solve_rate` (full training-time rho range) also jumped to
`0.72` (from round 1's `0.487`, the original baseline's `0.005`).
Interestingly `dim_feedforward=2048` did *worse* than `1024` at every
`num_layers` tested (e.g. `L=12`: `0.980` at `ff=1024` vs. `0.870` at
`ff=2048`) -- likely undertrained at 300 epochs for the larger param
count, not a real capacity disadvantage; not pursued further since
`ff=1024` already met the goal. **Conclusion: depth was the dominant
lever as hypothesized, but only paired with `dim_feedforward>=1024`** --
shallow-but-wide and deep-but-narrow configs both plateaued well below
~100% (see round 1). Recommended config for future sudoku-dd transformer
work: `d_model=64, nhead=4, num_layers=8-12, dim_feedforward=1024,
epochs=300, n_train_instances=5000`.
