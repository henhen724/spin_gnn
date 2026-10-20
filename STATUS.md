# STATUS — spin_gnn, branch `sudoku-dd` (2026-10-19)

## Current best result

`d0128_h04_L12_ff01024_lr1e-03_pre` @900ep (Slurm `48513926_1`), **agg_solve_rate = 0.767**.
Nothing beat it this session: scratchpad variant 0.740 (`48579019`), i.e. −0.027.

## Delivered today

Deck + executed notebook in `notebooks/sudoku_interp_2026-10-19/` (commit `4d5c3bb`),
uploaded to OneDrive. Both carry the caveats below rather than only the clean story.

## The two findings

1. **The hint-density transition is real and already measurable at N=24** (`48581589`):
   solve_rate 0.99 → 0.44 monotonically, `truncated_rate = 0.000` in *every* bin, so it is
   not a generation-budget artifact. DD's floor `rho_min ≈ 0.477` is N-independent
   (converged by N=96, `48581590`). The independent-**random**-hint ensemble has no
   transition at any N (`48579545`: flat 0.64–0.78, highest at both extremes) — this
   retires the old "scale N to ~1700 tokens" plan.
   *Not established:* whether depth `T` or density `rho` controls the collapse.
   `r(solve,T) = −0.966` vs `r(solve,rho) = +0.944` — collinear by construction.
2. **The model is not running DD.** Four independent lines agree:
   - probes (`48580216`): `known_mask` 0.91 vs `frontier_mask` 0.67 (layer-0 control 0.57)
   - algorithm ID on disagreements only (`48582589`): loses to `ascending_after`
     (DD 0.334 / heuristic 0.361); worse at d0064 (0.238 / 0.415)
   - causal counterfactual (`48583444`): flip rate 0.267 vs 0.25 chance
   - SAE (`48583439`, now sparse, `hit_target_l0=True`): no `frontier` family at any layer
   Best description: **bookkeeping of which nodes are colored + the sorted-emission
   convention + sub-threshold sensitivity to neighbour colors.**
   That sensitivity is real (logit gap moves 12.3) and localized: `edited_hint` (L0–5) →
   **recolored neighbour's** graph row (L6–9) → prediction position (L8–11). The predicted
   node's own row carries ~0.00 throughout — color-to-node binding, not DD's neighbour-list
   join. Head ablation (`48585025`): no single head necessary (worst L10H3 keeps 0.807),
   and the top 5 sit in exactly the L6–L10 handoff window.

## Running / queued

- `48580229` spin_gnn_sudoku2_series — checkpoint series, RUNNING ~47 min
- `48582169` spin_gnn_sudoku2_serprobe — PENDING, gated on the above. **This is the
  "when during training" answer**; the deck has a placeholder page for it.

## Open problems

- Depth vs density is confounded. Next measurement: bin **equal-rho** instances by `T`.
- SAE `n_dead_features = 0` with all 2048 alive at L4–12 → "alive" is a weak criterion
  (fires ≥once in 26,448 positions); sparse per position, not well concentrated.
- The 23 `next_node` SAE features at **layer 0** cannot be computing anything (raw
  token+position embedding) — they are position features riding the sorted-emission
  confound. Treat deeper `next_node` counts with the same suspicion.
- `frontier_mask` at 0.67 is weak but above the 0.57 control; unclear if that residual is
  real structure or probe leakage from `known_mask`.

## Next steps

1. Read `48582169` when it lands → "when during training does DD appear?" → rebuild deck
   (the page is already wired, `--series-probes`).
2. Equal-rho / varied-`T` eval to disentangle depth from density.
3. Untouched main task: `sudoku_mcmc.anneal` for `d=5`, `<m>` order parameter,
   `gamma(rho)` for `d=4,6,8`; `d_model=256` @900ep.
