# STATUS — spin_gnn, branch `sudoku-dd` (2026-10-21)

## Current best result — CORRECTED 2026-10-21

**`d0256_h04_L12_ff01024_lr1e-03_pre` @300ep, agg_solve_rate = 0.847**
(`sudoku_hparam_sweep/sweep.csv`). The 0.767 we had been quoting (`d0128` @900ep,
`48513926_1`) was never best on this metric — a stale ranking from when `mean_o1_solve_rate`
was the headline (d0064 wins that, 0.980), not redone when the metric changed. Same eval
both times: 300 held-out val instances over rho∈[0.3,0.95], `--n-per-bin 100`, default
`--eval-seed-start`; gap ~3.8σ, single seed.
Caveat: d0256 has `solve_rate_k0 = 0.24`, failing the degenerate rho=1.0 input (fully hinted,
nothing to deduce) that d0064 gets 0.99 on. That rho is *outside* agg's range so it does not
inflate 0.847, but `mean_o1_solve_rate` is worse (0.842 vs 0.968). "Best" depends on which of
the two disjoint rho ranges you score — worth settling explicitly.

## Findings (1–2 fully written up in the committed notebooks; see LOG for detail)

1. **Hint-density transition is real at N=24** (`48581589`): solve_rate 0.99 → 0.44,
   `truncated_rate = 0.000` every bin; DD floor `rho_min ≈ 0.477` N-independent. Random-hint
   ensemble has no transition at any N. *Open:* depth `T` vs density `rho` are collinear.
2. **The model is not running DD.** Probes, algorithm-ID, causal counterfactual and SAE all
   agree. Neighbour-color sensitivity is real (logit gap 12.3) and localized to `edited_hint`
   (L0–5) → recolored neighbour's row (L6–9) → prediction (L8–11); no single head necessary.
3. **DD never appears during training** (`48580229` + `48582169`). `known_mask` 0.925 by
   **epoch 5** at solve_rate 0.007; `frontier_mask` 0.654 → 0.679 over 895 more epochs while
   solve_rate grows ~100x. The only target developing *with* competence is `next_color`
   (0.327 → 0.655, r=0.963). Algorithm: **pick the node by the sorted-emission convention
   (learned in 5 epochs), then spend training on which color goes there.** One trajectory,
   no seed replicates → an observation, not yet a finding.

## Open problems

- Depth vs density still confounded. Next measurement: bin **equal-rho** instances by `T`.
- **`frontier_mask`'s 0.679 is uninterpretable, not weak evidence** (`48825742`, labels only).
  `1 − known` scores **0.734** (above the model), a procedure-matched probe on the known bits
  **0.588** (below it) — balanced accuracy on a ~15%-positive target is miscalibrated both
  ways (`shuffled = 0.5000`, so pooling is sound). Fix: AUC column. Quote nothing until then.
- Overfitting is the dominant failure mode: solve_rate saturates ~epoch 50 while `val_nll`
  rises monotonically after. No dropout/weight-decay flag exists in `sudoku_hparam_sweep`.
- SAE: all 2048 features "alive" at L4–12 → too weak a criterion. The 23 layer-0 `next_node`
  features cannot compute anything — sorted-emission confound; distrust deeper counts too.
- **27 `model.pt` files (267MB) are committed**; `.gitignore` had no `*.pt` until `3273932`.
  Patterns added, files left tracked — untracking won't shrink history, so rewriting is yours.

## Next steps

1. Read `48854232` / `48854233` (below); re-rank `sweep.csv` by agg_solve_rate afterwards.
2. AUC rescore of the probe sweep, so finding 2's probe line becomes quotable.
3. Equal-rho / varied-`T` eval to disentangle depth from density.
4. Settle the agg-vs-o1 metric question: which rho range is the target of record?
5. Untouched main task: `sudoku_mcmc.anneal` for `d=5`, `<m>` order parameter, `gamma(rho)`
   for `d=4,6,8`.

## Running / queued

Both change exactly one thing vs the 0.847 incumbent (d0256 @300ep, n_train=5000):
- `48854232` d256 @**900ep** (epochs 300→900). Hypothesis: will NOT beat 0.847 — the same
  change at d0128 gave 0.773→0.767 while val_nll rose, and d0256 is already at val_nll 2.41
  by epoch 300. A falsification run, submitted because it was explicitly requested.
- `48854233` d256 @300ep, **n_train 5000→20000**. Hypothesis: this is the lever that works,
  since every diagnostic says overfitting and data is the only regularizer available.
  If val_nll falls but agg_solve_rate does not, 0.847 is a real ceiling → go to architecture.
