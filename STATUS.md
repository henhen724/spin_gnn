# STATUS — spin_gnn, branch `sudoku-dd` (2026-10-21)

## Current best result

**`d0256_h04_L12_ff01024_lr1e-03_pre` @300ep, agg_solve_rate = 0.847**
(`sudoku_hparam_sweep/sweep.csv`). The 0.767 previously quoted (`d0128` @900ep) was a stale
ranking from when `mean_o1_solve_rate` was the headline; same eval both times (300 val
instances, rho∈[0.3,0.95], `--n-per-bin 100`), ~3.8σ, single seed. Re-derive with
`scripts/sweep_leaderboard.py`. Caveat: d0256 is the sweep's *worst* model at rho=1.0
(`solve_rate_k0 = 0.24` vs d0064's 1.00) — outside agg's range so it doesn't inflate 0.847,
but which of the two disjoint rho ranges is the target of record is unsettled.

## Findings (1–3 written up in the committed notebooks; detail in LOG)

1. **Hint-density transition is real at N=24** (`48581589`): solve_rate 0.99 → 0.44,
   `truncated_rate = 0.000`, DD floor `rho_min ≈ 0.477` N-independent, none in random-hint.
2. **The model is not running DD** — probes, algorithm-ID, causal counterfactual and SAE all
   agree. Neighbour-color sensitivity is real (logit gap 12.3), localized `edited_hint`
   (L0–5) → neighbour's row (L6–9) → prediction (L8–11); no single head necessary.
3. **DD never appears during training** (`48580229` + `48582169`). `known_mask` 0.925 by
   **epoch 5** at solve_rate 0.007; `frontier_mask` 0.654 → 0.679 over 895 more epochs while
   solve_rate grows ~100x. Only `next_color` develops *with* competence (0.327 → 0.655,
   r=0.963). Algorithm: **pick the node by the sorted-emission convention (learned in 5
   epochs), then spend training on which color goes there.**
4. **`val_nll` does not predict `agg_solve_rate` across models: r = +0.17 (31 runs)**;
   `n_params` is the best predictor at +0.713. "val_nll fell" is therefore *not* evidence of
   improvement — this weakens my own case for the data4x run, before it reported.

## Open problems

- Depth vs density still confounded. Next measurement: bin **equal-rho** instances by `T`.
- Overfitting is the dominant *within-run* failure mode (solve_rate saturates ~epoch 50 while
  val_nll rises) but finding 4 says val_nll is near-useless *across* runs. No dropout or
  weight-decay flag in `sudoku_hparam_sweep`.
- SAE: all 2048 features "alive" at L4–12 → criterion too weak; the 23 layer-0 `next_node`
  features are the sorted-emission confound, so distrust the deeper counts too.
- **27 `model.pt` files (267MB) are committed** (`.gitignore` had no `*.pt` until `3273932`;
  patterns added, files left tracked — untracking won't shrink history, so rewriting is yours).

## Next steps

1. Read the four jobs below; re-run `scripts/sweep_leaderboard.py` after each.
2. Equal-rho / varied-`T` eval to disentangle depth from density.
3. Settle the agg-vs-o1 metric question: which rho range is the target of record?
4. Seed replicate of the checkpoint-series *training* run (`48861799` gives probe noise only).
5. Untouched main task: `sudoku_mcmc.anneal` for `d=5`, `<m>` order parameter, `gamma(rho)`.

## Running / queued

- `48854232` d256 @**900ep** (vs the 0.847 incumbent). Predicted not to beat it. Epoch 616:
  train_nll 0.158, val_nll 2.80 and rising, val_acc 0.564 — overfitting as predicted.
- `48854233` d256 @300ep, **n_train 5000→20000**. Epoch 139: val_nll **1.105** (vs 2.41),
  val_acc 0.618. First run ever to vary `n_train_instances` (constant at 5000 for all 31).
- `48862411` **d384** @300ep, continuing the ladder 0.720→0.773→0.847. If agg rises while
  `solve_rate_k0` falls further, the two rho ranges are in real tension.
- `48861799` probe rescore: **AUROC + average precision**, seeds 0/1/2, same snapshots and
  splits as `48582169`. Balanced accuracy thresholds a BCE probe at 0, which on a
  ~15%-positive target ranks by how freely positives are predicted: `1 − known` scores 0.734
  (*above* the model's 0.679), a matched known-bits probe 0.588 (*below*). Quote nothing
  about frontier until this lands.
