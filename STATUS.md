# STATUS — spin_gnn, branch `sudoku-dd` (2026-10-21)

## Current best result

**`d0256_h04_L12_ff01024_lr1e-03_pre` @300ep, agg_solve_rate = 0.847** — still the leader
after today's runs. The 0.767 previously quoted was a stale ranking from when
`mean_o1_solve_rate` was the headline; same eval (300 val instances, rho∈[0.3,0.95],
`--n-per-bin 100`), single seed. Re-derive with `scripts/sweep_leaderboard.py`. Caveat: it is
the sweep's *worst* model at rho=1.0 (`solve_rate_k0 = 0.24`), a real trade — see finding 5.

## Findings (1–3 in the committed notebooks; detail in LOG)

1. **Hint-density transition is real at N=24** (`48581589`): 0.99 → 0.44, DD floor 0.477.
2. **The model is not running DD** — probes, algorithm-ID, causal counterfactual and SAE all
   agree. Neighbour-color sensitivity is real (logit gap 12.3), localized `edited_hint`
   (L0–5) → neighbour's row (L6–9) → prediction (L8–11); no single head necessary.
3. **CORRECTED by `48861799`: the frontier *is* represented, it just never develops.** On
   AUROC (seeds 0/1/2, std ≤0.0014) `frontier_mask` peaks at **0.863**, above every label-only
   baseline including the graph-aware `degree_aug` (0.783; `not_known` 0.734, `known_probe`
   0.767), with a **+0.186 layer-0 lift** (0.676 → 0.863) comparable to `known_mask`'s +0.208.
   The old "weak/absent" reading was a balanced-accuracy artifact.
   What survives: 0.863 is reached by **epoch 5** at solve_rate 0.007 and is 0.864 at epoch
   900 — flat while solve_rate grows ~100x; only `next_color` tracks competence (0.327 →
   0.675). New: training moves the computation *earlier* (peak layer 8 → 5, L1 0.708 →
   0.776) without raising the peak.
4. **`val_nll` does not predict `agg_solve_rate` across models: r = +0.17 (31 runs)**;
   `n_params` is the best at +0.713. "val_nll fell" is not evidence of improvement.
5. **Depth costs something, independent of density** (`48863464`) — *against* my
   pre-registered prediction of flat. At rho=0.75, where realized density is **identical**
   in every bin, solve_rate falls 0.980 → 0.931 over T=1→4 (d0256; d0128 0.960 → 0.914).
   At rho=0.60 (spread 0.0012): 0.915 → 0.798 over T=2→6. All six (rho, checkpoint) cells are
   negative, r −0.71 to −0.98; single cells are only ~1.5–2.5σ, so the consistency is the
   evidence. **Discount rho=0.50** — there `r(realized_rho, T) = −0.82`, confound intact.
   Also `48854232` (d256 300→900ep) *lowered* agg 0.847 → 0.817 while raising
   `solve_rate_k0` 0.24 → 0.87: the two rho ranges genuinely trade off.

## Open problems

- Overfitting is the dominant *within-run* failure mode (solve_rate saturates ~epoch 50 while
  val_nll rises); finding 4 says val_nll is near-useless *across* runs. No dropout/weight-decay
  flag in `sudoku_hparam_sweep`.
- SAE: all 2048 features "alive" at L4–12 → criterion too weak; layer-0 `next_node` features
  are the sorted-emission confound, so deeper counts are suspect too.
- **27 `model.pt` files (267MB) are committed** (`.gitignore` had no `*.pt` until `3273932`;
  files left tracked — untracking won't shrink history, so rewriting is yours).

## Next steps

1. Read `48854233` / `48862411` below; re-run `scripts/sweep_leaderboard.py` after each.
2. Reconcile findings 3 and 5 — frontier represented but static, yet depth costs solve_rate.
   Candidate: the cost is in *chaining* colors, not in locating the frontier.
3. Settle the agg-vs-o1 metric question — `48854232` makes the trade-off concrete.
4. Seed replicate of the checkpoint-series *training* run (`48861799` gives probe noise only).
5. Untouched main task: `sudoku_mcmc.anneal` for `d=5`, `<m>` order parameter, `gamma(rho)`.

## Running

- `48854233` d256 @300ep, **n_train 5000→20000** — the first run ever to vary it (constant at
  5000 for all 31 previous). Epoch 139: val_nll 1.105 vs the incumbent's 2.41.
- `48862411` **d384** @300ep, ladder 0.720→0.773→0.847, from `r(agg, n_params) = +0.713`.
