# STATUS — spin_gnn, branch `sudoku-dd` (2026-10-21)

## Current best result

`d0128_h04_L12_ff01024_lr1e-03_pre` @900ep (Slurm `48513926_1`), **agg_solve_rate = 0.767**.
Still unbeaten; nothing was trained this session, so there is no new number against it.

## The three findings

1. **The hint-density transition is real and measurable at N=24** (`48581589`): solve_rate
   0.99 → 0.44 monotonically, `truncated_rate = 0.000` in every bin. DD's floor
   `rho_min ≈ 0.477` is N-independent (`48581590`). The independent-**random**-hint ensemble
   has no transition at any N (`48579545`) — retires the "scale N to ~1700 tokens" plan.
   *Not established:* depth `T` vs density `rho`, collinear by construction
   (`r(solve,T) = −0.966` vs `r(solve,rho) = +0.944`).
2. **The model is not running DD.** Four lines agree: probes (`48580216`), algorithm ID on
   disagreements (`48582589`, loses to `ascending_after` 0.334/0.361), causal counterfactual
   (`48583444`, flip rate 0.267 vs 0.25 chance), SAE (`48583439`, no `frontier` family).
   Neighbour-color sensitivity *is* real (logit gap moves 12.3) and localized: `edited_hint`
   (L0–5) → **recolored neighbour's** graph row (L6–9) → prediction position (L8–11); the
   predicted node's own row ~0.00 throughout. Head ablation (`48585025`): no single head
   necessary, top 5 in the L6–L10 window.
3. **DD never appears during training** (`48580229` + `48582169`, new). `known_mask` hits
   0.925 by **epoch 5**, when solve_rate is 0.007; `frontier_mask` goes 0.654 → 0.679 over
   895 more epochs while solve_rate grows ~100x. The one target developing *with* competence
   is `next_color` (0.327 → 0.655, `r = 0.963`); `next_node`, `known_mask`, `remaining_mask`
   all finish by epoch 5. Best algorithm description: **pick the node by the sorted-emission
   convention (learned in 5 epochs), then spend training on which color goes there.**
   One trajectory, no seed replicates → an observation, not yet a finding.

Delivered: `notebooks/sudoku_when_during_training_2026-10-21/` (`7c50b99`), notebook + 9-page
deck, on OneDrive; supersedes the `sudoku_interp_2026-10-19` deck.

## Open problems

- Depth vs density still confounded. Next measurement: bin **equal-rho** instances by `T`.
- **`frontier_mask`'s 0.679 is uninterpretable, not weak evidence** (`48825742`, labels only,
  no model). The deducibility-free rule `1 − known` scores **0.734** (above the model), but a
  procedure-matched BCE probe on the known bits alone scores **0.588** (below it) — balanced
  accuracy on a ~15%-positive target is miscalibrated both ways (`shuffled = 0.5000`, so the
  pooling is sound). Fix: an AUC column on the existing sweep. Quote no number until then.
- Training is ~18x longer than needed: solve_rate saturates by epoch 50 (0.787) at the level
  it still has at epoch 900 (0.740) while `val_nll` rises monotonically 1.138 → 2.136.
- SAE: all 2048 features "alive" at L4–12 → too weak a criterion. The 23 layer-0 `next_node`
  features cannot compute anything — sorted-emission confound; distrust deeper counts too.
- **27 `model.pt` files (267MB) are committed**; `.gitignore` had no `*.pt` until `3273932`.
  Patterns added, files left tracked — untracking won't shrink history, so rewriting is yours.

## Next steps

1. AUC rescore of the probe sweep, so finding 2's probe line becomes quotable.
2. Equal-rho / varied-`T` eval to disentangle depth from density — last open transition question.
3. `d_model=256` @900ep — the untried config, and the only shot queued at beating 0.767.
4. Untouched main task: `sudoku_mcmc.anneal` for `d=5`, `<m>` order parameter, `gamma(rho)`
   for `d=4,6,8`.

## Running / queued

No `spin_gnn_*` jobs. (9 other jobs on the account — `mls_*`, `wr_*`, `h43moments`,
`eta10conv` — belong to other projects' sessions; leave them alone.)
