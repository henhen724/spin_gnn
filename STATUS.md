# STATUS — spin_gnn, branch `sudoku-dd` (2026-10-21)

## Current best result

**`d0256` @300ep with `n_train=20000`, agg_solve_rate = 0.973** (`48854233`), up from 0.847
on the identical eval (300 val instances, rho∈[0.3,0.95], `--n-per-bin 100`, single seed; SE ≈
0.009, so ~12σ). Also k0 0.24 → 0.60, o1 0.842 → 0.918, val_nll 2.41 → 1.27; k0 is what's left.

## Findings (1–3 in the committed notebooks; detail in LOG)

1. **Hint-density transition is real at N=24** (`48581589`): 0.99 → 0.44, DD floor 0.477.
2. **The model is not running DD** — probes, algorithm-ID, counterfactual, SAE all agree.
3. **CORRECTED by `48861799`: the frontier *is* represented, it just never develops.** On
   AUROC (seeds 0/1/2, std ≤0.0014) `frontier_mask` peaks at **0.863**, above every label-only
   baseline including graph-aware `degree_aug` (0.783; `not_known` 0.734, `known_probe` 0.767),
   a **+0.186 layer-0 lift**; the old "weak/absent" reading was a balanced-accuracy artifact.
   0.863 is reached by **epoch 5** at solve_rate 0.007 and is 0.864 at ep900 — flat while
   solve_rate grows ~100x; only `next_color` tracks competence. Training moves the computation
   *earlier* (peak layer 8 → 5) without raising the peak.
4. **`val_nll` barely predicts `agg_solve_rate` across models: r = +0.33 (34 runs)**;
   `n_params` +0.69, `epochs` +0.61. "val_nll fell" is not by itself evidence of improvement.
5. **Depth costs something, independent of density** (`48863464`) — *against* my pre-registered
   flat. At rho=0.75, realized density **identical** in every bin, solve_rate falls 0.980 →
   0.931 over T=1→4 (d0256; d0128 0.960 → 0.914); at rho=0.60 (spread 0.0012) 0.915 → 0.798 over
   T=2→6. All six cells negative, r −0.71…−0.98; single cells only ~1.5–2.5σ, so the consistency
   is the evidence. **Discount rho=0.50** — `r(realized_rho, T) = −0.82`, confound intact.
6. **Training-set size was the binding constraint and was never varied until today** — pinned
   at 5000 for all 31 earlier runs; 20000 gave the largest single gain in 34 runs (+0.126).
   **`d384` is the other win** (`48862411`): agg ties 0.847 but k0 0.24 → **0.99**, o1 →
   **0.994**, so agg-vs-o1 is not a trade for width. `48869017` tests whether the two add.
7. **The depth cost is NOT in the token-level slots** (`48868696`; 600 inst × 3 rho × 2 ckpts).
   Teacher-forced coloring is at ceiling and flat (0.977–0.997, ≤0.01/round); selection's
   apparent *rise* with round (+0.074…+0.115) collapses to ≈+0.009 once position (≡ candidate
   count) is fixed. **Refutes the chaining-colors guess I put on the deck.** Caveat: selection
   scores agreement with DD's *sorted order*, not correctness (0.61 vs solve_rate 0.86), so it
   is a weak competence measure. Finding 5 stays unexplained; next test needs generation.

## Open problems

- "Overfitting" was likely data starvation (finding 6); no dropout/weight-decay flag exists.
- SAE: all 2048 features "alive" at L4–12 → criterion too weak; layer-0 `next_node` = the
  sorted-order confound, so deeper counts are suspect too (see finding 7's caveat).
- **27 `model.pt` files (267MB) are committed** (no `*.pt` in `.gitignore` until `3273932`);
  untracking won't shrink history, so a rewrite is your call.

## Next steps

1. Read `48869017` below; re-run `sweep_leaderboard.py` after it.
2. If d384×20000 is not additive on k0, try `n_train 80000` at d256 instead of more width.
3. Re-run the interp probes on the 0.973 model — all interp above is from the data-starved
   regime (5000 instances).
4. Finding 5 follow-up: attribute generation failures by where the first error lands.
5. Untouched main task: `sudoku_mcmc.anneal` for `d=5`, `<m>` order parameter, `gamma(rho)`.

## Running

- `48869017` **d384 × n_train 20000** — one change from the new best. Pre-registered: agg
  above 0.973 but barely (little room), k0 ≥ 0.95, o1 ≥ 0.97. Anything inside 0.973 ± 0.02
  is a **tie** and must be reported as one.
- Correction deck sent (`19f5716`); its page-6 guess is already refuted by finding 7.
