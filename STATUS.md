# STATUS — spin_gnn, branch `sudoku-dd` (2026-10-21)

## Current best result

**`d0256` @300ep, `n_train=20000` — solve_rate 0.9847 [0.9771, 0.9898]** on the shared 1500-instance pool (`48912134`), ahead of `d0384 x 20000` at 0.9720 [0.9624, 0.9792]: paired McNemar on the 53 instances where they disagree is 36–17 for d256, **exact p = 0.013**. This **reverses** their own-val ranking (0.973 d256 vs 0.987 d384), which is why the 300-instance per-run `agg_solve_rate` is retired as the metric of record. Both are copied to `$OAK/hshunt/checkpoints/spin_gnn/`.

**The metric of record is now the shared pool** (`spin_gnn.sudoku_agg_eval`, 1500 instances at `seed_start=5e10`, disjoint from every training set, scored by every checkpoint and compared paired). Per-run `agg_solve_rate` validates on `seed + n_train_instances`, so each `n_train` sees *different* instances and the data4x training range even covers the 5000-run val seeds; at p≈0.98 its SE is 0.008, so nothing above ~0.97 can be ranked there. Raw solved/not matrices go to `*_per_instance.npz` so any pair can be re-tested without re-running the job.

**Shared-pool leaderboard:** d256×20000 0.9847 · d384×20000 0.9720 · d384×5000 0.8760 [0.8584, 0.8917] · d256×5000 0.8200 [0.7997, 0.8386]. The last two were a "tie" at n=300 and are not (disjoint 95% intervals).

## Findings (1–3 in the committed notebooks; detail in LOG)

1. **Hint-density transition is real at N=24** (`48581589`): 0.99 → 0.44, DD floor 0.477.
2. **The model is not running DD** — probes, algorithm-ID, counterfactual, SAE all agree.
3. **Frontier is represented early and strengthens only slightly** (`48861799` + ladder `48913017`). `frontier_mask` peak AUROC is 0.907–0.938 across 5 checkpoints spanning 0.82–0.985 solve_rate, not the flat 0.863 I reported — but **0.863 came from a `d0128` series, so width is confounded with data**. Clean within-width: d256 0.907→0.916, d384 0.916→0.938 (seeds 0/1/2, spread ±0.001; layer-0 control 0.676). A 10× cut in failures buys +0.03 AUROC, and the frontier peaks mid-stack (L5–L7; L12 only for d384×20000), so "represented but underused" survives; "flat" does not.
4. **`val_nll` barely predicts solve_rate across models, r = +0.33** (34 runs; `n_params` +0.69, `epochs` +0.61). "val_nll fell" is not by itself evidence.
5. **Depth costs something, independent of density** (`48863464`), *against* my pre-registered flat: at rho=0.75 realized density is **identical** across bins yet solve_rate falls 0.980 → 0.931 over T=1→4 (d0128 0.960→0.914), and at rho=0.60, 0.915 → 0.798 over T=2→6. All six cells negative (r −0.71…−0.98), each only ~1.5–2.5σ, so the consistency is the evidence. **Discount rho=0.50**: `r(realized_rho, T) = −0.82`, confound intact.
6. **Training-set size was the binding constraint and was never varied until 10-20** — pinned at 5000 for all 31 earlier runs; 20000 gave the largest single gain in 34 runs (+0.126 own-val, and 0.82→0.985 on the shared pool). **`d384` is a separate win** (`48862411`): agg ties 0.847 but k0 0.24 → **0.99**, o1 → **0.994**.
7. **The depth cost is NOT in the token-level slots** (`48868696`; 600 inst × 3 rho × 2 ckpts). Teacher-forced coloring is at ceiling and flat (0.977–0.997, ≤0.01/round); selection's apparent *rise* with round (+0.074…+0.115) collapses to ≈+0.009 once position (≡ candidate count) is fixed. **Refutes the chaining-colors guess I put on the deck.** Caveat: selection scores agreement with DD's *sorted order*, not correctness; finding 5 open.
8. **What tracks competence is `next_node`, not the frontier** (`48913017`). Peak `next_node` accuracy (0.789 / 0.810 / 0.957 / 0.964) orders the four models **exactly** as the shared pool does, reversal included; `next_color` gets 3 of 4 (0.69→0.83); `frontier_mask` gets it wrong (it ranks d384×20000 first). Both next-step probes **step up** with `n_train` rather than sloping, and both peak at the **last** layer while the frontier peaks mid-stack — so late layers carry the competence gain.

## Open problems

- Pre-registration scorecard for today: **2 held, 3 wrong**, and all three misses were me assuming a difference was unresolvable. Keep pre-registering, but stop predicting ties as the safe answer.
- "Overfitting" was data starvation (finding 6): `48912548` at `n_train=80000` has a train/val gap of **+0.042** at epoch 125 (0.8684 / 0.9086) versus **+0.817** for the data4x runs at ep300, and is already below their val minimum of 0.957. Where the within-run optimum sits is still untested.
- SAE: all 2048 features "alive" at L4–12 → criterion too weak; layer-0 `next_node` = the sorted-order confound, so deeper counts are suspect too (see finding 7's caveat).
- **27 `model.pt` files (267MB) are committed** (no `*.pt` in `.gitignore` until `3273932`); untracking won't shrink history, so a rewrite is your call.

## Next steps

1. Re-score `48912548` on the **shared pool**, not its own 300-instance val, and against its pre-registration (agg a tie, k0 ≥ 0.85, o1 ≥ 0.96).
2. **Train `d0128 × n_train=20000`** — the one run that separates width from data in the 0.863 → 0.907 frontier gap, and the only thing blocking a clean statement of finding 3.
3. Finding 5 follow-up: attribute generation failures by where the first error lands (free generation, not teacher forcing — that is what finding 7 ruled out).
4. Results **notebook** for the paired eval (the slide deck is built and sent; the notebook is not).
5. Untouched main task: `sudoku_mcmc.anneal` for `d=5`, `<m>` order parameter, `gamma(rho)` for `d=4,6,8`.

## Running

- `48912548` **d256 × n_train 80000**, one change from `48854233` (which is the new best). Epoch 125/300 at the last check, ~1.21 min/epoch, ~3h30 of training left inside an 8h limit. Curve healthy: val_nll flat at 0.906–0.909 over the last several epochs with almost no train/val gap, i.e. 80000 instances largely eliminated the overfitting. Pre-reg: own-val agg a tie (no room at n=300), k0 ≥ 0.85 continuing 0.24→0.60, o1 ≥ 0.96.
