# STATUS — spin_gnn, branch `sudoku-dd` (2026-10-21)

## Current best result

**`d0384` @300ep, `n_train=20000`, agg_solve_rate = 0.987** (`48869017`) vs **0.973** at d0256
(`48854233`), both up from 0.847. By my pre-registered ±0.02 band that gap is a **TIE** (z≈1.2 at
n=300): co-leaders until `48912134` settles it on a shared 1500-inst pool. d384×20000 also gave
k0 0.91 (predicted ≥0.95 — **missed**) and o1 0.980 (≥0.97 — met), so the two interventions are
only partly additive. Copied to `$OAK/hshunt/checkpoints/spin_gnn/sudoku_dd_d0384_..._job48869017`.
**Correction:** I called 0.847→0.973 "the identical eval"; the protocol is, the *instances are
not* — val seeds are `seed + n_train_instances`, so each n_train sees different instances and the
data4x range *contains* the 5000-run val seeds. The ~12σ gain stands (both unbiased draws); the
wording was wrong, and `48912134` is the fix.

## Findings (1–3 in the committed notebooks; detail in LOG)

1. **Hint-density transition is real at N=24** (`48581589`): 0.99 → 0.44, DD floor 0.477.
2. **The model is not running DD** — probes, algorithm-ID, counterfactual, SAE all agree.
3. **CORRECTED by `48861799`: the frontier *is* represented, it just never develops.** `frontier_mask`
   AUROC peaks at **0.863** (seeds 0/1/2, std ≤0.0014) vs 0.783 for graph-aware `degree_aug`, a +0.186 layer-0 lift — the old "weak/absent" reading was a balanced-accuracy artifact.
   Flat from **ep5** (solve_rate 0.007) to ep900 while solve_rate grows ~100x; peak layer 8 → 5.
4. **`val_nll` barely predicts `agg_solve_rate` across models, r = +0.33** (34 runs; `n_params` +0.69, `epochs` +0.61). "val_nll fell" is not by itself evidence.
5. **Depth costs something, independent of density** (`48863464`), *against* my pre-registered
   flat: at rho=0.75 realized density is **identical** across bins yet solve_rate falls 0.980 → 0.931 over T=1→4 (d0128 0.960→0.914), and at rho=0.60, 0.915 → 0.798 over T=2→6.
   All six cells negative (r −0.71…−0.98), each only ~1.5–2.5σ, so the consistency is the evidence. **Discount rho=0.50**: `r(realized_rho, T) = −0.82`, confound intact.
6. **Training-set size was the binding constraint and was never varied until today** — pinned
   at 5000 for all 31 earlier runs; 20000 gave the largest single gain in 34 runs (+0.126).
   **`d384` is the other win** (`48862411`): agg ties 0.847 but k0 0.24 → **0.99**, o1 → **0.994**.
7. **The depth cost is NOT in the token-level slots** (`48868696`; 600 inst × 3 rho × 2 ckpts).
   Teacher-forced coloring is at ceiling and flat (0.977–0.997, ≤0.01/round); selection's apparent
   *rise* with round (+0.074…+0.115) collapses to ≈+0.009 once position (≡ candidate count) is fixed. **Refutes the chaining-colors guess I put on the deck.**
   Caveat: selection scores agreement with DD's *sorted order*, not correctness; finding 5 open.

## Open problems

- "Overfitting" was likely data starvation (finding 6) and no dropout/weight-decay flag exists,
  yet both data4x runs still overfit hard (val_nll 0.957 @ep40 → 1.363 @ep300) while giving the best agg ever — finding 4 again. Where the within-run optimum sits is untested.
- SAE: all 2048 features "alive" at L4–12 → criterion too weak; layer-0 `next_node` = the
  sorted-order confound, so deeper counts are suspect too (see finding 7's caveat).
- **27 `model.pt` files (267MB) are committed** (no `*.pt` in `.gitignore` until `3273932`); untracking won't shrink history, so a rewrite is your call.

## Next steps

1. Finding 5 follow-up: attribute generation failures by where the first error lands (generation,
   not teacher forcing — that is what finding 7 ruled out).
2. If the ladder (`48913017`) shows frontier AUROC flat across models too, finding 3 generalizes
   and the next interp question is what *does* carry competence (`next_color` within a run).
3. Results notebook + slide deck for the 0.973/0.987 results are still owed.
4. Untouched main task: `sudoku_mcmc.anneal` for `d=5`, `<m>` order parameter, `gamma(rho)`.

## Running

- `48912134` **paired shared-pool agg eval** — top 4 ckpts × 1500 held-out instances at seeds 5e10,
  McNemar exact on discordant pairs. Pre-reg: both data4x beat both 5000-inst models at p<1e-6;
  d384 vs d256 data4x stays a tie. **Read this before quoting any new best.**
- `48912548` **d256 × n_train 80000**, one change from `48854233`. Pre-reg: agg a tie (no room at
  n=300), k0 ≥ 0.85 continuing 0.24→0.60, o1 ≥ 0.96. ~6h expected.
- `48913017` **probe competence ladder** — 5 ckpts spanning agg 0.816–0.986 × 3 seeds, probe
  hyperparameters identical to `48861799`. Pre-reg: frontier AUROC 0.863 ± 0.02, no trend against
  agg; >0.92 at agg 0.986 would make finding 3 a data-starved-regime artifact needing a rewrite.
