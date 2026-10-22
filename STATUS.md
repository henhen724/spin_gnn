# STATUS — spin_gnn, branch `sudoku-dd` (2026-10-22)

## Current best result

**`d0256` @300ep, `n_train=80000` — shared-pool solve_rate 0.9960 [0.9913, 0.9982]** (`48912548` trained, `48999377` scored), up from 0.9847 for `d0256 x 20000`: paired delta **+0.0113**, 27 discordant instances **22–5**, **exact p = 0.0015**. Archived to `$OAK/hshunt/checkpoints/spin_gnn/sudoku_dd_d0256_L12_ntrain80000_ep300_sharedpool0.9960_job48912548/`.

**Metric of record: the shared pool** (`spin_gnn.sudoku_agg_eval`, 1500 instances at `seed_start=5e10`, disjoint from every training set, every checkpoint scored on it, compared paired with McNemar exact; Wilson intervals). Per-run 300-instance `agg_solve_rate` is retired — it validates on `seed + n_train_instances`, so each `n_train` sees different instances, and it had the top two backwards on 10-21. `48999377` wrote one authoritative 7×1500 matrix (`agg_paired_full_grid.csv` + `_per_instance.npz`), so any future pair is testable offline without a GPU.

**Width × data grid, shared-pool solve_rate** (all 300 ep, L12/ff1024/lr1e-3):

|        | n=5000 | n=20000 | n=80000 |
|--------|--------|---------|---------|
| d0128  | 0.7707 | 0.9733  |   —     |
| d0256  | 0.8200 | 0.9847  | **0.9960** |
| d0384  | 0.8760 | 0.9720  |   —     |

**`d0128 x 20000` (4.0M params) ties `d0384 x 20000` (16.7M)** — 0.9733 vs 0.9720, overlapping intervals. At this scale data buys more than width. The six previously scored checkpoints reproduced their recorded rates to 4 decimals (pool is deterministic, nothing drifted).

## Findings (notebooks committed; detail in LOG)

1. **Hint-density transition is real at N=24** (`48581589`): 0.99 → 0.44, DD floor 0.477.
2. **The model is not running DD** — probes, algorithm-ID, counterfactual, SAE all agree.
3. **The frontier probe measures width, not competence** (rewritten, `48935109` + `48999377`). Across all 7 cells, `frontier_mask` peak AUROC has Spearman **+0.25** with shared-pool solve_rate but **+0.85** with `d_model`. Concretely: d256×5000 hits 0.9071 while solving 0.820; d128×20000 only 0.8831 while solving 0.973; the new best model **fell** to 0.9067 (from 0.9162) while its solve rate rose 0.985 → 0.996. Within-width data effects are small and not even signed consistently (d128 0.8578→0.8831, d256 0.9162→**0.9067**, d384 0.9163→0.9381). The old 0.863 → 0.907 "gap" was width. Frontier peaks mid-stack (L5–L7).
4. **`val_nll` barely predicts solve_rate across models, r = +0.33** (34 runs; `n_params` +0.69, `epochs` +0.61). "val_nll fell" is not by itself evidence.
5. **Depth costs something, independent of density** (`48863464`): at rho=0.75 realized density is identical across bins yet solve_rate falls 0.980 → 0.931 over T=1→4; at rho=0.60, 0.915 → 0.798 over T=2→6. All six cells negative, each only 1.5–2.5σ. Discount rho=0.50 (`r(realized_rho, T) = −0.82`).
6. **Training-set size was the binding constraint, unvaried until 10-20** — pinned at 5000 for all 31 earlier runs. It is still the strongest lever and **has not saturated at 80000**.
7. **The depth cost is NOT in the token-level slots** (`48868696`). Teacher-forced coloring is at ceiling and flat; selection's rise with round collapses to ≈+0.009 once position is fixed. Refutes the chaining-colors guess. Caveat: selection scores agreement with DD's *sorted* order, not correctness.
8. **What tracks competence is `next_node`** (strengthened to 7 cells). Peak accuracy has Spearman **+0.96** with shared-pool solve_rate and steps up with data at every width: d128 0.738→0.951, d256 0.789→0.964→**0.990**, d384 0.810→0.957. One inversion, between d384×20000 and d128×20000, whose solve rates differ by 0.0013 and are not separable. Peaks at the **last** layer, unlike the frontier.

## Open problems

- **The easy end collapsed while the full range improved.** `48912548`: `solve_rate_k0` 0.60 → **0.07**, `mean_o1` 0.918 → **0.776**, with shared pool at 0.9960. Width did the same on 10-18, so it is not specific to the data axis. k0 = 0.07 alongside 0.996 is a contradiction — decoding failure or representation failure, untested.
- Pre-registration scorecard for 10-22: **4 held, 3 missed** (held: ≥0.9847 with positive delta; next_node ≥0.96 → 0.9895; d128×20000 frontier in [0.855,0.895] → 0.8831; control reproduces 0.863±0.015 → 0.8578. Missed: predicted range [0.980,0.995] → 0.9960 just above; k0 ≥0.85 → 0.07; mean_o1 ≥0.96 → 0.776). Both remaining misses are the easy-end problem.
- SAE: all 2048 features "alive" at L4–12 → criterion too weak; layer-0 `next_node` = the sorted-order confound, so deeper counts are suspect (finding 7's caveat).
- **27 `model.pt` files (267MB) are committed** (no `*.pt` in `.gitignore` until `3273932`); untracking won't shrink history, so a rewrite is your call.
- Whether `d0128 x 20000` deserves an `$OAK` archive too (4.0M params tied with the 16.7M model) — your call, archiving is manual by policy.

## Next steps

1. **Push the data axis again: `d0256 x 320000`**, one change, now that 80000 did not saturate. Pre-register the shared-pool range before it runs.
2. Resolve the easy-end collapse: is k0=0.07 a decoding failure (greedy/ordering) or a representation failure? Inspect where the first error lands on k=0 instances.
3. Finding 5 follow-up: attribute generation failures by where the first error lands under **free** generation (finding 7 ruled out the teacher-forced slots).
4. Untouched main task: `sudoku_mcmc.anneal` for `d=5`, `<m>` order parameter, `gamma(rho)` for `d=4,6,8`.

## Running

- None. `48912548` (train, 5:33:11), `48935109` (d128×20000 + probes + eval, 1:07:26) and `48999377` (7-checkpoint pool + probes, 10:27) all COMPLETED, ExitCode 0:0.
