# STATUS — spin_gnn, branch `sudoku-dd` (2026-10-24)

## Current best result

**`d0256` @300ep, `n_train=80000` — shared-pool solve_rate 0.9960 [0.9913, 0.9982]** (`48912548` trained, `48999377` scored), up from 0.9847 for `d0256 x 20000`: paired delta **+0.0113**, 27 discordant **22–5**, **exact p = 0.0015**. Archived to `$OAK/hshunt/checkpoints/spin_gnn/sudoku_dd_d0256_L12_ntrain80000_ep300_sharedpool0.9960_job48912548/`.

**Metric of record: the shared pool** (`spin_gnn.sudoku_agg_eval`, `seed_start=5e10`, disjoint from every training set, paired McNemar exact + Wilson intervals). Per-run 300-instance `agg_solve_rate` is retired. `49430217` re-scored at **n=6000** (a strict superset — instance *i* is seeded at `seed_start+i`): 0.9955 [0.9935, 0.9969] vs 0.9837, delta **+0.0118**, 117 discordant 94–23, p = 2e-11 — reproducing the n=1500 delta at 4x resolution.

**Width × data grid, shared-pool solve_rate** (300 ep, L12/ff1024/lr1e-3; `n=6000` where measured):

|        | n_train=5000 | 20000 | 80000 |
|--------|--------|---------|---------|
| d0128  | 0.7707 | 0.9750  |   —     |
| d0256  | 0.8200 | 0.9837  | **0.9955** |
| d0384  | 0.8760 | 0.9740  |   —     |

At n=6000 the d0128/d0384 tie at 20000 holds (0.9750 vs 0.9740, Δ=0.001) but **both are now resolvably below d0256** (−0.0087 p=3e-4; −0.0097 p=5e-5), which n=1500 could not separate. `d_model=256` is a genuine optimum at this budget, not a shrug.

## Findings (notebooks committed; detail in LOG)

1. **There is no sharp critical density** (`notebooks/sudoku_finite_size_2026-10-24`, 7-page deck). Corrects the old finding 1, which read as if DD transitions at N=24 — `dd_resolved_frac ≡ 1.0` there and the 0.99→0.44 curve is the *transformer* degrading against a DD baseline pinned at 1.0. Training ensemble (`removal`) is DD-solvable by construction; `random`-hint solvability threshold **runs off to rho=1** (midpoint 0.9296→0.9875 for N=24→384, unhinted nodes at crossing 1.7→4.8). T ~ N^a at **every** density, a = +0.18 (rho=0.95) → +0.38 (rho=0.50), never 0, so rho_min≈0.478 is where greedy removal stops, not a critical point. **Depth only arrives by raising N.**
2. **The model is not running DD** — probes, algorithm-ID, counterfactual, SAE all agree.
3. **The frontier probe measures width, not competence** (`48935109`+`48999377`): peak AUROC Spearman **+0.25** with solve_rate, **+0.85** with `d_model`. The new best model *fell* to 0.9067 while solve_rate rose to 0.996.
4. **`val_nll` barely predicts solve_rate across models, r = +0.33** (34 runs). "val_nll fell" is not evidence.
5. **Depth costs something, independent of density** (`48863464`): at rho=0.75, realized density identical across bins yet solve_rate falls 0.980→0.931 over T=1→4; rho=0.60, 0.915→0.798 over T=2→6. Discount rho=0.50 (`r(realized_rho,T) = −0.82`).
6. **Training-set size was the binding constraint** — pinned at 5000 for 31 runs; still the strongest lever and **not saturated at 80000**.
7. **The depth cost is NOT in the token-level slots** (`48868696`). Refutes the chaining-colors guess.
8. **What tracks competence is `next_node`** — peak accuracy Spearman **+0.96** with solve_rate (d0256: 0.789→0.964→0.990). Peaks at the **last** layer, unlike the frontier.
9. **The easy-end collapse was an OOD artifact, now closed** (`49430217`). `removal_path` snapshots at `floor(rho*N)`, so `rho_max=0.95` at N=24 is **22** hints: k=0 (24) and k=1 (23) are outside training support and `mean_o1` averaged two OOD bins with three in-distribution ones. Best model scores **exactly 1.000, 0/300 failures, at every in-distribution hint count** (22/21/20/18/16) and only breaks above the ceiling (0.860 at 23, 0.060 at 24). The 24-hint failures are `empty BEGIN_ROUND` parse errors — the model opening a round on an instance with nothing to deduce, i.e. exactly the predicted signature, which the first classifier charged to `malformed`. **Caveat:** `d0384 x 5000` declines in-distribution too (0.970 at 18, 0.933 at 16) while handling 24 hints fine (0.983) — in-distribution competence is model-dependent, so this is not an ensemble-wide guarantee.

## Open problems

- Whether network depth must grow with reasoning depth is **open and being measured** (49433454). Finding 2 predicts L*(T) flat; the serial hypothesis predicts it rising.
- SAE: all 2048 features "alive" at L4–12 → criterion too weak; layer-0 `next_node` = sorted-order confound.
- **27 `model.pt` files (267MB) are committed** (no `*.pt` in `.gitignore` until `3273932`); untracking won't shrink history, so a rewrite is your call.
- Whether `d0128 x 20000` deserves an `$OAK` archive (4.0M params, now resolvably *below* d0256 — arguably no longer worth it).

## Next steps

1. Score 49433453 / 49433454 against their pre-registrations; then submit the **N=48 arm of the layer scan** if the recipe transferred, and a reduced-L arm at N=96.
2. `d0256 x 320000`, one change, now that 80000 did not saturate — held until the N axis is characterised, since it competes for the same GPU.
3. Finding 5 follow-up: where the first error lands under **free** generation (finding 7 ruled out the teacher-forced slots).
4. Untouched main task: `sudoku_mcmc.anneal` for `d=5`, `<m>` order parameter, `gamma(rho)` for `d=4,6,8`.

## Running

- **49433453** `sud2n48base` — first N=48 training run (d0256/L12, n_train=20000, 300 ep) + depth eval. 16 h limit.
- **49433454** `sud2n24Lscan` — `num_layers` in {1,2,3,4,6,8,12} at N=24, ff pinned at 1024 so only depth varies, + `sudoku_depth_eval`. 10 h limit.
- **49431733** `ddctrlsweep` — CPU-only (degree, n_colors) sweep, predictions (a)–(e) in the sbatch header. 3 h limit.
- `49430217` (easy end + n=6000 pool) COMPLETED 00:25:57, ExitCode 0:0.
