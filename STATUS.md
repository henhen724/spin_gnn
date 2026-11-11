# STATUS — spin_gnn, branch `sudoku-dd` (2026-11-11 04:45, stage S answered, stage T running)

**Findings 1–61 in full, and every line of detail trimmed out of this file, are in `FINDINGS.md`** — 107 KB: **grep it, never read it whole.** Session history is `LOG.md`: search it, never read it in full. This file is the handoff and is deliberately short.

## The objective, restated 2026-11-10 — read this before any rung table

**THE GOAL IS A CONCEPTUALLY SIMPLE MODEL THAT MIGHT BE THEORETICALLY ANALYZABLE, EXPLICITLY NOT A MODEL WITH THE FEWEST PARAMETERS.** Trainable count is no longer the figure of merit or the tie-break. Rank by what the OBJECT is — layers composed, operations per layer, weights shared, heads to read — and use `rho_50 ≤ 0.740` as a **gate** ("does it still learn DD"), not as a score.

- **Unchanged by the new objective:** free line 0.740, `s*` 0.1484 / 0.2918, the 14-point grid (floor 0.33, 400/bin), plateau 1.64, LOO bar 0.0423, 3 seeds with the spread quoted, every seed must cross.
- **Non-negotiables, every stage:** `--shuffle-adj`; 3 seeds, spread quoted; **validation token accuracy FIRST** (never censors); register the scorer before the first cell lands, array `--dependency=afterok` on it; **read every deck page back as a PNG**; one knob at a time, each run in its own dated directory; decks carry example traces and the tokenization.

## Current best result

- **Standing gate-passing cells (N=48, `shuffle_adj`, 3 seeds, pooled eval).** Attention-only `no_mlp L=16 d192` (J2), 2,504,890, **0.580**, 3/3 — the standing minimum, **unchanged by stage S**. Hub `full L=6 d256`, 4,899,898, **0.592** — 6 layers, the shallowest free cell, the most tractable object known to do DD. `full L=12 d256`, 9,638,458, **0.385**, best solver at any size.
- **Attention alone does direct deduction at N=48 — the RULE and the ITERATION** (findings 58/59, decked). Round-0 counterfactuals 1.000/0.997/1.000 on J2's seeds; chaining against self-derived facts 0.994 3/3. Limits: round-0 cases are easy by construction, the round profile is tested to 3 of 16 layers, and **the guard third of 27/28 — declining to deduce what does not follow — has no module in the tree.**
- **62 (11-11). STAGE S: ONE SHARED BLOCK DOES DD, 0.011 LESS ACCURATELY — the plateau was a CAPACITY FLOOR, not an obstruction from sharing.** S1 `tied_no_mlp T=16 d320`, 613,818 trainable, **`rho_50` 0.757** (0.754/0.757/0.765), **3/3 crossing**, mean_solve 0.3148, val acc 0.4830, **plateau escapes 3/3** (1.5851/1.5915/1.5864) against P1 d192's 0/3. So d192 censored because one block at that width cannot serve 16 roles — **not** because tying breaks deduction. The cell lands **0.017 past** the free line 0.740, so it does not clear the gate.
- **S4 (job `51904215`) is the headline over any level: tying moves the cell ALONG the accuracy-to-solve curve, it does not bend the relation.** All three d320 seeds sit on the N=48 depth-ladder curve — residuals **+0.0165 / −0.0034 / +0.0065** against the 0.0423 LOO bar, inside a fit where 26/26 well-determined held-out rungs sit on the curve and the worst miss among 30 discriminating cells is 0.0222. They are DISCRIMINATING, so the fit could have rejected them. **The whole 0.177 `rho_50` gap against J2 is what J2's extra 0.011 of token accuracy buys at this steepness.**
- **Accuracy has saturated in width:** d192 0.4567 → d320 0.4830 → d448 0.4837 (one seed), still 0.011 under J2's 0.4940. Epoch confound excluded: J2, P1 and S1 are all 240k × 50 epochs.
- **61 (11-10). STAGE P** returns no architecture result on its own: P1 d192 `rho_50` CENSORED, plateau 0/3. Superseded as an interpretation by stage S above; the CENSOR itself stands.
- **Never pool N=48 and N=96 `rho_50`** (finding 55). Read every gap against its OWN control.

## Running

- **STAGE T, array `51904636`** (3 cells, 1 GPU each, `--time 04:00:00`, registered by `scripts/staget_score.py`, committed by job `51904633`). **No training**: the one shared d320 block applied `T ∈ {4,8,12,16,20,24}` on FIXED weights, 3 seeds, 18 grids, same 14-point grid and same `--seed-start`, so every T is paired on identical instances. **Readings fixed in the scorer:** T1 (easy end still solves, read first, never censors) → level table → **T0 (iterable operator: |T=24 − T=16| ≤ 0.01 and no T ≥ 16 worse than best by > 0.01, against the competing "depth 16 is a trained schedule" hypothesis)** → T2 (smallest T within 0.01 of T=16) → T3 (paired low-rho gain at T=20/24).
- **STAGE S write-up `51905177`**, `--dependency=afterany:51853969`: commits `notebooks/make_stages_deck.py` + `notebooks/build_stages_nb.py`, renders the deck with that sha, then builds and executes `notebooks/sudoku_n48_stages_2026-11-11_results.ipynb`. **Artifacts are a second commit and the PNGs are unread — nothing is uploaded yet.**
- **STAGE S complete score `51904146`**, `--dependency=afterany:51853969`. The d320 numbers above come from the PARTIAL read; this is the quotable one, and the d448 arm lands with it.
- **Stage S array `51853969`**: the three d448 cells were still running at 04:30 (8h56m elapsed). Two of them have been the uniform predictor since inside epoch 0.
- **Test suite `51905194`** after the `rung_key`, `rho_transition_eval` and `run_tests` edits.
- **Other sessions are on sganguli** (`mirage_*`, `astra_*`, `p36_*`, `pullback_*` — other projects; `ListAgents` shows no peer on spin_gnn). Don't touch them.

## Open problems

- **The d448 arm cannot carry a reading.** Two of three seeds diverged inside epoch 0 at warmup 0; diagnostic `51860479` traced that to **warmup, not to the cell** (1.6228 NLL / 0.4804 acc at warmup 5, against 3.9563 / 0.0313 at warmup 0, still falling through the full-lr epochs). **The S5 warmup array (`scripts/slurm/sudoku2_n48_tiedwidth_S5warm_2026-11-11.sbatch`) is written and DELIBERATELY NOT SUBMITTED:** its motivation was the H1 censoring-floor confound for a *censored* `rho_50`, and d320 crossed 3/3 and produced a number, so the confound is moot. Its other motive — more width — is now contradicted by the saturation above.
- **The 0.011 accuracy gap is the only thing between the tied cell and the free line, and width will not close it.** Open question: whether it is an optimization budget (50 epochs, constant 1e-3, tail slope ≈ −0.00013/epoch) or a real ceiling imposed by one block serving 16 roles. The interpolation — `k` distinct blocks cycled `T/k` times, `k ∈ {1,2,4}` — is the experiment that would answer it and needs a new `arch_mode` plus tests. **Stage T's outcome decides whether that is the right next cut**, so it is not written yet.
- **Test suite: 27 files, 237 cases, 0 failures.** `test_exact_recovery` and `test_transformer_shapes` import `spin_gnn/model.py` → `torch_geometric`, which `ml spider` does not find on Sherlock. They are now reported as **SKIPPED by name** via an explicit `ABSENT` list in `scripts/run_tests.py` (with a stale-entry check), so the suite's exit code carries signal again instead of being permanently 1. An install is hshunt's call.
- **Deck provenance nit:** the O+P footer is stamped with the parent commit because builders and artifacts went in one commit. `51905177` uses stage N's order. Also still open: `notebooks/deck_tokenization.py` page 4's "… 1 further rounds" grammar.
- **Reading the PNGs is not a formality** — the O+P deck needed eleven fixes across three renders, none with an exit code. Lint first (`/scratch/users/hshunt/lint_claims.py`).
- **Every checkpoint is the final epoch** (no best-val selection; worth +0.0280 solve_rate, finding 14). **The whole L\* ladder is leaked-order**: `shuffle_adj` is in `INSTANCE_KEYS`.
- **MODULES ONLY** (`$SCRATCH/spin_gnn_venv` is purged): `module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312`, then `sys.path.insert(0, "/home/users/hshunt/spin_gnn")`. **Never pipe `module load`**, never set `PYTHONPATH=` inline.
- **$HOME I/O on sh02-ln02 wedges intermittently.** **Every git step runs in a job; never delete `.git/index.lock`, never kill the process.**

## Next steps

1. **Read stage T in its registered order: T1 first, then the level table, then T0.** T0 is the claim worth having — if the block is an iterable operator, the architecture is a fixed-point iteration and the tractability case is made independently of the level.
2. **Finish the stage S write-up:** read every PNG back, lint the titles, commit the artifacts as a second commit, then `~/.local/bin/claude-results-upload.sh spin-gnn <pdf> "stage S tied-width sweep" "rho_50 0.757 3/3, on-curve, vs 0.580 untied"`.
3. **Then the `k` distinct blocks ladder**, if stage T supports it — scorer registered first, array `--dependency=afterok`, one knob (`k`), 3 seeds.
4. **The guard third of findings 27/28** is still the gap worth closing on the Q/R route; localization (`--stages patching,heads`) is now safe on tied checkpoints and needs a GPU.
