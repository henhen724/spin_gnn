# STATUS — spin_gnn, branch `sudoku-dd` (2026-11-13 07:10; determinism pinned and shipped; NOTHING RUNNING)

**`FINDINGS.md` archives findings 1–54 in full** (grep it, never read it whole). **Findings 55–67 live in `LOG.md`** — search it, never read it in full — and in the decks. This file is the handoff and is deliberately short.

## The objective, restated 2026-11-10 — read this before any rung table

**THE GOAL IS A CONCEPTUALLY SIMPLE MODEL THAT MIGHT BE THEORETICALLY ANALYZABLE, EXPLICITLY NOT A MODEL WITH THE FEWEST PARAMETERS.** Trainable count is not the figure of merit or the tie-break. Rank by what the OBJECT is — layers composed, operations per layer, weights shared, heads to read — and use `rho_50 ≤ 0.740` as a **gate** ("does it still learn DD"), not a score.

- **Unchanged:** free line 0.740, `s*` 0.1484 / 0.2918, the 14-point grid (floor 0.33, 400/bin), plateau 1.64, LOO bar 0.0423, 3 seeds with the spread quoted, every seed must cross.
- **Non-negotiables, every stage:** `--shuffle-adj`; 3 seeds, spread quoted; **validation token accuracy FIRST** (never censors); register the scorer before the first cell lands, array `--dependency=afterok` on it; **read every deck page back as a PNG, twice**; one knob at a time, each run in its own dated directory; decks carry example traces and the tokenization.
- **Delivery is a password-gated WEB DECK, PDF ask-only.** Decks NEST: `decks/<project>/<deck>/deck.md` → `/<project>/<deck>/`, two segments max, **one password per project**. Publishing bare `spin-gnn` is **refused** while nested decks exist. `~/slides` is shared with other sessions — the publisher stages only `decks/<slug>` and `lib/passwords.js`, so never `git add -A` there.

## Current best result

- **Standing gate-passing cells (N=48, `shuffle_adj`, 3 seeds).** Attention-only `no_mlp L=16 d192` (J2), 2,504,890, **`rho_50` 0.580**, 3/3 — **the standing minimum, unchanged by S, T, U, V, W**. Hub `full L=6 d256`, 4,899,898, 0.592. `full L=12 d256`, 9,638,458, 0.385, best solver at any size.
- **62. STAGE S: ONE SHARED BLOCK DOES DD, 0.011 LESS ACCURATELY.** S1 `tied_no_mlp T=16 d320`, 613,818 trainable, **0.757** (0.754/0.757/0.765), 3/3, mean_solve 0.3148, val acc 0.4830 (spread 0.0010). **0.017 past the free line, gate NOT cleared.** S4: all seeds ON the accuracy-to-solve curve, so **0.017 of `rho_50` ≈ 0.0011 of token accuracy** (exactly 0.0010565 — `stagev_score` DERIVES it; 0.0011 is the rounding, never retype either). This is the control every later stage is read against.
- **63–65. T, U, V:** the shared block is not an iterable operator (`T*=16`); role count is not the binding constraint, the basin is; a longer *trained* schedule does not help and moved the seed spread 0.0010 → 0.0222.
- **66. STAGE W: doubling the budget does not close the gap.** W-a falsified 1/3 (spread 0.0263), W-b misses (median 0.757 → 0.766), residuals inside the 0.0423 LOO bar — nothing architectural happened.
- **67 (THIS MORNING). A MATCHED SEED NOW REPRODUCES A RUN; UNPINNED IT NEVER DID.** Job `52181659`, 4 cells of S1 at seed 0, 3 epochs. **H1 confirmed:** pinned pair bit-identical in train NLL, val NLL and accuracy at every epoch (max |Δ| 0.00e+00). **H2 confirmed:** unpinned pair at the same seed differs **0.1982** train NLL at epoch 0 → 0.0165 → 3.57e-04. **H3: 2.5 min per cell, ≈0.8 min/epoch, 7.7%.** Six runs of this one configuration now sit at five distinct epoch-0 values spanning **0.2294**; the only coincident pair is the pinned one. **Old floor σ_d 0.01412 → 1403 replicates per arm (11,081 GPU-h)**, 71 (562 h) dropping seed 2 — **2×–22× the effect, which is why S–W all read "within noise"**. Pinning removes run-to-run noise, **not** the seed-to-seed basin lottery.
- **Never pool N=48 and N=96 `rho_50`** (finding 55). Read every gap against its OWN control.

## Running

- **NOTHING OF MINE IS RUNNING OR QUEUED.** `52181659` (determinism check) and `52190645` (deck) are COMPLETED. `52183401` and `52189773` FAILED **by design** — see Open problems; both refusals were correct and neither is a bug to chase.
- `critic_kacf` (`52182580`) on sganguli belongs to another session — don't touch it. `~/slides` has an uncommitted `decks/arpes-detector-sim/deck.md` that is theirs — leave it unstaged.
- **Decks:** `https://slides-hshunt.pages.dev/spin-gnn/budget/` (stage W, 28 pages, builder `74709c4`) and **`/spin-gnn/determinism/`** (33 pages, 18 explanation, builder `29b451c`, rendered by `52190645`, seed 0 of job `52181659`). Password `sapling-yarrow-69` opens **all 5** `spin-gnn` decks.

## Open problems

- **THE 0.011 ACCURACY GAP IS STILL OPEN AND EVERY REGISTERED *KNOB* IS CLOSED:** width, cosine decay, more data, best-val selection, role count (U), schedule length (V), training budget (W). **The next move is a different question, not another knob — that is a decision for hshunt, not an arm to invent.** W-f (k=2 × 100 epochs) and stage V tasks 3–5 (T=32) stay **registered and deliberately unsubmitted**.
- **What finding 67 changes, and what it does not.** It does not close the gap; it makes the instrument able to see one. Pinned, a control and an arm sharing a seed with constant lr compute *identical* arithmetic up to the shorter one's last epoch, so a cross-run comparison becomes exact and a number that moves is a bug rather than weather. It says nothing about why one seed in three finds the basin, which is seed-to-seed.
- **`deterministic` is OFF by default and should stay off**, so the code as it stands still reproduces S–W exactly. A new stage opts in per cell (`--deterministic`, plus `CUBLAS_WORKSPACE_CONFIG=:4096:8` **in the launcher** — the trainer refuses to start without it) and pays 7.7% wall clock. Cells get a `_det` directory segment; the setting is recorded in `config.json` and `sweep.csv`.
- **Two guards earned their keep this morning and both now exist in the tree.** `check_claims` compares `noise_floor`'s effect size against the scorer's to twelve digits — it refused job `52183401` because the two disagreed by 4%, the published rounding having been typed into one of them. `deck_lint.py` now counts **bullets per `points_page`** (cap 3) as well as words per literal; not counting them cost job `52189773`.
- **A term a FIGURE introduces is a term the deck must define.** The shared tokenization figure asserts "THE ORDER WITHIN A ROW IS THE LEAK" in bold, so a deck that emits it must emit `deck_explain.LEAK`. Only `CENSOR` is safe to drop.
- **Guard 7 does not protect stage W:** at 613,818 its rungs sit BELOW H1's 879,418 censoring floor. What protected it was the *identical* count at 50 and 100 epochs.
- Cosmetic, pre-existing, shared builder: the PDF mixes two `/MediaBox` sizes. `slides.py` is shared with every project — not touched.
- The S5 warmup array is written and **deliberately not submitted**. The **guard third of findings 27/28** still has no module in the tree.
- **MODULES ONLY** (`$SCRATCH/spin_gnn_venv` is purged): `module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312`, then `sys.path.insert(0, …)`. **Never pipe `module load`**, never set `PYTHONPATH=` inline. `viz` + `py-matplotlib` are not optional. `scripts/noise_floor.py` is **no longer stdlib-only** — it imports NEED, so it needs the modules.
- **$HOME I/O on the login nodes wedges intermittently.** Never delete `.git/index.lock`, never kill the process.

## Next steps

1. **Put the choice of the next question to hshunt before building anything.** Every knob is closed and finding 67 does not reopen one. The two candidates, both honest about cost: (a) **chase the basin on purpose** — with pinning, "why one seed in three" is now askable with an exact control; (b) **change the measurement** — the 0.011 is read on a 3-seed median that *seed-to-seed* noise still moves by more than the gap, and pinning does not touch that.
2. **Nothing may be submitted that re-tests a closed knob**, finding 67 notwithstanding: an exact instrument does not make a dead knob live.
3. **If a new stage is authorized:** register the scorer first and `--dependency=afterok` the array on it, import bars from `stagew_score` rather than retyping them, pass `--deterministic` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and still prefer **within-run** comparisons where the quantity allows.
4. **The deck toolchain is current and reusable as-is.** `notebooks/deck_explain.py` (15 shared explanation pages; drop only `CENSOR`), `notebooks/make_determinism_web.py` (headline branches on the measured verdict, including the branch where the result would have been "pinning fixes nothing"), `scripts/noise_floor.py`, `scripts/deck_lint.py`, `scripts/step_epoch_census.py`.
5. **Read every page back as a PNG, twice — it keeps paying.** S–W: twenty-four defects, nine wrong *claims*, on decks that rendered clean. This morning: five more, including a title its own figure refuted and a bolded term the deck never defined.
