# STATUS — spin_gnn, branch `sudoku-dd` (2026-11-12 23:20; S, T, U, V, W all shipped; NOTHING RUNNING)

**`FINDINGS.md` archives findings 1–54 in full** (grep it, never read it whole). **Findings 55–66 live in `LOG.md`** — search it, never read it in full — and in the decks. This file is the handoff and is deliberately short.

## The objective, restated 2026-11-10 — read this before any rung table

**THE GOAL IS A CONCEPTUALLY SIMPLE MODEL THAT MIGHT BE THEORETICALLY ANALYZABLE, EXPLICITLY NOT A MODEL WITH THE FEWEST PARAMETERS.** Trainable count is not the figure of merit or the tie-break. Rank by what the OBJECT is — layers composed, operations per layer, weights shared, heads to read — and use `rho_50 ≤ 0.740` as a **gate** ("does it still learn DD"), not a score.

- **Unchanged:** free line 0.740, `s*` 0.1484 / 0.2918, the 14-point grid (floor 0.33, 400/bin), plateau 1.64, LOO bar 0.0423, 3 seeds with the spread quoted, every seed must cross.
- **Non-negotiables, every stage:** `--shuffle-adj`; 3 seeds, spread quoted; **validation token accuracy FIRST** (never censors); register the scorer before the first cell lands, array `--dependency=afterok` on it; **read every deck page back as a PNG, twice**; one knob at a time, each run in its own dated directory; decks carry example traces and the tokenization.
- **Delivery is a password-gated WEB DECK, PDF ask-only.** Decks NEST: `decks/<project>/<deck>/deck.md` → `/<project>/<deck>/`, two segments max, **one password per project**. Publishing bare `spin-gnn` is **refused** while nested decks exist. `~/slides` is shared with other sessions — the publisher stages only `decks/<slug>` and `lib/passwords.js`, so never `git add -A` there.

## Current best result

- **Standing gate-passing cells (N=48, `shuffle_adj`, 3 seeds).** Attention-only `no_mlp L=16 d192` (J2), 2,504,890, **`rho_50` 0.580**, 3/3 — **the standing minimum, unchanged by S, T, U, V and W**. Hub `full L=6 d256`, 4,899,898, 0.592. `full L=12 d256`, 9,638,458, 0.385, best solver at any size.
- **62. STAGE S: ONE SHARED BLOCK DOES DD, 0.011 LESS ACCURATELY.** S1 `tied_no_mlp T=16 d320`, 613,818 trainable, **0.757** (0.754/0.757/0.765), 3/3, mean_solve 0.3148, val acc 0.4830 (spread 0.0010). **0.017 past the free line, gate NOT cleared.** S4: all seeds ON the accuracy-to-solve curve, so **0.017 of `rho_50` ≈ 0.0011 of token accuracy**. This is the control every later stage is read against.
- **63. STAGE T: the shared block is not an iterable operator** — `T* = 16`, frozen T0/T1 both falsified.
- **64. STAGE U: role count is not the binding constraint; which basin the optimizer finds is.** k=2 / k=4 medians 0.752 / 0.755, all within 0.01 of k=1, but one seed in each k>1 arm left the plateau (0.603, 0.579) and U4 puts both ON the curve. One seed does NOT displace J2's three-seed 0.580.
- **65. STAGE V: a longer *trained* schedule does not help and destabilizes training.** T=24, same 613,818: V-a 0/3 at the 0.4841 bar, two seeds unchanged and one never trained, spread 0.0010 → 0.0222, median 0.757 → 0.769, 2/3 crossing.
- **66 (TONIGHT). STAGE W: DOUBLING THE BUDGET DOES NOT CLOSE THE GAP, AND THE TWO ARMS ARE NOT THE SAME RUN.** 100 epochs, same cell, array `52053221`, scored by `52062829`. **W-a FALSIFIED 1/3** — acc 0.4856 / 0.4814 / 0.4594, **spread 0.0263, 24× the 0.0011 needed**. **W-b misses:** median `rho_50` 0.757 → **0.766**, 2/3 crossing, mean_solve 0.3148 → 0.3082. W-c **holds** (within-run falls 0.0244/0.0059/0.0089, median ≥ 0.0065); W-d in seed 0 only (NLL 1.5494, acc 0.4856, yet `rho_50` 0.760); W-e falsified. Residuals −0.0158/+0.0005/−0.0030 vs the 0.0423 LOO bar — **nothing architectural happened**.
- **66b, and the part that outlives the stage: A MATCHED SEED DOES NOT REPRODUCE A RUN.** The 100-epoch cells and their "controls" are **independent runs of one configuration** — configs identical but for the epoch count, data identical (stored baseline agrees to 16 digits), nothing pinning GPU kernel nondeterminism. At the control's last epoch they already differ by **−0.0113 / +0.0078 / +0.0602** train NLL and **+0.0018 / −0.0027 / −0.0246** accuracy. Seed 2's control stepped at epoch 22 and its 100-epoch twin never did, which is most of why W-a reads 1/3. **Every cross-run comparison in this project carries that noise; W-c was registered within-run and is the only clean one.**
- **Never pool N=48 and N=96 `rho_50`** (finding 55). Read every gap against its OWN control.

## Running

- **NOTHING OF MINE IS RUNNING OR QUEUED.** Stage W array `52053221`, scorer `52062829`, and render jobs `52149320`, `52149494`, `52150186`, `52150992` are all COMPLETED. `critic_ksh` (`52148638`) on sganguli belongs to another session — don't touch it.
- `ListAgents` 23:05: peers `arpes-curious-crown`, `agent-management`; **neither on spin_gnn**. `~/slides` has an uncommitted `decks/arpes-detector-sim/deck.md` that is theirs — leave it unstaged.
- **Stage W is published:** `https://slides-hshunt.pages.dev/spin-gnn/budget/`, 28 pages / 14 explanation / 7 figures, builder commit `74709c4`, seeds 0–2 of array `52053221`, password `sapling-yarrow-69` (one password opens all 4 `spin-gnn` decks).

## Open problems

- **THE 0.011 ACCURACY GAP IS STILL OPEN AND EVERY REGISTERED ROUTE TO IT IS NOW CLOSED:** width (saturated), cosine decay (47), more data (`n_train_instances` is not in `rung_key`, so a bigger cell destroys its own control), best-val selection (final `model.pt` only), role count (U), schedule length (V), **training budget (W)**. **The next move is a different question, not another knob — that is a decision for hshunt, not an arm to invent.** W-f (k=2 at 100 epochs) stays **registered and deliberately unsubmitted**: it is the budget knob again, now known not to pay.
- **The strongest open lead is the recipe's variance, not the architecture.** One knob moved the seed spread 0.0010 → 0.0222 (V), and runs of an *identical* config differ by up to 0.06 train NLL and 0.025 accuracy at the same epoch (W). Three of the four best `rho_50` seeds the tied family has produced are basin seeds. **Whether "find the basin reliably" is the question worth asking next is the thing to put to hshunt.**
- **Determinism is unpinned** in `train_sudoku2.py` — no `torch.use_deterministic_algorithms`, no cuDNN flag, no seeded generator for the kernels. Pinning it would make matched-seed pairs reproducible and would cost speed; it has never been measured here.
- **Guard 7 does not protect stage W:** at 613,818 its rungs sit BELOW H1's 879,418 censoring floor. What protected it was the *identical* count at 50 and 100 epochs.
- Cosmetic, pre-existing, shared builder: the PDF mixes two `/MediaBox` sizes. `slides.py` is shared with every project — not touched.
- The S5 warmup array is written and **deliberately not submitted**. The **guard third of findings 27/28** still has no module in the tree.
- **MODULES ONLY** (`$SCRATCH/spin_gnn_venv` is purged): `module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312`, then `sys.path.insert(0, …)`. **Never pipe `module load`**, never set `PYTHONPATH=` inline. `viz` + `py-matplotlib` are not optional.
- **$HOME I/O on the login nodes wedges intermittently.** Every git step that matters runs in a job; never delete `.git/index.lock`, never kill the process.

## Next steps

1. **Put the choice of the next question to hshunt before building anything.** Stage W closed the last registered route; the brief's own instruction for that case is to say so rather than invent an arm. The two candidates to offer, both honest about cost: (a) **make the recipe reliable** — pin determinism, then ask whether a basin can be reached on purpose rather than 1 seed in 3; (b) **change the measurement** — the 0.011 gap is read on a 3-seed median that run noise alone moves by more than the gap.
2. **Nothing may be submitted that re-tests a closed knob.** W-f (k=2 × 100 epochs) and stage V tasks 3–5 (T=32) are both registered and both stay unsubmitted.
3. **If a new stage is authorized:** register the scorer first and `--dependency=afterok` the array on it, import the bars from `stagew_score` rather than retyping them, and make the comparison **within run** wherever the quantity allows — 66b is the reason.
4. **The deck toolchain is current and reusable as-is.** `notebooks/deck_explain.py` (14 stage-independent explanation pages, `check()` asserts every number against the scorer), `notebooks/stagew_figs.py` (`fig_training_curves`, `clearing_dy`, `describe`), `notebooks/make_stagew_web.py` (`check_claims()` asserts every published number, `MID_EP`/`END_EP` own every epoch label, `paired()` reads both arms at the control's last epoch). `scripts/step_epoch_census.py` carries the registered `step_epoch()` and the reporting-only `onset()`.
5. **Read every page back as a PNG, twice — it keeps paying.** Across S–W the read-backs caught twenty-four defects, **nine of them wrong *claims*** on decks that rendered clean. Tonight's: a title the figure refuted, "no step anywhere" false in both arms, one NLL value labelled epoch 50 on one page and epoch 49 on another, grey that stopped meaning control, a label printed on the floor line.
