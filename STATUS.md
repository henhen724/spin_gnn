# STATUS — spin_gnn, branch `sudoku-dd` (2026-11-11 23:00; S, T, U shipped, V running)

**`FINDINGS.md` archives findings 1–54 in full** (grep it, never read it whole; it has lagged since stage K). **Findings 55–64 live in `LOG.md`** — search it, never read it in full — and in the decks. This file is the handoff and is deliberately short.

## The objective, restated 2026-11-10 — read this before any rung table

**THE GOAL IS A CONCEPTUALLY SIMPLE MODEL THAT MIGHT BE THEORETICALLY ANALYZABLE, EXPLICITLY NOT A MODEL WITH THE FEWEST PARAMETERS.** Trainable count is no longer the figure of merit or the tie-break. Rank by what the OBJECT is — layers composed, operations per layer, weights shared, heads to read — and use `rho_50 ≤ 0.740` as a **gate** ("does it still learn DD"), not as a score.

- **Unchanged by the new objective:** free line 0.740, `s*` 0.1484 / 0.2918, the 14-point grid (floor 0.33, 400/bin), plateau 1.64, LOO bar 0.0423, 3 seeds with the spread quoted, every seed must cross.
- **Non-negotiables, every stage:** `--shuffle-adj`; 3 seeds, spread quoted; **validation token accuracy FIRST** (never censors); register the scorer before the first cell lands, array `--dependency=afterok` on it; **read every deck page back as a PNG, twice**; one knob at a time, each run in its own dated directory; decks carry example traces and the tokenization.
- **Delivery is a password-gated WEB DECK, PDF ask-only** (`~/CLAUDE.md`): source in `~/slides/decks/<project>/`, published with `claude-slides-publish.sh`. **Stage U is live** at `https://slides-hshunt.pages.dev/spin-gnn/` (18 pages, commit `084409b`, published `b579821`); `bash ~/slides/tools/check-gate.sh spin-gnn` prints **gate holds** (figures 401 without the cookie, the test that matters). `~/slides` is shared with other sessions — the publisher stages only `decks/spin-gnn`, so never `git add -A` there.

## Current best result

- **Standing gate-passing cells (N=48, `shuffle_adj`, 3 seeds, pooled eval).** Attention-only `no_mlp L=16 d192` (J2), 2,504,890, **0.580**, 3/3 — the standing minimum, **unchanged by stages S, T and U**. Hub `full L=6 d256`, 4,899,898, **0.592**. `full L=12 d256`, 9,638,458, **0.385**, best solver at any size.
- **62 (11-11). STAGE S: ONE SHARED BLOCK DOES DD, 0.011 LESS ACCURATELY.** S1 `tied_no_mlp T=16 d320`, 613,818 trainable, **`rho_50` 0.757** (0.754/0.757/0.765), 3/3 crossing, mean_solve 0.3148, val acc 0.4830. **0.017 past the free line, gate NOT cleared.** S4: all three seeds ON the N=48 accuracy-to-solve curve, so **0.017 of `rho_50` ≈ 0.0011 of token accuracy**.
- **63 (11-11). STAGE T: THE SHARED BLOCK IS NOT AN ITERABLE OPERATOR** — one weight set tuned to a 16-step schedule. Frozen weights at `T ∈ {4..24}`: T0 and T1 both FALSIFIED, **T\* = 16** (all sixteen applications used), seed 0 falls 0.887 → 0.000 easy-bin from T=16 to T=24.
- **64 (11-11). STAGE U: ROLE COUNT IS NOT THE BINDING CONSTRAINT; WHICH BASIN THE OPTIMIZER FINDS IS.** k distinct blocks cycled 16/k times, array `51922710`, scored `52000549`, U4 by `52025112`. **U0 FALSIFIED**: medians **0.752 (k=2) / 0.755 (k=4) vs 0.757 (k=1)**, all 3/3 crossing, all within 0.01 — the competing hypothesis' prediction. U1 flat (train NLL span 0.0067 ≤ the 0.01 band → the floor is the recipe, not the roles). U2 monotone 0.4830 → 0.4831 → 0.4837 but inside the worst seed spread 0.0114, so accuracy cannot adjudicate, as registered. **What the bound-aware median hid is the result:** one seed in EACH k>1 arm left the plateau — k=2 s2 **0.603**, k=4 s1 **0.579** (acc 0.4929 / 0.4948, train NLL 1.5371 / 1.5306) — while no k=1 seed did in three tight seeds (0.754/0.757/0.765). **U4: both sit ON the curve** (+0.0017 and −0.0025 against the 0.0423 LOO bar, both weakly determined), so nothing architectural happened; they reached J2-level accuracy and collected what the curve pays. **One seed at 0.579 does NOT displace J2's three-seed 0.580 and no improvement is claimed.** Both basin seeds were still descending 4–10× faster than the plateau seeds at epoch 50.
- **Never pool N=48 and N=96 `rho_50`** (finding 55). Read every gap against its OWN control.

## Running

- **STAGE V, array `51962715` tasks 0–2 (T=24), `-p sganguli`, 1 GPU each.** Started ~14:23, **8:29 elapsed at 22:52, 21:30 of wall left, 0 restarts** — healthy, nothing to debug. Registered `ca11c2b` by `51962711`. One knob: the block is TRAINED at a longer schedule; **613,818 trainable at every T**, so it costs no parameters. Score with `python3 -u -m scripts.stagev_score`, **V-a first** (val acc ≥ **0.4841** in ≥2/3 seeds = control + the 0.0011).
- **T=32 arm (tasks 3–5) deliberately NOT submitted** until T=24 is read. V-d needs both rungs but not concurrently.
- **Nothing else of mine is running.** The other sganguli jobs (`52025503`, `52027606`, `52001211`, `52026430`) belong to other sessions — don't touch.

## Open problems

- **The 0.011 accuracy gap is still the only thing between the tied cell and the free line.** Closed on evidence: **width** (saturated), **cosine decay** (finding 47 — both cells got worse), **more data** (`n_train_instances` is NOT in `rung_key`, so a 480k cell pools into stage S's rung and destroys its own control), **best-val selection** (final `model.pt` only), and now **role count** (stage U). Live: **schedule length** (stage V) and **budget** (stage W, below).
- **Stage W is now the strongly motivated arm:** both stage U basin seeds were cut off mid-descent at 50 epochs. 100 epochs at constant 1e-3, 3 seeds, differenced against stage U's own k=2 (or k=4) rung at 50; `rung_key` keys it distinctly as `(16,4,'tied_no_mlp',320,1024,0,100,'k2')`. **Register the scorer before the first cell lands, array `--dependency=afterok` on it.** Two seeds out of six is a coincidence, not a rate — stage W is what turns it into one.
- **Guard 7 does not protect stage V:** at 613,818 its rungs sit BELOW H1's 879,418 censoring floor. What protects it is the *identical* count at every T.
- Cosmetic, pre-existing, in the shared builder: the PDF mixes two `/MediaBox` sizes. Not touched — `slides.py` is shared with every project.
- The S5 warmup array is written and **deliberately not submitted** (the d448 arm still cannot carry a reading). The **guard third of findings 27/28** still has no module in the tree.
- **MODULES ONLY** (`$SCRATCH/spin_gnn_venv` is purged): `module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312`, then `sys.path.insert(0, …)`. **Never pipe `module load`**, never set `PYTHONPATH=` inline. `viz` + `py-matplotlib` are not optional (`51905177` died 19 s in without them).
- **$HOME I/O on sh02-ln02 wedges intermittently.** **Every git step runs in a job; never delete `.git/index.lock`, never kill the process.**

## Next steps

1. **When `51962715` lands (ETA 2026-11-12 ~02:00–07:30): `python3 -u -m scripts.stagev_score`**, `sacct -j 51962715 -X` first. Read **V-a first** — if it misses in 2 of 3 seeds the schedule-length route is dead on the statistic that cannot censor, and no reading of `rho_50` rescues it. Then V-b gate, V-c easy end, V-d trend, **V-e trained-at-24 vs stage T's frozen-at-24** (0.819, 2/3, worst easy bin 0.0000).
2. **Submit tasks 3–5 (T=32) only after that read.**
3. **Then stage W** (the 100-epoch budget arm above), registered scorer first. If V is also null, W is the last live route to the 0.011.
4. **Stage V's deck replaces stage U's** in the same gated project (re-publishing keeps the password `sapling-yarrow-69`; stage U's pages stay in the slides repo's history). Builder template is `notebooks/make_stageu_web.py` — it re-derives nothing, calling the scorers' own loaders and asserting every published number in `check_claims()`. Read every page back as a PNG, twice: across S, T and U the read-backs caught thirteen defects, five of them wrong *claims* on decks that rendered clean.
