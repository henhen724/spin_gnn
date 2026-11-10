# STATUS — spin_gnn, branch `sudoku-dd` (2026-11-09, after stage F)

**Findings 1–54 in full, and every line of detail trimmed out of this file, are in `FINDINGS.md`** —
archived verbatim from this file on 11-09, nothing removed. It is 107 KB: **grep it, never read it
whole.** Session history is `LOG.md`: search it, never read it in full. This file is the handoff.

## Current best result

**THE FIGURE OF MERIT IS THE STANDING MINIMUM, NOT A SOLVE RATE** (brief of 11-04), and **the
breaking point is the result**, not the smallest thing that still works.

- **N=48 STANDING MINIMUM: attention-only (no MLP anywhere) `L=16 d192`, 2,504,890 trainable,
  `rho_50` 0.580, 3/3 seeds crossing** (finding 53, rung J2). Hub, which `s*` is anchored to:
  `full L=6 d256`, 0.592 at 4,899,898. Best solver at any size: `full L=12 d256`, 0.385 at
  9,638,458. Yardstick `s*` = 0.1484 (`rho_50`) / 0.2918 (mean_solve); "free" iff `rho_50` ≤ 0.740;
  N=48 grid floor 0.33. **Replacement rule H3 = I3 = J3 = K2: a rung takes the minimum only if 3 of
  3 seeds cross AND the median is ≤ 0.740; two of three is never enough (finding 51).**
- **Stage K tried to go below it at L=16 and failed** (finding 54, `51668331`, scored `51699439`):
  KA `L=16 d160` 0.757 at 1,759,738 (3/3 cross, 3/3 escapes), KB `L=16 d128` 0.766 at 1,145,658
  (3/3 cross, 2/3 escapes) — both past the free line. **Before/after on the pooled `s*` eval:
  0.580 at 2,504,890 → 0.580 at 2,504,890, UNCHANGED.** K0 falsified high (the depth step worth
  −0.174 at d192 is worth −0.016 at d160, so depth and width are not separable below d192 and the
  ~2M floor is the TASK's, not L=12's); K1 falsified in the interesting direction (d128 censors at
  L=12, crosses 3/3 at L=16); K4 held 24/24, worst residual 0.0222 vs the 0.0423 LOO bar.
  **Any brief quoting 0.605 at 3,331,642 predates finding 53 — check a stated baseline's date.**
- **55 (NEW, 11-09). THE N=48 MINIMUM DOES NOT TRANSFER TO N=96 — the cut is load-bearing there**
  (stage F, `51656823` 4/4 COMPLETED 0:0 in 11:44–13:04; scored + deck + notebook in `51708442` by
  the registered `scripts/stagef_score.py` `6c66e3a`). Stripped `no_mlp L=16 d192` `rho_50`
  **0.850** (seeds 0.937/0.826/0.850, 3/3 crossing, 2,618,602 trainable / 8,929,514 total) vs
  paired control `full L=12 d256` **0.676** (0.679/0.564/0.676, 9,790,058 total): **+0.175 =
  1.47 `s*_96`** against the registered bar 0.119. **F0 FALSIFIED, DIRECTION HIGHER. F1 held**
  (3/3 measured, 0 censored), **F2 held** (third control seed 0.676 inside the registered bracket
  [0.564, 0.679]; 3-seed median 0.676 vs the 2-seed 0.621, +0.46 `s*_96`), **F3 is a registration,
  not a measurement** — no N=96 number moves the N=48 minimum. `s*_96` = 0.119 frozen at
  registration; re-estimated from 4 multi-seed rungs it is 0.117 (reported, never substituted).
  **CONSEQUENCE: every "smallest architecture" claim in this programme is an N=48 claim and is
  relabelled as one, the ~2M floor included.** Deck `notebooks/sudoku_n96_stagef_2026-11-09_slides.pdf`
  (7 pp, every page read back as a PNG, on OneDrive) + `..._results.ipynb` (17 cells).
- **Never pool N=48 and N=96 `rho_50`**: different grids (floors 0.33 and 0.32), different
  yardsticks (`s*` 0.1484 vs `s*_96` 0.119). Read each gap against its OWN control, never as a ratio.
- Older bests, kept for provenance: leak-free N=24 **1.0000** at 240k×100ep (finding 23, the working
  budget), leak-free N=48 **0.9780** (finding 38); on the FALL ensemble, 0.755 at rho=0.333/T=13.79
  (finding 44) — different ensembles, never differenced. **Is it learning DD? Yes — rule, chaining
  AND guard** (findings 27/28: 350/350 round-0 counterfactuals, 0/350 when the forcing is broken
  against a 0.940 placebo ceiling), with the within-round ORDER the one thing not copied.

## Running

**NOTHING IS RUNNING** — `squeue --me` empty at 19:14. Closed today: stage K `51668331` (6/6,
scored `51699439`, write-up `51702509`), stage F `51656823` (4/4, scored + written up `51708442`),
git/record jobs `51704309`, `51708483`, `51709564`. Commits `88b3150`, `f22c8b4`, `fb2029c` pushed.

## Open problems

- **`next_node` at `pre_node` is counting, not deducing — MEASURED in all three arms**
  (`scripts/order_baseline_next_node.py`): events come out in ascending node index for 100.0% of
  3,746 rounds, and a hand rule that deduces nothing (`min_unknown`) scores 0.632 against the
  SOLVER's 0.816, so most of that score is counting. "It knows which node comes next" stays retired.
  Findings 19/21 (`justifies_next` at `pre_color`) are untouched. Detail in `FINDINGS.md`.
- **Every checkpoint is the final epoch** — no best-val selection, no early stopping; measured worth
  +0.0280 solve_rate (finding 14), needs no code change (`checkpoint_epochs=()` already accepted).
  **The whole L\* ladder is still leaked-order**: `shuffle_adj` is in `INSTANCE_KEYS`, so re-measuring
  means retraining at 240k×100ep.
- **`$SCRATCH/spin_gnn_venv` IS DEAD (purged) — CPU analysis runs on MODULES ONLY**:
  `module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
  py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312`,
  then `sys.path.insert(0, "/home/users/hshunt/spin_gnn")`. **Never pipe `module load`** and never set
  `PYTHONPATH=` inline. Missing: `torch_geometric`, `nbformat`, `tqdm`, `pytest` (shimmed, 32/32 pass
  in a job). Rule 8 forbids a `$SCRATCH` rebuild, `$GROUP_HOME` is 99% full → `$HOME` is the only
  legal site and it is hshunt's call. 27 committed `model.pt` files (267 MB) still bloat history.
- **$HOME I/O on sh02-ln02 wedges intermittently** — four processes on 11-09 parked at 00:00:00 of
  CPU in `cl_sync_io_wait`, including a `git commit` that held a 0-byte `.git/index.lock` for 33
  minutes and then completed on its own, exit 0. **Move the step into a job; never delete the lock
  and never kill the process.** Reported to hshunt with a support draft; not worked around.

## Next steps

1. **NON-NEGOTIABLES, EVERY STAGE:** `--shuffle-adj` on every cell (findings 16/17/23 — the smaller
   the model, the more tempting the decoding shortcut); **3 seeds per rung with the spread quoted**
   (41/42/47/51); **score on validation token accuracy FIRST** (finding 4+6 — it never censors and
   predicts mean_solve to 0.009; `rho_50` only confirms, and **a rung that falls OFF
   `scripts/acc_vs_solve.py`'s curve is the interesting one**); register the scorer BEFORE the first
   cell of a stage lands; **read every deck page back as a PNG before shipping**; compare at a
   MATCHED TRAINABLE BUDGET, never against zero; every run into its own dated directory.
2. **THE LIVE QUESTION IS NOW WHICH CUT IS LOAD-BEARING AT N=96 — the MLP or the width.** Finding 55
   says the pair differs by 1.47 `s*_96`; it cannot say which half of the N=48 cut did it. That is a
   **two-cell experiment against this same control** (`full L=12 d256`, 3 seeds already on disk): add
   the MLP back at d192 (`full L=16 d192`), and strip the MLP at the control's width
   (`no_mlp L=12 d256`). Register the scorer first; `s*_96` for a NEW stage should be frozen from the
   re-estimate 0.117, not from 0.119, which belongs to F.
3. **J1's CONVEXITY IS NOW NOT WORTH BUYING.** It was queued behind F on exactly one condition — "worth
   3 cells only if F says the N=48 minimum transfers" — and F says it does not. The shape of the
   `rho_50` curve on [192, 256] at L=12 is an N=48 question about a floor that has just been
   relabelled as N=48-only. Reopen it only if item 2 finds the N=96 cut is the WIDTH.
4. **THE ONE STAGE-K LOOSE THREAD: re-run KB (`no_mlp L=16 d128`) on 3 fresh seeds.** d128 censoring
   at L=12 and crossing 3/3 at L=16 is the single surprise of that stage, it came with only 2/3
   plateau escapes, and K1's own registration says fresh seeds before it is quoted outside the
   scorer. 3 cells. Nothing else this N=48 grid can resolve: what is left is width steps smaller
   than the seed spread.
