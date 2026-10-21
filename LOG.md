# LOG

Dated history for the DD Sudoku project (`sudoku-dd` branch). Do not read
this file in full from the start of a session -- search it for whatever
you need. At most 3 lines appended per session.

- 2026-10-15: Picked up from a previous (killed) agent session that had
  committed the model/code/handoff doc (`431aee3`) but never run anything
  on Sherlock. Fixed a broken `pip install numba` (used Sherlock's
  `py-numba/0.60.0_py312` module instead; 88 tests pass) and ran the
  first real greedy hint-removal sweep (Slurm `47921119`, 840 runs).
- 2026-10-15: Analysis of that sweep recovers the claimed easy-to-hard
  transition for `d=7` (gamma(rho) jumps right at rho_c=0.27) but not yet
  for `d=5` (greedy stuck above its rho_c=0.46) -- see
  `notebooks/results_transition.ipynb`.
- 2026-10-15: Separate side task: built an event-based DD-Sudoku
  reasoning-trace tokenizer (`sudoku_trace.py`/`sudoku_tokens.py`) mirroring
  the master branch's k-SAT v2 lessons, reused `SATReasoningTransformer`
  as-is. First attempt sampled hints independently at random (wrong --
  not the slides' protocol); fixed `build_instances` to use
  `sudoku_dd.removal_path`'s greedy hint-removal instead (every instance
  guaranteed resolvable). Corrected run (`48032796`) gets
  outcome_accuracy=0.995 but `solve_rate=0.005` -- see STATUS.md. Added
  checkpoint snapshotting + a pinned-rho `eval_solve_by_rho` and ran a
  checkpoint-vs-difficulty sweep (`48144619`/`48145554`,
  `notebooks/sudoku_transition.ipynb`): confirms the model learns the
  easy/shallow end first (epoch 50: `0.16` at `rho=0.95`, `0.00` at
  `rho<=0.8`) rather than failing uniformly, though far short of ~100%.
  Extended to `rho` up to `1.0` (`48146991`/`48150261`): `solve_rate`
  plateaus at `~0.3` even on the fully trivial all-hinted case, while
  `outcome_accuracy`/`well_formed_rate`=1.0 -- a capacity/precision
  ceiling, not a difficulty-positioning issue.
- 2026-10-19: 900-epoch retrain of `d_model in {64,128}` at `nhead=4,L=12,
  ff=1024` (Slurm `48513926`, epochs the only variable; enrichment `48522086/
  48522087`). **H1 (ceiling is training-budget-limited): refuted** --
  `agg_solve_rate` 0.720->0.720 (d64) / 0.773->0.767 (d128) and enrichment
  1.255->1.246 (d64) / 1.249->1.129 (d128), while `val_nll` worsens
  1.20->1.37 and 1.67->2.16: the extra 600 epochs only overfit. **H2
  (d128's easy-case regression was undertraining): confirmed** --
  `mean_o1_solve_rate` 0.896->0.968, `solve_rate_k0` 0.55->0.94, so the
  d_model sweep's "wider is worse on easy cases" reading is retracted as a
  budget confound. New best balance: `d_model=128 @ 900ep`,
  `agg_solve_rate=0.767` (+0.047 over previous best) at
  `mean_o1_solve_rate=0.968`. Three levers (head count, head width, budget)
  are now all negative on content-targeting. See
  `notebooks/sudoku_long_train_2026-10-19_results.ipynb`.
- 2026-10-18: Tested the opposite direction from the nhead sweep: keep
  `nhead=4` fixed and scale `d_model` up (64->128->256, so `head_dim`
  16->32->64) to give each head more room without fragmenting into more
  heads (Slurm `48446043` train + `48449673/48449674` attention analysis).
  **Also doesn't help.** Mean attention-targeting enrichment is flat-to-down
  (`1.255@d64 -> 1.249@d128 -> 1.144@d256`), and the per-layer shape gets
  *worse distributed*: at `d64` enrichment peaks smoothly across middle
  layers (`~1.66-1.72` at layers 4-5); at `d128/d256` most layers drop to
  ~baseline or below (e.g. layer 2: `0.79@d128`, `0.46@d256` -- worse than
  attending by distance alone) while one or two late layers spike sharply
  (`2.89@d128` layer 9, `2.44/2.13@d256` layers 10-11). `mean_o1_solve_rate`
  (near-trivial end) also degrades with width (`0.980 -> 0.896 -> 0.842`),
  and `solve_rate_k0` (zero uncolored nodes -- pure copy, no reasoning
  needed) collapses (`~0.98 -> 0.55 -> 0.24`) even as `agg_solve_rate` (full
  training rho range) improves (`0.72 -> 0.77 -> 0.85`) and `final_val_nll`
  gets much worse (`1.2 -> 1.67 -> 2.41`). Net: bigger `d_model` buys some
  full-range solve rate but at the cost of precision on the easiest cases
  and no gain in content-addressing -- combined with the nhead-sweep result,
  this says `head_dim=16` (the original default) is close to optimal for
  this task's id-matching, and scaling either axis (more heads or more
  per-head width) alone doesn't raise the content-addressing ceiling found
  in the evolution analysis. See `notebooks/sudoku_dmodel_sweep.png`.
- 2026-10-17: Tested "more heads for id-matching" directly: swept `nhead in
  {2,4,8,16,32}` at fixed `d_model=64, num_layers=12, dim_feedforward=1024,
  300 epochs` (Slurm `48296020` train + `48297775/48297914/48297915/
  48297776` attention analysis). **Opposite of the hypothesis**: `nhead=4`
  (head_dim=16, the original default) is the peak -- `agg_solve_rate`
  degrades monotonically on both sides as head count moves away from it
  (`0.71@h2 -> 0.72@h4 -> 0.51@h8 -> 0.33@h16 -> 0.10@h32`), and attention-
  targeting enrichment tracks the same shape (peaks `~1.26` at `h=4`, falls
  to `~1.08` at `h=32`, barely above the untargeted baseline). More heads
  means smaller `head_dim` (`64/nhead`), which apparently isn't enough room
  to represent a clean id-matching subspace -- the lever is per-head
  dimensionality, not head count. See `notebooks/sudoku_nhead_sweep.png`.
- 2026-10-17: Attention-targeting evolution over training
  (`spin_gnn.sudoku_attention_evolution`, Slurm `48287024`, retrained
  L=12/ff=1024 with checkpoint snapshots at epochs
  0/5/10/25/50/75/100/150/200/250/300): targeting enrichment *does* train --
  rises from ~1.0 (untrained/random) to ~1.35-1.38 by epoch 50-75 alongside
  val_accuracy -- but then plateaus/slightly regresses to ~1.22-1.26 for the
  rest of training (epoch 100-300), while val_accuracy keeps slowly rising
  (0.56->0.57). So the attention head learns *some* targeting early, then
  stalls well before the model stops improving -- later accuracy gains come
  from something other than sharper attention targeting. See
  `notebooks/sudoku_attention_evolution.png`.
- 2026-10-17: Attention-distance analysis (`spin_gnn.sudoku_attention_distance[_analysis]`,
  Slurm `48267874`) on the best checkpoint (`L=12,ff=1024`): DD's required
  reference distance is huge (median 179/210 max_len -- BEGIN_GRAPH sits
  near the sequence start, every later round needs it) but measured
  attention does *not* decay sharply with distance -- it stays roughly flat
  out past distance 150. The real gap is targeting, not reach: attention at
  the exact needed token is only ~1.0-1.7x a distance-matched random
  baseline (peak at middle layers 4-7), i.e. barely better than attending
  by distance alone. See `notebooks/sudoku_attention_distance_comparison.png`.
- 2026-10-16: Model-size hyperparameter scan
  (`spin_gnn.sudoku_hparam_sweep`) over `num_layers x dim_feedforward` at
  `d_model=64, nhead=4`, targeting `mean_o1_solve_rate` -- `solve_rate`
  averaged over a fixed `k=0..4` uncolored-node grid (`rho=(24-k)/24`),
  the near-trivial end held at a constant small `k` rather than scaled by
  `n_nodes`. Round 1 (`48157903`, `L in {2,4,6,8} x ff in
  {128,256,512,1024}`, 150 epochs): best `L=8,ff=1024` reaches
  `mean_o1_solve_rate=0.904` (up from baseline's ~0.5) -- both depth and
  width matter, neither alone enough -- but still undertrained at epoch
  150 (`val_accuracy` still rising). Round 2 (`48162371`, `L in
  {8,10,12} x ff in {1024,2048}`, 300 epochs) reached the goal:
  `L=12,ff=1024` hits `mean_o1_solve_rate=0.980` (`agg_solve_rate=0.72`,
  vs. the original `0.005` baseline); `L=8,ff=1024` close behind at
  `0.968` with fewer params. `ff=2048` underperformed `ff=1024` at every
  `L` (likely undertrained at 300 epochs for the larger size).

- 2026-10-19: Located the hint-density transition. Random-hint ensemble has none at any N
  (48579545); removal-path does, already at N=24 — solve_rate 0.99→0.44, truncated_rate 0.000
  everywhere, rho_min≈0.477 N-independent (48581589, 48581590). rho and T collinear, so
  depth-vs-density is unresolved. Deck+notebook delivered (commit 4d5c3bb).
- 2026-10-19: Four tests agree the model is NOT running DD: probes known 0.91 vs frontier 0.67
  (48580216), loses the disagreement vote to ascending_after 0.334/0.361 (48582589), flip rate
  0.267 vs 0.25 chance (48583444), no frontier SAE family (48583439). Patching localizes a real
  sub-threshold color route: edited_hint L0-5 → recolored neighbour's row L6-9 → predict L8-11,
  never through the predicted node's own row.
- 2026-10-19: Fixed two measurement bugs that had produced publishable-looking nonsense — SAE
  l1_coeff didn't transport across d_model (dense dictionary at FVE 0.992), and head ablation
  hooked out_proj, which nn.MultiheadAttention never calls, giving effect_retained==1.0 on all
  7200 rows. Both now have regression tests.

## 2026-10-21

- `48582169`: DD never appears — `known_mask` 0.925 by ep5 (solve_rate 0.007), `frontier_mask` 0.654→0.679 over 895 more epochs, `next_color` the only target tracking solve_rate (0.327→0.655, r=0.963). Notebook + deck `7c50b99`, uploaded.
- `48825742` (labels only): `frontier_mask` 0.679 is uninterpretable, not weak evidence — `1−known` scores 0.734 above it, procedure-matched known-bits probe 0.588 below it, `shuffled` 0.5000. Needs AUC rescore; against my pre-registered reading.
- Series side-note: solve_rate saturates at ep50, `val_nll` rises monotonically after, so 850 of 900 epochs are overfitting (within a run — see the r=+0.17 line below before reading val_nll across runs).
- Best result was never 0.767: `sweep.csv` already held **0.847** (d0256 @300ep) on the same eval, unranked since the headline metric changed. `scripts/sweep_leaderboard.py` (`90d02f5`+) now prints both metrics so it cannot recur. d0256 is also the sweep's worst model at rho=1.0 (k0 0.24 vs 1.00).
- Over all 31 runs, `r(agg_solve_rate, val_nll) = +0.17` and `r(agg, n_params) = +0.713` — validation loss is ~uninformative about the metric of record, which undercuts my own case for the data4x run. Submitted `48854232` (ep900), `48854233` (n_train 4x, the first run ever to vary it), `48862411` (d384), `48861799` (probe rescore on AUROC/AP, seeds 0–2).
