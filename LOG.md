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
- 2026-10-21: **agg_solve_rate 0.847 → 0.973** (`48854233`, n_train 5000→20000, ~12σ; same *protocol*, but NOT the same instances — val seeds are `seed + n_train_instances`, so each n_train draws different ones and data4x even trains over the 5000-run val seeds, which is why `48912134` rescores everything on one shared pool) — the largest single gain in 34 runs, from the one hyperparameter never varied. `48862411` (d384) ties agg at 0.847 but takes k0 0.24→0.99 and o1 0.842→0.994, so agg-vs-o1 is not a trade for width. `48869017` submitted to test whether the two are additive.
- 2026-10-21: `48868696` **refutes the chaining-colors hypothesis I put on the deck hours earlier**. Teacher-forced coloring is at ceiling and flat (0.977–0.997, ≤0.01/round) and selection's marginal rise with round (+0.074…+0.115) collapses to ≈+0.009 once position (≡ candidate count) is fixed. Depth cost is not token-level; selection accuracy also only scores agreement with DD's sorted order, so 0.61 coexists with solve_rate 0.86. Finding 5 unexplained.
- 2026-10-21: runtime estimate for `48868696` was off by ~140x (51s against a 2h request, under the 10-min aggregation guideline) — no autoregressive decoding means these eval jobs are far cheaper than the solve_rate ones they were sized against.
- 2026-10-21: **the 300-instance per-run eval had the top two models backwards.** `48912134` scored 4 checkpoints on ONE pool of 1500 held-out instances (seed 5e10) and paired them with McNemar exact: d256x20000 **0.9847** beats d384x20000 0.9720 (discordant 53, 36-17, p=0.013), the reverse of their own-val 0.973/0.986; the two "tied" 5000-instance models separate 0.876 vs 0.820. Both data4x models still crush both 5000 models (p~1e-29, 1e-55). The shared pool is now the metric of record, and 3 of my 5 pre-registered predictions for this and `48913017` were wrong — all three by assuming differences were unresolvable.
- 2026-10-21: **probe ladder `48913017` revises finding 3 and finds what competence actually lives in.** frontier_mask peak AUROC is 0.907-0.938 across 5 checkpoints, not the 0.863 I called flat, and it rises with competence (within width: d256 0.907->0.916, d384 0.916->0.938, seed spread +-0.001) -- but 0.863 came from a narrower d0128 series, so width is confounded with data and a d0128 x 20000 run is needed to separate them. `next_node` peak accuracy (0.789/0.810/0.957/0.964) reproduces the shared-pool ranking **exactly**, reversal included, while frontier_mask does not; both next-step probes step up with n_train and peak at the last layer, where the frontier peaks mid-stack.
- 2026-10-22: **new best, and the data axis has not saturated.** `48912548` (d0256 x n_train=80000, 300 ep, 5:33) scores **0.9960 [0.9913, 0.9982]** on the shared 1500-instance pool vs 0.9847 for d0256 x 20000 -- paired delta +0.0113, 27 discordant 22-5, exact p = 0.0015 (`48999377`). Archived to $OAK. The own-val 0.9933 that prompted the re-score was directionally right but uncheckable at n=300; the shared pool is what settled it.
- 2026-10-22: **finding 3 rewritten: the frontier probe measures width, not competence.** Over the full 7-cell width x data grid, `frontier_mask` peak AUROC has Spearman +0.25 with shared-pool solve_rate and +0.85 with `d_model`; the new best model's frontier AUROC FELL (0.9162 -> 0.9067) while its solve rate rose 0.985 -> 0.996, and d128 x 20000 reaches only 0.8831 while solving 0.973 against d256 x 5000's 0.9071 at 0.820. The old 0.863 -> 0.907 gap was width all along. Finding 8 strengthened instead: `next_node` peak accuracy is Spearman +0.96 across all 7 cells, one inversion between two statistically tied models.
- 2026-10-22: `d0128 x 20000` (4.0M params, 0.9733) is **statistically tied with `d0384 x 20000`** (16.7M, 0.9720) -- at this scale data buys more than width. Also a real negative: `48912548`'s easy end went backwards, `solve_rate_k0` 0.60 -> **0.07** and `mean_o1` 0.918 -> **0.776**, the same anti-correlation width produced on 10-18, now reproduced by data scale. 4 of 7 pre-registrations held; both live misses are this collapse.
- 2026-10-24: **there is no sharp critical density, and my own "critical point at rho_min" framing from earlier today does not survive the test.** Re-analysing the existing CSVs over N=24..996 (no new measurement, no GPU): DD depth T at FIXED density grows as N^a at every density, a = +0.18 at rho=0.95 rising smoothly to +0.38 at rho=0.50 and never reaching 0 -- so rho_min~0.478 is where greedy removal stops, not a divergence. Separately the `random`-hint solvability threshold runs off toward rho=1 (50% crossing 0.9296 -> 0.9875 for N=24 -> 384, with only 1.7 -> 4.8 unhinted nodes at the crossing, i.e. 2.9x where extensive would need 16x). Old finding 1 also misread: `dd_resolved_frac` is identically 1.0 at N=24, so the 0.99 -> 0.44 curve was the transformer degrading, not DD. 7-page deck committed (`a94b0e9`) and sent. Consequence for the program: depth is only purchasable by raising N, which is why N=48 (49433453) and the num_layers scan (49433454) went out.
- 2026-10-24: **the easy-end collapse is closed, and it was an out-of-distribution artifact, not a regression** (`49430217`). `removal_path` stops at `floor(rho*N)` hints, so `rho_max=0.95` at N=24 is 22 hints -- k=0 and k=1 are OOD and `mean_o1` was averaging them in. The best model scores **exactly 1.000 with 0 failures of 300 at every in-distribution hint count** (22/21/20/18/16), breaking only past the ceiling (0.860 at 23 hints, 0.060 at 24). The 24-hint failures are `empty BEGIN_ROUND` parse errors -- the model opening a deduction round on an instance with nothing to deduce, which is the signature I predicted but which `classify_failure` was charging to `malformed` because it tested `well_formed` first (now split out as its own label). My prediction (d) was wrong in a useful direction: `d0384 x 5000` declines in-distribution too (0.970 at 18 hints, 0.933 at 16) while handling 24 hints at 0.983, so in-distribution perfection is model-dependent rather than a property of the ensemble.
- 2026-10-24: **the shared pool at 6000 instances separates what 1500 could not.** d0256x80000 0.9955 [0.9935, 0.9969] vs d0256x20000 0.9837, delta +0.0118 with 117 discordant 94-23 (p = 2e-11) -- reproducing the +0.0113 from n=1500. The d0128/d0384 tie at n_train=20000 survives (0.9750 vs 0.9740) but both are now resolvably BELOW d0256 (-0.0087, p=3e-4; -0.0097, p=5e-5), so `d_model=256` is a real optimum at this budget and the d0128 $OAK archive is probably not worth taking.
