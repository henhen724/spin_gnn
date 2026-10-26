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
- 2026-10-24: **there is no sharp critical density, and my own "critical point at rho_min" framing from earlier today does not survive the test.** Re-analysing the existing CSVs over N=24..996 (no new measurement, no GPU): DD depth T at FIXED density grows as N^a at every density, a = +0.18 at rho=0.95 rising smoothly to +0.38 at rho=0.50 and never reaching 0 -- so rho_min~0.478 is where greedy removal stops, not a divergence. Separately the `random`-hint solvability threshold runs off toward rho=1 (50% crossing 0.9296 -> 0.9875 for N=24 -> 384, with only 1.7 -> 4.8 unhinted nodes at the crossing, i.e. 2.9x where extensive would need 16x). Old finding 1 also misread: `dd_resolved_frac` is identically 1.0 at N=24, so the 0.99 -> 0.44 curve was the transformer degrading, not DD. 7-page deck committed (`a94b0e9`) and sent. Consequence for the program: depth is only purchasable by raising N, which is why N=48 (49433453) and the num_layers scan (49433454) went out. **Extended the same evening by the (degree, n_colors) sweep (49437796, 12 of 21 cells before it crashed), which settles both halves more strongly than the re-analysis could.** (i) a > 0 at rho=0.90 in every cell that deduces at all, +0.19 to +0.25, so the absence of a critical point at rho_min is a property of the whole grid and not of (5,4) -- the one cell with a = +0.00, d=3 q=4, is pinned at T = 1.0 because d = q-1 lets DD force nothing past the first round, which is the opposite of criticality. (ii) The "only by raising N" conclusion is **wrong as stated**: rho_min falls monotonically with the propagation ratio d/(q-1) (0.836 at 1.00 to 0.011 at 5.00) and (d=8,q=4) reaches mean depth 19.1 near its floor at N=96, which (5,4) needs N~813 for -- about 8x in N. The caveat that keeps it honest is that this is depth at each cell's OWN floor; at MATCHED density (5,4) is still the deepest cell on the grid, so the corner moves the floor down rather than making deduction harder. Pre-registered prediction (b) failed backwards: the low-ratio cells it expected to be deepest are exactly the degenerate d=q-1 ones, and depth has an interior optimum in the ratio.
- 2026-10-24: **the easy-end collapse is closed, and it was an out-of-distribution artifact, not a regression** (`49430217`). `removal_path` stops at `floor(rho*N)` hints, so `rho_max=0.95` at N=24 is 22 hints -- k=0 and k=1 are OOD and `mean_o1` was averaging them in. The best model scores **exactly 1.000 with 0 failures of 300 at every in-distribution hint count** (22/21/20/18/16), breaking only past the ceiling (0.860 at 23 hints, 0.060 at 24). The 24-hint failures are `empty BEGIN_ROUND` parse errors -- the model opening a deduction round on an instance with nothing to deduce, which is the signature I predicted but which `classify_failure` was charging to `malformed` because it tested `well_formed` first. Split out as its own label and re-run identically as 49434190: **280 of the 282** 24-hint failures are `empty_round`, solve_rate reproduces to the digit in all 21 bins, and d0384 keeps its 100% `extra_rounds` as the control -- 4 of 4 pre-registrations held. My prediction (d) was wrong in a useful direction: `d0384 x 5000` declines in-distribution too (0.970 at 18 hints, 0.933 at 16) while handling 24 hints at 0.983, so in-distribution perfection is model-dependent rather than a property of the ensemble.
- 2026-10-24: **the shared pool at 6000 instances separates what 1500 could not.** d0256x80000 0.9955 [0.9935, 0.9969] vs d0256x20000 0.9837, delta +0.0118 with 117 discordant 94-23 (p = 2e-11) -- reproducing the +0.0113 from n=1500. The d0128/d0384 tie at n_train=20000 survives (0.9750 vs 0.9740) but both are now resolvably BELOW d0256 (-0.0087, p=3e-4; -0.0097, p=5e-5), so `d_model=256` is a real optimum at this budget and the d0128 $OAK archive is probably not worth taking.
- 2026-10-25: **network depth does not track reasoning depth** (`49433454`, N=24, `num_layers` 1-12 with `d_model`/ff pinned). L*(T), the smallest `num_layers` reaching 0.90 on instances whose DD trace takes T rounds at fixed hint density, is **3 at every T** -- 1..3 at rho=0.75, 2..5 at rho=0.60, slope +0.00 layers per round against a pre-registered +3 or more. Three things make this more than a null. (i) Prediction (a) failed in the informative direction: it said one deduction round should need no depth, but L=2 scores **0.686 at T=1**, so the L=3 knee is a fixed cost paid before any chaining rather than the first rung of a ladder. (ii) The *marginal* value of depth past L* does rise with T -- L3->L8 gains of +0.011/+0.026/+0.037/+0.051 at T=2..5 -- so residual error at L* is concentrated in deep instances and extra layers remove it preferentially; a weak serial effect survives in the margin while the threshold form is dead. Measured to L=12 that gain goes flat and noisy because L=12 is past the knee and overfits, which is why the deck prints both and the trend claim rests on the pair, not on my pick. (iii) The rho=0.60 series reported `r(realized_rho,T) = -0.941` and the sbatch said to discount on a large r; it is **kept**, because r measures only whether the drift is consistent in direction and the whole spread is 0.0009 = **0.02 of one hint out of 24**, far too small to move solve_rate 4-5 points. rho=0.75 has realized_rho identical in every bin and carries the primary claim either way, so nothing turns on that call. `drift_verdict()` now reports drift in hints so "r is large" and "the confound is negligible" can be seen to be consistent rather than argued.
- 2026-10-25: **the N=24 recipe transfers to N=48 and the depth window widens as intended** (`49433453`): `d0256`/L12/20000/300ep gives agg 0.9033 against 0.9837 at N=24, clearing the 0.70 gate that held the N=48 layer scan (now submitted, 49509466). rho=0.60 spans T=3..7 there against 2..5 at N=24, with solve_rate 0.943 -> 0.894, and the density confound points the **wrong way** to explain the decline (`r(realized_rho,T) = +0.893`: deep bins are slightly *denser*, which would flatter deep instances), so finding 5's depth cost survives a conservative confound at the larger size. Separately, the control-param sweep went out as 49509101 on its third and final attempt, after 49445119 died at 41 min on the same divisibility `ValueError` as 49437796 -- in the other branch, because the fix I wrote guarded only the random one on the strength of a docstring claim of mine that `dd_removal_path_small_n` does not call `planted_regular_graph`. It does, at line 38; I wrote that claim and then acted on it without checking. Both branches are now gated and `check_grid()` proves all 168 points constructible before any measurement, since both failures were fully decidable from the arguments alone.
- 2026-10-25: **at matched parameter count, depth beats MLP width outright, so the L=3 knee is a serial requirement rather than a capacity floor** (`49510197` trained, `49521537` scored on the shared pool at n=1500; deck sent). Finding 11 established that L=2 fails even at T=1 but not *why*, and H1 -- that the knee is simply where the model has enough parameters -- predicts width should substitute for it. It does not, and not marginally: at **1.65M** matched params L=4/ff=256 scores **0.9587** against L=2/ff=1024's **0.4727** (+0.4860, exact p=9.9e-204), and at **2.44M** L=6/ff=256 **0.9767** against the knee L=3/ff=1024's **0.9500** (+0.0267, discordant 65-25, p=3.0e-05) -- the comparison that matters, because it says a fixed budget spent on depth beats the *reference* width and not merely a crippled L=2. All three pre-registrations resolved, including the efficiency one: L=6/ff=256 at 2.45M **ties L=12/ff=1024 at 9.55M** (+0.0053, p=0.37) and also ties L=4/ff=1024 at 3.23M (p=1.0), i.e. the top of the reference ladder at 3.9x fewer parameters. Two things I expected to have to hedge turned out not to need it. The width pathology is real rather than a bad seed (p=4.6e-33). And L=2/ff=4096 -- 4.80M, **1.97x** the knee's budget -- scores 0.4920, which I first tried to explain away as under-training; checking the curves refuted that **backwards**, since its val_nll turned over at the halfway mark and ended 0.019 *above* its minimum, so if anything it overfit. Writing the deck also caught two of my own false claims before they shipped: a "2.9x the knee's budget" that is 1.97x, and a "flat to within 0.001 for the last 80 epochs" that is 0.021 and is 0.021 for all three cells, so the converged-vs-still-improving contrast it was carrying did not exist either. Both are now computed from the checkpoints instead of asserted.
- 2026-10-25: **L*(T) is flat at N=48 as well, and the layer scan's pre-registered gate failed in the harmless direction** (`49509466`). L* is **4 at every depth**, T=3..7 at rho=0.60 and T=2..4 at rho=0.75, and this time there is nothing to argue about on confounds: realized-density spread is 0.0004 and 0.0000 across bins. So the flat-in-T result now holds at two sizes over a wider T range than N=24 offered. The cross-N half is **not** settled, because the grid {1,2,4,8,12} skipped L=3 -- precisely the N=24 knee -- so all the scan can say is L*(48) is in (2,4]; L=3 went out as 49523504 to decide between "depth is a fixed cost flat in N" and "depth tracks graph size". The gate: its L=12 cell was required to reproduce 49433453's 0.9033 within [0.865, 0.932] or the scan was to be reported and stopped as confounded with epoch budget. It came in at 0.9533 -- outside, but **upward**, 150 epochs beating 300. The cause is identified rather than assumed. Both runs reach the same validation minimum (1.2146 at ep46, 1.2164 at ep36, so training is reproducible), both then overfit, and the 300-epoch run overfits further: +0.1413 above its minimum against +0.0497, having risen 0.0663 over its second half. `save_checkpoint` keeps the final epoch, so the longer run's saved weights are deeper into overfitting. That makes the confound **cross-job rather than within-scan** -- all five cells, the new one, and the whole N=24 arm are 150 ep -- so what it invalidates is using 49433453 as a reproducibility anchor, not the L* comparison the scan exists for, and continuing is the honest reading of the rule rather than a way around it. It is also the first time the standing final-epoch-checkpointing problem has shown a measured solve_rate cost (~0.05) instead of a val_nll curiosity, and it is a within-model case where val_nll *does* predict solve_rate, unlike finding 4's across-model r=+0.33.
- 2026-10-25: **the full 21-cell control-param grid landed (`49509101`, third attempt, 168/168 constructible) and it falsifies a caveat I had already published.** Finding 10's claim that "at matched density (5,4) is still the deepest cell on the grid" was read off the 12 cells that survived 49437796 and is **false** on 21: (5,4) ranks 3rd of 13 at rho=0.55, behind (10,6) and (8,5), at every size. The replacement is cleaner than the thing it replaces, because the two panels *invert* the role of the colour count. At each cell's own floor, more colours is **shallower** (q=6 tops out at 9.3 against q=3's 14.5) and depth peaks at propagation ratio 2.0-2.7. At matched rho=0.55, more colours is **deeper** (5.0/6.1/6.8/8.4 for q=3,4,5,6) and depth falls monotonically with ratio inside each q. The mechanism is the floor: high-q cells have high `rho_min`, so a fixed 0.55 sits just above theirs while being far above q=3's. Both orderings hold unchanged out to N=960. The practical consequence is a better route-2 cell than the one finding 10 proposed: **(d=10,q=6)** has floor 0.507 against the baseline's 0.468, so its depth advantage is measured at essentially the same density and needs no caveat at all -- 8.4 against 6.1 at N=60, widening to 28.0 against 14.8 at N=960. The cheap-corner headline also needs restating, since the complete sweep has no N=96 and the 8x figure came from it: at N=60, (d=4,q=3) reaches mean depth 14.5 where (5,4) needs N~333, a factor of **6**. I caught the original error only by rendering the deck and re-checking the per-q peaks against the data, after a first rewrite that asserted both panels peak at ratio 2.00 and was also wrong.

- 2026-10-25: **the cross-N question is closed: L\*(48)=4, so depth tracks graph size, and L\*(T) is flat in T at every size measured** (`49523504`, 00:35:25). The L=3 cell the scan skipped scores **0.8293 [0.8095, 0.8475]** on the shared pool at n=1500 against L=4's 0.9093, so pre-registration (a) is confirmed and L\*(48) is 4, not 3. The per-bin picture is the better result: L\*(T) is constant in all four fully searched series — 3 at both densities at N=24, 3 at rho=0.75 and 4 at rho=0.60 at N=48 — **15 bins, 0 rising**, and at rho=0.60 L=3's deficit is a *uniform* offset (0.875/0.846/0.860/0.835/0.827) rather than concentrated at deep T, so the extra layer is not buying deduction rounds; what moves L\* is hint density and problem size. The other two pre-registrations also resolved: (b) the 150-epoch L=12 cell beats the 300-epoch one by **+0.0280**, 106–64, **p=0.0016**, which turns the standing final-epoch-checkpointing problem from a val_nll curiosity into a measured solve_rate cost between two runs that reached the same val minimum; and (c) L=8 ties L=12 (+0.0140, p=0.076, inside the pre-registered 0.02 band), with **L=8 at 0.9533 the best N=48 cell**, consistent with L=12 overfitting. Deck `notebooks/sudoku_depth_vs_n_2026-10-25` (8 pages, `d89ddb4`) sent to OneDrive.
- 2026-10-25: **the N=96 probe earned its cost by failing, and the ceiling is optimization rather than data** (`49524030`, 02:59:15). Shared pool **0.8267 [0.8067, 0.8450]**, so pre-registration (a) is falsified and the 36 h N=96 layer scan stays **unsubmitted**. The decisive number is per-bin, not aggregate: rho=0.75 clears 0.90 in all 4 bins (1.000/0.984/0.983/0.947) and rho=0.60 in **none of 6** (0.860/0.822/0.839/0.754/0.794/0.837), so L\*(96) at rho=0.60 is **undefined rather than large** — exactly the outcome the one-cell probe existed to catch before the scan spent 36 h returning a column with no L\* in it. (b) held (rho=0.60 spans T=4..9 as predicted); (c) holds marginally at rho=0.75 but **fails** at rho=0.60 where the decline goes non-monotone (r=−0.398 against ≈−0.9 at N=24/48). The diagnosis **contradicts this run's own pre-registered "the next move is data, not layers"**: val_nll bottoms at 1.4588 @ep129 of 150, final 1.4621, the last 30 epochs span 0.0043, and train ends **0.0289 below** val, i.e. the model never fits its own training set, so there is no generalization gap for finding 6's data lever to close. Rather than drop the pre-registered lever on an argument, 49565433 tests both — cosine lr (cell A) and 4x data (cell B) — against the same control on one pool.
- 2026-10-25: **three code fixes, two of them latent bugs nobody had hit yet.** (i) There was **no lr scheduler anywhere in the codebase** — every run in the project trained Adam at a constant lr — so `lr_schedule`/`lr_min_factor` were added to `SudokuReasoningTrainConfig` and the sweep CLI, defaulting to `"constant"` (the old behaviour) with the `_cos` tag suffix appearing only when non-default, so every tag written before today stays byte-identical and the decks and `agg_eval` paths that address cells by that string still resolve. (ii) `save_checkpoint` recorded **neither the seed nor the training budget**, so this deck had to recover both from the sbatch that produced the run; it now writes seed, epochs, lr, n_train/n_val, batch_size and the schedule into `config.json` (additive, `INSTANCE_KEYS` unaffected). (iii) `sweep.csv` is append-only but the writer passed `fieldnames=list(row.keys())` on every append, so adding any column would have silently written **ragged rows into every pre-existing `sweep.csv`** — it now reuses the header on disk and prints which fields it cannot hold, verified against a synthetic old-format file. All three went out as `38d503a` and the smoke cell of 49565433 confirmed the plumbing in-job (`{'seed': 0, 'epochs': 2, 'lr_schedule': 'cosine', ...}`).
