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
