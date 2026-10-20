# STATUS

Project: DD Sudoku (direct-deduction coloring) on random regular graphs --
reproducing the reasoning-depth phase transition, plus a side task
pretraining a transformer on DD reasoning traces. Branch `sudoku-dd`. See
`SUDOKU_DD.md` for model/code. Separate from `master` (k-SAT) and AMP work.

## Current best result
**Phase transition (main task):** `d=7` recovers the claimed transition
(`gamma(rho)` jumps at `rho_c=0.27`); `d=5` does not yet (greedy floor
`rho~0.478`). See LOG for the sweep (`47921119`).

**Where the hint-density transition actually is (2026-10-19, new):** NOT in
independent random hints -- that ensemble is subcritical at every density
(`48579017`: recovery beyond hints peaks at +0.12 and is identical N=24..384;
cascade depth only 1.64->3.36 over 16x N). This **retires the "scale N to
~1700 tokens" plan** on the delivered slide deck's last page. It is in the
removal-path ensemble the models already train on, as a divergence of DD
deduction time T as rho -> rho_min: `48581590` gives rho_min = 0.442/0.462/
0.475/0.475/0.479/0.477 and T(0.90)->T(~0.50) rising 4.1x/4.8x/5.4x/5.7x/
6.0x/7.9x for N = 24/48/96/192/384/996. The gradient exists at N=24, so it is
testable on existing checkpoints.

**Transformer side task:** best is `d_model=128, nhead=4, num_layers=12,
dim_feedforward=1024, epochs=900` (`48513926_1`, dir
`sudoku_hparam_sweep_2026-10-19_ep900/d0128_h04_L12_ff01024_lr1e-03_pre`):
`agg_solve_rate=0.767` at `mean_o1_solve_rate=0.968`. Three levers are all
negative on the attention-targeting metric: more heads (`48296020`), wider
heads (`48446043`), 3x budget (`48513926`).

**Interpretability (2026-10-19, new):** probes on the best checkpoint
(`48580216`) show the model represents bookkeeping but NOT DD's predicate:
`known_mask` balanced-acc 0.593 (layer-0 control) -> **0.920** by layer 3-4,
while `frontier_mask` (deducibility, needs graph table joined to known set)
only reaches **0.669** from a 0.569 control. `next_color` is computed late
(0.459 at L7 -> 0.651 at L12). Reads as "track what's colored, then guess the
next pair", not "compute the deducible set, then pick from it" -- but a weak
linear probe is not proof of absence, which is what the causal job tests.

## Running jobs
- `48579019` scratchpad training, `48579545` random-hint training (both R).
- `48580229` checkpoint series (R) -> `48582169` series probes (dependency,
  answers "when during training").
- `48581589` hint-density transition eval, `48581592` SAE, `48582016` circuits
  (all PD).
No other session is on `spin_gnn` (checked `ListAgents`/`squeue`).

## Open problems
1. Phase transition: `d=5` unconfirmed (needs `sudoku_mcmc.anneal`, never run
   for results); `<m>` order parameter not done; `d=4,6,8` data unanalyzed.
2. Transformer: content-addressing ceiling (1.13-1.26x enrichment) survives
   head count, width and budget. `agg_solve_rate` pinned ~0.77 while the
   near-trivial end is ~0.97.
3. `d_model=256` only ever run at 300 epochs, budget-confounded.
4. Whether the weak `frontier_mask` probe means the model lacks DD's frontier
   or merely encodes it nonlinearly -- `48582016` (causal) decides this.

## Next steps
1. Read out `48581589` (does solve_rate collapse where `dd_mean_T` diverges?
   `truncated_rate` is reported per bin to rule out the length confound),
   `48581592` (SAE -- check reported L0 before trusting feature labels) and
   `48582016`, then build a results notebook + slide PDF and upload.
2. `48582169` lines DD-structure emergence against the `series.csv`
   solve_rate curve.
3. Phase transition main task: `sudoku_mcmc.anneal` for `d=5`; `<m>` vs rho.
