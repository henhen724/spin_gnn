# DD Sudoku on random regular graphs: handoff

Goal: numerically explore the reasoning-depth phase transition from the talk
"A Transition in Reasoning Depth in Sudoku on Random Graphs" (slides not in
this repo; summarized below). This file explains the model, what code exists,
what has been verified, what's been observed so far, and what to do next.

## The model (from the slides)

- `G = (V, E)`: random **d-regular** graph, `N` nodes, with a **planted
  proper q-coloring** `x*`. Main case in the slides: `q = 4`, `d = 5` and
  `d = 7`.
- A hint set `V_hint ⊆ V` has its colors revealed; `rho = |V_hint| / N`.
- **Direct deduction (DD)**, synchronous rounds: a non-hint node is deduced
  one round after its deduced neighbors show all `q-1` colors other than its
  own ("naked single" -- this is exactly the slides' `A^T` factor). Only
  Sudokus that DD fully solves are considered.
- Deduction time `t_i` (we use hints at `t = 0`); `T = max_i t_i` = number of
  sequential rounds = "reasoning depth"; order parameter
  `m = (1/N) sum_i t_i / T`.
- **Claimed phases**: high rho is the *fast* phase (O(N) deductions per round,
  `T ~ log N`, `<m> -> 0` as `N -> inf`); below `rho_c` is the *slow* phase
  (o(N) per round, long sequential chains, `T ~ N^gamma(rho,q,d)`, `<m> > 0`).
- **Slide numbers to reproduce** (read off plots, approximate):
  - q=4, d=5: `rho_c ≈ 0.46`, `<m>` transition **continuous** (theory: linear
    from ~0.19 at rho=0.3 to 0 at rho_c).
  - q=4, d=7: `rho_c ≈ 0.27`, `<m>` transition **discontinuous** (jump to ~0.25-0.3).
  - Plot axes: `T/log N` vs rho (log y) for N in {996, 1416, 2028, 2904, 4140,
    5916, 8436, 12048, 17196, 24540, 35028, 49992} (all multiples of 12 = q(q-1));
    curves collapse above rho_c and fan out below. Curves span rho ≈ 0.3–1.0
    (d=5) and ≈ 0.15–1.0 (d=7).
- **Slides' numerical protocol**: start from the fully colored graph, remove
  colors step by step, keep a removal only if DD still recovers the full
  coloring; record (rho, T, m) along the way.
- **Theory (not implemented)**: dynamical cavity method on the measure over
  (hint set, coloring, times) solvable within budget T, with tilt `mu` on hints
  (cf. Behrens et al., *Backtracking Dynamical Cavity Method*, PRX 2023); RS
  ansatz with uniform messages `psi(h, h')`, `h = (x, t)`. The `T -> inf` limit
  is studied like bootstrap percolation (Guggiola & Semerjian, *Minimal
  contagious sets in random regular graphs*, 2014) through a 2D shooting
  problem in `X_t = sum_{l<=t} psi`, `Y_t`; three mechanisms between left/right
  boundary manifolds → one fast branch, two slow branches (continuous vs
  discontinuous `<m>`). They also compute a finite-T entropy `s_T(rho)`.

## Code (new, all in `spin_gnn/`)

| File | What it does |
|---|---|
| `sudoku_dd.py` | `planted_regular_graph(N, d, q, rng)`: balanced planted coloring + configuration model restricted to cross-color edges (each color-class pair gets exactly `dN/(q(q-1))` edges), multi-edges removed by switches. Needs `q | N` and `q(q-1) | dN`. `dd_times`: from-scratch DD (numba), returns times (INF = never). `remove_hint`: **incremental** update of all times when one hint is removed, reverting if DD would fail. `removal_path`: one random-order greedy removal path from rho=1 to a minimal hint set, recording (rho, T, m) after every accepted removal + full `t` snapshots at requested rhos. `on_grid`: sample a path on a rho grid. |
| `sudoku_sweep.py` | CLI: `python -m spin_gnn.sudoku_sweep --degrees 5 7 --sizes ... --seeds 24 --out-dir DIR --workers K`. One `.npz` per (q,d,N,seed): `rho_grid, T, m, rho_min, final_t, snap_<rho>`. Skips files that already exist, so it can be resumed. |
| `sudoku_mcmc.py` | Metropolis sampler over DD-solvable hint sets with weight `e^{mu |V_hint|}` (the T=∞ tilted ensemble). `add_hint` is the decreasing counterpart of `remove_hint`. `anneal(...)` sweeps mu downward from a greedy endpoint. CLI: `python -m spin_gnn.sudoku_mcmc --d 5 --N 4140 --mu-min -3`. **Exact moves are tested; not yet used for any results.** |
| `tests/test_sudoku_dd.py` | 32 tests: graph is simple/regular/properly colored; after **every** attempted removal / MCMC move, incremental times == `dd_times` from scratch; accept/reject matches from-scratch solvability; greedy endpoint is minimal. All pass. |
| `scripts/slurm/sudoku_dd_sweep.sbatch` | 16 CPUs / 4 h sweep over d ∈ {4..8}, the slide N values, 24 seeds → `$SCRATCH/spin_gnn/sudoku_dd_results`. Installs `numba` into the existing venv if missing. |

Dependency added: **numba** (in `pyproject.toml`).

Key implementation facts (useful if you modify things):
- DD times are the least fixed point of `t_i = 1 + max_{a≠x_i} min_{j∈∂i, x_j=a} t_j`.
  A node's time is justified only by neighbors with **strictly smaller** time,
  so justification is acyclic. That's what makes the incremental updates exact.
- Removing a hint only increases times. `T` is monotone along a greedy path.
- DD closure is monotone in the hint set, so one pass over a random order gives
  a minimal hint set: a node that can't be removed now can never be removed later.
- Conventions differ from slides possibly by an offset (slides may put hints at
  t=1). `<m> -> 0` at rho=1 in the slides is consistent with hints at t=0.

## Observations so far (few seeds; preliminary)

Performance: greedy paths take ~0.03 s at N=8k and ~3 s at N=200k for d=5. For d=7
at N=50k they take ~60 s because failed removals cascade in the slow phase.

| q,d | greedy endpoint rho_min | T at endpoint | Slide rho_c |
|---|---|---|---|
| 4,5 | ≈ 0.477 (N = 8k–200k, very stable) | 50–90, grows slowly with N | ≈ 0.46 |
| 4,7 | ≈ 0.225 | 370 (N=8k) → 1390 (N=50k), i.e. `T ~ N^0.74` | ≈ 0.27 |

Interpretation / open issue:
- **d=5: random-order greedy removal gets stuck just above rho_c**, so it cannot
  reach the slow phase the slides show down to rho=0.3. The slides' sampler
  must differ (non-random removal order? a different/bigger move set?). That's
  why `sudoku_mcmc.py` exists: anneal mu < 0 to push rho below the greedy endpoint.
- **d=7: greedy goes below rho_c into a clearly slow phase** (power-law T).
- Partial sweep output (~64 files, d=6 N=49992) exists only on Henry's laptop;
  ignore it and rerun on Sherlock.

## Next steps

1. **Run the greedy sweep on Sherlock** (never on the login node; see
   `scripts/setup_env.sh`): `sbatch scripts/slurm/sudoku_dd_sweep.sbatch`.
2. **Reproduce slides 11–12**: `T/log N` vs rho and `<m>` vs rho per d, curves
   over N, mean ± stderr over seeds (ignore grid points below a run's
   `rho_min`; report how many seeds contribute). Mark slide rho_c.
3. **Locate rho_c from data**: (a) where `T/log N` curves stop collapsing;
   (b) fit `log T = gamma(rho) log N + c` per rho and find where gamma → 0;
   (c) peak of seed-to-seed variance of m. Compare with 0.46 / 0.27.
4. **Go below the greedy endpoint for d=5** with `sudoku_mcmc.anneal`
   (start N ≈ 4k–17k; check equilibration by comparing `start="greedy"` vs
   `start="full"` at the same mu, and the per-sweep time series). Bin samples by
   rho and overlay on the greedy curves. Is there a slow phase at rho < 0.46, and
   is `<m>` continuous for d=5 and discontinuous for d=7?
5. **Order of the transition vs d**: d=6 (and 4, 8) to see where `<m>` switches
   from continuous to discontinuous.
6. **Per-round deduction profile**: from `snap_<rho>`, plot `#nodes deduced at
   round t / N` vs t: O(N) bursts (fast phase) vs a long o(N) tail (slow phase).
   Also the empirical cumulative `P(m) = (1/N) sum 1[t_i <= mT]` to compare with
   the slides' two slow-phase classes.
7. Optional: finite-budget picture (slide 16), i.e. the fraction of sampled hint
   sets with T ≤ budget vs rho; and the cavity-method theory itself.
8. Tie-in with the rest of this repo (k-SAT reasoning traces, `ksat2.py`): DD on
   these graphs is a clean generator of reasoning traces with *tunable depth*
   (rho controls fast vs slow), a natural dataset for the transformer
   experiments.

Add a short README section once results exist.
