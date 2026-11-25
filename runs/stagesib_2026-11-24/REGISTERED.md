# STAGE SIB — the gate-dropped sibling seeds of every single-seed cell

Registered 2026-11-24, **before the scorer ran and before the job was submitted**.
Branch `sudoku-dd`. Finding number to be 136.

## What hshunt approved

Finding 135 (read-only audit, committed `4370f01`) established that **all six
single-seed cells in the 79 are single-seed by *selection*, not by budget**:
every sibling seed was trained, has a `model.pt` on disk, was scanned at
ρ = 0.50, and was dropped because `stagepos_cells.successful_runs()` admits a
cell only on

    if len(at50) == 1 and float(at50[0]["solve_rate"]) > 0:

Asked in one line whether to score the 10 remaining dropped siblings on the rule
arm — no training, ~20 GPU-minutes — hshunt answered **"Yes, all 10"**.

**NOTHING IS TRAINED HERE.** This stage runs the existing stage LOGIC
measurements L and F on checkpoints that have been sitting on `$SCRATCH` since
2026-11-04 / 11-05 / 11-10, against stage POS's **cached** instances, so the
comparison against the admitted seed is on identical puzzles.

## The twelve checkpoints, and why twelve rather than ten

`POS_EXTRA` appends in order after the seven, so these take `r80` upward.

| run | cell | seed | checkpoint (`<scan>/<task>`) |
|---|---|---|---|
| r80 | tied_no_mlp L=16 d192 | 1 | `sudoku_n48_tied_P_2026-11-10/t01_…_tied_no_mlp_L16_s1` |
| r81 | tied_no_mlp L=16 d192 | 2 | `sudoku_n48_tied_P_2026-11-10/t02_…_tied_no_mlp_L16_s2` |
| r82 | no_mlp L=4 d256 | 1 | `sudoku_n48_minarch_2026-11-04/t22_L04_h04_no_mlp_s1` |
| r83 | no_mlp L=4 d256 | 2 | `sudoku_n48_minarch_2026-11-04/t23_L04_h04_no_mlp_s2` |
| r84 | rf_mlp L=4 d256 | 0 | `sudoku_n48_minarch_2026-11-04/t15_L04_h04_rf_mlp_s0` |
| r85 | rf_mlp L=4 d256 | 2 | `sudoku_n48_minarch_2026-11-04/t17_L04_h04_rf_mlp_s2` |
| r86 | no_mlp L=6 d128 ep100 | 0 | `sudoku_n48_minarch_G_2026-11-05/t00_…_no_mlp_L06_ep100_s0` |
| r87 | no_mlp L=6 d128 ep100 | 1 | `sudoku_n48_minarch_G_2026-11-05/t02_…_no_mlp_L06_ep100_s1` |
| r88 | no_mlp L=12 d128 | 0 | `sudoku_n48_minarch_H_2026-11-05/t00_…_no_mlp_L12_s0` |
| r89 | no_mlp L=12 d128 | 1 | `sudoku_n48_minarch_H_2026-11-05/t02_…_no_mlp_L12_s1` |
| r90 | tied_no_mlp L=16 d448 | 1 | `sudoku_n48_tiedwidth_S_2026-11-10/t04_…_tied_no_mlp_L16_s1` |
| r91 | tied_no_mlp L=16 d448 | 2 | `sudoku_n48_tiedwidth_S_2026-11-10/t05_…_tied_no_mlp_L16_s2` |

**`r80` and `r81` are the two d192 siblings stage SEED already measured, and they
are deliberately re-run.** Two reasons, both worth ~4 GPU-minutes:

1. **It removes a run-id collision.** Stage SEED's `POS_EXTRA` had two entries,
   so its shards call the d192 siblings `r80`/`r81`. If this stage appended only
   the ten new checkpoints, *its* `r80`/`r81` would be `no_mlp L=4` seeds 1 and 2
   — the same ids naming different checkpoints in two published stages. Putting
   d192 first makes `r80`/`r81` mean the same thing in both by construction.
2. **It is the only end-to-end check that this pipeline equals the one 134
   published from** (D5 below). Quoting 134's numbers instead would assume that.

## Baselines — the admitted seed of each cell, recomputed from the published shards

All 14 densities, 400 instances per bin, eval seed start 60,000,000,000.
`h_rule` pools every `j`; `u = 1 − s_F(0)` is `j = 0` only. **The two arms are
measured on different blocks** — that is 134's corrected D5 and is the
convention here.

| cell | run | seed | published `h_rule` | wrong / forced | `u(0)` | params | rule? |
|---|---|---|---|---|---|---|---|
| tied_no_mlp L=16 d192 | r68 | 0 | 0.08106 | 977 / 12,053 | 0.4359 | 270,010 | **breaks** |
| no_mlp L=6 d128 ep100 | r23 | 2 | 0.03618 | 677 / 18,714 | 0.2632 | 480,058 | **breaks** |
| no_mlp L=12 d128 | r44 | 2 | 0.00004 | 1 / 24,737 | 0.2550 | 879,418 | intact |
| tied_no_mlp L=16 d448 | r78 | 0 | 0.00000 | 0 / 26,949 | 0.2357 | 1,088,698 | intact |
| no_mlp L=4 d256 | r08 | 0 | 0.00000 | 0 / 24,955 | 0.2379 | 1,218,106 | intact |
| rf_mlp L=4 d256 | r09 | 1 | 0.00008 | 2 / 23,891 | 0.2712 | 2,267,706 | intact |

**Two of the six break the 0.01 bar and four are 125× to ∞ under it.** That split
is what makes this stage worth running: 134 seeded a cell that was *already* a
rule-breaker, so it could not distinguish "the gate hides bad seeds" from "this
particular cell is unstable". Four rule-intact cells with hidden siblings can.

## Predictions

**S1 — THE SELECTION EFFECT IS GENERAL: every cell gets worse.** For all five
newly-measured cells, pooled `h_rule` over (admitted + siblings) exceeds the
published value. HIT only if all five move up; a single cell that does not is a
MISS, because the claim is about the gate and not about any one architecture.

**S2 — AT LEAST ONE CURRENTLY RULE-INTACT CELL BREAKS THE 0.01 BAR.** Among
`r44`, `r78`, `r08`, `r09`, at least one sibling measures `h_rule > 0.01`.
**This is the consequential prediction.** If it HITS, "the rule is a property of
the architecture" fails for cells that are published as rule-intact, and four
more rows in the ladder rest on between 5 and 14 solved instances of 400. If it
MISSES, the selection effect is confined to cells already near the bar and the
intact rows survive.

**S3 — THE SIBLINGS CLUSTER HIGH RATHER THAN NEAR THEIR OWN CELL.** Median
`h_rule` over the ten new siblings **> 0.20**. Rationale: d192's two sat at
0.5656 and 0.5641, within counting noise of each other and 7× from their own
cell's published value, which suggests *being gate-dropped* is more predictive
than the architecture. A low median means the opposite — that a dropped sibling
of an intact cell is still mostly intact.

**S4 — `u` FALLS AGAIN, AS IT DID IN 134.** For **≥ 6 of the 10** siblings,
`u(0)` is below its own cell's published `u(0)`. 134 registered `u` to rise and
measured it falling (0.4359 → 0.2882, 0.2655); if that direction is a property
of bad seeds rather than of d192, it repeats here. Note four of the six cells
already sit near 0.24–0.27, the "half of chance" clump of finding 129, so there
is less room to fall than d192 had.

**S5 — SURVIVORSHIP SHOWS IN THE DENOMINATOR.** For **≥ 8 of the 10** siblings,
`n_F` is below the admitted seed's `n_F` on identical instances, because a
rollout is censored at its first error. d192 ran 12,053 against 6,531 and 6,745,
a factor of 1.82. Where this holds, `h_rule` **understates** the gap.

## Diagnostics

**D1 — THE TWELVE ARE THE CELLS, SEEDS AND PARAMETER COUNTS NAMED ABOVE**, all
with `in79 = False` and `in_seven = False`, and `r80`/`r81` resolve to the two
d192 sibling checkpoints exactly as in stage SEED.

**D2 — `POS_EXTRA` UNSET RETURNS THE PUBLISHED POPULATION UNCHANGED**: 80
records, 79 `in79` plus 1 flagged, ids `r00`–`r79`. Set, it returns 92 with
`r00`–`r79` untouched. `stagerule_fit.load()` drops `in79 = False`, and the
shards are written as `F_hazard_r80-r91.csv`, which the published `*of10` glob
cannot match — so **no published fit can absorb these rows even with `POS_EXTRA`
exported.**

**D3 — THE SIBLINGS ARE SCORED ON THE SAME INSTANCES AS THEIR ADMITTED SEED**:
14 densities, `n_instances = 400` on every bin, and `|realized_rho − r68's|`
= 0 over the grid. Instances are reused, not rebuilt, so this must be exactly 0.

**D4 — `n_err_in_F ≤ n_in_F ≤ n_at_risk` ON EVERY ROW**, zero violations.

**D5 — RE-MEASURING `r80`/`r81` REPRODUCES STAGE SEED EXACTLY.** Same
checkpoints, same cached instances, same scorer, so `h_rule` must come back
0.5656 (3,694 / 6,531) and 0.5641 (3,805 / 6,745), and `u(0)` 0.2882 and 0.2655.
Any disagreement means the pipeline moved between 134 and here and **this
stage's numbers may not be compared with 134's** until it is explained.

**D6 — THE SIX ADMITTED SEEDS RECOMPUTED FROM THE PUBLISHED SHARDS REPRODUCE THE
BASELINE TABLE ABOVE**, on the mixed-block convention (`h_rule` over all `j`,
`u` at `j = 0`). This is the check 134's first D5 got wrong by demanding one
count block serve both arms.

## What this stage cannot say

Two seeds per cell, five cells, one graph population (`d5q4`, N = 48). It cannot
establish that seed dispersion is general — 133 found d320's three seeds agreeing
*inside* counting noise while 134 found d192's differing 7×, so dispersion may
itself be width-dependent. It also cannot re-admit anything: a sibling scored
here stays `in79 = False` and is never pooled into the 79, because "the 79" is a
published population and findings quote its size. And it says nothing about the
*solving* gate itself being the right filter for a solver comparison — only that
it is not neutral on the rule arm.
