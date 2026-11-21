# Round width across an N ladder, 48 to 1020

Request from hshunt, 2026-11-20:

> For slide 23 of round width, I would be interested in a plot of the width
> with more different N's plotted. Preferably some range from 48-1024, so
> something like that.

Slide 23 of `/spin-gnn/round-width/` is "The pooled curves mix *wide round*
with *easy puzzle*", carrying `figures/confound.png` — mean hint density $\rho$
per round-width bin, at N=48 and N=96 only, because those were the only two
sizes with trained models.

## Why this needs no training and no checkpoint

Round width is the number of cells direct deduction forces in one pass. It is a
property of the **instance**, computed by `solve_with_dd_trace` as
`len(events)` for each `events` in `trace["rounds"]`. No model is involved, so
it costs ~0.03 s per instance even at N=1020 and the ladder is reachable on
CPU in minutes.

**This is the limit of what the new figure can say.** Models exist only at
N=48, N=96 and (training now, job 53062385) N=192. Every accuracy-vs-width
curve elsewhere in the deck stays at N=48/96. The new pages are instance
combinatorics and must be labelled so, or a reader will take them for model
performance at N=1020.

## Why 1020 and not 1024

`planted_regular_graph` requires `n_colors | n_nodes` and
`n_colors(n_colors-1) | degree·n_nodes`. With q=4, d=5 that is `4 | N` and
`12 | 5N`; since `gcd(5,12)=1`, **`12 | N`**. 1024/12 is not an integer, so
**N=1024 is not a legal graph size here**. The ladder is

    48, 96, 192, 288, 384, 576, 768, 1020

all divisible by 12, spanning the range hshunt asked for.

## Cells

Per N: five pinned densities 0.90 / 0.75 / 0.60 / 0.45 / 0.35 (matching stage
SCALE), plus one cell drawn from the full training mix $\rho \sim U[0.33,
0.95]$. 300 instances each, `fall=True`, `shuffle_adj=True`, seeds from
80e9 (disjoint from the 14-point eval grid at 60e9 and stage SCALE at 70e9).

The pinned cells give mean width against N. The **pooled** cell is the one that
reproduces slide 23's construction: within a pinned cell the density is
effectively constant, so a confound curve built from five discrete values would
be a staircase. Slide 23's figure came from the continuous mix and the new
version must too.

## Diagnostics, not predictions

This is a descriptive measurement made to draw a figure, not a hypothesis test,
so there is nothing to pre-register. Two things can actually fail:

| # | check | bar |
|---|---|---|
| DW1 | an instance's round widths sum to its blank-cell count | 0 violations |
| DW2 | achieved density is never *below* target by > 0.02 | 0 violations |

DW2 is **one-sided on purpose**. `build_instances` walks a removal path and
stops at the target; if that path's own sparsest DD-recoverable point is denser
than the target it falls back to that point. That is documented behaviour, and
it bites at small N and low $\rho$ — at N=48, $\rho = 0.35$ sits below
$\rho_c \approx 0.46$ and is often unreachable, so the achieved density
overshoots upward. A density *below* target would instead mean a broken
instance. The overshoot is reported per cell as `mean_achieved_rho` and
`max_rho_gap`, and the deck labels curves with what was sampled rather than
what was asked for.

Within a width bin every round contributes the same token count $w$, so
round-weighted and token-weighted mean density coincide; `rho_by_w` records the
summed instance density per width so the pooled curve is exact.

## Files

- `scripts/widthladder_probe.py`
- `scripts/slurm/widthladder_2026-11-20.sbatch` — `sganguli`, CPU only, 4 cores,
  2 h wall
- job 53177844 (53177722 was the same sweep cancelled after ~1 min, before the
  `rho_by_w` and pooled-cell fields were added)
- output `width_ladder.csv`, 48 rows
