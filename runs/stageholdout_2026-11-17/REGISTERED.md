# Stage HOLDOUT — registered before any of it was computed

Written and committed **before** `scripts/stageholdout_fit.py` existed. No number
below was chosen after seeing it. This stage tests a claim that is currently in
`STATUS.md` flagged as *not to be believed*, and it is set up so that it can
**retract that claim**.

## The claim under test

Stage GATE (99) reported, as a **post-hoc** read of data gathered for a different
purpose, that the cost of demanding DD-faithfulness

$$\text{ratio} \;=\; \frac{\log \widehat{S}_{\text{ddf}}}{\log \widehat{S}}$$

falls monotonically as the task gets harder: median over runs of 3.456, 3.574,
3.271, 3.130, 3.074, 2.968, 2.911, 2.798, 2.726, 2.668 at
rho = 0.95, 0.90, ..., 0.50, with Spearman against rho of **+0.988**.

Nobody predicted this before seeing it. `STATUS.md` next-step 1 part (ii) says it
"must be re-registered on held-out runs before being believed".

## Why these four densities are held out

`stageexp_fit.py:49` sets `FIT_RHOS = [0.95 ... 0.50] = GRID[:10]`, and
`stagegate_fit.py`, `stageregime_fit.py` and `stagebound_fit.py` all take their
density list from it. **No exponent, ratio, prefactor or regime fit in this
project has ever touched rho = 0.45, 0.40, 0.35 or 0.33.** The rollouts exist —
the 14-point grid was always evaluated — but those four columns have been
carried, never fitted.

**The honest caveat, stated before the test rather than after it.** These
densities are held out *with respect to the claim*, not untouched. They enter
`rho_50` through the downward scan across the grid, so for weak runs whose solve
curve crosses 0.50 below rho = 0.50 they have influenced a published number. They
have never entered a ratio. And they are the **same 79 runs at harder densities**,
not new runs: this tests whether the trend extrapolates, not whether it
generalises to unseen models. Only training would test the latter, and training
is not available here.

## The reason to doubt the claim, which is the point of the stage

The ratio is undefined when `S = 1` (the denominator vanishes) and when
`S_ddf = 0` (the numerator diverges). **Those two exclusions act at opposite ends
of the grid.** At high rho, runs are dropped for solving everything; at low rho,
runs are dropped for solving nothing. Dropping `S_ddf = 0` removes exactly the
cells with the *largest* numerator, so it biases the surviving median
**downward**, and it bites harder as rho falls.

That alone could manufacture the entire +0.988 trend. The population is changing
with density, and a median taken over a changing population is not a trend in any
run. So the decisive test here is not the extrapolation — it is H3, which holds
the population fixed.

## What was inspected before registering, and what was not

- `head -1` of `bound_cells.csv`: the 16 column names.
- `FIT_RHOS` and its uses, by grep, across the four fit scripts.
- **The per-density count of usable cells**, which is how I knew the test was
  possible at all, and which is the only low-density measurement I have seen:

  ```
     rho  cells   0<S<1  0<Sddf<1   both   S==0  Sddf==0
    0.50     79      79        79     79      0        0
    0.45     79      78        75     75      1        4
    0.40     79      71        62     62      8       17
    0.35     79      57        38     38     22       41
    0.33     79      53        30     30     26       49
  ```

- The ten fitted medians above, which are published in deck 25 and `STATUS.md`.
- **Not inspected:** any ratio value at any rho <= 0.45; any complete-case
  recomputation at any density; any paired per-run statistic; any hazard
  decomposition; anything H1-H4 are about.

## Predictions

**H1 — THE TREND CONTINUES BELOW THE FIT GRID.** Ordinary least squares on the
last six fitted points (rho 0.75 -> 0.50, values 3.074 -> 2.668) gives a slope of
about **1.62 per unit rho**. Extrapolating from 2.668 at rho = 0.50 predicts
**2.59** at 0.45 and **2.51** at 0.40.

*Prediction:* the median ratio over usable cells is in **[2.40, 2.70]** at
rho = 0.45 and in **[2.25, 2.65]** at rho = 0.40. *Falsified if* either reaches
**2.668** — the rho = 0.50 value, so the trend has stopped or reversed — or falls
below **1.8**, which would mean it accelerates into a different regime than the
one fitted.

**H2 — THE RANK CORRELATION SURVIVES THE EXTENSION.** Spearman of the median
ratio against rho over all **14** densities is **>= +0.90**. *Falsified if*
< +0.60.

H1 and H2 are the weak predictions: they are what the post-hoc fit already
asserts, extended by four points, and they inherit the censoring bias described
above. H3 is the one that decides whether the claim lives.

**H3 — THE TREND IS NOT A CENSORING ARTEFACT.** Two immunisations, both
pre-specified.

*H3a, fixed population.* Let `C10` be the set of runs for which
`0 < S < 1` **and** `0 < S_ddf < 1` at **all ten** fitted densities, and `C14` the
same over all fourteen. Recompute the median ratio by density over `C10` alone.

*Prediction:* Spearman of the `C10` median against rho over the ten fitted
densities is **>= +0.90**, and every `C10` median is within **0.50** of the
published all-cells median at the same density. *Falsified if* the `C10` Spearman
is **< +0.60**, in which case the +0.988 is an artefact of the population changing
with density, finding 99's exponent trend is **retracted**, and `STATUS.md` must
stop quoting 2.67-3.57 as a measured decline.

*H3b, paired within run.* For every run measurable at both rho = 0.95 and
rho = 0.50, take `ratio(0.50) - ratio(0.95)`. This compares a run with itself and
is immune to any change of population.

*Prediction:* the difference is **negative for >= 70%** of such runs, and its
median is **< -0.30**. *Falsified if* fewer than **55%** are negative — a coin
flip within runs would mean the between-density trend lives entirely in which
runs survive, not in any run's behaviour.

*Reporting rule, fixed now:* if `|C14| < 20` the fourteen-density version of H3a
is reported as **underpowered** and not used to decide anything. The counts above
make `|C14| <= 30` likely.

**H4 — WHICH HAZARD MOVES.** Column `njs` gives the number of contributing
indices, so the per-index geometric-mean hazards

$$\bar h = 1 - \widehat S^{\,1/n_{js}}, \qquad \bar h_{\text{ddf}} = 1 - \widehat S_{\text{ddf}}^{\,1/n_{js}}$$

satisfy `ratio = log(1 - h_ddf) / log(1 - h)` exactly. A falling ratio therefore
means one of two different things: the colour hazard catching up, or the
faithfulness hazard easing off. These are not the same story about the model and
the deck has to tell the right one.

*Prediction:* **both** hazards rise as rho falls — median `h_ddf` at rho = 0.50
exceeds median `h_ddf` at rho = 0.95 — and median `h` rises by a **strictly
larger factor** than median `h_ddf` over the same span. That is, the ratio falls
because premature naming becomes *relatively* less dominant, not because the
model becomes more DD-faithful on harder instances. *Falsified if* median `h_ddf`
falls with rho, or if it grows by at least as large a factor as median `h`.

## Diagnostics

**D1 — NO LOADER DRIFT.** The ten fitted medians recomputed here must reproduce
GATE's published values to **three decimal places**. Any mismatch invalidates
H1-H4 and the stage stops there.

**D2 — THE CENSORING LEDGER, PUBLISHED NOT HIDDEN.** At each of the 14 densities,
report the count excluded and the cause, split `S == 1`, `S == 0`, `S_ddf == 1`,
`S_ddf == 0`. rho = 0.35 and 0.33 are expected to be excluded from the headline by
their own censoring and must be **shown as censored, never dropped silently** —
the project's standing convention.

**D3 — MONOTONICITY AT THE NEW DENSITIES.** `S_ddf <= S` on all **1106** cells,
including the 316 below the fit grid. Any violation is a scorer bug.

## Figures

| | |
| --- | --- |
| `H1_extrapolation.png` | median ratio against rho, fitted ten solid, held-out four open, the OLS extrapolation drawn as a line |
| `H3a_population.png` | all-cells median against fixed-population median, by rho |
| `H3b_paired.png` | per-run `ratio(0.50) - ratio(0.95)`, sorted, zero line |
| `H4_hazards.png` | median `h` and median `h_ddf` against rho, normalised to their rho = 0.95 values |
| `D2_censoring.png` | stacked count of excluded cells by cause, all 14 densities |

## Output

`runs/stageholdout_2026-11-17/`: `holdout_density.csv` (per rho: medians, counts,
censoring split), `holdout_paired.csv` (per run: the two ratios and their
difference), `VERDICT.txt`, `figures/`.

Reads `runs/stagebound_2026-11-17/bound_cells.csv` only. Runs on a **login node
in seconds**. **Nothing is submitted, nothing is trained, no new instances.**
