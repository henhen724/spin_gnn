# Stage RULE5 — registration

Written and committed **before** `scripts/stagerule5_fit.py` existed, same
discipline as stage RULE4. Population `d5q4` at `N = 48`, the same 79 runs and
the same `F_hazard` shards as stages RULE2, RULE3 and RULE4. **Nothing is
trained and nothing is submitted.** No closed knob (width, cosine decay, more
data, best-val selection, role count, schedule length, training budget) is
touched.

## What stage RULE4 left open, in its own words

Stage RULE4 removed censoring exactly, by scoring only the first emission of
each rollout: at `j = 0` every rollout is still at risk, so nothing has been
filtered. It then said on its own slides what that does **not** fix:

> `h_rule` conditions on the model having **named a forced cell**, and that
> share is density-dependent — 96.3% at `rho = 0.95` down to 75.9% at
> `rho = 0.33`. Censoring-free is not unbiased.

That is the bias this stage removes. `h_rule(0)` is

$$h^{\mathrm{rule}}(0) \;=\; \frac{n_{\mathrm{err\,in\,}F}(0)}{n_F(0)}$$

whose **denominator is chosen by the model**: `n_F(0)` counts only those
rollouts whose first emission named a cell that one-step propagation forces.
If the model names an unforced cell exactly when it is unsure, the rollouts
remaining in that denominator are the confident ones, and the selection
strengthens as `rho` falls — which is the right shape to manufacture stage
RULE4's low-arm fall out of nothing.

## The quantity

$$h^{\mathrm{all}}(0) \;=\; \frac{n_{\mathrm{err}}(0)}{n_{\mathrm{at\,risk}}(0)}
\;=\; \Pr(\text{the first emitted colour disagrees with the planted trace})$$

Per finding 124, `n_err` counts a **wrong colour**, forced cell or not; naming
an unforced cell with the planted colour is not an error. So this is a rate over
*every* rollout, and:

- **censoring-free** — `j = 0`, nothing has been filtered yet (stage RULE4's
  D1: `n_at_risk(0) = 400` on all 1,106 `(run, rho)` sequences, 0 violations);
- **selection-free** — the denominator is all 400 rollouts, not the subset the
  model chose to name.

**Its denominator is a constant.** 79 runs x 400 instances = **31,600 at every
one of the 14 densities**, by construction rather than by measurement. So a
cross-density comparison of `h^all(0)` is a comparison of raw counts with no
weighting, no selection and no censoring anywhere in it. Nothing else in this
dataset has that property, and neither RULE2, RULE3 nor RULE4 computed it.

## Denominators, read in advance and tabled as design

`n_F(0)` is carried over from the stage RULE4 registration (where it was
likewise read before its numerator). The unforced count is `31600 - n_F(0)`.
**The new numerator `n_err(0)` has not been read.**

| rho | at risk | named forced `n_F(0)` | named unforced | forced share |
|---|---|---|---|---|
| 0.95 | 31600 | 30442 | 1158 | 96.3% |
| 0.90 | 31600 | 29687 | 1913 | 93.9% |
| 0.85 | 31600 | 28502 | 3098 | 90.2% |
| 0.80 | 31600 | 27681 | 3919 | 87.6% |
| 0.75 | 31600 | 27163 | 4437 | 86.0% |
| 0.70 | 31600 | 25198 | 6402 | 79.7% |
| 0.65 | 31600 | 24924 | 6676 | 78.9% |
| 0.60 | 31600 | 24225 | 7375 | 76.7% |
| 0.55 | 31600 | 24201 | 7399 | 76.6% |
| 0.50 | 31600 | 23607 | 7993 | 74.7% |
| 0.45 | 31600 | 23659 | 7941 | 74.9% |
| 0.40 | 31600 | 23530 | 8070 | 74.5% |
| 0.35 | 31600 | 24055 | 7545 | 76.1% |
| 0.33 | 31600 | 23990 | 7610 | 75.9% |

Also carried over, and likewise already read: `n_err_in_F(0)` = 35, 48, 74, 80,
87, 80, 77, 65, 62, 62 | 56, 52, 50, 50. The unforced numerator
`n_err(0) - n_err_in_F(0)` is what this stage reads for the first time.

## Hypothesis

The naming selection, not density, produces stage RULE4's low-arm fall.
Removing it should straighten the curve: `h^all(0)` rises monotonically as
`rho` falls, with no peak, because the unforced emissions that the selection
discards are both far more error-prone and a growing share of the total.

## Quantitative prediction, derived before the read

Finding 124 measured off-support colour agreement pooled over all `j`: 71% at
`rho = 0.95` falling to 50% at `rho = 0.33`, i.e. an error rate of 0.29 rising
to 0.50. Taking those as the unforced error rate at `j = 0` — which they are
not, they are pooled over index, so this is an extrapolation and may fail —

$$h^{\mathrm{all}}(0) \;\approx\; \frac{n_{\mathrm{err\,in\,}F}(0) + e_{\mathrm{unf}} \cdot (31600 - n_F(0))}{31600}$$

- `rho = 0.95`: `(35 + 0.29 x 1158) / 31600 = 370.8 / 31600 = 0.0117`
- `rho = 0.33`: `(50 + 0.50 x 7610) / 31600 = 3855 / 31600 = 0.1220`

a **10.4x rise**. Registering the predicted numbers rather than a direction, so
the test is whether the arithmetic closes and not whether a sign is guessed.

## Scored expectations

- **E1** — `h^all(0)` is monotone in `rho` with no interior peak: Spearman
  over all 14 densities `<= -0.95`. **MISS** if a peak survives with both
  flanks resolved (Poisson CIs on the ratio to the peak excluding 1 on both
  sides), which would mean the hump is not the naming selection.
- **E2** — the unforced error rate exceeds the forced rate by `>= 10x` at every
  one of the 14 densities. **MISS** at any density where the factor is `< 10`.
- **E3** — the predicted endpoints hold within a factor of 1.5: `h^all(0)` at
  `rho = 0.95` in `[0.0078, 0.0176]` and at `rho = 0.33` in `[0.0813, 0.1830]`.
  **MISS** if either falls outside, which would say finding 124's pooled
  off-support rate does not transfer to `j = 0`.
- **E4** — the low-arm fall disappears: stage RULE4 measured Spearman `+0.9758`
  for `h_rule(0)` over the ten fitted bins (a fall as `rho` drops). For
  `h^all(0)` the same ten bins give Spearman `<= 0`. **MISS** if it stays
  positive, i.e. the fall survives the selection fix.
- **E5** — the forced-cell curve is reproduced exactly: recomputing
  `h_rule(0)` from this stage's loader returns the RULE4 per-density table to
  `< 1e-12`. **MISS** on any disagreement — this is a wiring check, not a
  finding.

## Diagnostics, all must pass

- **D1** — `n_at_risk(0) = 400` on every `(run, rho)` sequence; 1,106
  sequences; pooled denominator exactly 31,600 at each of the 14 densities.
- **D2** — `n_err(0) >= n_err_in_F(0)` on every sequence. A violation would
  mean `n_err` is not a superset of the forced errors and invalidates the
  decomposition.
- **D3** — `n_F(0) <= n_at_risk(0)` on every sequence, so the unforced count is
  non-negative.
- **D4** — the `n_F(0)` pooled table reproduces the 14 numbers above exactly.
- **D5** — run count is 79 at every density.

## The interpretive limit, registered in advance

`h^all(0)` measures **agreement with the planted trace, not logical validity**,
and those two come apart on unforced cells:

- On a **forced** cell one-step propagation leaves exactly one colour, which is
  the planted one, so off-trace = logically wrong.
- On an **unforced** cell several colours are locally consistent. Above the
  density where the planted colouring stops being recoverable by propagation
  alone (`rho_c ~ 0.465` at `d5q4`) picking a different one is still a mistake
  — the cell is determined, just not by one step. **Below** `rho_c` it need not
  be: in the four added bins an off-trace colour may be perfectly consistent
  and simply a different valid colouring.

So `h^all(0)` in the four added bins is an **upper bound** on the logical error
rate, and the clean statement lives in the ten fitted bins. Every reported
figure splits the arms for this reason. This limit is named, not fixed: the
shards carry no per-instance solution count, so it cannot be fixed here.

A second limit carried over from RULE4: all 79 runs share the same 400
instances per bin and the shards carry no per-instance breakdown, so **no
interval in this stage can cluster by instance**, and the delete-one-run
jackknife treats 79 runs as 79 exchangeable replicates, which they are not.
Both intervals are reported where the numerator is small.

## What a null licenses

**No training follows from any outcome.** If E1 and E4 miss — if the hump
survives the selection fix — that is a stronger version of stage RULE4's
finding, not a reason to submit anything. If they hit, stage RULE4's low-arm
fall is explained as an artefact of the model's own naming and must be
withdrawn as a density effect, which is a correction to a published deck and
again not a training result. The repair-and-continue rollout eval stays
hshunt's call either way.

## Traps

1. `n_err` counts a wrong **colour**, not any departure from DD (finding 124).
   Do not read it as a stopping or formatting failure.
2. The `j = 0` slice is censoring-free but the **pooled** curve is not; never
   compare a `j = 0` number against a pooled reference, which is the error
   RULE4's clean window flagged in place.
3. 31,600 is a constant denominator **only at `j = 0`**. It is not the
   denominator of anything pooled over index.
4. The unforced error rate is scored against the planted trace; below `rho_c`
   it is an upper bound, per the limit above.
5. Spearman over 14 points with a constant denominator is still only 14
   points; report the permutation p-value, not the asymptotic one.
6. Do not convert `h^all(0)` into a density threshold. It is a rate, as
   `h_rule` is.
7. Stage RULE4's published deck says the low-arm fall is real. If this stage
   contradicts it, the deck gets a correction page — it does not get quietly
   rebuilt, and the talk deck `transition-and-minimal-models` is not touched
   at all.
