# Stage GATE — registered before any of it was computed

Written and committed **before** `scripts/stagegate_fit.py` existed. No number
below was chosen after seeing it. Serves the caveat stage REGIME (98) exposed in
the criterion that every rung table in this project has been read against.

## The question

98 found that the free-generation scorer calls a pair "ok" when
`node_valid and color_correct`, where `node_valid = node not in assigned` means
only **not already coloured** and never **forced**. So a cell emitted before
direct deduction has pinned it down is not an error at all, provided the colour
is right — and the model colours such cells right about half the time.

Therefore the quantity the whole project ranks architectures by —

```
rho_50 = the hint density at which the free-generation solve rate crosses 0.50
```

measures **"produces the planted colouring"**, not **"executes direct
deduction"**. The brief is the smallest or most analytically tractable
architecture that still learns *direct deduction*. The gate may be the wrong
instrument for the brief.

This stage recomputes `rho_50` under a **DD-faithful** criterion: a rollout
counts as solved only if it never named a cell that DD had not already forced.
It is a refit of CSVs already on disk: **no training, no GPU, no new instances,
nothing submitted.**

## Definitions

From `F_hazard_all_<i>of10.csv`, per (run, rho, j):

$$h_j^{\text{logic}} = \frac{n_{\text{err}}(j) - n_{\text{err\_in\_F}}(j)}{n_{\text{at\_risk}}(j)}
\qquad
h_j^{\text{strict}} = \frac{n_{\text{err}}(j)}{n_{\text{at\_risk}}(j)}$$

$$\widehat{S}_{\text{logic}} = \prod_j \bigl(1 - h_j^{\text{logic}}\bigr)
\qquad
\widehat{S}_{\text{strict}} = \prod_j \bigl(1 - h_j^{\text{strict}}\bigr)$$

over every `j` with `n_at_risk > 0`. `S` is the measured `solve_rate` column of
`LF_summary_all_<i>of10.csv`. `n = 48(1 - rho_real)`; magnitudes are scored only
at the six densities with `n >= 12` (76, 79). Group values are medians over runs
with the spread quoted. Runs are keyed on the **`cell`** label, which is the fix
96 demanded after keying on `(arch_mode, num_layers, d_model)` silently pooled
k = 1, 2, 4.

`rho_50` is read by scanning the 14-point grid from high rho downward and
linearly interpolating in rho across the first bracket where the rate crosses
0.50. Registered now so the method is not chosen to suit the answer. A run whose
curve never crosses inside the covered grid is **reported as censored, never
dropped silently.**

## The estimator is an UPPER bound, and that asymmetry is registered in advance

`h_j` counts only logical first errors at **node** positions. A rollout can also
die of grammar, early termination or truncation, classes 97 measured at solve
rate 0. So `S_logic` **ignores those failures** and is an upper bound on the true
DD-faithful solve rate; the `rho_50` it yields is correspondingly an optimistic
(low) bound.

**This makes the evidence asymmetric and the asymmetry is stated before the
result:** if a run FAILS the gate under this estimator it fails for real, while
if it PASSES, that is an upper-bound pass only and does not clear it.

A second, smaller bias runs the other way: the at-risk set drops by the
**strict** error count (registered fact of 98, 18,602 transitions), so benign
strict errors are censored out of the denominator. 98 measured those at
**0.73%** of all first errors, which bounds this term.

## What was inspected before registering, and what was not

- **Column names** of `F_hazard` (23 columns) and `LF_summary` (34), and that
  both cover all **14** grid densities, 0.95 down to 0.33.
- **Row count** of `LF_summary`: **1120** = 80 runs x 14 densities, over **32**
  distinct `cell` labels.
- **Not inspected:** any solve rate, any hazard value, any product, any `rho_50`.

## Predictions

**G1 — THE DD-FAITHFUL CRITERION SQUARES THE SOLVE RATE.** 97 decomposed
`-log S = R * n * e_node` with fatal share `R ~ 0.37-0.61`, and 98 showed that
`R` is `Pr(colour wrong | cell not yet forced)` rather than anything about
position. Requiring faithfulness removes exactly that forgiveness, so
`-log S_logic` should recover the undiscounted `n * e_node`, i.e.
`log S_logic / log S ~ 1/R ~ 1.6-2.7`.

*Prediction:* the median ratio lies in **[1.5, 2.8]** at a majority of the six
densities with `n >= 12`. *Falsified if* it is outside [1.2, 4.0] at a majority.
The content is that REGIME's `j = 0` identity — the two instruments agree on the
per-decision rate to four decimals — **extends across all `j`**, with the
deviation bounded by the Jensen term 98 measured at 0.64-1.00%.

**G2 — J2 STILL CLEARS THE GATE.** The standing minimum `no_mlp L=16 d192`
sits at `rho_50` 0.580 against a gate of 0.740, so it carries 0.160 of headroom.

*Prediction:* its DD-faithful `rho_50` is **<= 0.740**. *Falsified if* it exceeds
0.740 — in which case, by the upper-bound argument above, the gate is really
broken and every rung table that used it needs re-reading.

**G3 — THE RANKING SURVIVES EVEN IF THE CRITERION DOES NOT.** Whether past rung
tables stand as *orderings* is a separate question from whether their absolute
criterion was loose.

*Prediction:* the Spearman correlation of `rho_50` with DD-faithful `rho_50`,
over runs where both are defined, is **>= 0.9**. *Falsified if* it is < 0.6.

**G4 — THE PENALTY IS WEAKEST WHERE THE TASK IS HARDEST.** 97 found `R` rises as
rho falls (0.443 at rho 0.75 to 0.607 at rho 0.50), so `1/R` falls, so the
squaring penalty should shrink with difficulty — the opposite of the intuition
that a stricter criterion bites hardest on hard instances.

*Prediction:* the Spearman of the median ratio with rho, over the ten densities
with a fit, is **> +0.6**. *Falsified if* it is negative.

**G5 — THE ORDERING AT THE TOP OF THE RUNG TABLE IS UNCHANGED.** J2 (2,504,890,
attention-only) and the hub `full L=6 d256` (4,899,898) sit at 0.580 and 0.592,
0.012 apart, and which of them is "the smallest that still learns DD" is the
brief's live question.

*Prediction:* under the DD-faithful criterion they remain within **0.05** of each
other **and** J2 remains the lower. *Falsified if* they separate by more than
0.10, or swap order.

**D1 — THE PRODUCT ESTIMATOR REPRODUCES A QUANTITY ALREADY MEASURED.** Fed the
strict hazard it must return the measured solve rate, since that is what stage
POS's P2(a) checked directly.

*Expected:* median `|S_strict_hat - S|` **< 0.01** and max **< 0.05** over cells
with `n >= 12`, consistent with POS's known residual of 2.25e-02 traced to 101 of
448,000 rollouts. **A failure invalidates G1-G5**, every one of which rests on
the product form.

**D2 — MONOTONICITY, EXPECTED EXACTLY.** The DD-faithful criterion is strictly
stricter, so `S_logic <= S_strict_hat` on **every** cell and
`rho_50_logic >= rho_50` on **every** run. Any violation is a bug in the scorer,
not a finding.

## Figures

| | |
| --- | --- |
| `G1_ratio.png` | `log S_logic / log S` against rho, median over runs, with 97's `1/R` band |
| `G2_curves.png` | both solve-rate curves for J2, with the 0.50 crossings marked |
| `G3_scatter.png` | DD-faithful `rho_50` against ordinary `rho_50`, one point per run, identity line and the 0.740 gate |
| `D1_closure.png` | `S_strict_hat` against the measured solve rate |

## Output

`runs/stagegate_2026-11-17/`: `gate_cells.csv` (per run, rho: both hazard
products, the measured rate, the ratio), `gate_runs.csv` (per run: both
`rho_50`s, censoring flags, params, `cell`), `VERDICT.txt`, `figures/`.

Runs on a **login node in seconds** — CSV reading, products, medians, one
Spearman per density and matplotlib. **Nothing is submitted.**
