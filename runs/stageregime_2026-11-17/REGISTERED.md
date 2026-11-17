# Stage REGIME — registered before any of it was computed

Written and committed **before** `scripts/stageregime_fit.py` existed. No number
below was chosen after seeing it. Serves the one question stage EXPONENT (97)
left open, and it corrects my own STATUS claim that answering it "needs a new
instrument, not another refit" — it does not. The instrument was already
written by stage LOGIC.

## The question

Finding 97 decomposed the solve-rate exponent as `k/n = D x R`:

```
D = e_node / (-log p_logic)      dilution, 1.82-2.00   (why k is pushed above n)
R = (-log S) / (n e_node)        fatal share, 0.37-0.61 (why it is pulled back)
```

`D` is understood: round markers carry 16-20% of the node class's error mass and
are never fatal, so the denominator of `k` averages over positions that cannot
kill a rollout. **`R` is not understood.** `R < 1` says free generation produces
about half as many fatal node events per rollout as teacher forcing produces
node errors — i.e. **teacher forcing is the HARDER regime**, which is backwards
from the usual intuition that generating from your own prefix is harder than
being dragged along a correct one.

97 recorded three candidates and tested none:

- **(i) exposure/state** — teacher forcing scores the model at all `n` node
  positions along the *reference* prefix whether or not its own policy would
  ever have reached them;
- **(ii) order** — in free generation the model emits in an order of its own
  choosing, and stage POS found the per-step hazard *falls* with `j` (P1
  falsified at 31.1% positive);
- **(iii) difficulty** — `R` rises monotonically with difficulty, 0.443 at
  rho 0.75 to 0.607 at rho 0.50.

This stage separates them position by position. It is a refit of CSVs already
on disk: **no training, no GPU, no new instances, nothing submitted.**

## What was inspected before registering, and what was not

Honest disclosure, since I had to know the schemas and two definitional facts
to write predictions over them.

- **Column names and row counts** of `F_hazard_all_*of10.csv` (20,039 rows) and
  `L_position_all_*of10.csv` (9,849 rows per shard; scheme counts `all` 5,600,
  `j` 41,120, `round` 51,760; within scheme `j`, `node` and `color` 20,560 each).
- **Index ranges, structural:** `j` runs 0-32 in both tables, consistent with `j`
  as the deduction index and `n <= 32` at the rho = 0.33 floor.
- **Two definitional facts, checked as booleans over all cells, with no rate
  printed:**
  1. at `j = 0`, `n_at_risk == n_instances` on **1120 of 1120** cells — the
     first node decision is uncensored and both instruments see the same
     instances and the same prefix;
  2. the at-risk set drops by the **strict** error count — `n_at_risk(j+1) ==
     n_at_risk(j) - n_err(j)` on **18,602** transitions, versus the logical count
     `n_err - n_err_in_F` on **0**, with **317** transitions matching neither
     (completions mixed in). So the hazard denominator is "has made no error of
     any kind so far", **not** "has survived", and benign strict errors are
     censored out of it. This is registered as O5 rather than assumed away.
- **Not inspected:** any hazard value, any teacher-forced error rate, any ratio
  of the two, any fit, any per-density number.

O3 is **strongly indicated in advance** by POS's P1 together with LOGIC's P3 and
is registered as a confirmation on a matched pair of instruments, not as a
discovery.

## Inputs, population and conventions

- `runs/stagelogic_2026-11-17/F_hazard_all_<i>of10.csv` — per (run, rho, j):
  `n_at_risk`, `n_in_F`, `n_err`, `n_err_in_F`, `n_F_sum`, `n_instances`.
- `runs/stagelogic_2026-11-17/L_position_all_<i>of10.csv`, rows with
  `scheme = j` and `tok_type = node` — per (run, rho, j): `n_total`,
  `n_logic_correct`.
- `runs/stagelogic_2026-11-17/LF_summary_all_<i>of10.csv` — `solve_rate`,
  `p_logic_node`, `mean_n_blank`, `dd_mean_T`, `in79`, for `R` itself.

Definitions, both **logical** so that finding 94's mis-scoring is already
removed from both sides:

$$e_j \;=\; 1 - \frac{n_{\text{logic\_correct}}(j)}{n_{\text{total}}(j)}
\qquad\text{(teacher-forced logical node error at deduction index } j)$$

$$h_j \;=\; \frac{n_{\text{err}}(j) - n_{\text{err\_in\_F}}(j)}{n_{\text{at\_risk}}(j)}
\qquad\text{(free-generation logical FIRST-error hazard at } j)$$

$$r_j \;=\; h_j / e_j .$$

`in79` runs only; the six densities with `n >= 12` are where every magnitude is
scored, because a magnitude checked at a degenerate point is not checked (76,
79). Group values are medians over runs and the run spread is quoted. Cells with
`e_j = 0` or `n_at_risk = 0` are **dropped with the count reported** at every
density. The join key is `(run_id, rho, j)`; an unmatched row is **counted and
reported**, never dropped silently.

## Predictions

**O1 — THE `j = 0` ANCHOR. This is the pivot of the stage.** At the first node
decision the two instruments are on the same prefix, the same instances, and no
censoring has happened yet (verified above, 1120/1120). If they are the same
rule applied in two regimes, `r_0 = 1`.

*Prediction:* the median `r_0` lies in **[0.8, 1.25]** at a majority of the six
densities with `n >= 12`. *Falsified if* the median is outside [0.5, 2.0] at a
majority. **`r_0` ~ 1 means `R < 1` is ACCUMULATED over `j`** — the regimes start
identical and diverge, which is where candidates (i) and (ii) live. **`r_0` < 1
means it is DEFINITIONAL and present at the very first decision**, and then no
selection story is needed or admissible.

**O2 — SHAPE CANNOT EXPLAIN `R`, BECAUSE THE TEACHER-FORCED RATE IS FLAT IN
`j`.** Stage LOGIC's P3 found teacher-forced accuracy flat in position (1 of 80
runs falls, median last - first +0.0000). If `e_j` is flat then `R` is a ratio of
means, and the falling shape of `h_j` that candidate (ii) rests on is irrelevant
to `R` by construction, not by measurement.

*Prediction:* the coefficient of variation of `e_j` over `j` is **< 0.25** at a
majority of densities with `n >= 12`, **and** the Jensen term
`sum_j -log(1 - h_j)` minus `n * (-log(1 - h_bar))` is **< 0.10** of
`sum_j -log(1 - h_j)`. *Falsified if* the CV exceeds 0.5 at a majority. Note the
Jensen term can only make `-log S` LARGER, i.e. push `R` up, so it cannot be the
explanation either way; the prediction bounds its size.

**O3 — THE GAP GROWS WITH `j` AS THE SURVIVING POPULATION GETS EASIER.**
Registered as a **confirmation**, not a discovery: POS's P1 and LOGIC's P3
together nearly force it, and the content here is that it holds on a matched
pair of logically-scored instruments.

*Prediction:* the median Spearman correlation of `r_j` with `j`, over `j` with
`n_at_risk >= 30`, is **< -0.3** at a majority of densities with `n >= 12`.
*Falsified if* it is positive at a majority.

**O4 — SURVIVORSHIP IS LARGE ENOUGH FOR O3 TO BE ABOUT SELECTION.** If almost
nobody is censored, "the survivors are easier" has no room to be the mechanism.

*Prediction:* at rho = 0.50, the median of `n_at_risk` at the largest `j` with
data, divided by `n_instances`, is **< 0.40**. *Falsified if* it exceeds 0.80.

**O5 — THE CENSORING IS SMALL, AND IT PUSHES `R` THE WRONG WAY TO EXPLAIN
ITSELF.** The denominator is "no error of any kind so far", so a benign strict
error (a forced cell that is not the reference's next cell) removes a rollout
that had not actually failed. That SHRINKS the denominator and therefore RAISES
`h_j`, moving `R` toward 1 rather than away from it.

*Prediction:* pooled over the six densities, benign first errors `n_err_in_F` are
**< 5%** of all first errors `n_err`, and recomputing the at-risk path by
censoring only logical errors **raises** the median `r_j` at a majority of
densities. *Falsified if* benign errors exceed 20% of first errors.

**D1 — THE TWO TABLES ARE THE SAME POPULATION, POSITION BY POSITION.**
Teacher forcing scores every instance at every node position it has, so

$$n_{\text{total}}(j) = n_{\text{instances}} \ \text{for } j < n,
\qquad \sum_j n_{\text{total}}(j) = n_{\text{total}}(\text{scheme}=\text{all},\ \text{node}).$$

*Expected:* the sum identity holds **exactly** (difference 0) for every (run,
rho) cell, and the count of `(run_id, rho, j)` keys present in one table but not
the other is reported. A failure **invalidates O1-O5**, every one of which
compares the two tables at a matched `j`.

## Figures

| | |
| --- | --- |
| `O1_anchor.png` | `r_0` against rho, one point per run, median overlaid, with the `r = 1` line |
| `O2_profiles.png` | `e_j` and `h_j` against `j` at rho 0.50, median over runs |
| `O3_ratio.png` | `r_j` against `j` at three densities, median over runs |
| `O4_atrisk.png` | `n_at_risk / n_instances` against `j`, three densities |
| `PH_budget.png` | where `R` comes from: cumulative `sum_j h_j` against `sum_j e_j` |

## Output

`runs/stageregime_2026-11-17/`: `regime_positions.csv` (per run, rho, j: `e_j`,
`h_j`, `r_j`, `n_at_risk`, counts), `regime_summary.csv` (per rho: medians,
spreads, the Spearman, the Jensen term), `VERDICT.txt`, `figures/`.

Runs on a **login node in under three minutes** — CSV reading, medians, one
Spearman per density and matplotlib. **Nothing is submitted.**
