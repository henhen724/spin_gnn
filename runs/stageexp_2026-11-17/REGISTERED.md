# Stage EXPONENT — registered before any of it was computed

Written and committed **before** `scripts/stageexp_fit.py` existed. No number
below was chosen after seeing it. Serves STATUS next-step 1, the one question
stages POS / LOGIC / PREFACTOR leave open that needs **no training**.

## The question

Finding 95 closed the prefactor: the solve-rate law is `S = p_logic^k` with no
prefactor, once `k` is fitted where the extrapolation to `p = 1` is short. What
it left open is **`k` itself**. Fitted that way, `k_top/n` runs **1.11–1.26** at
`n ≥ 17` — *above* `n`. The reasoning section holds

```
P = 2n + T + 1          n NODE tokens, n COLOR tokens, T BEGIN_ROUND, 1 terminal
```

positions, so `k` sits strictly between `n` and `P ≈ 2.3n`. Three accounts:

- **(A) colour positions are independently fatal** — the model picks the right
  cell and the wrong colour for it, so each blank cell carries *two* fatal
  choices and `k` should approach `2n`;
- **(B) round markers or the terminal are fatal** — `k ≈ n + T + 1`;
- **(C) `k` is not a position count at all** — it is the ratio
  `(−log S)/(−log p_logic)`, and `p_logic` averages over **all** `P` positions
  including ones whose errors never kill a rollout. Non-fatal error mass in the
  denominator inflates `k` above the number of fatal positions.

(C) is the account this stage is built to separate from (A) and (B), because it
needs no new fatal position and it is the one that would make the law read
`S = p_node^n` — one fatal choice per blank cell, which is as analytically
simple as this law can get.

## What was inspected before registering, and what was not

Honest disclosure, since this is a refit of CSVs already on disk and I had to
know the schemas to write predictions over them:

- **Column names and row counts** of `L_position_all_*`, `F_errors_all_*`,
  `LF_summary_all_*` (stage LOGIC) and `A_position_*`, `B_hazard_*`,
  `B_instances_*` (stage POS).
- **Three marginal distributions**, no conditionals: `first_kind` over all
  448,000 rollouts (`color_wrong` 276,666 · `none` 170,630 · `node_invalid` 413
  · `early_term` 238 · `truncated` 41 · `grammar` 12), `solved` (170,605 of
  448,000) and `well_formed` (445,689). The `tok_type` vocabulary
  (`node`/`color`/`round`/`term`/`any`) and the `scheme` vocabulary
  (`j`/`round`/`all`).
- **One data row**, the first line of `LF_summary_all_1of10.csv` — run `r01` at
  ρ = 0.95, `n = 3`: `p_logic_node` 0.9700, `p_logic_color` 1.0000,
  `solve_rate` 0.9525. It gives `(−log S)/(n·e_node) ≈ 0.54`. **That is a
  degenerate point** (`n = 3`, one run, the easiest density) and *every* test
  below is restricted to `n ≥ 12`, per the standing rule that a magnitude
  checked at a degenerate point is not checked (76, 79). It is disclosed rather
  than hidden, and it points *against* X1.
- **Not inspected:** any per-class error rate at any fit density, any
  conditional solve rate, any join between the two tables, any fit.

X6 and D1 are **strongly indicated in advance** by finding 94's L4 (98.270%
of first errors name an unforced node) and are registered as confirmations on a
different table and a different scorer, not as discoveries. X1–X3 are open.

## Inputs, population and conventions

Every input is a CSV already on disk. **No training, no GPU, no new instances,
nothing submitted.**

- `runs/stagelogic_2026-11-17/LF_summary_all_<i>of10.csv` — per (run, ρ):
  `p_logic`, `p_logic_node`, `p_logic_color`, the strict twins, `solve_rate`,
  `mean_n_blank`, `dd_mean_T`, `n_instances`, `in79`.
- `runs/stagelogic_2026-11-17/L_position_all_<i>of10.csv`, rows with
  `scheme = all` and `index = -1` — per (run, ρ, `tok_type`): `n_total`,
  `n_logic_correct`, giving the `round` and `term` classes that `LF_summary`
  does not carry.
- `runs/stagelogic_2026-11-17/F_errors_all_<i>of10.csv` — per failed rollout:
  `first_kind`, `err_in_F`, `err_in_F_now`, `err_uncolored`,
  `err_color_correct`.
- `runs/stagepos_2026-11-17/B_instances_{seven,rest_<i>of10}.csv` — per rollout,
  **free generation**: `first_kind`, `first_error_j`, `solved`, `well_formed`.

Conventions. `e_c = 1 − p_logic_c` is the **logical** (not strict) per-position
error rate of class `c`, so finding 94's mis-scoring is already removed and no
result here can be bought by it. `n = 48(1 − ρ_real)`; `P = 2n + T + 1` is read
from the data as `n_total(tok_type = any)/n_instances`, never assumed. The ten
densities ρ = 0.95 … 0.50 where a fit is defined, `in79` runs only, and because
`S` enters as `log S`, runs with `S = 0` or `S = 1` are dropped — **the count
dropped is reported at every density**. Seeds are kept separate and the spread
is quoted. Group values are medians over runs. The join key between the two
tables is `(run_id, rho, inst)`; an unmatched row is **counted and reported**,
never dropped silently.

## Predictions

**X1 — the fatal mass is the node class, and `k = n`.** If node selection is the
one fatal choice per blank cell then `−log S = n · e_node` per run.

*Prediction:* the median of `(−log S)/(n · e_node)` lies in **[0.8, 1.3]** at a
majority of the ten densities with `n ≥ 12`. *Falsified if* the median is
outside [0.5, 2.0] at a majority. A ratio **below** 1 says node errors are
themselves partly survivable and the exponent is not a count of fatal positions
at all; **above** 1 says a second class is fatal too.

**X2 — the colour class carries real error mass but little fatal mass.** The
dilution mechanism of account (C), stated as two measurements that must both
hold.

*Prediction:* `(n · e_color)/(n · e_node) > 0.20` at a majority of densities with
`n ≥ 12` — colour errors are **not** negligible in the teacher-forced count —
**while** the share of fatal first errors whose named node is in `F_now` is
**< 0.05** pooled over those densities. *Falsified if* the first ratio is below
0.05 (nothing to dilute with) or the second exceeds 0.20 (colour positions are
genuinely fatal, i.e. account (A)).

**X3 — a zero-parameter node-only law fits as well as a free exponent.** OLS of
`−log S` on `n · e_node` through the origin, per density, over the same runs.

*Prediction:* the slope `a ∈ [0.8, 1.3]` at a majority of densities with
`n ≥ 12`, and `R² > 0.80` at a majority. *Falsified if* `a` is outside
[0.5, 2.0] at a majority, or `R² < 0.50` at a majority. Registered alongside X1
so that one robust statistic (the median) and one fit have to agree; **if they
disagree the fit loses**, because finding 95 is the standing lesson that a
number read off a fitted line can be a property of the fit.

**X4 — round markers and the terminal are not where the exponent lives.**
Account (B).

*Prediction:* `(T · e_round + e_term)/(n · e_node) < 0.10` at a majority of
densities with `n ≥ 12`. *Falsified if* it exceeds 0.25 at a majority.

**X5 — every free-generation first error is fatal, whatever its class.**
Registered so that X1's interpretation cannot be smuggled past the possibility
that some classes are survivable.

*Prediction:* pooled over the ten densities, `P(solved | first_kind =
color_wrong) < 0.02` and `P(solved | first_kind = node_invalid) < 0.10`.
*Falsified if* either exceeds 0.10 / 0.30 respectively. `node_invalid` has only
413 rollouts in the whole table, so its interval is reported and it is read as
suggestive at best.

**X6 — the 99.7% `color_wrong` label is a guard failure, not a colour failure.**
The interpretive link between the two tables, and the reason X1 and X2 are not
in tension with a table in which almost every failure is labelled a colour
error. Joins `B_instances` (free generation, labels the pair by its colour) to
`F_errors_all` (same rollouts, computes the forced set `F`).

*Prediction:* among first errors with `first_kind = color_wrong`, the share
whose named node is **not** in `F_now` exceeds **0.90**. *Falsified if* below
0.50 — that would mean the model is choosing forced cells and mis-colouring
them, which is account (A) and would refute X1 and X2 together.

**D1 — the class decomposition is an identity, not a model.** With `n_c` the
position count of class `c`,

$$1 - p_{\text{logic}} \;=\; \frac{\sum_c n_c\, e_c}{P},\qquad P=\sum_c n_c = 2n+T+1 .$$

*Expected:* the two sides agree to **< 1e-6** for every (run, ρ) cell, `P` read
from the data matches `2·mean_n_blank + dd_mean_T + 1` to < 0.1, and
`p_logic_node` / `p_logic_color` from `LF_summary` match the same quantities
rebuilt from `L_position` to < 1e-9. A failure of any part **invalidates
X1–X4**, because every one of them divides by a class error rate.

## Figures

| | |
| --- | --- |
| `X1_ratio_by_density.png` | `(−log S)/(n·e_node)` against ρ, per run, median overlaid |
| `X2_class_mass.png` | per-position error mass by class against ρ, stacked |
| `X3_fit.png` | `−log S` against `n·e_node` at the three hardest densities, with the `y = x` line |
| `X5_fatality.png` | conditional solve rate by `first_kind` |
| `X6_forced.png` | share of `color_wrong` first errors at unforced nodes, against ρ |

## Output

`runs/stageexp_2026-11-17/`: `exp_cells.csv` (per run, ρ: class error rates,
masses, `P`, `−log S`, the ratio), `exp_summary.csv` (per ρ: medians, spreads,
the fit), `VERDICT.txt`, `figures/`.

Runs on a **login node in under three minutes** — CSV reading, medians, one OLS
per density and matplotlib. No GPU, no model, no instance generation.
**Nothing is submitted.**
