# Stage CAND — does the model compute the candidate set, or something that agrees with it?

Written and committed **before** `scripts/stagecand_fit.py` existed. No number below
was chosen after seeing it.

## The question, which is STATUS next-step 1 part (iii)

98 established that the model has **DD's rule and not DD's stopping condition**: it
errs on a genuinely forced cell only 0.11–0.32% of the time, but 3.7–25.3% of its
emissions name a cell DD has not yet forced. 100 and 101 sharpened the measurement
of that gap without touching its interpretation.

`STATUS.md` records the open part as: *naming only forced cells is NECESSARY for DD
and not SUFFICIENT — the models' output is consistent with having deduced, and
nothing on disk distinguishes that from computing something else that agrees. That
needs a new instrument or hshunt, not another refit.*

**That claim is a hypothesis, not a finding (maxim 100), and this file has made it
twice and been wrong twice** — 98 and 100 were both answered by columns already
written. So it is tested before it is believed.

## The instrument, and why `j = 0` is the only place it works

`stagelogic_score.py:305 walk_forced` records, for the **first error of every
rollout**, the node the model named (`err_node`), the round (`err_round`), and the
cache index of the instance (`inst`). `instance_cache` is deterministic and
byte-identical across runs, so **for any recorded first error the exact graph,
hints and planted colouring can be rebuilt from disk.**

At `j = 0` — and only at `j = 0` — the model's pre-state is **exactly the hints**,
because nothing has been emitted yet. Everywhere else the pre-state depends on
which cells this rollout emitted, which is not recorded. `j = 0` is already this
project's clean anchor: 98 showed free generation and teacher forcing are provably
on the same prefix there, with `n_at_risk == n_instances` on 1120/1120 cells.

So define, for an uncoloured cell `v` under the hints,

$$c(v) \;=\; q \;-\; \bigl|\{\, \text{colours on } v\text{'s assigned neighbours} \,\}\bigr|$$

the number of colours **locally consistent** with the constraints, with `q = 4`.
This is `forced_mask`'s own arithmetic (`stagelogic_score.py:179`) read one step
earlier: that function returns `(asg < 0) & (cnt == q - 1)`, which is exactly
`c(v) = 1`. **Forced is the special case `c = 1`.** The planted colour is always one
of the `c(v)` candidates, because the planted colouring is proper.

## The two hypotheses this separates

**DEDUCTION.** The model propagates constraints: it computes, or represents
something equivalent to, the candidate set of a cell. Then when it misfires it
should misfire on **near-forced** cells — `c = 2`, one short of determined — and its
colour on an unforced cell should come from that cell's candidate set.

**A PROXY THAT AGREES.** The model has learned a surface rule that correlates with
forcedness without propagating anything. The strongest such rival on a 5-regular
graph is **coloured-neighbour count**: "name the uncoloured cell with the most
assigned neighbours". A cell with `k` coloured neighbours is more likely to be
forced as `k` grows, so this proxy reproduces the necessary condition most of the
time — but `k` and `c` are **not the same**: five coloured neighbours can show 1, 2,
3 or 4 distinct colours. Separating them is the whole point of C5.

## What is observable, and from which column

Per `(run, rho)` at `j = 0`, from `F_hazard` (four aggregate columns):

- `N = n_at_risk` (= 400), `n_in_F`, `n_err`, `n_err_in_F`
- unforced emissions `U = N - n_in_F`
- of those, wrong `E_u = n_err - n_err_in_F`
- **measured correct-rate on an unforced emission `M = 1 - E_u/U`**

Per erring rollout, from `F_errors` + the instance cache:

- `c(err_node)`, hence **`w(c)`, the candidate-count distribution of unforced cells
  the model named and got wrong**
- `k(err_node)`, the number of coloured neighbours, for C5

`M` and `w(c)` come from **different columns of different files**. Nothing links
them except a hypothesis about the model, which is what makes C2 a test rather
than a fit.

The matched null `b(c)` is **not** the global share. For every `(run, instance)`
pair that produced a `j = 0` error, take that instance's own within-instance share
of candidate count `c` among its uncoloured unforced cells, and average over those
pairs. This is the distribution a model choosing **blindly among unforced cells of
the same instances** would produce, so instance composition cannot confound it.
The unmatched all-instance version is reported alongside, never used to decide.

## What was inspected before registering, and what was not

- The `F_errors` header (28 columns) and the source of `walk_forced`,
  `forced_mask`, `instance_cache`.
- The `j = 0` row count over all 10 shards, all 14 densities, all runs: **45,888**,
  of which `first_kind` is `color_wrong` on 45,885 and `node_invalid` on 3.
- One pickle's structure (`rho = 0.60`: 400 instances, `adj` (48, 5) int32, `hints`
  a dict of 28 entries, `trace` keys `hints/resolved/rounds/unresolved_nodes`).
- **A mechanics probe, binary output only:** on the 307 `j = 0` rows at `rho = 0.60`
  in shard `1of10`, `(c == 1)` agreed with the scorer's own `err_in_F` on **307 of
  307**. No candidate-count distribution, no base rate, no correct-rate and no
  stratification was printed.
- **From `STATUS.md`, not newly computed:** `M` lies in **[0.4647, 0.5159]** across
  densities, against a blind `1/q = 0.25`; error on a genuinely forced cell
  0.0011–0.0032; unforced share 0.0366 → 0.2529 as rho falls. Which density carries
  the 0.5159 is not recorded there and is not known here.
- **Not inspected:** any `w(c)`, any `b(c)`, any `k` stratification, any `P̂`, any
  per-density `M`, anything C1–C5 are about.

## The honest disclosure that maxim 101 exists to force

**Model A below has a hard ceiling of `M = 0.50`**, because `1/c ≤ 1/2` for every
unforced cell. `STATUS.md` already records `M` reaching **0.5159**. So "is `M` at or
below 0.50" is **not a live prediction** — my own file has already half-answered it.
C3 is therefore written about the **count** of densities above the ceiling and the
**size** of the excess, neither of which is recorded anywhere, and C2 is written as
a quantitative-accounting test rather than a yes/no on Model A.

## Model A, stated so it can be refuted

*At `j = 0` the model picks a cell by an unknown rule, computes that cell's
candidate set correctly, and emits a colour uniform on it.* Then

$$\Pr(\text{wrong} \mid c) = 1 - \tfrac{1}{c}, \qquad
w(c) \;\propto\; \pi(c)\left(1 - \tfrac{1}{c}\right), \qquad
\pi(c) \;\propto\; w(c)\,\frac{c}{c-1}$$

where `π(c)` is the candidate-count distribution of the unforced cells it **names**
(wrong or right). Model A then predicts the correct-rate

$$\hat P \;=\; \sum_{c\ge 2} \frac{\pi(c)}{c}$$

from `F_errors` alone, to be compared against `M` from `F_hazard` alone.

## Predictions

**C1 — THE MODEL SELECTS NEAR-FORCED CELLS.** Pooled over the ten fit densities,
`w(2) > b(2)` by at least **+0.10** in share, and `w(2) > b(2)` at **≥ 8 of 10**
densities individually. *Falsified if* `w(2) ≤ b(2)` at ≥ 5 densities, or the pooled
gap is **< +0.02**.

This is deliberately the conservative direction: under blind cell choice with a
uniform-over-candidates colour rule, `w(c) ∝ b(c)(1 - 1/c)`, which **down**-weights
`c = 2` (factor 0.50 against 0.67 and 0.75). So a positive gap has to overcome a
null that is pushing the other way.

**C2 — THE OVER-IDENTIFYING TEST.** `|P̂ - M| ≤ 0.08` at **≥ 7 of 10** fit
densities, and the median absolute difference over the ten is **≤ 0.05**.
*Falsified if* the median absolute difference exceeds **0.12** — the candidate
structure would then fail to account for a correct-rate it is supposed to generate,
and Model A is the wrong description regardless of which direction it misses.

**C3 — WHERE THE CEILING BREAKS.** `M > 0.50` at **1 to 3** of the ten fit
densities, and wherever it does the excess is **≤ 0.03**. *Falsified if* `M > 0.50`
at **≥ 5** densities, or any excess exceeds **0.06**. Either would mean the model
systematically beats the local candidate set — it is using more than one step of
propagation, and "uniform over candidates" is not merely imprecise but the wrong
shape.

**C4 — THE SELECTION STRENGTHENS WITH DIFFICULTY.** Spearman of `w(2) - b(2)`
against rho over the ten fit densities is **≤ -0.60**. *Falsified if* **≥ +0.30**.
As rho falls, `b(2)` must fall — fewer coloured neighbours means more candidates —
so a model that keeps naming `c = 2` cells widens the gap, while one that degrades
into blind choice closes it.

**C5 — THE HEADLINE: THE SELECTION SURVIVES HOLDING COLOURED-NEIGHBOUR COUNT
FIXED.** Stratify named-and-wrong unforced cells by `k`, the number of coloured
neighbours, and recompute both shares **within** each stratum. In the modal stratum
— whichever `k` holds the most such cells, named after the counts are in and not
before — `w(2 \mid k) - b(2 \mid k) ≥ +0.05`. *Falsified if* that within-stratum
gap is **≤ 0**, in which case the apparent candidate-set sensitivity is a
coloured-neighbour-count effect, the proxy survives, and **C1 must not be reported
as evidence of propagation**.

C1–C4 are about the candidate set; C5 is the only one that separates it from the
nearest rival. If C1 hits and C5 misses, the finding is the rival, not the rule.

## Diagnostics

**D1 — NO RECONSTRUCTION DRIFT.** `(c(err_node) == 1)` must equal the scorer's own
`err_in_F` on **every** `j = 0` row at **all 14** densities, not the 307 probed. Any
mismatch means the instance index or the pre-state reconstruction is wrong and the
stage stops there.

**D2 — THE POPULATION LEDGER.** Per density: `N`, `n_at_risk` at `j = 0` (which must
equal `n_instances`, 98's anchor), the `j = 0` error count, how many are
`node_invalid` and excluded, how many have `err_in_F = 1`, and `U`, `E_u`, `M`.
Reported in full, nothing dropped silently.

## Scope

Decided on the **ten fit densities** so the numbers are comparable with 98's
published `M` range; all **14** are computed and reported, with 0.45/0.40/0.35/0.33
marked as extension and used to decide nothing. **`j = 0` has no `S = 0` / `S = 1`
censoring** — every rollout makes a first emission — so 101's censoring ledger does
not apply here, and that is stated rather than assumed. Population is the **79
in79 runs**, as everything since 92.

## Figures

| | |
| --- | --- |
| `C1_selection.png` | `w(c)` against `b(c)`, pooled over the ten fit densities |
| `C2_overid.png` | `P̂` and `M` against rho |
| `C3_ceiling.png` | `M` against rho with the 0.50 candidate ceiling and the 0.25 blind line |
| `C4_trend.png` | `w(2) - b(2)` against rho |
| `C5_stratified.png` | `w(2 \mid k) - b(2 \mid k)` by coloured-neighbour count `k` |

## Output

`runs/stagecand_2026-11-17/`: `cand_density.csv` (per rho: `w(c)`, `b(c)`, `P̂`, `M`,
the ledger), `cand_strata.csv` (per rho and `k`), `VERDICT.txt`, `figures/`.

Reads `F_errors_*.csv`, `F_hazard_*.csv` and the cached instance pickles only.
Runs on a **login node in seconds**. **Nothing is submitted, nothing is trained, no
new instances, no model is loaded.**
