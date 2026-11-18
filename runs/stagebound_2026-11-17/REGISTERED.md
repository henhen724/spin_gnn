# Stage BOUND — registered before any of it was computed

Written and committed **before** `scripts/stagebound_fit.py` existed. No number
below was chosen after seeing it. Closes the open item stage GATE (99) left
behind, and contradicts what `STATUS.md` says about that item.

## The question

Stage GATE recomputed every architecture's `rho_50` under a **DD-faithful**
criterion — a rollout counts as solved only if it never named a cell that direct
deduction had not already forced — and reported the result as an **upper bound**,
because the estimator

$$h_j^{\text{ddf}} = 1 - \frac{n_{\text{in\_F}}(j)}{n_{\text{at\_risk}}(j)}$$

asks only *was the named cell forced*. It is blind to three other ways a rollout
fails: a **wrong colour on a cell that was forced**, a **grammar** violation, and
**early termination or truncation**. `STATUS.md` records folding those in as
needing "the rollouts RE-SCORED — a job, with a new registration".

**That is wrong, and this stage is the demonstration.** Re-reading
`scripts/stagelogic_score.py` shows `walk_forced` appends the pair at the index
where a rollout dies *before* breaking, so the already-written column
`n_err_in_F(j)` counts exactly the first errors at `j` whose named cell **was**
forced — the wrong-colour-on-a-forced-cell class. The remaining class,
non-node-bearing deaths, is not resolved per `j`, but its **total per (run, rho)
is on disk exactly** as `n_first_err - n_err_node`.

So the joint criterion is computable from the same CSVs: **no training, no GPU,
no new instances, nothing submitted.**

## The estimator, derived rather than asserted

Fix a run, a density and an index `j`. Two sets of rollouts fail the joint
criterion at `j`:

- `A(j)` — emitted a node token naming a cell **not** in the forced set `F`.
  `|A(j)| = n_at_risk(j) - n_in_F(j)`.
- `B(j)` — committed their **first permissive error** at `j`.
  `|B(j)| = n_err(j)`.

Partition `B(j)` by whether the error is node-bearing. A node-bearing error on a
cell **not** in `F` is already in `A(j)`. A node-bearing error on a cell **in**
`F` is not in `A(j)` and is counted by `n_err_in_F(j)`. A non-node-bearing death
is not in `A(j)` either, because `walk_forced` breaks before recording the pair,
so it never enters `n_at_risk(j)`; write its count `m(j)`. Hence exactly

$$|A(j) \cup B(j)| = \underbrace{\bigl(n_{\text{at\_risk}}(j) - n_{\text{in\_F}}(j)\bigr)}_{\text{unforced cell named}} + \underbrace{n_{\text{err\_in\_F}}(j)}_{\text{wrong colour, forced cell}} + \underbrace{m(j)}_{\text{grammar / early stop}}$$

with no double counting and nothing omitted. The first two terms are exact per
`j`. Only the **distribution** of the third is missing; its total over `j` is
exact, `M = n_first_err - n_err_node` from `LF_summary_all_<i>of10.csv`.

Define, over every `j` with `n_at_risk > 0`,

$$h_j^{\text{joint}} = \frac{\bigl(n_{\text{at\_risk}}(j) - n_{\text{in\_F}}(j)\bigr) + n_{\text{err\_in\_F}}(j)}{n_{\text{at\_risk}}(j)}, \qquad \widehat{S}_{\text{joint}} = \prod_j \bigl(1 - h_j^{\text{joint}}\bigr)$$

and the **non-node mass** \(\varphi = M / n_{\text{instances}}\), the fraction of
rollouts whose first failure this product cannot see. The reported bracket on the
true joint solve rate is

$$\max\bigl(0,\ \widehat{S}_{\text{joint}} - \varphi\bigr) \ \le\ S_{\text{joint}} \ \le\ \widehat{S}_{\text{joint}}$$

— the upper end because dropping `m(j)` can only spare rollouts, the lower end
because at most `M` of the `n_instances` rollouts die that way and each removes
at most one survivor. **`rho_50` is reported for both ends of the bracket**, and
an architecture is called a real pass only if the **lower** end clears 0.740.

`S_ddf` and the permissive `S_perm = prod(1 - n_err/n_at_risk)` are recomputed
here with stage GATE's loaders so the three criteria are read off one table.
`rho_50` is read exactly as in GATE: scan the 14-point grid downward from high
rho, linearly interpolate across the first bracket crossing 0.50, report a run
whose curve never crosses as **censored, never dropped silently**. Runs are keyed
on the **`cell`** label, never on `(arch_mode, num_layers, d_model)` — the defect
GATE found in its own code and had to disclose.

## What was inspected before registering, and what was not

- **Column names only** of `F_hazard` (23) and `LF_summary` (34).
- The **source** of `walk_forced` and `measure_F` in `scripts/stagelogic_score.py`
  — which is how the estimator above was derived, and is the point of the stage.
- **Carried over from stage GATE, and it is not nothing:** GATE computed the
  mis-registered `h^{regd} = (n_err - n_err_in_F)/n_at_risk` and its products, so
  **a function of `n_err_in_F` aggregated over `j` has been seen**, and B1 below
  should be discounted for that. I have not seen `n_err_in_F` itself at any `j`,
  nor `n_err_node`, nor `varphi`, nor any joint product.
- **Not inspected:** every quantity B1–B5 and D3–D4 are about.

## Predictions

**B1 — FAITHFULNESS, NOT COLOUR, IS WHAT THE STRICT CRITERION COSTS.** GATE
showed requiring faithfulness more than squares the solve rate
(`log S_ddf / log S` 2.67–3.57). If the model's errors are overwhelmingly
premature guesses on unforced cells, those are *already* counted by `h^ddf`, and
adding wrong colours on forced cells should barely move it.

*Prediction:* the median of `log S_joint / log S_ddf` over runs lies in
**[1.00, 1.50]** at a majority of the six densities with `n >= 12`. *Falsified
if* it exceeds **2.5** at a majority — which would mean deduction errors on cells
the model had correctly identified as forced are a first-order failure mode, and
GATE's bound was loose in a way that matters.

**B2 — THE NON-NODE MASS IS SMALL, WHICH IS WHY NO JOB IS NEEDED.** 97 measured
the `early_term`, `truncated` and `grammar` first-error classes at solve rate 0,
but these are terminal-token mistakes and the models are fluent.

*Prediction:* the median over cells with `n >= 12` of `varphi` is **< 0.02** and
its 90th percentile is **< 0.10**. *Falsified if* the median exceeds **0.05**, in
which case the bracket is too wide to decide anything and STATUS's "needs a job"
claim stands after all. **This is the prediction the stage lives or dies by** —
B3 and B4 are only interesting if the bracket is tight.

**B3 — J2 STILL CLEARS, NOW WITHOUT THE UPPER-BOUND CAVEAT.** The standing
minimum `L=16 h=4 no_mlp d192` went 0.5955 -> 0.6739 under GATE's DD-faithful
criterion, keeping 0.066 of headroom under the 0.740 gate.

*Prediction:* its joint `rho_50` at the **lower** end of the bracket is
**<= 0.740**. *Falsified if* it exceeds 0.740 — in which case the project's
standing minimum does not survive a criterion that actually requires direct
deduction, and every rung table needs re-reading.

**B4 — THE VERDICT TABLE IS UNCHANGED, READ PER ARCHITECTURE.** GATE found 6 of
31 architectures clear the gate as published and the **same 6** clear
DD-faithfully.

*Prediction:* the set clearing at the lower end of the joint bracket is a
**subset of those 6** and has **at least 4** members. *Falsified if* it is empty,
or if it contains an architecture outside the 6 (which would be a monotonicity
bug, not a finding).

**B5 — THE RANKING IS UNTOUCHED.** The joint criterion adds a small hazard to
every index of every run.

*Prediction:* Spearman of joint `rho_50` with DD-faithful `rho_50`, over runs
where both are defined, is **>= 0.95**. *Falsified if* < 0.8.

**D3 — BY CONSTRUCTION, AND THE VIOLATIONS LOCATE THE MISSING MASS.** Adding a
non-negative term to the numerator gives `S_joint <= S_ddf` on **every** cell,
with equality iff `n_err_in_F = 0` everywhere. Against the permissive product the
relation holds only up to the dropped `m(j)`: `S_joint <= S_perm` wherever
`varphi = 0`.

*Expected:* `S_joint <= S_ddf` on **1106 of 1106** cells — any violation is a
scorer bug, not a finding — and every cell with `S_joint > S_perm` has
`varphi > 0`, with the excess bounded by `varphi`.

**D4 — THE GATE-ERA CLOSURE STILL HOLDS.** Recomputing `S_perm` here must
reproduce GATE's D1: median `|S_perm - S|` **< 0.01**, max **< 0.05**, over the
790 cells with `n >= 12`. A failure means the loaders drifted and invalidates
B1–B5.

## Figures

| | |
| --- | --- |
| `B1_cost.png` | `log S_joint / log S_ddf` against rho, median over runs with IQR |
| `B2_phi.png` | distribution of the non-node mass `varphi` by rho |
| `B3_curves.png` | the three solve-rate curves for J2, with the 0.50 crossings and the bracket |
| `B4_dumbbell.png` | per architecture: published, DD-faithful and joint `rho_50`, against parameters |
| `D3_monotone.png` | `S_joint` against `S_ddf`, identity line, cells with `varphi > 0` marked |

## Output

`runs/stagebound_2026-11-17/`: `bound_cells.csv` (per run, rho: the three
products, the measured rate, `varphi`), `bound_runs.csv` (per run: three
`rho_50`s plus the bracket ends, censoring flags, params, `cell`), `VERDICT.txt`,
`figures/`.

Runs on a **login node in seconds** — CSV reading, products, medians, Spearman
and matplotlib. **Nothing is submitted.**
