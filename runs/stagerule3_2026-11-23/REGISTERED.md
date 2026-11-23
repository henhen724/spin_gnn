# STAGE RULE3 — registered predictions, written BEFORE the scorer was run

Stage RULE2 ended by saying it **could not** tell whether DD's rule degrades
below ρ_c, because scoring stops at the first error and the deep bins are a
selected sample. That was a statement about the instrument, not a measurement of
it. **This stage measures the censoring itself** and asks how much of it would
be needed to explain RULE2's fall.

A **third refit of the same shards** (`runs/stagelogic_2026-11-17/
F_hazard_all_*of10.csv`). Seconds on a login node. **NOTHING IS SUBMITTED, NO
MODEL IS TRAINED, NO ROLLOUT IS REGENERATED.** Same 79 runs, same 31
architectures, same 400 instances per bin, same eval seed start 60,000,000,000.

## What the instrument actually counts, corrected

Reading the shard rows directly fixed a loose reading carried in earlier decks.
Per `(run_id, rho, j)`:

| column | meaning |
|---|---|
| `n_at_risk(j)` | rollouts with no error before index $j$ |
| `n_in_F(j)` | emissions at $j$ that named a **forced** cell |
| `n_err(j)` | first errors at $j$ — emissions of a **wrong colour**, forced cell or not |
| `n_err_in_F(j)` | first errors at $j$ that named a forced cell |
| `n_F_sum(j)` | $\sum$ over at-risk rollouts of $|F|$, the forced cells *available* |

So `n_err − n_err_in_F` is **a wrong colour on an unforced cell**: the model
guessed where DD would have waited, and guessed wrong. That is the dominant
censoring channel. Naming an unforced cell and getting the colour *right* is a
stopping-condition failure but **not** an error and does **not** censor.

## The one knob

Nothing about `h_rule` changes. The stage adds one quantity: the forced
emissions that censoring **removed**, and a sensitivity parameter on what they
would have carried.

Per `(run, rho)`, with survivors' mean remaining forced emissions from $j+1$,

$$ R(j) \;=\; \frac{\sum_{j' > j} n_{F}(j')}{n_{\text{at risk}}(j+1)},
   \qquad
   C(j) \;=\; n_{\text{err}}(j) - n_{\text{err},F}(j), $$

the **lost denominator** and the **censored count** are

$$ L \;=\; \sum_j C(j)\,R(j), \qquad C_{\text{tot}} \;=\; \sum_j C(j). $$

Let $\lambda$ be the ratio of the censored rollouts' counterfactual rule-error
rate to the survivors' observed rate. Then

$$ h(\lambda) \;=\; \frac{E_{\text{obs}} + \lambda\,h_{\text{obs}}\,L}
                          {F_{\text{obs}} + L},
   \qquad h(1) = h_{\text{obs}} . $$

$\lambda = 1$ is ignorable censoring and **must** reproduce RULE2 exactly
(diagnostic D3). $\lambda > 1$ means the rollouts that died by guessing would
also have been worse at the rule — which is the frailty RULE2 registered as the
bias direction.

**The tipping point** is the $\lambda$ at which the low-density bin stops
looking better than ρ = 0.50:

$$ \lambda^{*}(\rho) \;=\;
   \frac{h_{0.50}\,\bigl(F_{\text{obs}} + L\bigr) - E_{\text{obs}}}
        {h_{\text{obs}}\,L} . $$

## Why a tipping point rather than a worst-case bound

The all-or-nothing bound — assume **every** censored rollout would have broken
the rule — is expected to be vacuous here, because $C_{\text{tot}}$ is of the
same order as the whole denominator. A bound that permits $h^{\mathrm{rule}}$
above the 0.01 bar says nothing, and would be reported as saying nothing.
$\lambda^{*}$ is the informative version: **how much worse would the censored
rollouts have had to be** for the fall to be an artefact rather than a fact.

## HYPOTHESIS, stated before the numbers

**Censoring at ρ = 0.33 is severe enough that a small $\lambda$ overturns the
fall, so RULE2's "unanswerable" verdict stands and is now quantified.** The
censored channel is wrong guesses on unforced cells, which is exactly the
behaviour of a model that has the rule but not the stopping condition — the
population this ladder is full of.

## Predictions

| | prediction | MISS condition |
|---|---|---|
| **E1** | fewer than **10%** of rollouts reach the last scored index uncensored at ρ = 0.33, pooled over the 79 runs | 10% or more |
| **E2** | the lost denominator exceeds the observed one at ρ = 0.33: $L / F_{\text{obs}} > 1$ | $\le 1$ |
| **E3** | the all-or-nothing worst case is **vacuous** at ρ = 0.33: $h(\lambda_{\max}) > 0.01$, above the rule bar | it lands below the bar |
| **E4** | the tipping point is small: $\lambda^{*}(0.33) < 3$ | $\ge 3$ |
| **E5** | censoring is mild in the fast phase: $L / F_{\text{obs}} < 0.25$ at ρ = 0.95 | $\ge 0.25$ |

**E4 carries the stage.** If $\lambda^{*}$ is small, RULE2's fall is explained by
censoring at a frailty the data cannot rule out, and "unanswerable" is the
correct and now-quantified verdict. If $\lambda^{*}$ is large, censoring
**cannot** explain the fall, and the fall becomes a real effect that needs its
own explanation — which would amend RULE2's reading rather than confirm it.

**E5 is the control.** If censoring were severe at every density, the method
would be measuring the instrument everywhere and could not localise the problem
below ρ_c.

## Diagnostics (if any fails, nothing above is reported)

- **D1 — the censoring identity.** $n_{\text{at risk}}(j+1) = n_{\text{at
  risk}}(j) - n_{\text{err}}(j)$ on **every** `(run, rho)` sequence in the in79
  population. This is the whole basis of the stage: if it fails, the reading of
  `n_err` as "the event that censors" is wrong and nothing here is computed
  correctly. **Zero violations permitted.**
- **D2 — reproduce finding 100.** Total `n_err_in_F` over all 14 densities =
  **1,988**, affected cells **63 of 1,106**, as in RULE and RULE2.
- **D3 — $\lambda = 1$ reproduces stage RULE2.** $h(1)$ per density must equal
  RULE2's pooled `h_rule` to within $10^{-12}$ at all fourteen densities. This
  is what makes $\lambda$ interpretable as a departure from ignorability.
- **D4 — no negative mass.** $R(j) \ge 0$, $C(j) \ge 0$ and $L \ge 0$ on every
  cell; $n_{\text{in}F}(j) \le n_{\text{at risk}}(j)$ as before.

## Traps carried forward

1. **KEY ON `cell`, NEVER ON `(arch_mode, num_layers, d_model)`** — 79 runs →
   **31** `cell` labels (the 96/99 defect).
2. **NO SINGLE-SEED RUNG IS A MEASUREMENT OF ITS ARCHITECTURE** (41/42, 119/120).
3. **THE `h_rule` RANKING IS DENSITY-DEPENDENT AND MUST NEVER BE QUOTED** (122,
   E6, Spearman +0.8464). Nothing here ranks architectures.
4. **THIS IS THE `d5q4` POPULATION.** Blind guess $1 - 1/q = 0.75$. No `d4q3`
   constant enters. ρ_c ≈ 0.465 is this ensemble's own number.
5. **COUNT THE POPULATION BEFORE QUOTING A COUNT** (122): 15 of 31 measure
   exactly zero at ten densities, 12 of stage RULE's 25 **sub-gate**.
6. **A ZERO DENOMINATOR IS NOT A ZERO HAZARD.**
7. **CHECK FOR THE ARTEFACT, NOT THE EXIT CODE.** No `|| true` anywhere.
8. **$\lambda$ IS A SENSITIVITY PARAMETER, NOT A MEASUREMENT.** No value of
   $\lambda$ is "the" answer and $\lambda^{*}$ is never quoted as the frailty
   that occurred. It is the frailty that *would be required*, and the data
   cannot say which $\lambda$ is real — only a repair-and-continue eval can.

## What a null licenses

A report, and nothing else. **No training follows from any outcome**, and no
new eval is submitted: if E4 hits, the conclusion is that the repair-and-continue
eval is *necessary*, which is a request to hshunt and not an action. The two
seeding candidates and the polynomial mode stay where they are.
