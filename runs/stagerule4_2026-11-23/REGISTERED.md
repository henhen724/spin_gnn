# STAGE RULE4 — registered predictions, written BEFORE the scorer was run

Stage RULE2 found $h^{\mathrm{rule}}$ *falling* below $\rho_c$ and refused to
read it. Stage RULE3 measured why: at $\rho = 0.33$ only **0.7%** of rollouts
survive to the last scored index, and a frailty of $\lambda^{*} = 1.19$ erases
the fall entirely. Both stages ended with the same sentence — **whether DD's
rule degrades below $\rho_c$ is unanswerable from censored rollouts.**

There is one slice of the same data where that objection cannot apply.

## At deduction index $j = 0$ nothing has been censored yet

Censoring removes a rollout at its **first** error. At the first emission no
rollout has had a first error yet, so none has been removed. Verified before
this registration was written, on the design side only:

**`n_at_risk(0) = 400` on all 1,106 `(run, rho)` sequences, zero exceptions.**

So $h^{\mathrm{rule}}(0)$ compared across densities is **free of the
survivorship bias** that made the pooled comparison unreadable. Every rollout
that exists is in the denominator's reference class.

A **fourth refit of the same shards**, seconds on a login node. **NOTHING IS
SUBMITTED, NO MODEL IS TRAINED, NO ROLLOUT IS REGENERATED.** Same 79 runs,
31 architectures, 400 instances per bin, eval seed start 60,000,000,000.

## The one knob

Which indices the sum runs over. Everything else — the instrument, the bars,
the population, the densities — is unchanged.

$$ h^{\mathrm{rule}}(0) \;=\;
   \frac{\sum_{\text{runs}} n_{\text{err},F}(0)}
        {\sum_{\text{runs}} n_{F}(0)},
   \qquad\text{against}\qquad
   h^{\mathrm{rule}} \;=\;
   \frac{\sum_{\text{runs}}\sum_j n_{\text{err},F}(j)}
        {\sum_{\text{runs}}\sum_j n_{F}(j)} . $$

## The denominators, known in advance and part of the design

Pooled $n_F(0)$ per density, read before this file was written. The
**numerator was not read.**

| $\rho$ | 0.95 | 0.90 | 0.85 | 0.80 | 0.75 | 0.70 | 0.65 |
|---|---|---|---|---|---|---|---|
| $n_F(0)$ | 30,442 | 29,687 | 28,502 | 27,681 | 27,163 | 25,198 | 24,924 |

| $\rho$ | 0.60 | 0.55 | 0.50 | 0.45 | 0.40 | 0.35 | 0.33 |
|---|---|---|---|---|---|---|---|
| $n_F(0)$ | 24,225 | 24,201 | 23,607 | 23,659 | 23,530 | 24,055 | 23,990 |

Four added bins pool to **95,234**; the ten fitted to **265,630**. At
$h \approx 6\times10^{-4}$ that is roughly 57 and 160 expected events, so the
relative standard errors are about 13% and 8%. **The resolvable difference is
therefore a factor of about 1.3, not 1.05**, and E5 checks this rather than
assuming it.

## What $j = 0$ does NOT remove

**It removes survivorship. It does not make the comparison assumption-free.**
$h^{\mathrm{rule}}$ conditions on the model having *named a forced cell*, at
$j=0$ as at every index. The share of rollouts doing so differs by density —
$30{,}442 / 31{,}600 = 96.3\%$ at $\rho = 0.95$ against
$23{,}990 / 31{,}600 = 75.9\%$ at $\rho = 0.33$ — so the cells entering the
denominator are differently composed across densities. That is selection on the
**emission**, not on **survival**, and it is inherent to the definition of
$h^{\mathrm{rule}}$. It is reported, not corrected.

## HYPOTHESIS, stated before the numbers

**The rule degrades as hints are removed, and the pooled fall was entirely the
censoring artefact.** On the clean slice $h^{\mathrm{rule}}(0)$ should *rise* as
$\rho$ falls, because a forced cell at low $\rho$ sits in a sparser
neighbourhood and there are more candidate cells to confuse — even though
being forced still requires three distinct colours among the neighbours.

## Predictions

| | prediction | MISS condition |
|---|---|---|
| **E1** | the clean slice does **not** reproduce the pooled fall: the ratio $h^{\mathrm{rule}}(0)$[four added bins] $/$ $h^{\mathrm{rule}}(0)$[$\rho=0.50$] has a 95% interval reaching 1.0 or above | the interval lies entirely below 1.0 |
| **E2** | the rule genuinely degrades: $h^{\mathrm{rule}}(0)$[four added] $/$ $h^{\mathrm{rule}}(0)$[$\rho \ge 0.80$] $\ge 1.3$, with the interval excluding 1.0 | below 1.3, or the interval includes 1.0 |
| **E3** | the trend is monotone in the right direction: Spearman$(\rho,\ h^{\mathrm{rule}}(0))$ over the fourteen densities is negative | zero or positive |
| **E4** | the rule survives the clean slice everywhere: $h^{\mathrm{rule}}(0) < 0.01$ at all fourteen densities | any density at or above the bar |
| **E5** | the comparison is powered for its own claim: the 95% interval on the E2 ratio is narrower than a factor of 2 | wider, in which case **E2 is reported as underpowered and its verdict is not read as evidence either way** |

**E1 and E2 are different claims and are scored separately.** E1 hitting while
E2 misses is the outcome "the fall was an artefact, but we cannot show the rule
degrades either" — a null that is *informative* only if E5 hits.

## Diagnostics (if any fails, nothing above is reported)

- **D1 — zero censoring at $j=0$.** $n_{\text{at risk}}(0) = 400$ on every one
  of the 1,106 sequences. This is the entire basis of the stage.
- **D2 — reproduce finding 100.** Total $n_{\text{err},F}$ over all indices and
  all fourteen densities = **1,988**.
- **D3 — agree with stages RULE2 and RULE3.** Summing over all $j$ must
  reproduce their pooled per-density $h^{\mathrm{rule}}$ to $10^{-12}$.
- **D4 — no impossible counts.** $0 \le n_F(0) \le 400$ and
  $0 \le n_{\text{err},F}(0) \le n_F(0)$ on every sequence.

## Traps carried forward

1. **KEY ON `cell`, NEVER ON `(arch_mode, num_layers, d_model)`** (96/99).
2. **NO SINGLE-SEED RUNG IS A MEASUREMENT** (41/42, 119/120). Nothing here is
   per-architecture; everything is pooled over all 79 runs.
3. **THE `h_rule` RANKING IS DENSITY-DEPENDENT AND MUST NEVER BE QUOTED**
   (122, Spearman +0.8464). This stage ranks nothing.
4. **`d5q4` POPULATION.** Blind guess $1 - 1/q = 0.75$; $\rho_c \approx 0.465$
   is this ensemble's own number. No `d4q3` constant enters.
5. **A ZERO DENOMINATOR IS NOT A ZERO HAZARD**, and a handful of events is not
   a rate — E5 exists because this stage's numerators are tens, not thousands.
6. **CHECK THE SPECIFICATION BEFORE BELIEVING A DIAGNOSTIC FAILURE** (123: 308
   of 317 D1 violations were my own omission of the completion channel).
7. **$j=0$ REMOVES CENSORING, NOT ALL SELECTION.** Never describe this slice as
   unbiased, only as censoring-free.

## What a null licenses

A report. **No training follows from any outcome**, no new eval is submitted,
and no closed knob is reopened. If E1 hits and E2 misses under a passing E5,
the honest conclusion is that the pooled fall was an artefact and the clean
slice is flat — which would make the repair-and-continue eval *less* urgent,
not more, and that is still hshunt's call to make.
