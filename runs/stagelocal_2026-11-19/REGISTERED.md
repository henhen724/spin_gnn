# Stage LOCAL — how much of the adjacency join does the colour step actually do?

Written and committed **before** `scripts/stagelocal_fit.py` existed. No number
below was chosen after seeing it.

## The caveat this stage attacks is one I wrote myself

103's verdict closes with:

> Exactness of the colour step does not mean the model COMPUTES the candidate
> set and samples from it; the same record is produced by any mechanism that
> never emits a seen colour, including simple inhibition from the neighbour
> tokens in context. Distinguishing those needs the model's internals, not its
> output.

That is two different accounts wearing one sentence, and they are not equally
hard to test:

- **(B1) GLOBAL inhibition** — suppress colours seen anywhere in the visible
  context. Needs no adjacency at all.
- **(B2) ADJACENCY-LOCAL inhibition** — suppress the colours on *this named
  cell's own* assigned neighbours. That requires joining the `BEGIN_GRAPH`
  adjacency row for the named node against the `BEGIN_HINTS` colours, which is
  the same function as computing the candidate set.

**(B2) really is behaviourally identical to computing the candidate set and I
am not claiming this stage separates them.** But (B1), and every *partial*
version of (B2), make quantitatively different predictions, and the instance
cache settles them with no model and no job.

**STATUS now says this question "needs internals, not output". That is the
fourth time this file has called something a job. The first three were wrong.**

## The observable: properness under a PARTIAL join

Take the cells the model actually named at `j = 0` in rollouts whose colour was
wrong, restricted to **unforced** cells ($c \ge 2$) — exactly $Q_u$'s
denominator, exactly the population 103's $Q_{\text{null}}$ was matched on.

Let the named cell have $k$ assigned neighbours, $S$ the set of colours on all
of them, and $c = q - |S|$ its true candidate count. Consider an emitter that
joins against only a subset $A$ of those neighbours, sees $S_A \subseteq S$,
avoids the planted colour, and emits uniformly from what is left. Its chance of
being **proper** is $(c-1)/(q - |S_A| - 1)$, so define

$$\Pi(D) \;=\; \operatorname*{mean}_{\text{named cells}} \;\;
\operatorname*{mean}_{|A| = k - D} \; \frac{c-1}{q - |S_A| - 1}$$

$D$ is the number of assigned neighbours **dropped** from the join.

**This interpolates exactly between two numbers that are already published.**
At $D = 0$, $S_A = S$ and $\Pi = 1$ identically. At $D = k$, $S_A = \emptyset$
and $\Pi(k) = \operatorname{mean}(c-1)/3$, which is **$Q_{\text{null}} =
0.3379$ from 103, to the digit**. The ladder is the curve between them, and the
model sits at the $D = 0$ end with a measured $Q_u$ of exactly 1.0000.

## Why this is a real test and not arithmetic

$\Pi(D)$ is computed from the instances alone. The *model* contributes one
number: 103's count of improper draws, which is **0 out of 24,008** wrong
unforced emissions over the ten fit densities on the 77 non-carrier runs.

If the colour step dropped even one neighbour on a fraction $1-\pi$ of
emissions, expected violations are $(1-\pi)\,\bigl(1 - \Pi(1)\bigr) \times
24008$. Observing zero therefore bounds $\pi$ from below, and that bound is
the stage's headline.

## What was inspected before registering, and what was not

- `F_errors`' 29 columns, confirming there is **no emitted-colour column**, so
  nothing here can use the colour the model actually wrote — only whether it
  was proper, which exists at aggregate level only. Stated as a scope limit in
  advance, as in 103.
- `stagecand_fit.py`'s `hint_asg` and `cand_and_degree`, which this stage
  reuses verbatim so $c$ and $k$ cannot drift from 102's and 103's.
- 103's published $Q_{\text{null}} = 0.3379$ and the 0-of-24,008 count.
- **Not inspected:** any value of $\Pi(D)$ for any $D \ge 1$, any redundancy
  fraction, any per-$k$ stratum, any colour-presence rate.

## Disclosure, in the form 101–103 exist to force

**Two of the six predictions below are near-certainties and are registered as
checks, not as findings.** L1 asks whether all four colours appear among the
hints; with 24–46 hints over 4 colours it would be remarkable if they did not,
and I expect ~1.0000. L5's *direction* is largely forced by pigeonhole — more
neighbours means more duplicated colours means a dropped one matters less — so
only its *magnitude* is live. **The stage is carried by L2, L3, L4 and L6.**

Equally: $\Pi(k) = Q_{\text{null}}$ is an identity, not a result, and if the
scorer reproduces 0.3379 that is diagnostic E1 doing its job, not evidence.

## Predictions

**L1 — GLOBAL COLOUR PRESENCE CARRIES NO INFORMATION (check).** At every fit
density, the fraction of instances in which all $q = 4$ colours appear among
the hints is $\ge 0.99$. *Falsified if* below 0.90 at any fit density. A hit
kills account (B1) outright: if every colour is always "seen in context", a
global-inhibition emitter has no basis for choosing and cannot reach 1.0000.

**L2 — DROPPING ONE NEIGHBOUR IS ALREADY EXPENSIVE.** Pooled over the ten fit
densities, $\Pi(1) \in [0.70, 0.90]$. *Falsified if* $\Pi(1) \ge 0.95$ (one
neighbour is nearly free, and the join need not be complete) or $\le 0.55$.

**L3 — AND THE LADDER KEEPS FALLING.** $\Pi(2) \in [0.40, 0.70]$, and
$\Pi(D)$ is strictly decreasing in $D$ at every fit density. *Falsified if*
$\Pi(2) \ge 0.80$ or the sequence is non-monotone at $\ge 3$ of 10 densities.

**L4 — BUT THE JOIN HAS REAL SLACK, AND THAT IS THE ARCHITECTURAL CONTENT.**
The fraction of named cells having at least one *droppable* neighbour — one
whose colour also appears on another assigned neighbour, so removing it leaves
$S$ unchanged — is $\ge 0.50$ at every fit density. *Falsified if* $< 0.30$ at
$\ge 4$ of 10. A high value means the model need not attend to all $k$
neighbours equally, which is what an analyzable architecture would exploit.

**L5 — SMALL-$k$ CELLS ARE THE UNFORGIVING ONES (direction is a check).**
$\Pi(1 \mid k = 2) \le \Pi(1 \mid k = 5) - 0.10$. *Falsified if* the gap is
$\le 0$. Only the magnitude is live; the sign is pigeonhole.

**L6 — THE HEADLINE: THE JOIN COMPLETES ON ESSENTIALLY EVERY EMISSION.**
Combining $\Pi(1)$ with 103's 0 improper draws in 24,008 wrong unforced
emissions, the one-sided 95% lower bound on the join-completion rate,
$\pi \ge 1 - \dfrac{3}{24008\,\bigl(1 - \Pi(1)\bigr)}$ by the rule of three,
exceeds **0.999**. *Falsified if* it comes out below 0.99.

## Diagnostics

**E1 — THE LADDER REPRODUCES 103's NULL AT ITS FAR END.** $\Pi(k)$, the
full-drop rung, must equal 103's $Q_{\text{null}}$ to within 0.0005 pooled over
the ten fit densities. These are the same quantity computed by two different
code paths; a mismatch means the cell population drifted and the stage stops.

**E2 — THE NEAR END IS EXACTLY ONE.** $\Pi(0) = 1$ on every cell, by
construction. A violation is an arithmetic bug.

**Both diagnostics carry their domain:** E1 over the ten fit densities on the
matched named-cell population; E2 over every named cell read, forced ones
excluded since $c = 1$ makes the ratio $0/0$.

## Scope

Decided on the **ten fit densities** `FIT_RHOS = GRID[:10]`; all 14 computed,
the four below marked extension and deciding nothing. Population is the **79**
in79 runs for the ladder, with the **77 non-carrier** runs used for L6's count,
since 103 established the 2 carriers behave differently and pooling them would
understate the bound. Unforced named cells only. **Aggregate level only for
properness** — `color_proper` was never written per instance, so no claim here
pairs a specific emission with a specific cell.

## Figures

| | |
| --- | --- |
| `L2_ladder.png` | $\Pi(D)$ against $D$, with 103's $Q_{\text{null}}$ marked at the far end |
| `L3_density.png` | $\Pi(1)$ and $\Pi(2)$ against rho |
| `L4_redundancy.png` | droppable-neighbour fraction against rho |
| `L5_strata.png` | $\Pi(1)$ by $k$ |
| `L6_bound.png` | expected violations against $D$, against the observed 0 |

## Output

`runs/stagelocal_2026-11-19/`: `local_density.csv`, `local_strata.csv`,
`VERDICT.txt`, `figures/`.

Reads `F_errors_*.csv`, `F_hazard_*.csv`, `B_hazard_*.csv` and the cached
instance pickles only. Runs on a **login node in seconds**. **Nothing is
submitted, nothing is trained, no new instances, no model is loaded.**
