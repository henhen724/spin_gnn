# STAGE RULE2 — registered predictions, written BEFORE the scorer was run

A **second refit** of the same shards stage RULE read
(`runs/stagelogic_2026-11-17/F_hazard_all_*of10.csv` and
`LF_summary_all_*of10.csv`). Seconds on a login node. **NOTHING IS SUBMITTED,
NO MODEL IS TRAINED, NO ROLLOUT IS REGENERATED.** Same 79 runs, same 31
architectures, same 400 instances per bin, same eval seed start
60,000,000,000.

## The one knob

Stage RULE pooled `h_rule` over the **ten fitted** densities, ρ = 0.95 … 0.50,
because `FIT_RHOS = GRID[:10]` and ρ = 0.45 / 0.40 / 0.35 / 0.33 were generated
but never fitted (trap 2, finding 101). **This stage changes exactly that: pool
over all fourteen.** Nothing else moves — not the instrument, not the bars, not
the size order, not the population.

$$ h^{\text{rule}} \;=\; \frac{\sum_j n\_err\_in\_F(j)}{\sum_j n\_in\_F(j)}
   \;=\; \Pr(\text{colour wrong} \mid \text{named a forced cell}) $$

## Why the four added bins are not bookkeeping

**ρ_c ≈ 0.465 on this `d5q4` ensemble** (117, calibrated by recovering 0.465 on
the `d5q4` arm). The four never-fitted densities — 0.45, 0.40, 0.35, 0.33 — are
therefore **exactly the bins at or below the colouring transition**, where the
instances leave DD's fast phase. So the extension asks a real question:

> **Does DD's rule survive below ρ_c, or was "keeps the rule" only ever "keeps
> the rule in the fast phase"?**

That matters directly for the 2026-11-23 brief. If the rule breaks below ρ_c,
then the smallest architecture that "still learns direct deduction" is smallest
only on fast-phase instances, and the headline needs that qualifier.

## HYPOTHESIS, stated before the numbers

**Survivorship will make the extension nearly inert, and where it moves at all
it will move UP.** A rollout is censored at its first error of any kind, so the
only rollouts still contributing a denominator in the deep bins are those that
got that far without erring. At ρ = 0.33 that selection is already known to be
near-total: **49 of 79 runs never produce one DD-faithful rollout** (101).

So I expect the added four bins to contribute a small share of the total forced
emissions, the per-architecture numbers to barely move, and the ranking to
survive. I expect no architecture to cross the 0.01 bar.

**THE DIRECTION OF THE BIAS IS REGISTERED HERE SO IT CANNOT BE READ BACKWARDS.**
Censoring keeps the *lucky* rollouts and drops the broken ones, so it biases
`h_rule` **downward** at low density. Therefore:

- a **rise** at low ρ is a real effect, measured against a bias pushing the
  other way;
- a **fall** at low ρ is **not** evidence that the rule improves on harder
  instances. It is the survivorship artefact, and it must be reported as one.

## Predictions

| | prediction | MISS condition |
|---|---|---|
| **E1** | the four added bins contribute **< 25%** of total forced emissions, median over the 31 | median share ≥ 25% |
| **E2** | the headline cell `L=16 h=4 no_mlp d128` stays **below 0.01** on all-14 pooling | ≥ 0.01 |
| **E3** | the bar partition is unchanged: **2** architectures above 0.01, **29** below | any architecture crosses, either way |
| **E4** | the three largest `h_rule` are the **same three architectures** as at ten densities | any of the three is displaced |
| **E5** | pooled over the population, `h_rule` at ρ = 0.33 **exceeds** `h_rule` at ρ = 0.50 | it is lower or equal |
| **E6** | Spearman(`h_rule` at 10 densities, `h_rule` at 14) over the 31 is **≥ 0.90** | below 0.90 |

**E5 is the prediction that carries the science** and the one most likely to
MISS, because the survivorship bias pushes against it. **E1 is the one that
decides whether anything else here is interesting**: if the added bins are 2% of
the denominator, the whole extension is a footnote and should be reported as
one, not dressed up.

**E6 is the falsifier for stage RULE's conclusion.** If the ranking does not
survive the extension, then stage RULE's headline was an artefact of which
densities happened to be fitted, and finding 121 needs amending rather than
extending.

## Traps carried forward

1. **KEY ON `cell`, NEVER ON `(arch_mode, num_layers, d_model)`** — that triple
   pools `full L=6 d256` (4,899,898) with `full L=6 ff256` (2,535,994) into a
   fake six-seed median (the 96/99 defect). 79 runs → **31** `cell` labels.
2. **NO SINGLE-SEED RUNG IS A MEASUREMENT OF ITS ARCHITECTURE** (41/42, re-taught
   by 119/120: 0.1627 of seed spread at fixed architecture, one seed flipping a
   size ordering). Per-seed parts are printed beside every pooled number, and
   the headline stays the smallest **3+-seed** cell.
3. **`rho_50` IS A COLOURING CRITERION, NEVER A DD ONE** (98). It appears only as
   the thing `h_rule` is compared against, with that sentence attached.
4. **THIS IS THE `d5q4` POPULATION.** A blind colour guess is `1 − 1/q = 0.75`.
   **No `d4q3` constant enters** — not ρ_c ≈ 0.25, not `ln 3`, not 0.99191.
   ρ_c ≈ 0.465 is this ensemble's own number and is the reason the four bins
   matter.
5. **A ZERO DENOMINATOR IS NOT A ZERO HAZARD.** An architecture that never names
   a forced cell in a bin contributes nothing to that bin, and is not perfect
   there. Censored and empty bins are listed with their counts.
6. **`h_rule` SEPARATES, IT DOES NOT RANK** (121). Among the architectures that
   have the rule, a change of 0.00001 is not a result. Only crossings of the bar
   and changes in the three-breaker set are reported as findings.
7. **CHECK FOR THE ARTEFACT, NOT THE EXIT CODE** — DQ3's split exited `0:0` with
   no verdict because a `|| true` masked a crash. No `|| true` anywhere here.

## Diagnostics (if any fails, nothing above is reported)

- **D1 — reproduce finding 100, unchanged from stage RULE.** Total
  `n_err_in_F` over all 14 densities = **1,988**; affected cells **63 of
  1,106**; the three named architectures hold **39** of those 63 with counts
  **977 / 677 / 307**.
- **D2 — column sanity.** `n_in_F ≤ n_at_risk` and `n_err_in_F ≤ n_err` on every
  in79 row. Any violation stops the stage.
- **D3 — the pooling identity.** Summed over the 31 architectures, the all-14
  rule-error total must equal **1,988** exactly, and per architecture both
  `n_in_F` and `n_err_in_F` must be **non-decreasing** from 10 densities to 14.
  A decrease is arithmetically impossible and would mean the pooling set is
  wrong.
- **D4 — the shared scorer is untouched in behaviour.** `per_cell()` gains an
  optional `rhos` argument defaulting to `FIT_RHOS`, so stage RULE's
  `VERDICT.txt` must regenerate **byte-identical (md5)** after the patch. If it
  does not, the patch changed stage RULE's published numbers and is reverted.

## What a null licenses

A report, and nothing else. If E1 shows the added bins are negligible, the
honest headline is "the never-fitted densities carry almost no forced emissions,
so finding 121 is unchanged and could not have been otherwise" — **not** a
search for a density range where some other architecture wins. If E6 MISSes,
finding 121 is amended, not defended. **No training follows from any outcome
here**; the two seeding candidates in `STATUS.md` still wait on hshunt.
