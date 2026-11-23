# STAGE DQ4 — registered predictions, written BEFORE the array was submitted

Ensemble `d=4 q=3`, `N=120` only. Chance for `reason_color` is
`ln 3 = 1.098612` nats; chance accuracy is `1/3 = 0.3333`.

**NOTHING FROM `q=4` IS QUOTED ANYWHERE IN THIS STAGE** — not the free line
0.740, `s*` 0.1484/0.119, the `val_nll` floors 1.371/1.625/1.861,
`(5/16) log N`, nor `ln 4` itself. The two ensembles have different chance
levels, different critical densities and different sequence lengths.

## The one knob

**The seed, and only the seed.** Seeds **3, 4, 5, 6, 7** at `N=120`, with the
architecture, optimiser, data budget, generator and evaluation grid identical
to stages DQ (seeds 0, 1) and DQ3 (seed 2). Together with those three cells
this gives **8 seeds at one size**.

`--seed-start` is left at its default in `rho_transition_eval`, exactly as in
DQ and DQ3, so all eight cells are scored on the **same 400 instances per
bin** and are paired point for point.

Seed count is **not** a closed knob (the closed list is width, cosine decay,
more data, best-val selection, role count, schedule length, training budget).
This stage re-tests none of them.

## Why this, and why now

Stage DQ3 (`VERDICT.txt`, `runs/stagedq3_2026-11-22/`) found that **the spread
between seeds at fixed `N` is larger than the effect of `N`, and flips the
size ordering**:

| cell | `reason_color` | `a_eff` at ρ=0.32 |
|---|---|---|
| `N=120` s0 | 1.0987 (at chance) | `< 0.94202` censored |
| `N=120` s1 | 0.0001 (rule exact) | `< 0.94202` censored |
| `N=120` s2 | 0.0002 (rule exact) | 0.98542 |

Three cells cannot measure a distribution. Every comparison this project has
drawn on this ensemble — the `N` penalty, the depth lever, the D2/E3/E4
trends — is a two- or three-cell difference, and DQ3 showed that such a
difference is **not interpretable until the spread is known**. This stage
measures the spread. It is the cheapest thing that makes every past and future
comparison on this grid readable, and it is a measurement, not an attempt to
win a number.

## Predictions

| | prediction | MISS condition |
|---|---|---|
| **F1** | every new cell's `reason_color` is at an extreme, `< 0.10` or `> 0.99` nats | **any** new cell lands in `[0.10, 0.99]` |
| **F2** | at least 4 of the 8 `N=120` cells are rule-exact (`< 0.10` nats) | 3 or fewer |
| **F3** | `a_eff` spread at ρ=0.32 across the measured `N=120` cells exceeds **0.0065** | spread ≤ 0.0065 |
| **F4** | at least one new cell holds the rule (`reason_color < 0.10`) while search is broken (`reason_node > 0.50`) | no cell shows that combination |
| **F5** | the best new cell's `a_eff` at ρ=0.32 beats the standing **0.99191** | it does not |

**F1 has a real chance of MISSing** and that is the point of registering it:
stage DQ already produced one intermediate cell, 0.4468 at `N=96`. If
intermediate cells turn out to be common, "two basins" is the wrong
description and the right one is a continuum with mass near the ends.

**F3's bar, 0.0065, is not arbitrary** — it is the best-to-best `N=96`→`N=120`
penalty measured in DQ3 (0.99191 → 0.98542). F3 asks, with 8 seeds instead of
3, whether seed noise really is the larger of the two.

**F5 is the standing improvement bar and is expected to MISS.** 0.99191 was set
by `N=96` s1 and no `N=120` cell has approached it. It is registered anyway so
that a lucky cell could not be reported as an improvement after the fact.

## Registered falsifier for findings 119/120

If all five new cells land **within 0.01 nats of each other** on
`reason_color` **and** within **0.005** of each other on `a_eff` at ρ=0.32,
then there is no seed lottery, DQ3's seed 2 was a single outlier, and findings
119 and 120 are overturned rather than extended. That outcome is to be
reported as plainly as the expected one.

## Traps carried forward

1. **Scoring is PER CELL.** Pooling across seeds produced stage DQ's
   misleading D2; any pooled number is reported beside its parts.
2. **If a bin solves 0 of 400, `a_eff = 0` is NOT reported.** Zero successes
   bounds the rate: `p < 1 - 0.05^(1/400) = 0.00746136`,
   `a_eff < p^(1/n_blank)`. The bound **rises with `T`** purely because
   `n_blank` grows — an artefact, and it must be labelled as one wherever it
   is drawn. At `N=120`, ρ=0.32, `n_blank = 82`, the bound is `< 0.94202` and
   is **identical for every censored seed**, so it is drawn **once in grey**,
   never per seed.
3. **A censored cell contributes to F2 but not to F3.** `reason_color` is
   measured for every cell; `a_eff` only exists where something was solved.
   F3's spread is over the measured cells, and the count of censored ones is
   reported with it.
4. **`val_nll` cannot rank these cells.** DQ3: 0.019 nats of aggregate loss
   sat on top of a 47× difference in whole-instance solving. It is a health
   check and nothing else.
5. **Never `scripts/stagensplit_fit.py` on a q=3 cell** — its `FLOOR` dict is
   keyed on `N` alone, so it KeyErrors at `N=120` and would silently apply a
   `q=4` `val_nll` floor.
6. **Check for the artefact, not the exit code.** DQ3's split exited `0:0`
   with no verdict because a `|| true` masked a scorer crash. The DQ4 split
   must not wrap its scorer in `|| true`.

## What a null licenses

Nothing but a report. Not a bigger model, not a longer schedule, not another
size rung. If F1 MISSes the honest conclusion is "the state is a continuum",
and if the falsifier above fires the honest conclusion is "DQ3's seed 2 was an
outlier" — in both cases the deliverable is the deck, not a follow-up run.
