# Stage RULE6 — registered before the scorer exists

Written 2026-11-23, after stage RULE5 (findings 127/128) and before any
per-cell numerator has been read. The probe that preceded this file read the
architecture roster and the **structural** denominators only; it was written so
that it could not read `n_err(0)` or `n_in_F(0)` per cell.

Sixth refit of `runs/stagelogic_2026-11-17`. **Nothing is trained and nothing is
submitted.** No closed knob is reopened: `width` and `depth` appear only as
labels on cells that already exist.

## What RULE5 left open

Finding 127 established that `h_rule`'s denominator — the set of cells the model
chose to name — is picked by the model, and that it shrinks from 96.3% to 75.9%
of rollouts as hints are removed. The consequence was drawn against *density*.

It was not drawn against *architecture*, and that is where the line's headline
lives. Finding 121 reads:

> THE SMALLEST OBJECT THAT KEEPS DD'S RULE IS 2.19× UNDER J2, AND BY PARAMETER
> COUNT 4.08× UNDER IT.

That ranking is a ranking by `h_rule`. Direct deduction is two things — a
**rule** (the colour forced on a cell) and a **stopping condition** (which cell
may be named at all) — and 121 measured the rule with the stopping condition
divided out, deliberately. The only check on the stopping condition anywhere in
this line is the solving gate `rho_50 <= 0.740`, and findings 98/99/100 showed
that gate **never charges for naming an unforced cell**.

So the stopping condition has never been measured per architecture. At `j = 0`
it can be, with a denominator nothing chooses.

## The quantity

Per `(run_id, rho)` at **`j = 0` only**, pooled per architecture over all
fourteen densities `rho = 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55,
0.50, 0.45, 0.40, 0.35, 0.33`:

$$u \;=\; 1 - s_F(0) \;=\; \frac{\sum \big(n_{\text{at risk}}(0) - n_{\text{in }F}(0)\big)}{\sum n_{\text{at risk}}(0)}$$

the **stopping-condition failure rate at the first emission**: the fraction of
rollouts whose first emitted cell is one that one-step propagation does *not*
force. Alongside it, the two rates RULE5 defined, now per cell:

$$h^{\text{rule}}(0) = \frac{\sum n_{\text{err in }F}(0)}{\sum n_{\text{in }F}(0)}, \qquad h^{\text{all}}(0) = \frac{\sum n_{\text{err}}(0)}{\sum n_{\text{at risk}}(0)}$$

`u` and `h_all(0)` are different events. By finding 124, `n_err` counts a wrong
**colour**; naming an unforced cell with the *right* colour is counted by `u`
and not by `h_all(0)`. They are the stopping condition and the colour, measured
separately, on the same fixed denominator.

## Why the denominator cannot be argued with

`n_at_risk(0) = 400` on every `(run, rho)` by construction — 400 instances per
bin, every rollout alive at its own first emission. Verified in advance over the
whole roster, and this table is the stage's basis:

| cell | params | seeds | at_risk(0) |
| --- | --- | --- | --- |
| `L=16 h=4 tied_no_mlp d192` | 270,010 | 1 | 5,600 |
| `L=6 h=4 no_mlp d128 ep100` | 480,058 | 1 | 5,600 |
| `L=16 h=4 tied_no_mlp d320` | 613,818 | 3 | 16,800 |
| `L=12 h=4 no_mlp d128` | 879,418 | 1 | 5,600 |
| `L=16 h=4 tied_no_mlp d320 k2` | 1,025,978 | 3 | 16,800 |
| `L=16 h=4 tied_no_mlp d448` | 1,088,698 | 1 | 5,600 |
| `L=16 h=4 no_mlp d128` | 1,145,658 | 5 | 28,000 |
| `L=4 h=4 no_mlp` | 1,218,106 | 1 | 5,600 |
| `L=12 h=4 no_mlp d160` | 1,345,018 | 3 | 16,800 |
| `L=6 h=4 no_mlp` | 1,746,490 | 2 | 11,200 |
| `L=6 h=4 no_mlp ep100` | 1,746,490 | 3 | 16,800 |
| `L=16 h=4 no_mlp d160` | 1,759,738 | 3 | 16,800 |
| `L=16 h=4 tied_no_mlp d320 k4` | 1,850,298 | 3 | 16,800 |
| `L=12 h=4 no_mlp d192` | 1,908,922 | 3 | 16,800 |
| `L=16 h=4 no_mlp d176` | 2,115,930 | 3 | 16,800 |
| `L=4 h=4 rf_mlp` | 2,267,706 | 1 | 5,600 |
| `L=8 h=4 no_mlp` | 2,274,874 | 3 | 16,800 |
| `L=16 h=4 no_mlp d192` (J2) | 2,504,890 | 3 | 16,800 |
| `L=3 h=4 full` | 2,530,618 | 3 | 16,800 |
| `L=6 h=4 full ff256` | 2,535,994 | 3 | 16,800 |
| `L=12 h=4 no_mlp d224` | 2,571,130 | 3 | 16,800 |
| `L=4 h=4 full` | 3,320,378 | 2 | 11,200 |
| `L=4 h=4 linear` | 3,320,378 | 3 | 16,800 |
| `L=6 h=4 rf_mlp` | 3,320,890 | 2 | 11,200 |
| `L=12 h=4 no_mlp` | 3,331,642 | 3 | 16,800 |
| `L=6 h=4 full d192` | 3,381,562 | 3 | 16,800 |
| `L=6 h=4 no_mlp d384` | 3,799,354 | 3 | 16,800 |
| `L=6 h=4 full` | 4,899,898 | 3 | 16,800 |
| `L=6 h=4 linear` | 4,899,898 | 2 | 11,200 |
| `L=8 h=4 full` | 6,479,418 | 3 | 16,800 |
| `L=12 h=4 full` | 9,638,458 | 3 | 16,800 |

Every row is exactly `400 × 14 × seeds`, and the total is exactly
`79 × 14 × 400 = 442,400`. **Six of the thirty-one cells are single-seed** and
carry no spread: `tied_no_mlp d192`, `no_mlp d128 ep100`, `no_mlp L=12 d128`,
`tied_no_mlp d448`, `no_mlp L=4`, `rf_mlp L=4`. Any ranking claim must say so.

The population baseline from RULE5, equal weight per density, is
`s_F = 0.8157`, so **`u ≈ 0.1843` pooled over all 79 runs**.

## Hypothesis

The stopping condition is a *property of the architecture*, varies across the
roster by more than the measurement error, and is **not** what `h_rule` is
measuring. If so, 121's ranking survives as a statement about the rule — and
gains a second, independent axis it was never scored on.

## Predictions, with a size floor on every one

RULE5's lesson (ii) was that a significance test without a size floor "resolves"
any ripple once the numerators are large. Every prediction below carries an
explicit factor, not only an interval.

- **E1 — `u` carries architecture information.** Across the 31 cells,
  `max(u) / min(u) >= 2.0`. **MISS** if `< 2.0`: the stopping condition would
  then be fixed by the task, and the only thing an architecture controls at
  `j = 0` is the colour.
- **E2 — `u` and `h_rule(0)` are close to independent.** Spearman over the 31
  cells, `|rho| < 0.6`. **MISS** if `>= 0.6`: the rule and the stopping
  condition would be near-redundant, and 121's ranking would be measuring both
  at once rather than the rule alone.
- **E3 — the tractable answer is not keeping the rule by declining to be
  tested.** `u` for the answer cell `tied_no_mlp L=16 d320` (613,818, 3 seeds)
  is within **1.25×** of the standing J2 reference `no_mlp L=16 d192`
  (2,504,890, 3 seeds). **MISS** if `u(tied d320) > 1.25 × u(J2)` — that would
  mean the answer cell attains `h_rule = 0.00002` partly by naming cells outside
  the forced set far more often, and **finding 121's deck
  `/spin-gnn/rule-hazard/` gets a correction page.**
- **E4 — the broken tied cell breaks on the rule, not on the stopping
  condition.** `tied_no_mlp L=16 d192` reads `h_rule = 0.08106`, about 4,000×
  its `d320` sibling. Its `u` is within **1.5×** of `d320`'s. **MISS** if
  `> 1.5×`: losing the rule at `d192` would then be losing the stopping
  condition too, and "the tied map loses the rule between `d192` and `d320`"
  is the wrong description of what width does.
- **E5 — reproduction.** Summing the 31 cells' `j = 0` counts back over density
  reproduces stage RULE5's fourteen-row `h_all(0)`, `h_rule(0)`, `h_unf(0)` and
  `s_F` table to **`0.000e+00`**. **MISS** on any non-zero residual: the
  `run_id → cell` join would be wrong and nothing else in the stage is readable.

## Diagnostics

- **D1** — every cell's `Σ n_at_risk(0) = 400 × 14 × seeds`, total 442,400,
  14 densities present in every cell. Zero violations required.
- **D2** — `n_err >= n_err_in_F` and `n_in_F <= n_at_risk` on every `j = 0` row.
- **D3** — the `run_id → cell` map is a partition: 79 runs, 31 cells, no run
  mapped to two cells, no orphan run in the hazard shards, no cell without runs.
- **D4** — per-cell `n_in_F(0)` summed by density reproduces RULE4's
  `REG_DEN = {0.95: 30442, …, 0.33: 23990}` exactly.
- **D5** — keyed on **`cell`**, never on `(arch_mode, num_layers, d_model)`.
  Asserted explicitly, because `L=6 h=4 no_mlp` and `L=6 h=4 no_mlp ep100` have
  the **same 1,746,490 parameters** and the same triple; pooling them is the
  96/99 defect. They must stay two cells with 2 and 3 seeds.

## The interpretive limit

`u` measures **agreement with DD's stopping condition**, not logical validity. A
solver that guesses a cell propagation has not forced, and then checks it, is
reasoning; it is simply not reasoning in DD's trace format. So a high `u` is a
departure from the planted trace, and only the *rule* arm carries a claim about
correctness. Below `rho_c ≈ 0.465` even that weakens, exactly as in RULE5: an
off-trace colour need not be a logical error, so the four bins
`rho = 0.45, 0.40, 0.35, 0.33` bound the rule from above. `u` itself is
unaffected by that caveat, because it is about *which cell*, not which colour.

Everything here is `j = 0`. Nothing in this stage says anything about later
indices, where censoring returns and RULE3's `lambda* = 1.19` applies.

`u` is **not a solve rate** and is never converted into one.

## What a null licenses

If E1 misses, the stopping condition is a task constant and this stage ends
with that sentence. If E3 misses, 121's headline is qualified in public and the
deck gets a correction page. **No training follows from any outcome.** The width
sweep between `d192` and `d320` stays closed, seeding the single-seed cells
stays hshunt's call, and stage DQ4 stays unsubmitted.

## Traps

1. **121's `h_rule` and this stage's `h_rule(0)` are different numbers for the
   same cell** — 121 pools over every `j` and the ten fitted densities, this
   pools `j = 0` over all fourteen. Never quote one as the other, and never
   present a re-ranking as a correction to 121 without saying which quantity
   moved.
2. **Name the densities, never "the fourteen."** RULE5's E4 quoted a Spearman
   from a different ten bins than it had registered. Every scored set in this
   stage prints its own bins.
3. **Six single-seed cells.** Any "smallest object that…" sentence states the
   seed count in the same breath, as 121 did.
4. **The talk deck `transition-and-minimal-models` is not touched at all**,
   whatever this stage finds. hshunt is editing a local copy.
5. If this contradicts finding 121, `/spin-gnn/rule-hazard/` gets a
   **correction page**, not a quiet rebuild — the same way RULE4's deck was
   handled after RULE5.
6. **`u` is not the gate.** `rho_50 <= 0.740` remains a solving gate and is not
   replaced, reinterpreted or re-derived here.
7. A per-cell interval on a numerator in the thousands is narrow by
   construction. Report the **factor** first and the interval second,
   everywhere.
