# Stage PROPER — is the emitted COLOUR inside the named cell's candidate set?

Written and committed **before** `scripts/stageproper_fit.py` existed. No number
below was chosen after seeing it.

## Why this stage exists, and why it is not CAND again

102 (stage CAND) asked whether the model's **choice of cell** tracks the
candidate set, and the answer was no: the apparent preference for near-forced
cells is selection on `k`, the number of coloured neighbours, and `c` and `k`
are nearly collinear at `q = 4`, `d = 5` (an unforced cell with `k = 4` has two
candidates 100.00% of the time). CAND's own verdict concluded that separating
the two accounts "needs a GENERATOR change, not a scorer".

**That conclusion is about the CELL step. It says nothing about the COLOUR
step, and the colour step is not confounded by the `c`–`k` collinearity at all,
because the cell is held fixed: given whatever cell the model named, is the
colour it wrote one the constraints still allow?**

**And the column that answers it was written months ago by a scorer nobody has
read for this purpose.** `stagepos_score.py:385`:

```python
color_proper = bool(color_is and all(assigned.get(int(u)) != c for u in adj[node]))
```

— the emitted colour conflicts with **no already-coloured neighbour**, which is
exactly *the colour is in the named cell's candidate set*. It is aggregated per
`(run, rho, j)` as `n_color_proper_u` in `B_hazard_*.csv`, alongside
`n_color_correct_u` and `n_emitted`. **`grep` finds `color_proper` in no
analysis script, no `VERDICT.txt`, `STATUS.md`, `LOG.md` or `FINDINGS.md`: it
has never been read.** This is the fourth time the next question was already on
disk (98, 100, 102, and now this).

## What the columns mean, read from the writer and not from their names

- `n_emitted` counts **every** pair emitted at that `j`. The `_u` suffix means
  **unconditional** — not restricted to `j <= stop_j`, and **not** "unforced".
  At `j = 0` every rollout emits exactly one pair and `stop_j >= 0` always, so
  `n_emitted = n_at_risk = n_instances = 400` and the distinction vanishes.
- `n_color_correct_u` — the emitted colour equals the planted colour.
- `n_color_proper_u` — the emitted colour clashes with no assigned neighbour.
  At `j = 0` `assigned` is **exactly the hints**, so this is candidate-set
  membership under the same pre-state CAND reconstructed.
- **correct implies proper**, because the planted colouring is proper. So
  `n_color_proper_u >= n_color_correct_u` by construction (that is E2).
- On a **forced** cell `c = 1`: the single proper colour *is* the planted one,
  so there proper and correct coincide and a wrong colour is necessarily
  improper.

## The observable

Per `(run, rho)` at `j = 0`, with `N = n_emitted`:

$$Q \;=\; \frac{n_{\text{proper}} - n_{\text{correct}}}{N - n_{\text{correct}}}$$

— **among the emissions whose colour was wrong, the share that are nonetheless
locally consistent.** Wrong-but-proper can only happen on an unforced cell, so
restricting the denominator to unforced errors using `n_err_in_F` from stage
LOGIC's `F_hazard` gives the sharper

$$Q_u \;=\; \frac{n_{\text{proper}} - n_{\text{correct}}}{N - n_{\text{correct}} - n_{\text{err,}F}}.$$

**The matched null.** For each `j = 0` first-error record in `F_errors`, the
named cell is known and CAND's reconstruction gives its candidate count `c`. A
colour-blind emitter that avoids the planted colour has `c - 1` proper choices
out of 3 wrong ones, so

$$Q_{\text{null}} \;=\; \operatorname{mean}_{\text{erring rollouts}} \frac{c-1}{3},$$

which is 0 on a forced cell automatically. This is matched on the **same named
cells of the same instances**, so it is not a global base rate.

## What was inspected before registering, and what was not

- `walk_continuation` (`stagepos_score.py:345–397`) and `measure_B` (`400–480`)
  in full, including that `recs` carries **no** per-instance proper flag — so
  this stage is aggregate-level only and **cannot stratify `Q` by `c` or `k`**,
  which is stated here as a scope limit and not discovered later.
- The `B_hazard` header (28 columns) and that `j = 0` has **1124 rows over 80
  runs and 14 densities** — one run above the 79, to be filtered on `in79`.
- `grep` for `color_proper` across `scripts/`, `STATUS.md`, `LOG.md`,
  `FINDINGS.md`: two hits, both in the scorer that writes it.
- **Not inspected:** any value of `n_color_proper_u`, any `Q`, any `Q_null`, any
  per-density or per-run number this stage is about.

## The disclosure maxim 101 and 102 exist to force

**A high `Q` is half-implied by a number already published, and I will not claim
it as a discovery.** 98 records `M = Pr(colour correct | unforced)` in
`[0.4647, 0.5159]` against a blind `1/q = 0.25`. Under the two-component
mixture "with probability `p` draw uniformly from the candidate set, otherwise
uniformly from all `q` colours", `M ≈ 0.49` with `c ≈ 2` forces `p ≈ 0.96`,
hence `Q` near 0.97. So R1 is a **consistency check, deliberately scored as
such**, and the stage is carried by R2–R5, none of which follow from `M`.

## Predictions

**R1 — THE COLOUR IS ALMOST ALWAYS LOCALLY CONSISTENT (consistency check).**
Pooled over the ten fit densities, `Q >= 0.90` and `Q - Q_null >= +0.30`.
*Falsified if* `Q <= 0.60` or `Q - Q_null <= +0.20`. Scored, but a hit here is
worth little for the reason disclosed above.

**R2 — THE RESIDUAL IS REAL AND SMALL.** `1 - Q_u`, the rate at which the model
writes a colour that contradicts a neighbour it can already see, lies in
**[0.005, 0.15]** at **>= 7 of 10** fit densities. *Falsified* in either
direction: below 0.005 at >= 7 of 10 (the model never does it, and the colour
step is an exact candidate draw), or above 0.15 at >= 4 of 10 (it does it often
enough that "draws from the candidate set" is the wrong description).

**R3 — THE HEADLINE: THE COLOUR RULE DOES NOT DEGRADE WITH DIFFICULTY, EVEN
THOUGH THE CORRECT-RATE DOES.** `M` falls by 0.0544 over the ten fit densities
and CAND showed its predictor `P̂` is effectively constant, so something must
account for the fall. If the colour step is an unchanging candidate draw, then
`Q_u` is **flat** and the fall in `M` is entirely the shift of `π(c)` toward
larger `c`. Registered: the **range of `Q_u` over the ten fit densities is
<= 0.05** and `Spearman(Q_u, rho) <= +0.60`. *Falsified if* the range exceeds
**0.10** and `Spearman(Q_u, rho) >= +0.80` — the colour rule itself degrading
on hard puzzles, which would be a different and more interesting model.

**R4 — THE RESIDUAL EXPLAINS THE PART OF `M` THAT `P̂` MISSES.** Over the ten
fit densities, `Spearman(P̂ - M, 1 - Q_u) >= +0.60`. *Falsified if* `<= 0`. A
negative or absent correlation means the improper draws are not where `M`'s
density dependence comes from, and CAND's unexplained residual stays
unexplained.

**R5 — THE SMALL ARCHITECTURES CARRY THE IMPROPER DRAWS.** At `rho = 0.60`,
across the 79 runs, `Spearman(1 - Q_u, log n_trainable_params) <= -0.30`.
*Falsified if* `>= +0.10`. BOUND found the three smallest architectures carried
39 of 63 wrong-colour-on-forced-cell errors; if properness is the same facility,
it should scale the same way.

R1 is a check. R3 is the one that would change the account: a flat `Q_u` makes
the colour step a fixed, constraint-respecting rule whose only density
dependence is the puzzle's, and it is the first positive evidence in this line
that **something** in the model represents the candidate set — the cell step
does not, as 102 showed, but the colour step might.

## Diagnostics

**E1 — THE TWO SCORERS SAW THE SAME ROLLOUTS.** `B_hazard` (stage POS) and
`F_hazard` (stage LOGIC) are different scorers over the same deterministic
greedy rollouts on the same cached instances. At `j = 0`, on every in79 cell:
`n_emitted == n_at_risk == n_instances == 400`, and
`|(N - n_err) - n_{\text{correct}}|` must be at most that cell's `node_invalid`
count (an emission can be `node_invalid` and still carry the planted colour).
Any larger mismatch means the join is wrong and the stage stops there.

**E2 — PROPER DOMINATES CORRECT.** `n_color_proper_u >= n_color_correct_u` on
all cells, by construction. A violation is a scorer bug, not a finding.

**Both diagnostics carry their domain** — the lesson 102's D1 cost. E1 is
asserted only over in79 cells at `j = 0`; E2 over every row read.

## Scope

Decided on the **ten fit densities** `FIT_RHOS = GRID[:10]`; all 14 computed and
reported, the four below marked extension and used to decide nothing.
Population is the **79 in79 runs**; the 80th run present in `B_hazard` is
filtered out. `j = 0` has no `S = 0` / `S = 1` censoring. **Aggregate level
only** — no stratification of `Q` by `c` or `k` is possible, because
`color_proper` was never written per instance.

## Figures

| | |
| --- | --- |
| `R1_proper.png` | `Q` and `Q_null` against rho |
| `R2_residual.png` | `1 - Q_u` against rho |
| `R3_flat.png` | `Q_u` and `M` against rho, each normalised to its own value at rho 0.95 |
| `R4_explains.png` | `P̂ - M` against `1 - Q_u` over the ten fit densities |
| `R5_size.png` | `1 - Q_u` against trainable parameters at rho 0.60 |

## Output

`runs/stageproper_2026-11-17/`: `proper_density.csv`, `proper_runs.csv`,
`VERDICT.txt`, `figures/`.

Reads `B_hazard_*.csv`, `F_hazard_*.csv`, `F_errors_*.csv` and the cached
instance pickles only. Runs on a **login node in seconds**. **Nothing is
submitted, nothing is trained, no new instances, no model is loaded.**
