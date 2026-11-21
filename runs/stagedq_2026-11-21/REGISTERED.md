# STAGE DQ — registered before any cell lands

Registered 2026-11-21, commit `992162b`, branch `sudoku-dd`.
Ensemble pre-check: `runs/stagedq_ensemble_2026-11-21/`.
Timing probe: `53242607`. Array: **not yet submitted — sized from the probe.**

hshunt chose option (e) on 2026-11-21 — the `(degree, n_colors)` depth lever —
over the trace-format change (f) and a fourth size rung.

---

## The question this stage exists to answer

Finding 116 established that at `d=5, q=4` the model's `reason_color`
cross-entropy sits at `ln 4 = 1.3863` nats with accuracy `1/q` at **N=192**,
at every density and on both seeds, while at **N≤96** it is near-perfect
(0.0036 nats, 0.9991). It never learns the colouring rule at the larger size.

Two causes are consistent with every number we have, and they were not
separable on a size ladder because **N and DD depth move together there**:

- **H-N** — the collapse is driven by `N` itself: the node vocabulary
  (`N + n_special`), the graph section's length, the number of distinct node
  identities the model must keep straight.
- **H-T** — the collapse is driven by the **DD depth** that comes with larger
  `N`: more sequential rounds before the answer, so a longer credit path.

`(d=4, q=3)` separates them, because it buys depth from the **local**
parameters while `N` is held fixed. From `stagedq_datacheck.py`
(40 instances/cell, deepest-decile mean depth and mean encoded length):

| cell | deepest-decile `T` | mean seq len | vocab |
|---|---|---|---|
| `d4q3 N=96` | **32.8** | **687** | 105 |
| `d4q3 N=120` | **35.0** | 856 | 129 |
| `d5q4 N=96` (control, on disk) | 22.5 | 781 | 106 |
| `d5q4 N=192` (the cell that failed) | ~35.4 | ~1560 | 202 |

So `d4q3 N=96` is **deeper than `d5q4 N=96` while being shorter**, and
`d4q3 N=120` **matches the depth of the `d5q4 N=192` cell that failed** at
1.8× fewer tokens — both at an `N` where the colouring rule is known learnable.

---

## What does NOT carry over, and is not quoted anywhere in this stage

`(d=4, q=3)` is a **different ensemble**, not a new point on the old curve.
Measured in `runs/stagedq_ensemble_2026-11-21/`:

- **ρ_c moves ≈ 0.465 → ≈ 0.25.** Read off where `T/log N` stops collapsing
  across `N=48..384`. The same diagnostic recovers the **published 0.465** on
  the `d5q4` arm, so it is calibrated, not fitted.
- **ρ_min moves** ≈ 0.314 (`d5q4` N=192) → ≈ 0.150 (`d4q3` N=96).
- **The chance level moves**: `ln 3 = 1.0986`, not `ln 4 = 1.3863`. Quoting
  1.3863 anywhere in this stage is a category error.
- **Banned from this stage's write-up**: the free line 0.740, `s*` 0.1484 /
  0.119, the `val_nll` floors 1.371 / 1.625 / 1.861, `ρ_50 ≤ 0.740` as a gate,
  and finding 109's `(5/16) log N`. All are `q=4, d=5` constants. Stage DQ
  measures its own or reports none.

### Finding 10 needs amending and this stage does not lean on it

Finding 10's "`(degree, n_colors)` buys depth ~6× cheaper" came from sweep
`49509101`, whose scorer never passes `fall`. Reproduced on that ensemble I
measure **5.00× in N, 5.48× in tokens** — consistent with the published ~6×.
On `fall=True`, the ensemble **every training cell uses**, the same
measurement gives **1.58× in N, 1.77× in tokens**. The fall constraint helps
the `q=4 d=5` reference far more than the lever (at N=192 it moves ρ_min
0.4729 → 0.3134 for the reference, 0.1944 → 0.1437 for the lever), which eats
most of the gap.

**The 1.58× is not the reason to run this stage** and no prediction below
depends on it. The reason is the depth-vs-`N` separation.

---

## Cells

Array `0-3`, one GPU each, `--partition=sganguli`.

| task | N | d | q | seed | ρ range (train) |
|---|---|---|---|---|---|
| 0 | 96 | 4 | 3 | 0 | U[0.14, 0.95] |
| 1 | 96 | 4 | 3 | 1 | U[0.14, 0.95] |
| 2 | 120 | 4 | 3 | 0 | U[0.14, 0.95] |
| 3 | 120 | 4 | 3 | 1 | U[0.14, 0.95] |

Architecture **copied, not retyped**, from the N=192 ladder cell so the only
things changing are `(d, q, N, ρ)`: `--d-model 256 --nhead 4
--dim-feedforward 1024 --num-layers 12 --n-train-instances 240000 --epochs 50
--n-val-instances 300 --batch-size 32 --shuffle-adj --fall`.

ρ range starts at each ensemble's own floor, matching the q=4 protocol
(`rho=0.30` against ρ_min≈0.32 there; `rho=0.14` against ρ_min≈0.15 here).
Measured slow-phase fraction of the training mix: **27.5%** at `d4q3`, against
37.5% for `d5q4 N=96`. Lower, and stated in advance so it cannot be offered
later as an excuse for a null.

**Eval grid** (14 bins, 400 instances each, `--seed-start` left at default so
seeds are paired point for point):
`0.95 0.90 0.85 0.80 0.75 0.70 0.60 0.50 0.40 0.32 0.27 0.22 0.18 0.15`
— chosen to straddle ρ_c≈0.25 with **four bins below it**, bottoming at the
measured ρ_min. Identical grid *values* across ensembles do **not** license
pooling; the cross-ensemble axis is `dd_mean_T`.

---

## Registered predictions

### D1 — THE DISCRIMINATOR. `reason_color` at `d4q3 N=96`
Teacher-forced NLL on the deduced-colour token, pooled over eval densities,
both seeds. Chance is `ln 3 = 1.0986`.

- **H-N predicts < 0.10 nats** (N is safe at 96, so the rule is learned).
- **H-T predicts ≥ 0.55 nats** (half of chance) — depth is what breaks it.
- **0.10 – 0.55 is PARTIAL and is read as such, not rounded to either.**
  Registered in advance: a partial result means depth contributes but does not
  account for the collapse, and **D2's direction decides**, not a re-reading
  of D1.

### D2 — Monotonicity in depth inside the new ensemble
`reason_color(N=120) ≥ reason_color(N=96)`, since N=120 is the deeper cell.
- **Falsifier:** N=120 better than N=96 by **> 0.02 nats**. That would mean
  the degradation does not track depth even where depth is the thing varying,
  and D1 may not then be attributed to depth at all.

### D3 — Health check, gates whether D1 is readable
The deterministic sections stay learned: graph-row-head NLL **< 0.01** and
terminator NLL **< 0.01**, as they were at N=192 (0.0000 / 0.0006).
- **Breaking point:** either **> 0.05** means the cell is sick rather than
  specifically bad at colouring, and **D1/D2 are declared unreadable** for
  that cell instead of being interpreted.

### D4 — Slow-phase solving: the original brief
At least one eval bin with **ρ < 0.25** (below ρ_c) reaching
**solve_rate > 0.1**, on at least one seed.
- This has never been achieved at depth. The whole size ladder failed it:
  N=48 solved below the grid floor but at shallow depth, N=96 died at ρ_c,
  N=192 solved nothing anywhere.
- **Falsifier:** 0 solves in every sub-ρ_c bin on both seeds — the lever does
  not rescue slow-phase operation either, and that is reported as such.
- **If solve_rate is 0 in a bin, `a_eff = 0` IS NOT REPORTED.** Zero successes
  bounds the rate, it does not pin it: the one-sided Clopper–Pearson bound is
  `p < 1 − 0.05^(1/400) = 0.00746`, and `a_eff < p^(1/n_blank)`. Carried
  forward from the N=192 rung, where it is what kept N1 honest.

### D5 — Trainability
`val_nll` argmin falls in the last 5 epochs and the final-minus-best gap is
≤ 0. Same N4-analogue as the ladder.
- **Breaking point:** flat at the initial plateau for all 50 epochs means not
  trainable at this budget. That is the result, and **it is not retried
  larger** — the closed-knob rule applies here exactly as it did at N=192.

---

## Traps carried forward

1. **Chance is `ln 3 = 1.0986`.** A cell at 1.3863 here is not "at chance", it
   is worse than chance and means something is wrong.
2. **No `q=4` constant may appear in the verdict or the deck.** See the banned
   list above.
3. **Whole-instance solve rates are not comparable across ensembles.** A solve
   needs ~`N(1−ρ)` consecutive correct deductions and `n_blank` differs: ~82 at
   `d4q3 N=96, ρ=0.15` against ~65 at `d5q4 N=96, ρ=0.32`. Normalise to
   `a_eff = solve_rate^(1/n_blank)` before any cross-ensemble sentence.
4. **`k=0` in any small-k eval is degenerate** (it tests only the terminator)
   and is excluded from means, as in the N=192 write-up.
5. **This stage cannot re-open a closed knob.** Width, cosine decay, more data,
   best-val selection, role count, schedule length and training budget stay
   closed. If stage DQ nulls, the answer is not "train it longer".

## What a null licenses

If D1 lands at chance and D4 falsifies, the honest conclusion is that the
colour-head collapse tracks **depth**, not `N`, and the lever does not avoid
it — in which case the size ladder and the depth lever have the same wall, and
the next move is (f), the trace-format change, which is hshunt's call and not
an agent's. **It does not license a bigger model, a longer schedule, or a
fifth rung.**
