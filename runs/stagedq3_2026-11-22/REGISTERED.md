# STAGE DQ3 — registered BEFORE submit, 2026-11-22

One knob changes against stage DQ (`runs/stagedq_2026-11-21/`): **the seed, 0/1
→ 2.** Ensemble `d=4 q=3`, `N=96` and `N=120`, `L=12 d256 ff1024`, 240k train
instances, 50 epochs, `--shuffle-adj --fall`, `--rho 0.14 --rho-max 0.95`, the
same 14-bin grid at 400/bin with the default `--seed-start`, so the evaluation
instances are identical point for point to stage DQ's.

## Why this run exists

Stage DQ scored D1 PARTIAL at a pooled 0.2234 nats, and the pooled number is
**not seed-robust** — its own `VERDICT.txt` says so. The per-cell split is:

| cell | `reason_color` | acc | `reason_node` | acc |
|---|---|---|---|---|
| N=96 s0 | 0.4468 | 0.7671 | 0.6356 | 0.7803 |
| N=96 s1 | 0.0000 | 1.0000 | 0.0504 | 0.9851 |
| N=120 s0 | 1.0987 | 0.3297 | 0.6628 | 0.7729 |
| N=120 s1 | 0.0001 | 1.0000 | 1.0507 | 0.6479 |

`1.0987` is `ln 3` to four decimals and `0.3297` is `1/q`; that cell's colour
head is **at the prior**. Two other cells have the rule **exactly**. So D2's
"+0.3260 nats, deeper is worse" is the average of a collapsed cell and a
perfect one, and reads as a depth trend only because the pair was averaged.

**HYPOTHESIS: the colour head's state is a bimodal, seed-dependent training
outcome, not a function of `N` or of DD depth `T`.**

The competing reading, which this run can confirm instead, is that `N=120` is
genuinely past a threshold and `N=120 s1` was lucky.

## Predictions

**E1 — bimodality.** Pooled over the density grid, `reason_color` for each new
cell lands either **< 0.10 nats** (rule learned) or **> 0.99 nats** (within 10%
of `ln 3`, at the prior). **HIT** if both cells fall in one of those two bins;
**MISS** if either lands in `[0.10, 0.99]`. Stage DQ already produced one
intermediate cell (0.4468), so a miss is a live outcome, and a miss means the
state is continuous and "bimodal" is simply the wrong description.

**E2 — `N` is not the driver.** If `N` drove the collapse, every `N=120` seed
would be collapsed. **E2 is FALSIFIED if `reason_color` at `N=120 s2` is
> 0.99**, i.e. 2 of 3 `N=120` seeds at the prior. **HIT if < 0.10**, making 2
of 3 seeds hold the rule at the larger size.

**E3 — search degrades with `N` regardless of the colour head.**
`reason_node` at `N=120 s2` **>** `reason_node` at `N=96 s2`. This is the only
quantity that moved the same way on every stage DQ cell, and it is the one
finding 115 named as the failure mode at matched depth.

**E4 — `round_mark` is the `N`-sensitive zero-floor section.** `round_mark`
at `N=120 s2` **>** at `N=96 s2`, and **both > 0.01**. Registered because
D3's 0.01 bar covered `graph_lead` and `term` but **never covered
`round_mark`**, which was 0.1075–0.5410 on all four stage DQ cells and is the
section that measures "emit the whole forced set this round, then stop".

**E5 — `a_eff` at matched depth, the standing metric.** Report `a_eff` at the
`ρ=0.32` bin (`T≈12`) for both new cells. **HIT only if some cell exceeds
0.99191**, which is `N=96 s1`, the standing best on this eval. A zero-solve
bin is reported as its Clopper–Pearson upper bound (`< p_up^(1/n_blank)`,
`p_up = 1 − 0.05^(1/400) = 0.00746`) and **never as `a_eff = 0`.**

## Traps

1. **No `q=4` constant may be quoted** — not `ρ_c ≈ 0.465`, the free line
   0.740, `s*`, the `val_nll` floors, `(5/16) log N`, nor `ln 4`. Chance here
   is `ln 3 = 1.098612` and chance accuracy is `1/3`.
2. **Pooling across seeds is what produced the misleading D2.** Every E
   prediction is scored **per cell**, and any pooled number is reported beside
   its per-cell parts.
3. **Closed knobs stay closed.** This run re-tests none of width, cosine
   decay, more data, best-val selection, role count, schedule length or
   training budget — only the seed moves. Stage DQ's cells were still
   descending at epoch 50; that is reported, not fixed here.
4. **D5's FLAG on the `N=120` cells fired on a +0.0002-nat ordering** (best at
   epoch 47, not 49), not on flatness — those cells fell 3.06 → 1.74. The
   criterion is reported as it fell and is **not** adjusted for this run.
