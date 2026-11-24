# Stage RULE8 — registered 2026-11-24, before the scorer existed

Eighth refit of `runs/stagelogic_2026-11-17`. **Nothing is trained and nothing is
submitted.** No closed knob is reopened: this stage adds no architecture, no
width, no depth, no seed and no epoch.

## The object

The governing answer under the 2026-11-23 brief is
**`L=16 h=4 tied_no_mlp d320`** — one iterated operator $h_{t+1} = F(h_t)$,
613,818 parameters, 3 seeds. Its metric is

    h_rule = sum_j n_err_in_F / sum_j n_in_F = Pr(colour wrong | named a forced cell)

pooled over all 14 densities and all deduction indices $j$ — **the stage RULE2
eval**, on which it reads **0.00002 from 2 rule errors in 80,956 forced
emissions** (STATUS finding 122). The interval this line currently quotes for it
is Wilson 95% on those counts.

Today's question is the **interval**, not the point estimate. The point estimate
cannot move: the same eval on the same shards gives the same two counts, and
anything else would be a defect, not a result (E5 below). What this stage adds is
a **delete-one-run jackknife** over the cell's 3 seeds — each run dropped whole,
so anything shared within a run (seed, checkpoint, data order) sits inside the
resampled unit instead of being assumed away.

## Hypothesis, stated before the scorer was written

**The jackknife will not tighten this interval; it will widen it, and it will
stop excluding zero.** Three seeds give 2 degrees of freedom against a numerator
of 2 events. A resampling interval cannot manufacture information that 2 events
do not contain, and dropping whichever run carries an error moves the estimate by
a whole event. The useful thing the jackknife can still do is tell us whether
**seed is an extra variance component here or not** — if its standard error sits
on top of the Poisson standard error of the same 2 events, then 3 seeds buy
nothing beyond the count, and the only way to tighten this number is more
rollouts or more seeds, both of which are training and hshunt's call.

Registering the prediction that the requested procedure fails is the point of
registering. If it tightens instead, E1 misses and that is the finding.

## Scored predictions

Bars are fixed here and are not to be moved after a number is read.

- **E1 — the jackknife interval is wider than Wilson's.** Bar: width ratio
  (jackknife / Wilson) $\geq 1.0$ on the project's existing $Z = 1.96$
  convention. HIT if wider, MISS if it tightens.
- **E2 — the jackknife interval includes zero while Wilson's does not.** Bar:
  jackknife lower bound $\leq 0$ **and** Wilson lower bound $> 0$. The
  scientifically load-bearing bound on this cell is the *lower* one — it is what
  distinguishes "keeps DD's rule" from "makes no rule errors at all" — so an
  interval that cannot exclude zero is less informative however narrow it looks.
- **E3 — the two errors are concentrated in at most 2 of the 3 seeds.** Bar:
  at least one seed has `n_err_in_F = 0`. This is the mechanism for E1 and E2,
  and is registered separately so it can fail on its own.
- **E4 — seed is not an extra variance component on this cell.** Bar: the
  jackknife standard error over the 3 runs sits within a factor of 2 of the
  Poisson standard error of the same event count,
  $0.5 \leq SE_{\text{jack}} / SE_{\text{Poisson}} \leq 2.0$ with
  $SE_{\text{Poisson}} = \sqrt{k} / n_F$. If it holds, the 3 seeds add no
  dispersion beyond counting noise and the interval is a *count* problem, not a
  *seed* problem.
- **E5 — reproduction, at zero residual.** The stage must reproduce stage RULE2's
  published figures for this cell on the same eval: 2 rule errors, 80,956 forced
  emissions, `h_rule` 0.00002 over all 14 densities, and 0.00003 / 63,213 over
  the ten fitted densities. Bar: residual $= 0$ against recomputation
  (tol 1e-12). A miss here voids everything above it.

## Diagnostics — any failure stops the stage

- **D1 — the cell is resolved by its label, never by `(mode, L, d)`.** The triple
  `('tied_no_mlp', 16, 320)` covers **three different cells** on this roster —
  `d320`, `d320 k2` and `d320 k4` (RULE7 D5). The jackknife must run over the
  3 runs of `d320` alone. Bar: exactly 3 runs selected, and the `k2`/`k4` runs
  absent from the selection.
- **D2 — the per-seed counts sum to the pooled counts.** Bar: residual 0 on both
  `n_in_F` and `n_err_in_F` (tol 1e-9).
- **D3 — no seed has a density with an empty denominator.** Every (seed, density)
  pair must carry `n_in_F > 0`, so the pooled rate is not an average over bins
  one of which contributes nothing. Bar: 0 empty pairs over 3 × 14 = 42.
- **D4 — column semantics.** `n_err_in_F <= n_err` on every row of this cell
  (finding 124: `n_err` counts a wrong **colour**, forced or not). Bar: 0
  violations.
- **D5 — the pooling window is the one being claimed.** `GRID` has 14 entries and
  the ten-density value must come from `FIT_RHOS`, not from a re-typed list.
  Bar: `len(GRID) == 14`, `len(FIT_RHOS) == 10`, and both values reproduce E5's
  figures.

## What this stage may not do

It may not seed `tied_no_mlp d192`, submit the width sweep between `d192` and
`d320`, or submit stage DQ4. It may not quote an improvement: the point estimate
is identical before and after by construction, and only the interval is new.
