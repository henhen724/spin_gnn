# Stage POS — registered before the job ran

Written and committed **before** `sbatch`. Nothing here is edited afterwards;
what the measurement actually shows goes in the deck and in `STATUS.md`, next
to this.

Run directory: `$SCRATCH/spin_gnn/runs/stagepos_2026-11-17/`
Scripts: `scripts/stagepos_cells.py`, `scripts/stagepos_score.py`,
`scripts/_stagepos_dry_check.py`
Nothing is trained. Every model is an existing checkpoint, evaluated only.

## The claim being tested

hshunt's background fit over the 15-directory minarch scan: at fixed density,
solve rate is close to a power law in final validation token accuracy,
$S \sim \mathrm{acc}^{k(\rho)}$, $R^2 \approx 0.92$ for $\rho = 0.50$–$0.75$,
with $k$ = 2.0, 4.6, 19.9, 38, 69, 92 at $\rho$ = 0.95, 0.90, 0.80, 0.70,
0.60, 0.50. Against the number of blank cells $n = 48(1-\rho_{\rm real})$ the
exponent scales as $k \sim n^{1.84 \pm 0.03}$ for $\rho \ge 0.50$.

Independent per-step failure at a constant rate $p$ gives
$\log(1/S) = n\log\frac{1}{1-p}$, **linear** in $n$. A per-step failure rate
that rises linearly with position, $p_j \approx b\,j$, gives
$\log(1/S) \approx \sum_j p_j \approx b n^2/2$, **quadratic**. The measured
1.84 is close to 2.

**Hypothesis: the per-step error rate grows roughly linearly with position in
the reasoning trace, so total failure goes like $n^2$ rather than $n$.**

## Population

`stagepos_cells.successful_runs()`: the **79** (cell, seed) runs from
`ALL_DIRS` in `notebooks/make_minarch_web.py` whose solve rate at $\rho=0.50$
is nonzero — the same population the power law was fitted on. The script
**refuses to run** if the count is not 79.

Plus **one flagged extra**, `full L=2 d256 seed 0` (`in79 = False`): hshunt
named it by hand and it has solve rate 0.000 at $\rho=0.50$, so it fails the
filter. It is measured because it is the only cell that collapses *above*
$\rho = 0.50$ and is therefore where a hazard curve should be steepest. It is
excluded from everything fitted against the 79.

80 runs total. Eval: the rho curves' own 14-point grid 0.95→0.33, 400 per bin,
`seed_start = 60_000_000_000`, `fall` and `shuffle_adj` from each checkpoint's
config via `cfg_from_checkpoint`. All 80 share one instance distribution
(checked), so the instances are built once and reused — every curve is
comparable point by point.

## What will be plotted

Registered now so the figure list cannot be chosen after seeing the numbers.
One line per run wherever curves are overlaid; the cell key and seed are on
every CSV row.

1. **A1** — teacher-forced next-token accuracy vs token offset from
   `BEGIN_REASONING`. Small multiples, one panel per named run, three density
   lines ($\rho$ = 0.75, 0.60, 0.50).
2. **A2** — the same against deduction index $j$, **colour tokens only** and
   **node tokens only** on separate panels.
3. **A3** — teacher-forced accuracy vs round index, same layout.
4. **B1** — hazard $h(j)$ vs $j$, one panel per named run, three densities.
5. **B2** — hazard vs $j$ at $\rho = 0.60$, **one line per run**, all 79
   overlaid and coloured by validation token accuracy.
6. **B3** — survival $S(j) = \prod_{i<j}(1-h(i))$, same layout as B2.
7. **B4** — $\sum_j h(j)$ against $n$, log–log, all 79, with an $n^{1.84}$
   reference line. **This is the quantitative test.**
8. **C1** — hazard against position *within* a round (the P4 confound).
9. **C2** — hazard against $j/n$ (the P5 confound).
10. **D1** — this run's free-generation solve rate against the published rho
    curve's solve rate on the same eval, per run. The agreement check, and the
    comparison against the previous best on the same eval.

## Predictions, and what overturns them

**P1 — the hypothesis.** Hazard rises with $j$: Spearman $\rho_s(j, h(j))$
over $j$ with $n_{\rm at\ risk} \ge 100$ is positive at every density in
0.50–0.75, for ≥5 of the 7 named runs and for a majority of the 79.
*Falsified* by a flat ($|\rho_s| < 0.3$) or decreasing hazard on the majority
— which would mean the $n^2$ exponent comes from something other than
position-dependent error.

**P2 — the quantitative link.** (a) Arithmetic: $\prod_j (1 - h(j))$ must
reproduce the measured solve rate (an identity; a mismatch is a bug, not a
finding — already verified exactly on the CPU smoke). (b) The real prediction:
fitting $\sum_j h(j)$ against $n$ over $\rho \ge 0.50$ gives a slope in
$[1.5, 2.3]$, consistent with $1.84 \pm 0.03$. *Falsified* outside
$[1.3, 2.5]$.

**P3 — what A adds.** Under a perfect prefix the per-token error *also* rises
with position: teacher-forced colour-token accuracy at the last deduction is
below that at the first, on ≥5 of the 7 named runs at $\rho = 0.60$.
**I expect the opposite**, and that is the more interesting reading: A flat
while B rises would say the position dependence is entirely error
*accumulation* — the model corrupting the prefix it then reads — not an
intrinsic difficulty gradient. It will be reported that way, not as a failed
prediction.

**P4 — the confound that gets reported either way.** Later deductions sit in
later rounds, so "position in the trace" and "DD depth" are confounded by
construction. Every row carries the round index and hazard is reported both
against $j$ and against position *within* a round. If only the across-round
trend survives, the finding is "error grows with DD depth", not "with token
position", and that is what will be written.

**P5 — the second confound.** At fixed $\rho$, larger $j$ also means fewer
blanks left, so the remaining deductions are not drawn from a fixed difficulty
distribution. Hazard is therefore also reported against $j/n$.

## Definitions the measurement commits to

Reasoning-section grammar is `BEGIN_REASONING (BEGIN_ROUND (NODE COLOR)+)*
(SOLVED | UNRESOLVED NODE*)`, so the trace is $2n + T + 1$ tokens — checked in
the dry check, not assumed.

**Measurement A, teacher-forced.** One forward pass per instance over the
ground-truth sequence; at each reasoning position, is the next-token argmax
the true token given a *perfect* prefix. Indexed three ways (`offset` from
`BEGIN_REASONING`, deduction index `j`, `round`) and split by token type
(`node`, `color`, `round`, `term`). Raw correct/total counts are saved per
density so any re-binning is possible later.

**Measurement B, free generation.** Greedy rollout from the prompt through
`BEGIN_REASONING`, identical in decisions to
`train_sudoku_transformer.generate` (verified token-for-token against it in
the dry check; batched only for speed). For the $j$-th emitted pair:

- `node_valid` — a node token for a node not already coloured. "A legal
  move", **not** "the move the ground truth made": DD deductions within a
  round are order-free, so a different order is not an error.
- `color_correct` — the colour equals the planted colour of the node the model
  actually named.
- `matches_true`, `color_proper` — recorded, never used in the error
  definition, so the ordering and local-legality questions stay answerable.

$$\text{error at } j := \lnot(\text{node\_valid} \wedge \text{color\_correct}),
\qquad h(j) = \Pr[\text{first failure at } j \mid \text{none before } j].$$

A rollout can also fail without a bad pair — `SOLVED` before all $n$ blanks,
`UNRESOLVED` on a resolvable instance, or running out of budget. Those are
counted in a separate `n_stop` column at $j$ = pairs emitted, so a model that
quits early cannot look error-free, and `hazard_err` (pair errors alone) stays
separable from `hazard` (both).

## Outputs

`A_position_<tag>.csv`, `B_hazard_<tag>.csv`, `B_instances_<tag>.csv`,
`B_summary_<tag>.csv`, one shard per job/array task, every row carrying
`run_id`, `cell`, `key`, `seed`, `arch_mode`, `num_layers`, `d_model`,
`n_trainable_params`, `val_acc`, `in79`.

## Job layout

Two jobs on `sganguli`, 1 GPU each, explicit time limits, no duplicates
(`seven` and `rest` are disjoint and the 10 array shards cover `rest` exactly,
checked).

1. `stagepos_seven_2026-11-17.sbatch` — dry check, then A+B on the 7 named
   runs. Also warms the shared instance cache. `--time=03:00:00`.
2. `stagepos_rest_2026-11-17.sbatch` — `--array=0-9%5`, A+B on the other 73,
   `--dependency=afterok` on job 1. `--time=04:00:00` per task.

## Nothing is applied

No verdict is printed by the scorer. The exponent fit, the error definition
and whether anything gets re-run stay hshunt's call.
