# Stage NSPLIT — where does the N=192 excess live?

Registered 2026-11-20, **before the N=192 checkpoint exists**. Job 53062385 is
at epoch 29 of 50 and the sweep writes `model.pt` only at the end, so no number
below can have been peeked at.

## Why this exists, and a correction to the record

Finding 109 put the `val_nll` floor at 1.371 / 1.625 / 1.861 nats for
N = 48 / 96 / 192. **TB1** registered that the *excess* above that floor would
stay in [0.09, 0.15] at N=192, because it had been stable across a 2x in N:
0.113 (N=48 t08), 0.121 (N=96 t00), 0.103 (N=96 t04).

**TB1 is falsified.** N=192 t00 sits at excess 0.2864 at epoch 29 and is
descending at 0.00079/epoch, so it lands near 0.265.

**I previously attributed that to constant LR. That attribution was wrong and
is withdrawn.** It rested on a misreading of STATUS: the 0.103 figure belongs
to N=96 `t04`, a *fall-scan* cell trained at constant LR, not to a cosine run.
The real cosine-vs-constant comparison at N=96 is:

| cell | schedule | best val_nll | excess |
|---|---|---|---|
| n96 t00 | constant | 1.7455 | +0.1205 |
| n96 t04 | constant | 1.7283 | +0.1033 |
| n96 t00_cos | cosine | 1.8033 | +0.1783 |
| n96 t01_cos | cosine | 1.7435 | +0.1185 |

Cosine is no better than constant at N=96, and on one seed clearly worse. So a
cosine-decayed N=192 rerun is **not** the indicated test and is not proposed.

What the curves actually say, at matched epochs (both L12 d256, 240k x 50ep,
constant LR, seed 0):

| epoch | N=96 excess | N=192 excess |
|---|---|---|
| 9  | +0.2065 | +0.3274 |
| 14 | +0.1558 | +0.3040 |
| 19 | +0.1498 | +0.2943 |
| 24 | +0.1482 | +0.2895 |
| 29 | +0.1425 | +0.2864 |

The *slope* is the same — -0.00074/epoch at N=96 against -0.00079/epoch at
N=192 over epochs 19-29 — but there is a near-constant ~0.14 offset present
from epoch 14 onward. This is not an optimisation failure and not a schedule
failure. It is a different asymptote.

## The question

Of that ~0.14 of extra excess, how much is the model getting *deduction*
wrong, and how much is finding 109's floor model under-predicting the graph
section's true cost at N=192?

**The deduction sections have a data floor of exactly zero at every N.**
Within-round emission order is sorted by node index (`sudoku_trace.py:83`) and
the colour is forced, so `reason_node`, `reason_color`, `round_mark` and `term`
are deterministic given the prefix. Stage ENTROPY confirmed this empirically:
0.0219 and 0.0000 nats at the best N=48 cell. Therefore **any** NLL on those
sections is pure excess and needs no floor subtraction. That is the handle.

Write $w_s$ for a section's share of tokens and $\mathrm{NLL}_s$ for its mean
loss, so the pooled loss is $\sum_s w_s \mathrm{NLL}_s$. Define

- `ded(N)`   = $\sum_{s \in \mathrm{DET}} w_s \mathrm{NLL}_s$ — pure excess, floor 0
- `exc(N)`   = $\sum_s w_s \mathrm{NLL}_s - \mathrm{floor}(N)$ — total excess

Both measured by the probe at matched densities, so the probe's own density
mix cancels in the difference. These will NOT equal the sweep's 0.144, which
is over a different val mix; the registered claims are on the probe's numbers.

## Predictions

| # | prediction | bar | falsifier |
|---|---|---|---|
| TA1 | DEDUCTION account: deduction share of the excess growth, $\Delta$`ded` / $\Delta$`exc` | $\geq 0.60$ | $\leq 0.20$ |
| TA2 | FLOOR account (complement of TA1): non-deduction share | $\geq 0.60$ | $\leq 0.20$ |
| TA3 | `graph_nbr_novel` contribution grows by $(5/16)\log 2 = 0.2166$ as predicted by 109 | within $\pm 0.05$ | outside [0.10, 0.33] |
| TA4 | deduction NLL at N=192 exceeds N=96 at the matched L12_s0 cell | strict | fails |

TA1 and TA2 are mutually exclusive by construction — exactly one can hit, and
if both miss the excess is split and neither account is sufficient. That is the
registered discriminator.

TA4 is the weak consistency check against stage ENTROPY's E7, which already
found reasoning *entropy* rising at all three matched pairs from N=48 to N=96.
If TA4 fails, something is wrong with the N=192 checkpoint, not with the
accounts.

## Diagnostics

| # | check | bar |
|---|---|---|
| DN1 | reconstructed $\sum_s w_s \mathrm{NLL}_s$ matches the probe's pooled NLL | < 1e-6 |
| DN2 | $0 \leq$ NLL at every row, section shares sum to 1 | 0 violations |
| DN3 | the N=96 cell reproduces stage ENTROPY's numbers for the same cell | < 1e-3 |

DN3 is the important one: the N=96 L12_s0 cell is measured by both stage
ENTROPY and this probe, so it is a free cross-check that the N=192 extension
did not perturb the existing path.

### Amendment, 2026-11-21: DN3 is scored at matched instance count

As written, DN3 did not say how many instances the two sides use, and they did
not use the same number. Stage ENTROPY ran with `--limit 100`; this probe
defaults to the whole cache, which stage SCALE had already grown to 200 per
density bin. The first run of the cross-check (job 53203618's predecessor,
53203355, 200 instances) gave **max |diff| 3.157e-03 against the 1e-3 bar —
FAIL**.

That failure is recorded here rather than removed. The diagnosis: a per-row
comparison showed the largest deltas on `begin` (6.3e-02, a section the scorer
excludes) and `round_mark` (2.0e-02), with an `n_instances` column reading 200
on one side and 100 on the other and 389,960 more tokens in total — the
signature of extra sampling, not of a different code path.

Re-running the identical cell with `--limit 100` gives **max |diff| exactly
0.000e+00**: bit-identical to stage ENTROPY, not merely inside the bar. So the
extension demonstrably did not perturb the existing path, which is the only
thing DN3 was ever asking.

**The amendment is a clarification, not a loosening: the 1e-3 bar stands
unchanged, and DN3 is scored on the matched-count arm** written to
`dn3_limit100/`. The science cells keep the full 200 instances, since more data
is strictly better and the two counts move the reported split by ~2e-4
(pooled 1.7483 vs 1.7485, ded 0.0246 vs 0.0245, exc 0.1233 vs 0.1235) — far
below anything TA1–TA4 turn on.

## Method

Reuses `scripts/stageentropy_probe.py` unchanged apart from the cell list.
`scripts/stagescale_probe.py:run_path` has been extended with a 192 branch
(`SCAN_192`); 48 and 96 resolve exactly as before, and an unknown N now raises
instead of silently falling back to SCAN_96. `section_labels` already reads
`degree` from the instance, so no new encoding assumption is introduced.

Cells: `n96_L12_s0` (exists) and `n192_L12_s0`, `n192_L12_s1` (pending
53062385). Inference only, CPU, no training, no closed knob.

**Blocked until 53062385 writes its checkpoints.** Do not submit before then.

2026-11-21: submitted as **53203962** with `--dependency=afterok:53062385`,
held `PD` until the whole ladder array exits 0. The block above exists because
`resolve_inner` would raise on a missing `model.pt`; an `afterok` hold makes
that state unreachable, and `afterok` rather than `afterany` so a crashed
ladder cannot launch the probe onto a partial checkpoint. Chaining also avoids
polling for the checkpoint, which AGENTS.md rule 10 asks for directly. Nothing
about the predictions or the cell list changed.
