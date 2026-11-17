# Stage LOGIC — registered before the job was submitted

**2026-11-17.** Written and committed before `sbatch`. Nothing is trained; every
model is an existing checkpoint and every instance is read from stage POS's
cache, so the eval is point-for-point the one the rho curves, stage POS and
hshunt's power-law fit were measured on.

## The question

Stage POS's measurement A gave hshunt a much cleaner power law than the one
built on final validation accuracy. At each fixed density the solve rate is a
power of the teacher-forced reasoning-trace token accuracy `p`,

$$S(\rho) \;=\; C\,p^{\,k},\qquad C \approx 1 \ \text{for}\ \rho \ge 0.65,
\qquad k \approx 0.73\,n,$$

with `n = 48(1 - realized_rho)` the number of blank cells; a free power fit of
`k` against `n` gives `n^1.07`, i.e. **linear in n, not quadratic**, once the
exponent is measured against teacher-forced token accuracy rather than against
final validation accuracy. (Stage POS's own n^2.09 result is about the
free-generation *hazard*, a different quantity; these are not in conflict and
the deck will say so.)

The coefficient is 0.73 and not 1.0, and hshunt's reading is that it is a
scoring artifact: **many teacher-forced node errors are harmless.** DD
deductions inside a round are order-free, so when the model names a different
*forced* node than the one the ground-truth trace writes next, measurement A
scores it wrong although it is a perfectly legal DD move.

If every logical error were fatal and only logical errors were counted, the
exponent would have to be `k = n` exactly — one chance to fail per deduction,
each failure fatal — with prefactor 1. **The gap between 0.73 and 1 is then a
measurement of how much of A's error budget is bookkeeping rather than
reasoning**, and that is what stage LOGIC measures.

## Population — the same 80 runs as stage POS

`stagepos_cells.successful_runs()`: the **79** (cell, seed) runs whose solve
rate at ρ = 0.50 is nonzero, plus `full L=2 d256 s0` measured and flagged
`in79=False`. The script **raises** if the count is not 79. Every fit is on the
79; the L2 run is measured and plotted but never pooled.

Eval: the 14-point grid 0.95 … 0.33, 400 per bin, `seed_start` 60,000,000,000,
`fall` / `shuffle_adj` per checkpoint. **The instances are not rebuilt** — they
are read from `runs/stagepos_2026-11-17/instances/`, so an L curve and a stage
POS hazard curve are the same 5,600 instances per run.

## Definitions, committed now

**The forced set.** At a node position, `F` is the set of nodes DD forces in
the current round, computed from the hints plus all **completed earlier
rounds** (the usual synchronous round semantics — same-round siblings are not
evidence), minus the nodes already emitted in this round.

Measurement L reads `F` off the trace: `sudoku_trace`'s round `r` *is* the set
of nodes forced by hints plus rounds `< r`, so `F` at a node position is the
rest of the current round. **That identity is checked, not assumed** —
`_stagelogic_dry_check` recomputes the forced set from scratch at the start of
every round and demands set equality. It passed on 141 rounds across ρ = 0.85,
0.60 and 0.40.

**Logically correct**, teacher-forced on the true prefix:

| position | rule |
| --- | --- |
| node | argmax is a node token whose node is in `F` |
| colour | argmax colour equals the DD colour of the node in the true prefix (unchanged from A) |
| BEGIN_ROUND / terminator | argmax is that token |

At the end of a round `F` is empty and `BEGIN_ROUND` (or `SOLVED`) is the only
legal continuation, so the boundary rule is both A's and the logical one. **The
one exception** is the very first position of the section, which predicts the
first `BEGIN_ROUND` while `F` is *not* yet empty; the grammar still forces
`BEGIN_ROUND` there, it is scored that way, and it is emitted as scheme
`round` index 0 so it can be dropped from any re-binning.

**The argmax classes at a node position**, which is where the guard failures
are:

- `in_F` — a node DD forces right now. The logical hit.
- `premature` — a node token, not in `F`, **not yet coloured**. The model has
  named a cell it cannot yet know. A guard failure.
- `recolor` — a node token for a node already coloured (a hint, or emitted
  earlier in the trace).
- `nonnode` — not a node token at all.

**The accuracies.** `p_logic` and `p_strict` are both computed over the **same**
positions, so the logical and A rules are never compared across different
denominators:

$$p_{\text{logic}} = \frac{\#\{\text{logically correct}\}}{\#\{\text{reasoning-section positions}\}},
\qquad
p_{\text{logic,node}} = \frac{\#\{\text{logically correct node positions}\}}{\#\{\text{node positions}\}}$$

and `p_strict` the same with "argmax equals the true token". Node, colour and
boundary counts are saved separately, so any other denominator is a
re-binning, not a re-run.

**The free-generation check.** Stage POS's B found 99.7% of first errors name
an *uncoloured* node with the *wrong* colour. Two readings survive that: the
model picked a node DD had already forced and mis-coloured it, or it picked a
node DD had not forced and so could not have known the colour of. B did not
save **which** node was named, so this is **not** answerable from its CSVs; it
is answered by **replaying** the rollout — same greedy decode, same cached
instances, same batch size, bit-identical by construction. At the first error
the model's own prefix is still perfect, which is exactly what makes DD
runnable on it. Recorded: `err_in_F` (round-start rule, the registered one),
`err_in_F_now` (the model's full current state, same-round siblings included —
a second column, never a substitution), `|F|`, whether the node was uncoloured,
and whether the colour was right. `in_F` is also recorded for every pair up to
and including the first error, so "unforced but got away with it" has a
denominator.

**The replay is verified, not trusted:** the replayed solve rate must equal
measurement B's to the digit on all 1,120 (run, ρ) cells. A mismatch means the
rollout is not the same rollout and the comparison is void.

## Predictions, and what would overturn them

**L1 — hshunt's prediction.** Fitting `log S` on `log p_logic` per density over
the 79 `in79` runs with `0 < S < 1` gives an exponent close to `n` and a
prefactor close to 1. Registered band: the slope of `k_logic` against `n` lands
in **[0.85, 1.20]** and the median prefactor over ρ ≥ 0.65 lands in
**[0.5, 2.0]**.
**FALSIFIED** by a slope outside [0.7, 1.4].

**L2 — the identity.** `p_logic ≥ p_strict` at every (run, ρ). The logical rule
can only forgive, never create, errors; the dry check already proves the true
node is always in `F`. A violation is a **bug, not a finding**, and is reported
as one.

**L3 — how much of 0.73 is bookkeeping.** The median of `k_logic / k_strict`
over ρ ≥ 0.50 lands in **[1.1, 1.6]**, i.e. removing the harmless node errors
moves the exponent most of the way from 0.73n to n.
**FALSIFIED** below 1.0 (the logical rule would have made the fit *worse*) or
above 2.0.

**L4 — a majority of first errors name an unforced node.** Over the `in79` runs
at ρ = 0.50 … 0.75, the median fraction of first errors whose node is not in
`F` exceeds **0.5**.
**FALSIFIED** below 0.5, which would say the model picks the right cell and
mis-colours it, and would make "guard failure" the wrong name for this.

**L5 — the confound I expect to have to report.** `premature` is a *rate per
position*, and the number of uncoloured nodes outside `F` grows as the
instance gets sparser, so a premature rate rising with ρ falling is partly a
statement about how many wrong answers are available. Every row therefore
carries `|F|` and the node count, and the premature rate is reported both raw
and against `(N - coloured - |F|)/N`. If only the raw version trends, the
honest statement is "there are more ways to be premature", not "the guard is
weaker".

## Figures registered before the job ran

1. **L1 — `p_logic` against `p_strict`**, one point per (run, ρ), faceted by ρ,
   with the identity line. Shows how much A's rule was over-penalising.
2. **L2 — `log S` against `log p_logic`**, small multiples over ρ, one point
   per run, fitted line per panel, strict fit shown as a muted reference.
3. **L3 — `k_logic` and `k_strict` against `n`**, with the `k = n` line and
   hshunt's `k = 0.73n`. The headline figure.
4. **L4 — prefactor `C` against `n`**, logical and strict, with `C = 1` marked.
5. **L5 — `R²` of both fits against `n`.** Which rule gives the cleaner law.
6. **L6 — logical and strict node accuracy against deduction index `j`**, one
   line per model, three density panels.
7. **L7 — the node-position argmax class breakdown against `j`** (`in_F`,
   `premature`, `recolor`, `nonnode`), stacked, at one density.
8. **F1 — fraction of first errors naming an unforced node, against ρ**, one
   line per model, with the per-run point count.
9. **F2 — `in_F` rate per pair index `j` in free generation**, the
   unforced-but-survived rate, three density panels.
10. **F3 — the same-eval comparison:** replayed solve rate against stage POS's
    measurement-B solve rate, all 1,120 cells. Must be the identity line.

## Outputs

`runs/stagelogic_2026-11-17/`

- `L_position_<tag>.csv` — per (run, ρ, scheme, index, token type):
  `n_total`, `n_logic_correct`, `n_strict_correct`, `n_in_F`, `n_premature`,
  `n_recolor`, `n_nonnode`. Schemes `j`, `round`, `all`.
- `F_hazard_<tag>.csv` — per (run, ρ, pair index `j`): `n_at_risk`, `n_in_F`,
  `n_err`, `n_err_in_F`, `n_F_sum`.
- `F_errors_<tag>.csv` — one row per instance **that errs**: the named node,
  its round, `err_in_F`, `err_in_F_now`, `|F|`, `err_uncolored`,
  `err_color_correct`, plus `first_error_j`, `first_kind`, `solved`.
- `LF_summary_<tag>.csv` — one row per (run, ρ): `p_logic`, `p_strict`, the
  node-only and colour-only versions, the four class counts, `n_first_err`,
  `n_err_unforced`, `frac_unforced`, and the replayed `solve_rate`.
- `figures/`, `VERDICT.txt` from `scripts/stagelogic_fit.py`.

## Job layout

One array, `scripts/slurm/stagelogic_2026-11-17.sbatch`: sganguli, 1 GPU,
4 CPUs, 48 GB, `--time=03:00:00`, `--array=0-9%5`, round-robin shard
`LOGIC_SHARD="${SLURM_ARRAY_TASK_ID}/10"` over all 80 runs (8 per task). Round
robin, not contiguous, because the run list is sorted by depth. **No
dependency**: the instances already exist and are only read. Every task runs
the dry check first — it is seconds of CPU and it is what catches a stale
checkout — and it already passed on CPU before submission, together with a
2-run smoke of the scorer itself.

Nothing is applied and no verdict is printed by the measurement.
