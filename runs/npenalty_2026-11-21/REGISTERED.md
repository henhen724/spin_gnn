# Stage NPENALTY — where does the N-penalty actually live?

2026-11-21. Analysis only: **no training, no new checkpoint, no closed knob.**
Every number below comes from CSVs already on disk — the N=48 and N=96 fall
scans' `rho_curve_*.csv` (the same 14-point, 400-per-bin eval the whole
programme uses) and stage ENTROPY's per-section teacher-forced CSVs.

## The question

hshunt's metric today is the per-deduction rate at matched DD depth,
$a_\text{eff}$ at $T = 12$, which stands at **0.9940** (N=48 s0) against
**0.9122** and **0.9628** (N=96 s0, s1). The brief was to narrow that gap.
Before spending GPU hours, this stage asks what the gap is made of.

## What was tested, and what each outcome would mean

**Q1 — is the matched-$T$ gap a property of the model, or of the normaliser?**
At matched $T$ the larger graph has roughly twice as many blank cells, and
$a_\text{eff}$ falls with the number of cells emitted. Re-matching on
$n_\text{blank}$ instead of $T$ separates the two.

**Q2 — is the failure error propagation (exposure bias)?** $a_\text{eff}$ is a
free-running rollout. If teacher-forced accuracy on the same tokens is much
higher, the model knows the rule and loses it by conditioning on its own
output. If the two agree, it simply does not know the answer.

**Q3 — which tokens carry the loss?** The deduction trace has three decisions:
which node is forced next (`reason_node`), whether the round has ended
(`round_mark`), and what colour the named node takes (`reason_color`). These
are separable in the stage ENTROPY CSVs.

**Q4 — does depth narrow it?** Depth is the one capacity lever not on the
closed list (width, cosine decay, more data, best-val selection, role count,
schedule length, training budget are all closed). N=96 already has L08, L12 and
L16 cells at seed 0 and an L12 cell at seed 1, all on the same eval.

## The bar for claiming depth helps

$s^{*}_{96} = 0.119$ on $\rho_{50}$ is the published seed yardstick, and the
same two cells differ by **0.0506** in $a_\text{eff}$ at $T=12$. **A depth
effect counts only if it exceeds the seed spread measured on the same
quantity.** This is recorded before the comparison is read, so a gain inside
the noise cannot later be reported as an improvement.

## Not tested here, and why

Decoding was checked and is already greedy
(`train_sudoku_transformer.generate`, `temperature=0.0` → `argmax`), so there
is no inference-time lever to find.

Any change to the trace format — so that the model is not required to produce
the exhaustive forced set in one shot — is a change to the task representation.
It is **put to hshunt, not assumed**, in the same way the sbatch records
$(\text{degree}, n_\text{colors})$ as a registered follow-up rather than a
silent substitution.
