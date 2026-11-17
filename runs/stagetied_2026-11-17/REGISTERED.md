# Stage TIED — registered before looking at anything

Written and committed **before** the scorer was run. No number below was chosen
after seeing it. Serves the restated brief: *the smallest or most analytically
tractable architecture that still learns direct deduction.*

## The question

**S1** — `tied_no_mlp T=16 d320`, 613,818 trainable — is the most analytically
tractable object on the board: **one shared block applied 16 times**, i.e. an
iterated map rather than 16 distinct layers. It is held out of the gate by a
hair: `rho_50` 0.757 against the free line 0.740, which finding 62 converts to
**≈ 0.011 of validation token accuracy** against **J2** (`no_mlp L=16 d192`,
2,504,890 trainable, `rho_50` 0.580, the standing minimum).

STATUS records that gap as open with **every registered training knob closed**:
width, cosine decay, more data, best-val selection, role count, schedule
length, training budget.

Finding 94 now supplies a reason to re-examine the gap itself rather than
another knob. Validation token accuracy is **strict**: it scores the argmax
against the reference trace, which writes each round's cells in sorted node
order even though direct deduction fixes no order within a round. Stage LOGIC
measured the resulting mis-scoring at a median **+0.0106** of accuracy — the
same order of magnitude as the 0.011 gap S1 is held out by.

**Hypothesis.** Weight tying costs a model the capacity to memorise an
arbitrary *output-ordering convention* while leaving the deduction intact. If
so, part of S1's measured deficit is not a deduction deficit at all, and the
strict scorer has been overstating how far the tractable architecture is from
the untied one.

## Inputs, population and conventions

Every input is a CSV already on disk. **No training, no GPU, no new
instances.** `runs/stagelogic_2026-11-17/LF_summary_all_*.csv`, per (run, ρ):
`arch_mode`, `num_layers`, `d_model`, `seed`, `val_acc`, `in79`, `p_logic`,
`p_strict`, the `_node` and `_color` splits, `node_in_F`, `node_premature`,
`node_recolor`, `node_nonnode`, `solve_rate`.

- **S1** = every `in79` run with `arch_mode = tied_no_mlp`, `num_layers = 16`,
  `d_model = 320`. **J2** = every `in79` run with `no_mlp`, 16, 192.
- The ten densities ρ = 0.95 … 0.50 where a fit is defined, as in stages LOGIC
  and PREFACTOR. Seeds are kept separate and the **spread is quoted**, never a
  bare mean.
- `gain = p_logic − p_strict`, always at matched ρ. Group values are medians
  over the runs in that group at that density.

## Predictions

**T1 — tying is penalised more by strict scoring.** The strict→logical gain is
larger for S1 than for J2.

*Prediction:* median `gain(S1) − gain(J2) > 0.002`, and `gain(S1) > gain(J2)`
at a majority of the ten densities. *Falsified if* `gain(S1) ≤ gain(J2)` at a
majority of densities.

**T2 — the gap narrows under logical scoring.** Write the strict gap
`Δ_strict(ρ) = p_strict(J2) − p_strict(S1)` and likewise `Δ_logic`.

*Prediction:* `Δ_logic < 0.5 · Δ_strict` at a majority of densities with
`n ≥ 12`. *Falsified if* `Δ_logic ≥ 0.9 · Δ_strict` at a majority — that would
mean the gap is real deduction and mis-scoring is beside the point.

**T3 — it does not fully rescue S1.** The honest prior, registered so that a
partial result cannot be read as a total one.

*Prediction:* `Δ_logic > 0` — S1 is still behind J2 — at a majority of
densities with `n ≥ 12`.

**T4 — the residual deficit is the GUARD, not the rule.** Findings 63–81 found
the shared block has DD's rule and not DD's guard, on an instrument whose rates
were conditional on `clean_emits_v` with denominators 236/228/217. Stage
LOGIC's `node_premature` is an unconditional version of the same thing: the
rate at which the argmax names an uncoloured node that DD does **not** force
yet.

*Prediction:* `node_premature(S1) > node_premature(J2)` at a majority of
densities, and the median ratio exceeds **1.2**. *Falsified if* the median
ratio is below 1.0.

**T5 — the gate does not move, and this stage cannot move it.** Solve rate is
scored against the planted colouring and is therefore **already** invariant to
within-round emission order. Rescoring tokens cannot change it.

*Prediction:* `solve_rate` is bit-identical to what stage LOGIC recorded, so
`rho_50` and the gate verdict for S1 are **unchanged**. Registered explicitly
so that nothing here can later be reported as having cleared the gate.

**D1 — colour positions are a definitional check, not a result.** Logical
colour scoring asks whether the argmax colour equals the DD colour of the node
in the *true prefix*, which on a correct prefix is exactly the reference
colour.

*Expected:* `p_logic_color = p_strict_color` exactly, for every run and
density. A nonzero colour gain means the scorer disagrees with its own
definition and invalidates T1–T2.

## Figures

| | |
| --- | --- |
| `T1_gain_by_density.png` | strict→logical gain against ρ, S1 vs J2, seed spread shown |
| `T2_gap_closure.png` | `Δ_strict` and `Δ_logic` against `n` |
| `T3_premature.png` | `node_premature` against ρ, S1 vs J2 |
| `T4_acc_vs_solve.png` | `p_strict` and `p_logic` against solve rate, both groups |

## Output

`runs/stagetied_2026-11-17/`: `tied_cells.csv` (per run, ρ, group: the accuracy
and class columns), `tied_summary.csv` (per ρ, group: medians and spreads),
`VERDICT.txt`, `figures/`.

Runs on a login node in **under two minutes** — CSV reading, medians and
matplotlib. No GPU, no model, no instance generation. **Nothing is submitted.**
