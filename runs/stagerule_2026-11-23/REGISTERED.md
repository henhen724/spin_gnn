# STAGE RULE — registered predictions, written BEFORE the scorer was run

A **refit** of stage LOGIC's `F_hazard_all_*of10.csv` and `LF_summary_all_*of10.csv`
in `runs/stagelogic_2026-11-17/`. Seconds on a login node. **NOTHING IS
SUBMITTED AND NO MODEL IS TRAINED.** No new data is generated; every number
below comes from rollouts that already exist.

## The brief this answers

As of **2026-11-23** the goal is *the smallest or most analytically tractable
architecture that still learns direct deduction, not a stronger solver.*

**Reading of "smallest" used here, stated so it can be disagreed with.** This
file ranks architectures by what the OBJECT is — layers composed, operations
per layer, weights shared — and reports `n_trainable_params` beside each rung
WITHOUT using it as the figure of merit or the tie-break. That follows the
standing note in `STATUS.md` ("explicitly not a model with the fewest
parameters") and reads the new brief's "smallest" as smallest *architecture*.
If hshunt meant parameter count, the ranking column changes and the predictions
below do not.

## Why the existing gate cannot answer the new brief

The published figure of merit is `rho_50 ≤ 0.740`, a **threshold density for
solving**. Findings 98–100 established what that criterion is made of:

- **98**: free generation scores a pair "ok" when the cell is unassigned and
  the colour is right, **never when the cell is *forced***. Naming a cell DD
  has not yet forced is not an error under the gate. Pooled at `j = 0`:
  error on a genuinely forced cell **0.0011–0.0032**, share of emissions naming
  an UNFORCED cell **0.0366 → 0.2529** as ρ falls. Conclusion, verbatim: **"the
  model has DD's rule and not DD's stopping condition."**
- **99/100**: tightening the gate to DD-faithful and then to the joint
  criterion changes **nothing per architecture** — 6 of 31 clear 0.740 and
  **the same 6** clear it under every criterion, `s_lo ⊆ s_pub` True and
  identical, bracket width **zero** at all 31.

So every version of the gate is a *solving* score dominated by the stopping
condition. Under the new brief that is the part we no longer care about. **The
rule is the part we do care about, and it has never been read per
architecture.**

## The instrument

Per `(run_id, rho, j)` the hazard shards carry `n_at_risk`, `n_in_F`
(emissions naming a **forced** cell), `n_err` and `n_err_in_F` (first errors
that named a forced cell — i.e. **the colour was wrong on a cell DD had
genuinely forced**, finding 100). The **rule hazard** is therefore

$$ h^{\text{rule}} \;=\; \frac{\sum_j n\_err\_in\_F(j)}{\sum_j n\_in\_F(j)}
   \;=\; \Pr(\text{colour wrong} \mid \text{named a forced cell}) $$

pooled over the ten **fitted** densities. This is DD's *rule* with the stopping
condition divided out, and it is the quantity the new brief asks about. A blind
guess would be `1 - 1/q = 0.75` at this ensemble's `q = 4`.

It is **not** a density threshold and is **never** converted into one.

## Predictions

| | prediction | MISS condition |
|---|---|---|
| **R1** | all 6 gate-passing architectures have `h_rule < 0.01` | any of the 6 is `≥ 0.01` |
| **R2** | the three architectures finding 100 named carry `h_rule` ≥ **10×** the median of the 6 | ratio `< 10×` |
| **R3** | across the 25 non-clearing architectures (those with ≥ 100 forced emissions) `h_rule` spans ≥ **5×** max/min | span `< 5×` |
| **R4** | Spearman(`h_rule`, published `rho_50`) over the 31 architectures is weaker than **\|0.5\|** | `\|ρ_s\| ≥ 0.5` |
| **R5** | at least one architecture **strictly smaller than J2** has `h_rule < 0.01` while FAILING the 0.740 gate | no such architecture |

**R5 is the prediction that answers the brief.** If it HITs, the minimal
architecture that *learns DD* is strictly smaller than the minimal architecture
that *solves to density 0.740*, because the gate was charging it for a stopping
condition that finding 98 says even the hub lacks. If it MISSes, the gate was
already the right selector and the standing minimum J2 stands unchanged.

**R2 is the one most likely to MISS and that is why it is registered.** Finding
100 reported a dramatic concentration — `tied_no_mlp d192` 977 errors,
`no_mlp d128 ep100` 677, `no_mlp d160` 307, with the other 28 architectures
sharing **27 between them** — but those are raw **counts**. They can be
produced entirely by the denominator (how often each architecture names a
forced cell at all, and how many cells it has). Normalising is the whole
contribution, and the honest outcome may be that the concentration evaporates.

**R3 tests whether this instrument has resolution where the gate has none.**
Finding 99: below the gate, DD-faithful `rho_50` spans **0.0173** across the 25
non-clearing architectures — narrower than one `s*`, median 0.873, i.e. `n ≈ 6`
blanks — so "the ranking there is noise". If `h_rule` also fails to separate
them, this stage reports that and the sub-gate ranking stays unreadable.

**R4 is a falsifier for the whole premise.** If rule fidelity tracks the
published gate strongly, then the gate already encodes it, this instrument adds
nothing, and the right conclusion is that J2 remains the answer under the new
brief as well.

## Traps carried forward

1. **KEY ON `cell`, NEVER ON `(arch_mode, num_layers, d_model)`.** Finding 99
   repeated 96's defect in exactly this place: that triple pools
   `full L=6 d256` (4,899,898) with `full L=6 ff256` (2,535,994) into a fake
   six-seed median. 79 runs → **31** `cell` labels.
2. **`FIT_RHOS = GRID[:10]`.** ρ 0.45/0.40/0.35/0.33 were generated but never
   fitted (101). Pool over the ten fitted densities only, and say so. At ρ 0.33
   **49 of 79 runs never produce one DD-faithful rollout**, so the held-out
   tail decides nothing here either.
3. **SEED SPREAD IS LARGER THAN THE LADDER'S RUNGS.** Findings 41/42: 0.1627 at
   fixed architecture, bigger than the depth gaps the ladder is built from, and
   **no single-seed rung is a measurement of its architecture**. Every `h_rule`
   is reported per cell with its **per-seed parts beside it** — the
   no-pooling lesson that stage DQ's misleading D2 taught, applied in advance.
4. **`rho_50` IS A COLOURING CRITERION (98).** It appears in R4 only as the
   thing `h_rule` is correlated against, and that sentence is attached wherever
   it is printed.
5. **THIS IS THE `d5q4` STAGE LOGIC POPULATION.** Chance on a colour is
   `1/q = 0.25` and a blind guess has `h_rule = 0.75`. **No `d4q3` constant
   enters** — not ρ_c ≈ 0.25, not `ln 3`, not the 0.99191 `a_eff`.
6. **A ZERO DENOMINATOR IS NOT A ZERO HAZARD.** An architecture that never
   names a forced cell has `h_rule` undefined, not perfect. Such cells are
   listed separately with their counts and excluded from R3's span, and the
   exclusion list is printed.
7. **CHECK FOR THE ARTEFACT, NOT THE EXIT CODE** — DQ3's split exited `0:0`
   with no verdict because a `|| true` masked a crash.

## Diagnostics (if either fails, nothing above is reported)

- **D1 — reproduce finding 100.** Total `n_err_in_F` over `in79` rows must come
  back at **1,988**, the affected-cell count at **63 of 1,106**, and the three
  named architectures must hold **39** of those 63 with counts 977/677/307. A
  mismatch means the loader has drifted and the stage stops.
- **D2 — column sanity.** `n_in_F ≤ n_at_risk` and `n_err_in_F ≤ n_err` on
  **every** row. Any violation means the columns do not mean what this file
  assumes and the stage stops.

## What a null licenses

A report, and nothing else. If R5 MISSes, J2 is still the answer and the
correct next move is to say so — **not** to train a smaller model hoping it
clears. If R1 MISSes, the gate-passing six do not have DD's rule either, and
the honest headline is that no architecture in this ladder learns direct
deduction, which would redirect the whole brief rather than rank anything.
