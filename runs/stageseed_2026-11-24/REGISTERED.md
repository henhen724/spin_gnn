# Stage SEED -- is d192's rule-breaking a seed property or an architecture property?

Registered 2026-11-24, BEFORE the scorer ran and before any shard existed.
Nothing here may be edited after the first `F_hazard` row is written.

## What hshunt asked for, and why this is not what was approved

hshunt approved **2 more training seeds** of `tied_no_mlp L=16 d192` (270,010
params), the ladder's smallest object and its worst rule-breaker
(`h_rule` 0.08106, `u` 0.4359 at `r68`), because findings 130/131 say clump
membership is a seed property and this cell is single-seed.

**Those two seeds already exist and were trained on 2026-11-10.** Stage P
(array 51815311) trained all three seeds of this cell. All three `model.pt`
files survive:

    sudoku_n48_tied_P_2026-11-10/t0{0,1,2}_d0192_..._tied_no_mlp_L16_s{0,1,2}/

So this stage trains nothing. It runs the EXISTING stage LOGIC hazard eval
(measurement F) on the two checkpoints that were never evaluated, on stage
POS's cached instances -- the same 5,600 instances per run, the same 14-point
grid, the same 400/bin, the same eval seed start 60,000,000,000 -- so the three
seeds are directly comparable to each other and to every other cell in the
roster.

## Why the cell was single-seed, which is the finding that reframes the question

Seeds 1 and 2 were not skipped. They were **selected out**.
`stagepos_cells.successful_runs` admits a cell only if its solve rate at
rho = 0.50 is strictly positive:

    at50 = [r for r in rs if abs(float(r["rho"]) - 0.50) < 1e-9]
    if len(at50) == 1 and float(at50[0]["solve_rate"]) > 0:

At rho = 0.50 the three seeds solved:

| seed | run | solve@0.50 | val token acc | in the 79? |
| --- | --- | --- | --- | --- |
| 0 | r68 | 0.0025 = **1 of 400** | 0.4715 | yes |
| 1 | r80 | 0.0000 = 0 of 400 | 0.4561 | no |
| 2 | r81 | 0.0000 = 0 of 400 | 0.4567 | no |

**Seed 0 is in the population by one instance out of 400.** One Bernoulli
event decides membership, and seed 0 is also the best of the three on val
token accuracy. The published d192 rule number is therefore the *best* seed of
three, not a typical one -- which is a selection effect nobody registered.

That is why adding seeds cannot be done inside the 79: a new seed enters the
population only if it clears the same 1/400 gate. These two are appended with
`in79=False` through the new `POS_EXTRA` hook and are **never pooled with the
79**.

## Registered predictions

Hypothesis: at d192 the rule failure is a property of the architecture, not of
the seed, and seed 0's published number is the optimistic end of the spread.

- **S1 -- BOTH NEW SEEDS ALSO BREAK THE RULE.** `h_rule > 0.01` (the bar) for
  r80 and for r81, each on its own 14-density counts.
- **S2 -- THE THREE SEEDS AGREE WITHIN COUNTING NOISE.** A 3-way chi-square
  homogeneity test on `(n_err_in_F, n_in_F)` does not reject at 95%
  (`p > 0.05`, 2 d.o.f.). MISS means seed is resolvable at d192 on the rule arm.
- **S3 -- SEED 0 IS THE OPTIMISTIC END.** `h_rule(r68) < h_rule(r80)` and
  `h_rule(r68) < h_rule(r81)`. Registered because seed 0 is the only seed that
  solved at rho = 0.50 and has the highest val accuracy, so selection should
  have picked the least rule-breaking seed.
- **S4 -- THE STOPPING CONDITION MOVES WITH IT.** `u = 1 - s_F(0) >= 0.4359`
  for both new seeds.
- **S5 -- POOLING DOES NOT RESCUE THE CELL.** The 3-seed pooled `h_rule` stays
  within a factor of 2 of 0.08106, and stays above the 0.01 bar.

S1, S3 and S5 are all consistent with "architecture, not seed"; S2 is the test
that could overturn it; S3 is the one that could embarrass the published
number. A MISS on S3 with a MISS on S2 would mean the d192 rule number is
seed-driven and the single-seed measurement was lucky in the *bad* direction.

## Diagnostics that must pass before any of the above is scored

- **D1** r80 and r81 resolve to exactly one cell key each, `in79=False`,
  `in_seven=False`, 270,010 trainable params, `num_layers` 16,
  `arch_mode` `tied_no_mlp`, `d_model` 192 -- the same triple as r68.
- **D2** With `POS_EXTRA` unset, `successful_runs()` returns exactly 80 records
  (79 + the one flagged extra) and r00..r79 are unchanged. Already checked by
  hand on 2026-11-24 before submission; re-checked in the scorer.
- **D3** All three seeds carry 14 densities x 400 instances, and the realized
  rho per bin agrees with r68's to 1e-9 -- the instances are shared, not
  rebuilt.
- **D4** `n_err_in_F <= n_in_F <= n_at_risk` on every row.
- **D5** r68's own numbers recomputed from the shards reproduce the published
  `h_rule` 0.08106 and `u` 0.4359 exactly; if they do not, the comparison is
  not on the same eval and nothing is scored.

## Provenance

- branch `sudoku-dd`; commit recorded in VERDICT.txt at scoring time
- checkpoints: stage P, 2026-11-10, array 51815311, seeds 0/1/2
- eval: stage LOGIC measurement F, instances
  `runs/stagepos_2026-11-17/instances`, seed start 60,000,000,000
- no training; no new instances; nothing written outside this directory
