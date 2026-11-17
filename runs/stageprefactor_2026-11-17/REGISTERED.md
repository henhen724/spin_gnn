# Stage PREFACTOR — registered before looking at anything

Written and committed **before** the scorer was run. Nothing below was chosen
after seeing a number.

## The question

Stage LOGIC fitted, per density, over the 79 solving runs with `0 < S < 1`:

$$S \;=\; C\,p_{\rm logic}^{\,k}$$

and the exponent came out at `0.932 ± 0.025 · n` through the origin — L1 hit,
a logical error is fatal. But the **prefactor** did not stay at 1:

| n | 3 | 5 | 8 | 10 | 12 | 15 | 17 | 20 | 22 | 24 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | 1.018 | 1.011 | 1.012 | 1.005 | 0.991 | 0.985 | 0.953 | 0.900 | 0.821 | 0.797 |

`C` is the solve rate the fit extrapolates to at `p = 1`. `C = 0.797` at
ρ = 0.50 says a model that never errs on any scored position would still lose
a fifth of the instances. Either that is true and there is a second failure
mode per-token accuracy cannot see, or **the fit is lying about its own
intercept** and `C < 1` is an artefact of extrapolating a straight line from
`p ≈ 0.9` out to `p = 1`.

This stage decides which. **No training, no GPU, no new instances** — every
input is a CSV already on disk:

- `runs/stagelogic_2026-11-17/LF_summary_all_*.csv` — per (run, ρ): `p_logic`,
  `p_strict`, `solve_rate`, `realized_rho`, `in79`.
- `runs/stagepos_2026-11-17/B_instances_*.csv` — per (run, ρ, instance):
  `n_blank`, `T_true`, `n_pairs`, `first_error_j`, `first_kind`, `solved`,
  `terminated`, `well_formed`. 448,000 rows.

## Population and conventions, unchanged from stage LOGIC

The **79** runs with a nonzero solve rate at ρ = 0.50 (`in79`). The flagged
`full L=2 d256` seed 0 is excluded from every fit, as before. Densities
ρ = 0.95 … 0.50, the ten where a fit is defined. `n = 48(1 − ρ_real)`. All
logs natural. `C` always means the fitted intercept `exp(b)` of
`log S = b + k log p`.

## Predictions

**Q1 — curvature.** Refit each density with a quadratic,

$$\log S \;=\; \log C_q + k \log p + q\,(\log p)^2 .$$

*Prediction:* `q` is significantly nonzero (|q| > 2 se) at the densities where
`C < 0.95`, and the quadratic intercept `C_q` is **closer to 1** than the
linear `C`. *Falsified if* `|C_q − 1| ≥ |C − 1|` at a majority of those
densities — that would mean the shortfall is not curvature.

**Q2 — extrapolation distance.** The straight line is extrapolated from the
best model's accuracy `p_max` out to 1. Regress `(1 − C)` on `(1 − p_max)`
across the ten densities. *Prediction:* positive slope with R² > 0.5 — the
further the extrapolation, the larger the apparent shortfall. *Falsified if*
R² < 0.2.

**Q3 — top-decile refit.** Refit each density using only the 8 runs with the
highest `p_logic`, which sit closest to `p = 1` and so need the least
extrapolation. *Prediction:* `C_top > C` at every density with `n ≥ 17`, and
the median `C_top` over ρ ≤ 0.60 is ≥ 0.95. *Falsified if* `C_top ≤ C` at a
majority of those densities.

**Q4 — is there a hard residual subset?** For the single run with the highest
`p_logic` at ρ = 0.50, tabulate its per-instance solve rate against the
instance's own `n_blank` and against `T_true`. *Prediction:* its solve rate
falls monotonically in `n_blank` (Spearman < −0.5 over the strata with ≥ 100
instances). *Reported either way:* the `first_kind` breakdown of its failures,
because a residual subset that fails for a **single** reason is a mechanism and
one that fails for all reasons is just difficulty.

**Q5 — is aggregation over the n spread the cause?** Within one density bin the
instances do **not** all have the same `n_blank`. If each instance obeys
`S_i = p^{a n_i}`, the bin's mean solve rate is `E[p^{a n_i}]`, and since
`p^{an}` is convex in `n` that mean is **larger** than `p^{a E[n]}`. So
aggregation should push the measured curve **up**, flattening it and pulling
`C` **down**.

Test: at each density, stratify instances by exact `n_blank`, compute the solve
rate per (run, `n_blank`) stratum with ≥ 100 instances, and fit
`log S = log C_s + a·n_blank·log p` with `p` the run's bin-level `p_logic`.

*Prediction:* the stratified intercept `C_s` is **further below 1** than the
pooled `C`, i.e. `C_s < C`, because removing the Jensen lift removes the thing
that was flattening the line. *Falsified if* `C_s > C` at a majority of the
densities with `n ≥ 17` — which would make aggregation the cause after all,
and the opposite of what is written here.

**The honest prior.** Q1–Q3 are three versions of "the intercept is an
extrapolation artefact" and Q5 is "it is an aggregation artefact". If all four
miss, `C < 1` is real, and Q4's breakdown is the only evidence this stage has
about what the second failure mode is.

## Figures

| | |
| --- | --- |
| `P1_residuals_vs_p.png` | fit residuals against `p`, ρ = 0.75 / 0.60 / 0.50 — curvature is visible or it is not |
| `P2_C_vs_extrapolation.png` | `(1 − C)` against `(1 − p_max)`, one point per density |
| `P3_C_linear_quad_top.png` | `C` from the three fits against `n` |
| `P4_best_model_vs_nblank.png` | best model's solve rate against `n_blank`, ρ = 0.50 |
| `P5_stratified_C.png` | `C_s` against `C`, one point per density |

## Output

`runs/stageprefactor_2026-11-17/`: `prefactor_fits.csv` (per density: `C`, `k`,
`C_q`, `q`, `se_q`, `C_top`, `k_top`, `C_s`, `a_s`, `p_max`, counts),
`prefactor_strata.csv` (per run, ρ, `n_blank`: instances, solved),
`VERDICT.txt`, `figures/`.

Runs on a login node in under two minutes — it is CSV reading, a few dozen
least-squares fits and matplotlib, no GPU and no model. Nothing is submitted.
