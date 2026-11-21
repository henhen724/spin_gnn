# Stage ENTROPY — registered before the probe ran

Question from hshunt, 2026-11-20:

> I am interested in the entropy of the generated reasoning distribution. In
> theory, a well pretrained model should just learn the DD-algorithm perfectly
> and it should have zero entropy distribution. Is it true the entropy of next
> token distribution is concentrating towards zero?

Predictions committed in `946a7ec` (probe) and the scorer commit, both BEFORE
job 53151109 produced any numbers.

## Why entropy is read per section, not pooled

`reason_node`, `reason_color`, `round_mark` and `term` are deterministic given
the prefix, so their entropy floor is zero. `hint_node`, `hint_color`,
`graph_lead` and `graph_nbr_novel` are genuinely random draws with a positive
floor. `graph_nbr_copy` is in between, because `shuffle_adj=True` randomises
slot order within a graph row. A single pooled number is dominated by the graph
section and answers nothing.

## Predictions

| # | prediction | bar | falsifier |
|---|---|---|---|
| E1 | mean H on `reason_color`, best cell | < 0.05 nats | > 0.30 |
| E2 | mean H on `reason_node`, best cell | < 0.15 nats | > 0.60 |
| E3 | fraction of `reason_node` tokens with H < 0.10 nats, best cell | >= 0.70 | < 0.30 |
| E4 | mean H on `graph_nbr_novel`, every cell | > 1.0 nats | < 0.5 |
| E5 | Spearman(mean reasoning H, `reason_node` accuracy) over 9 cells | <= -0.70 | > -0.30 |
| E6 | reasoning H at rho 0.35 exceeds rho 0.90, every cell | all | fails in > 2 |
| E7 | reasoning H rises with N at matched `pair_key` | 3 of 3 | 0 of 3 |

E3 is the actual question. E4 and E6 are controls: a probe that is measuring an
artefact rather than deduction would collapse everywhere and fail both.

## Diagnostics

| # | check | bar |
|---|---|---|
| DE1 | section token shares agree with stage SCALE | max abs diff < 2e-2 |
| DE2 | 0 <= H <= log(vocab) at every row | 0 violations |
| DE3 | uniform logits recover H = log V | see amendment |

### DE3 amendment, recorded because a registered bar was loosened

DE3's original bar was 1e-9 and the probe measured 9.537e-07. That is a
tolerance error in the registration, not an implementation fault: the forward
pass is float32, log 58 = 4.06, and float32 carries ~1.2e-7 relative
resolution, so the achievable floor is ~5e-7. No correct implementation could
have passed 1e-9.

DE3 is now scored at two precisions — float64 to 1e-12, which tests the
entropy estimator itself, and float32 to 1e-5, which tests the pipeline the
probe actually runs. Both pass. The amendment is also in the probe docstring
and in commit `4f41d1c`.

## Method

Nine checkpoints from the stage SCALE grid (N=48: L04_s0, L08_s0, L08_s1,
L12_s0; N=96: L08_s0, L08_s1, L12_s0, L12_s1, L16_s0), five densities
(0.90 0.75 0.60 0.45 0.35), 100 cached instances each, teacher-forced.
Inference only — nothing trained, no checkpoint written, no closed knob
touched. Entropy from `log_softmax` to avoid `0 * -inf = nan`.

Run as job 53151109, array 0-2, partition `sganguli`, CPU.

## Outcome

7 of 7 predictions HIT, 3 of 3 diagnostics PASS. See `VERDICT.txt`.
Deck: https://slides-hshunt.pages.dev/spin-gnn/entropy/
