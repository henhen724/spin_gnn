# Stage ABLATE (finding 106) -- registered before the scorer exists

Written 2026-11-19, committed before `scripts/stageablate_probe.py` and
`scripts/stageablate_fit.py` were created.  Nothing is trained.  Checkpoints are
opened read-only.

## The hypothesis, stated first

Stage HEADS (105) established *where* a model looks when it emits the colour of
the first cell direct deduction can fix: a median of **2** `(layer, head)` pairs
out of `L x H` carry half the attention the query spends on the named cell's own
`BEGIN_GRAPH` row, and the top 3 pairs carry a median of **0.698** of it.  That
is a statement about attention, and attention is not causation.  The deck shipped
with that caveat in its scope slide and STATUS named it the first open item.

Stage ABLATE asks the causal question.  **If those 2-3 pairs are the lookup, then
preventing exactly those pairs from reading exactly that row should break the
colour step, while preventing the same number of other pairs from reading it, or
preventing those same pairs from reading a different row, should not.**

## What is ablated, exactly

For one query position `q` and one row `R` (a set of 6 token positions), ablating
pair `(l, h)` means: in application `l`, head `h`, add `-inf` to the attention
logit at `(q, p)` for every `p` in `R`, before the softmax.  Softmax then
renormalises over the keys that remain.  Every other query row, every other head
and every other application keep the plain causal mask.

Three consequences are part of the definition and not side effects:

- The intervention is **local to row `q`**.  The model is causal and the readout
  is taken at `q`, so changing row `q` at application `l` changes the hidden
  state at `q` from `l` onward and changes nothing at any other position.
- It removes **that head's access at that application**, not the information.
  Earlier applications may already have copied the row into the residual stream
  at `q`.  A drop therefore shows the pairs are *necessary at their layer*; it
  cannot show they are the only route.
- It is **well defined on a tied model**, which corrects the caveat STATUS
  carried.  The weights of a tied block are shared across applications, but the
  mask is applied per application inside the hand-written forward loop, so
  ablating application 7 leaves applications 0-6 and 8-15 untouched.  The caveat
  would bite an intervention on the *weights*; this one is on the *pattern*.
  In the event it does not arise for the primary claims at all: all 18
  gate-passing runs are untied.

## Populations

**Query.** Identical to HEADS: `v` is the first cell of round 0, `q =
round_color_pos[v] - 1` is the NODE token whose hidden state must emit that
cell's colour, `j = 0`.  `R(v)` is `v`'s own `BEGIN_GRAPH` row: the header NODE
token and its 5 neighbour tokens, 6 positions.

**Instances.** The **held-out half** of every cached bin, instances `[200:400]`
at rho in `{0.90, 0.75, 0.60}`, 200 usable per density.  HEADS used `[0:200]`.
The top-m pairs are selected from the HEADS byhead CSV, i.e. from instances
`[0:200]`, so **the selection is out of sample with respect to the evaluation**.

**Runs.** The 18 gate-passing runs (6 cells x 3 seeds, all untied) carry the
primary predictions.  A contrast group of the 25 below-gate cells at their lowest
seed carries A6 only.  43 runs in total.

## Conditions

| name | pairs ablated | row ablated |
|---|---|---|
| `INTACT` | none | -- |
| `TOP-m`, m in {1,2,3,6} | the m pairs with the largest own-row mass in HEADS at the same `(run, rho)` | `R(v)` |
| `RAND-m`, m in {1,3,6} | m pairs drawn uniformly without replacement from the pairs outside the top **6** | `R(v)` |
| `OTHERROW-3` | the top 3 pairs | `R(u)`, `u` uniform over the other 47 nodes |

Nine conditions.  The `RAND-m` pool excludes the top 6 for every `m`, so the
three random conditions are drawn from one pool and are comparable with each
other.  Random draws are reseeded per `(run_id, rho, condition, instance)` so the
run is reproducible and the draw is averaged over instances rather than fixed.

**Metric.** `acc` = the fraction of held-out queries at which `argmax` of the
logits at `q` is `color_token(true_c)`.  Per run, then median over runs.

## Predictions, with bars and falsifiers

Main density is rho = 0.75 unless stated.  "Median" is the unweighted median over
the 18 gate-passing runs.

- **A1 -- the causal claim.** median `acc(TOP-3)` <= **0.50**.
  Falsifier: >= 0.90.
- **A2 -- specificity in which heads.** median `acc(RAND-3)` >= **0.90**.
  Falsifier: <= 0.70.
- **A3 -- specificity in which row.** median `acc(OTHERROW-3)` >= **0.90**.
  Falsifier: <= 0.70.
- **A4 -- dose-response.** `acc(TOP-1) >= acc(TOP-2) >= acc(TOP-3) >= acc(TOP-6)`
  in at least **15 of 18** runs, ties satisfying the inequality.
  Falsifier: <= 9 of 18.
- **A5 -- one head is already load-bearing.** median `acc(TOP-1)` <= **0.95**.
  Falsifier: >= 0.995.
- **A6 -- a registered null for the contrast group.** `|median acc(TOP-3) over
  the 25 below-gate cells - median acc(TOP-3) over the 18 gate-passing runs|` <
  **0.15**.  Falsifier: >= 0.30.
- **A7 -- the effect is not a property of one density.** median `acc(TOP-3)`
  <= **0.50** at each of rho in {0.90, 0.75, 0.60}.  Falsifier: >= 0.90 at any.

## Support disclosed, including what argues against each bar

Registering a bar after looking at the data is only honest if what was looked at
is stated.  Everything below is from HEADS, instances `[0:200]`, already on disk.

- Intact `argmax == true colour` is **1.0000 on all 54** gate-passing
  `(run, rho)` and averages 0.9971 below the gate.  The baseline is saturated, so
  every prediction has full headroom downward and A1 cannot be won by a weak
  baseline.
- Median `n50` is **2** and the top-3 pairs hold a median **0.698** of own-row
  mass across the 18 gate-passing runs, range `[0.380, 0.875]`.  **This argues
  against A1's magnitude**: TOP-3 removes roughly 70% of own-row attention, not
  all of it, and I have no prior mapping "mass removed" to "accuracy lost".
  `TOP-6` is in the design because of this.  Three runs sit at `n50` 4, 6 and 7
  with top-3 shares 0.458, 0.392, 0.380 and are the likeliest individual misses.
- **A6 is a null predicted from matched support, not from indifference.** Median
  `n50` is 2 in *both* groups and the top-3 share is 0.685 below the gate against
  0.698 above it.  Concentration does not separate competent from incompetent
  models, so I do not expect ablation to either.  My first reading of this stage
  assumed the opposite and the data refused it before it was registered.
- The below-gate tail is genuinely less concentrated (`n50` up to 15, top-3 share
  down to 0.150, five of the six least concentrated runs being tied cells), so if
  A6 misses, the direction to expect is the below-gate group being *harmed less*.
- No ablation of any kind has been run on these checkpoints.  Every number above
  is an attention mass; the sign of every prediction is a genuine prediction.

## Execution checks

Scored as PASS/FAIL beside the predictions; a FAIL invalidates the stage rather
than being reported as a result.

- **E1** `acc(INTACT)` = 1.0000 on all 18 gate-passing runs at all three
  densities, reproducing HEADS E1 on held-out instances.  A deviation means this
  harness's forward differs from HEADS's.
- **E2** Under every ablated condition, the attention from `q` to the ablated row
  at each ablated pair is exactly 0, and that row still sums to 1 within 1e-4.
- **E3** Under every ablated condition, the attention at every non-ablated
  `(l, h)` at row `q` of application 0 is unchanged from `INTACT` within 1e-6,
  and rows other than `q` are unchanged at every application.  This is the
  locality claim, checked rather than asserted.
- **E4** The token at `graph_row_pos[v]` is `node_token(v)` and the token at
  `q + 1` is `color_token(true_c)`, on every query (HEADS E3, repeated because
  the instance slice is new).

## Scope -- what this stage cannot establish

1. Necessity at a layer is not sufficiency, and not uniqueness of route.  See the
   second consequence above.
2. Zeroing and renormalising raises every surviving weight in row `q`.  `RAND-m`
   shares that artefact, which is what makes it the control, **but it does not
   match the mass removed**: by construction the top 3 pairs carry more own-row
   mass than any other 3.  A mass-matched control does not exist outside the top
   few pairs.  This confound is accepted and disclosed, not removed, and A4's
   dose-response is the partial answer to it.
3. `j = 0` only, as in HEADS.  The first deduction of a trace is the easiest one;
   nothing here speaks to deductions that depend on the model's own earlier
   output.
4. 200 held-out queries per `(run, rho)` resolve no accuracy difference below
   about 0.5%.
5. The 25 below-gate cells are one seed each, so A6 compares a 3-seed median
   against a 1-seed-per-cell median and is correspondingly noisier.
6. This stage trains nothing and changes no checkpoint.  It touches no closed
   knob and none of the four questions waiting on hshunt.
