# Stage HEADS — registered before the scorer exists

Finding 105.  Registered 2026-11-19.  Branch `sudoku-dd`.

## The question

Stage LOCAL (104) closed the *what* and left the *how*.  It showed that the
colour step must complete the adjacency join on >= 99.90% of emissions: an
emitter that dropped one of the named cell's assigned neighbours would have
been improper 12.5% of the time, which over 103's 24,008 wrong unforced
emissions is ~3,011 expected violations against 0 observed.  It also showed
that 96.8% of named cells have a DROPPABLE neighbour — one whose colour is
duplicated on another assigned neighbour, so dropping it leaves the candidate
set unchanged.  From that came the one sentence LOCAL could not test:

> the join must be complete in EFFECT, not in EFFORT — it need not attend
> equally to all five.

This stage asks where the model actually looks.  Three questions, in order of
what they buy:

1. **Is there a hop-1 lookup at all** — does the query that must emit the
   colour attend to the named cell's own row of `BEGIN_GRAPH`, above the other
   47 rows?
2. **Is there a hop-2 lookup** — does it attend to the hint COLOUR tokens of
   that cell's neighbours, above the other hints?
3. **Is the 96.8% redundancy exploited** — within the named cell's row, does a
   CRITICAL neighbour (its colour is the only one of its kind among the
   assigned neighbours) draw more attention than a DROPPABLE one?

(3) is 104's open question stated so it can be measured.  (1) and (2) are the
two hops the answer to (3) presupposes.

**THIS IS THE FIRST STAGE SINCE 97 THAT LOADS A MODEL.**  Stages 100–104 each
beat a STATUS claim that the next question "needs a job" by finding the answer
already on disk, four times running.  I checked that first and it does not
hold here: `grep -rln 'attn_weights|need_weights|output_attentions|attn_probs|
attention_weights' spin_gnn/ scripts/` matches exactly one file,
`spin_gnn/sudoku_attention_distance.py`, which *computes* attention but has
never been run to disk — no CSV, verdict, figure or deck in this project
contains an attention weight.  So the instrument exists and the measurement
does not.  **That is a weaker claim than "this needs a job" and I am making
the weaker one.**

## What was inspected before registering — read this before scoring any band

Four things, all of which narrow what is live here:

1. **A 3-instance smoke on two checkpoints**, at rho 0.90, printing exactly
   the hop-1 selectivity `S_row` defined below.  It returned **0.85, 0.94,
   1.00** for `L=3 h=4 full` and **4.78, 5.36, 6.41** for `L=12 h=4 full`.
   So the EXISTENCE of hop-1 attention in a deep model, and its absence in a
   shallow one, is **already seen** — H1's sign is near-certain and only its
   magnitude across 31 cells is live.  H2, which holds depth fixed, was not
   seen at all.
2. **The support counts at j = 0**, over all 400 cached instances at each of
   the ten fit densities, with no model loaded.  Every j = 0 cell is DD-forced
   (400/400 at all ten densities), so its `k` assigned neighbours carry
   exactly 3 distinct colours and therefore droppable = `k - 3`, blank =
   `5 - k`, with `k` in {3,4,5}.  Cells with >= 1 droppable neighbour:
   0.993 / 0.980 / 0.932 / 0.895 / 0.828 / 0.743 / 0.682 / 0.620 / 0.552 /
   0.463 at rho 0.95 -> 0.50.  Cells with >= 1 blank neighbour: 0.200 / 0.328
   / 0.482 / 0.578 / 0.680 / 0.725 / 0.805 / 0.835 / 0.890 / 0.930.
   **I counted these BECAUSE L5 of stage 104 was scored VOID for comparing
   against a `k = 5` stratum that held zero cells.**  Both contrasts this
   stage needs are populated at the three densities it will run.
3. **The gate-passing list**, from `bound_runs.csv`: 6 of 31 cells, 18 runs.
4. **The accuracy axis**: cell-median `val_acc` spans 0.4670–0.4993 overall
   and 0.4715–0.4940 within depth 16, where **five of the nine cells sit
   inside 0.0008 of each other**.  That tie structure is why H2 is written as
   a single contrast and not as a Spearman: R5 of stage 103 was void because a
   Spearman ran over 60 tied zeros and reported tie-breaking, and nine cells
   with five ties is the same shape.  **Below the gate, rho_50 spans only
   0.742–0.856 and STATUS already records that ranking as noise, so no
   prediction here is scored against rho_50 below the gate.**

## The instrument

**The query.**  For each cached instance, let `v` be the first node of round 0
of the planted trace (`trace["rounds"][0][0]`), and `q` the position of `v`'s
NODE token in the reasoning section, i.e. `round_color_pos[v] - 1` from
`sudoku_attention_distance.token_positions`.  The sequence is teacher-forced
on the planted trace, so the context at `q` is exactly

```
BEGIN_HINTS (NODE COLOR)*  BEGIN_GRAPH (NODE NODE^5)^48  BEGIN_REASONING BEGIN_ROUND NODE(v)
```

and the next token the model must emit is `v`'s colour.  This is the same
j = 0 anchor stages 102, 103 and 104 used: the pre-state is exactly the hints,
so `c(v)` and `k(v)` are reconstructible with no rollout.  Attention is
causal, so the row at `q` of a forward pass over the FULL sequence is
identical to the row at `q` of a pass over the prefix; one pass per instance
serves every query.

**Capture.**  `sudoku_attention_distance.forward_capture_attention` returns one
`(heads, L, L)` map per APPLICATION — it iterates `applied_blocks`, so a tied
cell contributes its shared block `model_depth` times rather than once.  Only
row `q` of each map is retained.  Write `A^{l,h}_j` for the weight from `q` to
key `j`, with `sum_{j<=q} A^{l,h}_j = 1` for every `(l,h)`.

**Token groups** (all at positions < q; every node has degree 5, so all 48
graph rows are exactly 6 tokens long and no length correction is needed):

- `OWNROW(v)` — `v`'s `BEGIN_GRAPH` row: its header NODE token plus the 5
  neighbour tokens after it.
- `ROWS` — all 48 graph rows, 288 tokens.
- `OWNNBR(v)` — just the 5 neighbour tokens of `OWNROW(v)`, each classified as
  **CRITICAL** (assigned, and its colour appears on no other assigned
  neighbour of `v`), **DROPPABLE** (assigned, colour duplicated on another
  assigned neighbour) or **BLANK** (not assigned by a hint).
- `NBRHINT(v)` — the hint COLOUR tokens of `v`'s `k` assigned neighbours.
- `HINTS` — all hint COLOUR tokens.

**The five quantities.**  Masses are summed over every `(l,h)` and over
instances BEFORE any ratio is taken, so each is mass-weighted and no ratio of
ratios appears:

$$S_{\text{row}} \;=\; 48\,\frac{m(\mathrm{OWNROW})}{m(\mathrm{ROWS})}
\qquad
S_{\text{hint}} \;=\; \frac{m(\mathrm{NBRHINT})/k}{m(\mathrm{HINTS})/n_{\text{hints}}}$$

$$R_{dc} \;=\; \frac{\bar a_{\text{crit}}}{\bar a_{\text{drop}}}
\qquad
R_{ab} \;=\; \frac{\bar a_{\text{assigned}}}{\bar a_{\text{blank}}}$$

where $\bar a_X$ is the total mass on class-$X$ neighbour tokens divided by the
number of such tokens.  $S_{\text{row}} = 1$ and $S_{\text{hint}} = 1$ mean no
preference; $R_{dc} = 1$ means the five neighbours are read uniformly.

$n_{50}$ is the smallest number of `(l,h)` pairs whose own-row mass
$g_{l,h} = \sum_{j \in \mathrm{OWNROW}} A^{l,h}_j$, sorted descending, reaches
half of $\sum_{l,h} g_{l,h}$.  Out of $L \times H$ pairs (up to 64).

**Population.**  All 80 runs, rho in {0.90, 0.75, 0.60}, the first 200 cached
instances per density, one query each: 48,000 forward passes.  Scored on the
79; the 80th is reported separately as every stage has done.  Per-run values
pool the 200 instances; per-cell values are the median over that cell's seeds.
$R_{dc}$ is computed on cells with `k >= 4` (which is where a droppable
neighbour exists) and $R_{ab}$ on cells with `k <= 4` (where a blank exists).

## Predictions

**H1 — hop 1 exists and tracks the gate.**  Median $S_{\text{row}}$ at
rho 0.75 over the 6 gate-passing cells, divided by the median over the 25
non-passing cells, is **>= 1.5**.  Falsifier: <= 1.0.
*Sign disclosed as near-certain* — the smoke already saw ~1 for `L=3 full` and
~5 for `L=12 full`, and the latter is a passing cell.  Only the magnitude over
31 cells is live.

**H2 — and it is not only depth.**  At fixed depth 16, the one gate-passing
cell (`L=16 h=4 no_mlp d192`) has a higher $S_{\text{row}}$ at rho 0.75 than
the median of the other eight depth-16 cells, on **all 3 of its seeds**.
Falsifier: 0 of 3.  This is a one-cell contrast with three seeds as its error
bar, chosen over a 9-cell Spearman because five of those nine cells are tied
in `val_acc` to within 0.0008.

**H3 — hop 2 exists.**  Pooled $S_{\text{hint}}$ at rho 0.75 over the 18
gate-passing runs is **>= 1.5**.  Falsifier: <= 1.0.  Not seen in the smoke.

**H4 — 104's open question: the join is complete in EFFORT, not only in
EFFECT.**  Pooled $R_{dc}$ at rho 0.75 over the 18 gate-passing runs lies in
**[0.90, 1.10]**.  Falsifier: >= 1.25 or <= 0.80.
*Why this direction:* droppability is not visible at the neighbour's token in
`BEGIN_GRAPH` — that token names a node, and whether its colour is duplicated
is only knowable after hop 2.  A head doing hop 1 therefore has nothing to
condition on, and uniform attention over the five slots is the prediction.
A hit says the model pays the full cost of a join it could have shortened; the
falsifier firing high would be the more interesting outcome, because it would
mean hop-2 information is reaching the hop-1 read.

**H5 — blanks are discriminated.**  Pooled $R_{ab}$ at rho 0.75 over the 18
gate-passing runs is **>= 1.3**.  Falsifier: <= 1.0.  A purely structural
row-read is uniform over all five slots and gives 1.0; anything above means
the read is already content-dependent.  Note H4 and H5 can both hold: blank
vs assigned is decidable at hop 1 if the hint section is consulted at all,
droppable vs critical is not.

**H6 — the circuit is small, which is the brief's number.**  Median $n_{50}$
at rho 0.75 over the 18 gate-passing runs is **<= 4** of the `L x H` pairs
(<= 64).  Falsifier: >= 12.

## Diagnostics

**E1 — the query is the right position.**  Teacher-forced argmax at `q` equals
`v`'s true colour token on >= 0.90 of queries for the 18 gate-passing runs,
and the argmax is *some* COLOR token on >= 0.99 of queries for all 80.

**E2 — the capture is a distribution.**  `|sum_{j<=q} A^{l,h}_j - 1| <= 1e-4`
for every `(l,h)` on every query, and `A^{l,h}_j = 0` for all `j > q`.

**E3 — the layout mirrors the encoder.**  The token at `graph_row_pos[v]`
equals the NODE token for `v`, and the token at `q + 1` equals the COLOR token
for `v`'s planted colour, on every query.  `token_positions` reimplements
`encode_instance`'s layout rather than reading it, so this is the check that
the two have not drifted.

## Scope — what this cannot show

- **Attention is not causation.**  Mass on `OWNROW` does not prove the model
  uses it; that needs an ablation, which this stage does not do.  Every claim
  here is about where the model looks.
- **It does not separate (B2) from candidate-set computation.**  LOCAL
  disclosed that adjacency-local inhibition and computing the candidate set
  are the same function and no output-only measurement separates them.  This
  is an internals measurement but still not a separation: it says the two-hop
  lookup is *there*, not which of two descriptions of the same arithmetic is
  the right one.
- **It is about the COLOUR step on FORCED cells only.**  The j = 0 query is
  teacher-forced at the planted trace, so every cell measured has `c = 1`.
  102 showed the CELL step is selected on `k` and not on `c`; nothing here
  touches that, and no claim about the cell step may be read off it.
- **H1 is confounded with depth** and H2 is the control.  If H1 hits and H2
  misses, the honest reading is that hop-1 selectivity is a depth effect and
  the stage has not shown it tracks competence.
- 200 instances per (run, density) is 200 queries; no per-instance rate below
  ~0.5% is resolvable and none is claimed.

## Outputs

`runs/stageheads_2026-11-19/`:
`heads_byhead.csv` (one row per run, rho, layer, head: the group masses and
counts), `heads_runs.csv` (one row per run, rho: the five quantities, `n_50`,
the rank statistics, `cell`, `params`, `val_acc`, `rho50_pub`, depth, heads),
`VERDICT.txt`, `figures/`.
Measurement: `scripts/stageheads_probe.py` (torch).  Scoring:
`scripts/stageheads_fit.py` (matplotlib) — the two module sets are mutually
exclusive on this cluster, so they cannot be one script.

Nothing is trained.  Every model is an existing checkpoint, loaded read-only.
