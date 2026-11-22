---
title: "Does the colour head break because of size, or because of depth?"
author: "hshunt"
date: "2026-11-21"
---

<!--
DRAFT -- EXPLANATION PAGES ONLY. The results pages are deliberately absent:
array 53243550 is still training and the section split 53264789 is chained
behind it. This file lives in the PROJECT repo, not in ~/slides, precisely so
that a half-finished deck cannot be picked up by a build and listed on the
spin-gnn contents page. When the cells land, copy this into
~/slides/decks/spin-gnn/depth-lever/deck.md, append the results pages, and
publish with claude-slides-publish.sh.

No q=4 constant appears anywhere below. That is a rule of this stage, not a
style choice: a reader who sees a q=4 number on the same page as a q=3 number
will compare them, and those two ensembles have different chance levels,
different critical densities and different sequence lengths. Where a prior
q=4 result has to be described, it is described structurally (`at that
ensemble's own chance level`) and the numeral is withheld on purpose, with
the page saying so.
-->

# Does the colour head break because of size, or because of depth?

A transformer is trained to solve graph-colouring puzzles by writing out its
deductions one at a time. At the largest puzzle size we have tried, it stops
learning the colouring rule entirely. Two explanations fit every number we
have, and this run is built to tell them apart.

---

## The task: fill in a partly revealed colouring, one deduction at a time

We generate a graph whose nodes have already been validly coloured, hide most
of the colours, and ask the model to recover them. The model does not output
an answer in one shot. It writes a **reasoning trace**: a sequence of
`(node, colour)` steps, in a fixed order, until every hidden node is filled
in. Everything measured in this deck is measured on that trace.

---

## An instance is a random 4-regular graph with a planted 3-colouring

- **$d$-regular**: every node has exactly $d$ neighbours. Here $d = 4$.
- **$q$-colouring**: every node carries one of $q$ colours. Here $q = 3$.
- **Proper**: no edge joins two nodes of the same colour.
- **Planted**: we *first* choose a proper colouring, *then* keep only graphs
  consistent with it. So a valid answer is guaranteed to exist.

An instance is legal only if $q \mid N$ and $q(q-1) \mid dN$. At $d=4, q=3$
that reduces to $3 \mid N$.

---

## Hint density $\rho$ is the fraction of node colours we reveal

We reveal the colours of $\rho N$ nodes, chosen at random, and hide the rest.

$$n_{\text{blank}} = N - \mathrm{round}(\rho N)$$

is the number of nodes the model must deduce. High $\rho$ is an easy puzzle
with few blanks; low $\rho$ is a hard one. $\rho$ is the single knob that
moves an instance from easy to hard, and every evaluation below is a curve
over $\rho$.

---

## "Fall colouring" means every hidden node is reachable by deduction alone

A random subset of hints usually leaves nodes that **no** amount of local
reasoning can pin down — you would have to guess and backtrack. We do not
want to measure guessing, so hints are removed greedily and a removal is kept
only if the puzzle is still solvable by deduction alone. The surviving hint
set is the **fall colouring**. Every training and evaluation instance in this
project has this property.

It requires $d \ge q-1$, and it is the reason $\rho$ cannot be pushed
arbitrarily low: below some $\rho_{\min}$ no hint set works at all.

---

## Direct deduction fills a node when its neighbours exclude all but one colour

**Direct deduction (DD)**, also called naked-single elimination: look at one
node, list the colours already taken by its coloured neighbours, and if
exactly one colour survives, that colour is forced.

With $q = 3$, a node is forced as soon as two distinct colours appear among
its neighbours. No search, no guessing, no backtracking — DD either forces a
node or says nothing about it.

---

## A round applies DD to every node at once, and $T$ counts the rounds

One **round** evaluates DD at every still-blank node simultaneously, using
only the colours known at the *start* of the round, and fills in whichever
nodes were forced. Newly filled nodes then force more nodes in the next
round.

$T$ is the number of rounds needed to finish an instance. It is the **length
of the reasoning chain**: a node filled in round 7 depends on a node filled
in round 6, and so on back to the hints.

**$T$ is the quantity this whole run is about.** Call it the *DD depth*.

---

## The instance is serialised into one token sequence with four parts

1. `BEGIN_HINTS`, then each revealed node and its colour, as a pair.
2. `BEGIN_GRAPH`, then one row per node: the node, then its $d$ neighbours.
   Every undirected edge is therefore written **twice**, once in each
   endpoint's row.
3. `BEGIN_REASONING`, then for each round a `BEGIN_ROUND` marker followed by
   the forced `(node, colour)` pairs, sorted by node index.
4. A terminator, `SOLVED` or `UNRESOLVED`.

Neighbour order within a row is shuffled, so the model cannot read anything
off the ordering.

---

## The vocabulary has $N + 6 + q$ symbols and nothing else

| ids | meaning |
|---|---|
| 0–5 | `BEGIN_HINTS`, `BEGIN_GRAPH`, `BEGIN_REASONING`, `BEGIN_ROUND`, `SOLVED`, `UNRESOLVED` |
| 6 … 5+$q$ | `COLOR_0` … `COLOR_{q-1}` |
| 6+$q$ … | one id per node, `v0` … `v{N-1}` |

$$|V| = N + 6 + q$$

At $q=3$ the first node id is 9; at $q=4$ it would be 10. That one-symbol
shift is why every checkpoint is checked for
$|V| = N + 6 + 3$ before it is scored.

---

## A worked 12-node example: the graph and the colouring to recover

$N=12$, $d=4$, $q=3$, $\rho=0.5$. (The trained cells are $N=96$ and $N=120$;
this one is small enough to print.)

Planted colouring: `v0=2 v1=1 v2=1 v3=0 v4=0 v5=1 v6=1 v7=0 v8=2 v9=2 v10=0 v11=2`

Revealed: `v0=2 v3=0 v5=1 v6=1 v8=2 v9=2` — six of twelve.

| node | neighbours | | node | neighbours |
|---|---|---|---|---|
| v0 | v6 v7 v4 v3 | | v6 | v11 v3 v8 v0 |
| v1 | v7 v11 v9 v10 | | v7 | v5 v1 v2 v0 |
| v2 | v7 v3 v8 v4 | | v8 | v10 v2 v6 v4 |
| v3 | v6 v9 v0 v2 | | v9 | v5 v1 v3 v10 |
| v4 | v8 v2 v0 v11 | | v10 | v9 v1 v8 v5 |
| v5 | v10 v11 v9 v7 | | v11 | v5 v6 v4 v1 |

---

## The same example, solved: three rounds, six deductions

- **Round 1** — `v2←1`, `v7←0`, `v10←0`
- **Round 2** — `v1←1`, `v4←0`
- **Round 3** — `v11←2`

So $T = 3$ and $n_{\text{blank}} = 6$. Every deduced colour matches the
planted one, which is guaranteed: DD only ever writes a forced colour.

Read round 2 to see the chain: `v1` becomes forced only *because* `v7` and
`v10` were filled in round 1.

---

## The same example as the exact token stream the model reads

```
BEGIN_HINTS
  v0 COLOR_2  v3 COLOR_0  v5 COLOR_1  v6 COLOR_1  v8 COLOR_2  v9 COLOR_2
BEGIN_GRAPH
  v0 v6 v7 v4 v3   v1 v7 v11 v9 v10   v2 v7 v3 v8 v4   v3 v6 v9 v0 v2
  v4 v8 v2 v0 v11  v5 v10 v11 v9 v7   v6 v11 v3 v8 v0  v7 v5 v1 v2 v0
  v8 v10 v2 v6 v4  v9 v5 v1 v3 v10    v10 v9 v1 v8 v5  v11 v5 v6 v4 v1
BEGIN_REASONING
BEGIN_ROUND  v2 COLOR_1  v7 COLOR_0  v10 COLOR_0
BEGIN_ROUND  v1 COLOR_1  v4 COLOR_0
BEGIN_ROUND  v11 COLOR_2
SOLVED
```

The model sees only these ids. There is no grid, no picture and no colour
names — `COLOR_1` is just symbol 7.

---

## The sequence length is exactly $N(d+3) + T + 4$

$$L \;=\; \underbrace{1 + 2\rho N}_{\text{hints}} \;+\; \underbrace{1 + N(d+1)}_{\text{graph}} \;+\; \underbrace{1 + T + 2N(1-\rho)}_{\text{reasoning}} \;+\; \underbrace{1}_{\text{terminator}} \;=\; N(d+3) + T + 4$$

The $\rho$ terms cancel: a hint costs two tokens up front and a deduction
costs two tokens later. The example above has
$12 \times 7 + 3 + 4 = 91$ tokens, which is what it printed.

**Most of the sequence is the graph**, not the reasoning — $N(d+1)$ of it.

---

## The model is a 12-layer decoder transformer predicting the next token

| | |
|---|---|
| layers | 12 |
| model width $d_{\text{model}}$ | 256 |
| attention heads | 4 |
| feed-forward width | 1024 |
| trainable parameters | 9.77 M at $N=96$ |

Trained with plain next-token cross-entropy over the whole sequence —
hints, graph and reasoning alike. There is no special loss on the deductions
and no auxiliary objective. The architecture is **copied unchanged** from the
previous run so that the only things differing are $(d, q, N, \rho)$.

---

## Training: 240k fresh instances, 50 epochs, $\rho$ uniform on $[0.14, 0.95]$

Every instance is generated once and never repeated. $\rho$ is drawn
uniformly per instance, so the model sees easy and hard puzzles mixed
together; the lower end, 0.14, is this ensemble's own hint floor
$\rho_{\min} \approx 0.150$, not a number carried from anywhere else.

Four cells are trained: $N = 96$ and $N = 120$, two random seeds each. Two
seeds, because in earlier runs the spread between seeds was larger than the
effect of changing the size.

---

## Per-section NLL: only some of the sequence has a floor of zero

The sequence mixes two kinds of token, and pooling them hides everything.

- **Genuinely random** — the hint identities, the hint colours, and the first
  mention of each edge. These are fresh random draws. No model can beat the
  draw's own entropy, so their loss has a **positive floor**.
- **Forced by the prefix** — `reason_node`, `reason_color`, the round markers
  and the terminator. Emission order within a round is sorted by node index
  and the colour is forced by DD, so a perfect model would put all its mass
  on one symbol. Their floor is **exactly zero at every size**.

So every nat of loss on a forced section is pure excess, and needs no floor
model to interpret.

---

## `reason_color` is the loss on the deduced-colour token, and it is the metric

Let $S$ be the set of positions holding a deduced colour. Teacher-forced (the
model is fed the true prefix, so one mistake cannot cascade):

$$\mathrm{NLL}_{\text{color}} \;=\; -\frac{1}{|S|}\sum_{t\in S}\log p_\theta\!\left(x_t \mid x_{<t}\right)$$

Units are **nats**. $0$ means the colour rule has been learned exactly. It is
reported together with the top-1 accuracy on the same positions.

This is the headline number of the whole deck.

---

## Chance for `reason_color` is $\ln q = \ln 3 = 1.0986$ nats

A model that has learned nothing about colours but *has* learned that a
colour symbol goes here can do no better than guessing uniformly among $q$:

$$-\log \tfrac{1}{q} \;=\; \log q, \qquad \log 3 = 1.0986 \text{ nats}, \qquad \text{accuracy } \tfrac{1}{3} = 0.333$$

A measured value **at** 1.0986 with accuracy 0.333 means the colour rule was
never learned. A value *above* 1.0986 means something is actively wrong, not
merely unlearned.

---

## `solve_rate` is the fraction of instances completed entirely correctly

At evaluation the model generates the whole reasoning section itself, with no
teacher forcing. An instance counts as solved only if **every** deduced
colour is right and the trace terminates properly. One wrong node fails the
instance.

Evaluation is a grid of 14 hint densities, 400 instances per density, with
both seeds given the identical instances so they can be compared point for
point.

---

## $a_{\text{eff}}$ turns a whole-instance rate into a per-deduction rate

`solve_rate` is brutal: it is a product over $n_{\text{blank}}$ consecutive
deductions, so it collapses toward zero as puzzles get larger even if
per-step competence is unchanged. Dividing that out:

$$a_{\text{eff}} \;=\; s^{\,1/n_{\text{blank}}}, \qquad s = \text{solve\_rate}, \qquad n_{\text{blank}} = N - \mathrm{round}(\rho N)$$

$a_{\text{eff}}$ is the per-deduction success rate implied by the whole-trace
result. $a_{\text{eff}} = 0.99$ with 80 blanks gives $s \approx 0.45$;
$a_{\text{eff}} = 0.95$ gives $s \approx 0.017$.

---

## Zero solves bounds $a_{\text{eff}}$, it does not pin it

If a bin solves 0 of 400, then $s = 0$ and $a_{\text{eff}} = 0$ for **every**
$n_{\text{blank}}$ — which says only that the product underflowed, not
anything about per-step competence. Reporting it as 0 would be a false
precision, so we report the bound instead.

One-sided Clopper–Pearson at confidence $c$, with 0 successes in $n$ trials:

$$p \;<\; p_{\text{up}} = 1-(1-c)^{1/n}, \qquad p_{\text{up}} = 0.00746 \text{ at } n=400,\ c=0.95$$

and since $a_{\text{eff}}$ is increasing in $p$,

$$a_{\text{eff}} \;<\; p_{\text{up}}^{\,1/n_{\text{blank}}}$$

---

## $\rho_c$ is where the puzzles stop being shallow, and it moves with $(d,q)$

Above a critical hint density $\rho_c$, DD finishes in a number of rounds that
grows only like $\log N$. Below it, the chain lengthens sharply: the same
puzzle needs many more sequential rounds. That is the **fast–slow
transition**.

$\rho_c$ is a property of the instance ensemble, not of the model. For
$d=4, q=3$ we measure $\rho_c \approx 0.25$, by finding where $T/\log N$ stops
collapsing onto one curve across $N = 48 \ldots 384$. The same procedure,
applied to the previously published ensemble, recovers its published value —
so the method is calibrated rather than fitted.

---

## $T/\log N$ lies on one curve above $\rho_c$ and fans apart below it

![](figures/collapse.png)

Each panel is one ensemble, each curve one size. Density runs **right to
left**: easy puzzles on the left, hard on the right. Where the three sizes
overlap, $T$ is just $\log N$ times a function of $\rho$; where they separate,
depth has started growing faster than $\log N$. The dashed line marks that
density. The left panel is the calibration: the same reading applied to the
previously published ensemble returns its published $\rho_c$. Its curves stop
early because that ensemble has no instances below its own hint floor.

---

## The prior result: at the largest size, the colour head sat exactly at chance

In the previous run, on a different ensemble, `reason_color` landed **at that
ensemble's own chance level**, with top-1 accuracy equal to $1/q$, at every
density and on both seeds. At the two smaller sizes the same architecture
learned the rule essentially perfectly. The data was checked and was sound.

*The numerals from that run are withheld deliberately throughout this deck.*
They belong to an ensemble with a different chance level, a different
critical density and sequences roughly twice as long, and putting them on a
page beside a number from this run would invite a comparison that is not
valid.

---

## Two explanations fit that result, and they have opposite consequences

- **H-N — it is the size $N$.** More nodes means a bigger vocabulary, a
  longer graph section, and more distinct identities to keep straight. The
  reasoning itself is fine; the bookkeeping is not.
- **H-T — it is the depth $T$.** Bigger puzzles need more sequential rounds,
  so the chain from a hint to a late deduction is longer. The bookkeeping is
  fine; the reasoning chain is too long.

H-N says the method is sound and needs scale engineering. H-T says the
*format* is the limit and no amount of scale fixes it.

---

## A size ladder cannot separate them, because $N$ and $T$ move together

Every run so far has varied $N$. But raising $N$ raises the DD depth at the
same time — that is what $\rho_c$ and the $\log N$ scaling mean. So on a size
ladder H-N and H-T predict the *same* ordering, at every rung, always.

Adding a fourth rung would add a fourth confounded point. The confound is
structural, not statistical, so more data along the same axis cannot fix it.

---

## Raising $N$ raises $T$ at every single density — the two never move apart

![](figures/depth_vs_rho.png)

DD rounds against hint density for three sizes of the $d=4, q=3$ ensemble,
density again running right to left. There is no density at which the three
curves cross or meet below $\rho_c$: bigger is deeper, everywhere. That is the
confound stated as a picture. To change $T$ without changing $N$ you have to
leave the curve — which is what changing $(d,q)$ does.

---

## $(d,q) = (4,3)$ buys depth from the local parameters at fixed $N$

Lowering the number of colours and the degree makes each individual deduction
*rarer*, so the chain gets longer — without touching $N$. Measured on 40
instances per cell, deepest decile of DD depth:

| cell | DD depth $T$ | mean tokens | $|V|$ |
|---|---|---|---|
| **$d4q3$, $N=96$** | **32.8** | **687** | 105 |
| **$d4q3$, $N=120$** | **35.0** | 856 | 129 |
| previous ensemble, same $N=96$ | 22.5 | 781 | 106 |
| previous ensemble, the size that failed | ~35.4 | ~1560 | 202 |

$N=96$ here is **deeper than the matched cell while being shorter**, and
$N=120$ **matches the depth of the cell that failed** at 1.8× fewer tokens —
both at an $N$ where the colouring rule is known to be learnable.

---

## The depth gap holds at every size, 1.3× at $N=48$ widening to 1.6×

![](figures/depth_vs_n.png)

The same lever across the whole size range, 24 seeds per point. Each curve is
taken at **its own** ensemble's lowest reachable density, because the two do
not share a hint floor; comparing them at a common density would instead be
comparing a hard puzzle to an impossible one.

This is the **mean** depth at the floor, which is a different estimator from
the deepest-decile figure in the table on the previous page — that is why
$N=96$ reads 28.8 here and 32.8 there. Both say the same thing, and neither
is quoted as a result.

---

## So the two hypotheses now disagree, which is the point of the run

At $N=96$, $d=4$, $q=3$:

- **H-N predicts the colour rule is learned.** $N$ is small, so whatever
  broke at the large size is absent.
- **H-T predicts the colour head degrades anyway.** The depth is higher than
  anything the small sizes ever had.

One cell, two incompatible predictions, and the bar between them was written
down before the job was submitted.

---

## Nothing numeric carries across the two ensembles, and nothing is quoted

$(d,q)=(4,3)$ is a different instance family, not another point on the old
curve. Its chance level, critical density and hint floor all differ, and its
sequences are shorter at the same $N$.

So this stage measures its own constants or reports none, and the comparison
axis between runs is the DD depth $T$ — never the hint density, whose numeric
value means different things in the two families, and never a pooled solve
rate, whose $n_{\text{blank}}$ differs.

---

## Five predictions were registered in git before the array was submitted

| | prediction | falsifier |
|---|---|---|
| **D1** | `reason_color` at $N=96$: H-N says $< 0.10$ nats, H-T says $\ge 0.55$ | the 0.10–0.55 band is **registered in advance as partial**, decided by D2's direction |
| **D2** | $\mathrm{NLL}(N{=}120) \ge \mathrm{NLL}(N{=}96)$, as $N{=}120$ is deeper | $N=120$ better by more than 0.02 nats |
| **D3** | the forced graph and terminator sections stay learned, $< 0.01$ | either above 0.05 ⇒ the cell is sick and **D1/D2 are unreadable** |
| **D4** | some bin below $\rho_c$ reaches `solve_rate` $> 0.1$ | nothing solves below $\rho_c$ on either seed |
| **D5** | still improving at epoch 50, no overfitting turn-up | flat at the initial plateau for all 50 epochs |

Registering the partial band in advance is the part that matters: it is what
stops a middling result from being rounded afterwards to whichever hypothesis
is convenient.

---

## A null result here licenses nothing except reporting it

If D1 lands at chance and D4 fails, the honest conclusion is that the colour
collapse tracks depth, the lever does not avoid it, and the size ladder and
the depth lever hit the same wall.

That would **not** license a wider model, a longer schedule, more data or a
fifth size — all of those were tested and closed earlier. It would point at
changing the trace format itself, which is a decision for hshunt and not one
this run gets to make.

---

## Four data checks ran on CPU before any GPU time was spent

Because $q=3$ had never been trained here, the generator and tokeniser were
checked first, on 40 instances per cell, at $N=96$, $N=120$, and a control
from the previous ensemble:

- **F1** every deduction is genuinely forced by its neighbours as of the
  previous round — 0 bad.
- **F2** every deduced colour equals the planted colour — 0 bad.
- **F3** the context that forces a deduction has already been emitted, so
  nothing is deducible only from the future — 0 bad.
- **F4** every node id round-trips through the shifted vocabulary — 0 bad.

The measured marginal on `reason_color` came out at 1.0985 against
$\ln 3 = 1.098612$, confirming the chance level empirically.

---

## Three more checks guard the scoring code, not the data

- **DQ-P0** the entropy estimator returns $\log|V|$ on a uniform
  distribution, in both float32 and float64.
- **DQ-P1** the measurement code is re-run, through this stage's own
  modified configuration, on an **already published** checkpoint, and must
  reproduce the published per-section numbers. It reproduced 20 of 20
  quantities to $10^{-8}$.
- **DQ-P2** every checkpoint scored must report $|V| = N + 6 + 3$, so a
  checkpoint from the other ensemble cannot be scored against $\ln 3$ by
  accident and made to look spectacular.

The scorer carries no constant from the other ensemble, and a separate check
confirms it *refuses* to read that ensemble's files rather than merely
avoiding them.

---

<!-- RESULTS PAGES GO HERE when 53243550 and 53264789 land:
     D5 trainability curve; D3 health; D1 per seed; D1 per density; D1
     verdict; D2; D4 solve-vs-rho; D4 a_eff vs depth with the CP bounds
     drawn as upper arrows; and a final page comparing against the previous
     best ON THE SAME EVAL, or stating that there is none because this
     ensemble has never been run before -- which is the likely outcome and
     must be said plainly rather than skipped. Record the git commit, the
     config and the seeds on the last page. -->
