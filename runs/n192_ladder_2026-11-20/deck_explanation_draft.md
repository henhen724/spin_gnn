# Explanation pages for the N=192 ladder deck — DRAFT

Written 2026-11-21, while 53062385 was still training, so nothing here can have
been shaped by the results. These pages are results-independent by
construction: they define the task, the data, the tokenisation, the model and
every reported quantity. The results pages get appended after the run lands,
and then the whole file becomes
`~/slides/decks/spin-gnn/n192-ladder/deck.md`.

The worked example on the "complete instance" page is a REAL encoding, produced
by `spin_gnn.sudoku_tokens.encode_instance` at `n_nodes=12, degree=5,
n_colors=4, rho=0.55, fall=True, shuffle_adj=True, seed 12345`. Its three
deductions were checked by hand against the adjacency rows printed on the same
page. Do not replace it with a hand-written one.

---

## Colour the graph so that no edge joins two equal colours

The model is given a graph $G = (V, E)$ with $N = \lvert V \rvert$ nodes. Every
node must be assigned one of $q$ colours so that no edge has the same colour at
both ends. This deck uses $q = 4$ colours throughout.

The graph is **random $d$-regular**: every node has exactly $d$ neighbours, and
subject to that the edges are drawn uniformly at random. This deck uses
$d = 5$ throughout.

"Cell" and "node" mean the same thing here. The Sudoku vocabulary is used
because the puzzle a reader meets is the same shape: some cells are revealed,
the rest must be worked out from the constraint that neighbours differ.

---

## Every instance is built around a planted solution, so an answer exists

Instances are not sampled and then tested for solvability. The generator first
draws a proper $q$-colouring, then draws a $d$-regular graph consistent with
it. The colouring is **planted**: it is known to the generator before the graph
is finished.

This matters for reading every number in the deck. A failure is never the
puzzle being impossible — a correct answer always exists and the generator
knows it. A failure is the model not finding it.

The construction needs the colour classes to come out equal and the edges to
divide evenly among ordered pairs of distinct colours, which requires
$q \mid N$ and $q(q-1) \mid dN$. At $q=4$, $d=5$ this reduces to $12 \mid N$,
and $192 = 12 \times 16$ qualifies.

---

## A fall colouring makes every cell deducible in principle

The planted colouring is not just proper, it is a **fall colouring**: every
node has at least one neighbour of **every** other colour. Equivalently, each
colour class is an independent *dominating* set (Dunbar et al., 2000).

This is forced because of how direct deduction works. A cell is only ever
deduced once its neighbours display all $q - 1$ other colours — so a cell whose
neighbourhood is missing a colour can **never** be deduced, and has to stay a
hint forever.

Under an unconstrained planted colouring that happens to a fixed fraction of
nodes: **0.3827** of them at $q=4$, $d=5$, by inclusion–exclusion over which
colour classes a node misses. Those permanently-hinted cells put a hard
structural floor under $\rho_{\min}$, stalling hint removal at about 0.475.

Since $\rho_c \approx 0.46$, that floor sits **above** the transition: without
fall colouring the slow phase is unreachable at any size, and this deck's whole
question is unaskable. Fall colouring removes the floor and lets $\rho_{\min}$
fall to 0.320. It requires $d \ge q - 1$.

---

## Hints are removed only while deduction can still finish the puzzle

A **hint** is a cell whose colour is revealed before the puzzle starts.

The generator begins with every cell hinted and removes them in a random order,
accepting a removal only if deduction can still recover the entire colouring
from what is left. The walk stops when no further removal survives that test.

So an instance is never merely solvable in principle — it is solvable **by the
specific procedure defined on the next pages**, which is what makes the model's
target well defined.

One consequence is used later: if a target density is below what a particular
removal path can reach, the generator stops at that path's sparsest point
instead. Achieved density can therefore overshoot the target upward, never
downward.

---

## Hint density $\rho$ is the fraction of cells revealed at the start

$$\rho = \frac{\text{number of hinted cells}}{N}, \qquad
n_{\text{blank}} = N(1 - \rho)$$

$\rho = 1$ is a fully revealed puzzle with nothing to do. $\rho \to 0$ is a
blank grid. $n_{\text{blank}}$ is the number of cells the model must fill.

$\rho$ is the **difficulty axis of this entire deck**. Almost every plot has it
on the x-axis, and every quantity defined below is quoted at a stated $\rho$ or
as a function of it.

Each size has a floor it cannot go below, because at some point no removal
path survives: $\rho_{\min}$ is 0.333 at $N = 48$, 0.325 at $N = 96$ and
$0.320 \pm 0.013$ at $N = 192$.

---

## Direct deduction fills any cell whose colour its neighbours have forced

**Direct deduction (DD)** is the one rule used to solve these puzzles, and the
one the model is trained to imitate. For a blank cell, look at the colours
already assigned to its $d$ neighbours. Any colour appearing there is
unavailable. If exactly one of the $q$ colours survives, the cell is **forced**
and takes it.

$$\text{cell } v \text{ is forced} \iff
\lvert \{1 \ldots q\} \setminus \{\text{colours of } v\text{'s neighbours}\}
\rvert = 1$$

DD never guesses, never branches and never backtracks. If no cell anywhere is
forced, DD simply stops. This is the "naked single" rule a Sudoku solver uses,
transplanted to a graph.

---

## One round fills every forced cell at once; $T$ counts rounds

A **round** applies the rule above to all cells simultaneously: find every cell
that is currently forced, fill all of them, then look again. Cells unlocked by
this round's fills are not taken until the next round.

**$T$** is the number of rounds needed to finish — the **depth** of the
deduction. The **width** $w_t$ of round $t$ is how many cells it fills.

Since every blank cell is filled in exactly one round,

$$\sum_{t=1}^{T} w_t = n_{\text{blank}}$$

$T$ is the deck's measure of how much sequential reasoning an instance demands:
a puzzle with $T = 3$ needs three waves of inference, one depending on the last.

---

## Below $\rho_c \approx 0.46$ the depth $T$ stops growing logarithmically

This is the phenomenon the ladder exists to reach. As $\rho$ falls, DD depth
changes character at a critical density $\rho_c \approx 0.46$:

$$T \sim \log N \quad (\rho > \rho_c, \text{ the FAST phase}) \qquad
T \sim N^{\gamma} \quad (\rho < \rho_c, \text{ the SLOW phase})$$

Above $\rho_c$ hints are dense enough that deduction sweeps the graph in a few
waves. Below it, information has to propagate along chains, and depth grows as
a power of the size.

**$\gamma \approx 0.727$** is this project's measured exponent for the slow
phase. It is a *local* exponent taken at $\rho_{\min}(N)$, a density that
itself moves with $N$ — so it must not be compared against an exponent fitted
at a held-fixed $\rho$.

The instances reach the slow phase at every size here, since $\rho_{\min}$ is
below $\rho_c$ at all three rungs. Whether the trained models do is the
question.

---

## The whole puzzle is flattened into one left-to-right token sequence

There is no graph encoder. The model is a plain decoder-only transformer over a
token sequence with four sections in a fixed order:

$$\texttt{BEGIN\_HINTS}\; (\textit{node}\; \textit{colour})^{*}$$
$$\texttt{BEGIN\_GRAPH}\; (\textit{node}\; \textit{node}^{5})^{*}$$
$$\texttt{BEGIN\_REASONING}\;
(\texttt{BEGIN\_ROUND}\; (\textit{node}\; \textit{colour})^{+})^{*}$$
$$\texttt{SOLVED}$$

Hints come first, then the graph as one 6-token row per node (the node, then
its 5 neighbours), then the deduction trace as one group per round, then a
terminator. A trace that fails to finish ends `UNRESOLVED` followed by the
cells left blank, instead of `SOLVED`.

Two details are load-bearing later: **each undirected edge is written twice**,
once in each endpoint's row; and within a round, cells are emitted **sorted by
node index**, so the trace has no ordering freedom left to guess.

---

## A complete instance at $N = 12$, all 103 tokens of it

```
[BEGIN_HINTS]
  N0=COLOR_1  N2=COLOR_1  N7=COLOR_0  N8=COLOR_2  N9=COLOR_2  N10=COLOR_0
[BEGIN_GRAPH]
  N0  -> N6  N7  N5  N9  N8        N6  -> N4  N9  N0  N2  N10
  N1  -> N3  N5  N8  N10 N4        N7  -> N11 N9  N0  N2  N3
  N2  -> N4  N11 N7  N3  N6        N8  -> N0  N10 N1  N5  N11
  N3  -> N4  N11 N7  N1  N2        N9  -> N6  N0  N7  N10 N11
  N4  -> N1  N6  N5  N2  N3        N10 -> N9  N8  N6  N1  N5
  N5  -> N8  N4  N1  N0  N10       N11 -> N8  N9  N7  N3  N2
[BEGIN_REASONING]
  [BEGIN_ROUND]  N5=COLOR_3   N6=COLOR_3   N11=COLOR_3
  [BEGIN_ROUND]  N1=COLOR_1   N3=COLOR_2
  [BEGIN_ROUND]  N4=COLOR_0
[SOLVED]
```

A real encoding, not an illustration: $N = 12$, $d = 5$, $q = 4$, $\rho = 0.55$,
6 hints, 6 blanks, $T = 3$ rounds of widths 3, 2, 1. The model sees exactly
this, as integers. $N = 192$ is the same object about 15 times longer.

---

## Walking round 1: three cells have only one colour left

Take **N5** from the example. Its row reads `N5 -> N8 N4 N1 N0 N10`, and the
hints give `N8=COLOR_2`, `N0=COLOR_1`, `N10=COLOR_0`. Three of the four colours
are already spoken for among its neighbours, so N5 is forced to **COLOR_3**.
N4 and N1 are still blank and simply contribute no constraint.

**N6** (`N4 N9 N0 N2 N10`) sees `N9=COLOR_2`, `N0=COLOR_1`, `N2=COLOR_1`,
`N10=COLOR_0` — forced to **COLOR_3**. **N11** (`N8 N9 N7 N3 N2`) sees
`COLOR_2`, `COLOR_2`, `COLOR_0`, `COLOR_1` — forced to **COLOR_3**.

Those three are every forced cell at the start, so they are round 1, and they
are emitted in sorted order: N5, N6, N11.

Round 2 only becomes possible because of round 1: **N1** (`N3 N5 N8 N10 N4`)
needed `N5=COLOR_3` before `COLOR_1` was its only option. That dependency of
one round on the last is exactly what $T$ counts.

---

## Vocabulary is $N+10$ symbols, sequence length $8N+T+4$

The vocabulary is 6 structural symbols (`BEGIN_HINTS`, `BEGIN_GRAPH`,
`BEGIN_REASONING`, `BEGIN_ROUND`, `SOLVED`, `UNRESOLVED`), $q = 4$ colours, and
one symbol per node:

$$\lvert V_{\text{tok}} \rvert = N + q + 6 = N + 10$$

58 symbols at $N=48$, 106 at $N=96$, **202 at $N=192$**. Length follows from the
grammar — $N(d+1)$ graph tokens, $2N$ hint-and-fill tokens, $T$ round markers
and 4 section markers:

$$L = N(d + 3) + T + 4 = 8N + T + 4 \quad (d = 5)$$

103 for the $N=12$ example above, and about **1540** at $N=192$. The longest
training sequence measured was 1598, and `max_len` was set to 1984 — a 386-token
generation margin. This is the first run in the programme to need a context
near 2000.

---

## This rung is a 12-layer decoder trained on 240,000 instances

| setting | value |
|---|---|
| layers / $d_{\text{model}}$ / heads / feed-forward | 12 / 256 / 4 / 1024 |
| normalisation | pre-norm |
| parameters | 10,089,162 trainable |
| learning rate | $10^{-3}$, constant, no warmup |
| batch size | 32 |
| training instances $\times$ epochs | $240{,}000 \times 50$ = 375k steps |
| ensemble | $N{=}192$, $d{=}5$, $q{=}4$, fall, shuffled adjacency |
| seeds | 0 and 1, two independent cells |

Training is next-token prediction over the whole sequence. **Every one of these
settings is identical to the $N=48$ and $N=96$ rungs except $n\_nodes$** — the
ladder changes size and nothing else, so any difference is attributable.

"fall" is the colouring ensemble defined earlier — the one that keeps every
cell deducible. "Shuffled adjacency" randomises the order of the 5 neighbours
inside each graph row, so that order carries no signal to learn.

---

## $\mathrm{val\_nll}$ is the mean negative log-likelihood per held-out token

$$\mathrm{val\_nll} = -\frac{1}{L}\sum_{i=1}^{L} \log p_\theta(x_i \mid x_{<i})$$

in **nats per token**, averaged over held-out instances the model never trained
on. Lower is better. A model that assigned every token its correct probability
would score the data's own entropy, not zero.

It is the quantity the training curve plots and the one used to judge whether a
cell is still improving. It is **not** a measure of whether puzzles get solved —
that is the solve rate, two pages on. A model can improve its `val_nll` by
getting better at the easy bulk of the sequence while solving nothing.

---

## Most of $\mathrm{val\_nll}$ is a floor the model cannot get below

A large part of the sequence is genuinely unpredictable: the graph section lists
each node's neighbours, and in a random $d$-regular graph those are random. No
amount of training predicts them. So `val_nll` cannot fall to zero, and the
floor **grows with $N$** because there are more possible neighbours to name:

$$f_{\text{graph}}(N) = \tfrac{5}{16}\log N$$

Estimated floors are **1.371 / 1.625 / 1.861** nats at $N = 48 / 96 / 192$.

What is comparable across sizes is the **excess** above that floor:

$$\text{excess} = \mathrm{val\_nll} - f(N)$$

The deduction half of the sequence has a floor of exactly **zero** at every $N$,
because within-round order is sorted and each colour is forced — so any loss
there is pure excess and needs no floor model at all.

---

## Solve rate counts only puzzles where every blank cell came out right

$$\text{solve rate} = \Pr[\text{all } n_{\text{blank}}
\text{ cells correct}]$$

It is all-or-nothing per puzzle: one wrong cell out of 125 scores the same as
125 wrong. It is measured on a grid of densities, 400 fresh instances at each,
with seeds disjoint from training.

This strictness creates a trap the next page exists to fix. Solving needs
roughly $n_{\text{blank}}$ consecutive correct deductions, and
$n_{\text{blank}} = N(1-\rho)$ grows with $N$. **So solve rate must fall as $N$
grows even if the model's per-step skill is identical**, and a raw solve-rate
comparison across sizes measures mostly arithmetic.

---

## $a_{\mathrm{eff}}$ converts a solve rate into a per-deduction accuracy

$$a_{\text{eff}} = (\text{solve rate})^{1/n_{\text{blank}}},
\qquad n_{\text{blank}} = \mathrm{round}(N(1-\rho))$$

This is the per-deduction accuracy that would reproduce the observed solve rate
if the deductions were independent. It is well defined because
$n_{\text{blank}}$ is constant inside a density bin.

It divides out the length penalty of the previous page, so it **can** be
compared across sizes. $a_{\text{eff}} = 0.99$ means about one error per hundred
cells, whatever $N$ is.

Even cross-$N$, it is compared at **matched DD depth** $T$, not at matched
$\rho$, because the three rungs use different density grids and a given $\rho$
is not equally hard at different sizes.

---

## $\rho_{50}$ is the density at which the model solves half the puzzles

Read the solve-rate curve across the density grid and find where it crosses
$0.5$:

$$\rho_{50} = \rho \;\text{ such that }\; \text{solve rate}(\rho) = 0.5$$

**Lower is better.** A model with $\rho_{50} = 0.55$ still solves half of
puzzles that reveal only 55% of cells; one with $\rho_{50} = 0.70$ needs 70% of
the answer handed to it.

$\rho_{50} \le 0.740$ is this project's **gate** — a threshold a model must
clear to count, not a score to maximise.

Two reference points: at $N = 96$ the two seeds gave $\rho_{50} = 0.679$ and
$0.564$. Both sit **above** $\rho_c \approx 0.46$, meaning those models fail
before the slow phase begins. $\rho_{50}$ is undefined when the curve never
crosses 0.5 within the grid, and is then reported as censored rather than
extrapolated.

---

## $s^{*}$ measures how much of a result is just the random seed

$$s^{*} = \lvert \rho_{50}(\text{seed }0) - \rho_{50}(\text{seed }1) \rvert$$

Two cells identical but for the seed give two different $\rho_{50}$ values, and
$s^*$ is the gap. It is the yardstick any claimed improvement has to beat: a
change that moves $\rho_{50}$ by less than $s^*$ has not been shown to do
anything.

$s^*$ is **specific to a size** and is never carried across rungs. It measured
$0.1484$ at $N = 48$ and $0.119$ at $N = 96$; $N = 192$ gets its own, which is
why this rung buys two seeds rather than one.

That is a large number. At $N = 96$ the seed gap of 0.119 is comparable to the
gap between the sizes, which is why the two-seed design is not optional here.

---

## The ladder's three rungs change $N$ and nothing else

| | $N = 48$ | $N = 96$ | $N = 192$ |
|---|---|---|---|
| layers | 12 | 12 | 12 |
| instances $\times$ epochs | 240k $\times$ 50 | 240k $\times$ 50 | 240k $\times$ 50 |
| vocabulary | 58 | 106 | 202 |
| sequence length | ~390 | ~775 | ~1540 |
| `val_nll` floor | 1.371 | 1.625 | 1.861 |
| $\rho_{\min}$ reachable | 0.333 | 0.325 | 0.320 |
| GPU hours per cell | — | ~13.6 | ~29.3 |

Everything above the line is held fixed by design. The cost line is the reason
the ladder stops at three rungs: $N{=}192$ is 2.15$\times$ $N{=}96$, tracking
the linear bound rather than the quadratic one, because attention does not yet
dominate at this length.

---

## Five predictions were registered before either cell started training

| # | prediction | falsifier |
|---|---|---|
| N1 | $a_{\text{eff}}$ at $T \approx 12$ is below $N{=}96$'s worse seed, 0.9118 | it lands at or above 0.9626 |
| N2 | $\rho_{50} > \rho_c = 0.46$ for both seeds | $\rho_{50} \le 0.46$ for either |
| N3 | $s^{*}_{192} \ge s^{*}_{96} = 0.119$ | $s^{*}_{192} < 0.119$ |
| N4 | both cells' `val_nll` still descending at epoch 50 | either turns up |
| N5 | truncation rate $\le 0.01$ wherever solve rate $> 0.1$ | it exceeds 0.01 there |

**This grid was registered as expected to show the model getting WORSE**, so
that outcome could not later be reported as a surprise or as a bad run. N1 and
N2 both predict degradation.

N4 deliberately counts no "escape" from a loss plateau, because $N{=}192$'s
plateau has never been measured and the $N{=}48$ figure does not transfer. Its
breaking point is stated too: if both cells sit flat for all 50 epochs, then
$N{=}192$ is not trainable at this budget, and that is the result — not
something to retry at a larger size.

---
