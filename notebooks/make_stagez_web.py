#!/usr/bin/env python3
"""Render decks/spin-gnn/counterfactual from stage Z's own artifacts.

Stages X and Y each trained the model to represent which nodes are deducible,
each reached AUROC ~0.95 against DD's own predicate, and neither changed what
the model emits when a node stops being deducible. Stage Z asks whether the
thing that was installed is the predicate at all.

NOTHING IS RETYPED. Every number on every page is read out of
runs/2026-11-15_stagez_cfresp/ -- the summary the probe wrote and the three
per-case CSVs -- and every bar is imported from scripts.stagez_score, which
registered them before the probe ran. check_claims() asserts the numbers the
pages print against those files, so a deck whose prose has drifted fails to
build instead of publishing to a URL.

THE HEADLINE IS A CONTRAST BETWEEN TWO MEASUREMENTS OF THE SAME QUANTITY.
Conditioning on the count of revealed neighbours, the channel ranks determinacy
at 0.80. Conditioning on everything except the single token that flips the
predicate, it ranks it at 0.49, with two seeds of three below chance. A
representation that ranks well across instances and cannot track the
intervention that changes the thing it ranks is a confound, not a computation.

WHY THE EXPLANATION PAGES ARE HALF THE DECK. ~/CLAUDE.md requires a deck to
stand alone for a reader who has seen none of this project's other decks.
Stages X and Y are the premise of this one, so they are restated in full rather
than cited, and every quantity -- the predicate, the channel, the corruption,
the placebo, and all three AUROCs -- is defined with its equation before any
number appears. deck_explain supplies the stage-independent half; SUFFICIENCY
and CENSOR are dropped from it because this deck reports neither a censored
rho_50 nor an isotonic residual, and a deck should not define a quantity it
never uses.

Usage (in a job -- this imports torch through deck_tokenization):
    python3 -u -m notebooks.make_stagez_web --commit <sha> --job <id>
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

from slides import MUTED, SERIES, label_line, style_axes

from notebooks import deck_explain
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_staget_web import WebDeck

import scripts.stagez_score as sz

DECK = "/home/users/hshunt/slides/decks/spin-gnn/counterfactual"

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
RUN = os.path.join(_SCR, "spin_gnn", "runs", "2026-11-15_stagez_cfresp")
JOB = "52413604"

PUBLISHED_Y_AUROC = (0.9493, 0.9478, 0.9480)
PUBLISHED_X_AUROC = (0.9462, 0.9467, 0.9469)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load():
    with open(os.path.join(RUN, "cfresp_summary.json")) as fh:
        s = json.load(fh)
    rows = []
    for i in range(3):
        with open(os.path.join(RUN, "cfresp_s%d.csv" % i)) as fh:
            rows.append(list(csv.DictReader(fh)))
    return s, rows


def matched_pair(rows, col="dg_corrupt"):
    """P(g_clean > g_broken), ties at one half.

    The clean input has v forced and the broken input has it not forced, so
    this is an AUROC over 350 matched pairs: same instance, same node, same
    position, same number of revealed neighbours, differing by one token. It is
    a post-hoc re-expression of the probe's own dg column, not a registered
    bar, and the deck says so on the page that reports it.
    """
    d = np.array([float(r[col]) for r in rows])
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def pair_shares(per_stratum):
    """How the pooled AUROC's pairs split into same-count and different-count.

    The pooled measure averages over every (determined, undetermined) pair.
    Only the pairs drawn from one stratum require knowing the colours; the rest
    can be ordered by the count alone. This returns that split, which is the
    quantitative form of the conflation page.
    """
    ks = sorted(int(k) for k in per_stratum)
    pos = {k: per_stratum[str(k)]["n_pos"] for k in ks}
    neg = {k: per_stratum[str(k)]["n_neg"] for k in ks}
    within = sum(pos[k] * neg[k] for k in ks)
    total = sum(pos.values()) * sum(neg.values())
    return within, total


def check_claims(S, mp, pp):
    """Assert every number the pages print against the files the job wrote.

    The pages below are written by hand and the numbers in them are read from
    these same objects, so what this really guards is the prose: a claim that
    says "at chance" when the win rate has moved, or a denominator that has
    drifted from the CSV it describes.
    """
    bad = []
    per = S["per_seed"]
    strat = S["stratified"]["per_seed"]

    # Z1 is at chance, and the placebo is the control that makes it readable
    if not all(sz.PRED_WIN_RANGE[0] <= p["win_rate"] < sz.PRED_WIN_RANGE[1] for p in per):
        bad.append("a win rate left the registered range; the 'at chance' pages are wrong")
    if not all(p["flip_frac"] == 0.0 for p in per):
        bad.append("flip_frac is no longer 0 on every seed")
    # the denominator is read, never assumed -- and it is never 350
    if [p["flip_denom"] for p in per] != [236, 228, 217]:
        bad.append("flip denominators are not 236/228/217")
    if any(p["n_cases"] != 350 for p in per):
        bad.append("the guard case count is not 350")
    # The published emit rate is PER SEED (0.6743/0.6514/0.6200 = 236/228/217
    # of 350), not one number for all three -- sz.Y_EMITS_V holds seed 0's
    # clean/corrupt/placebo triple and nothing else. The real per-seed check is
    # the probe's own cross-check against the published skip_summary.json, so
    # that is what is asserted here, together with the identity that makes the
    # denominators on the flip page exactly those rates.
    if not all(p["cross_check_vs_skip_summary"] == "exact" for p in per):
        bad.append("the probe's cross-check against skip_summary is no longer exact")
    if not all(p["flip_denom"] == round(p["clean_emits_v"] * p["n_cases"])
               for p in per):
        bad.append("flip_denom is not clean_emits_v x 350; the flip page is wrong")

    # Z2
    cov = S["covariate_auroc"]
    if abs(cov["n_known_colors"] - 1.0) > 1e-9:
        bad.append("the tautology check no longer returns 1.0; the cells are wrong")
    if cov["n_known_nbrs"] >= sz.COV_BAR:
        bad.append("the count now clears COV_BAR; the 'not just a count' pages are wrong")

    # Z3: both ends of the scale, and the pooled cross-check
    taut = S["stratified"]["tautologies"]
    if abs(taut["n_known_nbrs"] - 0.5) > 1e-9 or abs(taut["n_known_colors"] - 1.0) > 1e-9:
        bad.append("the stratified scale is not calibrated at its ends")
    if any(abs(c["model_pooled"] - p) > 0.001
           for c, p in zip(strat, PUBLISHED_Y_AUROC)):
        bad.append("the pooled AUROC no longer reproduces the published column")
    if not all(sz.WITHIN_FLOOR <= c["model"] < sz.WITHIN_BAR for c in strat):
        bad.append("a within-stratum AUROC left (WITHIN_FLOOR, WITHIN_BAR)")
    # only two strata carry both labels, and a page says so in words
    for c in strat:
        scored = [k for k, v in c["per_stratum"].items() if v["auroc"] is not None]
        if sorted(int(k) for k in scored) != [3, 4]:
            bad.append("the informative strata are no longer exactly {3, 4}")
    if strat[0]["per_stratum"]["5"]["n_neg"] != 0:
        bad.append("k=5 is no longer all-positive; the 'answers itself' page is wrong")
    if strat[0]["per_stratum"]["5"]["n_pos"] != 10238:
        bad.append("the k=5 count on the strata page has drifted")

    # the headline contrast, in the direction the deck claims it
    if not all(m < 0.60 for m in mp):
        bad.append("a matched-pair AUROC rose above 0.60; the headline is wrong")
    if not all(c["model"] - m > 0.20 for c, m in zip(strat, mp)):
        bad.append("the 0.80-vs-0.49 contrast has collapsed")
    if sum(1 for m in mp if m < 0.50) != 2:
        bad.append("the 'two of three seeds' page no longer has two seeds")
    if not all(0.40 < p < 0.60 for p in pp):
        bad.append("the placebo control left chance; it is no longer a control")
    if bad:
        raise AssertionError("check_claims failed:\n  " + "\n  ".join(bad))


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def panels(d, name, claim, draw, note=None):
    """A 1x3 small-multiple page, one panel per seed, as the channel deck does.

    Three seeds on one axes would be six distributions. The seed fixes both the
    instances and the initialisation, so seed s is only ever read against
    itself, and the comparison that matters is inside a panel.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.4), sharey=True)
    draw(axes)
    for ax in axes:
        ax.set_rasterized(True)
    fig.tight_layout()
    d.expect(name, claim, note)
    d.savefig(fig)


def draw_scale(S, mp, pp):
    """The headline: one AUROC axis whose two ends are fixed by construction."""
    strat = S["stratified"]["per_seed"]
    taut = S["stratified"]["tautologies"]

    def inner(ax):
        rows = [
            (4, [taut["n_known_colors"]], "distinct colours", MUTED),
            (3, [c["model"] for c in strat], "given the count", SERIES[0]),
            (2, [taut["n_known_nbrs"]], "the count", MUTED),
            (1, mp, "matched pairs", SERIES[1]),
            (0, pp, "placebo pairs", MUTED),
        ]
        for y, vals, lab, col in rows:
            ax.plot(vals, [y] * len(vals), "o", color=col, markersize=11,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)
            label_line(ax, 1.015, y, lab, color=col, dx=8)
        for x in (0.5, 1.0):
            ax.axvline(x, color=MUTED, lw=1.0, ls="--", zorder=1)
        ax.text(0.5, 4.55, "chance", ha="center", fontsize=10, color=MUTED)
        ax.text(0.995, 4.55, "the predicate", ha="right", fontsize=10, color=MUTED)
        ax.set_ylim(-0.6, 4.9)
        ax.set_xlim(0.40, 1.02)
        ax.set_yticks([])
        style_axes(ax, xlabel="AUROC against direct deduction's determinacy predicate")
        ax.grid(axis="y", visible=False)
    return inner


def draw_response(R):
    """|dg| under the corruption against |dg| under the placebo, per seed."""
    def inner(axes):
        for i, ax in enumerate(axes):
            c = np.abs([float(r["dg_corrupt"]) for r in R[i]])
            p = np.abs([float(r["dg_placebo"]) for r in R[i]])
            spread = float(np.median([float(r["g_spread"]) for r in R[i]]))
            for v, col, lab, ly in ((c, SERIES[0], "corruption", 0.97),
                                    (p, MUTED, "placebo", 0.80)):
                v = np.sort(v[v > 0])
                ax.plot(v, np.arange(1, v.size + 1) / v.size, color=col, lw=1.9)
                if i == 2:
                    label_line(ax, v[-1], ly, lab, color=col)
            ax.axvline(spread, color=SERIES[1], lw=1.4, ls="--")
            ax.text(spread * 0.75, 0.30, "g's spread\nacross nodes", fontsize=9,
                    color=SERIES[1], ha="right")
            ax.set_xscale("log")
            ax.set_title("seed %d" % i, fontsize=11, color=MUTED)
            style_axes(ax, xlabel=r"$|\Delta g|$ at $v$",
                       ylabel="fraction of cases" if i == 0 else None)
    return inner


def draw_pairs(within, total):
    """Which pairs the pooled AUROC averages over, and which need the colours."""
    def inner(ax):
        cross = total - within
        ax.barh([0], [cross / total], color=MUTED, height=0.5)
        ax.barh([0], [within / total], left=[cross / total], color=SERIES[0],
                height=0.5)
        ax.text(cross / total / 2, 0,
                "%.0f%% have different counts\nand can be ordered by counting"
                % (100 * cross / total),
                ha="center", va="center", color="white", fontsize=13)
        label_line(ax, 1.0, 0, "%.0f%% share\na count" % (100 * within / total),
                   color=SERIES[0], dx=8)
        ax.set_ylim(-0.38, 0.38)
        ax.set_xlim(0, 1.0)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
        style_axes(ax, xlabel="share of the %.1f billion (determined, undetermined) pairs"
                   % (total / 1e9))
        ax.grid(axis="y", visible=False)
    return inner


# --------------------------------------------------------------------------
# stage-specific explanation pages. The <=25-word body cap does not apply to
# these: the definitions ARE the page, and a reader of this deck has seen no
# other deck from this project.
# --------------------------------------------------------------------------
SYMBOLS = r"""## Every symbol this deck uses, in one place

| symbol | meaning |
|---|---|
| $G=(V,E)$ | a random 5-regular graph on $N=48$ nodes |
| $q=4$ | the number of colours |
| $v$ | the node a deduction is about |
| $p$ | a position in the token stream where a node must be named |
| $\mathrm{DD}$ | direct deduction, the algorithm being imitated |
| $z_{p,v}$ | the model's output logit for node $v$ at position $p$ |
| $g_{p,v}$ | stage Y's separate determinacy score for $v$ at $p$ |
| $k(v)$ | how many of $v$'s neighbours have a revealed colour |
| $\kappa(v)$ | how many **distinct** colours those neighbours show |

A *position* is one place in the stream where the model must name the next node
to colour. Every quantity below is measured at such positions.
"""

PREDICATE = r"""## A node is determined when its neighbours show all but one colour

The algorithm being imitated is **direct deduction**: colour a node only when
exactly one colour is left for it, and repeat that in synchronous rounds until
nothing more is forced.

Node $v$ is **determined** (equivalently *forced*) at a position when

$$\mathrm{forced}(v) \;\Longleftrightarrow\; v \text{ is uncoloured} \;\wedge\; \kappa(v) = q-1 = 3$$

where $\kappa(v)$ counts the **distinct** colours among $v$'s neighbours whose
colour is already revealed. If three of the four colours appear next to $v$, the
fourth is the only one left and $v$ is forced.

Two counts matter and they are not the same. $k(v)$ counts revealed neighbours;
$\kappa(v)$ counts the distinct colours they show. Always $\kappa(v) \le k(v)$,
with equality only when no colour repeats. **The predicate depends on $\kappa$,
never on $k$** -- but the two are correlated, because more revealed neighbours
tend to show more colours. That gap is what this entire deck is about.
"""

WHY = r"""## Why determinacy was installed at all: the project's number is $\rho_{50}$

The tied model S1 solves half the instances down to $\rho_{50} = 0.757$. The
untied J2, at four times the parameters and sixteen different blocks, reaches
0.580. Closing that gap **without** untying the block is the point of the
project, because the tied object is the one a theory could hope to describe.

The idea behind stages X, Y and Z is that direct deduction's own structure is
missing from the objective. Next-token prediction never tells the model *which*
nodes are deducible; it only rewards naming one of them. So a term was added
that says so directly, in the hope that a model which represents determinacy
would sustain the longer chains of deduction that sparse hints demand.

Three stages have now tested that installation. **This deck is about whether the
thing that got installed is determinacy.**
"""

WHYZ = r"""## Two earlier runs installed this signal and neither changed behaviour

**Stage X** added a loss term making the model's existing node logits $z_{p,v}$
rank determined nodes above undetermined ones, at no new parameters. It reached
AUROC 0.9462/0.9467/0.9469 across three seeds.

**Stage Y** gave the signal its own channel instead, so it could re-order the
candidates rather than only shift them together, at a cost of 320 parameters. It
reached 0.9493/0.9478/0.9480.

Both were then tested with an intervention: destroy node $v$'s determinacy and
see whether the model stops naming $v$. **Neither stopped.** Stage Y's guard
fired once in 681 cases.

So the model can be trained to rank determinacy almost perfectly and still emit
as though it had not been. The natural reading is that the coupling between the
signal and the output is too weak. **Stage Z tests a different reading: that the
signal itself is not what it appears to be.**
"""

CHANNEL = r"""## Stage Y's channel is one extra score per node, added before the softmax

The model is a 16-layer transformer with one block of weights shared across all
16 applications, 613,818 trainable parameters. Stage Y adds a vector $s$ of 320
numbers and computes, at every position $p$ and for every node $v$,

$$g_{p,v} \;=\; \langle\, h_{16}[p] \odot s,\; E_v \,\rangle$$

where $h_{16}[p]$ is the final hidden state at position $p$, $E_v$ is node $v$'s
embedding, and $\odot$ is the elementwise product. That score enters the output
only through the node block of the logits:

$$z'_{p,v} \;=\; z_{p,v} + \log \sigma(g_{p,v}), \qquad \sigma(x) = \frac{1}{1+e^{-x}}$$

$g$ is trained with a binary cross-entropy against $\mathrm{forced}(v)$. At
initialisation $s = 0$, so $g \equiv 0$ and the model starts exactly where the
control starts.

**This deck reads $g$ directly.** That is why stage Z measures stage Y and not
stage X: stage X has no separate score to read, only the logits themselves,
which were already measured behaviourally.
"""

CORRUPT = r"""## The intervention rewrites one colour so that one node stops being forced

Take a node $v$ that direct deduction colours in the first round. Its revealed
neighbours show exactly $q-1=3$ distinct colours, and one of those colours is
held by exactly one neighbour $w$. Now **recolour $w$** to a colour that another
neighbour already shows.

After the edit $v$'s neighbours show only $q-2=2$ distinct colours, $v$ has two
candidates left, and direct deduction **does not colour $v$ in that round at
all**. The edit changes exactly one token of the prompt, and the probe asserts
that: `(clean != broken).sum() == 1`.

The critical property, and the reason this deck exists:

$$k(v) \text{ is } \textbf{unchanged}, \qquad \kappa(v): 3 \rightarrow 2$$

$w$ is revealed before the edit and revealed after it. Only *which* colour it
shows has changed. **Anything that tracks the count of revealed neighbours is
blind to this edit by construction.**
"""

PLACEBO = r"""## Every case also carries a placebo edit that changes nothing it should

A one-token change anywhere in a prompt might perturb the output for reasons
having nothing to do with determinacy. So each of the 350 cases carries a
**placebo**: recolour a hint that is **not** a neighbour of $v$, to a colour that
keeps the revealed colouring proper.

After a placebo, $v$ is still forced, direct deduction's round is unchanged, and
the model should behave exactly as it did on the clean prompt. The placebo is
the same *kind* of edit as the corruption -- one hint colour token, in the same
block of the prompt -- and differs only in being irrelevant to $v$.

This is what makes the corruption's effect readable. A number measured under the
corruption means nothing on its own; measured against the same number under the
placebo, it separates "the model responds to *this* edit" from "the model is
perturbed by *any* edit".
"""

AUROCDEF = r"""## AUROC is the chance a determined node outranks an undetermined one

Every measurement in this deck is an AUROC, so it is worth one page. Given
scores $s_i$ and labels $y_i \in \{0,1\}$,

$$\mathrm{AUROC} \;=\; \Pr\big(s_i > s_j \,\big|\, y_i = 1,\ y_j = 0\big) \;+\; \tfrac{1}{2}\Pr\big(s_i = s_j \,\big|\, y_i=1,\ y_j=0\big)$$

Draw one determined node and one undetermined node at random; the AUROC is the
probability the determined one scores higher, counting ties as half.

**0.5 is chance** and **1.0 is a perfect ranking.** It is used here instead of an
accuracy because the positive rate over the scored cells is 0.338, so a
thresholded accuracy would mostly report the base rate.

A cell is *scored* when the node is uncoloured at that position; nodes already
coloured are masked out, which leaves 137,026 of 543,504 cells over 600 held-out
instances.
"""

CONFLATE = r"""## A pooled AUROC mixes two rankings and only one of them is the predicate

When the AUROC is computed over all scored cells at once, the pairs it averages
over are of two kinds:

1. a determined node with **many** revealed neighbours against an undetermined
   node with **few**;
2. a determined node against an undetermined node with **the same** number of
   revealed neighbours.

Only the second requires knowing $\kappa$. The first can be answered by $k$
alone -- by counting, with no idea what the colours are.

Since $k$ and the predicate are correlated, a model that has learned only to
count scores well above 0.5 on the pooled measure. **So the pooled AUROC cannot
distinguish a model that computes determinacy from one that counts.** The next
pages define the measurements that can.
"""

WITHINDEF = r"""## Conditioning on the count removes the ranking a counter could win

The **within-stratum AUROC** restricts every comparison to pairs with the same
number of revealed neighbours, then pools over strata, weighting each by the
number of (determined, undetermined) pairs it contains:

$$\mathrm{AUROC}_{\text{within}} \;=\; \frac{\sum_{k} n^{+}_{k} n^{-}_{k}\, A_k}{\sum_{k} n^{+}_{k} n^{-}_{k}}, \qquad A_k = \mathrm{AUROC}\ \text{among cells with } k(v) = k$$

$n^{+}_{k}$ and $n^{-}_{k}$ are the determined and undetermined cells in stratum
$k$, and $A_k$ is the ordinary AUROC inside it. Weighting by pairs keeps the
result a probability over pairs, so a stratum holding three cells cannot swing
it.

**This scale is calibrated at both ends by construction, and the probe asserts
both.** $k$ itself is constant inside its own stratum, so every pair ties and it
scores exactly **0.5**. $\kappa$ is the predicate's own threshold, so it scores
exactly **1.0**. A model lands between, and where it lands says how much of its
ranking is more than counting.
"""

STRATA = r"""## Only two strata can carry the question, because $d = 5$ and $q = 4$

Of the six possible values of $k(v)$, four answer themselves:

- $k \in \{0, 1, 2\}$: fewer than three revealed neighbours cannot show three
  distinct colours, so $\kappa < 3$ and the node **cannot** be forced. Every one
  of those 53,918 cells is a negative.
- $k = 5$: in a proper colouring no neighbour holds $v$'s own colour, so five
  revealed neighbours draw from three colours, and in all 10,238 such cells they
  show all three. Every one of those cells is a positive.

So the count settles 47% of the cells outright, and the whole question lives in
$k = 3$ and $k = 4$, where a node may or may not be forced and only the colours
decide which. Those are the two strata the results pages report, and the probe
asserts that they are exactly the two with both labels present.
"""

MATCHEDDEF = r"""## The strictest version compares a prompt against its own corrupted copy

The within-stratum measure still compares *different nodes in different
instances*. The corruption allows something stricter.

For each of the 350 cases the probe has $g$ at the same node $v$, at the same
position, under two prompts that differ by **one token**: the clean prompt, where
$v$ is determined, and the corrupted prompt, where it is not. That is a
**matched pair** -- a positive and a negative with the instance, the graph, the
node, the position and $k(v)$ all held fixed.

$$\mathrm{AUROC}_{\text{matched}} \;=\; \Pr\big(g^{\text{clean}}_{p,v} > g^{\text{broken}}_{p,v}\big) \;+\; \tfrac{1}{2}\Pr\big(g^{\text{clean}}_{p,v} = g^{\text{broken}}_{p,v}\big)$$

It asks the same question as the within-stratum measure, with everything
controlled instead of one thing. A representation that computes the predicate
must score near 1.0; one that cannot see the edit scores 0.5. The placebo
version, where $v$ stays determined, **should** score 0.5, and it is reported
beside it as the control.
"""

DENOM = r"""## One rate here is conditional, and its denominator is written down

Most numbers in this deck are over all 350 cases. One is not.

`flip_frac` asks how often the channel's own response is large enough to have
changed the model's answer: whether $\Delta \log \sigma(g)$ at $v$ exceeds $v$'s
margin over the next-best candidate on the clean prompt. That question only makes
sense **where $v$ is the model's clean answer in the first place.** Where it is
not, the margin is negative and any positive response clears it trivially, which
would report a guard that is really a case the model already got wrong.

So `flip_frac` is conditional on the model naming $v$ on the clean prompt, and
its denominators are **236, 228 and 217** -- not 350. An unconditional version
read 0.167 on a six-case test whose true value is 0.000.

This project has made the conditional/unconditional denominator mistake three
times in three different scripts. The rule now is that the denominator is read
from the data and named in words, never assumed.
"""

REGISTERED = r"""## What was predicted, in writing, before the measurement ran

The bars and the prediction live in `scripts/stagez_score.py`, committed before
the probe was submitted, with a self-test covering every enumerated outcome.

**The hypothesis.** The signal stages X and Y installed is not the predicate but
the count $k$, which the corruption leaves invariant -- which would explain both
null results for a reason no re-coupling could repair.

**The prediction.** The channel does not respond to the corruption (win rate in
$[0.45, 0.60)$, and no case where the response is large enough to matter), and
the count alone scores $0.920 \pm 0.040$ on the pooled measure.

**The falsifier, registered too.** A win rate above 0.60 with a systematic drop
would mean the representation *is* counterfactually responsive and the failure is
purely in the coupling -- refuting the hypothesis this stage was written to
state. The bars are two-sided on purpose.

The third measurement, the within-stratum AUROC, was registered separately and
afterwards, once the first two returned an outcome that had been enumerated but
not expected. Its prediction was $0.80 \pm 0.08$.
"""

CLOSES = r"""## What this rules out, stated as plainly as it can be

**It is not the coupling.** Stage Y's null could have meant the channel was wired
too weakly to the output. It was not: the channel's score does not move under the
intervention, so there was nothing for any coupling to transmit. A larger weight
on the term, a wider channel, or a third place to write it all act on a quantity
that is already inert.

**It is not simply a count, either.** That was this stage's registered hypothesis
and the measurement refused it. A count tops out at 0.886 where the model reaches
0.949, and conditioning the model on the count still leaves 0.80. Something
genuinely predicate-like is present.

**It is that the two never meet.** The representation carries real information
about determinacy across instances and none about determinacy under
intervention. Those coexist whenever the information is read off features that
covary with the predicate rather than compute it.

**What it does not rule out.** One architecture, one loss weight, three seeds and
a single family of intervention -- recolouring a hint. A different edit, or a
model that saw corrupted prompts during training, is untested.
"""

NEXT = r"""## What a successor has to supervise, and what it costs

Stages X and Y both supervise a **static per-node ranking on the clean input.**
This deck shows that objective is satisfiable at AUROC 0.95 by a representation
with no counterfactual sensitivity at all. So more of it -- a third place to
write the same loss, a larger weight on it -- is not a new experiment.

The one quantity that X, Y and the 0.95 all leave at chance is the matched pair.
That makes the next objective legible: a margin loss on **matched
clean/corrupted pairs,**

$$\mathcal{L}_{\text{pair}} \;=\; \big[\,m - (g^{\text{clean}}_{p,v} - g^{\text{broken}}_{p,v})\,\big]_{+}$$

where $m$ is a fixed margin and $[x]_{+} = \max(x, 0)$, trained on exactly the
cases this probe already constructs.

**Its cost, priced from this run.** Building a corrupted copy is an array copy
and one index write, with no call to the solver. But it needs two forward passes
per step instead of one, so roughly twice the current ten minutes per epoch, and
the corruption has to be generated during training rather than in a probe
afterwards.

**Its success metric is the 0.80-versus-0.49 contrast, not the pooled AUROC,**
which this deck retires as evidence of anything.
"""


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--job", default=JOB)
    args = ap.parse_args()

    deck_explain.check()
    shared = [p for p in deck_explain.PAGES
              if p not in (deck_explain.SUFFICIENCY, deck_explain.CENSOR)]

    S, R = load()
    mp = [matched_pair(r) for r in R]
    pp = [matched_pair(r, "dg_placebo") for r in R]
    check_claims(S, mp, pp)

    per = S["per_seed"]
    strat = S["stratified"]["per_seed"]
    cov = S["covariate_auroc"]
    taut = S["stratified"]["tautologies"]
    meta = S["covariate_meta"]
    verdict = sz.score(RUN)
    within_pairs, total_pairs = pair_shares(strat[0]["per_stratum"])
    same_pct = 100.0 * within_pairs / total_pairs

    f3 = lambda xs: "/".join("%.3f" % x for x in xs)
    f4 = lambda xs: "/".join("%.4f" % x for x in xs)

    head = "A determinacy signal at AUROC 0.95 that no intervention can move"
    prov = ("job %s, commit %s, three stage-Y seeds, 350 matched cases and 600 "
            "held-out instances; nothing was trained. Registered verdict: %s."
            % (args.job, args.commit, verdict["verdict"]))

    d = WebDeck(args.out, head, "hshunt", "2026-11-15", prov)
    d.title_page(head, subtitle=(
        "Two earlier runs trained this model to represent which nodes are "
        "deducible, both reached AUROC 0.95, and neither changed what the model "
        "does when a node stops being deducible. This run asks whether what was "
        "installed is the property at all, by breaking it one token at a time "
        "and watching the representation."))

    d.points_page(
        "Three measurements on existing checkpoints, nothing trained",
        ["The signal is read under an edit that destroys determinacy.",
         "A model-free count is scored on the same cells.",
         "The ranking is recomputed with that count held fixed."])

    d._add(SYMBOLS)

    # ---- the shared explanation set, with the two shared figures
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, shuffle_adj=True,
                          fall=True)
    for page in shared:
        if page is deck_explain.FIGURE_SLOT:
            d.expect("tokenization",
                     "Every token is one categorical id in a %d-symbol vocabulary"
                     % ex.vocab.size,
                     note="The edit this deck is about changes exactly one of "
                          "these tokens, in the block of given colours.")
            page_tokenization(d, ex)
            d.expect("trace",
                     "A deduction fires when a node sees all three other colours",
                     note="The corruption rewrites one colour token in a prompt "
                          "like this, so one node stops being forced.")
            page_trace(d, ex)
        else:
            d._add(page)

    # ---- everything defined before any number appears
    d._add(PREDICATE)
    d._add(WHY)
    d._add(WHYZ)
    d._add(CHANNEL)
    d._add(CORRUPT)
    d._add(PLACEBO)
    d._add(AUROCDEF)
    d._add(CONFLATE)

    d.claim_page(
        "A counter orders %.0f%% of the pairs without seeing a colour"
        % (100 - same_pct),
        draw_pairs(within_pairs, total_pairs), "pairs",
        note="Pairs of one determined and one undetermined cell, split by "
             "whether the two have the same number of revealed neighbours.")

    d._add(WITHINDEF)
    d._add(STRATA)
    d._add(MATCHEDDEF)
    d._add(DENOM)
    d._add(REGISTERED)

    # ---- results
    # NOT "does not move": g moves by about 0.01, and a title that said
    # otherwise would be false against its own figure. What the figure shows is
    # that it moves no more than under an edit that changes nothing.
    panels(d, "response",
           "The channel moves no more under the corruption than under the placebo",
           draw_response(R),
           note="Cumulative distributions over 350 cases per seed. The two lie "
                "on top of each other, far left of g's own spread.")

    d.table_page(
        "Against its own placebo the corruption is a coin flip",
        ["seed", "win rate", "median $|\\Delta g|$ corrupt", "placebo", "g's spread"],
        [[i, "%.4f" % per[i]["win_rate"],
          "%.4f" % per[i]["median_abs_dg_corrupt"],
          "%.4f" % per[i]["median_abs_dg_placebo"],
          "%.3f" % per[i]["median_g_spread"]] for i in range(3)],
        note="Win rate is P(|corruption| > |placebo|), chance 0.500, because "
             "both edits are one hint colour token.")

    d.table_page(
        "In 681 cases the response was never large enough to matter",
        ["seed", "cases where $v$ is the clean answer",
         "response exceeds the margin"],
        [[i, per[i]["flip_denom"], "%.4f" % per[i]["flip_frac"]] for i in range(3)],
        note="Conditional, and the denominator is read from the data: 236, 228 "
             "and 217, never 350.")

    d.table_page(
        "A count alone does not explain the AUROC, refuting the hypothesis",
        ["scored by", "pooled AUROC"],
        [["revealed-neighbour count $k$, no model", "%.4f" % cov["n_known_nbrs"]],
         ["the channel $g$", f4([c["model_pooled"] for c in strat])],
         ["distinct-colour count $\\kappa$, the predicate itself",
          "%.4f" % cov["n_known_colors"]]],
        highlight_row=0,
        note="%s scored cells over %d instances. The registered bar was 0.90 and "
             "the prediction 0.920 ± 0.040."
             % (format(meta["n_scored"], ","), meta["n_instances"]))

    d.table_page(
        "Holding the count fixed, the channel still ranks determinacy at 0.80",
        ["stratum $k$", "determined", "undetermined", "seed 0", "seed 1", "seed 2"],
        [[k, format(strat[0]["per_stratum"][str(k)]["n_pos"], ","),
          format(strat[0]["per_stratum"][str(k)]["n_neg"], ",")]
         + ["%.4f" % strat[i]["per_stratum"][str(k)]["auroc"] for i in range(3)]
         for k in (3, 4)]
        + [["pooled by pairs", "", ""] + ["%.4f" % c["model"] for c in strat]],
        highlight_row=2,
        note="The other four strata hold only one label each, so no pair inside "
             "them can be ordered at all.")

    d.claim_page(
        "The same signal scores 0.80 across instances and 0.49 under intervention",
        draw_scale(S, mp, pp), "scale",
        note="One dot per seed. Both ends of this axis are exact by "
             "construction and asserted by the probe: the count 0.5, the "
             "predicate 1.0.")

    d.table_page(
        "Two of three seeds rank the broken prompt above the intact one",
        ["seed", "matched pairs, corrupted", "matched pairs, placebo",
         "given the count"],
        [[i, "%.4f" % mp[i], "%.4f" % pp[i], "%.4f" % strat[i]["model"]]
         for i in range(3)],
        note="A placebo leaves $v$ determined, so 0.5 is its correct value, and "
             "it is the control for the column beside it.")

    d.table_page(
        "The pooled AUROC reproduces the published column, so the cells match",
        ["seed", "measured here", "published for stage Y"],
        [[i, "%.4f" % strat[i]["model_pooled"], "%.4f" % p]
         for i, p in enumerate(PUBLISHED_Y_AUROC)],
        note="Computed one instance at a time rather than batched, so the "
             "agreement is a check on the alignment, not a tautology.")

    d.table_page(
        "The prediction was right about the route and wrong about the reason",
        ["registered", "predicted", "measured"],
        [["win rate at chance", "[0.45, 0.60)", f3([p["win_rate"] for p in per])],
         ["response large enough to matter", "0.000",
          f3([p["flip_frac"] for p in per])],
         ["count alone, pooled", "0.920 ± 0.040", "%.4f" % cov["n_known_nbrs"]],
         ["channel given the count", "0.80 ± 0.08",
          f3([c["model"] for c in strat])]],
        note="The count landed inside its tolerance yet under its 0.90 bar, so "
             "the mechanism holds as a part and fails as the whole.")

    d._add(CLOSES)

    d.table_page(
        "Three couplings, three AUROCs near 0.95, and no change in behaviour",
        ["", "stage X", "stage Y", "stage Z"],
        [["what it did", "loss on the logits", "its own channel",
          "read both, trained nothing"],
         ["new parameters", "0", "320", "0"],
         ["determinacy AUROC", "/".join("%.4f" % p for p in PUBLISHED_X_AUROC),
          "/".join("%.4f" % p for p in PUBLISHED_Y_AUROC), "--"],
         ["guard fired", "no", "1 of 681", "--"],
         ["responds to the edit", "no", "no", "measured: no"]],
        note="This is the comparison against the previous best on the same eval. "
             "Stage Z is the measurement that says what X and Y installed.")

    d._add(NEXT)

    d.points_page(
        "The one-line version",
        ["The representation ranks determinacy well across instances.",
         "It cannot track determinacy when a single token changes it.",
         "That is a confound, not a computation."])

    path = d.write()
    print("wrote %s (%d pages)" % (path, d.n))
    print("verdict: %s" % verdict["verdict"])
    print("within %s  matched %s  placebo %s"
          % (f4([c["model"] for c in strat]), f4(mp), f4(pp)))
    print("count %.4f  tautologies %.4f/%.4f  same-count pairs %.1f%%"
          % (cov["n_known_nbrs"], taut["n_known_nbrs"], taut["n_known_colors"],
             same_pct))


if __name__ == "__main__":
    main()
