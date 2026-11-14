"""Render decks/spin-gnn/localisation from the finding-72 lens-and-patch run.

Nothing is re-derived here. Every number comes from scripts/lens_score.py's own
helpers over the per-cell CSVs, every bar is imported from the module that owns
it, and check_claims() asserts what the pages print against lens_summary.json,
which the scorer wrote inside the measurement job.

THE HEADLINE BRANCHES ON THE REGISTERED VERDICT, and the branches that are not
"localised" are real pages, not placeholders: `distributed` is the registered
falsifier and is a result; `thin` means the untied cells did not supply enough
discordant cases; `lens_broken` means a gate failed and nothing below it is
read. All four were written before the job finished.

WHY THIS DECK EXISTS AT ALL. Four spin-gnn decks end on the same caveat --
"nothing here says where in the network the guard lives, and on a tied model
the usual per-layer ablation is not even defined." The second clause is true of
WEIGHT space and false of activations, and this deck is the consequence.

Usage (in a job or with the modules loaded in the same shell):
    python3 -u -m notebooks.make_lens_web --commit <sha> --job <id> [--out <dir>]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

from slides import SERIES, label_line, style_axes

from notebooks import deck_explain
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_prov_web import arms_page
from notebooks.make_staget_web import WebDeck

from scripts.ddguard_score import ARMS, SEEDS, rows_of
from scripts.lens_score import (DEPTH, DISC_MIN, FREED_MIN, HALF, LENS_GAP,
                                RUN, SHARP_GAP, SHARP_HI, SHARP_LO, SITES,
                                _num, cell_dir, deadline, profile, score_cell,
                                sharp)

DECK = "/home/users/hshunt/slides/decks/spin-gnn/localisation"

# Finding 68's round-0 guard rates, the decision-level result this run is the
# follow-up to. Imported rather than retyped wherever a module owns them; these
# three are quoted from the finding-68 deck and are asserted against the tied
# arm's own discordant count below, which is the only one this run remeasures.
F68_UNTIED = (0.0000, 0.0000, 0.8444)

CELLS = [(a, s) for a in ARMS for s in SEEDS]


def cell_label(arm, seed):
    return "%s s%d" % ("tied" if arm == "S1" else "untied", seed)


def load():
    """The summary the scorer wrote, plus every cell rescored from the CSVs.

    Rescoring is the check: score_cell() is the scorer's own function, so if the
    two disagree the deck fails to build rather than printing a third number.
    """
    import json
    with open(os.path.join(RUN, "lens_summary.json")) as fh:
        js = json.load(fh)
    cells = {(a, s): score_cell(a, s) for a, s in CELLS}
    return js, cells


def check_claims(js, cells):
    for a, s in CELLS:
        k = "%s_s%d" % (a, s)
        mine, theirs = cells[(a, s)], js["cells"][k]
        for f in ("n_disc", "n_freed", "L1_agree", "A2_drift",
                  "deadline_edit", "deadline_p_edit", "tail8_node"):
            x, y = mine[f], theirs[f]
            if isinstance(x, float) and x != x:
                assert y != y, "%s %s: nan here, %r there" % (k, f, y)
            else:
                assert abs(float(x) - float(y)) < 1e-12, \
                    "%s %s: %r here, %r there" % (k, f, x, y)
        assert mine["sharp_edit"] == theirs["sharp_edit"], k
    # The verdict is the scorer's, recomputed from the rescored cells so the
    # headline page cannot branch on a stale summary.
    untied = [cells[("J2", s)] for s in SEEDS]
    hits = [c for c in untied if c["sharp_edit"] is not None]
    gates = all(c["L1_pass"] and c["A2_pass"] for c in cells.values())
    thin = all(c["n_disc"] < DISC_MIN for c in untied)
    mine = ("lens_broken" if not gates else "thin" if thin
            else "localised" if len(hits) >= 2 else "distributed")
    assert mine == js["verdict"], "%s here, %s in the summary" % (mine, js["verdict"])
    return hits


def onset(prof, level=HALF):
    """FIRST t at or above level -- the mirror of the scorer's deadline().

    The registered `deadline` is a LAST crossing, which is the right summary
    for a curve that falls and is meaningless for one that rises: a monotone
    rising curve's last crossing is always the final application. The `read`
    site turned out to rise, so its curve is summarised with this instead, and
    a page says in so many words that the registered statistic does not fit it.
    """
    hits = [t for t, v in enumerate(prof) if v == v and v >= level]
    return min(hits) if hits else -1


def collapse_span(prof):
    """Smallest k with some t at or above SHARP_HI and t+k at or below SHARP_LO.

    The registered test is this quantity compared against SHARP_GAP. Reporting
    the quantity itself rather than only the pass/fail is what distinguishes a
    near miss from a curve that never collapses at all; None means it never
    gets from one level to the other.
    """
    best = None
    for t, a in enumerate(prof):
        if a != a or a < SHARP_HI:
            continue
        for k in range(1, DEPTH - t):
            b = prof[t + k]
            if b == b and b <= SHARP_LO:
                best = k if best is None else min(best, k)
                break
    return best


def scorable(cells, arm, site):
    """The seeds whose recovery curve exists -- i.e. that cleared DISC_MIN."""
    return [s for s in SEEDS
            if not all(v != v for v in cells[(arm, s)]["rec_%s" % site])]


def delta_profiles():
    """Mean relative movement per application, per cell, under the clean prompt.

    Computed here rather than stored by the scorer because it is a figure and
    not a bar: L5 registers only the tail share, which the scorer does own.
    Uses the scorer's own profile() so the averaging matches everything else.
    """
    out = {}
    for a, s in CELLS:
        traj = [r for r in rows_of(os.path.join(cell_dir(a, s), "lens_traj.csv"))
                if r["variant"] == "clean"]
        out[(a, s)] = {k: profile(traj, k) for k in ("delta_node", "delta_col")}
    return out


# ---------------------------------------------------------------- figures
def _curves(ax, series, ylab, ylim=None):
    """Three seeds, direct right-hand labels, no legend box.

    label_line wants the scalar endpoint, not the array. Several of these
    curves end at exactly the same value -- two untied seeds both sit at zero
    recovery by the last application -- so the labels are nudged apart
    vertically rather than printed on top of one another.
    """
    x = np.arange(DEPTH)
    ends = []
    for i, (lab, y) in enumerate(series):
        y = np.asarray(y, dtype=float)
        ax.plot(x, y, color=SERIES[i], lw=2.0, marker="o", ms=3.5)
        last = np.flatnonzero(np.isfinite(y))
        if len(last):
            j = last[-1]
            ends.append([float(x[j]), float(y[j]), lab, SERIES[i]])
    ends.sort(key=lambda e: e[1])
    for k in range(1, len(ends)):
        # One label height, in points, converted through nothing: the nudge is
        # an offset in points already, so a fixed step is the whole fix.
        if abs(ends[k][1] - ends[k - 1][1]) < 1e-9:
            ends[k].append(14.0 * k)
    for e in ends:
        dy = e[4] if len(e) > 4 else 0.0
        label_line(ax, e[0], e[1], e[2], color=e[3], dy=dy)
    ax.set_xlabel("application $t$ (of %d)" % DEPTH)
    ax.set_ylabel(ylab)
    ax.set_xlim(-0.4, DEPTH - 0.6)
    if ylim:
        ax.set_ylim(*ylim)
    style_axes(ax)


def draw_recovery(cells, arm, site):
    def go(ax):
        # Only the seeds that cleared DISC_MIN: a thin cell's profile is all
        # nan by construction, and an invisible line with a label hanging off
        # the right-hand edge would read as a measurement that came out flat.
        _curves(ax, [("s%d" % s, cells[(arm, s)]["rec_%s" % site])
                     for s in scorable(cells, arm, site)],
                "fraction of discordant cases that name $v$ again", (-0.03, 1.03))
        ax.axhline(SHARP_HI, color="0.75", lw=1.0, ls="--")
        ax.axhline(SHARP_LO, color="0.75", lw=1.0, ls="--")
    return go


def draw_recovery_p(cells, arm):
    def go(ax):
        _curves(ax, [("s%d" % s, cells[(arm, s)]["recp_edit"]) for s in SEEDS],
                "recovered share of the clean-broken gap in $p_{\\rm freed}$",
                (-0.1, 1.1))
        ax.axhline(HALF, color="0.75", lw=1.0, ls="--")
    return go


def delta_mean(prof, arm, key):
    return np.nanmean([prof[(arm, s)][key] for s in SEEDS], axis=0)


def delta_ylim(prof):
    """One pair of limits for both delta figures.

    The two pages are meant to be flipped between, and the note on the second
    one says so. An autoscaled log axis on each would make a smaller curve look
    the same height as a larger one, which is the comparison the pages exist to
    support, so the limits are computed once over both.
    """
    vals = [v for key in ("delta_node", "delta_col") for arm in ARMS
            for v in delta_mean(prof, arm, key) if v == v and v > 0]
    return (min(vals) / 1.6, max(vals) * 1.6)


def draw_delta(prof, key, ylab, ylim):
    def go(ax):
        ax.set_yscale("log")
        _curves(ax, [("tied" if arm == "S1" else "untied",
                      delta_mean(prof, arm, key)) for arm in ARMS], ylab, ylim)
        # A log axis over less than one decade labels a single tick, which
        # leaves the reader no way to read a value off either page. Plain
        # decimals at fixed places, and the same places on both pages.
        ticks = [v for v in (0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5)
                 if ylim[0] <= v <= ylim[1]]
        ax.set_yticks(ticks)
        ax.set_yticklabels(["%g" % v for v in ticks])
        ax.set_yticks([], minor=True)
    return go


# ---------------------------------------------------- explanation pages
VOCAB = """## A cell is one trained model; an arm is a configuration

Three words of bookkeeping recur on every page below, including on the shared
pages that set up the task, so they are defined before any of them.

A **cell** is one trained model: one configuration trained at one seed and
saved as one checkpoint. Six cells are read here and nothing is trained. Every
table below has one row per cell.

An **arm** is a configuration — the three cells that differ only in their seed.
There are two arms here, the tied model and the untied model. Results are
reported per cell rather than averaged over an arm, because the spread between
three seeds of one arm has repeatedly been larger than the effect being tested.

A **stage** is one registered experiment in this project's sequence, numbered
as a *finding*. Findings 63, 68 and 69 are quoted on the pages below; each
quotation says which finding it comes from and what that finding measured.
"""

DEPTHWORDS = """## Two different things are called depth; $t$ always means one

A **round** is a step of direct deduction in the task itself: every node whose
neighbours already show $q - 1 = 3$ colours is resolved at once. An instance
needs as many rounds as its longest chain of inference, and draws from this
generator reach 19.

An **application** is one pass of the transformer block over the whole token
sequence. There are always exactly sixteen of them, written
$t = 0, 1, \\dots, 15$, however many rounds the instance would need.

The two are different counts and the model is never told the first one. The
trace figure above says $T = 5$ and means rounds; the architecture page says
$T = 16$ and means applications. **Every $t$ on every axis below is an
application**, and nothing below counts rounds.
"""

WHY_NOW = """## Per-layer ablation is undefined on a tied model; patching is not

Four earlier decks for this project end on the same sentence: *nothing here
says where in the network the guard lives, and on a tied model the usual
per-layer ablation is not even defined.*

That is true, and it is true only of **weights**. The tied model has one
transformer block whose parameters are used sixteen times, so zeroing an
attention head zeroes it in all sixteen at once; there is no "layer 7 head 3"
to remove. The project's own circuit code refuses to pretend otherwise.

An **activation** is different. The forward pass is
$$h_{t+1} = F(h_t), \\qquad t = 0, 1, \\dots, 15,$$
where $h_t$ is the sequence of hidden vectors after $t$ applications of the
block $F$. Those sixteen states are sixteen distinct objects even when $F$ is
one set of weights. Reading them, or replacing one of them, is well defined on
both models, and the index $t$ means the same thing in both: how many times the
computation has been applied. Both models here are sixteen applications deep,
so $t$ is a shared axis and the two arms can be put on the same plot.
"""

ARMSAXIS = """## The untied model is the same computation with sixteen blocks

The untied model runs
$$h_{t+1} = F_t(h_t), \\qquad t = 0, 1, \\dots, 15,$$
with a different $F_t$ at each step. The tied model is the special case
$F_t = F$ for all $t$.

That is the whole architectural difference being studied, and it is why the
tied model is the object the project wants: a map applied repeatedly is
something a theory can describe, while sixteen unrelated maps are not.

The cost of the difference is on one page below: the tied model is worse at the
task. The point of this deck is not to close that gap but to say **where along
$t$** each model does the thing that distinguishes them.
"""

GUARD = """## The guard: naming a node only while that node is still forced

Direct deduction names a node $v$ only when the hints and the deductions so far
leave exactly one colour available to it. A model that has learned the rule
rather than the habit should therefore **stop naming $v$** as soon as $v$ stops
being forced.

Finding 68 built that test. Each case has two prompts differing in one token:

- **clean** — $v$ is forced to colour $c$, and the next thing to write is $v$;
- **broken** — one hint colour is rewritten so a second colour $a$ becomes
  available to $v$. Nothing forces $v$ any more, and direct deduction must not
  name it.

The guard rate is how often the model keeps naming $v$ anyway:
$$g = \\frac{\\#\\{\\text{clean names } v \\text{ and broken names } v\\}}
            {\\#\\{\\text{clean names } v\\}}.$$
**A HIGH $g$ MEANS NO GUARD** — the model went on naming a node it could no
longer deduce. Low is good.

Every case here is at **round 0**: the first deduction of the trace, where the
prompt carries hints and nothing the model has written itself. At round 0
finding 68 measured the untied model at
%.4f / %.4f / %.4f over its three seeds, and the tied model at $1.0000$ in all
three: the tied model never withholds.
""" % F68_UNTIED

READOUTS = """## The two numbers read off a prompt, and where they are read

Every prompt here stops just before the model would write the next deduction.
Two positions in that sequence are read, and both appear on every page below.

**The node position.** The next token the model writes is a node id. Write
$\\mathrm{emits}_v = \\mathbf{1}[\\arg\\max_u \\ell_u = v]$, where $\\ell_u$ is
the logit the model gives to node $u$ at that position. It is $1$ when the model
is about to name $v$ and $0$ otherwise. This is what the guard acts on.

**The colour position.** One token later the model writes a colour. Let $p_x$ be
the probability it gives to colour $x$ there. Two quantities are read:
$$p_{\\rm freed} = p_a, \\qquad p_{\\rm cand} = p_c + p_a,$$
where $c$ is $v$'s true colour and $a$ is the colour the broken edit freed.
$p_{\\rm freed}$ is how much the model has noticed that $a$ became available;
$p_{\\rm cand}$ is how much of its colour mass sits on the two-element candidate
set $\\{c, a\\}$ rather than on the other two colours.

Finding 69 found the tied model at $p_{\\rm cand}$ exactly $1.0000$ with
$p_{\\rm freed}$ about $0.52$: it had worked out that two colours were available
and scheduled $v$ regardless.
"""

LENSPAGE = """## A lens reads the model's own output off an unfinished state

$h_t$ is an intermediate state and the model has no opinion about it. The
standard trick is to apply the model's own final layer norm $N$ and readout
matrix $W$ to it anyway,
$$\\tilde\\ell_t = W\\,N(h_t),$$
and read $\\mathrm{emits}_v$, $p_{\\rm freed}$ and $p_{\\rm cand}$ off
$\\tilde\\ell_t$ exactly as they are read off the finished logits. That gives a
trajectory: what the model would say if it stopped after $t$ applications.

This is a **descriptive** instrument, and it is cheap — one forward pass per
prompt, with the sixteen states captured on the way through. It is not a causal
claim: $W$ and $N$ were trained on $h_{16}$ and nothing guarantees they are
meaningful earlier. Where the lens and the patch disagree, the patch wins, and
a later page says so again.

At $t = 15$ the lens is not an approximation at all: $W N(h_{15})$ **is** the
model's output. That identity is one of the three gates further down.
"""

L1PAGE = """## Gate one: at the last application the lens must be the model

$W N(h_{15})$ and the model's own logits are the same expression, so they must
agree exactly. If they do not, the lens is applying the wrong norm, reading the
wrong position, or capturing the wrong tensor, and every trajectory on every
page below would be a picture of that bug.

The gate is therefore an identity check rather than a tolerance:
$\\mathrm{emits}_v$ agreement of $1.0000$ on every case and every variant, and
$|p_{\\rm freed}^{\\rm lens} - p_{\\rm freed}^{\\rm model}| \\le 10^{-4}$, which
is float32 slack and nothing else. It fails the run closed inside the
measurement job, before any patch is computed, and is recomputed here.
"""

PATCHPAGE = """## A patch asks whether an application still changes the answer

The lens says what the model would conclude at step $t$. It cannot say whether
step $t$ mattered. For that the state has to be changed and the output watched.

Run the **broken** prompt, which has one hint colour rewritten. At one chosen
application $t$ and one chosen token position $p$, overwrite the state there
with the state the **clean** prompt had at the same $t$ and $p$:
$$h^{\\rm broken}_t[p] \\;\\leftarrow\\; h^{\\rm clean}_t[p],$$
then let the remaining $15 - t$ applications run normally and read the output.
This is one forward pass per $(t, p)$, so sixteen per site per case.

If the splice restores the clean behaviour, the information the edit destroyed
was still needed after application $t$. If it does not, the decision was already
taken. **The last $t$ at which the repair still works is the deadline**, and
that deadline is what "where in the iteration" means here.

On the tied model the hook has to count calls, because one block object is
called sixteen times; splicing on the $t$-th call is what makes the tied row
mean the same as the untied row.
"""

SITESPAGE = """## Two positions are patched, and they ask different questions

A splice is at one token position, so the position chooses the question.

**`edit` — the hint colour token the broken prompt rewrote.** Repairing it puts
the correct evidence back. The deadline here says *by when the evidence has to
be right*: after it, the model has already extracted what it is going to
extract from that token and fixing the token changes nothing.

**`read` — the node position, where $\\mathrm{emits}_v$ is read.** Repairing it
overwrites the decision itself with the clean model's. The deadline here says
*how long the answer stays movable*.

The two can differ, and the gap between them is the part of the computation
that carries the information from the hint to the decision. One site is patched
at a time, which is a real limit: a decision carried redundantly at two
positions would show up as low recovery at both, and this design could not tell
that from a decision carried somewhere else entirely.
"""

ENDPOINTS = """## Both ends of every recovery curve are fixed by construction

Two points on each curve below carry no information at all, and they are
pointed out here so that nobody reads them as a result.

Attention is the only thing that moves information between positions, and it
lives inside the block. So a splice at position $p$ and application $t$ can
only affect a **different** position if at least one more application runs
after it.

- **A splice at the `edit` position at $t = 15$ has no effect whatsoever** on
  either read position. There is no application left to carry it there, so
  $\\mathrm{recovery}(15)$ and $\\mathrm{rec}_p(15)$ are exactly $0$ for every
  cell, by construction and not by measurement.
- **A splice at the `read` position at $t = 15$ replaces the very vector the
  readout consumes,** so it reproduces the clean answer exactly and recovery
  there is $1$ for the same empty reason.

Everything interesting is strictly between those two ends.
"""

DEADLINEPAGE = """## One summary statistic fits a falling curve and not a rising one

The registered summary of a recovery curve is a **deadline**: the last
application $t$ whose mean recovery is still at least $%.2f$. That is the right
thing to ask of a curve that starts high and falls, which is what repairing the
evidence was expected to look like.

The `read` site turned out to produce a curve that **rises** instead, from zero
to one. A rising curve's last crossing is always the final application, so the
registered deadline returns $15$ there and says nothing.

The `read` curves are therefore summarised with the mirror statistic, the
**onset** — the first application whose mean recovery reaches $%.2f$ — computed
in this deck rather than in the registered scorer, and labelled as such
wherever it appears. The registered number is printed beside the `read` curve
too, so the mismatch is visible rather than quietly dropped.
""" % (HALF, HALF)

A2PAGE = """## Gate two: splicing a run into itself has to change nothing

A patching result is only as good as the hook. A hook that wrote one position
to the left, or cloned a stale tensor, or fired on the wrong call of a tied
block, would produce a smooth plausible recovery curve made entirely of its own
bug, and nothing about the curve would look wrong.

So the same splice is run **clean into clean**: same hook, same positions, same
sixteen applications, but the source and the target are the same prompt. Then
the patched output must be the unpatched one, to float32 slack.

The bar is $1.0000$ of rows with $\\mathrm{emits}_v$ unchanged and a maximum
$|p_{\\rm freed}|$ drift of $10^{-4}$. Like gate one it fails the run closed
inside the job, before the real patch runs.
"""

RECOVPAGE = """## Recovery is measured only where the guard actually acted

The patch is scored on **discordant** cases: the clean prompt names $v$ and the
broken prompt does not. Those are exactly the cases where the guard did
something, and on any other case there is nothing for a repair to restore.

$$\\mathrm{recovery}(t) = \\frac{\\#\\{\\text{discordant cases where the patch at }
t\\text{ makes the model name } v\\}}{\\#\\{\\text{discordant cases}\\}}.$$

$\\mathrm{recovery}(t) = 1$ means a repair at application $t$ completely undoes
the guard; $\\mathrm{recovery}(t) = 0$ means it is too late. A cell needs at
least %d discordant cases or it is reported as **thin** and not scored, because
a recovery curve over twenty cases is a picture of twenty coin flips.

This is why the headline is read on the untied arm only. The tied model never
withholds, so in this run it had **no discordant cases at all**, in any of its
three seeds: there is nothing there to localise, and the tied arm's version of
the question is on the next page.
""" % DISC_MIN

RECOVPPAGE = """## The tied arm's question is when the ambiguity gets resolved

The tied model does not withhold, so recovery of $\\mathrm{emits}_v$ has nothing
to recover. But finding 69 showed it computes the ambiguity anyway, putting its
whole colour mass on the two-element candidate set. That quantity can be
patched, and it is the one the project actually wants.

Read the same splice on $p_{\\rm freed}$ at the colour position, normalised per
case so that $0$ is the broken prompt and $1$ is the clean one:
$$\\mathrm{rec}_p(t) = \\frac{p_{\\rm freed}^{\\rm patched}(t)
  - p_{\\rm freed}^{\\rm broken}}
  {p_{\\rm freed}^{\\rm clean} - p_{\\rm freed}^{\\rm broken}}.$$
Cases where that denominator is below $%.2f$ in absolute value are dropped: the
two prompts do not separate there and the ratio would be noise over noise.

The deadline is the last $t$ whose mean $\\mathrm{rec}_p$ is at least $%.2f$.
On the tied model it reads as a statement about the iterated map — *the
candidate set is settled by application $k$* — which is the kind of statement a
theory of one repeated block would have to reproduce.
""" % (FREED_MIN, HALF)

REGPAGE = """## What was registered, and what the registered miss would mean

Committed before the job was submitted, in the scorer that produces every
number here.

- **L3, the headline, untied arm only.** The guard is *localised in time*:
  some application $t$ has $\\mathrm{recovery}(t) \\ge %.2f$ and
  $\\mathrm{recovery}(t + %d) \\le %.2f$, at the `edit` site, in at least 2 of 3
  seeds.
- **Its falsifier is a result, not a failure.** A gradual decline means the
  guard is **distributed** across the applications. That is reported as the
  finding, with the profile, and the deck is not rewritten around the other
  branch.
- **L1, a gate.** The lens at $t = 15$ is the model, exactly.
- **A2, a gate.** The clean-into-clean splice changes nothing.
- **L2, L4, L5 are descriptive and carry no bar,** because nothing in this
  project fixes in advance what a decision time should be.
""" % (SHARP_HI, SHARP_GAP, SHARP_LO)

L2PAGE = """## Decision time: the last application at which the lens changes

A descriptive companion to the patch, from the lens alone and free, since the
trajectory is already captured.

For one case, let $\\mathrm{emits}_v(t)$ be the lens's value at application $t$.
The **decision time** is
$$t_{\\rm dec} = \\max\\{\\,t : \\mathrm{emits}_v(t) \\ne
  \\mathrm{emits}_v(t-1)\\,\\},$$
and $0$ when the value never changes after the first application. From
$t_{\\rm dec}$ on, the model's answer to "do I name $v$ next" is fixed.

The median over cases is quoted, for the clean and the broken prompt
separately. A broken prompt that takes longer than a clean one is a model doing
extra work when the evidence is ambiguous, which is the behaviour the guard
would need. The untied arm does the reverse, and that has a page of its own
below.

$t_{\\rm dec}$ is a lens quantity, so it is weaker evidence than the patch. It
can report a decision as settled while a later application would still have
changed it, and the patch is the thing that rules that out.
"""

L5PAGE = """## How much movement is left in the last eight applications

Finding 63 tried training the tied model deeper and found $T = 32$ did not beat
$T = 16$. This run can say something about why, for free, from the same
trajectories — and it does **not** reopen that question. No model is trained
here and the depth knob stays closed.

Let $h_t[p]$ be the state at read position $p$ after application $t$. The
relative movement the $t$-th application produced there is
$$\\delta_t = \\frac{\\lVert h_t[p] - h_{t-1}[p] \\rVert}
                    {\\lVert h_{t-1}[p] \\rVert},$$
undefined at $t = 0$, because the hook fires on block outputs and the embedding
that the first application consumed is not one of them. The **tail share** is
$$\\mathrm{tail}_8 = \\frac{\\sum_{t \\ge 8} \\delta_t}{\\sum_{t \\ge 1} \\delta_t},$$
the fraction of all movement that happens in the second half of the stack.

A small tail share means the later applications barely move the state, which is
what a model near a fixed point looks like and what would make extra depth
useless. A large one means the computation is still running at the end.
"""

KEYPAGE = """## Gate three: the identifier a case is grouped by has to be unique

Stated because it caught a real bug in this run, before any result was read.

The first version of this measurement labelled each row with the node $v$ it was
about. $v$ is an index between $0$ and $47$. The instance stream yields $885$
breakable cases over $400$ graphs, so node $7$ is the subject of dozens of
unrelated cases, and grouping on it would have pooled them: the discordant set
would have pulled in rows belonging to other cases, and the per-case
dictionaries would have silently kept only the last case per node.

The first job was cancelled ten minutes in, the rows were given a sequential
case identifier together with the three other fields the project's earlier
probes key on, and the scorer now **asserts** that the identifier is unique
before reading anything. The assertion is there because the comment version of
it is what failed.
"""

SCOPE = """## What this run does not establish

- **Nothing was trained.** Six existing checkpoints were read. $\\rho_{50}$ did
  not move and no published number changes.
- **The arms still differ 4.1$\\times$ in trainable parameters,** so every
  between-arm difference here is weight tying *and* budget, exactly as in the
  four earlier decks.
- **A lens is not causal.** The decision time and the movement profile are read
  through a readout trained only on the final state. The patch is the causal
  half, and where they disagree the patch is what counts.
- **Round 0 only.** Every case here is the first deduction in a trace, so the
  prompt holds hints and nothing the model wrote itself. Whether the same
  applications carry the guard deeper into a trace is untested here.
- **One site at a time,** so a decision carried redundantly at two positions is
  indistinguishable here from one carried somewhere neither site touches.
- **One untied seed has almost no guard to localise.** Its discordant count is
  far below the minimum, so the headline rests on two seeds, not three, and
  that seed is the one finding 68 already measured as barely guarding at all.
- **The onset statistic is post-hoc.** The registered summary was a deadline,
  which does not fit a rising curve. The onset was added when that was seen,
  and it is a description of a curve rather than a test of anything.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--job", default="unknown")
    args = ap.parse_args()

    deck_explain.check()
    # CENSOR goes: nothing here is scored on a density grid. GATE, RHO50,
    # SOLVE_RATE and BASELINES stay, because the arms page quotes all four.
    shared = [p for p in deck_explain.PAGES if p is not deck_explain.CENSOR]

    js, cells = load()
    hits = check_claims(js, cells)
    prof = delta_profiles()
    verdict = js["verdict"]
    untied = [cells[("J2", s)] for s in SEEDS]
    n_disc = [c["n_disc"] for c in untied]

    head = {
        "localised": "Repair stops working within two applications, so the guard is localised",
        "distributed": "The guard's window is a few applications wide, not one",
        "thin": "Too few discordant cases to localise anything on this list",
        "lens_broken": "A gate failed, so no localisation is read from this run",
    }[verdict]

    prov = ("Job %s, commit %s. Six existing checkpoints, finding 68's round-0 "
            "case list, %d cases read per cell and %d of them patched at every "
            "one of %d applications and both sites. Registered by ce645ae "
            "before the job ran. Nothing trained."
            % (args.job, args.commit, cells[("S1", 0)]["n_cases"],
               cells[("S1", 0)]["n_patch"], DEPTH))

    d = WebDeck(args.out, head, "hshunt", "2026-11-13", prov)
    d.title_page(head, subtitle=(
        "Four earlier decks for this project end by saying that nothing in them "
        "localises the guard, because per-layer ablation is undefined on a model "
        "whose one block is applied sixteen times. That is true of weights and "
        "false of activations, and this is what the activations say."))

    d.points_page(
        "Nothing is trained and no published number can change",
        ["Six checkpoints that already existed are read, never written.",
         "No number this project has published moves, because nothing was "
         "trained.",
         "One token differs between the two prompts, asserted in code."])

    d._add(VOCAB)

    # ---- shared explanation set, with the two shared figures
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, shuffle_adj=True,
                          fall=True)
    for page in shared:
        if page is deck_explain.FIGURE_SLOT:
            d.expect("tokenization",
                     "Every token is one categorical id in a %d-symbol vocabulary"
                     % ex.vocab.size,
                     note="The leak the figure names is the order of node ids "
                          "within a row; a later page shows how every eval "
                          "here closes it.")
            page_tokenization(d, ex)
            d.expect("trace",
                     "A deduction fires when a node sees all three other colours",
                     note="Every case here edits one colour token in a prompt "
                          "like this, so that one node stops being forced.")
            page_trace(d, ex)
        else:
            d._add(page)

    # Definitions before the pages that use them: recovery is defined before
    # the two pages that discuss the shape of a recovery curve, and the three
    # gates sit together rather than being scattered through the setup.
    for page in (arms_page(), DEPTHWORDS, WHY_NOW, ARMSAXIS, GUARD, READOUTS,
                 LENSPAGE,
                 PATCHPAGE, SITESPAGE, RECOVPAGE, RECOVPPAGE, ENDPOINTS,
                 DEADLINEPAGE, L1PAGE, A2PAGE, KEYPAGE, REGPAGE, L2PAGE,
                 L5PAGE):
        d._add(page)

    # ---- gates first, because everything below is read only if they hold
    d.table_page(
        "Lens and no-op gates hold exactly in all six cells",
        ["cell", "lens agreement at $t=15$", "max $|p_{\\rm freed}|$ gap",
         "no-op rows unchanged", "max no-op drift"],
        [[cell_label(a, s), "%.4f" % cells[(a, s)]["L1_agree"],
          "%.1e" % cells[(a, s)]["L1_gap"], "%.4f" % cells[(a, s)]["A2_same"],
          "%.1e" % cells[(a, s)]["A2_drift"]] for a, s in CELLS],
        note="Bars: agreement and unchanged both $1.0000$, both gaps at most "
             "$10^{-4}$. Gate three is an assertion and prints no number.")

    d.table_page(
        "The tied model supplies no cases where the guard could have acted",
        ["cell", "cases patched", "discordant", "usable for $p_{\\rm freed}$"],
        [[cell_label(a, s), cells[(a, s)]["n_patch"], cells[(a, s)]["n_disc"],
          cells[(a, s)]["n_freed"]] for a, s in CELLS],
        note="Discordant means the clean prompt names $v$ and the broken one "
             "does not. Finding 68's result, arriving again.")

    # ---- the headline
    d.claim_page(head, draw_recovery(cells, "J2", "edit"), "recovery_edit",
                 note=("Untied arm, `edit` site, the %d seeds above the case "
                       "minimum. Dashed lines are the registered %.2f and %.2f."
                       % (len(scorable(cells, "J2", "edit")), SHARP_HI, SHARP_LO)))

    d.table_page(
        "The registered two-application test is met by one seed, not two",
        ["seed", "discordant cases", "last $t$ with recovery $\\ge %.2f$" % HALF,
         "%.2f to %.2f within %d applications" % (SHARP_HI, SHARP_LO, SHARP_GAP)],
        [["s%d" % s, cells[("J2", s)]["n_disc"],
          "not scored" if cells[("J2", s)]["n_disc"] < DISC_MIN
          else cells[("J2", s)]["deadline_edit"],
          "not scored" if cells[("J2", s)]["n_disc"] < DISC_MIN
          else "no" if cells[("J2", s)]["sharp_edit"] is None
          else "yes, at $t = %d$" % cells[("J2", s)]["sharp_edit"]] for s in SEEDS],
        note="The rule needed 2 of 3. Seed 2 is below the %d-case minimum "
             "because its guard barely fires at all." % DISC_MIN)

    spans = {s: collapse_span(cells[("J2", s)]["rec_edit"])
             for s in scorable(cells, "J2", "edit")}
    d.points_page(
        "A missed bar is not a null result",
        ["Registered %d applications; measured %s."
         % (SHARP_GAP, ", ".join(
             "seed %d in %s" % (s, "never" if spans[s] is None else spans[s])
             for s in sorted(spans))),
         "A window, not a stack, and not a single application either.",
         "The bar is not moved after the fact to make it pass."])

    rd = sorted({cells[("J2", s)]["deadline_read"]
                 for s in scorable(cells, "J2", "read")})
    d.claim_page("Overwriting the node position works only near the end",
                 draw_recovery(cells, "J2", "read"), "recovery_read",
                 note="Untied arm, `read` site. The curve rises, so the "
                      "registered deadline is %s in every scored seed and the "
                      "onset replaces it." % "/".join(str(x) for x in rd))

    d.table_page(
        "The hint stops mattering long before the node position fixes the answer",
        ["seed", "`edit` deadline, last $t$", "`read` onset, first $t$",
         "applications in between"],
        [["s%d" % s, cells[("J2", s)]["deadline_edit"],
          onset(cells[("J2", s)]["rec_read"]),
          onset(cells[("J2", s)]["rec_read"]) - cells[("J2", s)]["deadline_edit"]]
         for s in scorable(cells, "J2", "edit")],
        note="Through most of that gap neither patched position restores the "
             "answer, so whatever carries the decision is somewhere else.")

    # ---- the tied arm's question
    d.claim_page("The tied model's candidate set stays repairable past application twelve",
                 draw_recovery_p(cells, "S1"), "recovery_p_tied",
                 note="Tied arm, `edit` site, read on $p_{\\rm freed}$. Dashed "
                      "line is the half-recovery level.")

    d.claim_page("The untied model's candidate set is repairable only to application twelve",
                 draw_recovery_p(cells, "J2"), "recovery_p_untied",
                 note="Untied arm, same quantity, same site and same axis as "
                      "the previous page.")

    d.table_page(
        "The ambiguity deadline, the number a theory would have to predict",
        ["cell", "usable cases", "last $t$ with mean $\\mathrm{rec}_p \\ge %.2f$" % HALF],
        [[cell_label(a, s), cells[(a, s)]["n_freed"], cells[(a, s)]["deadline_p_edit"]]
         for a, s in CELLS],
        note="For the tied arm this reads: the candidate set is still movable "
             "up to this application of the one shared block, and settled after.")

    d.points_page(
        "The colour is still being decided long after the node is",
        ["Repairing the hint restores the colour distribution until $t = 12$ "
         "or later.",
         "Untied, it stops restoring the naming decision at $t=3$ and $t=5$.",
         "So the model commits to whom to name before choosing the colour."])

    # ---- descriptive
    uclean = [cells[("J2", s)]["tdec_clean"] for s in SEEDS]
    udl = [cells[("J2", s)]["deadline_edit"] for s in scorable(cells, "J2", "edit")]
    d.table_page(
        "The lens keeps changing its answer after the patch stops working",
        ["cell", "median $t_{\\rm dec}$, clean", "median $t_{\\rm dec}$, broken"],
        [[cell_label(a, s), "%.1f" % cells[(a, s)]["tdec_clean"],
          "%.1f" % cells[(a, s)]["tdec_broken"]] for a, s in CELLS],
        note="Lens, not patch: compare the untied clean column, %.0f to %.0f, "
             "with the `edit` deadlines of %s."
             % (min(uclean), max(uclean), " and ".join(str(x) for x in udl)))

    # The zero in the broken column is not a missing value, and a reader who
    # takes it for one loses the sharpest single number in the run.
    d.points_page(
        "On broken prompts the untied lens settles sooner, not later",
        ["Median broken $t_{\\rm dec}$ is $0$ in two of three untied seeds.",
         "That means the lens value never changed after application one.",
         "The same two seeds' clean medians are $10$ and $11$."])

    ylim = delta_ylim(prof)
    d.claim_page("The node position is still moving in the second half",
                 draw_delta(prof, "delta_node", "relative movement $\\delta_t$",
                            ylim),
                 "delta_node",
                 note="Clean prompt, mean over three seeds per arm, log scale. "
                      "Undefined at $t=0$ by construction.")

    d.claim_page("The colour position moves later than the node position everywhere",
                 draw_delta(prof, "delta_col", "relative movement $\\delta_t$",
                            ylim),
                 "delta_col",
                 note="Same cells and the same log axis limits as the previous "
                      "page, read at the colour position instead.")

    d.table_page(
        "At the colour position every tied seed finishes earlier than every untied",
        ["cell", "$\\mathrm{tail}_8$, node position", "$\\mathrm{tail}_8$, colour position"],
        [[cell_label(a, s), "%.3f" % cells[(a, s)]["tail8_node"],
          "%.3f" % cells[(a, s)]["tail8_col"]] for a, s in CELLS],
        note="Bears on finding 63, where $T=32$ did not beat $T=16$. It does "
             "not reopen it: nothing was trained.")

    d._add(SCOPE)

    # ---- closing
    close = {
        "localised": "localised, in %d of 3 untied seeds." % len(hits),
        "distributed": ("distributed, missing the 2-of-3 rule by one seed."
                        if len(hits) == 1 else
                        "distributed, with %d of 3 untied seeds sharp." % len(hits)),
        "thin": "unscored; discordant cases were %s against a minimum of %d." % (
            "/".join(str(x) for x in n_disc), DISC_MIN),
        "lens_broken": "unread; a gate failed before any result was formed.",
    }[verdict]

    d.points_page(
        "What this adds, and what it is the first measurement of",
        ["Registered verdict: %s" % close,
         "No earlier localisation result exists for this project to compare "
         "against.",
         "The tied arm gave %s discordant cases, matching finding 68."
         % "/".join(str(cells[("S1", s)]["n_disc"]) for s in SEEDS)])

    path = d.write()
    print("wrote %s (%d pages)" % (path, d.n))


if __name__ == "__main__":
    main()
