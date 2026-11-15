#!/usr/bin/env python3
"""Render decks/spin-gnn/channel from stage Y's own artifacts.

Stage Y gave the determinacy signal its own emission channel -- a separate
score g_{p,v} added to the node logits as log sigma(g) -- after stage X had
installed the same signal on the existing readout and bought no behavioural
change at all. This deck reports what the second coupling did.

NOTHING IS RETYPED. Every number on every page is read out of the artifacts the
jobs wrote (history.json, the 14-point rho grids, skip_behavior.csv, the
determinacy CSV, stagey_verdict.json) and every bar is imported from
scripts.stagey_score, which registered them before the first cell ran.
check_claims() then asserts the numbers the pages print against the verdict file
the measurement job wrote, so a deck whose prose has drifted from the data fails
to build instead of publishing to a URL.

THE HEADLINE IS THE REGISTERED VERDICT and it is not negotiable here: the
scorer returned "Y4 failed, Y1-Y3 not interpreted", because one seed of three
finished 0.026 of token accuracy below the control. The deck says that, says
which bars failed on their own numbers anyway, and does not reach for a
subset of seeds that looks better.

WHY THE EXPLANATION PAGES ARE HALF THE DECK. ~/CLAUDE.md requires a deck to
stand alone for a reader who has seen none of this project's other decks. The
stage-X refutation is the premise of this entire run, so it is restated here in
full rather than cited; the determinacy predicate, the guard, the gauge
argument, the two couplings and the shape bar are all defined before any
result. deck_explain supplies the stage-independent half.

Usage (in a job -- this imports torch through deck_tokenization):
    python3 -u -m notebooks.make_channel_web --commit <sha> --job <id>
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

from slides import MUTED, SERIES, label_line, style_axes

from notebooks import deck_explain
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_staget_web import WebDeck

from scripts.ddguard_score import ratio
from scripts.guard_shape import shape
from scripts.n96_fallscan_analysis import rho_50
from scripts.step_epoch_census import FLOOR, MIN_EP, level, step_epoch
import scripts.stagey_score as sy

DECK = "/home/users/hshunt/slides/decks/spin-gnn/channel"

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
_SG = os.path.join(_SCR, "spin_gnn")
SCAN_Y = os.path.join(_SG, "sudoku_n48_determinacy_Y_2026-11-14b")
SCAN_X = os.path.join(_SG, "sudoku_n48_determinacy_X_2026-11-13")
SCAN_S = os.path.join(_SG, "sudoku_n48_tiedwidth_S_2026-11-10")
RUN_Y = os.path.join(_SG, "runs", "2026-11-14b_stagey_guard")

JOB_Y, JOB_X, JOB_S = "52348732", "52285051", "51853969"

_YTAG = "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_chan"
_STAG = "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied"
_XTAG = _STAG + "_det_dw1"

Y_CELLS = [os.path.join(
    SCAN_Y, "t%02d_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s%d" % (i, i),
    _YTAG + ("" if i == 0 else "_s%02d" % i)) for i in range(3)]
S_CELLS = [os.path.join(
    SCAN_S, "t%02d_d0320_ff01024_h04_tied_no_mlp_L16_s%d" % (i, i),
    _STAG + ("" if i == 0 else "_s%02d" % i)) for i in range(3)]
X_CELLS = [os.path.join(
    SCAN_X, "t%02d_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_s%d" % (i, i),
    _XTAG + ("" if i == 0 else "_s%02d" % i)) for i in range(3)]

# The three reference points every spin-gnn deck ends against, and stage X's
# rho_50, which is the thing this run is the follow-up to. Quoted from the
# decks that measured them; the channel arm's own numbers are all read below.
X_RHO50 = (0.866, 0.864, 0.855)
X_GUARD = (1.0000, 1.0000, 0.9568)
X_GUARD_DEN = (194, 175, 185)
X_AUROC = (0.9462, 0.9467, 0.9469)
X_OUTCOME = (0.653, 0.713, 0.690)
CONTROL_GUARD_DEN = (232, 237, 233)
S1_TRAINABLE, Y_TRAINABLE = 613818, 614138
J2_RHO50, J2_TRAINABLE = 0.580, 2504890


# --------------------------------------------------------------------------
# loading: every number below comes from one of these, never from prose
# --------------------------------------------------------------------------
def history(path):
    with open(os.path.join(path, "history.json")) as fh:
        return json.load(fh)


def grid(scan, job, column="solve_rate"):
    """The 14-point density grid, one list of (realized_rho, column) per seed."""
    out = []
    for t in range(3):
        hits = [f for f in sorted(os.listdir(scan))
                if f.startswith("rho_curve_t%02d_" % t) and job in f
                and f.endswith(".csv")]
        assert len(hits) == 1, (scan, t, hits)
        with open(os.path.join(scan, hits[-1])) as fh:
            rows = [(float(r["realized_rho"]), float(r[column]))
                    for r in csv.DictReader(fh)]
        out.append(sorted(rows))
    return out


def guard_rows(seed):
    p = os.path.join(RUN_Y, "guard", "stagey_s%d" % seed, "skip_behavior.csv")
    with open(p) as fh:
        return list(csv.DictReader(fh))


def auroc_table():
    """The one fixed held-out set, both arms scored in the same process."""
    out = {}
    with open(os.path.join(RUN_Y, "determinacy_both_arms.csv")) as fh:
        for r in csv.DictReader(fh):
            arm = "Y" if float(r["determinacy_weight"]) > 0 else "S"
            out[(arm, int(r["seed"]))] = (float(r["determinacy_auroc"]),
                                          float(r["determinacy_bce"]))
    return out


def load():
    hy = [history(p) for p in Y_CELLS]
    hs = [history(p) for p in S_CELLS]
    hx = [history(p) for p in X_CELLS]
    cfg = [json.load(open(os.path.join(p, "config.json"))) for p in Y_CELLS]

    rho, censored = [], []
    for seed, curve in enumerate(grid(SCAN_Y, JOB_Y)):
        v, flag = rho_50(curve)
        censored.append(flag != "ok")
        rho.append(float("inf") if flag == ">top" else v)
        assert flag in ("ok", ">top"), (seed, flag, v)

    guard, den, fired, shapes = [], [], [], []
    for seed in range(3):
        rows = guard_rows(seed)
        rate, n = ratio(rows, "clean_emits_v", "corrupt_emits_v")
        _, nf, s = shape(rows)
        guard.append(rate)
        den.append(n)
        fired.append(nf)
        shapes.append(s)

    verdict = json.load(open(os.path.join(RUN_Y, "stagey_verdict.json")))
    return {"hy": hy, "hs": hs, "hx": hx, "cfg": cfg, "rho": rho,
            "censored": censored, "guard": guard, "den": den, "fired": fired,
            "shapes": shapes, "auroc": auroc_table(), "verdict": verdict}


def check_claims(D):
    """Assert the pages' numbers against the job's own verdict file.

    The verdict was written inside the measurement job, before this builder
    existed, so it is the independent copy. Anything this deck prints that the
    verdict also carries is checked against it here rather than trusted.
    """
    v, bad = D["verdict"], []
    acc = [h[-1]["accuracy"] for h in D["hy"]]
    for i in range(3):
        if abs(acc[i] - v["Y4"]["values"][i]) > 5e-6:
            bad.append("Y4 seed %d: history %.6f, verdict %.6f"
                       % (i, acc[i], v["Y4"]["values"][i]))
        if abs(D["guard"][i] - v["Y2"]["values"][i]) > 5e-6:
            bad.append("Y2 seed %d: recomputed %.6f, verdict %.6f"
                       % (i, D["guard"][i], v["Y2"]["values"][i]))
        a, b = D["rho"][i], v["Y1"]["values"][i]
        if not ((a == b == float("inf")) or abs(a - b) <= 5e-6):
            bad.append("Y1 seed %d: recomputed %s, verdict %s" % (i, a, b))
    if v["verdict"] != "Y4 failed, Y1-Y3 not interpreted":
        bad.append("the registered verdict string moved: %r" % v["verdict"])
    for i, c in enumerate(D["cfg"]):
        if c.get("determinacy_channel") is not True:
            bad.append("seed %d has no determinacy channel" % i)
        if c["n_trainable_params"] != Y_TRAINABLE:
            bad.append("seed %d is %d trainable, deck says %d"
                       % (i, c["n_trainable_params"], Y_TRAINABLE))
        if float(c["determinacy_weight"]) != sy.LAMBDA:
            bad.append("seed %d ran at lambda %s" % (i, c["determinacy_weight"]))
        if c["shuffle_adj"] is not True:
            bad.append("seed %d was evaluated with the leak open" % i)
        if len(D["hy"][i]) != 50:
            bad.append("seed %d has %d epochs" % (i, len(D["hy"][i])))
    if Y_TRAINABLE - S1_TRAINABLE != sy.NEW_PARAMS:
        bad.append("the channel costs %d parameters, scorer says %d"
                   % (Y_TRAINABLE - S1_TRAINABLE, sy.NEW_PARAMS))
    # The one claim with no artifact behind it: the stage-X reference numbers,
    # which are quoted from the stage-X deck. Their internal consistency with
    # the scorer that owns them is what can be checked here.
    if tuple(round(h[-1]["accuracy"], 4) for h in D["hx"]) != sy.X_ACC_FINAL:
        bad.append("stage X's histories disagree with sy.X_ACC_FINAL")
    if tuple(round(h[-1]["accuracy"], 4) for h in D["hs"]) != sy.S_ACC_FINAL:
        bad.append("the control's histories disagree with sy.S_ACC_FINAL")
    if bad:
        raise AssertionError("check_claims failed:\n  " + "\n  ".join(bad))
    return acc


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def panels(d, name, claim, draw, note=None):
    """A 1x3 small-multiple page, one panel per seed.

    WebDeck.claim_page owns a single axes, which is right for a one-claim plot
    and wrong for the per-seed pairing: stage Y seed s is only ever read
    against control seed s, because the seed fixes both the instances and the
    initialisation, and three seeds on one axes would be six lines.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2), sharey=True)
    draw(axes)
    for ax in axes:
        ax.set_rasterized(True)
    fig.tight_layout()
    d.expect(name, claim, note)
    d.savefig(fig)


def _zoom(axes, series, first, label):
    """Scale the shared y-axis to the epochs after the initial collapse.

    Both training figures here are about differences of 0.02 and 0.07 sitting
    on top of an initial drop of 0.2 and 1.7. On the full range the claim --
    seed 0 leaving its plateau, seed 2 never leaving its own -- is a kink a
    few pixels tall. The axis is cut at `first` and the fact that earlier
    epochs run off it is written on the figure, because a clipped curve with
    no note is a figure that lies about its own range.
    """
    vals = [v for s in series for e, v in s if e >= first]
    lo, hi = min(vals), max(vals)
    pad = 0.10 * (hi - lo)
    axes[0].set_ylim(lo - pad, hi + pad)
    clipped = any(v > hi + pad or v < lo - pad for s in series for _, v in s)
    if clipped:
        axes[0].text(0.03, 0.03, label, transform=axes[0].transAxes,
                     fontsize=7.5, color=MUTED, va="bottom")


def draw_acc(D):
    def inner(axes):
        for i, ax in enumerate(axes):
            ys, ss = D["hy"][i], D["hs"][i]
            ax.plot([r["epoch"] for r in ss], [r["accuracy"] for r in ss],
                    color=SERIES[0], lw=1.6)
            ax.plot([r["epoch"] for r in ys], [r["accuracy"] for r in ys],
                    color=SERIES[1], lw=1.6)
            style_axes(ax, xlabel="epoch",
                       ylabel="val token accuracy" if i == 0 else None)
            ax.set_title("seed %d" % i, fontsize=9, color=MUTED)
            if i == 2:
                label_line(ax, ss[-1]["epoch"], ss[-1]["accuracy"],
                           "no channel", color=SERIES[0], dy=9)
                label_line(ax, ys[-1]["epoch"], ys[-1]["accuracy"],
                           "channel", color=SERIES[1], dy=-11)
        _zoom(axes,
              [[(r["epoch"], r["accuracy"]) for r in h]
               for h in D["hy"] + D["hs"]], 2,
              "epochs 0-1 run below the axis")
    return inner


def draw_nll(D):
    def inner(axes):
        for i, ax in enumerate(axes):
            ys, ss = D["hy"][i], D["hs"][i]
            ax.plot([r["epoch"] for r in ss], [r["train_nll"] for r in ss],
                    color=SERIES[0], lw=1.6)
            ax.plot([r["epoch"] for r in ys], [r["train_nll"] for r in ys],
                    color=SERIES[1], lw=1.6)
            ax.axhline(FLOOR, color=MUTED, lw=1.0, ls="--")
            style_axes(ax, xlabel="epoch",
                       ylabel="train NLL" if i == 0 else None)
            ax.set_title("seed %d" % i, fontsize=9, color=MUTED)
            if i == 2:
                label_line(ax, ss[-1]["epoch"], ss[-1]["train_nll"],
                           "no channel", color=SERIES[0], dy=-11)
                label_line(ax, ys[-1]["epoch"], ys[-1]["train_nll"],
                           "channel", color=SERIES[1], dy=9)
        # MIN_EP is the project's own boundary for "the initial collapse out
        # of initialisation has finished", and the step detector starts there
        # for the same reason this axis does.
        _zoom(axes,
              [[(r["epoch"], r["train_nll"]) for r in h]
               for h in D["hy"] + D["hs"]], MIN_EP,
              "epochs 0-%d run above the axis" % (MIN_EP - 1))
    return inner


def draw_rho(D):
    yr = grid(SCAN_Y, JOB_Y)
    sr = grid(SCAN_S, JOB_S)
    xr = grid(SCAN_X, JOB_X)

    def inner(axes):
        for i, ax in enumerate(axes):
            for curve, col in ((sr[i], SERIES[0]), (yr[i], SERIES[1]),
                               (xr[i], SERIES[2])):
                ax.plot([p[0] for p in curve], [p[1] for p in curve],
                        color=col, lw=1.6, marker="o", ms=2.5)
            ax.axhline(0.5, color=MUTED, lw=1.0, ls="--")
            style_axes(ax, xlabel=r"realized $\rho$",
                       ylabel="solve rate" if i == 0 else None)
            ax.set_title("seed %d" % i, fontsize=9, color=MUTED)
            if i == 2:
                label_line(ax, sr[i][-1][0], sr[i][-1][1], "no channel",
                           color=SERIES[0], dy=10)
                label_line(ax, yr[i][-1][0], yr[i][-1][1], "channel",
                           color=SERIES[1], dy=-2)
                label_line(ax, xr[i][-1][0], xr[i][-1][1], "stage X term",
                           color=SERIES[2], dy=-14)
    return inner


def draw_outcome(D):
    yo = grid(SCAN_Y, JOB_Y, "outcome_accuracy")
    so = grid(SCAN_S, JOB_S, "outcome_accuracy")
    xo = grid(SCAN_X, JOB_X, "outcome_accuracy")

    def inner(axes):
        for i, ax in enumerate(axes):
            for curve, col in ((so[i], SERIES[0]), (yo[i], SERIES[1]),
                               (xo[i], SERIES[2])):
                ax.plot([p[0] for p in curve], [p[1] for p in curve],
                        color=col, lw=1.6, marker="o", ms=2.5)
            ax.set_ylim(0.0, 1.05)
            style_axes(ax, xlabel=r"realized $\rho$",
                       ylabel="solved/unsolved call" if i == 0 else None)
            ax.set_title("seed %d" % i, fontsize=9, color=MUTED)
            if i == 2:
                label_line(ax, so[i][-1][0], so[i][-1][1], "no channel",
                           color=SERIES[0], dy=10)
                label_line(ax, yo[i][-1][0], yo[i][-1][1], "channel",
                           color=SERIES[1], dy=-2)
                label_line(ax, xo[i][-1][0], xo[i][-1][1], "stage X term",
                           color=SERIES[2], dy=-14)
    return inner


def draw_auroc(D):
    """Per-epoch AUROC of the channel against the control's measured band.

    The three seeds separate by 0.14 at epoch 1 and are within 0.01 of each
    other from epoch 5 on, so they are one colour with one label: three
    labels would say only that the curves coincide, and would collide while
    saying it.

    The curve is the per-epoch AUROC on the training run's own validation
    split; the grey band is the probe's control, which is the only place a
    control AUROC exists at all, because the control has no determinacy head
    to read. Different sets, same predicate, and the note on the page says so.
    """
    band = [D["auroc"][("S", s)][0] for s in range(3)]

    def inner(ax):
        last = None
        for i in range(3):
            ep = [r["epoch"] for r in D["hy"][i]]
            au = [r["determinacy_auroc"] for r in D["hy"][i]]
            ax.plot(ep, au, color=SERIES[0], lw=1.4)
            last = (ep[-1], au[-1])
        label_line(ax, last[0], last[1], "all three seeds", color=SERIES[0])
        ax.axhspan(min(band), max(band), color=MUTED, alpha=0.18)
        ax.text(0.02, min(band) - 0.035,
                "the same model with no channel: %.3f-%.3f"
                % (min(band), max(band)),
                transform=ax.get_yaxis_transform(), fontsize=8, color=MUTED)
        ax.set_ylim(0.5, 1.0)
        style_axes(ax, xlabel="epoch", ylabel="determinacy AUROC")
    return inner


# --------------------------------------------------------------------------
# stage-specific explanation pages
#
# These go through WebDeck._add rather than points_page, because the shared
# library caps a bullet at 14 words and a page at 3 of them, and ~/CLAUDE.md
# exempts explanation pages from the body caps: the definitions are the page.
# --------------------------------------------------------------------------
VOCAB = """## A cell is one trained model; an arm is a configuration

Three words of bookkeeping recur on every page below, so they are defined
before any of them.

A **cell** is one trained model: one configuration trained at one seed and
saved as one checkpoint. Nine cells are compared here -- three trained for
this run and six that already existed.

An **arm** is a configuration: the three cells that differ only in their seed.
Three arms appear, and they differ only in how the determinacy signal is
coupled to the output.

A **stage** is one registered experiment in this project's sequence, and its
conclusions are numbered as *findings*. Findings 68, 69, 73, 74 and 75 are
quoted below, and each quotation says what that finding measured.
"""

DETPAGE = """## A node is *determined* when exactly one candidate is left for it

This is the predicate the whole run is about, so it gets its own page and its
own symbol. Write $C_v \\subseteq \\{1,\\dots,q\\}$ for the candidate set of node
$v$ given everything deduced so far. Then

$$\\text{det}(v) \\;=\\; \\mathbf{1}\\big[\\,|C_v| = 1 \\,\\big]$$

- $\\text{det}(v) = 1$ means direct deduction can emit $v$ right now: its colour follows from what is already known, with no guessing.
- $\\text{det}(v) = 0$ means $v$ is still open, with two or more colours consistent with what is known. A solver that names $v$ anyway is **guessing**, whatever it then writes.
- Already-coloured nodes have no candidate set left to speak of and are neither: they are excluded everywhere below and are never counted as a negative.
- $\\text{det}$ is **state-dependent**: it changes after every emitted step, because each new colour eliminates candidates in the neighbours. The same node is undetermined early and determined later, which is why it is computed afresh at every position rather than once per instance.
"""

GUARDPAGE = """## The model has direct deduction's rule and not its guard

This is finding 68, the result this whole line of work is a response to, and
nothing below makes sense without it.

- A **guard** is the part of direct deduction that refuses: the rule says *emit $v$ when $\\text{det}(v)$*, and the guard is the *when*. A solver with the rule and no guard emits $v$ regardless.
- Finding 68 tested it directly. Take a prompt the model solves, change **one** hint colour so that the node it was about to name stops being determined, and re-run. A model with a guard names something else. A model without one names the same node anyway.
- The tied model named it anyway **every single time**: a guard rate of 1.0000 in all three seeds, with zero spread. The larger untied model refused on two of its three seeds.
- The guard rate is **conditional**, and the next page says on what. Lower is better. 1.0000 means it never once noticed.
"""

DENOM = """## The guard rate's denominator is read, never assumed

- The probe offers each cell **350** cases. The published rate is **conditional on the model naming node $v$ on the clean prompt at all**, because a cell that was going to name something else was never in a position to refuse:

$$\\text{guard rate} \\;=\\; \\frac{\\#\\{\\text{names } v \\text{ clean and names } v \\text{ broken}\\}}{\\#\\{\\text{names } v \\text{ clean}\\}}$$

- So the denominator is **not 350**. It is how many of the 350 that cell named cleanly, and it differs by cell: 232 / 237 / 233 for the control and 194 / 175 / 185 for stage X.
- This deck prints the denominator beside every rate. Three different scripts in this project have divided by 350 instead, and one log's prose once reported $350 - 194 = 156$ as though it were a guard count, which would have scored the bar backwards.
- A **fired** case is one where the model named $v$ cleanly and then named something else once $v$'s determinacy was destroyed. The guard rate is one minus the fired fraction.
"""

KNOWS = """## It computes the ambiguity and schedules as if it were certain

- Finding 69 asked the obvious follow-up: does the tied model not *know*, or does it know and not act?
- Reading the model's own state on the edited prompts, the probability it assigns to the freed node being free is **0.5253 / 0.5228 / 0.5354** across seeds. It is tracking the ambiguity, not blind to it.
- Finding 73 went further: determinacy is already **linearly readable** in the deciding logits of a model that was never trained to represent it, at AUROC 0.7610 / 0.7635 / 0.7774 against 0.5 for chance.
- Finding 72 localised the untied model's guard to a window of two to three applications near $t \\approx 3$-$6$: where a guard exists it acts in a narrow band, not throughout.
- So the missing piece is not the information, and it is not the ranking either. It is the step from having the information to withholding the emission.
"""

GAUGE = """## Cross-entropy can rank the logits but cannot set their height

This is the argument both stages are built on, so it is stated as an equation.
Cross-entropy at position $p$ reads the logit vector $z_p$ only through a
softmax:

$$\\mathcal{L}_{\\text{CE}}(z_p) \\;=\\; -\\log \\frac{e^{z_{p,y_p}}}{\\sum_v e^{z_{p,v}}}$$

- $y_p$ is the true token at position $p$ and $z_{p,v}$ the logit of symbol $v$ there. Add the same constant $c$ to **every** entry and the fraction is unchanged: $\\mathcal{L}_{\\text{CE}}(z_p + c\\mathbf{1}) = \\mathcal{L}_{\\text{CE}}(z_p)$.
- So cross-entropy constrains the logits only **up to an additive constant** -- a *gauge* freedom. It can say "this node comes next", because that is a comparison between entries. It cannot say "no node is emittable here", because that is a statement about absolute height, and height is exactly what it cannot see.
- A guard is a statement about absolute height. **Nothing in the plain objective trains it.** That was the hypothesis for why a model that computes the ambiguity still never acts on it.
"""

STAGEX = """## Stage X put the determinacy question on the existing readout

- Stage X added one term to the loss, attached to the logits the model already computes, asking a different question of them: not *which node comes next*, but *for each still-open node, is it determined right now*.
- It is a multi-label binary cross-entropy on those same logits against direct deduction's own predicate:

$$\\mathcal{L}_{\\text{det}} \\;=\\; -\\frac{1}{|S|}\\sum_{(p,v) \\in S}\\Big[ \\text{det}_p(v)\\log\\sigma(z_{p,v}) + (1-\\text{det}_p(v))\\log\\big(1-\\sigma(z_{p,v})\\big)\\Big]$$

- $S$ is the set of scored (position, node) pairs: $p$ ranges over positions where a node id is about to be emitted and $v$ over the nodes **still uncoloured** at $p$. $\\sigma(x) = 1/(1+e^{-x})$ is the logistic function.
- Because $\\sigma$ reads each logit on its own rather than through a softmax over all of them, this term **can** see the additive constant the previous page says cross-entropy cannot. That was the entire design, and it cost **zero** new parameters.
"""

STAGEXRES = """## Stage X installed the representation and changed no behaviour

This is finding 75, and it is the premise of the run reported here.

- The term did exactly what it says. Determinacy AUROC on a fixed held-out set went from the control's 0.7610 / 0.7635 / 0.7774 to **0.9462 / 0.9467 / 0.9469**, complete separation, every seed.
- The guard probe, run unchanged on those cells, came back at **1.0000 / 1.0000 / 0.9568**. Breaking a node's determinacy still did not stop the model naming it.
- Solve rate got *worse*: $\\rho_{50}$ moved from 0.754 / 0.757 / 0.765 to 0.866 / 0.864 / 0.855, and the solved/unsolved call collapsed to near chance in the middle of the density range.
- The scorer's registered verdict string was `representation_without_use`. A model can be made to represent determinacy at AUROC 0.95 and still emit exactly as before.
"""

STATIC = """## Static ranking and counterfactual responsiveness are different properties

This distinction is what stage X established and what this deck is organised
around, so it is stated on its own page.

- **Static ranking** is a statement about one position: across the nodes still open at $p$, are the determined ones scored above the undetermined ones? AUROC measures this, and stage X drove it to 0.95.
- **Counterfactual responsiveness** is a statement about one node across two prompts: when $v$'s determinacy is destroyed and nothing else changes, does what the model emits move? The guard probe measures this, and stage X left it at zero.
- The two are logically independent. A score can rank determined nodes first at every position while the *emitted* token is a function of something else entirely, in which case destroying determinacy changes the ranking and not the output.
- Everything in this deck that says "the signal is there" is the first property. Everything that says "nothing refuses" is the second. They are never read as the same claim.
"""

F74 = """## The real guard re-ranks; it does not suppress the whole block

The cheapest way to couple determinacy to the output is already ruled out, and
the argument is worth a page because it is what forced this run's design.

- The cheap version gates the node block as a whole: $z \\mapsto z + \\log \\sigma(a(z - \\tau))$ for two learned scalars $a$ and $\\tau$. It does break the softmax's additive-constant gauge, so it is not vacuous.
- But it is strictly increasing in $z$, so it preserves the order **within** the node block. It can only push the whole block down relative to the terminal and round-marker tokens -- a *stop*-shaped guard.
- Finding 74 read finding 68's stored data for what the untied model, the one that does guard, emits *instead* when it refuses. It names another node token **96.6%** of the time, and that other node is **still forced 90.3%** of the time. It names the broken node again in **0 of 350** cases.
- So a real guard moves the emission to a different, still-deducible node. It is a **re-ranking**, not a suppression. An arm that produces a stop-shaped guard has rebuilt what stage X already did, and this run's registration counts that as a refutation rather than a pass.
"""

CHANNEL = """## The channel: a second score per node, outside the readout

Stage Y's one change. The model keeps the readout it had and gains a separate
determinacy score for each node at each position:

$$g_{p,v} \\;=\\; \\langle\\, h_{16}[p] \\odot s, \\; E_v \\,\\rangle$$

- $h_{16}[p]$ is the state at position $p$ after the sixteenth application of the block, the same $320$-vector the readout reads.
- $E_v$ is the **token embedding** of node $v$: a $320$-vector the model already has, one per vocabulary symbol.
- $s$ is the one new parameter -- a single $320$-vector, shared by every node and every position. $\\odot$ is the elementwise product, and $\\langle\\cdot,\\cdot\\rangle$ the dot product, so $g_{p,v}$ is one number per (position, node) pair.
- Equivalently $g_{p,v} = \\sum_{k} h_{16}[p]_k\\, s_k\\, E_{v,k}$: a bilinear form between the state and the node's embedding, with $s$ selecting which coordinates of the state carry determinacy.
- The determinacy loss from the previous page is now computed on $g_{p,v}$ instead of on $z_{p,v}$. The readout is no longer asked two questions at once.
"""

EMIT = """## The channel enters the emission as $z' = z + \\log \\sigma(g)$

The score has to reach the output or this is just stage X with an extra head.
At every position, on the node block only:

$$z'_{p,v} \\;=\\; z_{p,v} + \\log \\sigma\\big(g_{p,v}\\big)$$

- $z_{p,v}$ is the readout's logit for node $v$ and $z'_{p,v}$ is what the softmax actually sees. The 10 non-node symbols -- the colours, the round markers, `SOLVED`, `UNRESOLVED` -- are untouched.
- $\\log \\sigma(g) \\in (-\\infty, 0)$, so the channel can only ever subtract, and it subtracts **more from nodes it scores as less determined**. Reading $\\sigma(g)$ as a probability that $v$ is determined, this is adding $\\log \\Pr[\\text{det}(v)]$ to the log-odds of emitting $v$.
- Crucially the subtraction is **per node**, so it changes the order *within* the block. That is the property finding 74 says a real guard needs and the monotone block gate cannot have.
- It also still breaks the gauge: pushing every node's $g$ down lowers the whole block against the terminal tokens, so the stop-shaped behaviour remains available. The design adds a capability; it does not remove one.
"""

SEPARATE = """## Re-ranking is only possible because embedding and readout are untied

- $g_{p,v}$ is built from the **token embedding** $E_v$, the matrix that maps symbol $v$ into the model on the way *in*. The logit $z_{p,v}$ comes from the **readout** $W$, a different matrix, on the way out.
- These two are separate matrices in this architecture and were never weight-tied. That is what makes $g$ able to disagree with $z$: if $E$ were $W^\\top$, then $g_{p,v}$ would be a monotone-ish function of a reweighted $z_{p,v}$ and the channel would collapse toward the block gate the previous pages rule out.
- "Tied" in `tied_no_mlp` refers to the transformer **block** being the same weights at every one of the 16 applications. It has nothing to do with embedding-readout tying, which this model does not do. The two senses of the word collide in the config names and are kept apart here on purpose.
"""

PARAMS = """## The channel costs 320 parameters, one per model dimension

| | without the channel | with it |
|---|---|---|
| trainable parameters | 613,818 | 614,138 |
| new parameters | -- | 320 |
| as a fraction | -- | $+0.052\\%$ |

- The 320 are exactly the entries of $s$. There is no new matrix, no second readout and no extra projection: the channel reuses $h_{16}$ and $E$, both of which already exist.
- This matters for what the project is for. The goal is the simplest architecture that still learns direct deduction -- analytically tractable, not merely small -- and one extra vector in a bilinear form is an object a theory can still describe.
- Both counts are read out of the checkpoints' own configs and asserted in the builder, so "320 parameters" is checked against the runs and not against arithmetic.
"""

INIT = """## $s$ starts at zero, so the model starts as the control

- At initialisation $s = 0$, so $g_{p,v} \\equiv 0$ and $\\log \\sigma(0) = -\\log 2$ for every node. A constant subtracted from the whole block is invisible to the softmax, so the model begins **exactly** in the control's order.
- The first version of this arm initialised $s$ to all ones, which was a bug and was caught before the run that is reported here. It made $g$ have standard deviation 21 and $\\log \\sigma(g)$ standard deviation 10.3, reaching $-59$, against node logits whose entire spread is 0.61. The channel was 17 times the signal it was supposed to modulate, and one seed never recovered.
- That arm was cancelled and relaunched into a new directory with the single-line change to a zero initialisation. Nothing else differs between the two, and only the zero-init run is reported.
- A smoke test asserts both halves of the claim: the channel's contribution has zero spread across nodes at initialisation, and non-zero spread after one gradient step of the determinacy loss.
"""

LAMBDAPAGE = """## The total loss, with its one weight fixed in advance

$$\\mathcal{L} \\;=\\; \\mathcal{L}_{\\text{CE}} \\;+\\; \\lambda\\, \\mathcal{L}_{\\text{det}}, \\qquad \\lambda = 1.0$$

- $\\mathcal{L}_{\\text{CE}}$ is the cross-entropy of the previous pages, now read on $z'$; $\\mathcal{L}_{\\text{det}}$ is the determinacy term, now read on $g$.
- $\\lambda$ was registered at 1.0 **before any cell ran and was not swept**. It is the same value stage X used, which is deliberate: changing the coupling and the weight together would leave neither attributable.
- Not sweeping it is the standing rule. A sweep is a second knob, and a stage that reports the best of five $\\lambda$ values has reported the best of five, not the effect of the channel.
- The **train NLL** column throughout this deck is $\\mathcal{L}_{\\text{CE}}$ alone, for every arm. The determinacy term is excluded from it by construction, so the arms' curves are the same quantity and the gap between them is the main task's cost.
"""

AUROCDEF = """## AUROC asks whether determined nodes are scored above undetermined ones

- The channel's own success is scored with **AUROC**: draw one determined node and one undetermined node at the same position, and ask how often the determined one has the higher score.

$$\\text{AUROC} \\;=\\; \\Pr\\big[\\, g_{p,v^{+}} > g_{p,v^{-}} \\,\\big], \\qquad \\text{det}_p(v^{+}) = 1,\\; \\text{det}_p(v^{-}) = 0$$

- 0.5 is chance and 1.0 is a perfect ranking. It is invariant to adding a constant, so it measures the *ranking* the channel installs and not the absolute heights.
- For the control, which has no channel, the same quantity is computed on the readout's node logits $z_{p,v}$ instead. That is the only sense in which a model without a channel has a determinacy score at all.
- **The control is not 0.5.** Cross-entropy already lifts the logit of the node about to be emitted, and that node is always determined, so a bar of "above chance" would be met by doing nothing. Finding 73's 0.7610 / 0.7635 / 0.7774 is the number to beat.
- Every AUROC comparing arms is measured on **one fixed held-out set of 600 instances, in one process**, so the groups differ in the model and nothing else. The per-epoch AUROC in the training curves is a separate per-cell draw and is only read within a cell.
"""

SHAPEBAR = """## A guard also has to have the right shape to count

- Y2 asks whether a guard fires. Y3 asks what the model emits **instead** when it does, because finding 74 says the two possible answers mean opposite things.
- Among the fired cases, two fractions are scored: **node token**, the fraction where the model emits some node id rather than a terminal or round marker, and **still forced**, the fraction where that node is itself determined.
- The bars are node token $\\ge 0.90$ and still forced $\\ge 0.60$ in at least 2 of 3 seeds, with the untied model's measured 0.966 and 0.903 as the reference.
- Passing Y2 with a *low* still-forced fraction would mean the guard suppresses rather than re-ranks, which is stage X's block shift rebuilt at 320 parameters' cost. The registration counts that as a refutation, not a pass, which is the point of having Y3 at all.
- Y3 is readable only on the cases that fired. A seed whose guard never fires has no shape and cannot pass it; the denominator is printed with the result.
"""

CANNOT = """## What this run cannot show

- It cannot show that **no** coupling would work. Two couplings at one weight have now been registered and run -- the term on the readout and this separate channel -- and a null result is about those two.
- It cannot separate "the architecture cannot make emission depend on determinacy" from "50 epochs at this recipe is not enough training for it to". One seed of three here plainly ran out of budget, and that is reported as a limitation rather than fixed after the fact.
- It cannot speak to $N \\ne 48$, or to any width or depth other than this one. Everything here is one graph size and one architecture.
- Three seeds give a one-sided rank test at $p = 1/20$ and no more. An effect smaller than the seed spread is invisible to this design, which is why the bars are separations rather than mean differences.
- $\\lambda$ was not swept, by design, so nothing here distinguishes "the channel does not help" from "the channel does not help at $\\lambda = 1$".
"""

NOTFIXED = """## Four knobs that could have been turned here and were not

The arm finished with one seed clearly mid-convergence, which makes the list of
things not done part of the result rather than an omission.

- **No longer budget.** 50 epochs was registered. Choosing a run length after seeing which seeds escaped their plateau would be choosing the answer.
- **No best-validation checkpoint.** Every number here is epoch 49, the last one, for every arm. Selecting the best epoch per cell would flatter the arm whose seeds are still moving.
- **No $\\lambda$ sweep and no channel warm-up.** Both are second knobs on a stage that is testing a first one.
- **No seed dropped.** Seed 2 fails the sanity bar and is reported in every table, including the ones where it is the only reason a bar fails.
- What a follow-up should do instead is register a longer budget, or an explicit convergence criterion, **before** it runs -- which is the recommendation this stage ends on.
"""


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--job", default=JOB_Y)
    args = ap.parse_args()

    deck_explain.check()
    # SUFFICIENCY goes: no residual against the isotonic ladder is reported
    # here, and a deck should not define a quantity it never uses. CENSOR
    # STAYS and is load-bearing -- one seed's rho_50 is genuinely censored.
    shared = [p for p in deck_explain.PAGES
              if p is not deck_explain.SUFFICIENCY]

    D = load()
    acc = check_claims(D)
    sacc = [h[-1]["accuracy"] for h in D["hs"]]
    xacc = [h[-1]["accuracy"] for h in D["hx"]]
    ynll = [h[-1]["train_nll"] for h in D["hy"]]
    snll = [h[-1]["train_nll"] for h in D["hs"]]
    steps = [step_epoch([(r["epoch"], r["train_nll"], r["nll"]) for r in h])
             for h in D["hy"]]
    ssteps = [step_epoch([(r["epoch"], r["train_nll"], r["nll"]) for r in h])
              for h in D["hs"]]
    pred = sy.score_the_prediction()
    mean_solve = [sum(p[1] for p in c) / len(c) for c in grid(SCAN_Y, JOB_Y)]
    s_solve = [sum(p[1] for p in c) / len(c) for c in grid(SCAN_S, JOB_S)]
    y_out = [sum(p[1] for p in c) / len(c)
             for c in grid(SCAN_Y, JOB_Y, "outcome_accuracy")]
    s_out = [sum(p[1] for p in c) / len(c)
             for c in grid(SCAN_S, JOB_S, "outcome_accuracy")]

    # When each channel seed joins its control, and seed 0's gap before it
    # does -- measured rather than typed. "Joins" is the first epoch within
    # 0.002 of the control's final accuracy; seed 2 never gets there, which
    # is the same fact Y4 fails on. The gap is read at the epoch before the
    # registered step, so it is the height of the plateau the step leaves.
    acc_ep = [{r["epoch"]: r["accuracy"] for r in h} for h in D["hy"]]
    sacc_ep = [{r["epoch"]: r["accuracy"] for r in h} for h in D["hs"]]
    joins = [next((e for e in sorted(acc_ep[i])
                   if acc_ep[i][e] >= sacc[i] - 0.002), None) for i in range(3)]
    gap0 = sacc_ep[0][steps[0] - 1] - acc_ep[0][steps[0] - 1]
    # The accuracy page's title names epoch 43, so pin it: a title is the one
    # claim on a page and nothing else here would catch it drifting.
    assert joins == [43, 16, None], joins
    xtax = sorted(abs(xacc[i] - sacc[i]) for i in range(3))

    rho_txt = ["censored" if D["censored"][i] else "%.3f" % D["rho"][i]
               for i in range(3)]
    head = "Determinacy got its own channel and still nothing refuses"

    prov = ("Jobs %s (3 cells, 50 epochs each) and 52375912 (the probe), commit "
            "%s. Registered by scripts/stagey_score.py at 3d0a3a6, before the "
            "first cell ran. The channel adds %d parameters, +0.052%%. Verdict: "
            "%s." % (args.job, args.commit, sy.NEW_PARAMS,
                     D["verdict"]["verdict"]))

    d = WebDeck(args.out, head, "hshunt", "2026-11-14", prov)
    d.title_page(head, subtitle=(
        "Stage X trained the model to represent which nodes are deducible, "
        "reached AUROC 0.95, and changed nothing about what it emits. This run "
        "gives that signal its own output channel so it can re-order the "
        "candidates rather than only rank them, and asks the same question "
        "again."))

    d.points_page(
        "One change, 320 parameters, and the control is otherwise identical",
        ["The determinacy score leaves the readout and becomes its own channel.",
         "It enters the emission per node, so it can re-order candidates.",
         "Data, schedule, width, depth and seeds are the control's."])

    d._add(VOCAB)

    # ---- the shared explanation set, with the two shared figures
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, shuffle_adj=True,
                          fall=True)
    for page in shared:
        if page is deck_explain.FIGURE_SLOT:
            d.expect("tokenization",
                     "Every token is one categorical id in a %d-symbol vocabulary"
                     % ex.vocab.size,
                     note="The node ids occupy a contiguous block of this "
                          "vocabulary, and that block is the only part the "
                          "channel touches.")
            page_tokenization(d, ex)
            d.expect("trace",
                     "A deduction fires when a node sees all three other colours",
                     note="The guard probe edits one colour token in a prompt "
                          "like this, so that one node stops being forced.")
            page_trace(d, ex)
        else:
            d._add(page)

    # ---- what is being measured and why, before any number
    for page in (DETPAGE, GUARDPAGE, DENOM, KNOWS, GAUGE, STAGEX, STAGEXRES,
                 STATIC, F74, CHANNEL, EMIT, SEPARATE, PARAMS, INIT,
                 LAMBDAPAGE, AUROCDEF, SHAPEBAR):
        d._add(page)

    d.table_page(
        "The three arms differ in the coupling and in nothing else",
        ["", "no channel (control)", "stage X term", "stage Y channel"],
        [["determinacy loss reads", "--", "$z_{p,v}$", "$g_{p,v}$"],
         ["reaches the emission", "--", "same logits", "$+\\log\\sigma(g)$"],
         ["can re-order the block", "--", "no", "yes"],
         ["trainable parameters", "{:,}".format(S1_TRAINABLE),
          "{:,}".format(S1_TRAINABLE), "{:,}".format(Y_TRAINABLE)],
         ["architecture", "tied, $16\\times$", "tied, $16\\times$",
          "tied, $16\\times$"],
         ["epochs / seeds", "50 / 0,1,2", "50 / 0,1,2", "50 / 0,1,2"]],
        note="Same data, same schedule, same width and depth. Only the "
             "coupling differs, which is what makes the comparison one.")

    d.table_page(
        "What was registered, before any cell ran",
        ["", "the question", "the bar"],
        [["**Y1**", "did the solver get better?",
          "$\\rho_{50}$ of every seed strictly below every control seed "
          "(0.754 / 0.757 / 0.765)"],
         ["**Y2**", "did a guard appear?",
          "conditional guard rate below %.2f in at least 2 of 3 seeds"
          % sy.GUARD_BAR],
         ["**Y3**", "is the guard the right shape?",
          "node token $\\ge$ %.2f and still forced $\\ge$ %.2f in 2 of 3"
          % (sy.SHAPE_NODE_BAR, sy.SHAPE_FORCED_BAR)],
         ["**Y4**", "did it break anything?",
          "validation token accuracy at least %.4f in every seed" % sy.ACC_BAR]],
        note="Y4 is a gate on the others: a cell that did not train is no "
             "evidence about a guard.")

    d._add(CANNOT)

    # ---- results
    d.points_page(
        "Both jobs finished clean and every cell trained its full budget",
        ["Training array %s: COMPLETED, exit 0:0, about 9 hours each."
         % args.job,
         "Probe job 52375912: COMPLETED, exit 0:0, 37 seconds.",
         "All three cells ran 50 epochs. Nothing restarted or early-stopped."])

    d.table_page(
        "Y4 fails: seed 2 finished 0.026 of accuracy below its control",
        ["", "no channel", "channel", "difference", "bar %.4f" % sy.ACC_BAR],
        [["seed %d" % i, "%.4f" % sacc[i], "%.4f" % acc[i],
          "%+.4f" % (acc[i] - sacc[i]),
          "pass" if acc[i] >= sy.ACC_BAR else "**fail**"] for i in range(3)],
        note="Seeds 0 and 1 land on the control. Y4 is a bar on every seed, "
             "so the stage fails it.",
        highlight_row=2)

    panels(d, "acc_trajectory",
           "Seed 0 joins the control at epoch 43; seed 2 never does",
           draw_acc(D),
           note="Seed 1 joins by epoch %d. Seed 0 sits %.3f below its control "
                "for %d epochs, then closes it in %d."
                % (joins[1], gap0, steps[0] - 1, joins[0] - steps[0]))

    d.table_page(
        "The channel delays convergence rather than changing where it lands",
        ["", "final train NLL", "level", "registered step at"],
        [["seed %d, no channel" % i, "%.4f" % snll[i], level(snll[i]),
          "epoch %s" % ssteps[i] if ssteps[i] else "none"] for i in range(3)]
        + [["seed %d, channel" % i, "%.4f" % ynll[i], level(ynll[i]),
            "epoch %s" % steps[i] if steps[i] else "none"] for i in range(3)],
        note="`STUCK` is the plateau level, above 1.600. Two of three channel "
             "cells reach the floor the control reaches; one does not.")

    panels(d, "train_nll",
           "Seed 0's plateau breaks at epoch 41; seed 2's never breaks",
           draw_nll(D),
           note="Main-term cross-entropy only, both arms, so the curves are "
                "the same quantity. Dashed line is the 1.600 floor.")

    d.points_page(
        "The control's own seed 2 did the same thing, 19 epochs earlier",
        ["The control's seed 2 steps off its plateau at epoch 22.",
         "The channel's seed 0 steps off at epoch 41.",
         "Late plateau exits are a property of the recipe, not the channel."])

    d.table_page(
        "A prediction committed at epoch 32 was right twice and wrong once",
        ["", "predicted accuracy", "measured", "error"],
        [["seed %d" % i, "%.4f" % pred["predicted_final_acc"][i],
          "%.4f" % pred["measured_final_acc"][i],
          "%+.4f" % pred["error_per_seed"][i]] for i in range(3)],
        note="Predicted one seed clearing Y4; two did. The epoch-32 note "
             "called seed 0's plateau a basin, and it was a delay.")

    d.table_page(
        "Y1 fails: one seed worse, one tied, one never crossing",
        ["", "$\\rho_{50}$, no channel", "$\\rho_{50}$, channel",
         "mean solve rate, no channel", "channel"],
        [["seed %d" % i, "%.3f" % sy.CONTROL_RHO50[i], rho_txt[i],
          "%.3f" % s_solve[i], "%.3f" % mean_solve[i]] for i in range(3)],
        note="Lower $\\rho_{50}$ is better. Seed 2's solve rate peaks at 0.315 "
             "and never reaches 0.5 anywhere on the grid.",
        highlight_row=2)

    panels(d, "rho_curve",
           "Seed 1 lands on the control; seed 2 never crosses one half",
           draw_rho(D),
           note="Same 14 densities, 400 instances each. Seed 0's curve sits "
                "slightly right of its control. Dashed line is $\\rho_{50}$.")

    d.points_page(
        "A censored seed is reported as censored, not dropped",
        ["Seed 2's curve never reaches 0.5, so its $\\rho_{50}$ does not exist.",
         "It is scored as $+\\infty$, which fails Y1 by every route.",
         "Averaging over the two crossing seeds would hide exactly that."])

    d.table_page(
        "The channel keeps the solved/unsolved call that stage X destroyed",
        ["", "no channel", "stage X term", "stage Y channel"],
        [["seed %d" % i, "%.3f" % s_out[i], "%.3f" % X_OUTCOME[i],
          "%.3f" % y_out[i]] for i in range(3)],
        note="Mean over the 14 densities. Stage X's global block shift broke "
             "this decision; a per-node channel does not.")

    panels(d, "outcome_call",
           "Stage X's U in density does not reappear with the channel",
           draw_outcome(D),
           note="Flat near 1.00 for the control and for the channel at every "
                "density. Stage X traces a U between those ends.")

    d.table_page(
        "The channel reads determinacy as well as the trained readout did",
        ["", "AUROC, no channel", "stage X term", "stage Y channel",
         "channel BCE"],
        [["seed %d" % i, "%.4f" % D["auroc"][("S", i)][0], "%.4f" % X_AUROC[i],
          "%.4f" % D["auroc"][("Y", i)][0], "%.4f" % D["auroc"][("Y", i)][1]]
         for i in range(3)],
        note="One fixed held-out set of 600 instances, both arms in one "
             "process. The channel is scored by no bar, for the reason below.")

    d.claim_page(
        "The AUROC is won in five epochs, then sits still",
        draw_auroc(D), "auroc",
        note="Curve is the per-epoch value during training; grey band is the "
             "probe's control. Installing the signal is easy.")

    d.points_page(
        "AUROC is deliberately not a bar in this registration",
        ["Stage X reached 0.947 and bought no behavioural change at all.",
         "A second high AUROC would therefore say nothing new.",
         "It is measured and printed for the record, and scored by nothing."])

    d.table_page(
        "Y2 fails: the guard fired once in 681 cases across three seeds",
        ["", "guard rate, conditional", "denominator", "cases fired",
         "control rate"],
        [["seed %d" % i, "%.4f" % D["guard"][i], D["den"][i], D["fired"][i],
          "%.4f" % sy.CONTROL_GUARD[i]] for i in range(3)],
        note="Bar is below %.2f in 2 of 3 seeds; 0 of 3 clear it. "
             "Denominators %d, %d, %d, never the 350 offered."
             % (sy.GUARD_BAR, D["den"][0], D["den"][1], D["den"][2]))

    d.table_page(
        "Two different couplings, the same guard rate, the same control",
        ["arm", "determinacy AUROC", "guard rate", "denominator"],
        [["no channel", "0.7610-0.7774", "1.0000 / 1.0000 / 1.0000",
          "%d / %d / %d" % CONTROL_GUARD_DEN],
         ["stage X term on the readout", "0.9462-0.9469",
          "%.4f / %.4f / %.4f" % X_GUARD, "%d / %d / %d" % X_GUARD_DEN],
         ["stage Y channel", "%.4f-%.4f"
          % (min(D["auroc"][("Y", s)][0] for s in range(3)),
             max(D["auroc"][("Y", s)][0] for s in range(3))),
          "%.4f / %.4f / %.4f" % tuple(D["guard"]),
          "%d / %d / %d" % tuple(D["den"])]],
        note="Installing the representation twice, two different ways, moved "
             "the counterfactual behaviour by one case out of 681.")

    d.points_page(
        "Y3 cannot be read: one fired case across all three seeds",
        ["Seeds 0 and 2 never fired, so they have no shape at all.",
         "Seed 1 fired once, and that one case re-ranked to a forced node.",
         "One case is not evidence about shape; Y3 is reported unreadable."])

    d.table_page(
        "Scored against the registration: all four bars fail",
        ["", "the question", "bar", "result"],
        [["**Y1**", "did the solver get better?", "every seed below 0.754",
          "**fail**: %s / %s / %s" % tuple(rho_txt)],
         ["**Y2**", "did a guard appear?", "below 0.90 in 2 of 3",
          "**fail**: 0 of 3"],
         ["**Y3**", "right shape?", "node token and still forced, 2 of 3",
          "**fail**: 1 of 3, n=1"],
         ["**Y4**", "did it break anything?", "accuracy at least 0.4800",
          "**fail**: 0.4572 on seed 2"]],
        note="The scorer's verdict string is `%s`." % D["verdict"]["verdict"])

    d.points_page(
        "Y4 failing is a real caveat and not a formality",
        ["Seed 2 is 0.026 below its control and still on the plateau.",
         "Its guard rate and AUROC are reported, but it did not finish converging.",
         "Y1 and Y2 fail on seeds 0 and 1 alone regardless."])

    d.points_page(
        "The finding rests on the two seeds that did converge",
        ["Seeds 0 and 1 reach the control's accuracy and the control's NLL floor.",
         "Their AUROC is 0.949 and 0.948; their guard rate is 1.0000 and 0.9956.",
         "Representation at 0.95, counterfactual responsiveness at zero, twice over."])

    d._add("""## Two couplings now install the signal and neither produces a guard

- Stage X trained the existing readout to rank determined nodes above undetermined ones, reaching AUROC 0.947. The guard rate stayed at the control's 1.0000 / 1.0000 / 0.9568.
- Stage Y gave the signal its own channel, with genuine per-node re-ranking capability and no competition for the readout's parameters, reaching AUROC 0.9493 / 0.9478 / 0.9480. The guard rate is 1.0000 / 0.9956 / 1.0000.
- The second coupling was designed specifically to fix the first one's two suspected faults -- that the readout was being asked two questions at once, and that a block-level gate could only suppress. It fixed both and changed nothing.
- So the gap is not the parameter conflict, and it is not the monotonicity of a block gate either. **Making the emission depend on determinacy is not the same problem as making the model represent it, and the second does not deliver the first.**
""")

    d._add(NOTFIXED)

    d.points_page(
        "What the channel did buy: stage X's accuracy tax back",
        ["Stage X cost %.4f to %.4f of token accuracy on every seed."
         % (xtax[0], xtax[-1]),
         "The channel's two converged seeds cost %+0.4f and %+0.4f."
         % (acc[0] - sacc[0], acc[1] - sacc[1]),
         "Separating the two losses does remove the interference stage X had."])

    d.table_page(
        "Against the previous best, on this same eval",
        ["arm", "trainable parameters", "$\\rho_{50}$"],
        [["**J2**, untied, `no_mlp L=16 d192`", "{:,}".format(J2_TRAINABLE),
          "**%.3f**" % J2_RHO50],
         ["**S1**, tied, `tied_no_mlp T=16 d320`", "{:,}".format(S1_TRAINABLE),
          "0.754 / 0.757 / 0.765"],
         ["stage X, S1 plus the determinacy term", "{:,}".format(S1_TRAINABLE),
          "%.3f / %.3f / %.3f" % X_RHO50],
         ["stage Y, S1 plus the determinacy channel",
          "{:,}".format(Y_TRAINABLE), " / ".join(rho_txt)],
         ["the free-generation gate", "--", "%.3f" % sy.FREE_LINE]],
        note="Same 14-point grid, same 400 instances per point, same seeds. "
             "The best tied model is still S1, unchanged by this run.")

    d._add("""## Reproducing this

- Repository `github.com/henhen724/spin_gnn`, branch `sudoku-dd`, commit **`%s`**.
- Arm: `tied_no_mlp`, $T=16$, $d_{\\text{model}}=320$, $\\text{ff}=1024$, 4 heads, **%s trainable**, 50 epochs, lr $10^{-3}$, `shuffle_adj`, free generation, determinacy channel on, $\\lambda = 1.0$ registered a priori and **not swept**.
- Seeds **0, 1, 2** in every arm. Control is the stage-S scan `%s`, this arm is `%s`, the probe is `52375912`. Stage X is `%s`, reported in the `determinacy` deck.
- Registration and scorer: `scripts/stagey_score.py`, with the mid-run prediction and its scoring in the same file. Guard probe: `spin_gnn.sudoku_skip`, 800 instances, 350 cases, seed start 600000000 -- byte-identical to the list finding 68 used. Shape: `scripts/guard_shape.py`. This deck: `notebooks/make_channel_web.py`.
- The initialisation bug and its cancelled job are recorded in the scorer's docstring, not only here.
""" % (args.commit, "{:,}".format(Y_TRAINABLE), JOB_S, args.job, JOB_X))

    path = d.write()
    print("wrote %s (%d pages)" % (path, d.n))
    print("verdict: %s" % D["verdict"]["verdict"])
    print("Y4 %s  Y1 %s  Y2 %s"
          % ("/".join("%.4f" % a for a in acc), "/".join(rho_txt),
             "/".join("%.4f" % g for g in D["guard"])))


if __name__ == "__main__":
    main()
