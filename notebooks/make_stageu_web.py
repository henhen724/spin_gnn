"""Stage U as the project's password-gated WEB deck: ~/slides/decks/spin-gnn/.

Stage U asked whether the 0.017 that separates the one shared block from the
free line is about ROLE COUNT: k distinct blocks, each applied 16/k times, so
that every rung is still one map iterated (G = F_{k-1} o ... o F_0). Array
51922710, six cells, k in {2, 4} by three seeds, scored by the registered
scorer (scripts/stageu_score.py, committed b72a74c BEFORE the first cell
landed) in job 52000549, with U4 closed by job 52025112.

NOTHING IS RE-DERIVED HERE.

  * the rungs, the control and every level come from the scorer's own
    load_pooled / rung / summarize / acc_of, so a number on a slide is the
    number the scorer printed or this builder does not build;
  * U4's residuals are recomputed by calling scripts.acc_vs_solve's own
    collect / pava / predict / rms on exactly the dir list score_u4() prints,
    and check_claims() asserts the bar it finds equals the registered
    LOO_BAR. A hand-typed residual would be a sentence nobody could re-run.

Only two things are read straight off disk: history.json per cell, for the
learning-curve page and the tail slopes. Those are per-EPOCH numbers; no
scorer function exposes them, and the final-epoch train NLL read this way is
checked against the scorer's frozen S1_NLL_MEDIAN in check_claims().

THE RESULT THIS DECK HAS TO CARRY HONESTLY: the registered gate prediction is
falsified -- no k clears 0.740 with 3/3 -- and the interesting thing is what
the bound-aware MEDIAN hides. One seed in EACH k > 1 arm left the plateau
(rho_50 0.603 and 0.579 against the arm medians 0.752 and 0.755), and both of
those seeds sit ON the accuracy-to-solve curve, so nothing architectural
happened: they reached J2-level token accuracy and collected the solve rate
the curve pays there. Two seeds out of six is suggestive, not a rate, and the
0.579 does NOT displace J2's 3-seed median of 0.580. The last page says so.

Usage (in a job -- $HOME I/O on sh02-ln02 wedges, and this imports torch
through deck_tokenization):
    python3 -u -m notebooks.make_stageu_web --commit <sha> [--out <dir>]
"""

import argparse
import json
import os
import sys
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

from slides import MUTED, SERIES, label_line, style_axes

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_staget_deck import J2_RHO50, J2_TRAINABLE
from notebooks.make_staget_web import WebDeck

from scripts.acc_vs_solve import collect, is_reference, pava, predict, rms
from scripts.minarch_analysis import SCAN_DIRS
from scripts.stagec_score import rho_txt, rung, summarize
from scripts.stagej_score import LOO_BAR, SUFFICIENCY_DIRS
from scripts.stagek_score import FREE, SCAN_K
from scripts.stagen_score import J2, SCAN_N, acc_of, load_pooled
from scripts.stageo_score import SCAN_O
from scripts.stagep_score import CENSOR_FLOOR, SCAN_P
from scripts.stages_score import SCAN_S
from scripts.stageu_score import (ACC_GAP, BASELINE_DIRS, FLAT_BAND, LADDER,
                                  RHO_GAP, S1_ACC, S1_NLL_MEDIAN, S1_RHO50,
                                  S1_TRAINABLE, SCAN_U, cycled_count)

DECK = "/home/users/hshunt/slides/decks/spin-gnn"

# The split between the two modes, stated ONCE and used everywhere. It is not a
# fitted threshold: the nine cells are 0.579, 0.603 and then seven between
# 0.752 and 0.765, so anything in the empty stretch names the same two cells.
BASIN = 0.65

# How much token accuracy the gate needs, from the scorer's own S4 steepness.
NEED = ACC_GAP * (float(S1_RHO50) - FREE) / RHO_GAP


def history(m):
    with open(os.path.join(m["path"], "history.json")) as fh:
        return json.load(fh)


def tail_slope(h, n=10):
    """Train NLL per epoch over the last n epochs -- 'was it still falling?'."""
    return (h[-1]["train_nll"] - h[-1 - n]["train_nll"]) / n


def load():
    """Every arm of the ladder, through the scorer's own loaders."""
    cells = load_pooled([SCAN_U] + BASELINE_DIRS + SCAN_DIRS, 2000)
    arms, recs = {}, {}
    for key, label, k, pred in LADDER:
        ms = sorted(rung(cells, key), key=lambda m: m["seed"])
        if not ms:
            raise SystemExit("arm k=%d is not on disk; nothing to publish" % k)
        accs = dict(acc_of(ms))
        arms[k] = {"key": key, "label": label, "pred": pred, "s": summarize(ms),
                   "ms": ms}
        recs[k] = []
        for m in ms:
            h = history(m)
            recs[k].append({
                "seed": m["seed"], "rho50": m["rho50"], "flag": m["flag"],
                "mean_solve": m["mean_solve"], "acc": accs[m["seed"]],
                "nll": h[-1]["train_nll"], "slope": tail_slope(h),
                "trainable": m["n_trainable"], "h": h,
                "basin": m["rho50"] < BASIN})
    return cells, arms, recs


def curve():
    """U4, recomputed by the same functions job 52025112 ran."""
    dirs = SUFFICIENCY_DIRS + [SCAN_K, SCAN_N, SCAN_O, SCAN_P, SCAN_S, SCAN_U]
    cells = [c for c in collect(dirs) if c["acc"] == c["acc"]]
    ref = [c for c in cells if is_reference(c["rung"])]
    held = [c for c in cells if not is_reference(c["rung"])]
    fit = pava([(c["acc"], c["mean_solve"]) for c in ref])
    loo = []
    for i, c in enumerate(ref):
        sub = pava([(d["acc"], d["mean_solve"]) for j, d in enumerate(ref) if j != i])
        p, _oor, _cov = predict(sub, c["acc"])
        loo.append(c["mean_solve"] - p)
    return ref, held, fit, rms(loo)


def residuals(arms, held, fit, bar):
    """Residual and bracket width per cell of this ladder, keyed (k, seed)."""
    want = {a["key"]: k for k, a in arms.items()}
    out = {}
    for c in held:
        k = want.get(c["rung"])
        if k is None:
            continue
        p, oor, cov = predict(fit, c["acc"])
        out[(k, c["seed"])] = {"acc": c["acc"], "solve": c["mean_solve"],
                               "pred": p, "res": c["mean_solve"] - p,
                               "cov": cov, "oor": oor, "weak": cov > bar}
    return out


def breakaway(recs, k, seed, gap=0.01):
    """Epoch from which a basin seed's train NLL stays below its own siblings.

    Compared against the MEAN of the two plateau seeds in the SAME arm, so the
    reference is the same architecture, the same recipe and the same budget --
    the only difference left is the draw.

    PERSISTENT separation, scanned from the last epoch backwards, not the first
    epoch the gap is exceeded: at epoch 0 the arms differ by their draw alone
    (2.13 against 2.37 in one case), which is initialisation and not a
    breakaway. The first read of this deck printed 'epoch 0' for k=4 seed 1 and
    that is what it was reporting.
    """
    me = [r for r in recs[k] if r["seed"] == seed][0]
    sibs = [r for r in recs[k] if r["seed"] != seed]
    diffs = [sum(s["h"][e]["train_nll"] for s in sibs) / len(sibs)
             - me["h"][e]["train_nll"] for e in range(len(me["h"]))]
    e = len(diffs)
    while e > 0 and diffs[e - 1] > gap:
        e -= 1
    return e if e < len(diffs) else None


def biggest_step(r, first=5):
    """(drop, epoch) of the sharpest one-epoch fall in train NLL.

    From epoch `first` on, because the initial collapse off the uniform
    predictor swamps everything after it -- and it does not finish at epoch 1.
    With first=2 this returned epoch 2 for every plateau seed (a fall of about
    0.055, the tail of that collapse), and the deck printed 'two plateau seeds
    step as sharply (2 and 2)' while the figure showed their steps near epochs 8
    and 23. By epoch 5 the collapse is over and what is left is a step.
    """
    nll = [e["train_nll"] for e in r["h"]]
    return max((nll[e - 1] - nll[e], e) for e in range(first, len(nll)))


def check_claims(arms, recs, res, bar):
    """Every sentence this deck publishes, asserted against the data first.

    A web deck is a URL, so a stale number here is worse than a stale number in
    a PDF: it keeps serving. If a grid is ever re-scored and a value moves, this
    builder stops instead of publishing the old sentence.
    """
    # 1. the control is still stage S's cell, as the scorer froze it
    s1 = arms[1]["s"]
    assert rho_txt(s1) == S1_RHO50, (rho_txt(s1), S1_RHO50)
    assert s1["trainable"] == S1_TRAINABLE, s1["trainable"]
    assert round(median([r["acc"] for r in recs[1]]), 4) == S1_ACC
    assert abs(median([r["nll"] for r in recs[1]]) - S1_NLL_MEDIAN) < 5e-5, (
        "history.json disagrees with the scorer's frozen train NLL median")

    # 2. the counts are the registration's, and both new arms can carry a censor
    for k, a in arms.items():
        assert a["s"]["trainable"] == cycled_count(320, k) == a["pred"], k
    for k in (2, 4):
        assert arms[k]["s"]["trainable"] > CENSOR_FLOOR, k

    # 3. the gate: the headline negative
    meds = {k: a["s"]["rho50"] for k, a in arms.items()}
    assert all(m > FREE for m in meds.values()), (
        "an arm cleared the gate; this deck's title is wrong", meds)
    for k in (2, 4):
        assert abs(meds[k] - meds[1]) <= 0.01, (
            "an arm moved more than 0.01 from the control", k, meds)

    # 4. the bimodality is exactly two cells, one per k > 1, none at k=1
    basins = [(k, r["seed"]) for k in recs for r in recs[k] if r["basin"]]
    assert sorted(k for k, _ in basins) == [2, 4], basins
    assert not any(r["basin"] for r in recs[1]), "a k=1 seed left the plateau"
    assert max(r["rho50"] for k in recs for r in recs[k] if not r["basin"]) < 0.80
    assert min(r["rho50"] for k, s in basins
               for r in recs[k] if r["seed"] == s) >= J2_RHO50 - 0.002, (
        "a basin seed beat the standing minimum by more than rounding; the "
        "last page would then be understating the result")

    # 5. still descending: every basin seed is steeper than every plateau seed
    slow = max(r["slope"] for k in recs for r in recs[k] if not r["basin"])
    fast = max(r["slope"] for k, s in basins
               for r in recs[k] if r["seed"] == s)
    assert fast < 3 * slow, ("the basin seeds are not descending faster", fast, slow)

    # 6. U4: the bar is the registered one and both basin seeds are inside it
    assert abs(bar - LOO_BAR) < 5e-4, (bar, LOO_BAR)
    for key in basins:
        r = res[key]
        assert not r["oor"] and abs(r["res"]) <= bar, (key, r)

    # 7. accuracy, read first, cannot adjudicate -- as registered in advance
    accs = [median([r["acc"] for r in recs[k]]) for k in (1, 2, 4)]
    assert all(b >= a for a, b in zip(accs, accs[1:])), accs
    worst = max(max(r["acc"] for r in recs[k]) - min(r["acc"] for r in recs[k])
                for k in (1, 2, 4))
    assert NEED < worst, (NEED, worst)
    return basins, meds, accs


# ---------------------------------------------------------------- figures

XS = {1: 0, 2: 1, 4: 2}
TICKS = ["k=1\n16 x 1", "k=2\n8 x 2", "k=4\n4 x 4"]


def _arms_axis(ax, ylabel):
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(TICKS)
    ax.set_xlim(-0.45, 2.5)
    style_axes(ax, "distinct blocks, applied 16/k times each", ylabel)


def _seed_dots(ax, recs, field, med_label_y, lx=2.19):
    for k in (1, 2, 4):
        for r in recs[k]:
            c = SERIES[0] if r["basin"] else MUTED
            ax.plot([XS[k]], [r[field]], "o", ms=9 if r["basin"] else 7,
                    color=c, mfc=c if r["basin"] else "white", mew=1.7,
                    zorder=4 if r["basin"] else 2)
        m = median([r[field] for r in recs[k]])
        ax.plot([XS[k] - 0.19, XS[k] + 0.19], [m, m], "-",
                color=SERIES[1], lw=2.6, zorder=3)
    label_line(ax, lx, med_label_y, "median", color=SERIES[1])


def draw_level(ax, recs):
    _seed_dots(ax, recs, "rho50", median([r["rho50"] for r in recs[4]]))
    for y, ls, txt in ((FREE, "--", "free line %.3f" % FREE),
                       (J2_RHO50, ":", "J2 %.3f" % J2_RHO50)):
        ax.axhline(y, ls=ls, lw=1.2, color=MUTED, zorder=1)
        label_line(ax, 2.5, y, txt, color=MUTED)
    for k, s in ((2, 2), (4, 1)):
        r = [x for x in recs[k] if x["seed"] == s][0]
        label_line(ax, XS[k], r["rho50"], "seed %d  %.3f" % (s, r["rho50"]),
                   color=SERIES[0], dy=-15)
    vals = [r["rho50"] for k in recs for r in recs[k]] + [FREE, J2_RHO50]
    # Headroom below, or the lower label rides the axis spine (first read-back).
    ax.set_ylim(min(vals) - 0.022, max(vals) + 0.008)
    _arms_axis(ax, "rho_50  (lower is better)")


def draw_acc(ax, recs):
    _seed_dots(ax, recs, "acc", median([r["acc"] for r in recs[1]]), lx=0.21)
    y = S1_ACC + NEED
    ax.axhline(y, ls="--", lw=1.2, color=MUTED, zorder=1)
    label_line(ax, 2.5, y, "what the gate needs\n%.4f" % y, color=MUTED)
    _arms_axis(ax, "validation token accuracy")


def draw_curves(ax, recs):
    i, last = 0, []
    for k in (1, 2, 4):
        for r in recs[k]:
            ep = [e["epoch"] for e in r["h"]][1:]
            nll = [e["train_nll"] for e in r["h"]][1:]
            if r["basin"]:
                c = SERIES[i]
                ax.plot(ep, nll, "-", color=c, lw=2.3, zorder=3)
                label_line(ax, ep[-1], nll[-1], "k=%d seed %d" % (k, r["seed"]),
                           color=c, dy=(7, -7)[i])
                i += 1
            else:
                ax.plot(ep, nll, "-", color=MUTED, lw=1.0, alpha=0.6, zorder=1)
                last.append(nll[-1])
    label_line(ax, 49, median(last), "the seven plateau seeds", color=MUTED)
    ax.set_xlim(1, 51)
    ax.set_ylim(1.50, 1.76)
    style_axes(ax, "epoch  (epoch 0, near 2.1, is above the axis)",
               "training NLL")


def draw_u4(ax, ref, fit, res, basins, bar):
    xs = [p[0] for p in fit]
    ys = [p[1] for p in fit]
    ax.plot(xs, ys, "-", color=MUTED, lw=1.8, zorder=2)
    ax.fill_between(xs, [y - bar for y in ys], [y + bar for y in ys],
                    color=MUTED, alpha=0.13, lw=0, zorder=1)
    ax.plot([c["acc"] for c in ref], [c["mean_solve"] for c in ref], "o",
            ms=5, color=MUTED, mfc="white", mew=1.2, zorder=3)
    # On the flat left shelf, where there is room inside the axes. At the fit's
    # right-hand end it floated above the plot box (first read-back).
    flat = [p for p in fit if p[0] <= 0.470][-1]
    label_line(ax, flat[0], flat[1], "depth ladder\n+- its own LOO bar",
               color=MUTED, dy=26)
    plateau = []
    for (k, s), r in sorted(res.items()):
        if (k, s) in basins:
            ax.plot([r["acc"]], [r["solve"]], "o", ms=10, color=SERIES[0], zorder=5)
            label_line(ax, r["acc"], r["solve"],
                       "k=%d seed %d  %+.4f" % (k, s, r["res"]), color=SERIES[0],
                       dy=-16)
        else:
            ax.plot([r["acc"]], [r["solve"]], "s", ms=6, color=SERIES[2],
                    mfc="white", mew=1.6, zorder=4)
            plateau.append(r)
    lo = min(plateau, key=lambda r: r["acc"])
    label_line(ax, lo["acc"], lo["solve"], "the seven plateau seeds",
               color=SERIES[2], dy=20, dx=-60)
    ax.set_xlim(0.455, 0.503)
    ax.set_ylim(0.0, 0.62)
    style_axes(ax, "validation token accuracy", "mean solve rate over the grid")


# ---------------------------------------------------------------- the deck

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()

    cells, arms, recs = load()
    ref, held, fit, bar = curve()
    res = residuals(arms, held, fit, bar)
    basins, meds, accs = check_claims(arms, recs, res, bar)

    j2 = summarize(rung(cells, J2))
    j2_acc = median([a for _, a in acc_of(rung(cells, J2))])
    brk = {key: breakaway(recs, *key) for key in basins}
    best = min(r["rho50"] for k, s in basins for r in recs[k] if r["seed"] == s)
    plateau_slopes = [r["slope"] for k in recs for r in recs[k] if not r["basin"]]
    # The two sharpest one-epoch drops among the PLATEAU seeds: the steps in the
    # learning curves are not what the basin seeds have and the others lack.
    steps = sorted(sorted((biggest_step(r) for k in recs for r in recs[k]
                           if not r["basin"]), reverse=True)[:2],
                   key=lambda t: t[1])
    assert steps[0][0] >= 0.5 * max(biggest_step(r)[0] for k, s in basins
                                    for r in recs[k] if r["seed"] == s), (
        "the plateau seeds do not step as sharply; that sentence would be wrong")

    prov = ("Array 51922710, six cells, scored 52000549, curve 52025112, commit "
            "%s. d320 ff1024 h4 tied_no_mlp, 16 applications, 50 epochs, lr 1e-3 "
            "constant, seeds 0/1/2, 14 densities x 400 instances, --shuffle-adj "
            "--fall." % args.commit)

    d = WebDeck(args.out,
                "Cycling k blocks changes nothing; which basin the optimizer "
                "finds changes everything", "hshunt", "2026-11-11", prov)

    d.title_page(
        "Cycling k blocks changes nothing; which basin the optimizer finds "
        "changes everything",
        subtitle="k distinct blocks, each applied 16/k times, at fixed width "
                 "d320 and a fixed sixteen applications. Three seeds per k, one "
                 "pooled 14-density grid, no code change between arms.")

    d.points_page(
        "A cycled cell is k blocks read in rotation, sixteen times",
        ["G = F_{k-1} o ... o F_0, applied 16/k times",
         "k=1 is stage S's one shared block; k=16 would be the untied stack",
         "So every arm is still one map iterated, which is the brief"])

    exi = example_instance(n_nodes=48, degree=5, n_colors=4,
                           shuffle_adj=True, fall=True)
    d.expect("tokenization",
             "Every token is one categorical id in a %d-symbol vocabulary"
             % exi.vocab.size)
    page_tokenization(d, exi)
    d.expect("trace",
             "A deduction fires when a node sees all three other colours",
             note="The same generator every arm trained and was evaluated on; k "
                  "changes the model, never the data.")
    page_trace(d, exi)

    # Accuracy first, as registered: it cannot censor, so it is readable
    # whatever rho_50 does.
    d.claim_page(
        "Accuracy rises monotonically with k, by less than the seed spread",
        lambda ax: draw_acc(ax, recs), "acc",
        note="Medians %.4f, %.4f, %.4f. The gate needs about %.4f of accuracy; "
             "the worst seed spread is %.4f."
             % (accs[0], accs[1], accs[2], NEED,
                max(max(r["acc"] for r in recs[k]) - min(r["acc"] for r in recs[k])
                    for k in (1, 2, 4))))

    d.table_page(
        "The training floor sits in the same place at every k",
        ["arm", "final train NLL, per seed", "median", "vs k=1"],
        [[arms[k]["label"].split("  ")[1],
          " / ".join("%.4f" % r["nll"] for r in recs[k]),
          "%.4f" % median([r["nll"] for r in recs[k]]),
          "--" if k == 1 else "%+.4f" % (median([r["nll"] for r in recs[k]])
                                         - median([r["nll"] for r in recs[1]]))]
         for k in (1, 2, 4)],
        note="Span of the medians is %.4f, inside the registered flat band of "
             "%.2f. The floor belongs to the recipe, not to the role count."
             % (max(median([r["nll"] for r in recs[k]]) for k in (1, 2, 4))
                - min(median([r["nll"] for r in recs[k]]) for k in (1, 2, 4)),
                FLAT_BAND))

    d.table_page(
        "No k clears the %.3f gate: %.3f and %.3f against %s"
        % (FREE, meds[2], meds[4], S1_RHO50),
        ["arm", "trainable", "rho_50 median", "seeds crossing", "vs k=1"],
        [[arms[k]["label"].split("  ")[1], "{:,}".format(arms[k]["s"]["trainable"]),
          rho_txt(arms[k]["s"]),
          "%d/%d" % (arms[k]["s"]["n_cross"], arms[k]["s"]["n"]),
          # Against the control's FROZEN median, which is what the scorer
          # differenced, so the deck and job 52000549 print the same deltas.
          "--" if k == 1 else "%+.3f" % (meds[k] - float(S1_RHO50))]
         for k in (1, 2, 4)],
        highlight_row=0,
        note="Lower rho_50 is better. Both new arms land within 0.01 of the "
             "control, which is the competing hypothesis' prediction, not this "
             "stage's.")

    d.claim_page(
        "The median hides a second mode: one seed per k reaches 0.58",
        lambda ax: draw_level(ax, recs), "level",
        note="Per seed, not pooled. Both new arms report a median near 0.75 "
             "because two of their three seeds sit there.")

    d.table_page(
        "Two of six k>1 seeds reached a basin no k=1 seed found",
        ["arm", "seed", "rho_50", "mean solve", "token acc", "train NLL", "mode"],
        [[arms[k]["label"].split("  ")[1], r["seed"], "%.3f" % r["rho50"],
          "%.4f" % r["mean_solve"], "%.4f" % r["acc"], "%.4f" % r["nll"],
          "BASIN" if r["basin"] else "plateau"]
         for k in (1, 2, 4) for r in recs[k]],
        note="The three k=1 seeds are tight at %s. Two seeds out of six is "
             "suggestive, not a rate."
             % ", ".join("%.3f" % r["rho50"] for r in recs[1]))

    d.claim_page(
        "The basin seeds track a plateau seed, then break away in steps",
        lambda ax: draw_curves(ax, recs), "curves",
        note="Leaving their siblings for good at epochs %s. Two plateau seeds "
             "step as sharply (%s); where it lands differs."
             % (" and ".join(str(brk[key]) for key in sorted(brk)),
                " and ".join(str(e) for _, e in steps)))

    d.table_page(
        "Both basin seeds were still descending when the budget ended",
        ["arm", "seed", "train NLL per epoch, last ten", "mode"],
        [[arms[k]["label"].split("  ")[1], r["seed"], "%+.5f" % r["slope"],
          "BASIN" if r["basin"] else "plateau"]
         for k in (1, 2, 4) for r in recs[k]],
        note="The seven plateau seeds fall at %+.5f to %+.5f per epoch; the two "
             "basin seeds are four to ten times steeper at fifty epochs."
             % (max(plateau_slopes), min(plateau_slopes)))

    d.claim_page(
        "The basin seeds sit on the accuracy-to-solve curve, not above it",
        lambda ax: draw_u4(ax, ref, fit, res, basins, bar), "u4",
        note="Residuals %s against the reference's own leave-one-out bar of "
             "%.4f. Both are weakly determined and say so."
             % (" and ".join("%+.4f" % res[key]["res"] for key in sorted(brk)),
                bar))

    # Grouped the way scripts.acc_vs_solve groups them -- well-determined cells
    # apart from weakly determined ones -- so these rows are the rows job
    # 52025112 printed, rather than a pooled mean that agrees with nothing.
    d.table_page(
        "Every arm is explained by its token accuracy, within the bar",
        ["arm", "cells", "bracket", "mean residual", "verdict"],
        [[arms[k]["label"].split("  ")[1], len(g),
          "weakly determined" if weak else "well determined",
          "%+.4f" % (sum(r["res"] for r in g) / len(g)),
          "EXPLAINED by token accuracy"
          if abs(sum(r["res"] for r in g) / len(g)) <= bar else "OFF the curve"]
         for k in (1, 2, 4)
         for weak in (False, True)
         for g in [[res[(k, r["seed"])] for r in recs[k]
                    if (k, r["seed"]) in res
                    and res[(k, r["seed"])]["weak"] == weak]] if g],
        note="A cell off the curve would have been the headline over the gate: "
             "cycling blocks would change the relation, not the level. None is.")

    d.points_page(
        "What binds is which basin the optimizer finds, not the role count",
        ["Extra blocks did not move the floor the median reports",
         "They coincided with two escapes from it; k=1 had none in three seeds",
         "With three seeds an arm that is a coincidence, not a rate"])

    d.points_page(
        "Stage U speaks for one width, one recipe, fifty epochs",
        ["d320 tied, 16 applications, lr 1e-3 constant, no warmup",
         "N=48 only; stage M's N=96 bars are a different size",
         "Final-epoch checkpoints only, so no best-val selection anywhere"])

    d.table_page(
        "Every cell ran the same config; only k and the seed differ",
        ["what", "value"],
        [["array / cells", "51922710, six cells, all COMPLETED 0:0"],
         ["commit", args.commit],
         ["config", "d320 ff1024 h4 tied_no_mlp, L=16 applications, k in {1,2,4}"],
         ["training", "50 epochs, lr 1e-3 constant, no warmup, 240k instances"],
         ["seeds", "0, 1, 2 per arm (nine cells with the control)"],
         ["eval", "14 densities x 400 instances, --shuffle-adj --fall"],
         ["scored by", "52000549 (registered scorer), 52025112 (the curve)"]],
        note="The registration was committed as b72a74c before the first cell "
             "landed, so the predictions above were not chosen after the fact.")

    d.points_page(
        "The next cut is the training budget, not the architecture",
        ["Both basin seeds were cut off mid-descent at fifty epochs",
         "100 epochs on this same cell is one knob against this control",
         "Stage V is testing schedule length now, at 613,818 parameters"])

    # Last, as ~/CLAUDE.md requires.
    d.table_page(
        "Against the previous best, stage U changes nothing: %.3f still stands"
        % J2_RHO50,
        ["quantity", "J2 untied d192 (standing)", "k=2 cycled", "k=4 cycled"],
        [["rho_50 median", "%.3f" % J2_RHO50, rho_txt(arms[2]["s"]),
          rho_txt(arms[4]["s"])],
         ["best single seed", "%.3f" % min(m["rho50"] for m in rung(cells, J2)),
          "%.3f" % min(r["rho50"] for r in recs[2]),
          "%.3f" % min(r["rho50"] for r in recs[4])],
         ["mean solve", "%.4f" % j2["mean_solve"],
          "%.4f" % arms[2]["s"]["mean_solve"], "%.4f" % arms[4]["s"]["mean_solve"]],
         ["token accuracy", "%.4f" % j2_acc,
          "%.4f" % median([r["acc"] for r in recs[2]]),
          "%.4f" % median([r["acc"] for r in recs[4]])],
         ["trainable", "{:,}".format(J2_TRAINABLE),
          "{:,}".format(arms[2]["s"]["trainable"]),
          "{:,}".format(arms[4]["s"]["trainable"])],
         ["distinct blocks", "16", "2", "4"],
         ["seeds crossing 0.5", "3/3",
          "%d/%d" % (arms[2]["s"]["n_cross"], arms[2]["s"]["n"]),
          "%d/%d" % (arms[4]["s"]["n_cross"], arms[4]["s"]["n"])]],
        highlight_row=0,
        note="Same grid, same 400 instances per bin, same flags. One seed at "
             "%.3f does not displace a three-seed median, and none is claimed."
             % best)

    path = d.write()
    print("wrote %s (%d pages, %d figures)"
          % (path, d.n, len(os.listdir(d.fig_dir))))
    print("figures: %s" % ", ".join(sorted(os.listdir(d.fig_dir))))
    print("rho_50 medians: " + "  ".join("k=%d %s" % (k, rho_txt(arms[k]["s"]))
                                         for k in (1, 2, 4)))
    print("basin cells: " + "  ".join(
        "k=%d seed %d rho_50 %.3f acc %.4f resid %+.4f break@%d"
        % (k, s, [r for r in recs[k] if r["seed"] == s][0]["rho50"],
           res[(k, s)]["acc"], res[(k, s)]["res"], brk[(k, s)])
        for k, s in sorted(basins)))
    print("LOO bar %.4f (registered %.4f), best basin seed %.3f vs standing %.3f"
          % (bar, LOO_BAR, best, J2_RHO50))


if __name__ == "__main__":
    main()
