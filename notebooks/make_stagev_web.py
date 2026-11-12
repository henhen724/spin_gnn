"""Stage V as the project's password-gated WEB deck: ~/slides/decks/spin-gnn/.

Stage V asked whether the 0.017 that separates the one shared block from the
free line is about SERIAL DEPTH: train the block at T = 24 applications instead
of 16, which costs ZERO parameters. Array 51962715 tasks 0-2, registered scorer
scripts/stagev_score.py committed ca11c2b by job 51962711 BEFORE the first cell
landed.

THE RESULT THIS DECK HAS TO CARRY HONESTLY, and the shape of it matters: V-a is
FALSIFIED 0/3, but NOT because the cell got worse. Two seeds land at 0.4833 and
0.4834 against the control's 0.4830 -- unchanged -- and the registered bar needed
+0.0011. The third seed never trained: accuracy 0.4612, train NLL 1.6334, the
easy end collapsed to 0.065, rho_50 censored. So the honest headline is "does not
help, and destabilizes", not "makes it worse": the median rho_50 does move the
wrong way, 0.757 -> 0.769, but one broken seed moved it, and the page titles say
so. A title claiming every seed got worse would be false -- seed 2 is 0.757
against its own control's 0.765 -- and an earlier draft of this builder said
exactly that until the numbers were read back against it.

NOTHING IS RE-DERIVED HERE. The rungs, the control, every level and every
accuracy come from the scorer's own load_pooled / rung / summarize / acc_of; the
per-bin curves are load_all rows filtered by rung_key, so a point on a slide
belongs to the cell the scorer scored; the sufficiency residuals are recomputed
by scripts.acc_vs_solve's own collect / pava / predict / rms; stage T's
frozen-at-24 comparison goes through cell_metrics and summarize and is checked
against stage T's published number before it is drawn. Only history.json is read
straight off disk, for the per-epoch curves no scorer function exposes, and its
final-epoch value is checked against the scorer's frozen literals.

EVERY COMPARATIVE SENTENCE IS DERIVED AND ASSERTED. Across stages S, T and U the
page-by-page read-backs caught thirteen defects, five of them wrong CLAIMS on
decks that rendered clean. A title computed from the data and asserted against
it cannot be one of those.

Usage (in a job -- $HOME I/O on sh02-ln02 wedges, and this imports torch
through deck_tokenization):
    python3 -u -m notebooks.make_stagev_web --commit <sha> [--out <dir>]
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
from scripts.minarch_analysis import SCAN_DIRS, cell_metrics, load_all, rung_key
from scripts.stagec_score import rho_txt, rung, summarize
from scripts.stagej_score import LOO_BAR, SUFFICIENCY_DIRS
from scripts.stagek_score import FREE, SCAN_K
from scripts.stagen_score import J2, SCAN_N, acc_of, load_pooled
from scripts.stageo_score import SCAN_O
from scripts.stagep_score import CENSOR_FLOOR, SCAN_P
from scripts.stages_score import SCAN_S
from scripts.staget_score import EASY_BAR, EASY_RHO, SCAN_T
from scripts.stageu_score import SCAN_U
from scripts.stagev_score import (ACC_BAR, BASELINE_DIRS, NEED, S1, S1_ACC,
                                 S1_ACC_SPREAD, S1_MEAN_SOLVE, S1_NLL_MEDIAN,
                                 S1_RHO50, S1_TRAINABLE, SCAN_V, T24_CROSS,
                                 T24_EASY_WORST, T24_RHO50, V1, sched_count)

DECK = "/home/users/hshunt/slides/decks/spin-gnn"

WORD = {0: "Zero", 1: "One", 2: "Two", 3: "Three"}
FROZEN24 = S1 + ("T24",)            # stage T's 16-trained weights RUN at 24


def plural(n, one, many):
    return one if n == 1 else many


def history(path):
    with open(os.path.join(path, "history.json")) as fh:
        return json.load(fh)


def arm(cells, key):
    """One rung, per seed, through the scorer's own loaders.

    rung() returns None -- not [] -- when nothing has landed, so a missing cell
    has to stop the build rather than sort None into an empty page.
    """
    ms = rung(cells, key)
    if not ms:
        raise SystemExit("rung %r is not on disk; nothing to publish" % (key,))
    ms = sorted(ms, key=lambda m: m["seed"])
    accs = dict(acc_of(ms))
    out = []
    for m in ms:
        h = history(m["path"])
        out.append({"seed": m["seed"], "rho50": m["rho50"], "flag": m["flag"],
                    "ok": m["flag"] == "ok", "mean_solve": m["mean_solve"],
                    "acc": accs[m["seed"]], "nll": h[-1]["train_nll"], "h": h,
                    "trainable": m["n_trainable"]})
    return out, summarize(ms)


def curves(dirs, key):
    """Per-bin solve_rate by seed for ONE cell, filtered by rung_key.

    The scan dirs hold more than one cell -- stage S alone has d192, d320 and
    d448 -- so grouping by seed alone would average three widths into one curve.
    """
    out = {}
    for r in load_all(dirs):
        if rung_key(r) != key:
            continue
        out.setdefault(int(r["seed"]), []).append(r)
    for seed in out:
        out[seed].sort(key=lambda r: r["realized_rho"])
    return out


def frozen24():
    """Stage T's frozen-at-24 re-read, per seed, through cell_metrics."""
    by = curves([SCAN_T], FROZEN24)
    return {s: cell_metrics(rs, 2000) for s, rs in by.items()}


def sufficiency():
    """The N=48 accuracy-to-solve curve, by acc_vs_solve's own functions."""
    dirs = SUFFICIENCY_DIRS + [SCAN_K, SCAN_N, SCAN_O, SCAN_P, SCAN_S, SCAN_U,
                               SCAN_V]
    cells = [c for c in collect(dirs) if c["acc"] == c["acc"]]
    ref = [c for c in cells if is_reference(c["rung"])]
    held = [c for c in cells if not is_reference(c["rung"])]
    fit = pava([(c["acc"], c["mean_solve"]) for c in ref])
    loo = []
    for i, c in enumerate(ref):
        sub = pava([(d["acc"], d["mean_solve"]) for j, d in enumerate(ref) if j != i])
        p, _oor, _cov = predict(sub, c["acc"])
        loo.append(c["mean_solve"] - p)
    bar = rms(loo)
    res = {}
    for c in held:
        if c["rung"] != V1:
            continue
        p, oor, cov = predict(fit, c["acc"])
        res[int(c["seed"])] = {"acc": c["acc"], "solve": c["mean_solve"],
                               "pred": p, "res": c["mean_solve"] - p,
                               "oor": oor, "weak": cov > bar}
    return ref, fit, bar, res


def ok_median(ms):
    """Stage T's own level rule: the median over the CROSSING seeds only.

    staget_score.py:216 reports a T as median([m["rho50"] for m in ok]) when a
    majority crossed, which is NOT summarize()'s bound-aware median: with two
    crossing seeds and one censored, stage T's rule averages the two (0.819) and
    the bound-aware rule returns the worse of them (0.875). Both are honest; they
    are different estimators, and a figure that mixed them across its own three
    points would be the exact defect the read-backs exist to catch. This deck
    draws the bound-aware number everywhere, because that is what stage V's
    registered scorer reports, and reproduces stage T's rule only to CHECK that
    the right cells were loaded.
    """
    ok = [m for m in ms if m["flag"] == "ok"]
    return median([m["rho50"] for m in ok]) if len(ok) * 2 > len(ms) else float("nan")


def worst_easy(by_seed):
    """Worst solve_rate in the easy bins, per seed -- V-c's statistic."""
    out = {}
    for seed, rs in by_seed.items():
        easy = [r for r in rs if r["realized_rho"] >= EASY_RHO]
        if easy:
            out[seed] = min(r["solve_rate"] for r in easy)
    return out


def check_claims(ctrl, cs, v1, vs, froz, fs, bar, res):
    """Every sentence this deck publishes, asserted against the data first.

    A web deck is a URL: a stale number here keeps serving. If a grid is
    re-scored and a value moves, this builder stops instead of republishing the
    old sentence.
    """
    # 1. the control is still stage S's cell, exactly as the scorer froze it
    assert rho_txt(cs) == S1_RHO50, (rho_txt(cs), S1_RHO50)
    assert cs["trainable"] == S1_TRAINABLE, cs["trainable"]
    assert round(median([r["acc"] for r in ctrl]), 4) == S1_ACC
    assert abs(median([r["nll"] for r in ctrl]) - S1_NLL_MEDIAN) < 5e-5, (
        "history.json disagrees with the scorer's frozen train NLL median")
    assert abs(max(r["acc"] for r in ctrl) - min(r["acc"] for r in ctrl)
               - S1_ACC_SPREAD) < 5e-5
    assert cs["n_cross"] == 3, cs["n_cross"]

    # 2. THE HEADLINE GUARD: the knob cost nothing, which is the whole case for
    # having tried it before width, roles or data.
    assert vs["trainable"] == cs["trainable"] == sched_count(320) == 613818, (
        "the application count moved the parameter count", vs["trainable"])
    assert vs["trainable"] < CENSOR_FLOOR, "guard 7 does not protect this stage"

    # 3. the negative itself, on both statistics
    assert sum(1 for r in v1 if r["acc"] >= ACC_BAR) == 0, (
        "a seed cleared V-a's bar; this deck's title is wrong",
        [r["acc"] for r in v1])
    assert vs["rho50_readable"], "the pooled T=24 median is a bound, not a number"
    assert vs["rho50"] > float(S1_RHO50), (
        "the longer schedule did not make rho_50 worse", vs["rho50"])
    assert vs["n_cross"] < cs["n_cross"], (vs["n_cross"], cs["n_cross"])
    assert vs["mean_solve"] < S1_MEAN_SOLVE, vs["mean_solve"]

    # 4. the spread, which is the other half of the result
    sv = max(r["acc"] for r in v1) - min(r["acc"] for r in v1)
    assert sv > 10 * S1_ACC_SPREAD, ("the spread did not blow up", sv)

    # 5. V-e's pair must be two different cells, or the comparison is a cell
    # against itself (the registration's own guard 8). The frozen arm is
    # recomputed here and checked against stage T's published number.
    assert FROZEN24 != V1, FROZEN24
    assert fs, "stage T's frozen-at-24 curves are not on disk"
    assert len(froz) == len(v1), (len(froz), len(v1))
    # Loaded the right cells: stage T's OWN rule, applied to these rows,
    # reproduces the number stage T published.
    assert abs(ok_median(list(froz.values())) - T24_RHO50) < 5e-4, (
        "stage T's crossing-seed median over these rows is not its published one",
        ok_median(list(froz.values())), T24_RHO50)
    assert fs["n_cross"] == T24_CROSS, (fs["n_cross"], T24_CROSS)
    assert fs["trainable"] == vs["trainable"], (fs["trainable"], vs["trainable"])
    # And the page's claim holds under BOTH estimators, so the sentence does not
    # depend on which one the reader has in mind.
    assert vs["rho50"] < min(fs["rho50"], T24_RHO50), (
        "trained-at-24 did not beat frozen-at-24, which V-e predicted it would",
        vs["rho50"], fs["rho50"], T24_RHO50)

    # 6. the sufficiency bar is the registered one
    assert abs(bar - LOO_BAR) < 5e-4, (bar, LOO_BAR)
    assert res, "no stage V cell reached the accuracy-to-solve curve"
    return sv


# ---------------------------------------------------------------- figures

XS = {16: 0, 24: 1}
TICKS = ["T=16\ntrained at 16", "T=24\ntrained at 24"]


def _arms_axis(ax, ylabel):
    ax.set_xticks([0, 1])
    ax.set_xticklabels(TICKS)
    ax.set_xlim(-0.45, 1.6)
    style_axes(ax, "applications of the one shared block", ylabel)


def _seed_dots(ax, arms, field, meds, only_ok=False):
    """Seeds as dots, with the median bar the PAGE TITLE quotes.

    `meds` is passed in rather than computed here. The first render drew the
    median of the crossing seeds (0.763) under a title that said 0.769, because
    rho_50's median is bound-aware and includes the censored seed's bound while
    a median over the plotted dots cannot. A figure must draw the number its own
    title names.
    """
    for t, rows in arms.items():
        for r in rows:
            ok = r["ok"] or not only_ok
            ax.plot([XS[t]], [r[field]], "o" if ok else "^", ms=8,
                    color=SERIES[0], mfc=SERIES[0] if ok else "white", mew=1.7,
                    zorder=4)
        ax.plot([XS[t] - 0.17, XS[t] + 0.17], [meds[t]] * 2, "-",
                color=SERIES[1], lw=2.6, zorder=3)
    label_line(ax, 1.33, meds[24], "median", color=SERIES[1])


def draw_acc(ax, arms, meds):
    _seed_dots(ax, arms, "acc", meds)
    ax.axhline(ACC_BAR, ls="--", lw=1.2, color=MUTED, zorder=1)
    label_line(ax, 1.6, ACC_BAR, "gate needs\n%.4f" % ACC_BAR, color=MUTED)
    # Explicit, because the one broken seed is far below the other five and
    # matplotlib's autoscale put it exactly on the bottom spine.
    vals = [r["acc"] for rows in arms.values() for r in rows] + [ACC_BAR]
    pad = 0.12 * (max(vals) - min(vals))
    ax.set_ylim(min(vals) - pad, max(vals) + pad)
    _arms_axis(ax, "validation token accuracy")


def draw_level(ax, arms, meds):
    _seed_dots(ax, arms, "rho50", meds, only_ok=True)
    for y, ls, txt in ((FREE, "--", "free line %.3f" % FREE),
                       (J2_RHO50, ":", "J2 %.3f" % J2_RHO50)):
        ax.axhline(y, ls=ls, lw=1.2, color=MUTED, zorder=1)
        label_line(ax, 1.6, y, txt, color=MUTED)
    cens = [r for r in arms[24] if not r["ok"]]
    if cens:
        label_line(ax, XS[24], cens[0]["rho50"], "never crossed 0.5",
                   color=SERIES[0], dy=-16)
    vals = [r["rho50"] for rows in arms.values() for r in rows] + [FREE, J2_RHO50]
    ax.set_ylim(min(vals) - 0.02, max(vals) + 0.03)
    _arms_axis(ax, "rho_50  (lower is better)")


def draw_curves(ax, ctrl, v1, high):
    for r in ctrl:
        ax.plot([e["epoch"] for e in r["h"]][1:],
                [e["train_nll"] for e in r["h"]][1:], "-", color=MUTED, lw=1.0,
                alpha=0.6, zorder=1)
    label_line(ax, 49, median([r["nll"] for r in ctrl]), "the three T=16 seeds",
               color=MUTED, dy=-14)
    for r in v1:
        c = SERIES[1] if r["seed"] == high else SERIES[0]
        ax.plot([e["epoch"] for e in r["h"]][1:],
                [e["train_nll"] for e in r["h"]][1:], "-", color=c, lw=2.1,
                zorder=3)
    hi = [r for r in v1 if r["seed"] == high][0]
    lo = max((r for r in v1 if r["seed"] != high), key=lambda r: r["nll"])
    label_line(ax, 49, hi["nll"], "T=24 seed %d" % high, color=SERIES[1], dy=6)
    label_line(ax, 49, lo["nll"], "the other two", color=SERIES[0], dy=10)
    ax.set_xlim(1, 51)
    ax.set_ylim(1.50, 1.80)
    style_axes(ax, "epoch  (epoch 0, near 2.1, is above the axis)",
               "training NLL")


def _median_curve(by):
    xs = sorted({round(r["realized_rho"], 4) for rs in by.values() for r in rs})
    ys = [median([r["solve_rate"] for rs in by.values() for r in rs
                  if round(r["realized_rho"], 4) == x]) for x in xs]
    return xs, ys


def draw_solve(ax, c16, c24):
    for i, (by, name, dy) in enumerate(((c16, "trained at 16", 12),
                                        (c24, "trained at 24", -4))):
        xs, ys = _median_curve(by)
        ax.plot(xs, ys, "-o", ms=4, color=SERIES[i], lw=2.0, zorder=3 - i)
        label_line(ax, xs[-1], ys[-1], name, color=SERIES[i], dy=dy)
    ax.axhline(0.5, ls="--", lw=1.2, color=MUTED, zorder=1)
    label_line(ax, _median_curve(c16)[0][0], 0.5,
               "the 0.5 crossing that defines rho_50", color=MUTED, dy=8)
    ax.set_ylim(-0.03, 1.05)
    style_axes(ax, "realized hint density", "fraction of puzzles solved")


def draw_pair(ax, cs, fs, vs):
    """V-e: same shape, same 24 applications, trained for it or not."""
    pts = [("16-trained,\nrun at 16", cs["rho50"], SERIES[0]),
           ("16-trained,\nrun at 24", fs["rho50"], MUTED),
           ("24-trained,\nrun at 24", vs["rho50"], SERIES[1])]
    ax.plot(range(len(pts)), [p[1] for p in pts], "-", color=MUTED, lw=1.0,
            alpha=0.5, zorder=1)
    for i, (name, y, c) in enumerate(pts):
        ax.plot([i], [y], "o", ms=11, color=c, zorder=4)
        label_line(ax, i, y, "%.3f" % y, color=c, dy=14)
    ax.axhline(FREE, ls="--", lw=1.2, color=MUTED, zorder=1)
    # At the xlim edge, not inside it: at x=2.1 the dashed line ran through the
    # text and read as a strikethrough.
    label_line(ax, 2.45, FREE, "free line %.3f" % FREE, color=MUTED)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([p[0] for p in pts])
    ax.set_xlim(-0.4, 2.45)
    ax.set_ylim(min(p[1] for p in pts) - 0.03, max(p[1] for p in pts) + 0.03)
    style_axes(ax, "", "rho_50  (lower is better)")


def draw_fit(ax, ref, fit, bar, res):
    xs = [p[0] for p in fit]
    ys = [p[1] for p in fit]
    ax.plot(xs, ys, "-", color=MUTED, lw=1.8, zorder=2)
    ax.fill_between(xs, [y - bar for y in ys], [y + bar for y in ys], color=MUTED,
                    alpha=0.13, lw=0, zorder=1)
    ax.plot([c["acc"] for c in ref], [c["mean_solve"] for c in ref], "o", ms=5,
            color=MUTED, mfc="white", mew=1.2, zorder=3)
    low = [p for p in fit if p[0] <= 0.470] or fit[:1]
    label_line(ax, low[-1][0], low[-1][1], "depth ladder\n+- its own LOO bar",
               color=MUTED, dy=26)
    for i, seed in enumerate(sorted(res)):
        r = res[seed]
        ax.plot([r["acc"]], [r["solve"]], "o", ms=10, color=SERIES[0], zorder=5)
        label_line(ax, r["acc"], r["solve"], "seed %d  %+.4f" % (seed, r["res"]),
                   color=SERIES[0], dy=(-2, -20, 36)[i % 3])
    style_axes(ax, "validation token accuracy", "mean solve rate over the grid")


# ---------------------------------------------------------------- the deck

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()

    cells = load_pooled([SCAN_V] + BASELINE_DIRS + SCAN_DIRS, 2000)
    ctrl, cs = arm(cells, S1)
    v1, vs = arm(cells, V1)
    c16, c24 = curves([SCAN_S], S1), curves([SCAN_V], V1)
    froz = frozen24()
    fs = summarize(list(froz.values())) if froz else None
    ref, fit, bar, res = sufficiency()
    spread_v = check_claims(ctrl, cs, v1, vs, froz, fs, bar, res)

    arms = {16: ctrl, 24: v1}
    # The medians the figures draw are the ones the titles and tables quote:
    # accuracy's over all three seeds, rho_50's bound-aware from the scorer.
    acc_meds = {16: median([r["acc"] for r in ctrl]),
                24: median([r["acc"] for r in v1])}
    lvl_meds = {16: cs["rho50"], 24: vs["rho50"]}
    j2ms = rung(cells, J2)
    assert j2ms, "the standing minimum J2 is not on disk; the last page needs it"
    j2 = summarize(j2ms)
    j2_acc = median([a for _, a in acc_of(j2ms)])

    # Derived counts: every number a page title depends on, computed once.
    n_clear = sum(1 for r in v1 if r["acc"] >= ACC_BAR)
    n_stopped = cs["n_cross"] - vs["n_cross"]
    high = max(v1, key=lambda r: r["nll"])["seed"]
    ctrl_floor = median([r["nll"] for r in ctrl])
    n_high = sum(1 for r in v1 if r["nll"] > ctrl_floor + 0.03)
    assert 1 <= n_high <= 2, (
        "the training-curve page's title assumes a minority of seeds missed the "
        "control's floor", n_high, ctrl_floor)
    easy16, easy24 = worst_easy(c16), worst_easy(c24)
    assert len(easy16) == len(ctrl) and len(easy24) == len(v1)
    n_easy = sum(1 for v in easy24.values() if v >= EASY_BAR)
    assert 1 <= 3 - n_easy <= 2, (
        "the easy-end page's title assumes a minority of seeds collapsed", n_easy)
    # The per-seed table's claim: ONE seed carries the whole failure. Asserted as
    # a conjunction, because the sentence says "every bar" -- and the same seed
    # has to be the one the training-curve page points at, or the deck is telling
    # two stories about two different seeds.
    worst_seed = min(v1, key=lambda r: r["acc"])["seed"]
    worst = [r for r in v1 if r["seed"] == worst_seed][0]
    assert worst["acc"] < ACC_BAR, worst["acc"]
    assert not worst["ok"], "the worst-accuracy seed is not the censored one"
    assert easy24[worst_seed] < EASY_BAR, easy24[worst_seed]
    assert worst_seed == high, (
        "the worst-accuracy seed is not the one that missed the training floor",
        worst_seed, high)
    grid, y16 = _median_curve(c16)
    g24, y24 = _median_curve(c24)
    assert len(grid) == len(g24) == 14, (len(grid), len(g24))
    n_below = sum(1 for x, y in zip(grid, y16) if y24[g24.index(x)] < y)
    assert n_below >= len(grid) // 2, (
        "the longer schedule is not worse at most densities; the solve page's "
        "title would be wrong", n_below, len(grid))
    n_on = sum(1 for r in res.values() if not r["oor"] and abs(r["res"]) <= bar)

    prov = ("Array 51962715 tasks 0-2, registered scorer committed ca11c2b by job "
            "51962711 before the first cell landed, builder commit %s. d320 ff1024 "
            "h4 tied_no_mlp k=1, 50 epochs, lr 1e-3 constant, seeds 0/1/2, 14 "
            "densities x 400 instances, --shuffle-adj --fall." % args.commit)

    # The honest headline, and it is NOT "a longer schedule is worse". Two of the
    # three seeds land within 0.0004 of the control's accuracy -- they did not
    # get worse, they just did not get BETTER, and the registered bar needed
    # +0.0011. The third seed never trained. "Does not help, and destabilizes"
    # is what those three numbers say; "makes it worse" would overclaim from a
    # median that one broken seed moved.
    TITLE = "A longer trained schedule does not help, and destabilizes training"
    d = WebDeck(args.out, TITLE, "hshunt", "2026-11-12", prov)

    d.title_page(
        TITLE,
        subtitle="One shared block trained at twenty-four applications instead "
                 "of sixteen, at identical parameter count. Three seeds, one "
                 "pooled grid of fourteen densities, no code change.")

    exi = example_instance(n_nodes=48, degree=5, n_colors=4, shuffle_adj=True,
                           fall=True)
    deepest = max(exi.pool_depths)

    d.points_page(
        "The knob costs nothing, which is why it was tried first",
        ["%s trainable at sixteen applications and at twenty-four"
         % format(vs["trainable"], ","),
         "The object stays one block and one map, h_{t+1} = F(h_t)",
         "Width, decay and data were closed on evidence before this"])

    d.points_page(
        "The question: are sixteen applications too few for deep chains?",
        ["Draws from this generator reach %d rounds of deduction" % deepest,
         "Sixteen applications cannot serially propagate a %d-round chain" % deepest,
         "The deepest chains are the sparse instances, where rho_50 is read"])

    d.expect("tokenization",
             "Every token is one categorical id in a %d-symbol vocabulary"
             % exi.vocab.size)
    page_tokenization(d, exi)
    d.expect("trace",
             "A deduction fires when a node sees all three other colours",
             note="The same generator both arms trained and were evaluated on; "
                  "the schedule changes the model, never the data.")
    page_trace(d, exi)

    # Accuracy first, as registered: it cannot censor, and here it adjudicates.
    d.claim_page(
        "%s of three seeds reach the accuracy the gate needs" % WORD[n_clear],
        lambda ax: draw_acc(ax, arms, acc_meds), "acc",
        note="Registered in advance: %.4f in at least two of three seeds, the "
             "control's %.4f plus the %.4f the gate needs."
             % (ACC_BAR, S1_ACC, NEED))

    d.table_page(
        "The seed spread grew %.0f-fold when the schedule lengthened"
        % (spread_v / S1_ACC_SPREAD),
        ["arm", "val token accuracy, per seed", "median", "spread"],
        [["T=16 (control)", " / ".join("%.4f" % r["acc"] for r in ctrl),
          "%.4f" % median([r["acc"] for r in ctrl]), "%.4f" % S1_ACC_SPREAD],
         ["T=24", " / ".join("%.4f" % r["acc"] for r in v1),
          "%.4f" % median([r["acc"] for r in v1]), "%.4f" % spread_v]],
        highlight_row=0,
        note="Three tight seeds became three scattered ones. A longer schedule "
             "did not only fail; it made the recipe less reliable.")

    d.claim_page(
        "rho_50 rose from %s to %.3f and %s %s stopped crossing"
        % (S1_RHO50, vs["rho50"], WORD[n_stopped].lower(),
           plural(n_stopped, "seed", "seeds")),
        lambda ax: draw_level(ax, arms, lvl_meds), "level",
        note="Per seed, not pooled. The open triangle never reached a 0.5 solve "
             "rate at any density, so its rho_50 is only a bound.")

    d.table_page(
        "The worst T=24 seed misses every bar: accuracy, level and easy end",
        ["arm", "seed", "rho_50", "mean solve", "token acc", "train NLL"],
        [[lbl, str(r["seed"]),
          "%.3f" % r["rho50"] if r["ok"] else ">%.3f" % r["rho50"],
          "%.4f" % r["mean_solve"], "%.4f" % r["acc"], "%.4f" % r["nll"]]
         for lbl, rows in (("T=16", ctrl), ("T=24", v1)) for r in rows],
        note="Lower rho_50 is better. The other two T=24 seeds sit within %.4f "
             "of the control's accuracy: unchanged, not improved."
             % max(abs(r["acc"] - S1_ACC) for r in v1 if r["seed"] != worst_seed))

    d.claim_page(
        "%s of three seeds never reached the floor the rest found" % WORD[n_high],
        lambda ax: draw_curves(ax, ctrl, v1, high), "curves",
        note="Seed %d finished at %.4f against %s. The gap is training, not "
             "evaluation: it is on the training objective."
             % (high, max(r["nll"] for r in v1),
                " and ".join("%.4f" % r["nll"] for r in v1 if r["seed"] != high)))

    d.claim_page(
        "Mean solve rate fell from %.4f to %.4f across the grid"
        % (S1_MEAN_SOLVE, vs["mean_solve"]),
        lambda ax: draw_solve(ax, c16, c24), "solve",
        note="Median over three seeds at each realized density, 400 instances "
             "per bin. The T=24 curve is the lower one at %d of %d densities."
             % (n_below, len(grid)))

    d.table_page(
        "Training at the length fixes the easy end, except %s %s"
        % (WORD[3 - n_easy].lower(), plural(3 - n_easy, "seed", "seeds")),
        ["arm", "worst easy-bin solve rate, per seed", "registered bar"],
        [["T=16 trained and run (control)",
          " / ".join("%.3f" % easy16[s] for s in sorted(easy16)),
          "%.2f at density %.2f and above" % (EASY_BAR, EASY_RHO)],
         ["T=24 trained and run",
          " / ".join("%.3f" % easy24[s] for s in sorted(easy24)), ""],
         ["16-trained weights RUN at 24 (stage T)",
          "%.4f, worst of three seeds" % T24_EASY_WORST, ""]],
        note="V-c asked whether training at the length fixes the easy-end "
             "collapse frozen weights showed at every T.")

    d.claim_page(
        "Trained at 24 beats frozen weights there, and still loses to 16",
        lambda ax: draw_pair(ax, cs, fs, vs), "pair",
        note="All three medians bound-aware. Stage T published %.3f for the "
             "middle point under its crossing-seed rule; 24-trained wins under "
             "both." % T24_RHO50)

    d.claim_page(
        "%s of three seeds %s on the accuracy-to-solve curve"
        % (WORD[n_on], plural(n_on, "lands", "land")),
        lambda ax: draw_fit(ax, ref, fit, bar, res), "fit",
        note="Residuals %s against the reference ladder's own leave-one-out bar "
             "of %.4f. Nothing architectural happened."
             % (" / ".join("%+.4f" % res[s]["res"] for s in sorted(res)), bar))

    d.points_page(
        "The second rung was cancelled by this stage's own registration",
        ["V-d needed T=32 to read a trend across trained lengths",
         "V-a adjudicates, and it failed on the statistic that cannot censor",
         "V-f named the budget, not more applications, as what runs next"])

    d.points_page(
        "Stage V speaks for one width, one budget, one application count",
        ["d320 tied, k=1, 50 epochs, lr 1e-3 constant, no warmup",
         "N=48 only; stage M's N=96 bars are a different size",
         "Final-epoch checkpoints only, so no best-val selection anywhere"])

    d.table_page(
        "Every cell ran the same config; only the schedule and seed differ",
        ["what", "value"],
        [["array / cells", "51962715 tasks 0-2, three cells, COMPLETED 0:0"],
         ["builder commit", args.commit],
         ["registration", "ca11c2b, committed by 51962711 before cell one landed"],
         ["config", "d320 ff1024 h4 tied_no_mlp, k=1, T=24 applications"],
         ["training", "50 epochs, lr 1e-3 constant, no warmup, 240k instances"],
         ["seeds", "0, 1, 2 (three more read from disk as the T=16 control)"],
         ["eval", "14 densities x 400 instances, --shuffle-adj --fall"],
         ["trainable", "%s, identical to the control's" % format(vs["trainable"], ",")]],
        note="The predictions on every page above were committed before any "
             "number existed to tempt them.")

    d.points_page(
        "What runs next is the training budget, registered before this result",
        ["100 epochs at constant 1e-3, k=1, T=16: one knob, zero parameters",
         "Stage S's cells were still descending at epoch fifty",
         "Explicitly not more width, not cosine decay, not more applications"])

    # Last, as ~/CLAUDE.md requires.
    d.table_page(
        "Against J2's %.3f, stage V moves nothing: the schedule is not it"
        % J2_RHO50,
        ["quantity", "J2 untied d192 (standing)", "T=16 shared block",
         "T=24 shared block"],
        [["rho_50 median", "%.3f" % J2_RHO50, S1_RHO50, rho_txt(vs)],
         ["best single seed", "%.3f" % min(m["rho50"] for m in j2ms),
          "%.3f" % min(r["rho50"] for r in ctrl),
          "%.3f" % min(r["rho50"] for r in v1 if r["ok"])],
         ["mean solve", "%.4f" % j2["mean_solve"], "%.4f" % S1_MEAN_SOLVE,
          "%.4f" % vs["mean_solve"]],
         ["token accuracy", "%.4f" % j2_acc, "%.4f" % S1_ACC,
          "%.4f" % median([r["acc"] for r in v1])],
         ["trainable", "{:,}".format(J2_TRAINABLE),
          "{:,}".format(cs["trainable"]), "{:,}".format(vs["trainable"])],
         ["seeds crossing 0.5", "%d/%d" % (j2["n_cross"], j2["n"]),
          "%d/%d" % (cs["n_cross"], cs["n"]), "%d/%d" % (vs["n_cross"], vs["n"])]],
        highlight_row=0,
        note="Same grid, same 400 instances per bin, same flags. No improvement "
             "is claimed and the standing minimum does not move.")

    path = d.write()
    used = {"tokenization", "trace", "acc", "level", "curves", "solve", "pair",
            "fit"}
    stale = sorted(f for f in os.listdir(d.fig_dir)
                   if f.endswith(".png") and f[:-4] not in used)
    print("wrote %s (%d pages, %d figures used)" % (path, d.n, len(used)))
    print("stale figures from the previous deck, unreferenced: %s"
          % (", ".join(stale) or "none"))
    print("V-a: %d/3 clear %.4f, accs %s, spread %.4f (control %.4f)"
          % (n_clear, ACC_BAR, "/".join("%.4f" % r["acc"] for r in v1), spread_v,
             S1_ACC_SPREAD))
    print("V-b: rho_50 %s, %d/3 crossing, per seed %s"
          % (rho_txt(vs), vs["n_cross"],
             "/".join(("%.3f" % r["rho50"]) + ("" if r["ok"] else "*")
                      for r in v1)))
    print("V-c: worst easy bin %s (bar %.2f at rho %.2f), control %s"
          % ("/".join("%.3f" % easy24[s] for s in sorted(easy24)), EASY_BAR,
             EASY_RHO, "/".join("%.3f" % easy16[s] for s in sorted(easy16))))
    print("V-e: trained-24 %.3f (%d/3) vs frozen-24 %.3f bound-aware / %.3f by "
          "stage T's crossing-seed rule (%d/3, published %.3f)"
          % (vs["rho50"], vs["n_cross"], fs["rho50"],
             ok_median(list(froz.values())), fs["n_cross"], T24_RHO50))
    print("curves: control floor %.4f, T=24 %s, %d seed(s) above floor+0.03"
          % (ctrl_floor, "/".join("%.4f" % r["nll"] for r in v1), n_high))
    print("solve: T=24 below the control at %d of %d densities"
          % (n_below, len(grid)))
    print("sufficiency: bar %.4f (registered %.4f), residuals %s"
          % (bar, LOO_BAR,
             " ".join("s%d %+.4f%s" % (s, res[s]["res"],
                                       " weak" if res[s]["weak"] else "")
                      for s in sorted(res))))
    print("mean_solve: control %.4f -> T=24 %.4f"
          % (S1_MEAN_SOLVE, vs["mean_solve"]))


if __name__ == "__main__":
    main()
