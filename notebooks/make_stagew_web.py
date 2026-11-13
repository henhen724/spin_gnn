#!/usr/bin/env python3
"""Stage W as a nested web deck: ~/slides/decks/spin-gnn/budget/.

WHAT THIS REPORTS. One knob, zero parameters: the same tied d320 cell stage S
trained for 50 epochs, trained for 100. W-c holds (the loss did fall) and
W-a/W-b/W-e fail (the solving did not follow), so the honest headline is that
the budget bought training loss and nothing downstream of it.

NESTED, NOT TOP-LEVEL. ~/CLAUDE.md changed on 2026-11-12: decks now live at
decks/<project>/<deck>/ and /<project>/ becomes a contents page. Stage V's deck
moved down to decks/spin-gnn/schedule/ in commit 96324f2; this one is
decks/spin-gnn/budget/. One password covers both, so the link already shared
keeps working, and `claude-slides-publish.sh spin-gnn` would now be REFUSED --
publish `spin-gnn/budget`.

NOTHING IS RE-DERIVED. Every number comes through the registered scorer's own
loaders (scripts/stagew_score.py, committed before the first cell landed) and
the shared explanation pages come from notebooks/deck_explain.py. The training
curves come from notebooks/stagew_figs.py, which takes its log parse, step
definition and NLL levels from scripts/step_epoch_census.py. check_claims()
re-asserts every sentence against the data: a web deck is a URL, and a stale
number here keeps serving.

Usage (in a job -- this imports torch, and $HOME I/O on the login node wedges):
    python3 -u -m notebooks.make_stagew_web --commit <sha> [--out <dir>]
"""

import argparse
import os
import sys
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

from slides import MUTED, SERIES, label_line, style_axes

from notebooks import deck_explain
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_staget_deck import J2_RHO50, J2_TRAINABLE
from notebooks.make_staget_web import WebDeck
# The loaders are stage V's, unchanged: same cells table, same bound-aware
# summary, same per-bin curves. Only the rung key and the arm differ.
from notebooks.make_stagev_web import arm, curves, history, plural
from notebooks.stagew_figs import by_seed as logs_by_seed
from notebooks.stagew_figs import fig_training_curves

from scripts.acc_vs_solve import collect, is_reference, pava, predict, rms
from scripts.minarch_analysis import SCAN_DIRS
from scripts.stagec_score import escapes, rho_txt, rung, summarize
from scripts.stagej_score import LOO_BAR, SUFFICIENCY_DIRS
from scripts.stagek_score import FREE, SCAN_K
from scripts.stagen_score import J2, SCAN_N, acc_of, load_pooled
from scripts.stageo_score import SCAN_O
from scripts.stagep_score import CENSOR_FLOOR, SCAN_P
from scripts.stages_score import SCAN_S
from scripts.staget_score import SCAN_T
from scripts.stageu_score import SCAN_U
from scripts.step_epoch_census import BASIN, FLOOR, level, step_epoch
from scripts.stagev_score import SCAN_V
from scripts.stagew_score import (ACC_BAR, ADDED_EPOCHS, BASELINE_DIRS,
                                  BASIN_NLL, EPOCHS, NEED, NLL_BAR, S1, S1_ACC,
                                  S1_ACC_SPREAD, S1_MEAN_SOLVE, S1_NLL_MEDIAN,
                                  S1_RHO50, S1_TRAINABLE, SCAN_W, W1, nll_at,
                                  sched_count)

DECK = "/home/users/hshunt/slides/decks/spin-gnn/budget"

WORD = {0: "Zero", 1: "One", 2: "Two", 3: "Three"}
CONTROL_EPOCHS = EPOCHS - ADDED_EPOCHS      # 50

# step_epoch_census.level() returns STUCK for the level the LEVELS
# explanation page calls the plateau. One level, one word, on every page.
LEVEL_WORD = {"BASIN": "basin", "FLOOR": "floor", "STUCK": "plateau"}


def sufficiency(target):
    """The N=48 accuracy-to-solve curve, by acc_vs_solve's own functions.

    Stage V's builder hardcodes its own rung in the held-out filter, so the
    function is rewritten here with the rung as an argument rather than
    imported and monkey-patched.
    """
    dirs = SUFFICIENCY_DIRS + [SCAN_K, SCAN_N, SCAN_O, SCAN_P, SCAN_S, SCAN_T,
                               SCAN_U, SCAN_V, SCAN_W]
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
        if c["rung"] != target:
            continue
        p, oor, cov = predict(fit, c["acc"])
        res[int(c["seed"])] = {"acc": c["acc"], "solve": c["mean_solve"],
                               "pred": p, "res": c["mean_solve"] - p,
                               "oor": oor, "weak": cov > bar}
    return ref, fit, bar, res


def within_run(rows, ms):
    """W-c's pair: train NLL at the control's last epoch and at this run's.

    The whole point of W-c is a number from the MIDDLE of the run, so epoch 50
    is read by the scorer's OWN nll_at() -- from the cell's path, which arm()
    does not carry out -- rather than by a second index written here. The
    cross-check against the same entry of the history arm() already loaded is
    what would catch an off-by-one in either of them.
    """
    paths = {m["seed"]: m["path"] for m in ms}
    out = []
    for r in rows:
        at50 = nll_at(paths[r["seed"]], CONTROL_EPOCHS)
        if at50 is None:
            raise SystemExit("seed %d has no epoch-%d entry; W-c cannot be read"
                             % (r["seed"], CONTROL_EPOCHS))
        h = r["h"]
        if len(h) != EPOCHS:
            raise SystemExit("seed %d has %d history entries, not %d"
                             % (r["seed"], len(h), EPOCHS))
        if abs(h[CONTROL_EPOCHS - 1]["train_nll"] - at50) > 1e-9:
            raise SystemExit("nll_at() and history.json disagree at epoch %d"
                             % CONTROL_EPOCHS)
        out.append({"seed": r["seed"], "at50": at50, "end": r["nll"],
                    "fall": at50 - r["nll"], "level": LEVEL_WORD[level(r["nll"])]})
    return out


def check_claims(ctrl, cs, w1, ws, wc, steps, bar, res):
    """Every sentence this deck publishes, asserted against the data first."""
    # 1. the control is still stage S's cell, exactly as the scorer froze it
    assert rho_txt(cs) == S1_RHO50, (rho_txt(cs), S1_RHO50)
    assert cs["trainable"] == S1_TRAINABLE, cs["trainable"]
    assert round(median([r["acc"] for r in ctrl]), 4) == S1_ACC
    assert abs(median([r["nll"] for r in ctrl]) - S1_NLL_MEDIAN) < 5e-5, (
        "history.json disagrees with the scorer's frozen train NLL median")
    assert abs(max(r["acc"] for r in ctrl) - min(r["acc"] for r in ctrl)
               - S1_ACC_SPREAD) < 5e-5
    assert cs["n_cross"] == 3, cs["n_cross"]

    # 2. THE HEADLINE GUARD: the knob cost nothing. An epoch count cannot move a
    # parameter count, but asserting it is what makes "zero parameters" a
    # measurement rather than an argument -- and guard 7 does not protect this
    # stage, so the count has to be shown to be the control's.
    assert ws["trainable"] == cs["trainable"] == sched_count(320) == 613818, (
        "the budget moved the parameter count", ws["trainable"])
    assert ws["trainable"] < CENSOR_FLOOR, "guard 7 does not protect this stage"

    # 3. W-c HOLDS: the loss fell, in every seed, and by the registered bar in a
    # majority. This is the one prediction of the five that survived.
    assert all(r["fall"] > 0 for r in wc), [r["fall"] for r in wc]
    assert median([r["fall"] for r in wc]) >= NLL_BAR, (
        "the median fall misses W-c's bar; this deck's title is wrong",
        [r["fall"] for r in wc], NLL_BAR)
    assert sum(1 for r in wc if r["fall"] >= NLL_BAR) >= 2

    # 4. W-a FAILS, on the statistic that cannot censor, and the spread is why a
    # single clearing seed is not evidence.
    n_clear = sum(1 for r in w1 if r["acc"] >= ACC_BAR)
    assert n_clear == 1, ("the accuracy page's title names one seed",
                          [r["acc"] for r in w1])
    sw = max(r["acc"] for r in w1) - min(r["acc"] for r in w1)
    assert sw > NEED, ("the spread no longer swamps the required gain", sw, NEED)
    assert sw > 10 * S1_ACC_SPREAD, ("the spread did not blow up", sw)

    # 5. W-b and W-e: the gate did not move, on both statistics.
    assert ws["rho50_readable"], "the pooled 100-epoch median is a bound"
    assert ws["rho50"] > float(S1_RHO50), (
        "the budget improved rho_50", ws["rho50"])
    assert ws["n_cross"] < cs["n_cross"], (ws["n_cross"], cs["n_cross"])
    assert ws["mean_solve"] < S1_MEAN_SOLVE, ws["mean_solve"]

    # 6. W-d in one seed, and the page that reports it needs the SAME seed to be
    # the basin one, the accuracy-clearing one and the no-better-rho_50 one --
    # that conjunction is the finding, and a deck telling it about two different
    # seeds would be telling two stories.
    basin = [r for r in wc if r["end"] < BASIN_NLL]
    assert len(basin) == 1, [r["end"] for r in wc]
    best = max(w1, key=lambda r: r["acc"])
    assert basin[0]["seed"] == best["seed"], (basin[0]["seed"], best["seed"])
    assert best["acc"] >= ACC_BAR, best["acc"]
    assert best["rho50"] > float(S1_RHO50), (
        "the basin seed beat the control's rho_50 after all", best["rho50"])
    assert abs(BASIN - BASIN_NLL) < 5e-4, (BASIN, BASIN_NLL)

    # 7. No cell stepped. The curves page says the basin was reached GRADUALLY,
    # which is only a claim if the registered step detector agrees.
    assert all(s is None for s in steps.values()), steps

    # 8. the sufficiency bar is the registered one, and every seed is on the
    # curve -- which is what makes this a move ALONG the relation, not a change
    # to it.
    assert abs(bar - LOO_BAR) < 5e-4, (bar, LOO_BAR)
    assert len(res) == len(w1), (len(res), len(w1))
    assert all(abs(r["res"]) <= bar for r in res.values()), (
        "a seed left the accuracy-to-solve curve; the fit page is wrong",
        {s: r["res"] for s, r in res.items()})
    return n_clear, sw, basin[0], best


# ---------------------------------------------------------------- figures

XS = {CONTROL_EPOCHS: 0, EPOCHS: 1}
TICKS = ["%d epochs\n(the control)" % CONTROL_EPOCHS, "%d epochs" % EPOCHS]


def _arms_axis(ax, ylabel):
    ax.set_xticks([0, 1])
    ax.set_xticklabels(TICKS)
    ax.set_xlim(-0.45, 1.6)
    style_axes(ax, "training budget, at identical architecture", ylabel)


def _seed_dots(ax, arms, field, meds, only_ok=False):
    """Seeds as dots, with the median bar the PAGE TITLE quotes.

    `meds` is passed in rather than computed here: rho_50's median is
    bound-aware and carries a censored seed's bound, which a median over the
    plotted dots cannot reproduce. A figure must draw the number its own title
    names.
    """
    for t, rows in arms.items():
        for r in rows:
            ok = r["ok"] or not only_ok
            ax.plot([XS[t]], [r[field]], "o" if ok else "^", ms=8,
                    color=SERIES[0], mfc=SERIES[0] if ok else "white", mew=1.7,
                    zorder=4)
        ax.plot([XS[t] - 0.17, XS[t] + 0.17], [meds[t]] * 2, "-",
                color=SERIES[1], lw=2.6, zorder=3)
    label_line(ax, 1.33, meds[EPOCHS], "median", color=SERIES[1])


def draw_acc(ax, arms, meds):
    _seed_dots(ax, arms, "acc", meds)
    ax.axhline(ACC_BAR, ls="--", lw=1.2, color=MUTED, zorder=1)
    label_line(ax, 1.6, ACC_BAR, "bar needs\n%.4f" % ACC_BAR, color=MUTED)
    vals = [r["acc"] for rows in arms.values() for r in rows] + [ACC_BAR]
    pad = 0.12 * (max(vals) - min(vals))
    ax.set_ylim(min(vals) - pad, max(vals) + pad)
    _arms_axis(ax, "validation token accuracy")


def draw_level(ax, arms, meds):
    _seed_dots(ax, arms, "rho50", meds, only_ok=True)
    for y, ls, txt in ((FREE, "--", "gate %.3f" % FREE),
                       (J2_RHO50, ":", "J2 %.3f" % J2_RHO50)):
        ax.axhline(y, ls=ls, lw=1.2, color=MUTED, zorder=1)
        label_line(ax, 1.6, y, txt, color=MUTED)
    cens = [r for r in arms[EPOCHS] if not r["ok"]]
    if cens:
        label_line(ax, XS[EPOCHS], cens[0]["rho50"], "never crossed 0.5",
                   color=SERIES[0], dy=-16)
    vals = [r["rho50"] for rows in arms.values() for r in rows] + [FREE, J2_RHO50]
    ax.set_ylim(min(vals) - 0.02, max(vals) + 0.03)
    _arms_axis(ax, "rho_50  (lower is better)")


def _median_curve(by):
    xs = sorted({round(r["realized_rho"], 4) for rs in by.values() for r in rs})
    ys = [median([r["solve_rate"] for rs in by.values() for r in rs
                  if round(r["realized_rho"], 4) == x]) for x in xs]
    return xs, ys


def draw_solve(ax, c50, c100):
    for i, (by, name, dy) in enumerate(((c50, "%d epochs" % CONTROL_EPOCHS, 12),
                                        (c100, "%d epochs" % EPOCHS, -4))):
        xs, ys = _median_curve(by)
        ax.plot(xs, ys, "-o", ms=4, color=SERIES[i], lw=2.0, zorder=3 - i)
        label_line(ax, xs[-1], ys[-1], name, color=SERIES[i], dy=dy)
    ax.axhline(0.5, ls="--", lw=1.2, color=MUTED, zorder=1)
    label_line(ax, _median_curve(c50)[0][0], 0.5,
               "the 0.5 crossing that defines rho_50", color=MUTED, dy=8)
    ax.set_ylim(-0.03, 1.05)
    style_axes(ax, "realized hint density", "fraction of puzzles solved")


def draw_fit(ax, ref, fit, bar, res):
    xs = [p[0] for p in fit]
    ys = [p[1] for p in fit]
    ax.plot(xs, ys, "-", color=MUTED, lw=1.8, zorder=2)
    ax.fill_between(xs, [y - bar for y in ys], [y + bar for y in ys],
                    color=MUTED, alpha=0.13, lw=0, zorder=1)
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

    n_explain = deck_explain.check()

    cells = load_pooled([SCAN_W] + BASELINE_DIRS + SCAN_DIRS, 2000)
    ctrl, cs = arm(cells, S1)
    w1, ws = arm(cells, W1)
    wms = rung(cells, W1)
    c50, c100 = curves([SCAN_S], S1), curves([SCAN_W], W1)
    wc = within_run(w1, wms)
    steps = {s: step_epoch(h) for s, h in logs_by_seed(
        "sud2N48BudgetW_52053221_%d.out").items()}
    ref, fit, bar, res = sufficiency(W1)
    n_clear, spread_w, basin, best = check_claims(ctrl, cs, w1, ws, wc, steps,
                                                  bar, res)

    arms = {CONTROL_EPOCHS: ctrl, EPOCHS: w1}
    acc_meds = {CONTROL_EPOCHS: median([r["acc"] for r in ctrl]),
                EPOCHS: median([r["acc"] for r in w1])}
    lvl_meds = {CONTROL_EPOCHS: cs["rho50"], EPOCHS: ws["rho50"]}
    j2ms = rung(cells, J2)
    assert j2ms, "the standing minimum J2 is not on disk; the last page needs it"
    j2 = summarize(j2ms)
    j2_acc = median([a for _, a in acc_of(j2ms)])

    n_esc, _nll = escapes(wms)
    grid, y50 = _median_curve(c50)
    g100, y100 = _median_curve(c100)
    assert len(grid) == len(g100) == 14, (len(grid), len(g100))
    n_below = sum(1 for x, y in zip(grid, y50) if y100[g100.index(x)] < y)
    n_on = sum(1 for r in res.values() if not r["oor"] and abs(r["res"]) <= bar)
    med_fall = median([r["fall"] for r in wc])

    prov = ("Array 52053221 tasks 0-2 (COMPLETED 0:0, 15h51m-15h56m each), "
            "registered scorer committed beedf66 before the first cell landed, "
            "scored by job 52062829, builder commit %s. tied_no_mlp T=16 d320 "
            "ff1024 h4 k=1, %d epochs, lr 1e-3 constant, seeds 0/1/2, 14 "
            "densities x 400 instances, --shuffle-adj --fall."
            % (args.commit, EPOCHS))

    # The honest headline. W-c is the one prediction that survived: the loss did
    # fall, in all three seeds, by more than the bar in two. Everything
    # downstream of it -- accuracy, rho_50, crossing count, mean solve -- is flat
    # or worse. "Bought training loss" is exactly what those numbers say.
    TITLE = "Twice the budget bought training loss, and solving did not follow"
    d = WebDeck(args.out, TITLE, "hshunt", "2026-11-12", prov)

    # The subtitle carries what a separate "the arm" page would have said. The
    # deck is at the 25-page cap with the explanation pages ~/CLAUDE.md now
    # requires, and the title page's subtitle is the one place exempt from the
    # body cap, so the setup goes here rather than costing a page the per-seed
    # table needs more.
    d.title_page(
        TITLE,
        subtitle="One hundred epochs instead of fifty, at an identical "
                 "architecture and an identical parameter count -- %s trainable "
                 "either way. Stage S's cells were still descending at epoch "
                 "fifty, which is why the budget was tried before more width, "
                 "decay or applications. Three seeds, one pooled grid of "
                 "fourteen densities, no code change."
                 % format(ws["trainable"], ","))

    exi = example_instance(n_nodes=48, degree=5, n_colors=4, shuffle_adj=True,
                           fall=True)

    # The shared explanation pages, in the order deck_explain fixes, with its
    # one FIGURE_SLOT replaced by this deck's two rendered figure pages.
    for page in deck_explain.PAGES:
        if page is deck_explain.FIGURE_SLOT:
            d.expect("tokenization",
                     "Every token is one categorical id in a %d-symbol vocabulary"
                     % exi.vocab.size)
            page_tokenization(d, exi)
            d.expect("trace",
                     "A deduction fires when a node sees all three other colours",
                     note="The same generator trained and evaluated both arms; "
                          "the budget changes the model, never the data.")
            page_trace(d, exi)
        else:
            d._add(page)

    # Accuracy first, as registered: it cannot censor, and here it adjudicates.
    d.claim_page(
        "%s of three seeds reaches the accuracy the bar needs" % WORD[n_clear],
        lambda ax: draw_acc(ax, arms, acc_meds), "acc",
        note="Registered in advance: %.4f in at least two of three seeds -- "
             "the control's %.4f plus a %.4f margin, fixed before any cell "
             "landed." % (ACC_BAR, S1_ACC, NEED))

    d.table_page(
        "The seed spread is %.4f and the bar asked for %.4f" % (spread_w, NEED),
        ["arm", "val token accuracy, per seed", "median", "spread"],
        [["%d epochs (control)" % CONTROL_EPOCHS,
          " / ".join("%.4f" % r["acc"] for r in ctrl),
          "%.4f" % median([r["acc"] for r in ctrl]), "%.4f" % S1_ACC_SPREAD],
         ["%d epochs" % EPOCHS, " / ".join("%.4f" % r["acc"] for r in w1),
          "%.4f" % median([r["acc"] for r in w1]), "%.4f" % spread_w]],
        highlight_row=0,
        note="One seed over the bar is not evidence when the spread between "
             "seeds is %.0f times the gain the bar asks for."
             % (round(spread_w, 4) / round(NEED, 4)))

    d.table_page(
        "Training loss fell in all three seeds, by %.4f at the median" % med_fall,
        ["seed", "train NLL at epoch %d" % CONTROL_EPOCHS,
         "at epoch %d" % EPOCHS, "fall", "bar %.4f" % NLL_BAR, "level reached"],
        [[str(r["seed"]), "%.4f" % r["at50"], "%.4f" % r["end"],
          "%.4f" % r["fall"], "clears" if r["fall"] >= NLL_BAR else "below",
          r["level"]] for r in wc],
        note="Within each run, not across runs: the same cell at epoch %d and "
             "at epoch %d. The bar is stage S's measured tail slope."
             % (CONTROL_EPOCHS, EPOCHS))

    # This figure is rendered by notebooks/stagew_figs.py, which saves the PNG
    # itself rather than handing a figure to savefig(), so the page is added
    # through the same _fig_md() the other figure pages use -- which is what
    # enforces the title and note caps on it too.
    fig_training_curves(d.fig_dir)
    d._add(d._fig_md(
        "curves",
        "Seed %d reached the basin gradually, with no step anywhere" % basin["seed"],
        "Each cell against its own %d-epoch control, same seed, same config. "
        "No step fires in either arm." % CONTROL_EPOCHS))

    d.claim_page(
        "rho_50 rose from %s to %.3f and one seed stopped crossing"
        % (S1_RHO50, ws["rho50"]),
        lambda ax: draw_level(ax, arms, lvl_meds), "level",
        note="Per seed, not pooled. The open triangle never reached a 0.5 solve "
             "rate at any density, so its rho_50 is only a bound.")

    d.table_page(
        "The seed that reached the basin did not solve any better",
        ["arm", "seed", "train NLL", "level", "token acc", "rho_50"],
        [[lbl, str(r["seed"]), "%.4f" % r["nll"], LEVEL_WORD[level(r["nll"])],
          "%.4f" % r["acc"],
          "%.3f" % r["rho50"] if r["ok"] else ">%.3f" % r["rho50"]]
         for lbl, rows in (("%d ep" % CONTROL_EPOCHS, ctrl),
                           ("%d ep" % EPOCHS, w1)) for r in rows],
        note="Lower rho_50 is better. Seed %d is the first one-block cell to end "
             "below %.3f, and its rho_50 is %.3f against the control's %s."
             % (basin["seed"], BASIN_NLL, best["rho50"], S1_RHO50))

    d.claim_page(
        "Mean solve rate fell from %.4f to %.4f across the grid"
        % (S1_MEAN_SOLVE, ws["mean_solve"]),
        lambda ax: draw_solve(ax, c50, c100), "solve",
        note="Median over three seeds at each realized density, 400 instances "
             "per bin. The %d-epoch curve is the lower one at %d of %d densities."
             % (EPOCHS, n_below, len(grid)))

    d.claim_page(
        "%s of three seeds %s on the accuracy-to-solve curve"
        % (WORD[n_on], plural(n_on, "lands", "land")),
        lambda ax: draw_fit(ax, ref, fit, bar, res), "fit",
        note="Residuals %s against the reference ladder's own leave-one-out bar "
             "of %.4f. The budget moved cells along the relation."
             % (" / ".join("%+.4f" % res[s]["res"] for s in sorted(res)), bar))

    # Last, as ~/CLAUDE.md requires.
    d.table_page(
        "Against J2's %.3f, twice the budget moves nothing" % J2_RHO50,
        ["quantity", "J2 untied d192 (standing)", "%d epochs" % CONTROL_EPOCHS,
         "%d epochs" % EPOCHS],
        [["rho_50 median", "%.3f" % J2_RHO50, S1_RHO50, rho_txt(ws)],
         ["best single seed", "%.3f" % min(m["rho50"] for m in j2ms),
          "%.3f" % min(r["rho50"] for r in ctrl),
          "%.3f" % min(r["rho50"] for r in w1 if r["ok"])],
         ["mean solve", "%.4f" % j2["mean_solve"], "%.4f" % S1_MEAN_SOLVE,
          "%.4f" % ws["mean_solve"]],
         ["token accuracy", "%.4f" % j2_acc, "%.4f" % S1_ACC,
          "%.4f" % median([r["acc"] for r in w1])],
         ["trainable", "{:,}".format(J2_TRAINABLE),
          "{:,}".format(cs["trainable"]), "{:,}".format(ws["trainable"])],
         ["seeds crossing 0.5", "%d/%d" % (j2["n_cross"], j2["n"]),
          "%d/%d" % (cs["n_cross"], cs["n"]), "%d/%d" % (ws["n_cross"], ws["n"])]],
        highlight_row=0,
        note="Same grid, same 400 instances per bin, same flags. No improvement "
             "is claimed and the standing minimum does not move.")

    path = d.write()
    used = {"tokenization", "trace", "acc", "level", "curves", "solve", "fit"}
    stale = sorted(f for f in os.listdir(d.fig_dir)
                   if f.endswith(".png") and f[:-4] not in used)
    print("wrote %s (%d pages, %d explanation, %d figures)"
          % (path, d.n, n_explain, len(used)))
    print("stale figures, unreferenced: %s" % (", ".join(stale) or "none"))
    print("W-a: %d/3 clear %.4f, accs %s, spread %.4f (control %.4f, needs %.4f)"
          % (n_clear, ACC_BAR, "/".join("%.4f" % r["acc"] for r in w1), spread_w,
             S1_ACC_SPREAD, NEED))
    print("W-b: rho_50 %s, %d/3 crossing, per seed %s (control %s, %d/3)"
          % (rho_txt(ws), ws["n_cross"],
             "/".join(("%.3f" % r["rho50"]) + ("" if r["ok"] else "*")
                      for r in w1), S1_RHO50, cs["n_cross"]))
    print("W-c: falls %s, median %.4f against bar %.4f"
          % ("/".join("%.4f" % r["fall"] for r in wc), med_fall, NLL_BAR))
    print("W-d: basin seed %d at %.4f with acc %.4f, rho_50 %.3f"
          % (basin["seed"], basin["end"], best["acc"], best["rho50"]))
    print("steps: %s; plateau escapes %d/3; levels %s"
          % ("/".join("%s" % (steps[s] if steps[s] is not None else "none")
                      for s in sorted(steps)), n_esc,
             "/".join(r["level"] for r in wc)))
    print("solve: %d epochs below the control at %d of %d densities; mean_solve "
          "%.4f -> %.4f" % (EPOCHS, n_below, len(grid), S1_MEAN_SOLVE,
                            ws["mean_solve"]))
    print("sufficiency: bar %.4f (registered %.4f), residuals %s"
          % (bar, LOO_BAR,
             " ".join("s%d %+.4f%s" % (s, res[s]["res"],
                                       " weak" if res[s]["weak"] else "")
                      for s in sorted(res))))
    print("levels: basin <= %.3f, floor <= %.3f" % (BASIN, FLOOR))


if __name__ == "__main__":
    main()
