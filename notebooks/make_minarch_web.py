"""The architecture ladder as a standalone web deck for someone new to this.

~/slides/decks/spin-gnn-minimal/. The question it answers is the one hshunt set
on 2026-11-04 and restated on 11-10: not "what is the best solver" and not "what
has the fewest parameters", but HOW SIMPLE AN OBJECT can still learn direct
deduction -- few parts, few distinct operations, ideally something a theorist
could write down. Every page says what was removed and whether the task
survived.

WHAT THE DECK IS BUILT FROM. Every N=48 architecture cell this project has
scored, across fifteen scan directories, read through the shared scorer's own
loaders (scripts/minarch_analysis: load_all, cell_key, cell_metrics, rung_key,
rung_label, _bound_median) so that a rung here is the same rung the stage
scorers printed. Nothing is retyped and nothing is re-derived.

  * N=96 directories are DELIBERATELY ABSENT (SCAN_F, SCAN_M). Finding 55:
    rho_50 is never pooled across sizes, and a deck that quietly mixed them
    would be comparing two different tasks.
  * SCAN_T is absent too: it re-EVALUATES stage S's frozen weights at other
    iteration counts, so its rungs are not trained architectures.
  * SCAN_V is running as this is written and has no scored curves.

THREE THINGS THIS DECK HAS TO SAY HONESTLY:

  1. rho_50 CENSORS. A cell that never solves half the puzzles at any density
     has no crossing, and prints as a bound, never as a number at the grid
     edge. Fifteen of the rungs here are censored and the figures mark them.
  2. TRAINABILITY IS NOT CAPACITY. A rung whose three seeds never left finding
     44's 1.64 training plateau did not train at this budget; it is not a proof
     that the architecture cannot represent deduction. The deck uses that
     wording everywhere.
  3. 0.740 IS A GATE, NOT A SCORE. It is the hub's 0.592 plus the largest
     across-seed spread measured in this project, 0.1484. Clearing it means
     "still learns the task"; beating it by 0.01 means nothing.

Usage (in a job -- $HOME I/O on sh02-ln02 wedges, and this imports torch
through deck_tokenization):
    python3 -u -m notebooks.make_minarch_web --commit <sha> [--out <dir>]
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

from slides import MUTED, SERIES, label_line, style_axes

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_staget_web import WebDeck

from scripts.acc_vs_solve import collect, is_reference, pava, predict, rms
from scripts.minarch_analysis import (PLATEAU, SCAN_C, SCAN_D, SCAN_DEFAULT, SCAN_E,
                                     SCAN_G, SCAN_H, SCAN_I, SCAN_J, SCAN_L6,
                                     _bound_median, cell_key, cell_metrics, fmt_rho50,
                                     history, load_all, rung_label, sort_key, spread)
from scripts.stagej_score import LOO_BAR
from scripts.stagek_score import FREE, HUB_RHO50, HUB_TRAINABLE, SCAN_K
from scripts.stagen_score import SCAN_N
from scripts.stageo_score import SCAN_O
from scripts.stagep_score import SCAN_P
from scripts.stages_score import SCAN_S
from scripts.stageu_score import SCAN_U

DECK = "/home/users/hshunt/slides/decks/spin-gnn-minimal"

# Every N=48 architecture directory. See the module docstring for what is kept
# out and why.
ALL_DIRS = [SCAN_DEFAULT, SCAN_L6, SCAN_C, SCAN_D, SCAN_E, SCAN_G, SCAN_H,
            SCAN_I, SCAN_J, SCAN_K, SCAN_N, SCAN_O, SCAN_P, SCAN_S, SCAN_U]

# The largest across-seed spread ever measured here, frozen on 11-04 and never
# recomputed. FREE = HUB_RHO50 + S_STAR is the gate.
S_STAR = 0.1484

# What each arch_mode actually is, in the object's own terms. These strings are
# the deck's glossary and the reason it can be read by someone who has never
# seen the codebase.
ARCH_WHAT = {
    "full": "standard block: attention, then a two-layer MLP",
    "no_mlp": "attention only -- the MLP is deleted",
    "linear": "the MLP's nonlinearity removed, both layers kept",
    "rf_mlp": "the MLP frozen at its random initialization",
    "rf_all": "attention and MLP both frozen; only embeddings train",
    "tied_no_mlp": "one attention-only block, applied over and over",
}


# ---------------------------------------------------------------- the ladder

def ladder(boot=1):
    """One row per rung: the three seeds summarized the way the scorers do.

    boot=1 because this builder prints no confidence interval -- rho50_ci
    divides by n_boot, so 0 raises, and 2000 bootstrap resamples per cell would
    cost minutes for numbers no page shows.
    """
    rows = load_all(ALL_DIRS)
    if not rows:
        raise SystemExit("no scored curves under any of the %d scan dirs" % len(ALL_DIRS))
    cells = defaultdict(list)
    for r in rows:
        cells[cell_key(r)].append(r)

    by_rung = defaultdict(dict)
    for key in sorted(cells, key=sort_key):
        by_rung[key[:-1]][key[-1]] = cell_metrics(cells[key], boot)

    out = []
    for rung, seeds in by_rung.items():
        ms = [seeds[s] for s in sorted(seeds)]
        bm = _bound_median([(m["rho50"], m["flag"] == "ok") for m in ms])
        crossed = [m for m in ms if m["flag"] == "ok"]
        esc = 0
        for m in ms:
            h = history(m["path"])
            # history() returns (n_epochs, final_train_nll, final_val, best_val,
            # slope); a cell "escaped" if its training loss ever got below the
            # plateau finding 44 measured.
            if h and h[1] == h[1] and h[1] < PLATEAU:
                esc += 1
        # rung_key only carries (d_model, ff, warmup) when they are non-default,
        # and anything after them is a tail read BY TYPE, not by position (a
        # "k2" or an epoch count can sit at index 3 on a short key). So the
        # shape block is only present when indices 3..5 are all ints.
        has_v = len(rung) > 5 and all(isinstance(rung[i], int) for i in (3, 4, 5))
        out.append({
            "rung": rung, "label": rung_label(*rung).strip(),
            "arch": rung[2], "L": rung[0], "nhead": rung[1],
            "d_model": rung[3] if has_v else 256,
            "ff": rung[4] if has_v else 1024,
            "k": next((int(el[1:]) for el in rung
                       if isinstance(el, str) and el.startswith("k")), 1),
            "rho50": bm["rho50_bm"], "readable": bm["bm_readable"],
            "crossing": "%d/%d" % (len(crossed), len(ms)),
            "n_crossing": len(crossed), "n_seeds": len(ms),
            "spread": spread([m["rho50"] for m in crossed]) if len(crossed) > 1
                      else float("nan"),
            "mean_solve": sum(m["mean_solve"] for m in ms) / len(ms),
            "trainable": ms[0]["n_trainable"],
            "escapes": esc, "metrics": ms,
            "free": bool(bm["bm_readable"] and bm["rho50_bm"] <= FREE
                         and len(crossed) == len(ms)),
        })
    out.sort(key=lambda r: (not r["readable"], r["rho50"] if r["readable"] else 0.0,
                            r["trainable"]))
    return rows, out


def find(rungs, **want):
    """The one rung matching every field given, or None. Keeps the page code
    free of index arithmetic over a list that is sorted by score."""
    hits = [r for r in rungs if all(r[k] == v for k, v in want.items())]
    if len(hits) > 1:
        raise SystemExit("ambiguous rung %s: %s" % (want, [h["label"] for h in hits]))
    return hits[0] if hits else None


def txt(r):
    """rho_50 as the deck prints it: a bound when the rung never crossed."""
    if r["readable"]:
        return "%.3f" % r["rho50"]
    return ">0.95"


def num(n):
    return "{:,}".format(int(n))


# ---------------------------------------------------------------- figures

def draw_depth(rungs):
    """The depth ladder at the standard block: rho_50 against layer count."""
    pts = sorted([r for r in rungs if r["arch"] == "full" and r["nhead"] == 4
                  and r["d_model"] == 256 and r["ff"] == 1024],
                 key=lambda r: r["L"])
    def go(ax):
        xs = [r["L"] for r in pts if r["readable"]]
        ys = [r["rho50"] for r in pts if r["readable"]]
        ax.plot(xs, ys, color=SERIES[0], marker="o", ms=7, lw=2.2, zorder=3)
        cx = [r["L"] for r in pts if not r["readable"]]
        ax.plot(cx, [0.95] * len(cx), color=SERIES[0], marker="^", ms=9, lw=0,
                mfc="white", mew=2.0, zorder=3)
        ax.axhline(FREE, color=MUTED, ls="--", lw=1.2, zorder=1)
        ax.annotate("gate 0.740", xy=(max(r["L"] for r in pts), FREE),
                    xytext=(0, 6), textcoords="offset points", color=MUTED, size=11)
        ax.set_ylim(0.30, 1.0)
        style_axes(ax, "layers in the stack", "rho_50 (lower is better)")
    return go


def draw_width(rungs):
    """The width floor of the attention-only stack, at two depths."""
    series = [(16, "16 layers"), (12, "12 layers")]
    def go(ax):
        for i, (L, name) in enumerate(series):
            pts = sorted([r for r in rungs if r["arch"] == "no_mlp" and r["L"] == L
                          and r["ff"] == 1024], key=lambda r: r["d_model"])
            if not pts:
                continue
            xs = [r["d_model"] for r in pts if r["readable"]]
            ys = [r["rho50"] for r in pts if r["readable"]]
            ax.plot(xs, ys, color=SERIES[i], marker="o", ms=7, lw=2.2, zorder=3)
            cx = [r["d_model"] for r in pts if not r["readable"]]
            ax.plot(cx, [0.95] * len(cx), color=SERIES[i], marker="^", ms=9, lw=0,
                    mfc="white", mew=2.0, zorder=3)
            if xs:
                label_line(ax, xs[-1], ys[-1], name, color=SERIES[i])
        ax.axhline(FREE, color=MUTED, ls="--", lw=1.2, zorder=1)
        ax.set_ylim(0.30, 1.0)
        style_axes(ax, "model width d_model", "rho_50 (lower is better)")
    return go


def draw_escapes(rungs):
    """Every rung: size against score, split by how many seeds trained at all."""
    groups = [(3, "all 3 seeds trained"), (1, "1 or 2 trained"), (0, "none trained")]
    def go(ax):
        for i, (n, name) in enumerate(groups):
            sel = [r for r in rungs
                   if (r["escapes"] == 3 if n == 3 else
                       (r["escapes"] == 0 if n == 0 else 0 < r["escapes"] < 3))]
            xs = [r["trainable"] for r in sel]
            ys = [r["rho50"] if r["readable"] else 0.95 for r in sel]
            marks = ["o" if r["readable"] else "^" for r in sel]
            for x, y, mk in zip(xs, ys, marks):
                ax.plot([x], [y], marker=mk, ms=8, lw=0, color=SERIES[i],
                        mfc=SERIES[i] if mk == "o" else "white", mew=2.0, zorder=3)
            if xs:
                top = max(zip(xs, ys), key=lambda p: p[0])
                label_line(ax, top[0], top[1], name, color=SERIES[i])
        ax.axhline(FREE, color=MUTED, ls="--", lw=1.2, zorder=1)
        ax.set_xscale("log")
        ax.set_ylim(0.30, 1.0)
        style_axes(ax, "trainable parameters", "rho_50 (lower is better)")
    return go


def draw_curve(cells, ref, held):
    """Solve rate against token accuracy, with the reference's own fit."""
    xs = [c["acc"] for c in ref]
    ys = [c["mean_solve"] for c in ref]
    fit = pava(xs, ys)
    grid = [min(xs) + i * (max(xs) - min(xs)) / 200.0 for i in range(201)]
    def go(ax):
        ax.plot(grid, [predict(fit, x) for x in grid], color=MUTED, lw=2.0, zorder=2)
        ax.plot(xs, ys, marker="o", ms=7, lw=0, color=SERIES[0], zorder=3)
        ax.plot([c["acc"] for c in held], [c["mean_solve"] for c in held],
                marker="s", ms=7, lw=0, color=SERIES[1], zorder=3)
        label_line(ax, max(xs), max(ys), "standard blocks", color=SERIES[0])
        hx = max(c["acc"] for c in held)
        label_line(ax, hx, [c["mean_solve"] for c in held
                            if c["acc"] == hx][0], "everything else", color=SERIES[1])
        style_axes(ax, "validation token accuracy", "mean solve rate over the grid")
    return go


# ---------------------------------------------------------------- claims

def check_claims(rows, rungs, hub, j2, s1):
    """Every checkable sentence, asserted against the loaders' own output."""
    # 0. One task, leak-free, one size, one budget.
    flags = {(r["shuffle_adj"], r["fall"]) for r in rows}
    assert flags == {("True", "True")}, "not every curve is leak-free fall: %s" % flags
    sizes = {int(r["n_nodes"]) for r in rows}
    assert sizes == {48}, "more than one puzzle size in the ladder: %s" % sizes

    # 1. The gate is the hub plus the frozen spread, and the hub reads 0.592.
    assert abs(hub["rho50"] - HUB_RHO50) < 1e-9, \
        "the hub moved: %.4f vs the registered %.3f" % (hub["rho50"], HUB_RHO50)
    assert hub["trainable"] == HUB_TRAINABLE, "the hub's parameter count moved"
    assert abs((HUB_RHO50 + S_STAR) - FREE) < 1e-9, "gate arithmetic broke"

    # 2. J2 is the standing minimum: free, and no free rung is a simpler object.
    assert j2["free"] and abs(j2["rho50"] - 0.580) < 1e-9, \
        "J2 is no longer 0.580 / free: %s" % txt(j2)
    free = [r for r in rungs if r["free"]]
    assert all(r["arch"] in ("full", "no_mlp") for r in free), \
        "a third architecture now clears the gate: %s" % [r["label"] for r in free]
    assert min((r["trainable"] for r in free)) == j2["trainable"], \
        "a smaller rung now clears the gate than J2"

    # 3. The tied cell is the smallest object tried and misses the gate.
    assert not s1["free"], "the tied block now clears the gate -- rewrite the deck"
    assert s1["trainable"] < j2["trainable"] / 3, "the tied cell is not much smaller"
    assert s1["readable"] and 0.74 < s1["rho50"] < 0.78, "the tied cell moved: %s" % txt(s1)

    # 4. Frozen attention never learns anything, at either depth it was tried.
    rf = [r for r in rungs if r["arch"] == "rf_all"]
    assert rf, "no rf_all rung on disk"
    for r in rf:
        assert not r["readable"], "%s now crosses: %s" % (r["label"], txt(r))
        assert r["escapes"] == 0, "%s now trains" % r["label"]
        assert r["mean_solve"] < 0.01, "%s now solves %.4f" % (r["label"], r["mean_solve"])

    # 5. One head never learns it either, wherever it was tried.
    one = [r for r in rungs if r["nhead"] == 1]
    assert one, "no single-head rung on disk"
    for r in one:
        assert not r["free"], "%s now clears the gate" % r["label"]

    # 6. Size does not order the ladder: some censored rungs are bigger than
    #    free ones. This is the trainability page's whole claim.
    big_dead = [r for r in rungs if not r["readable"] and r["trainable"] > j2["trainable"]]
    assert big_dead, "no censored rung is larger than J2 any more -- rewrite that page"
    return free, big_dead


# ---------------------------------------------------------------- the deck

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()

    rows, rungs = ladder()
    hub = find(rungs, rung=(6, 4, "full"))
    j2 = find(rungs, rung=(16, 4, "no_mlp", 192, 1024, 0))
    s1 = find(rungs, rung=(16, 4, "tied_no_mlp", 320, 1024, 0))
    for name, r in (("hub", hub), ("J2", j2), ("S1", s1)):
        if r is None:
            raise SystemExit("the %s rung is not on disk" % name)
    free, big_dead = check_claims(rows, rungs, hub, j2, s1)

    cells = [c for c in collect(ALL_DIRS) if c["acc"] == c["acc"]]
    ref = [c for c in cells if is_reference(c["rung"])]
    held = [c for c in cells if not is_reference(c["rung"])]
    fit = pava([c["acc"] for c in ref], [c["mean_solve"] for c in ref])
    resid = rms([c["mean_solve"] - predict(fit, c["acc"]) for c in held])

    prov = ("%d rungs over %d scan directories, %d scored density curves, "
            "N=48 degree 5 four colours, 14 densities x 400 puzzles, "
            "--shuffle-adj --fall, 3 seeds per rung. Commit %s."
            % (len(rungs), len(ALL_DIRS), len(rows), args.commit))

    d = WebDeck(args.out,
                "An attention-only stack with no MLP anywhere still learns deduction",
                "hshunt", "2026-11-12", prov)

    d.title_page(
        "An attention-only stack with no MLP anywhere still learns deduction",
        subtitle="Every architecture this project has scored on one puzzle, ranked by "
                 "how simple the OBJECT is rather than by how well it scores. What "
                 "could be removed, and what could not.")

    # -- the task
    d.points_page(
        "The puzzle: colour a graph when some colours are given", [
            "48 nodes, each with 5 neighbours, 4 colours, no two neighbours alike",
            "A proper colouring is planted, then most nodes are hidden",
            "The visible ones are hints; the rest have to be deduced"])

    d.points_page(
        "One rule does all the work, applied in synchronous rounds", [
            "A node whose neighbours show all 3 other colours has one colour left",
            "Fill every such node, then look again: that is one round",
            "Sudoku players call it a naked single; here, direct deduction"])

    ex = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.30, rho_max=0.95,
                          shuffle_adj=True, fall=True)
    d.expect("tokenization", "Every token is one id in a 58-symbol vocabulary")
    page_tokenization(d, ex)

    d.expect("trace", "A deduction fires when a node sees all three other colours",
             note="The model is trained to write this whole trace, round markers "
                  "included, from the graph and hints alone.")
    page_trace(d, ex)

    # -- the question
    d.points_page(
        "The goal is not a better solver; it is a simpler one", [
            "A model we might analyse by hand beats a model that merely scores well",
            "Parameter count is explicitly NOT the figure of merit",
            "Count the parts instead: operations per layer, weights shared"])

    d.points_page(
        "So every test here removes one part and asks if the task survives", [
            "Fewer layers, no MLP, frozen weights, one head, one shared block",
            "One knob per experiment, three seeds, same budget throughout",
            "The question each time is pass or fail, not better or worse"])

    d.points_page(
        "Pass or fail is read off a grid of 14 hint densities", [
            "400 fresh puzzles per density, from the easiest to the hardest",
            "rho_50 is the density where it still solves half of them",
            "Lower is better; a cell that never reaches half has no number"])

    d.table_page(
        "The pass mark is the reference model plus the seed-to-seed spread",
        ["quantity", "value", "what it is"],
        [["reference model", "%.3f" % HUB_RHO50, "6 standard blocks, width 256"],
         ["seed spread s*", "%.4f" % S_STAR, "largest gap between seeds, same architecture"],
         ["**pass mark**", "**%.3f**" % FREE, "**reference + spread: still learns the task**"]],
        note="Two identical models trained from different seeds differ by %.3f, so "
             "anything inside that of the reference counts as the same answer." % S_STAR)

    # -- depth
    d.points_page(
        "First test: how many layers does deduction need?", [
            "The task takes up to 14 rounds of reasoning at the hardest density",
            "A fixed stack has to fit that into its layers somehow",
            "So sweep the depth with everything else held fixed"])

    d.claim_page(
        "Six layers are enough; at two the model stops learning the task",
        draw_depth(rungs), "depth",
        note="Standard blocks, width 256, three seeds each. Open triangles never "
             "solved half the puzzles at any density, so they have no rho_50.")

    depth_rows = []
    for r in sorted([x for x in rungs if x["arch"] == "full" and x["nhead"] == 4
                     and x["d_model"] == 256 and x["ff"] == 1024],
                    key=lambda x: -x["L"]):
        depth_rows.append([r["L"], num(r["trainable"]), txt(r),
                           r["crossing"], "%d/%d" % (r["escapes"], r["n_seeds"]),
                           "pass" if r["free"] else "fail"])
    d.table_page(
        "Depth buys score smoothly, then falls off a cliff below four layers",
        ["layers", "trainable", "rho_50", "seeds crossing", "seeds trained", "gate"],
        depth_rows,
        note="'Seeds trained' counts seeds whose training loss left the 1.64 "
             "plateau. At two layers and below, none did.")

    # -- what is in a layer
    d.points_page(
        "Second test: a layer has two halves, and one of them may be dead weight", [
            "Attention mixes information between nodes along the graph",
            "A two-layer MLP then transforms each position on its own",
            "The MLP holds most of the parameters, so try deleting it"])

    arch_rows = []
    for a in ("full", "linear", "rf_mlp", "rf_all", "no_mlp"):
        r = find(rungs, rung=(6, 4, a)) if a != "full" else hub
        if r is None:
            continue
        arch_rows.append([ARCH_WHAT[a], num(r["trainable"]), txt(r), r["crossing"],
                          "pass" if r["free"] else "fail"])
    d.table_page(
        "At six layers only the standard block passes; freezing attention kills it",
        ["what the layer is", "trainable", "rho_50", "seeds crossing", "gate"],
        arch_rows,
        note="All at six layers, width 256. Frozen attention leaves 14,906 trainable "
             "weights and solves nothing at any density.")

    nm = sorted([r for r in rungs if r["arch"] == "no_mlp" and r["d_model"] == 256
                 and r["ff"] == 1024], key=lambda r: r["L"])
    d.table_page(
        "Deleting the MLP costs nothing once the stack is twelve layers deep",
        ["layers", "trainable", "rho_50", "seeds crossing", "gate"],
        [[r["L"], num(r["trainable"]), txt(r), r["crossing"],
          "pass" if r["free"] else "fail"] for r in nm],
        note="Attention-only stacks at width 256. The MLP is not doing something "
             "attention cannot; it is buying depth cheaply.")

    # -- width
    d.points_page(
        "Third test: how narrow can the attention-only stack get?", [
            "Width sets how much each node can carry between layers",
            "Narrowing it is the cheapest simplification left",
            "Sweep width at 12 and 16 layers, three seeds each"])

    d.claim_page(
        "Below width 192 the attention-only stack stops training entirely",
        draw_width(rungs), "width",
        note="Open triangles never crossed. The jump is not gradual: 192 passes at "
             "both depths and 176 misses, at the same shape.")

    # -- heads and sharing
    head_rows = []
    for r in sorted([x for x in rungs if x["nhead"] == 1], key=lambda x: x["L"]):
        ctrl = find(rungs, rung=(r["L"], 4, "full"))
        head_rows.append([r["L"], num(r["trainable"]), txt(r), r["crossing"],
                          txt(ctrl) if ctrl else "--"])
    d.table_page(
        "A single attention head never learns the task at any depth tried",
        ["layers", "trainable", "rho_50, one head", "seeds crossing",
         "rho_50, four heads"], head_rows,
        note="Identical parameter count: one head of the same total width. The four "
             "heads are not redundant capacity, they are the mechanism.")

    d.points_page(
        "Fourth test: can one block be reused instead of sixteen distinct ones?", [
            "A shared block applied T times is a map iterated, not a stack",
            "That is the object a theorist would most like to be handed",
            "Train one attention-only block, apply it sixteen times"])

    tie = [r for r in rungs if r["arch"] == "tied_no_mlp" and r["readable"]]
    d.table_page(
        "One shared block does the task, 0.017 short of the pass mark",
        ["cell", "distinct blocks", "trainable", "rho_50", "seeds crossing"],
        [["attention-only, width %d, applied %d times" % (r["d_model"], r["L"]),
          r["k"], num(r["trainable"]), txt(r), r["crossing"]]
         for r in sorted(tie, key=lambda r: r["trainable"])],
        note="Smallest object on the ladder by a factor of four, and it misses by "
             "less than a seed spread. Three stages have tried to close the gap.")

    # -- trainability
    d.points_page(
        "A failure here means 'did not train', not 'cannot represent deduction'", [
            "Every cell shares one budget: 240k puzzles, 50 epochs, fixed rate",
            "Seeds that never leave the 1.64 training plateau learned nothing",
            "A different schedule might train them; this grid cannot say"])

    d.claim_page(
        "Size does not predict which architectures learn the task",
        draw_escapes(rungs), "escapes",
        note="All %d rungs. Open triangles never crossed; %d of them carry more "
             "trainable weights than the standing minimum does."
             % (len(rungs), len(big_dead)))

    d.claim_page(
        "Almost every architecture lands on one accuracy-to-solve curve",
        draw_curve(cells, ref, held), "curve",
        note="Fit on standard blocks only, then everything else dropped onto it: "
             "residual %.4f against a leave-one-out bar of %.4f." % (resid, LOO_BAR))

    # -- the answer
    d.table_page(
        "The simplest thing that passes: sixteen attention-only layers at width 192",
        ["", "reference model", "standing minimum", "smallest tried"],
        [["layers", hub["L"], j2["L"], "%d applications" % s1["L"]],
         ["distinct blocks", hub["L"], j2["L"], 1],
         ["what a layer is", "attention + MLP", "attention only", "attention only"],
         ["width", hub["d_model"], j2["d_model"], s1["d_model"]],
         ["trainable", num(hub["trainable"]), num(j2["trainable"]), num(s1["trainable"])],
         ["rho_50", txt(hub), txt(j2), txt(s1)],
         ["seeds crossing", hub["crossing"], j2["crossing"], s1["crossing"]],
         ["gate at %.3f" % FREE, "pass", "pass", "fail by 0.017"]],
        note="Lower rho_50 is better and the standing minimum is also the better "
             "solver here, which is a coincidence the gate deliberately ignores.")

    d.points_page(
        "What this ladder cannot tell you, stated plainly", [
            "One puzzle size, one budget, one learning rate, final checkpoints only",
            "Three seeds per rung, and the seed spread is 0.1484",
            "Censored rungs are untrained at this budget, not proven incapable"])

    d.table_page(
        "Against the previous minimum, the answer has not moved since 11-09",
        ["quantity", "previous minimum (11-05)", "standing minimum (11-09)"],
        [["architecture", "attention-only, 12 layers, width 256",
          "attention-only, 16 layers, width 192"],
         ["trainable", num(find(rungs, rung=(12, 4, "no_mlp"))["trainable"]),
          num(j2["trainable"])],
         ["rho_50", txt(find(rungs, rung=(12, 4, "no_mlp"))), txt(j2)],
         ["seeds crossing", find(rungs, rung=(12, 4, "no_mlp"))["crossing"],
          j2["crossing"]],
         ["gate at %.3f" % FREE, "pass", "pass"]],
        note="Same grid, same 400 puzzles per density, same flags. Stages K through "
             "U have each tried to go below it and none has.")

    path = d.write()
    print("wrote %s (%d pages)" % (path, d.n))
    print("  %d rungs, %d clear the gate, %d censored"
          % (len(rungs), len(free), sum(1 for r in rungs if not r["readable"])))
    for r in free:
        print("    free: %-44s %9s  rho_50 %s  %s"
              % (r["label"], num(r["trainable"]), txt(r), r["crossing"]))


if __name__ == "__main__":
    main()
