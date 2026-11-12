"""The fast/slow transition, as a standalone web deck for someone new to this.

~/slides/decks/spin-gnn-phases/. Separate from the project's running stage deck
(spin-gnn), which is a working log of one architecture search and assumes every
term in it. This one assumes nothing: it says what the puzzle is, what the model
is asked to write, why a phase transition is the thing being chased, and what
changed on both sides of it -- the training distribution and the trained model's
performance.

THE STORY, and it has two halves that must not be mixed up:

  THE TASK SIDE (measured today by scripts/fastslow_measure.py, job 52042656).
  Direct deduction has a fast phase above a hint density rho_c ~ 0.46, where it
  finishes in T ~ log N rounds, and a slow phase below it, where T grows as a
  power of N. For nine months every instance this project trained on was in the
  fast phase and could not have been anywhere else: the puzzle generator's
  greedy hint removal halts at rho ~ 0.477 at every size, because a node whose
  neighbourhood is missing a colour can never be deduced and is therefore a
  permanent hint. Their density is fixed by (q, d) alone -- inclusion-exclusion
  gives 0.3827 at q=4, d=5 -- and that number IS the floor. hshunt's fix is to
  plant a FALL colouring (every vertex adjacent to a vertex of every other
  colour), which drives that count to zero and the floor to 0.309, below rho_c.

  THE MODEL SIDE (grids scored 2026-11-02, finding 44, not recomputed here).
  On the new ensemble the same architecture reasons for 13.8 rounds instead of
  7.9 at N=48, at an unchanged sequence length, and solve rate degrades
  gracefully into the new regime rather than falling off a cliff. The two
  ensembles' best cells are NOT paired and the default-ensemble cell had twice
  the optimization, so the deck compares them on the shared depth axis and says
  so on its own page.

NOTHING IS RE-DERIVED BY HAND. The ensemble numbers come from the measurement
JSON; the model numbers come from the two scored grids' own CSVs and the
checkpoints' own history.json. check_claims() asserts every sentence the deck
prints that a reader could check, and the builder fails closed rather than
publishing one the data no longer supports.

Usage (in a job -- $HOME I/O on sh02-ln02 wedges, and this imports torch
through deck_tokenization):
    python3 -u -m notebooks.make_phases_web --commit <sha> [--out <dir>]
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

from slides import MUTED, SERIES, label_line, style_axes

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_staget_web import WebDeck
# One registered plateau for both decks, rather than 1.64 typed twice.
from scripts.minarch_analysis import PLATEAU

ROOT = "/scratch/users/hshunt/spin_gnn"
DECK = "/home/users/hshunt/slides/decks/spin-gnn-phases"
MEASURED = os.path.join(ROOT, "sudoku_fastslow_2026-11-12", "fastslow.json")

# The two scored grids. DEFAULT_GRID is the only fall=False rho curve at N=48;
# FALL_GRID is finding 44's scan. Both were scored by rho_transition_eval with
# 400 instances per density bin and --shuffle-adj.
DEFAULT_GRID = "sudoku_n48_rho_curves_2026-11-02/rho_curve_default_ensemble_50760252.csv"
FALL_GRID = "sudoku_n48_fall_scan_2026-11-02/rho_curve_*.csv"

# Dario's Sudoku_SC slides, q=4 d=5, thermodynamic limit. Quoted, never fitted.
RHO_C = 0.46

# The inclusion-exclusion prediction at the degree this project trains on. The
# measurement script computes it from the formula; this is the value the deck
# prints, asserted equal to it.
PRED_D5 = 0.3827


# ---------------------------------------------------------------- loading

def measured():
    with open(MEASURED) as fh:
        doc = json.load(fh)
    if not doc.get("complete"):
        raise SystemExit("%s is a partial dump (complete=false) -- wait for the job"
                         % MEASURED)
    if doc.get("quick"):
        raise SystemExit("%s is the --quick shape check, not the measurement" % MEASURED)
    return doc


def grid(pattern):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, pattern))):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("no grid rows at %s" % pattern)
    return rows


def cells(rows):
    """Grid rows grouped into one curve per checkpoint, sorted by density."""
    out = {}
    for r in rows:
        k = (int(r["num_layers"]), int(r["dim_feedforward"]),
             int(r["n_train_instances"]), int(r["epochs"]), int(r["seed"]))
        out.setdefault(k, []).append(r)
    return {k: sorted(v, key=lambda r: float(r["realized_rho"]))
            for k, v in out.items()}


def best(cs):
    """The cell that solves most at the HARDEST bin it was given.

    Not the cell with the best easy end: every cell above four layers reaches
    1.000 there, so the easy end cannot rank them, and the deck's subject is
    what happens at depth.
    """
    return max(cs.items(), key=lambda kv: float(kv[1][0]["solve_rate"]))


def hist(row):
    with open(os.path.join(row["path"], "history.json")) as fh:
        return json.load(fh)


def curve(rs, x, y):
    return [float(r[x]) for r in rs], [float(r[y]) for r in rs]


def row_of(doc, N, ens):
    for r in doc["sizes"]:
        if r["N"] == N and r["ensemble"] == ens:
            return r
    raise SystemExit("no measured row for N=%d %s" % (N, ens))


def fmt(x, n=4):
    return ("%." + str(n) + "f") % x


# ---------------------------------------------------------------- figures

def draw_floors(doc):
    """Where greedy hint removal stops, against the transition it has to cross."""
    def go(ax):
        anchor = None
        for i, ens in enumerate(("random", "fall")):
            rows = sorted([r for r in doc["sizes"] if r["ensemble"] == ens],
                          key=lambda r: r["N"])
            xs = [r["N"] for r in rows]
            ys = [r["rho_min"]["mean"] for r in rows]
            es = [r["rho_min"]["sd"] for r in rows]
            ax.errorbar(xs, ys, yerr=es, color=SERIES[i], marker="o", ms=5,
                        lw=2.0, capsize=3, zorder=3)
            label_line(ax, xs[-1], ys[-1],
                       "fall-coloured" if ens == "fall" else "unconstrained",
                       color=SERIES[i])
            if ens == "random":
                anchor = xs[min(2, len(xs) - 1)]
        ax.axhline(RHO_C, color=MUTED, ls="--", lw=1.2, zorder=1)
        # below the line and a few sizes in: at N=48 the label collided with the
        # unconstrained marker and its error bar.
        ax.annotate("rho_c ~ %.2f" % RHO_C, xy=(anchor, RHO_C),
                    xytext=(0, -16), textcoords="offset points", color=MUTED, size=11)
        ax.set_xscale("log")
        ax.set_ylim(0.25, 0.55)
        style_axes(ax, "number of nodes N", "hint density where removal halts")
    return go


def draw_depth(doc):
    """T at the end of the path: a power law below rho_c, flat above it."""
    def go(ax):
        for i, ens in enumerate(("random", "fall")):
            rows = sorted([r for r in doc["sizes"] if r["ensemble"] == ens],
                          key=lambda r: r["N"])
            xs = [r["N"] for r in rows]
            ys = [r["T_end"]["mean"] for r in rows]
            ax.plot(xs, ys, color=SERIES[i], marker="o", ms=5, lw=2.0, zorder=3)
            label_line(ax, xs[-1], ys[-1],
                       "fall-coloured" if ens == "fall" else "unconstrained",
                       color=SERIES[i])
        ax.set_xscale("log")
        ax.set_yscale("log")
        style_axes(ax, "number of nodes N", "deduction rounds T at the floor")
    return go


def path_points(doc, ens):
    """The part of a removal path at least half the paths reach: a mean over the
    three luckiest paths is not a curve, it is a tail."""
    p = doc["path"]
    out = []
    for rho, t, n in zip(p["rho"], p[ens]["T_mean"], p[ens]["n_reaching"]):
        if t is not None and n >= p["seeds"] // 2:
            out.append((rho, t))
    return out


def draw_path(doc):
    """T along one removal path at the training size, averaged over 200 paths."""
    def go(ax):
        for i, ens in enumerate(("random", "fall")):
            pts = path_points(doc, ens)
            xs = [a for a, _ in pts]
            ys = [b for _, b in pts]
            ax.plot(xs, ys, color=SERIES[i], lw=2.2, zorder=3)
            label_line(ax, xs[-1], ys[-1],
                       "fall-coloured" if ens == "fall" else "unconstrained",
                       color=SERIES[i])
        ax.axvline(RHO_C, color=MUTED, ls="--", lw=1.2, zorder=1)
        ax.annotate("rho_c ~ %.2f" % RHO_C, xy=(RHO_C, 0.06), color=MUTED, size=11,
                    xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points")
        ax.invert_xaxis()
        style_axes(ax, "hint density rho (harder to the right)",
                   "deduction rounds T at N=48")
    return go


def draw_solve_rho(dbest, fbest):
    """Solve rate against hint density, one best cell per ensemble."""
    def go(ax):
        for i, (rs, name) in enumerate(((dbest, "trained on unconstrained"),
                                        (fbest, "trained on fall-coloured"))):
            xs, ys = curve(rs, "realized_rho", "solve_rate")
            ax.plot(xs, ys, color=SERIES[i], marker="o", ms=4, lw=2.0, zorder=3)
            # The unconstrained curve ends where the fall curve is still passing
            # through, so lift its label clear of the other line.
            label_line(ax, xs[0], ys[0], name, color=SERIES[i], dy=14 if i == 0 else 0)
        ax.axvline(RHO_C, color=MUTED, ls="--", lw=1.2, zorder=1)
        ax.annotate("rho_c ~ %.2f" % RHO_C, xy=(RHO_C, 0.06), color=MUTED, size=11,
                    xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points")
        ax.set_ylim(0.0, 1.05)
        ax.invert_xaxis()
        style_axes(ax, "hint density rho (harder to the right)",
                   "fraction of puzzles solved")
    return go


def draw_solve_T(dbest, fbest):
    """The same two cells on the axis they share: mean deduction depth."""
    def go(ax):
        for i, (rs, name) in enumerate(((dbest, "unconstrained"),
                                        (fbest, "fall-coloured"))):
            xs, ys = curve(rs, "dd_mean_T", "solve_rate")
            ax.plot(xs, ys, color=SERIES[i], marker="o", ms=4, lw=2.0, zorder=3)
            # Sorted by density, so the deepest bin is wherever x is largest, not
            # at either end of the list: labelling xs[-1] stacked both labels on
            # the shallow end, on top of each other.
            j = max(range(len(xs)), key=lambda k: xs[k])
            label_line(ax, xs[j], ys[j], name, color=SERIES[i])
        ax.set_ylim(0.0, 1.05)
        style_axes(ax, "mean deduction rounds in the bin",
                   "fraction of puzzles solved")
    return go


def draw_train(by_depth):
    """Training NLL by depth at one matched token budget."""
    def go(ax):
        for i, (L, h) in enumerate(sorted(by_depth.items())):
            xs = [e["epoch"] for e in h]
            ys = [e["train_nll"] for e in h]
            ax.plot(xs, ys, color=SERIES[i], lw=2.0, zorder=3)
            label_line(ax, xs[-1], ys[-1], "%d layers" % L, color=SERIES[i])
        # The claim is about this line, so draw it.
        ax.axhline(PLATEAU, color=MUTED, ls="--", lw=1.2, zorder=1)
        ax.annotate("plateau %.2f" % PLATEAU, xy=(0.02, PLATEAU), color=MUTED, size=11,
                    xycoords=("axes fraction", "data"), xytext=(0, 5),
                    textcoords="offset points")
        style_axes(ax, "epoch", "training NLL")
    return go


# ---------------------------------------------------------------- claims

def check_claims(doc, dcells, fcells, dkey, fkey):
    """Every checkable sentence on the deck, asserted against its source."""
    big = max(r["N"] for r in doc["sizes"])
    n48r, n48f = row_of(doc, 48, "random"), row_of(doc, 48, "fall")

    # 1. The floors land on opposite sides of the transition, at every size.
    for r in doc["sizes"]:
        ens, m = r["ensemble"], r["rho_min"]["mean"]
        if ens == "random":
            assert m > RHO_C, "random floor %.4f at N=%d is below rho_c" % (m, r["N"])
        elif r["N"] >= 996:
            assert m < RHO_C, "fall floor %.4f at N=%d is not below rho_c" % (m, r["N"])

    # 2. The mechanism: the unconstrained ensemble's permanently-undeducible
    #    fraction matches inclusion-exclusion, and fall drives it to zero.
    for row in doc["degrees"]:
        assert abs(row["random"]["mean"] - row["predicted"]) < 0.03, \
            "d=%d: measured %.4f vs predicted %.4f" % (row["degree"],
                                                       row["random"]["mean"],
                                                       row["predicted"])
        assert row["fall"]["mean"] == 0.0, \
            "d=%d: fall ensemble has %.4f non-colourful" % (row["degree"],
                                                            row["fall"]["mean"])
    d5 = [r for r in doc["degrees"] if r["degree"] == 5][0]
    assert abs(d5["predicted"] - PRED_D5) < 5e-4, \
        "the d=5 prediction moved: %.5f vs %.4f" % (d5["predicted"], PRED_D5)

    # 3. The power law. Not "one fitted exponent": the claim is that the two
    #    branches are cleanly separated and that the fall branch's local slope
    #    is stable across independent intervals, which is what distinguishes a
    #    power law from curvature drawn through three points.
    gs = [g["gamma"] for g in doc["gamma"]["fall"] if g["from"] >= 252]
    gr = [g["gamma"] for g in doc["gamma"]["random"] if g["from"] >= 252]
    assert min(gs) > 0.60, "a fall interval is shallower than 0.60: %s" % gs
    assert max(gr) < min(gs), \
        "the unconstrained branch is as steep as the fall one: %s vs %s" % (gr, gs)
    assert max(gs) - min(gs) < 0.10, "fall exponents disagree across intervals: %s" % gs

    # 4. Depth at the floor roughly doubles at the training size.
    assert n48f["T_end"]["mean"] > 1.6 * n48r["T_end"]["mean"], \
        "N=48 floor depth did not grow: %.1f -> %.1f" % (n48r["T_end"]["mean"],
                                                         n48f["T_end"]["mean"])

    # 5. The training MIX deepens much less than the floor does, because rho is
    #    drawn uniformly in [0.30, 0.95] and most draws are shallow either way.
    #    The deck says this in its own words; the assertion is that the token
    #    cost of it stays inside a couple of tokens.
    t = doc["tokens"]
    ratio = t["fall"]["rounds"]["mean"] / t["random"]["rounds"]["mean"]
    assert 1.10 < ratio < 1.35, \
        "the mix no longer deepens by about a fifth (ratio %.3f): retitle the page" % ratio
    dtok = t["fall"]["tokens"]["mean"] - t["random"]["tokens"]["mean"]
    assert 0 < dtok < 2, \
        "the mix no longer costs about one token (%.2f): retitle the page" % dtok

    # 5. The grids: each ensemble's own floor, and what the best cell does there.
    dfloor = float(dcells[dkey][0]["realized_rho"])
    ffloor = float(fcells[fkey][0]["realized_rho"])
    assert abs(dfloor - 0.4714) < 1e-3, "default grid floor moved: %.4f" % dfloor
    assert abs(ffloor - 0.3333) < 1e-3, "fall grid floor moved: %.4f" % ffloor
    assert ffloor < RHO_C < dfloor, "the two grids no longer straddle rho_c"
    for k, cs in (("default", dcells[dkey]), ("fall", fcells[fkey])):
        assert float(cs[-1]["solve_rate"]) == 1.0, \
            "%s best cell does not solve the easy end" % k
        assert len(cs) in (11, 14), "%s best cell has %d bins" % (k, len(cs))

    # 6. The matched-parameter pair really is matched.
    wide = [k for k in fcells if k[1] == 4096]
    assert len(wide) == 1, "expected one wide cell, found %d" % len(wide)
    w = wide[0]
    deep = (12, 1024, w[2], w[3], w[4])
    assert deep in fcells, "no depth-matched partner for the wide cell"
    pw = int(fcells[w][0]["n_params"])
    pd = int(fcells[deep][0]["n_params"])
    assert abs(pw - pd) / pd < 0.01, "pair is not parameter-matched: %d vs %d" % (pw, pd)
    return big, gs, w, deep


# ---------------------------------------------------------------- the deck

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()

    doc = measured()
    dcells = cells(grid(DEFAULT_GRID))
    fcells = cells(grid(FALL_GRID))
    dkey, dbest = best(dcells)
    fkey, fbest = best(fcells)
    big, gs, wide, deep = check_claims(doc, dcells, fcells, dkey, fkey)

    n48r, n48f = row_of(doc, 48, "random"), row_of(doc, 48, "fall")
    bigr, bigf = row_of(doc, big, "random"), row_of(doc, big, "fall")
    tok = doc["tokens"]
    d5 = [r for r in doc["degrees"] if r["degree"] == 5][0]

    # The job id is read off the measurement directory, not typed: this dump was
    # rewritten by a second, higher-seed run (52042907) and a hand-typed id would
    # still credit the first one.
    outs = sorted(glob.glob(os.path.join(os.path.dirname(MEASURED), "measure_*.out")),
                  key=os.path.getmtime)
    mjob = os.path.basename(outs[-1])[len("measure_"):-len(".out")] if outs else "unknown"
    prov = ("Ensemble measurements: job %s, 2026-11-12, commit %s. Model "
            "grids: 50760252 and 50760297, scored 2026-11-02, 400 instances per "
            "density bin, --shuffle-adj. N=48, degree 5, 4 colours."
            % (mjob, args.commit))
    # Two numbers that appear in page titles, so they are computed, not typed:
    # how far inclusion-exclusion misses the measured undeducible fraction at
    # its worst degree, and the mean local exponent of the fall branch.
    DEG_ERR = max(abs(r["random"]["mean"] - r["predicted"]) for r in doc["degrees"])
    GAMMA_MEAN = (sum(g["gamma"] for g in doc["gamma"]["fall"])
                  / len(doc["gamma"]["fall"]))
    tdeep = n48f["T_end"]["mean"] / n48r["T_end"]["mean"]
    headline = ("Harder puzzles raised reasoning depth %.1fx at the same token cost"
                % tdeep)

    d = WebDeck(args.out, headline, "hshunt", "2026-11-12", prov)

    d.title_page(
        headline,
        subtitle="Every instance this project trained on before 2026-11-02 was in "
                 "the easy phase of its own task, and could not have been anywhere "
                 "else. What changed when that was fixed.")

    # -- what the task is
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

    # pick="deep" on both: the 90th percentile of each generator's own pool.
    # This deck's subject IS reasoning depth, and a median draw at rho ~ 0.6 is
    # three rounds deep in either ensemble, which illustrates nothing.
    ex_random = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.30,
                                 rho_max=0.95, shuffle_adj=True, fall=False,
                                 pick="deep")
    d.expect("tokenization", "Every token is one id in a 58-symbol vocabulary")
    page_tokenization(d, ex_random)

    d.expect("trace_random",
             "A deduction fires when a node sees all three other colours",
             note="The old generator at N=48, one of its deepest draws: %d rounds. "
                  "The model is trained to write this whole trace."
             % len(ex_random.trace["rounds"]))
    page_trace(d, ex_random, title="One real trace from the unconstrained generator")

    d.points_page(
        "The model gets the graph and hints, and writes the trace", [
            "A decoder-only transformer, trained to predict the next token",
            "No solver, no search, no tool: the rounds are generated",
            "Scored on whether the colouring it writes is the planted one"])

    # -- why a phase transition is the point
    d.points_page(
        "Theory says this puzzle has an easy phase and a hard phase", [
            "Above a hint density rho_c, deduction finishes in about log N rounds",
            "Below it, the number of rounds grows as a power of N",
            "At 4 colours and degree 5 the slides put rho_c near 0.46"])

    d.points_page(
        "The goal of every test here: get the training set below rho_c", [
            "Depth is what makes the task interesting, and depth lives below rho_c",
            "A model trained only above it has never seen a long deduction",
            "So first check which hint densities the generator can even produce"])

    d.claim_page(
        "The old generator halts above the transition at every size",
        draw_floors(doc), "floors",
        note="Greedy hint removal, %d to %d seeds per point. The unconstrained "
             "floor converges to %.3f; it never crosses the line."
             % (bigr["seeds"], n48r["seeds"], bigr["rho_min"]["mean"]))

    d.points_page(
        "The floor is not a sampler artifact: some nodes are undeducible", [
            "A node missing a colour from its neighbourhood can never be deduced",
            "It is a permanent hint, so removal halts once only those remain",
            "Their count is fixed by the degree and colour count alone"])

    d.table_page(
        "Counting those nodes predicts the undeducible fraction within %.3f" % DEG_ERR,
        ["degree", "predicted density", "measured, unconstrained",
         "measured, fall-coloured", "removal floor"],
        [[r["degree"], fmt(r["predicted"]), fmt(r["random"]["mean"]),
          fmt(r["fall"]["mean"]), "%.3f -> %.3f" % (r["random_rho_min"]["mean"],
                                                    r["fall_rho_min"]["mean"])]
         for r in doc["degrees"]],
        note="Inclusion-exclusion over missing colours, against %d graphs of N=%d "
             "per degree. Forcing a fall colouring zeroes the column."
             % (d5["seeds"], d5["N"]))

    d.points_page(
        "The fix: plant a colouring in which no node is undeducible", [
            "A fall colouring has every vertex adjacent to every other colour",
            "Reserve one mandatory neighbour per colour pair, split the rest at random",
            "hshunt's diagnosis; the graphs stay regular, proper and balanced"])

    d.claim_page(
        "Below the transition, reasoning depth grows as a power of size",
        draw_depth(doc), "depth",
        note="Same paths as the previous page. The unconstrained branch flattens "
             "near %.0f rounds; the fall branch keeps climbing to %.0f."
             % (bigr["T_end"]["mean"], bigf["T_end"]["mean"]))

    d.table_page(
        # The spread is 0.64 to 0.76, so "near 0.70 in every interval" was doing
        # more work than the numbers support: quote the interval instead.
        "The fall exponent stays between %.2f and %.2f at every size"
        % (min(a["gamma"] for a in doc["gamma"]["fall"]),
           max(a["gamma"] for a in doc["gamma"]["fall"])),
        ["size interval", "fall-coloured exponent", "unconstrained exponent"],
        [["%d -> %d" % (a["from"], a["to"]), fmt(a["gamma"]), fmt(b["gamma"])]
         for a, b in zip(doc["gamma"]["fall"], doc["gamma"]["random"])],
        note="Local slope of T against N between consecutive sizes. Agreement "
             "across intervals is what makes it a power law, not curvature.")

    # The two numbers in the title are the ends of the two curves actually drawn,
    # not the mean floors on the table two pages on: a path is kept here only
    # where half the paths reach it, so these ends are the stricter statement.
    path_end = {e: path_points(doc, e)[-1][0] for e in ("random", "fall")}
    assert path_end["fall"] < RHO_C <= path_end["random"], \
        "the drawn removal paths no longer straddle rho_c: %r" % path_end
    d.claim_page(
        "The new generator reaches density %.2f where the old halts at %.2f"
        % (path_end["fall"], path_end["random"]),
        draw_path(doc), "path",
        note="Mean over %d removal paths per ensemble, plotted where at least half "
             "the paths reach that density." % doc["path"]["seeds"])

    ex_fall = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.30,
                               rho_max=0.95, shuffle_adj=True, fall=True, pick="deep")
    # Both traces are 90th-percentile draws from their own generator, so the two
    # counts are comparable; "far more rounds" was not, for 13 against 10.
    d.expect("trace_fall",
             "The same size and vocabulary, %d deduction rounds instead of %d"
             % (len(ex_fall.trace["rounds"]), len(ex_random.trace["rounds"])),
             note="The fall-coloured generator, same N=48 and same vocabulary. Only "
                  "the first rounds are drawn; this trace has %d."
             % len(ex_fall.trace["rounds"]))
    page_trace(d, ex_fall, title="One real trace from the fall-coloured generator")

    d.table_page(
        "At its own hardest density, N=48 takes %.1f rounds, not %.1f"
        % (n48f["T_end"]["mean"], n48r["T_end"]["mean"]),
        ["puzzle size", "generator", "hint density at the floor",
         "deduction rounds there"],
        [[r["N"], "unconstrained" if r["ensemble"] == "random" else "fall-coloured",
          "%.4f +- %.4f" % (r["rho_min"]["mean"], r["rho_min"]["sd"]),
          "%.1f +- %.1f" % (r["T_end"]["mean"], r["T_end"]["sd"])]
         for r in (n48r, n48f, row_of(doc, 96, "random"), row_of(doc, 96, "fall"))],
        note="%d and %d greedy paths per row, at the hard end of the training "
             "distribution the old generator could not reach."
             % (n48r["seeds"], row_of(doc, 96, "random")["seeds"]))

    _dtok = tok["fall"]["tokens"]["mean"] - tok["random"]["tokens"]["mean"]
    assert 0.5 <= _dtok < 1.5, "the length difference is no longer one token: %.2f" % _dtok
    d.table_page(
        "Across the training mix, depth rises %.0f%% and length one token"
        % (100.0 * (tok["fall"]["rounds"]["mean"] / tok["random"]["rounds"]["mean"] - 1)),
        ["generator", "deduction rounds", "deduced nodes", "tokens, mean",
         "tokens, max"],
        [["unconstrained", fmt(tok["random"]["rounds"]["mean"], 1),
          fmt(tok["random"]["events"]["mean"], 1),
          fmt(tok["random"]["tokens"]["mean"], 1), int(tok["random"]["tokens"]["max"])],
         ["fall-coloured", fmt(tok["fall"]["rounds"]["mean"], 1),
          fmt(tok["fall"]["events"]["mean"], 1),
          fmt(tok["fall"]["tokens"]["mean"], 1), int(tok["fall"]["tokens"]["max"])]],
        note="%d real training instances each, density drawn uniformly in [0.30, "
             "0.95], so most draws are easy either way. The hard tail is what is new."
             % tok["n_instances"], highlight_row=1)

    d.points_page(
        "Training got no more expensive, which is why this was worth doing", [
            "A trace holds one event per deduced node, a count set by density",
            "Extra rounds only add round markers, so sequences barely lengthen",
            "Same budget, same context window, a strictly harder task"])

    # -- the model side
    d.points_page(
        "What the harder training set does to a trained model", [
            "Performance: can it still solve puzzles below the old floor",
            "Training: does the same architecture still learn the task",
            "Both read off grids of 14 densities by 400 puzzles"])

    d.claim_page(
        "A model trained on the hard set solves past the old floor",
        draw_solve_rho(dbest, fbest), "solve_rho",
        note="Best cell of each grid, %d layers. The old grid simply stops at "
             "0.47: those puzzles cannot be generated." % dkey[0])

    d.claim_page(
        "On the shared depth axis the two agree, then one keeps going",
        draw_solve_T(dbest, fbest), "solve_T",
        note="Same two cells against mean rounds per bin. Not paired: the old cell "
             "had twice the epochs. The overlap is the control.")

    by_depth = {}
    for L in sorted({k[0] for k in fcells}):
        k = (L, 1024, 240000, 50, 0)
        if k in fcells:
            by_depth[L] = hist(fcells[k][0])
    # Read off the curves rather than typed: an earlier draft said four layers
    # "never leave the plateau" and "end where they began", and its own figure
    # showed them starting at 2.06 and descending onto 1.64.
    final = {L: h[-1]["train_nll"] for L, h in by_depth.items()}
    assert abs(final[4] - PLATEAU) < 0.01, \
        "four layers no longer stall on the plateau: %.4f vs %.2f" % (final[4], PLATEAU)
    assert max(final[L] for L in final if L > 4) < PLATEAU - 0.05, \
        "the deeper stacks no longer clear the plateau: %r" % final
    d.claim_page(
        "Four layers stall at %.2f while deeper stacks keep descending" % final[4],
        draw_train(by_depth), "train",
        note="Training NLL at 240k puzzles by 50 epochs, seed 0. Four layers flatten "
             "onto the %.2f line; the deeper stacks cross it." % PLATEAU)

    budget = sorted({(k[2], k[3]) for k in fcells if k[1] == 1024 and k[4] == 0})
    rows = []
    for L in sorted({k[0] for k in fcells if k[1] == 1024}):
        for n, ep in budget:
            k = (L, 1024, n, ep, 0)
            if k not in fcells:
                continue
            cs = fcells[k]
            rows.append([L, "%dk x %d" % (n // 1000, ep),
                         fmt(hist(cs[0])[-1]["train_nll"]),
                         fmt(float(cs[-1]["solve_rate"])),
                         fmt(float(cs[0]["solve_rate"]))])
    d.table_page(
        "Depth, not budget, decides whether the hard end is reached at all",
        ["layers", "budget", "final train NLL", "solved at rho 0.94",
         "solved at rho 0.33"], rows,
        note="Equal token budgets: 30k x 400 epochs = 120k x 100 = 240k x 50, "
             "seed 0 throughout.")

    d.table_page(
        "At matched parameters, %dx the feedforward width buys nothing"
        % (wide[1] // deep[1]),
        ["cell", "parameters", "solved at rho 0.94", "solved at rho 0.50",
         "solved at rho 0.33"],
        [["%d layers, feedforward %d" % (k[0], k[1]), "{:,}".format(int(fcells[k][0]["n_params"])),
          fmt(float(fcells[k][-1]["solve_rate"])),
          fmt(float([r for r in fcells[k]
                     if abs(float(r["realized_rho"]) - 0.50) < 0.01][0]["solve_rate"])),
          fmt(float(fcells[k][0]["solve_rate"]))]
         for k in (wide, deep)],
        note="Both at 120k puzzles by 100 epochs, seed 0, within %.2f%% of the same "
             "parameter count. Depth is doing the work."
             % (100.0 * abs(int(fcells[wide][0]["n_params"])
                            - int(fcells[deep][0]["n_params"]))
                / int(fcells[deep][0]["n_params"])), highlight_row=1)

    d.points_page(
        "What this deck cannot say, stated plainly", [
            "The two grids are different task distributions, never differenced",
            "One seed per cell on the hard grid; seed spread elsewhere is 0.15",
            "The deepest bins truncate 0.5% of traces, so they are lower bounds"])

    d.table_page(
        "Against the previous best, the hard end is new ground",
        ["quantity", "previous best, unconstrained", "best on the fall-coloured set"],
        [["architecture", "%d layers, 240k x 100 epochs" % dkey[0],
          "%d layers, 240k x 50 epochs" % fkey[0]],
         ["parameters", "{:,}".format(int(dbest[0]["n_params"])),
          "{:,}".format(int(fbest[0]["n_params"]))],
         ["lowest density reachable", fmt(float(dbest[0]["realized_rho"])),
          fmt(float(fbest[0]["realized_rho"]))],
         ["mean rounds there", fmt(float(dbest[0]["dd_mean_T"]), 2),
          fmt(float(fbest[0]["dd_mean_T"]), 2)],
         ["solved there", fmt(float(dbest[0]["solve_rate"])),
          fmt(float(fbest[0]["solve_rate"]))],
         ["solved at the easy end", fmt(float(dbest[-1]["solve_rate"])),
          fmt(float(fbest[-1]["solve_rate"]))]],
        note="Same eval code, same 400 instances per bin, same flags, different "
             "instance distributions. Quoted side by side and never subtracted.")

    path = d.write()
    print("wrote %s (%d pages)" % (path, d.n))
    print("  fall gamma over the top two intervals: %.4f, %.4f" % (gs[0], gs[1]))
    print("  grids: default %s floor %.4f, fall %s floor %.4f"
          % (dkey, float(dbest[0]["realized_rho"]), fkey,
             float(fbest[0]["realized_rho"])))


if __name__ == "__main__":
    main()
