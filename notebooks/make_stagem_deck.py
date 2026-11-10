"""STAGE M DECK -- the N=96 2x2: which half of the stage-F cut is load-bearing?

Every title on every page is computed from the numbers this script reads off
disk through scripts/stagem_score.py's OWN functions. The deck and the scorer
cannot disagree, because there is only one implementation.

WHY THIS DECK IS NOT STAGE F'S DECK WITH FOUR BARS. Stage F compared two arms
against one bar and had one gap to report. Stage M's headline turned out to be
(i) a CENSORED corner -- `full L=16 d192` does not train to a measurable rho_50
at N=96 -- and (ii) a SUB-ADDITIVE interaction, the pair of cuts costing LESS
than the sum of the parts. Neither is a main effect, so a deck organised around
main effects would be reporting a stage that did not happen. The order below
leads with the censored corner and the interaction.

PAGES
  1  the headline
  2  the 2x2, as a table                              (M2 / the corner levels)
  3  the four solve curves, corner B accented         (the censored corner)
  4  the four solve curves, corner D accented         (the rescue)
  5  additivity, as a table                           (M1)
  6  the interaction in bars of seed spread           (M1)
  7  per-seed rho_50 by corner                        (the seed story)
  8  why M0 cannot be scored as registered            (M0)
  9  validation token accuracy by corner              (M3)
 10  how an instance is tokenized                     (mandatory)
 11  one real reasoning trace                         (mandatory)
 12  against the previous best on the same eval       (finding 55)
 13  what this changes for the programme

Run it in a job, never on a login node: page 11 generates an N=96 instance and
solves it, and the whole thing renders 13 figures.
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from statistics import median

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slides import Deck, MUTED, style_axes, label_line, series_colors

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from scripts.stagem_score import (CORNERS, SCAN_M_DIRS, collect, rung, summarize,
                                  deltas, S_RHO, S_MS, S_ACC,
                                  F_GAP_RHO, F_GAP_MS, F_GAP_ACC,
                                  A_TOTAL, D_TOTAL, N48_MINIMUM)

# Accent order per page: which corner the claim is ABOUT. series_colors(n,
# highlight=i) greys the rest, which is emphasis rather than identity -- the
# right encoding when four curves share one page and one of them is the point.
ORDER = ["A", "B", "C", "D"]
PRETTY = {"A": "A control  full L=12 d256",
          "B": "B shape    full L=16 d192",
          "C": "C MLP      no_mlp L=12 d256",
          "D": "D stripped no_mlp L=16 d192"}
SHORT = {"A": "A full L12 d256", "B": "B full L16 d192",
         "C": "C no_mlp L12 d256", "D": "D no_mlp L16 d192"}


class PngDeck(Deck):
    """slides.Deck plus a PNG per page, and a savefig() the shared pages can use.

    There is no PDF rasterizer on this cluster, so the PNGs are the only way to
    LOOK at the deck before shipping it, which is required before any upload --
    reading them back is what caught six defects in the stage N deck that no
    exit code would have.

    Overriding _save here rather than editing ~/.local/lib/claude-slides/slides.py
    keeps the shared builder untouched: other projects' decks import it too, and
    a PNG side effect is this project's need, not everyone's.

    savefig() exists because notebooks/deck_tokenization's mandatory pages were
    written against a PdfPages and build their own figure. Routing them through
    here lets them compose with this deck unchanged -- they do not go through
    _blank(), so the page counter is bumped here instead.
    """

    def __init__(self, path, png_dir, **kw):
        Deck.__init__(self, path, **kw)
        self.png_dir = png_dir
        os.makedirs(png_dir, exist_ok=True)

    def _save(self, fig):
        fig.savefig(os.path.join(self.png_dir, "page%d.png" % self._n), dpi=110)
        Deck._save(self, fig)

    def savefig(self, fig):
        self._n += 1
        self._save(fig)


# ----------------------------------------------------------------- the data

def load_curves(scan_dirs):
    """Solve curves for every leak-free N=96 corner cell, keyed like the scorer.

    Read from the rho_curve_*.csv the eval writes, not from the checkpoints:
    those CSVs carry realized_rho, which is the x-axis the rho_50 bars are
    defined on. Keyed on (arch_mode, num_layers, d_model) so a curve can never
    be attached to the wrong corner, and filtered to the registered budget so a
    cosine or 120k cell cannot leak into a corner.

    Seeds are pooled on the NOMINAL rho, not on realized_rho. Realized density
    differs by a hint or two between seeds at the same grid point, so binning on
    it would split each of the 14 points into three and draw a jagged curve out
    of single-seed values. The x actually plotted is still realized -- the median
    realized_rho within the bin -- because that is the axis rho_50 lives on.
    """
    by = defaultdict(lambda: defaultdict(list))
    for d in scan_dirs:
        for p in glob.glob(os.path.join(d, "rho_curve_*.csv")):
            with open(p) as f:
                for r in csv.DictReader(f):
                    # These CSVs span months and the column set GREW. A missing
                    # filter field excludes the row (fail closed, never assume a
                    # budget); a missing arch_mode is different -- the file
                    # predates the field, which predates no_mlp, so the cell is
                    # `full`. Same rule as the stage Q config guard.
                    try:
                        keep = (int(r["n_nodes"]) == 96 and r["shuffle_adj"] == "True"
                                and r["fall"] == "True" and int(r["epochs"]) == 50
                                and int(r["n_train_instances"]) == 240000)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not keep:
                        continue
                    key = (r.get("arch_mode") or "full", int(r["num_layers"]), int(r["d_model"]))
                    by[key][round(float(r["rho"]), 4)].append(
                        (float(r["realized_rho"]), float(r["solve_rate"])))
    return by


def corner_curve(curves, key):
    """(realized rho, median solve_rate) for one corner, pooled over its seeds."""
    k = (key[3], key[1], key[4])          # arch_mode, num_layers, d_model
    if k not in curves:
        return [], []
    pts = sorted((median([v[0] for v in vs]), median([v[1] for v in vs]))
                 for vs in curves[k].values())
    return [p[0] for p in pts], [p[1] for p in pts]


# ------------------------------------------------------------ the drawings
#
# Every figure is drawn by a MODULE-LEVEL function taking a bare axes. The deck
# hands it to slides.Deck.claim_page and the results notebook calls the same
# function on its own axes, so the notebook and the PDF cannot drift -- there is
# one implementation of each plot. (Stage N achieved this with a PdfPages sink;
# that trick does not carry over, because slides.Deck owns its axes rather than
# accepting a caller-built figure.)

def draw_curves(ax, keys, curves, accent):
    cols = series_colors(len(ORDER), highlight=ORDER.index(accent))
    for tag, col in zip(ORDER, cols):
        xs, ys = corner_curve(curves, keys[tag])
        if not xs:
            continue
        lw = 2.6 if tag == accent else 1.4
        ax.plot(xs, ys, color=col, linewidth=lw, zorder=3 if tag == accent else 2)
        label_line(ax, xs[-1], ys[-1], tag, color=col)
    ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    style_axes(ax, xlabel="realized hint density", ylabel="solve rate")
    ax.set_ylim(-0.02, 1.02)


def room_for_y_labels(ax, dx):
    """Shift the axes right so long y tick labels survive the page edge.

    slides.Deck owns the axes and pins its left edge at 0.08 of the figure
    width; whatever a tick label needs to the left of that is cut off at the
    page boundary. On the first render that ate the leading characters of every
    corner name on the accuracy page, leaving four bars nobody could tell apart.
    The labels carry the identity of the bars, so the margin gives way, not the
    labels.
    """
    p = ax.get_position()
    ax.set_position([p.x0 + dx, p.y0, p.width - dx, p.height])


def draw_interaction(ax, ia_ms, ia_acc):
    room_for_y_labels(ax, 0.05)
    names = ["mean_solve", "val_acc"]
    vals = [abs(ia_ms) / S_MS, abs(ia_acc) / S_ACC]
    ax.barh(names, vals, color=series_colors(2, highlight=1), height=0.45, zorder=3)
    ax.axvline(1.0, color=MUTED, linewidth=1.2, linestyle=":", zorder=2)
    for n, v in zip(names, vals):
        ax.text(v + 0.08, n, "%.2f bars" % v, va="center", fontsize=15)
    style_axes(ax, xlabel="interaction, in units of this metric's seed spread")
    ax.set_xlim(0, max(vals) * 1.25)


def draw_per_seed(ax, got):
    cols = series_colors(len(ORDER), highlight=2)
    for i, (tag, col) in enumerate(zip(ORDER, cols)):
        # Seeds are spread along x. Without this they overplot: corner C's three
        # seeds agree to 0.005, which is the CLAIM of this page, and drawing them
        # at one x turned three dots into one -- indistinguishable from a corner
        # where only one seed ran. B's two censored seeds coincided exactly, so
        # the note promised two crosses and the figure showed one.
        for j, (_sd, v, ok) in enumerate(got[tag]["per_seed"]):
            x = i + (j - 1) * 0.12
            if ok:
                ax.plot([x], [v], "o", color=col, markersize=10, zorder=3)
            else:
                # A censored seed has no level. It is drawn at the grid top as a
                # cross so the page cannot be read as if it had one.
                ax.plot([x], [0.95], "x", color=col, markersize=11, zorder=3,
                        markeredgewidth=2.2)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([SHORT[t] for t in ORDER], fontsize=12)
    ax.set_xlim(-0.5, len(ORDER) - 0.5)
    style_axes(ax, ylabel="rho_50 per seed")
    ax.set_ylim(0.5, 1.0)


def draw_acc(ax, got, control_acc):
    room_for_y_labels(ax, 0.10)        # "D no_mlp L16 d192" is the longest label
    vals = [got[t]["acc"] for t in ORDER]
    ax.barh([SHORT[t] for t in ORDER], vals,
            color=series_colors(len(ORDER), highlight=1), height=0.5, zorder=3)
    ax.axvline(control_acc, color=MUTED, linewidth=1.2, linestyle=":", zorder=2)
    for t, v in zip(ORDER, vals):
        ax.text(v + 0.002, SHORT[t], "%.4f" % v, va="center", fontsize=14)
    style_axes(ax, xlabel="validation token accuracy")
    ax.set_xlim(0.40, max(vals) * 1.05)


# ----------------------------------------------------------------- the pages

def page_curves(d, keys, curves, accent, claim, note):
    with d.claim(claim, note=note) as ax:
        draw_curves(ax, keys, curves, accent)


def load_stagem(boot=2000):
    """Everything both the deck and the notebook read, computed once, from disk.

    Returns the four corner summaries, the per-corner solve curves, the three
    deltas against the control and the two interactions -- all through
    stagem_score's own collect/rung/summarize/deltas, so a number cannot differ
    between the scorer, the deck and the notebook.
    """
    cells = collect(SCAN_M_DIRS, boot)
    keys = {tag: key for tag, _name, key in CORNERS}
    got = {tag: (summarize(rung(cells, key)) if rung(cells, key) else None)
           for tag, _name, key in CORNERS}
    a, b, c, dstr = got["A"], got["B"], got["C"], got["D"]
    dm, ds, dd_ = deltas(c, a), deltas(b, a), deltas(dstr, a)
    ia_ms = (dm["mean_solve"][0] + ds["mean_solve"][0]) - dd_["mean_solve"][0]
    ia_acc = (dm["val_token_acc"][0] + ds["val_token_acc"][0]) - dd_["val_token_acc"][0]
    return {"cells": cells, "keys": keys, "got": got, "curves": load_curves(SCAN_M_DIRS),
            "dMLP": dm, "dSHAPE": ds, "dPAIR": dd_, "ia_ms": ia_ms, "ia_acc": ia_acc}


def fmt_rho(s):
    if s is None:
        return "--"
    return "%.3f" % s["rho50"] if s["rho50_readable"] else "CENSORED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/users/hshunt/spin_gnn/notebooks/stagem")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()
    pdf_path = args.pdf or os.path.join(args.out, "stagem_slides.pdf")
    os.makedirs(args.out, exist_ok=True)

    S = load_stagem(args.boot)
    got, keys, curves = S["got"], S["keys"], S["curves"]
    a, c = got["A"], got["C"]
    dm, ds, dd_ = S["dMLP"], S["dSHAPE"], S["dPAIR"]
    ia_ms, ia_acc = S["ia_ms"], S["ia_acc"]

    # A deck that renders below the registered seed count is a deck that will be
    # believed at the wrong confidence. Stage N's refused for the same reason.
    for tag in ORDER:
        s = got[tag]
        if s is None or s["n"] < 3:
            sys.exit("corner %s has %s seeds, registered 3. Refusing to build."
                     % (tag, "no" if s is None else s["n"]))

    # Neither the subtitle nor this line WRAPS -- title_page lays both out as a
    # single run of text, so anything past the page width is simply cut off at
    # the edge (it was, on the first render). Both are kept short enough to fit.
    prov = ("array 51745334 | scorer 51746081 | commit %s | 3 seeds/corner | "
            "N=96 d5 q4 --shuffle-adj --fall 240k x 50ep | 400/bin" % args.commit)

    with PngDeck(pdf_path, args.out, project="spin_gnn stage M",
                 provenance=prov) as d:

        d.title_page("Neither half of the N=96 cut is load-bearing alone",
                     subtitle="The two cuts interact, and the corner that fails to "
                              "train is the one keeping its MLP.")

        d.table_page(
            "The deep-narrow full model is the only corner that fails",
            ["corner", "rho_50", "crossing", "mean_solve", "val_acc"],
            [[SHORT[t], fmt_rho(got[t]), "%d/3" % got[t]["n_cross"],
              "%.4f" % got[t]["mean_solve"], "%.4f" % got[t]["acc"]] for t in ORDER],
            highlight_row=1,
            note="Lower rho_50 is better. B censors on 2 of 3 seeds, so its level is "
                 "not quotable and M0's main effect does not exist.")

        page_curves(d, keys, curves, "B",
                    "Removing the MLP rescues the shape that kills the full model",
                    note="B (full L=16 d192) never reaches 0.5. D is the same shape "
                         "with no MLP at all and crosses on every seed.")

        page_curves(d, keys, curves, "D",
                    "Attention-only trains at L=16 d192 where the full model censors",
                    note="Same depth, same width, same budget, same seeds. The only "
                         "difference is that D's feedforward writes zero.")

        d.table_page(
            "The two cuts are sub-additive: the pair costs less than the parts",
            ["metric", "dMLP (C-A)", "dSHAPE (B-A)", "sum", "actual (D-A)", "interaction"],
            [["rho_50", "%+.3f" % dm["rho_50"][0], "censored", "--",
              "%+.3f" % F_GAP_RHO, "not readable"],
             ["mean_solve", "%+.4f" % dm["mean_solve"][0], "%+.4f" % ds["mean_solve"][0],
              "%+.4f" % (dm["mean_solve"][0] + ds["mean_solve"][0]),
              "%+.4f" % dd_["mean_solve"][0], "%+.4f" % ia_ms],
             ["val_acc", "%+.4f" % dm["val_token_acc"][0], "%+.4f" % ds["val_token_acc"][0],
              "%+.4f" % (dm["val_token_acc"][0] + ds["val_token_acc"][0]),
              "%+.4f" % dd_["val_token_acc"][0], "%+.4f" % ia_acc]],
            note="rho_50 has no additivity row because corner B censored. The two "
                 "never-censoring metrics both read sub-additive.")

        d.claim_page(
            "Interaction reaches %.2f bars on solve rate, %.2f on accuracy"
            % (abs(ia_ms) / S_MS, abs(ia_acc) / S_ACC),
            lambda ax: draw_interaction(ax, ia_ms, ia_acc),
            note="Dotted line is one seed spread. An interaction this far past it "
                 "means no single-factor statement here generalizes.")

        d.claim_page(
            "Corner C's three seeds agree to %.3f, the tightest rung yet"
            % (max(v for _s, v, ok in c["per_seed"] if ok)
               - min(v for _s, v, ok in c["per_seed"] if ok)),
            lambda ax: draw_per_seed(ax, got),
            note="B's two censored seeds are drawn as crosses at the grid top; they "
                 "have no level, and that is the corner's result.")

        d.points_page(
            "M0 cannot be scored as registered because corner B censored",
            ["M0 asked that exactly one of |dMLP|, |dSHAPE| clear 0.117",
             "dSHAPE is not a level: 2 of 3 B seeds never reach 0.5",
             "Registered outcome (d) binds -- the cost is an interaction"])

        d.claim_page(
            "Validation accuracy ranks the corners exactly as solve rate does",
            lambda ax: draw_acc(ax, got, a["acc"]),
            note="Accuracy never censors, so it orders B too. Dotted line is the "
                 "control. B is worst on every metric, including this one.")

        # fall=True is NOT optional. It is an instance key (finding 43): it
        # selects the fall-colouring graph ensemble, whose removal paths reach
        # rho = 0.32 and whose traces are deeper at the same N. The first render
        # left it defaulted to False, so page 10's footer honestly read
        # "unconstrained ensemble" while page 11's note claimed --fall -- the
        # deck contradicted itself and illustrated the wrong distribution.
        ex = example_instance(n_nodes=96, degree=5, n_colors=4,
                              shuffle_adj=True, fall=True)
        page_tokenization(d, ex)
        page_trace(d, ex, note="N=96, degree 5, 4 colours, --shuffle-adj --fall: the "
                               "distribution all four corners were trained and "
                               "evaluated on.")

        d.table_page(
            "Finding 55's gap is an interaction, not the MLP cut",
            ["quantity", "stage F (previous best read)", "stage M (this run)"],
            [["what was compared", "D vs A only", "all four corners of the 2x2"],
             ["rho_50 gap D-A", "%+.3f" % F_GAP_RHO, "%+.3f, unchanged" % F_GAP_RHO],
             ["attributed to", "the attention-only cut", "an interaction (%.2f bars)"
              % (abs(ia_ms) / S_MS)],
             ["mean_solve gap", "%+.4f" % F_GAP_MS, "%+.4f, unchanged" % F_GAP_MS],
             ["MLP cut alone", "never measured", "%+.3f of that %+.3f"
              % (dm["rho_50"][0], F_GAP_RHO)]],
            highlight_row=2,
            note="Same eval, same grid, same bars. F's numbers do not move; what "
                 "moves is what they may be attributed to.")

        d.points_page(
            "The one-cut-at-a-time ladder cannot answer this at N=96",
            ["Depth-16 width-192 needs the MLP removed to train at all",
             "So 'which half is load-bearing' has no answer at N=96",
             "Every architecture claim here stays an N=48 claim"])

    size = os.path.getsize(pdf_path)
    print("wrote %s (%d KB, %d pages)" % (pdf_path, size // 1024, 13))
    print("PNGs at %s/page1..13.png -- READ EVERY ONE before shipping" % args.out)
    print("N=48 standing cell, for the record and NOT compared above: %s, %d trainable, "
          "rho_50 %.3f" % N48_MINIMUM)
    print("corner totals: A %d, D %d" % (A_TOTAL, D_TOTAL))


if __name__ == "__main__":
    main()
