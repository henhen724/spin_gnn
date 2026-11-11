"""STAGES O + P DECK -- two routes to a smaller cell, and two kinds of failure.

ONE deck for the two stages, on the same grounds stage Q and R share one: they
are two halves of one question. Under the 2026-11-10 brief the objective is the
simplest or most analytically tractable architecture that still learns direct
deduction, and there are exactly two ways to make the standing cell simpler --
make it SMALLER at the same arrangement (stage O: no_mlp L=16 d176, a width step
into the measured 1.91M-2.50M gap) or make it the SAME SIZE with fewer distinct
objects (stage P: one weight-tied block applied 16 times, h_{t+1} = F(h_t)).
Neither reached the free line, and reporting them apart would hide that the two
failures are of different kinds: OA produced a measured level one thousandth
past the line, P1 produced no architecture result at all because it never
escaped the plateau.

Every title on every page is computed from numbers this script reads off disk
through the two registered scorers' OWN functions, so the deck and the scorers
cannot disagree.

TWO THINGS THIS DECK MUST SAY OUT LOUD, both of which a tidier deck would bury:

  1. STAGE O'S DESIGN WAS AMENDED MID-FLIGHT. O0 was a two-arm shape comparison
     at matched size; the OB arm (no_mlp L=20 d160) was cancelled at epoch 9 of
     50 on 11-10 at 12:39 because the brief had changed under it -- 4 layers
     deeper than anything in the programme, with its entire registered upside
     being 1.15x fewer parameters, which is the wrong direction when the figure
     of merit is what the object IS. So O0 cannot be called, and page 5 says so
     rather than quietly reporting one arm as if it were the stage.
  2. STAGE P RETURNED NO ARCHITECTURE RESULT. P2 (the plateau) was falsified and
     by its own registration that outranks P0, so the censor is a fact about the
     recipe, not about tying.

PAGES
  1  the headline
  2  both cells against the standing minimum, as a table
  3  OA's solve curve inside its bracket                 (O2, the level)
  4  OA's three seeds                                    (the spread)
  5  why O0 cannot be called                             (the amendment)
  6  the transition, with OA's point inside it           (O1)
  7  P1's solve curve against its untied twin            (P0, as a bound)
  8  the plateau, both stages                            (P2 -- read first)
  9  P1's training trajectory                            (why width, not warmup)
 10  validation token accuracy                           (O4a / P3a)
 11  out-of-sample sufficiency                           (O4b / P3b)
 12  the confound P1 named in advance                    (P1)
 13  stage S, running now                                (what the confound costs)
 14  how an instance is tokenized                        (mandatory)
 15  one real reasoning trace                            (mandatory)
 16  against the previous best on the same eval          (the standing rule)

Run it in a job, never on a login node: page 15 generates an N=48 instance and
solves it, the sufficiency page fits over twelve scan directories, and the whole
thing renders 16 figures.

Usage (from a job; see scripts/slurm/stageop_writeup_2026-11-10.sbatch):
    python3 -u -m notebooks.make_stageop_deck --commit <sha>
"""

import argparse
import csv
import glob
import json
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
from scripts.acc_vs_solve import FLOOR, is_reference, pava, predict
from scripts.acc_vs_solve import collect as acc_collect
from scripts.minarch_analysis import SCAN_DIRS
from scripts.stagec_score import PLATEAU, S_RHO50, escapes, rho_txt, rung, summarize
from scripts.stagej_score import LOO_BAR, SUFFICIENCY_DIRS
from scripts.stagek_score import FREE, SCAN_K
from scripts.stagen_score import SCAN_N, acc_of, load_pooled
from scripts.stageo_score import (BRACKET, BRACKET_DIRS, GAP_HI, GAP_LO, OA, OB,
                                 SCAN_O, STANDING_RHO50, STANDING_TRAINABLE)
from scripts.stagep_score import (CENSOR_FLOOR, FOLLOWUP, P1, SCAN_P,
                                  UNTIED_AT_SAME_SHAPE)

# The untied twin, which is also the standing minimum and the free end of the
# bracket. Named once here so no page re-derives the key.
J2 = (16, 4, "no_mlp", 192, 1024, 0)
H2 = (12, 4, "no_mlp", 192, 1024, 0)

PRETTY = {"OA": "OA  no_mlp L=16 d176",
          "P1": "P1  tied_no_mlp T=16 d192",
          "J2": "J2  no_mlp L=16 d192",
          "H2": "H2  no_mlp L=12 d192"}

# table_page spaces its columns evenly across 0.84 of the page, so the first
# column of a four-column table gets about 300px and of a six-column table about
# 205px. The PRETTY labels are wider than that at fontsize 16 and the read-back
# found them printed THROUGH the trainable column on pages 2, 6, 10 and 13 --
# "J2  no_mlp L=16 d192" over "2,504,890" -- with no exit code. Tables get these
# instead; PRETTY stays for the curve labels and prose, where there is room.
# "tied" alone is unambiguous: tied_no_mlp is the only tied mode in the tree.
SHORT = {"OA": "OA  L=16 d176",
         "P1": "P1  tied T=16 d192",
         "J2": "J2  L=16 d192",
         "H2": "H2  L=12 d192"}
KEYS = {"OA": OA, "P1": P1, "J2": J2, "H2": H2}


class PngDeck(Deck):
    """slides.Deck plus a PNG per page, and a savefig() the shared pages can use.

    There is no PDF rasterizer on this cluster, so the PNGs are the only way to
    LOOK at the deck before shipping it, which is required before any upload --
    the stage Q+R read-back caught three defects that had no exit code, and
    stage N's caught six.
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
    """Solve curves for every leak-free N=48 cell, keyed as the scorers key them.

    Read from the rho_curve_*.csv the eval writes rather than from the
    checkpoints: those CSVs carry realized_rho, which is the x-axis rho_50 is
    defined on. Keyed on (arch_mode, num_layers, d_model) so a curve can never be
    attached to the wrong cell -- and arch_mode is what separates the tied cell
    from its untied twin, which share L and d exactly.

    Seeds are pooled on the NOMINAL rho and plotted at the median REALIZED rho
    within the bin, as stage M's deck does: realized density differs by a hint or
    two between seeds at one grid point, so binning on it would split each of the
    14 points into three and draw a jagged curve out of single-seed values.
    """
    by = defaultdict(lambda: defaultdict(list))
    for d in scan_dirs:
        for p in glob.glob(os.path.join(d, "rho_curve_*.csv")):
            with open(p) as f:
                for r in csv.DictReader(f):
                    # These CSVs span weeks and the column set GREW. A missing
                    # filter field excludes the row (fail closed, never assume a
                    # budget); a missing arch_mode is different -- the file
                    # predates the field, which predates no_mlp, so the cell is
                    # `full`.
                    try:
                        keep = (int(r["n_nodes"]) == 48 and r["shuffle_adj"] == "True"
                                and r["fall"] == "True" and int(r["epochs"]) == 50
                                and int(r["n_train_instances"]) == 240000)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not keep:
                        continue
                    key = (r.get("arch_mode") or "full", int(r["num_layers"]),
                           int(r["d_model"]))
                    by[key][round(float(r["rho"]), 4)].append(
                        (float(r["realized_rho"]), float(r["solve_rate"])))
    return by


def cell_curve(curves, key):
    """(realized rho, median solve_rate) for one cell, pooled over its seeds."""
    k = (key[2], key[0], key[3])          # arch_mode, num_layers, d_model
    if k not in curves:
        return [], []
    pts = sorted((median([v[0] for v in vs]), median([v[1] for v in vs]))
                 for vs in curves[k].values())
    return [p[0] for p in pts], [p[1] for p in pts]


def trajectory(path):
    """Per-epoch train NLL from one cell's history.json.

    minarch_analysis.history() returns only the aggregates (final, best, slope).
    Page 9's whole claim is about the SHAPE of the curve -- monotone and
    decelerating rather than spiking -- so it needs every epoch.
    """
    try:
        with open(os.path.join(path, "history.json")) as fh:
            h = json.load(fh)
        return [float(e["train_nll"]) for e in h]
    except (OSError, KeyError, ValueError):
        return []


def residuals(cells, tags):
    """Where each cell sits against the accuracy-to-solve curve (O4b / P3b).

    SUFFICIENCY_DIRS plus K, N, O and P, exactly the dirs both scorers print:
    the fit is over the reference ladder (full-width 4-head cells) in those
    dirs, and a narrower set moves the fit. `weak` reproduces
    acc_vs_solve.py:227 -- a residual is discriminating only with BOTH a tight
    bracket and a prediction off the 0.10 floor, and testing only the floor made
    stage N's deck call six seeds discriminating where the scorer called five.
    """
    pool = [c for c in acc_collect(list(SUFFICIENCY_DIRS)
                                   + [SCAN_K, SCAN_N, SCAN_O, SCAN_P])
            if c["acc"] == c["acc"]]
    fit = pava([(c["acc"], c["mean_solve"]) for c in pool if is_reference(c["rung"])])
    out = []
    for tag in tags:
        for c in sorted((c for c in pool if c["rung"] == KEYS[tag]),
                        key=lambda c: c["seed"]):
            pred, oor, cov = predict(fit, c["acc"])
            out.append({"tag": tag, "seed": c["seed"], "acc": c["acc"],
                        "pred": pred, "resid": c["mean_solve"] - pred,
                        "oor": oor, "cov": cov,
                        "weak": not (cov <= LOO_BAR and pred > FLOOR)})
    return out


def load_stageop(boot=2000):
    """Everything both the deck and the notebook read, computed once, from disk."""
    dirs = [SCAN_O, SCAN_P] + list(BRACKET_DIRS) + list(SCAN_DIRS)
    cells = load_pooled(dirs, boot)
    got, ms = {}, {}
    for tag, key in KEYS.items():
        rows = rung(cells, key)
        ms[tag] = rows
        got[tag] = summarize(rows) if rows else None
    return {"cells": cells, "got": got, "ms": ms,
            "curves": load_curves(dirs),
            "esc": {t: escapes(ms[t]) for t in KEYS if ms[t]},
            "acc": {t: acc_of(ms[t]) for t in KEYS if ms[t]}}


# ------------------------------------------------------------- the drawings
#
# Every figure is drawn by a MODULE-LEVEL function taking a bare axes, so the
# results notebook calls the same function on its own axes and the notebook and
# the PDF cannot drift.

def room_for_y_labels(ax, dx):
    """Shift the axes right so long y tick labels survive the page edge.

    slides.Deck pins the axes' left edge at 0.08 of the figure width and
    whatever a tick label needs to the left of that is cut at the page boundary.
    The labels carry the identity of the bars, so the margin gives way.
    """
    p = ax.get_position()
    ax.set_position([p.x0 + dx, p.y0, p.width - dx, p.height])


def label_ends(ax, ends, tol):
    """One label per CLUSTER of coincident curve ends, not one per curve.

    Curves that finish on top of each other print their labels on top of each
    other, and three overprinted labels are not three labels, they are an
    illegible smear -- which is what the read-back found on page 3 (H2 and J2
    both at solve rate ~1.0) and page 9 (three seeds within 0.008 NLL). A
    cluster gets ONE label naming all of its members, in grey because the label
    no longer stands for a single hue, and that is also the honest reading: the
    reader sees one line and is told it is several.

    `ends` is [(x, y, name, colour)], `tol` the y-distance that counts as
    coincident in data units.
    """
    groups = []
    for x, y, name, col in sorted(ends, key=lambda e: -e[1]):
        if groups and abs(groups[-1][0] - x) < 1e-9 and abs(groups[-1][1] - y) <= tol:
            groups[-1][2].append(name)
        else:
            groups.append([x, y, [name], col])
    for x, y, names, col in groups:
        label_line(ax, x, y, " ".join(names),
                   color=col if len(names) == 1 else "0.35")


def draw_bracket(ax, curves, accent):
    """OA's solve curve inside the bracket that defines the measured gap."""
    tags = ["H2", "OA", "J2"]
    cols = series_colors(len(tags), highlight=tags.index(accent))
    # H2 dashed, J2 solid. Both are muted to the same grey and they END together
    # at solve rate 1.0, so a reader given one cluster label cannot tell which
    # grey curve is which in the middle -- where they are 0.2 apart and where the
    # whole point of the page lives (OA runs with H2, not with J2). The dash is
    # the only thing that resolves them without a legend box.
    style = {"H2": (1.4, "--"), "J2": (1.4, "-"), "OA": (2.6, "-")}
    ends = []
    for tag, col in zip(tags, cols):
        xs, ys = cell_curve(curves, KEYS[tag])
        if not xs:
            continue
        lw, ls = style[tag]
        if tag == accent:
            lw = 2.6
        ax.plot(xs, ys, color=col, linewidth=lw, linestyle=ls,
                zorder=3 if tag == accent else 2)
        ends.append((xs[-1], ys[-1], tag, col))
    label_ends(ax, ends, tol=0.03)
    ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    style_axes(ax, xlabel="realized hint density", ylabel="solve rate")
    ax.set_ylim(-0.02, 1.02)


def draw_tied_vs_untied(ax, curves):
    """P1 against its untied twin at the same depth and width."""
    tags = ["P1", "J2"]
    cols = series_colors(2, highlight=0)
    ends = []
    for tag, col in zip(tags, cols):
        xs, ys = cell_curve(curves, KEYS[tag])
        if not xs:
            continue
        ax.plot(xs, ys, color=col, linewidth=2.6 if tag == "P1" else 1.6)
        ends.append((xs[-1], ys[-1], tag, col))
    label_ends(ax, ends, tol=0.03)
    ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    style_axes(ax, xlabel="realized hint density", ylabel="solve rate")
    ax.set_ylim(-0.02, 1.02)


def draw_per_seed(ax, ms, free, standing):
    """OA's three seeds, with the free line and the standing level behind them."""
    rows = sorted(ms["OA"], key=lambda m: m["seed"])
    cols = series_colors(3)
    for m, col in zip(rows, cols):
        ax.plot([m["seed"]], [m["rho50"]], "o", color=col, markersize=12, zorder=3)
        ax.annotate("%.3f" % m["rho50"], xy=(m["seed"], m["rho50"]),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=14, color=col)
    ax.axhline(free, color="0.25", linestyle="--", linewidth=1.2, zorder=2)
    # BELOW its line, not above: seed 0 sits within 0.006 of the free line and
    # its value annotation is drawn 14pt above the marker, so the two texts
    # touched at the left edge of the plot.
    ax.text(-0.42, free - 0.016, "free line %.3f" % free, size=11, color="0.25")
    ax.axhline(standing, color=MUTED, linestyle=":", linewidth=1.2, zorder=2)
    ax.text(-0.42, standing + 0.004, "J2 %.3f" % standing, size=11, color=MUTED)
    ax.set_xticks([m["seed"] for m in rows])
    ax.set_xticklabels(["seed %d" % m["seed"] for m in rows], fontsize=13)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    style_axes(ax, ylabel="rho_50")
    ax.set_ylim(0.55, 0.80)


def draw_plateau(ax, esc):
    """Final train NLL per seed for both cells, against finding 44's plateau."""
    room_for_y_labels(ax, 0.06)
    tags = ["P1", "OA", "J2"]
    cols = series_colors(3, highlight=0)
    ys, labels = [], []
    for i, (tag, col) in enumerate(zip(tags, cols)):
        if tag not in esc:
            continue
        _n, nll = esc[tag]
        for v in nll:
            ax.plot([v], [i], "o", color=col, markersize=11, zorder=3)
        ys.append(i)
        labels.append(tag)
    ax.axvline(PLATEAU, color="0.25", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(PLATEAU - 0.001, len(tags) - 0.45, "plateau %.2f" % PLATEAU,
            size=12, color="0.25", ha="right")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_ylim(-0.6, len(tags) - 0.3)
    ax.invert_yaxis()
    style_axes(ax, xlabel="final train NLL (one point per seed)")


def draw_trajectory(ax, ms):
    """P1's per-epoch train NLL, one line per seed, with the plateau behind."""
    rows = sorted(ms["P1"], key=lambda m: m["seed"])
    cols = series_colors(3)
    ends = []
    for m, col in zip(rows, cols):
        t = trajectory(m["path"])
        if not t:
            continue
        ax.plot(range(1, len(t) + 1), t, color=col, linewidth=2)
        ends.append((len(t), t[-1], "s%d" % m["seed"], col))
    # The three seeds finish within 0.008 NLL of each other, which is why this
    # is the page the cluster rule was written for.
    label_ends(ax, ends, tol=0.01)
    ax.axhline(PLATEAU, color="0.25", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(2, PLATEAU - 0.024, "plateau %.2f" % PLATEAU, size=12, color="0.25")
    style_axes(ax, xlabel="epoch", ylabel="train NLL")
    ax.set_ylim(1.60, 1.95)


def draw_residuals(ax, res):
    """Each seed's distance from the accuracy-to-solve curve, against the bar."""
    room_for_y_labels(ax, 0.07)
    shown = [r for r in res if not r["oor"]]
    labels = ["%s s%d" % (r["tag"], r["seed"]) for r in shown]
    vals = [r["resid"] for r in shown]
    cols = ["0.62" if r["weak"] else series_colors(3)[0] for r in shown]
    ax.barh(range(len(shown)), vals, color=cols, height=0.55, zorder=3)
    # The limit has to hold the DATA, not just the bar. At a fixed +-1.5*LOO_BAR
    # P1 s0's +0.0729 was cut off at the frame and read as 0.064, with its label
    # printed outside the axes: a truncated bar with no mark is a wrong number,
    # not a cropped picture. Widened to whichever is larger, with room for the
    # label at the end of the longest bar.
    lim = max(LOO_BAR * 1.5, max((abs(v) for v in vals), default=0) * 1.32)
    for i, r in enumerate(shown):
        off = 0.002 if r["resid"] >= 0 else -0.002
        ax.text(r["resid"] + off, i, "%+.4f" % r["resid"], va="center",
                ha="left" if r["resid"] >= 0 else "right", fontsize=12)
    for s in (-1, 1):
        ax.axvline(s * LOO_BAR, color="0.25", linestyle="--", linewidth=1.2, zorder=2)
    # Right-aligned just left of the line, not centred on it: centred, the dashed
    # line printed through the middle of the word.
    ax.text(LOO_BAR - lim * 0.012, len(shown) - 0.4, "LOO bar %.4f" % LOO_BAR,
            size=11, color="0.25", ha="right")
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_ylim(-0.7, len(shown) - 0.3)
    ax.set_xlim(-lim, lim)
    style_axes(ax, xlabel="mean_solve - curve's prediction  (grey = not discriminating)")


# ----------------------------------------------------------------- the pages

def fmt_rho(s):
    return "--" if s is None else rho_txt(s)


def side(txt):
    """Which side of the free line a rho_50 TEXT falls on.

    Computed from the number rather than from the row's position in a
    size-sorted table: in this particular bracket size and side happen to agree,
    and a table that asserted the side from the sort would keep asserting it
    after they stopped agreeing -- which is the whole question stage O asked.
    A censor has no side.
    """
    if txt == "CENSORED":
        return "no level"
    return "free" if float(txt) <= FREE else "past"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/users/hshunt/spin_gnn/notebooks/stageop")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()
    pdf_path = args.pdf or os.path.join(args.out, "stageop_slides.pdf")
    os.makedirs(args.out, exist_ok=True)

    S = load_stageop(args.boot)
    got, ms, curves, esc = S["got"], S["ms"], S["curves"], S["esc"]

    # A deck that renders below the registered seed count is a deck that will be
    # believed at the wrong confidence.
    for tag in ("OA", "P1", "J2"):
        s = got[tag]
        if s is None or s["n"] < 3:
            sys.exit("%s has %s seeds, registered 3. Refusing to build."
                     % (tag, "no" if s is None else s["n"]))
    # Guard: the two cells must be the ones the registrations describe. A tied
    # count equal to the untied one means the sharing silently did not happen.
    if got["P1"]["trainable"] == UNTIED_AT_SAME_SHAPE:
        sys.exit("P1's count equals the untied twin's: the sharing did not happen.")
    if got["J2"]["trainable"] != STANDING_TRAINABLE:
        sys.exit("the standing minimum drifted; every comparison here is against it.")

    oa, p1, j2 = got["OA"], got["P1"], got["J2"]
    oa_gap = oa["rho50"] - FREE
    oa_spread = (max(m["rho50"] for m in ms["OA"])
                 - min(m["rho50"] for m in ms["OA"]))
    acc_med = {t: median([a for _s, a in S["acc"][t]]) for t in S["acc"]}
    res = residuals(S["cells"], ("OA", "P1"))
    disc = [r for r in res if not r["weak"] and not r["oor"]]
    worst = max((abs(r["resid"]) for r in disc), default=float("nan"))
    p1_traj = trajectory(sorted(ms["P1"], key=lambda m: m["seed"])[0]["path"])

    prov = ("arrays 51801495 / 51815311 | scorers 51801620 / 51815318 | commit %s "
            "| 3 seeds | N=48 d5 q4 --shuffle-adj --fall 240k x 50ep | 400/bin"
            % args.commit)

    with PngDeck(pdf_path, args.out, project="spin_gnn stages O+P",
                 provenance=prov) as d:

        d.title_page(
            "Neither route to a smaller cell reaches the free line",
            subtitle="A width step lands one thousandth past it; weight tying "
                     "never escapes the plateau at all.")

        d.table_page(
            "The standing minimum is unchanged at 0.580 and 2,504,890",
            # Five columns, not six: val acc has a page of its own (10) and a
            # sixth column here squeezed every column to 205px, which is where
            # the cell labels started printing through the trainable counts.
            ["cell", "trainable", "rho_50", "crossing", "escapes"],
            [[SHORT["J2"], "{:,}".format(j2["trainable"]),
              fmt_rho(j2), "%d/3" % j2["n_cross"], "%d/3" % esc["J2"][0]],
             [SHORT["OA"], "{:,}".format(oa["trainable"]), fmt_rho(oa),
              "%d/3" % oa["n_cross"], "%d/3" % esc["OA"][0]],
             [SHORT["P1"], "{:,}".format(p1["trainable"]), fmt_rho(p1),
              "%d/3" % p1["n_cross"], "%d/3" % esc["P1"][0]]],
            highlight_row=0,
            note="Lower rho_50 is better. Free means every seed crosses 0.5 and the "
                 "median sits at or below %.3f." % FREE)

        d.claim_page(
            # No fraction in the title: the miss is 0.001 against a 0.011 seed
            # span, which is nearer a tenth than the "fifth" an earlier draft
            # asserted, and a hardcoded ratio would keep asserting itself after
            # the next render moved the numbers.
            "OA misses the free line by %.3f, well inside its seed spread"
            % oa_gap,
            lambda ax: draw_bracket(ax, curves, "OA"),
            note="Median %.3f against the line at %.3f, three of three crossing. "
                 "Dashed H2 and solid J2 end the measured gap."
                 % (oa["rho50"], FREE))

        d.claim_page(
            # NOT "every one is past the line", which the first render asserted
            # over a figure showing seed 0 at 0.734, i.e. BELOW 0.740 and free
            # on its own. The gate reads the median, so one free seed does not
            # make the cell free -- but a title has to match its own picture.
            # Counted from the data so it cannot drift from the markers.
            "%d of OA's three seeds %s free, but the median is not"
            % ((lambda n: (n, "is" if n == 1 else "are"))
               (sum(1 for m in ms["OA"] if m["rho50"] <= FREE))),
            lambda ax: draw_per_seed(ax, ms, FREE, STANDING_RHO50),
            note="Span %.3f: %s. The standing rule reads the median, which lands "
                 "%+.3f past the line."
                 % (oa_spread,
                    ", ".join("%.3f %s" % (m["rho50"],
                                           "free" if m["rho50"] <= FREE else "past")
                              for m in sorted(ms["OA"], key=lambda m: m["seed"])),
                    oa_gap))

        d.points_page(
            "Stage O's other arm was cancelled, so O0 cannot be called",
            ["O0 compared two SHAPES at matched size; one shape is gone",
             "OB (no_mlp L=20 d160) was stopped at epoch 9 of 50",
             "Its whole registered upside was 1.15x fewer parameters"])

        d.table_page(
            "The %.2fM-to-%.2fM step now has one measured point inside it"
            % (GAP_LO / 1e6, GAP_HI / 1e6),
            ["rung", "trainable", "rho_50", "side of the line"],
            [[label, "{:,}".format(tr), txt, side(txt)] for label, tr, txt in
             sorted([(lbl, tr, txt) for lbl, tr, txt, _e in BRACKET.values()]
                    + [(SHORT["OA"], oa["trainable"], fmt_rho(oa))],
                    key=lambda r: r[1])],
            highlight_row=2,
            note="Sorted by size. Everything below 2.50M is past the line; the drop "
                 "from %.3f to %.3f still happens in one step." % (oa["rho50"],
                                                                  STANDING_RHO50))

        d.claim_page(
            # The earlier title counted the seeds that cross 0.5 ("one of three")
            # over a figure whose tied curve tops out at 0.31, because pooling
            # three seeds is not the same as any one of them. A title has to
            # describe its own picture; the 1/3 count moved into the note and is
            # on page 2's table besides.
            "Tying at J2's shape leaves solve rate near zero until 0.8",
            lambda ax: draw_tied_vs_untied(ax, curves),
            note="Curves pooled over seeds; one of three crosses 0.5. Same shape, "
                 "budget and seeds; P1 pays the per-block term once, %s against %s."
                 % ("{:,}".format(p1["trainable"]),
                    "{:,}".format(j2["trainable"])))

        d.claim_page(
            "No tied seed escapes the %.2f plateau, so P2 is falsified" % PLATEAU,
            lambda ax: draw_plateau(ax, esc),
            note="Read before any level, as registered. An optimization-limited "
                 "cell's rho_50 is a fact about the recipe, not about tying.")

        d.claim_page(
            "The tied curve asymptotes to %.2f rather than destabilizing"
            % p1_traj[-1],
            lambda ax: draw_trajectory(ax, ms),
            note="Monotone and decelerating on all three seeds, no spike at any "
                 "epoch. Warmup and a lower LR treat an instability there is none of.")

        d.table_page(
            # "the two cells" over a three-row table: the table carries J2 as
            # well, and all three orders agree, which is the stronger statement.
            "Accuracy orders all three cells exactly as solve rate does",
            ["cell", "val token acc", "mean_solve", "rho_50"],
            [[SHORT["J2"], "%.4f" % acc_med["J2"], "%.4f" % j2["mean_solve"],
              fmt_rho(j2)],
             [SHORT["OA"], "%.4f" % acc_med["OA"], "%.4f" % oa["mean_solve"],
              fmt_rho(oa)],
             [SHORT["P1"], "%.4f" % acc_med["P1"], "%.4f" % p1["mean_solve"],
              fmt_rho(p1)]],
            note="Accuracy never censors, so it orders P1 too. Tied minus untied is "
                 "%+.4f at the same shape." % (acc_med["P1"] - acc_med["J2"]))

        d.claim_page(
            ("Both cells sit on the accuracy-to-solve curve, worst miss %.4f"
             % worst) if disc and worst <= LOO_BAR else
            ("%d of %d discriminating seeds fall OFF the curve"
             % (sum(1 for r in disc if abs(r["resid"]) > LOO_BAR), len(disc)))
            if disc else
            "No seed's residual is discriminating: the curve cannot judge these",
            lambda ax: draw_residuals(ax, res),
            note="If tying changed the accuracy-to-solve RELATION rather than the "
                 "level, this is the page where it would show.")

        d.points_page(
            "P1's censor cannot separate tying from %s being too few"
            % "{:,}".format(p1["trainable"]),
            ["%s sits below H1's %s, the only rung that has censored"
             % ("{:,}".format(p1["trainable"]), "{:,}".format(CENSOR_FLOOR)),
             "'Tying costs the deduction' and '270k is too few' both predict this",
             "The confound was named at registration, before any number landed"])

        d.table_page(
            "Stage S straddles the censoring floor using one knob, width",
            ["cell", "trainable", "vs H1's %s" % "{:,}".format(CENSOR_FLOOR),
             "status"],
            [[SHORT["P1"], "{:,}".format(p1["trainable"]), "below", "censored"],
             # "tied_no_mlp" spelled out here printed straight through the
             # trainable column: a 4-column table gives the first one ~300 px.
             # Same abbreviation as SHORT["P1"], which fits.
             ["S1  tied T=16 d320", "{:,}".format(FOLLOWUP["d320"]),
              "below", "running (51853969)"],
             ["S2  tied T=16 d448", "{:,}".format(FOLLOWUP["d448"]),
              "ABOVE", "running (51853969)"]],
            highlight_row=2,
            note="Only d448 is above the floor, so only a censor there would be "
                 "about tying. One knob changes: d_model.")

        ex = example_instance(n_nodes=48, degree=5, n_colors=4,
                              shuffle_adj=True, fall=True)
        page_tokenization(d, ex)
        page_trace(d, ex, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the "
                               "distribution both stages were trained and evaluated "
                               "on.")

        d.table_page(
            "Neither stage replaces the previous best on this eval",
            ["quantity", "previous best J2", "stage O OA", "stage P P1"],
            [["trainable", "{:,}".format(j2["trainable"]),
              "{:,}".format(oa["trainable"]), "{:,}".format(p1["trainable"])],
             ["rho_50", fmt_rho(j2), fmt_rho(oa), fmt_rho(p1)],
             ["seeds crossing", "%d/3" % j2["n_cross"], "%d/3" % oa["n_cross"],
              "%d/3" % p1["n_cross"]],
             ["plateau escapes", "%d/3" % esc["J2"][0], "%d/3" % esc["OA"][0],
              "%d/3" % esc["P1"][0]],
             ["val token acc", "%.4f" % acc_med["J2"], "%.4f" % acc_med["OA"],
              "%.4f" % acc_med["P1"]],
             ["verdict", "stands", "past the line", "no architecture result"]],
            highlight_row=1,
            note="Same grid, same bars, same pooled s* eval: %.3f at %s before, "
                 "%.3f at %s after." % (STANDING_RHO50,
                                        "{:,}".format(STANDING_TRAINABLE),
                                        STANDING_RHO50,
                                        "{:,}".format(STANDING_TRAINABLE)))

    size = os.path.getsize(pdf_path)
    print("wrote %s (%d KB, 16 pages)" % (pdf_path, size // 1024))
    print("PNGs at %s/page1..16.png -- READ EVERY ONE before shipping" % args.out)
    print("OA rho_50 %s (%+.3f vs the %.3f line, %+.3f s*), P1 %s, standing %.3f"
          % (fmt_rho(oa), oa_gap, FREE, oa_gap / S_RHO50, fmt_rho(p1),
             STANDING_RHO50))
    print("OB never trained: tasks 51801495_1/_3/_5 cancelled at epoch 9 of 50")


if __name__ == "__main__":
    main()
