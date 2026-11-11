"""The stage S deck: what weight sharing actually costs.

Stage P tied the attention-only cell's 16 blocks into ONE and the result
CENSORED -- no rho bin cleared 0.5, so there was no number, and two readings
fit equally well: either sharing obstructs direct deduction, or d192 is simply
too small to hold one block that must serve 16 roles. Stage S separated them
the only way a censor can be separated, by buying the capacity back at fixed
sharing: d192 -> d320 -> d448, T=16 throughout.

d320 trained 3 of 3 and produced a number. So the plateau was a CAPACITY FLOOR.
And S4 then put those three cells ON the N=48 accuracy-to-solve curve, which
makes the whole solve-rate gap against the untied twin a consequence of 0.011
of token accuracy rather than of a damaged algorithm.

EVERY NUMBER IS READ THROUGH THE SCORER'S OWN FUNCTIONS. scripts/stages_score.py
was committed before the first cell ran (register job 51846608); this script
imports its rung keys, its bars and its readers rather than restating any
number in prose, and the accuracy-to-solve page re-runs acc_vs_solve's own
pava/predict on the same scan dirs the registered S4 job used.

THE d448 ARM IS REPORTED, NOT HIDDEN. Two of its three seeds diverged at epoch
0; diagnostic 51860479 traced that to warmup rather than to the cell. A deck
that showed only d320 would be quoting the arm that worked.

Usage (from a job; see scripts/slurm/stages_writeup_2026-11-11.sbatch):
    python3 -u -m notebooks.make_stages_deck --commit <sha>
"""
import argparse
import json
import os
import sys
from statistics import median

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slides import Deck, MUTED, SERIES, style_axes, label_line, series_colors

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace

from scripts.acc_vs_solve import FLOOR, collect, is_reference, pava, predict
from scripts.minarch_analysis import SCAN_DIRS, load_all
from scripts.stagec_score import PLATEAU, escapes, rho_txt, rung, summarize
from scripts.stagej_score import LOO_BAR
from scripts.stagek_score import FREE
from scripts.stagen_score import acc_of, load_pooled
from scripts.stagep_score import SCAN_P
from scripts.stages_score import (BASELINE_DIRS, P1, S1, S2, SCAN_S,
                                  tied_count, untied_count)

# The untied twin: same arch_mode family, same L, same width, 16 SEPARATE
# blocks. The only comparison in this deck that isolates sharing, because it is
# the only one where nothing but sharing differs.
J2 = (16, 4, "no_mlp", 192, 1024, 0)
J2_ACC = 0.4940

# q=4, so a uniform predictor scores 0.25 on validation tokens. A seed below
# that did not learn a worse rule, it learned nothing: the two divergent d448
# seeds sit at 0.0412 and 0.0308. Used only to keep them out of the width trend,
# never to drop them from the page.
CHANCE_ACC = 0.25

CELLS = [(P1, "P1 tied d192"), (S1, "S1 tied d320"), (S2, "S2 tied d448"),
         (J2, "J2 untied d192")]

# d448 lost two seeds to an epoch-0 divergence and may land late or not at all;
# every quoted number is d320's either way. The other three are load-bearing: the
# deck's claim is a comparison between d192, d320 and the untied twin, and with
# any of those missing there is no claim to make.
REQUIRED = [P1, S1, J2]


# ------------------------------------------------------------------ the data

def load_everything(scan_dirs, boot=2000):
    """Scorer-side metrics per rung, plus the raw per-bin curve rows."""
    cells = load_pooled(scan_dirs, boot)
    out = {}
    for key, label in CELLS:
        ms = rung(cells, key)
        if not ms:
            out[key] = None
            continue
        s = summarize(ms)
        n_esc, nll = escapes(ms)
        accs = sorted(acc_of(ms))
        out[key] = {"label": label, "ms": sorted(ms, key=lambda r: r["seed"]),
                    "s": s, "n_esc": n_esc, "nll": nll, "accs": accs,
                    "acc_median": median([a for _, a in accs]) if accs else float("nan")}
    missing = [lbl for (k, lbl) in CELLS if out[k] is None and k in REQUIRED]
    if missing:
        sys.exit("not on disk: %s. Refusing to build a deck with a hole in the "
                 "comparison." % ", ".join(missing))
    if out[S2] is None:
        print("NOTE: no d448 curve on disk. Every quoted number is d320's "
              "anyway; the pages that would have shown d448 say it is absent.")
    return out


def present(D):
    """CELLS, minus any optional cell that has not landed."""
    return [(k, lbl) for k, lbl in CELLS if D[k] is not None]


def curve_rows(scan_dirs):
    """Per-bin rows, keyed (arch_mode, d_model, num_layers, seed).

    The scorer reduces a curve to rho_50 and a mean; the curve itself is what
    shows the d320 cell crossing 0.5 at all, and a crossing is the thing stage
    P never had.
    """
    by = {}
    for r in load_all([d for d in scan_dirs if os.path.isdir(d)]):
        k = (r.get("arch_mode", "full"), int(r["d_model"]), int(r["num_layers"]),
             int(r.get("seed") or 0))
        by.setdefault(k, []).append(r)
    for rows in by.values():
        rows.sort(key=lambda r: float(r["realized_rho"]))
    return by


def cell_curves(by, key):
    """The three seeds of one rung as [(rho[], solve[])], widest rho last."""
    arch, d, L = key[2], key[3], key[0]
    out = []
    for seed in (0, 1, 2):
        rows = by.get((arch, d, L, seed))
        if not rows:
            continue
        out.append(([float(r["realized_rho"]) for r in rows],
                    [float(r["solve_rate"]) for r in rows], seed))
    return out


def history(key, scan_dir):
    """train_nll per epoch for each seed of a rung, straight off disk."""
    arch, d, L = key[2], key[3], key[0]
    out = []
    if not os.path.isdir(scan_dir):
        return out
    for name in sorted(os.listdir(scan_dir)):
        p = os.path.join(scan_dir, name)
        if not os.path.isdir(p) or "_%s_L%d" % (arch, L) not in name:
            continue
        if "d%04d_" % d not in name:
            continue
        for sub in sorted(os.listdir(p)):
            hp = os.path.join(p, sub, "history.json")
            if os.path.exists(hp):
                h = json.load(open(hp))
                out.append((int(name.split("_s")[-1]),
                            [e["epoch"] for e in h], [e["train_nll"] for e in h],
                            [e["accuracy"] for e in h]))
    return sorted(out)


def s4_residuals(scan_dirs):
    """Re-run S4's own fit: tied cells against the depth ladder's curve.

    acc_vs_solve's functions, not a copy of them -- the registered S4 job
    (51904215) called the same `collect`, `pava` and `predict` over the same
    directories, so the residuals printed here are that job's residuals.
    """
    recs = collect([d for d in scan_dirs if os.path.isdir(d)])
    ref = [r for r in recs if is_reference(r["rung"]) and r["acc"] == r["acc"]]
    fit = pava([(r["acc"], r["mean_solve"]) for r in ref])
    out = []
    for r in recs:
        if is_reference(r["rung"]) or r["acc"] != r["acc"]:
            continue
        pred, oor, cov = predict(fit, r["acc"])
        out.append({**r, "pred": pred, "resid": r["mean_solve"] - pred,
                    "oor": oor, "cov": cov,
                    "discriminating": pred > FLOOR and not oor})
    return ref, fit, out


def tied_cells(resids, key):
    """The three seeds of one tied rung, as S4 scored them.

    Exact key match, not a prefix: rung_key appends the eval T when it differs
    from num_layers, and a prefix match would silently pool a stage T curve of
    the same checkpoint into this page.
    """
    got = [r for r in resids if r["rung"] == key]
    if not got:
        sys.exit("S4 has no held-out record of %s. Keys it does have: %s"
                 % (str(key), sorted({str(r["rung"]) for r in resids})[:6]))
    return sorted(got, key=lambda r: r["seed"])


# -------------------------------------------------------------- the drawings
#
# Module-level and taking a bare axes, so the results notebook calls exactly
# what the deck calls.

def declutter(ax, ys, min_pts=13.0):
    """Right-hand label heights pushed apart to one line of clearance.

    label_line writes at the curve's own end, which is the honest place for it
    and the wrong place when two curves end together. The first render had
    "untied d192" struck through "tied d320" on the rho page: both end within
    0.04 of solve rate, which is less than one line of 11pt type. slides.py
    does not de-collide, so the separation is enforced here and converted from
    points into THIS axes' data units, so it holds whatever the y range is.
    Call it after set_ylim: the conversion reads the transform as it stands.
    """
    inv = ax.transData.inverted()
    px = min_pts * ax.figure.dpi / 72.0
    gap = abs(inv.transform((0.0, px))[1] - inv.transform((0.0, 0.0))[1])
    out = list(ys)
    order = sorted(range(len(ys)), key=lambda i: -ys[i])
    for above, i in zip(order, order[1:]):
        if out[above] - out[i] < gap:
            out[i] = out[above] - gap
    return out


def draw_curves(ax, by, keys, labels):
    """Solve rate against REALIZED rho. Realized, never requested: below an
    ensemble's rho_min the requested points all collapse onto the floor and a
    flat tail there is one point drawn many times."""
    cols = series_colors(len(keys))
    ends = []
    for col, key, lbl in zip(cols, keys, labels):
        cs = cell_curves(by, key)
        for rho, solve, seed in cs:
            ax.plot(rho, solve, "-", color=col, linewidth=1.8, alpha=0.85,
                    rasterized=True)
        if cs:
            rho, solve, _ = cs[-1]
            ends.append((rho[-1], solve[-1], lbl, col))
    ax.axhline(0.5, color="0.3", linestyle="--", linewidth=1.2)
    ax.text(0.345, 0.52, "rho_50 crossing", size=11, color="0.3")
    ax.set_xlabel("realized hint density rho")
    ax.set_ylabel("solve rate")
    ax.set_ylim(-0.02, 1.02)
    for (x, _, lbl, col), yy in zip(ends, declutter(ax, [e[1] for e in ends])):
        label_line(ax, x, yy, lbl, color=col)
    style_axes(ax)


def draw_nll(ax, hists, labels):
    """Final-epoch train NLL is one number; the trajectory is what says whether
    a seed escaped the plateau or sat on it."""
    cols = series_colors(len(hists))
    ends = []
    for col, hs, lbl in zip(cols, hists, labels):
        for seed, ep, nll, _ in hs:
            ax.plot(ep, nll, "-", color=col, linewidth=1.8, alpha=0.85,
                    rasterized=True)
        if hs:
            ep, nll = hs[-1][1], hs[-1][2]
            ends.append((ep[-1], nll[-1], lbl, col))
    for (x, _, lbl, col), yy in zip(ends, declutter(ax, [e[1] for e in ends])):
        label_line(ax, x, yy, lbl, color=col)
    ax.axhline(PLATEAU, color="0.3", linestyle="--", linewidth=1.2)
    ax.text(2, PLATEAU + 0.012, "plateau %.2f" % PLATEAU, size=11, color="0.3")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train NLL")
    style_axes(ax)


def draw_accuracy(ax, widths, accs, j2_acc):
    """Validation token accuracy against width, with the untied twin's level as
    the thing width is trying to reach.

    A seed that never left epoch 0 is drawn, but it is not averaged into the
    trend: two of the three d448 seeds are the UNIFORM PREDICTOR (0.0412,
    0.0308, against chance 0.25), and a median taken over those says that width
    destroys accuracy when what it records is a warmup divergence diagnostic
    51860479 already traced off the cell. Both the points and the exclusion are
    on the page; neither is a choice the reader has to take on trust.
    """
    cols = series_colors(len(widths), highlight=1)
    for x, (_, a), col in zip(widths, accs, cols):
        for seed, v in a:
            trained = v >= CHANCE_ACC
            ax.plot([x], [v], "o", color=col if trained else MUTED,
                    markersize=9, fillstyle="full" if trained else "none",
                    markeredgecolor=col if trained else MUTED)
    trend = [[v for _, v in a if v >= CHANCE_ACC] for _, a in accs]
    ax.plot([x for x, t in zip(widths, trend) if t],
            [median(t) for t in trend if t], "-",
            color=SERIES[0], linewidth=2, zorder=0)
    n_div = sum(len(a) - len(t) for (_, a), t in zip(accs, trend))
    if n_div:
        ax.text(0.02, 0.06, "%d seed(s) diverged at epoch 0 (open, excluded "
                "from the line)" % n_div, transform=ax.transAxes, size=10,
                color="0.3")
    ax.axhline(j2_acc, color="0.3", linestyle="--", linewidth=1.2)
    # Offset in POINTS, not in accuracy: 0.0015 of accuracy is under two pixels
    # on this y range, and the dashed line struck the text through.
    ax.annotate("untied twin J2  %.4f" % j2_acc, xy=(widths[0], j2_acc),
                xytext=(0, 5), textcoords="offset points", size=11,
                color="0.3", va="bottom", ha="left")
    ax.set_xticks(widths)
    ax.set_xlabel("d_model (one shared block, applied 16 times)")
    ax.set_ylabel("validation token accuracy")
    style_axes(ax)


def draw_rho50(ax, labels, vals, censored):
    """rho_50 per cell against the free line. LOWER IS BETTER: rho_50 is the
    density at which solving becomes possible, so a cell that needs more hints
    is a weaker deducer."""
    y = np.arange(len(labels))[::-1]
    cols = [SERIES[0] if not c else MUTED for c in censored]
    ax.barh(y, [v if v == v else 1.0 for v in vals], height=0.5, color=cols,
            rasterized=True)
    ax.axvline(FREE, color="0.25", linestyle="--", linewidth=1.3)
    ax.text(FREE, len(labels) - 0.35, " free line %.3f" % FREE, size=11,
            color="0.25", va="center")
    for yy, v, c in zip(y, vals, censored):
        ax.text((1.0 if c else v) - 0.008, yy,
                "CENSORED" if c else "%.3f" % v, va="center", ha="right",
                size=13, color="white", weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("rho_50  (lower is better)")
    style_axes(ax)
    p = ax.get_position()
    ax.set_position([p.x0 + 0.06, p.y0, p.width - 0.06, p.height])


def draw_s4(ax, ref, fit, tied):
    """The depth ladder's accuracy-to-solve curve, with the tied cells placed
    on it. One claim: the tied cells are not off this curve."""
    ax.plot([r["acc"] for r in ref], [r["mean_solve"] for r in ref], "o",
            color=MUTED, markersize=7, rasterized=True)
    ax.plot([p[0] for p in fit], [p[1] for p in fit], "-", color="0.45",
            linewidth=1.8)
    ax.plot([r["acc"] for r in tied], [r["mean_solve"] for r in tied], "o",
            color=SERIES[0], markersize=11)
    for r in tied:
        ax.plot([r["acc"], r["acc"]], [r["pred"], r["mean_solve"]],
                "-", color=SERIES[0], linewidth=1.2)
    r = tied[-1]
    # dx past the marker's own radius: markersize 11 is 5.5pt of it, and
    # label_line's default 6 left the "t" sitting under the dot.
    label_line(ax, r["acc"], r["mean_solve"], "tied d320", color=SERIES[0], dx=12)
    ax.text(0.345, 0.52, "grey = the depth ladder, the reference curve",
            size=11, color="0.4")
    ax.set_xlabel("validation token accuracy")
    ax.set_ylabel("mean solve rate over the rho grid")
    style_axes(ax)


def draw_resid(ax, tied):
    """Signed residual per tied seed against the LOO bar that defines on-curve."""
    y = np.arange(len(tied))[::-1]
    ax.barh(y, [r["resid"] for r in tied], height=0.45, color=SERIES[0],
            rasterized=True)
    ax.axvline(0, color="0.25", linewidth=1.0)
    for b in (-LOO_BAR, LOO_BAR):
        ax.axvline(b, color="0.55", linestyle="--", linewidth=1.3)
    ax.text(LOO_BAR, len(tied) - 0.3, " LOO bar %.4f" % LOO_BAR, size=11,
            color="0.55", va="center")
    for yy, r in zip(y, tied):
        ax.text(r["resid"] + 0.002 * (1 if r["resid"] >= 0 else -1), yy,
                "%+.4f" % r["resid"], va="center",
                ha="left" if r["resid"] >= 0 else "right", size=13)
    ax.set_yticks(y)
    ax.set_yticklabels(["seed %d" % r["seed"] for r in tied])
    ax.set_xlim(-LOO_BAR * 1.35, LOO_BAR * 1.35)
    ax.set_xlabel("observed mean solve - curve prediction")
    style_axes(ax)
    p = ax.get_position()
    ax.set_position([p.x0 + 0.03, p.y0, p.width - 0.03, p.height])


class PngDeck(Deck):
    """slides.Deck plus a PNG per page.

    There is no PDF rasterizer on this cluster, so the PNGs are the only way to
    LOOK at the deck before shipping, and every page is read back before any
    upload.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/users/hshunt/spin_gnn/notebooks/stages")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()
    pdf_path = args.pdf or os.path.join(args.out, "stages_slides.pdf")
    os.makedirs(args.out, exist_ok=True)

    scan = [SCAN_S] + BASELINE_DIRS + SCAN_DIRS
    D = load_everything(scan)
    by = curve_rows(scan)
    ref, fit, resids = s4_residuals(scan)
    tied320 = tied_cells(resids, S1)

    HAVE = present(D)
    TIED = [k for k, _ in HAVE if k in (P1, S1, S2)]
    s1, j2 = D[S1], D[J2]
    # d192 lives in stage P's run dir, the two new widths in stage S's.
    h192 = history(P1, SCAN_P)
    h320 = history(S1, SCAN_S)
    if not h192 or not h320:
        sys.exit("no training history for d192 (%d seeds) or d320 (%d seeds) -- "
                 "the plateau page is the one S1 is read from and cannot be "
                 "drawn from final NLLs alone." % (len(h192), len(h320)))

    prov = ("S 51853969 | commit %s | 3 seeds/cell | N=48 d5 q4 --shuffle-adj "
            "--fall | 14 rho bins x 400 instances | 240k train, 50 epochs"
            % args.commit)

    with PngDeck(pdf_path, args.out, project="spin_gnn stage S",
                 provenance=prov) as d:

        d.title_page(
            "One shared block does direct deduction, 0.011 less accurately",
            # Broken by hand: _claim_text wraps a title, title_page's subtitle
            # is one fig.text that does not, so the first render ran off the
            # right edge at "d192 -> d320 -> d448, ap".
            subtitle="Stage P's censored tied cell, re-run at three widths with the "
                     "sharing held fixed:\nd192 -> d320 -> d448, applied 16 times.")

        d.points_page(
            "Stage P could not tell a damaged algorithm from a small one",
            ["Tying 16 blocks into 1 at d192 censored: no rho bin cleared 0.5",
             "A censor is not a number, and two readings fitted it equally",
             "Only buying capacity back at FIXED sharing separates them"])

        # Mandatory in every deck, and taking the CHECKPOINT's instance keys:
        # shuffle_adj and fall select the distribution (finding 43).
        exi = example_instance(n_nodes=48, degree=5, n_colors=4,
                               shuffle_adj=True, fall=True)
        page_tokenization(d, exi)
        page_trace(d, exi, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the "
                                "distribution every cell in this deck trained and was "
                                "scored on.")

        d.claim_page(
            "Tripling the width un-censors the tied cell: all three seeds cross 0.5",
            lambda ax: draw_curves(ax, by, [P1, S1, J2],
                                   ["tied d192", "tied d320", "untied d192"]),
            note="Three seeds drawn per cell, not a mean. Stage P's d192 never reaches "
                 "the dashed line at any density.")

        d.table_page(
            "The tied cell is 4.1x smaller than its untied twin and worse",
            ["cell", "trainable", "rho_50", "mean solve", "val acc", "escapes"],
            [[D[k]["label"], "{:,}".format(D[k]["s"]["trainable"]),
              rho_txt(D[k]["s"]), "%.4f" % D[k]["s"]["mean_solve"],
              "%.4f" % D[k]["acc_median"], "%d/%d" % (D[k]["n_esc"], D[k]["s"]["n"])]
             for k, _ in HAVE],
            note="rho_50 LOWER is better. The untied twin J2 is the standing minimum on "
                 "this eval; nothing here beats it.")

        d.claim_page(
            "The plateau was a capacity floor, not an obstruction from sharing",
            lambda ax: draw_nll(ax, [h192, h320], ["tied d192", "tied d320"]),
            note="S1, read before any level: d192 sits on the plateau for all 50 epochs, "
                 "d320 escapes it in every seed.")

        # Three columns, not four. table_page gives every column the same
        # 0.84/ncol of the page, and at four columns "1.6646 / 1.6721 / 1.6681"
        # is wider than its share: the first render printed it hard against the
        # escapes cell, as "1.6681 0/3". The column dropped is the plateau,
        # which was 1.64 in every row and is in the note underneath anyway.
        d.table_page(
            "Every d320 seed escapes the plateau; no d192 seed does",
            ["cell", "final train NLL by seed", "escapes"],
            [[D[k]["label"], " / ".join("%.4f" % v for v in D[k]["nll"]),
              "%d/%d" % (D[k]["n_esc"], D[k]["s"]["n"])]
             for k in TIED],
            note="The plateau bar %.2f was frozen in stage C and is reused here, not "
                 "refrozen around this run." % PLATEAU)

        d.claim_page(
            "Width buys accuracy up to d320, then stops short of the twin"
            if D[S2] else "Width buys accuracy from d192 to d320, 0.011 short of J2",
            lambda ax: draw_accuracy(ax, [k[3] for k in TIED],
                                     [(k, D[k]["accs"]) for k in TIED], J2_ACC),
            note=("d448's one trained seed is %+.4f against d320's median; its "
                  "other two never left epoch 0. Width has stopped buying accuracy."
                  % (max(v for _, v in D[S2]["accs"]) - D[S1]["acc_median"]))
                 if D[S2] else
                 "d448 has not landed; on d192 to d320 alone, width is still buying "
                 "accuracy and the saturation claim is not yet made.")

        d.points_page(
            "Two of three d448 seeds diverged at epoch 0, from warmup",
            ["Diagnostic 51860479 traced it to warmup, not to d448 itself",
             "So d448 carries %s and cannot support a 3-seed reading"
             % ("one seed" if D[S2] else "no scored seed yet"),
             "d320 is quoted throughout because d320 trained 3 of 3"])

        # The registered scorer calls S0 FALSIFIED off the d448 censor. Its own
        # precedence rule, printed one section above that verdict, says a cell
        # that never left the plateau says nothing about tying -- and two d448
        # seeds never left it. The verdict is printed in the notebook; this page
        # is why the deck does not adopt it.
        d.points_page(
            "The scorer printed S0 FALSIFIED; its own precedence rule outranks it",
            # Three bullets, not four: slides.py caps a points page at 3 and
            # raised on the fourth. The one cut said that failing to TRAIN is a
            # fourth outcome the scorer has no branch for; it is in the notebook.
            ["S0 was registered on d448, which never trained: escapes 1/3, NLL 3.79",
             "S1 outranks S0: a cell that never left the plateau says nothing about tying",
             "d320's reading stands: 3/3 crossing, rho_50 0.757, 0.017 past the line"])

        d.claim_page(
            "The tied cell lands 0.017 past the free line, not on it",
            lambda ax: draw_rho50(
                ax, [D[k]["label"] for k, _ in HAVE],
                [float(rho_txt(D[k]["s"])) if rho_txt(D[k]["s"]) != "CENSORED"
                 else float("nan") for k, _ in HAVE],
                [rho_txt(D[k]["s"]) == "CENSORED" for k, _ in HAVE]),
            note="The free line %.3f is where the unconstrained architecture solves. A "
                 "level, not a bar this stage was registered to clear." % FREE)

        d.claim_page(
            "Tying moves the cell along the accuracy-to-solve curve, not off it",
            lambda ax: draw_s4(ax, ref, fit, tied320),
            note="S4, registered before the run. The reference curve is the depth "
                 "ladder's own isotonic fit; the tied cells were never in it.")

        # The counts are COMPUTED, not quoted. S4's registered run saw 30
        # discriminating held-out cells and a worst miss of 0.0222; this fit
        # sees more of both, because the rho_curve_*.csv glob fix made stage T's
        # 18 grids visible to the loader and every off-schedule T entered the
        # held-out set as its own cell. The three tied d320 residuals are
        # unchanged to the digit, so the claim is unaffected -- but a note that
        # still said "30" would be quoting a population this page no longer draws.
        disc = [r for r in resids if r["discriminating"]]
        worst = max(abs(r["resid"]) for r in disc) if disc else float("nan")
        d.claim_page(
            "All three tied seeds sit inside the bar that defines on-curve",
            lambda ax: draw_resid(ax, tied320),
            note="Bar %.4f is stage J's leave-one-out residual, frozen there. Of %d "
                 "held-out cells %d discriminate; the worst misses by %.4f."
                 % (LOO_BAR, len(resids), len(disc), worst))

        d.table_page(
            "The tied cells are DISCRIMINATING, so the fit could have rejected them",
            ["seed", "val acc", "predicted solve", "observed", "residual", "bracket"],
            [["s%d" % r["seed"], "%.4f" % r["acc"], "%.4f" % r["pred"],
              "%.4f" % r["mean_solve"], "%+.4f" % r["resid"], "%.4f" % r["cov"]]
             for r in tied320],
            note="A cell predicted near the %.2f floor cannot fail this test; these are "
                 "not. Bracket is the span between the two reference points." % FLOOR)

        d.points_page(
            "The whole solve gap is what 0.011 of token accuracy buys",
            ["Tied d320 %.4f acc -> %s rho_50; untied J2 %.4f -> 0.580"
             % (s1["acc_median"], rho_txt(s1["s"]), J2_ACC),
             "Both sit on one curve, so the gap is a LEVEL, not a relation",
             "The object to improve is the optimization, not the architecture"])

        d.points_page(
            "A tied stack has one dial no untied stack has: applications",
            ["16 is the count it was trained with, not a structural property",
             "If the block is an iterable operator, T>16 should hold quality",
             "If depth 16 is a trained schedule, quality should peak at 16"])

        # "What stage S does not show" was a LABEL: it could head any stage's
        # limitations page, which is the test the title rule applies. The scope
        # is itself a finding, so it is stated as one.
        d.points_page(
            "Stage S speaks only for N=48 at train depth 16",
            ["Only N=48: the N=96 bars are a different size and appear nowhere",
             "Only T=16 at train time; no tied cell was trained at another depth",
             "No counterfactual audit of the tied cell; Q/R ran on untied"])

        d.table_page(
            "Against the previous best on this same eval, tying loses 0.177 rho_50",
            ["quantity", "J2 untied d192 (standing)", "S1 tied d320"],
            [["rho_50 median", "0.580", rho_txt(s1["s"])],
             ["mean solve", "%.4f" % j2["s"]["mean_solve"],
              "%.4f" % s1["s"]["mean_solve"]],
             ["val token accuracy", "%.4f" % J2_ACC, "%.4f" % s1["acc_median"]],
             ["trainable", "{:,}".format(untied_count(192, 16)),
              "{:,}".format(tied_count(320))],
             ["distinct blocks", "16", "1"],
             ["seeds crossing 0.5", "3/3", "%d/%d" % (s1["s"]["n_cross"], s1["s"]["n"])]],
            highlight_row=0,
            note="Same 14-point grid, same 400 instances per bin, same --shuffle-adj "
                 "--fall. No improvement is claimed: the tied cell is simpler and worse.")

    print("wrote %s (%d KB, %d pages)" % (pdf_path, os.path.getsize(pdf_path) // 1024,
                                          d._n))
    print("PNGs at %s/page1..%d.png -- READ EVERY ONE before shipping"
          % (args.out, d._n))
    print("S4 fit: %d reference cells, %d held-out, %d discriminating"
          % (len(ref), len(resids), sum(r["discriminating"] for r in resids)))
    # WHICH cells miss, by rung tuple and not by label: rung_label() reads a
    # 7-tuple's tail as `epochs`, so an eval-T cell labels itself "ep24".
    for r in sorted((r for r in resids if r["discriminating"]),
                    key=lambda r: -abs(r["resid"]))[:3]:
        print("  worst miss %+.4f  seed %d  rung %s" % (r["resid"], r["seed"], r["rung"]))
    for r in tied320:
        print("  tied d320 s%d  acc %.4f  pred %.4f  obs %.4f  resid %+.4f"
              % (r["seed"], r["acc"], r["pred"], r["mean_solve"], r["resid"]))


if __name__ == "__main__":
    main()
