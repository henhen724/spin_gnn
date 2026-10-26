"""Results deck for 49565433 (N=96 plateau) and 49587538 (N=48 ff/depth at the knee).

Both runs finished 2026-10-26 with ExitCode 0:0 (14:41:55 and 01:45:31).

PROVENANCE. spin_gnn/ package at commit 38d503a for both runs (the last commit
touching the package; everything after it is notebooks, STATUS and sbatch
headers). seed=0 everywhere. Metric of record is the shared pool via
sudoku_agg_eval at seed_start=50000000000, n=1500, instance i at seed_start+i,
regenerated per problem size -- so rows at equal N are the same instances and
paired-testable by index, and rows at different N share only the protocol.

Configs, both one change from an existing reference:
  49565433  N=96, degree=5, n_colors=4, rho=0.3, rho_max=0.95, removal hints,
            d_model=256, nhead=4, L=12, ff=1024, pre-norm, batch=32, 150 epochs.
            cell A = --lr-schedule cosine; cell B = --n-train-instances 80000.
            Control is 49524030 (20000, constant lr).
  49587538  N=48, same base, ff=256 at L=4/6/8, lr=1e-3 CONSTANT, n_train=20000.
            Wide (ff=1024) column is 49509466/49523504, already on the pool.

Run as: python -m notebooks.make_two_ceilings_deck
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from spin_gnn.sudoku_agg_eval import mcnemar_exact_p
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

ROOT = os.environ.get("SCRATCH", "") + "/spin_gnn"
OUT = os.path.join(ROOT, "notebooks", "sudoku_two_ceilings_2026-10-26")
PDF = os.path.join(OUT, "two_ceilings_moved.pdf")
THRESH = 0.90

N96 = f"{ROOT}/sudoku_n96_plateau_2026-10-26"
FF = f"{ROOT}/sudoku_ff_depth_tradeoff_2026-10-25_n48"
SCAN = f"{ROOT}/sudoku_layer_scan_2026-10-24_n48"
N48_POOL = f"{ROOT}/sudoku_n48_sharedpool_2026-10-25"

# agg_eval appends " [run_dir]" only to tags that collide across run dirs, so the
# N=96 names carry it (control and cell B share a tag) and the N=48 ones do not.
CTRL = "d0256_h04_L12_ff01024_lr1e-03_pre [sudoku_n96_baseline_2026-10-25]"
CELL_A = "d0256_h04_L12_ff01024_lr1e-03_pre_cos [sudoku_n96_plateau_2026-10-26]"
CELL_B = "d0256_h04_L12_ff01024_lr1e-03_pre [sudoku_n96_plateau_2026-10-26]"

GREY, BLUE, RED, GREEN = "0.45", "#1f77b4", "#d62728", "#2ca02c"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pool(path):
    """{checkpoint: (solve_rate, lo, hi, delta, p)} from a sharedpool CSV."""
    return {r["checkpoint"]: (float(r["solve_rate"]), float(r["wilson_lo"]),
                              float(r["wilson_hi"]), float(r["delta"]),
                              float(r["mcnemar_exact_p"])) for r in read_csv(path)}


def masks(path):
    z = np.load(path, allow_pickle=True)
    return {str(n): z["solved"][i].astype(bool) for i, n in enumerate(z["names"])}


def paired(a, b):
    """(delta, a_won, b_won, exact p) for two boolean masks over the same pool."""
    won, lost = int((a & ~b).sum()), int((b & ~a).sum())
    return a.mean() - b.mean(), won, lost, mcnemar_exact_p(lost, won)


def bins(rows, ckpt, rho):
    """[(T, n, solve_rate)] for one checkpoint at one requested density.

    Matched EXACTLY, not by prefix: "d0256_h04_L12_ff01024_lr1e-03_pre" is a
    prefix of the cosine cell's tag, so a startswith match silently merges the
    two N=96 cells into one series and doubles the bin count.
    """
    out = [(int(r["T"]), int(r["n"]), float(r["solve_rate"])) for r in rows
           if r["checkpoint"] == ckpt and abs(float(r["rho_requested"]) - rho) < 1e-9]
    return sorted(out)


def thin(L):
    return f"d0256_h04_L{L:02d}_ff00256_lr1e-03_pre"


def table(ax, header, rows, colw, fontsize=SMALL, row_colors=None):
    """Monospace-ish table drawn with fig.text; matplotlib's table() fights
    tight_layout on landscape slides."""
    ax.axis("off")
    y = 1.0
    x0 = 0.0
    for j, (h, w) in enumerate(zip(header, colw)):
        ax.text(x0 + sum(colw[:j]), y, h, size=fontsize, weight="bold",
                va="top", ha="left", transform=ax.transAxes)
    ax.plot([0, 1], [y - 0.075, y - 0.075], color="0.6", lw=1.0,
            transform=ax.transAxes, clip_on=False)
    for i, row in enumerate(rows):
        yy = y - 0.13 - i * 0.105
        col = (row_colors or {}).get(i, "black")
        for j, (c, w) in enumerate(zip(row, colw)):
            ax.text(x0 + sum(colw[:j]), yy, c, size=fontsize, va="top",
                    ha="left", color=col, transform=ax.transAxes)


# ---------------------------------------------------------------- pages

def page_title(deck, p96, p48):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.875, "Two ceilings moved at once", ha="center", size=BIG + 11,
             weight="bold")
    fig.text(0.5, 0.795, "N=96 and N=48, 2026-10-26", ha="center", size=MED + 3,
             color=GREY)

    b = p96[CELL_B][0]
    thin8 = p48["d0256_h04_L08_ff00256_lr1e-03_pre"][0]
    lines = [
        ("N=96: the ceiling was TRAINING DATA, not the learning rate.",
         f"4x data takes the shared pool from 0.8267 to {b:.4f} [{p96[CELL_B][1]:.4f}, "
         f"{p96[CELL_B][2]:.4f}] -- the Wilson floor clears 0.90. Cosine lr helps too "
         "(+0.0473) but only half as much, and it does not clear."),
        ("I had that diagnosis backwards, and said so in advance.",
         "Finding 15 argued the plateau was optimization because train sat 0.0289 BELOW "
         "val -- no generalization gap for data to close. Pre-registration (b) said the "
         "data cell would gain less. It gained more, at p=1.6e-07."),
        ("N=48: thin-deep wins at every matched budget, as predicted.",
         f"ff=256 beats ff=1024 at all three matched-parameter pairs, +0.5327 / +0.1007 / "
         f"+0.0480, worst p=3.4e-09. L=8/ff=256 scores {thin8:.4f} -- the best N=48 cell "
         "ever, at 1.95x fewer parameters than the previous best."),
        ("But the knee is NOT a property of the problem alone.",
         "Narrowing the MLP 4x moves the N=48 knee from L=4 to L=6. Depth and MLP width "
         "are partial substitutes, with depth the cheaper currency -- so 'L* is set by the "
         "problem' is false as stated in findings 11/14."),
    ]
    y = 0.705
    for head, body in lines:
        fig.text(0.07, y, head, size=MED + 1, weight="bold", va="top")
        wrapped, cur = [], ""
        for w in body.split():
            if len(cur) + len(w) + 1 > 112:
                wrapped.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        wrapped.append(cur)
        fig.text(0.07, y - 0.042, "\n".join(wrapped), size=SMALL + 1, va="top",
                 color="0.2", linespacing=1.45)
        y -= 0.042 + 0.036 * len(wrapped) + 0.030
    footer(fig, "Jobs 49565433 (14:41:55) and 49587538 (01:45:31), both ExitCode 0:0. "
                "spin_gnn/ at 38d503a, seed=0. Metric of record: shared pool, "
                "seed_start=5e10, n=1500, Wilson intervals and paired McNemar exact.")
    deck.savefig(fig)


def page_n96_pool(deck, p96):
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(16, 9),
                                  gridspec_kw={"width_ratios": [1.0, 1.15]})
    cells = [("control\n20000, const lr", CTRL, GREY, "control, 20k, const"),
             ("cell A\n20000, cosine lr", CELL_A, BLUE, "cell A, 20k, cosine"),
             ("cell B\n80000, const lr", CELL_B, GREEN, "cell B, 80k, const")]
    for i, (lab, key, col, _short) in enumerate(cells):
        sr, lo, hi, _, _ = p96[key]
        ax.bar(i, sr, color=col, width=0.62)
        ax.errorbar(i, sr, yerr=[[sr - lo], [hi - sr]], color="black", lw=2,
                    capsize=9, capthick=2)
        ax.text(i, hi + 0.012, f"{sr:.4f}", ha="center", size=MED, weight="bold")
    ax.axhline(THRESH, color=RED, ls="--", lw=2)
    ax.text(2.46, THRESH, " 0.90", color=RED, size=SMALL, va="center", ha="left")
    ax.set_xticks(range(3))
    ax.set_xticklabels([c[0] for c in cells], size=SMALL + 1)
    ax.set_xlim(-0.6, 2.75)
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("solve_rate, shared pool (n=1500)")
    ax.set_title("N=96: one change each from the control")

    rows = []
    for lab, key, _col, short in cells:
        sr, lo, hi, d, p = p96[key]
        rows.append([short, f"{sr:.4f}", f"[{lo:.4f}, {hi:.4f}]",
                     "--" if key == CTRL else f"{d:+.4f}",
                     "--" if key == CTRL else f"{p:.2g}"])
    table(axt, ["cell", "rate", "95% Wilson", "vs ctrl", "p"],
          rows, [0.33, 0.14, 0.26, 0.14, 0.13], fontsize=SMALL + 1,
          row_colors={2: GREEN})
    axt.text(0, 0.56, "Cell B vs cell A, paired on the same 1500 instances:",
             size=SMALL + 1, weight="bold", transform=axt.transAxes)
    axt.text(0, 0.50, "+0.0533,  156-76 discordant,  p = 1.6e-07", size=MED,
             color=GREEN, transform=axt.transAxes)
    axt.text(0, 0.40, "Pre-registration (b) said B would gain LESS than A.\n"
                      "It gained 2.1x more. The reasoning behind (b) -- that an\n"
                      "absent generalization gap implies data-insensitivity --\n"
                      "was wrong, and page 3 shows why.",
             size=SMALL, color="0.2", va="top", linespacing=1.6,
             transform=axt.transAxes)
    axt.text(0, 0.15, "Pre-registration (c) said neither cell would clear 0.90.\n"
                      "Cell B clears it with the Wilson floor at 0.9131.",
             size=SMALL, color="0.2", va="top", linespacing=1.6,
             transform=axt.transAxes)
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.97])
    footer(fig, "Same 1500 instances for all three cells, so every comparison is paired; "
                "deltas are McNemar exact on the discordant pairs, not a test of "
                "independent proportions.")
    deck.savefig(fig)


def page_n96_curves(deck, hists):
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
    titles = [("control: 20000, constant lr", GREY),
              ("cell A: 20000, cosine lr", BLUE),
              ("cell B: 80000, constant lr", GREEN)]
    for ax, (label, h), (title, col) in zip(axes, hists.items(), titles):
        ep = [e["epoch"] for e in h]
        tr = [e["train_nll"] for e in h]
        va = [e["nll"] for e in h]
        ax.plot(ep, tr, color=col, lw=2.5, label="train_nll")
        ax.plot(ep, va, color=col, lw=2.5, ls="--", label="val_nll")
        gap = tr[-1] - va[-1]
        ax.set_title(title, size=MED, color=col)
        ax.set_xlabel("epoch")
        ax.legend(loc="upper right", fontsize=SMALL)
        ax.annotate(f"train - val = {gap:+.4f}", xy=(0.5, 0.52),
                    xycoords="axes fraction", ha="center", size=MED,
                    weight="bold", color=RED if gap < 0 else GREEN)
        ax.annotate("underfits" if gap < 0 else "fits, then generalizes",
                    xy=(0.5, 0.45), xycoords="axes fraction", ha="center",
                    size=SMALL + 1, color=RED if gap < 0 else GREEN)
        ax.annotate(f"val min {min(va):.4f} @ep{ep[int(np.argmin(va))]}",
                    xy=(0.5, 0.37), xycoords="axes fraction", ha="center",
                    size=SMALL + 1, color="0.25")
    axes[0].set_ylabel("NLL")
    fig.suptitle("Why more data worked: only the 80000 cell stops underfitting",
                 size=BIG, weight="bold")
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.93])
    footer(fig, "The sign of train - val flips only in cell B. At 20000 the N=96 model "
                "never fits its own training set, with or without cosine, so the lever "
                "that helps is the one that makes each epoch see more distinct "
                "instances. The inference 'no overfitting, therefore more data will not "
                "help' does not hold: a model can be data-limited while underfitting.")
    deck.savefig(fig)


def page_n96_depth(deck, rows96):
    """rows96 carries all three cells; the control's bins come from 49524030's
    own depth CSV, re-keyed to "CONTROL" in main() because its tag is identical
    to cell B's."""
    series = [("CONTROL", "control 20000", GREY),
              ("d0256_h04_L12_ff01024_lr1e-03_pre_cos", "cell A cosine", BLUE),
              ("d0256_h04_L12_ff01024_lr1e-03_pre", "cell B 80000", GREEN)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for ax, rho in zip(axes, (0.75, 0.60)):
        for ckpt, lab, col in series:
            bb = bins(rows96, ckpt, rho)
            ax.plot([t for t, _, _ in bb], [sr for _, _, sr in bb], "o-", color=col,
                    lw=2.5, ms=9, label=lab)
        ax.axhline(THRESH, color=RED, ls="--", lw=2)
        ax.set_xlabel("DD trace depth T (rounds)")
        ax.set_title(f"rho = {rho:.2f}", size=MED + 1)
        ax.set_ylim(0.70, 1.03)
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted({t for c, _, _ in series for t, _, _ in bins(rows96, c, rho)}))
    axes[0].set_ylabel("solve_rate in bin")
    axes[0].legend(loc="lower left", fontsize=SMALL)
    for ax in axes:
        ax.text(0.985, (THRESH - 0.70) / (1.03 - 0.70) + 0.006, "0.90", color=RED,
                size=SMALL, ha="right", va="bottom", transform=ax.transAxes)
    b60 = bins(rows96, "d0256_h04_L12_ff01024_lr1e-03_pre", 0.60)
    cleared = [t for t, _, sr in b60 if sr >= THRESH]
    axes[1].text(0.03, 0.13, f"cell B clears 0.90 in ALL {len(cleared)} of {len(b60)} "
                             "rho=0.60 bins\n(T=4..9), and all 4 at rho=0.75.\n"
                             "That is the gate: L*(96) is measurable.",
                 size=SMALL + 1, color=GREEN, va="bottom", linespacing=1.6,
                 transform=axes[1].transAxes)
    a60 = bins(rows96, "d0256_h04_L12_ff01024_lr1e-03_pre_cos", 0.60)
    miss = [t for t, _, sr in a60 if sr < THRESH]
    axes[1].text(0.03, 0.015, f"cell A misses at T={miss[0]} -- 5 of 6 is not the gate.",
                 size=SMALL, color=BLUE, va="bottom", transform=axes[1].transAxes)
    c60 = bins(rows96, "CONTROL", 0.60)
    axes[1].text(0.03, 0.86, f"control clears {sum(sr >= THRESH for _, _, sr in c60)} "
                             f"of {len(c60)}",
                 size=SMALL, color=GREY, va="bottom", transform=axes[1].transAxes)
    fig.suptitle("The gate that unblocks the N=96 layer scan", size=BIG,
                 weight="bold")
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.93])
    footer(fig, "The N=96 layer scan was gated on a cell clearing 0.90 in EVERY rho=0.60 "
                "bin, not merely on aggregate, because an aggregate-only pass buys a "
                "column with no L* in it. Cell B passes, so the scan went in as 49639312 "
                "at n_train=80000 -- cell B's recipe, one change (num_layers).")
    deck.savefig(fig)


def page_n48_pairs(deck, p48, m48):
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(16, 9),
                                  gridspec_kw={"width_ratios": [1.1, 1.0]})
    pairs = [("1.72M", "d0256_h04_L04_ff00256_lr1e-03_pre",
              "d0256_h04_L02_ff01024_lr1e-03_pre", "L=4", "L=2"),
             ("2.51M", "d0256_h04_L06_ff00256_lr1e-03_pre",
              "d0256_h04_L03_ff01024_lr1e-03_pre", "L=6", "L=3"),
             ("3.30M", "d0256_h04_L08_ff00256_lr1e-03_pre",
              "d0256_h04_L04_ff01024_lr1e-03_pre", "L=8", "L=4")]
    w = 0.34
    rows, disc = [], []
    for i, (budget, thinkey, widekey, tl, wl) in enumerate(pairs):
        st, sw = p48[thinkey][0], p48[widekey][0]
        ax.bar(i - w / 2, st, width=w, color=GREEN,
               label="thin-deep (ff=256)" if i == 0 else None)
        ax.bar(i + w / 2, sw, width=w, color=GREY,
               label="wide-shallow (ff=1024)" if i == 0 else None)
        ax.text(i - w / 2, st + 0.015, f"{st:.3f}", ha="center", size=SMALL + 1)
        ax.text(i + w / 2, sw + 0.015, f"{sw:.3f}", ha="center", size=SMALL + 1)
        d, won, lost, p = paired(m48[thinkey], m48[widekey])
        # Short cell labels only: "L=4 / ff=256   0.7113" is 21 characters, which
        # overruns its column at SMALL and overprints the next one. The ff values
        # are in the header and the legend already.
        rows.append([budget, f"{tl}  {st:.4f}", f"{wl}  {sw:.4f}",
                     f"{d:+.4f}", f"{p:.2g}"])
        disc.append(f"{won}-{lost}")
    # The 0.90 rule goes in the legend, not in floating text: every free spot at
    # y=0.90 on this panel is either under the legend or inside a bar.
    ax.axhline(THRESH, color=RED, ls="--", lw=2, label="0.90 threshold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([p[0] + "\nparameters" for p in pairs], size=SMALL + 1)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("solve_rate, shared pool (n=1500)")
    ax.set_title("N=48, matched parameters, one pool", size=MED + 1, pad=14)
    ax.legend(loc="upper center", ncol=3, fontsize=SMALL - 1, frameon=False)

    # Headers carry the ff values in short form; the legend on the same page
    # spells out which column is which.
    table(axt, ["budget", "thin (256)", "wide (1024)", "delta", "p"],
          rows, [0.15, 0.25, 0.26, 0.17, 0.17],
          fontsize=SMALL, row_colors={0: GREEN, 1: GREEN, 2: GREEN})
    axt.text(0, 0.56, "Thin-deep wins all three, replicating finding 13 at N=48.",
             size=SMALL + 1, weight="bold", color=GREEN, transform=axt.transAxes)
    axt.text(0, 0.49,
             "Discordant pairs (thin solves / wide solves): 811-12, 212-61, 111-39.",
             size=SMALL, color="0.2", transform=axt.transAxes)
    axt.text(0, 0.42,
             "Pre-registration (a), the load-bearing one, tested the pair AT the\n"
             "N=48 knee: L=8/ff=256 vs L=4/ff=1024. +0.0480 at p=3.4e-09, so the\n"
             "thin-deep advantage survives at a size where more depth is actually\n"
             "required. Finding 13 is not an N=24 artifact.\n\n"
             "Pre-registration (b) needed > 0.30 at the smallest budget and got\n"
             "+0.5327, matching the +0.4860 seen at N=24.\n\n"
             "Measured parameter counts put the thin side only +0.21% ahead\n"
             "(1718074/1714490, 2509626/2504250, 3301178/3294010), so the margins\n"
             "are 23x to 254x the parameter slack. The sbatch header predicted\n"
             "+0.4% from arithmetic; the real slack is tighter.",
             size=SMALL, color="0.2", va="top", linespacing=1.55,
             transform=axt.transAxes)
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.96])
    footer(fig, "Matched pairs come from 2 thin layers costing 791552 parameters against "
                "1 wide layer at 789760 -- a ratio of 1.9955, both columns sharing a "
                "134970-parameter base, all read off the scan's own n_params column.")
    deck.savefig(fig)


def page_n48_knee(deck, p48, p48wide):
    fig, ax = plt.subplots(figsize=(16, 9))
    wide = [(1, 924730), (2, 1714490), (3, 2504250), (4, 3294010), (8, 6453050),
            (12, 9612090)]
    wx, wy = [], []
    for L, npar in wide:
        key = [k for k in p48wide if f"_L{L:02d}_ff01024" in k and "layer_scan" in k]
        if key:
            wx.append(npar)
            wy.append(p48wide[key[0]][0])
    thin = [(4, 1718074), (6, 2509626), (8, 3301178)]
    tx = [n for _, n in thin]
    ty = [p48[f"d0256_h04_L{L:02d}_ff00256_lr1e-03_pre"][0] for L, _ in thin]

    ax.plot(wx, wy, "o-", color=GREY, lw=2.5, ms=11, label="wide: ff=1024")
    ax.plot(tx, ty, "s-", color=GREEN, lw=2.5, ms=11, label="thin: ff=256")
    # Wide labels below the markers, except the two that would land on the 0.90
    # line; thin labels above. Keeps every label off the threshold rule.
    for x, y, L in zip(wx, wy, [L for L, _ in wide]):
        dy = 20 if L in (8, 12) else -28
        ax.annotate(f"L={L}", (x, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", size=SMALL, color=GREY)
    for x, y, L in zip(tx, ty, [L for L, _ in thin]):
        ax.annotate(f"L={L}", (x, y), textcoords="offset points", xytext=(0, 18),
                    ha="center", size=SMALL, color=GREEN)
    ax.axhline(THRESH, color=RED, ls="--", lw=2)
    ax.text(8.6e5, THRESH + 0.025, "0.90", color=RED, size=SMALL)
    ax.annotate("wide knee\n3.29M", (3294010, 0.9093), textcoords="offset points",
                xytext=(58, -72), size=SMALL + 1, color=GREY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.8))
    ax.annotate("thin knee\n2.51M", (2509626, 0.9300), textcoords="offset points",
                xytext=(-96, 62), size=SMALL + 1, color=GREEN, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8))
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("solve_rate, shared pool (n=1500)")
    ax.set_ylim(-0.11, 1.13)
    ax.legend(loc="lower right", fontsize=MED)
    ax.grid(alpha=0.25, which="both")
    ax.set_title("The knee moves with MLP width: L=4 wide, L=6 thin", size=BIG,
                 weight="bold", pad=26)
    fig.tight_layout(rect=[0.03, 0.09, 0.98, 0.95])
    footer(fig, "Pre-registration (c) said the thin column's knee would also sit at L=4. "
                "It does not: L=4/ff=256 scores 0.7113 and clears 0.90 in none of the "
                "five rho=0.60 bins, while L=6/ff=256 clears all five. So L* in LAYERS "
                "is not fixed by the problem -- it is 4 at ff=1024 and 6 at ff=256. In "
                "PARAMETERS the thin knee is still the cheaper one, 2.51M against 3.29M "
                "(+0.0207, p=0.011), which is why depth remains the better currency "
                "even though it is not the only one.")
    deck.savefig(fig)


def page_n48_lstar(deck, rows48):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    cols = {4: RED, 6: GREEN, 8: BLUE}
    for ax, rho, ylo in zip(axes, (0.75, 0.60), (0.55, 0.42)):
        for L in (4, 6, 8):
            bb = bins(rows48, thin(L), rho)
            ax.plot([t for t, _, _ in bb], [sr for _, _, sr in bb], "o-",
                    color=cols[L], lw=2.5, ms=9, label=f"L={L}")
            for t, n, sr in bb:
                if L == 4 and rho == 0.75 and t == 4:
                    lo, hi = 0.7376, 0.9047
                    ax.errorbar([t], [sr], yerr=[[sr - lo], [hi - sr]], color=RED,
                                lw=2.5, capsize=10, capthick=2.5)
        ax.axhline(THRESH, color="0.2", ls="--", lw=2)
        ax.set_xlabel("DD trace depth T (rounds)")
        ax.set_title(f"rho = {rho:.2f}", size=MED + 1)
        ax.set_ylim(ylo, 1.03)
        ax.set_xticks([t for t, _, _ in bins(rows48, thin(6), rho)])
        ax.grid(alpha=0.25)
        ax.legend(loc="lower left", fontsize=SMALL)
    axes[0].set_ylabel("solve_rate in bin")
    axes[0].text(0.03, 0.30,
                 "L=4 crosses below 0.90 at T=4:\n0.944 / 0.913 / 0.838.\n"
                 "That would be L*(T) = 4, 4, 6 --\nthe first RISING series here.\n"
                 "But that bin holds n=74, and\n62/74 = 0.838 [0.738, 0.905]\n"
                 "straddles 0.90. Not resolvable.",
                 size=SMALL, color=RED, va="bottom", linespacing=1.6,
                 transform=axes[0].transAxes)
    axes[1].text(0.40, 0.03,
                 "L=6 and L=8 clear 0.90 in all five bins;\n"
                 "L=4 clears none (0.61 - 0.79).\n"
                 "So L*(T) = 6, flat across T=3..7.",
                 size=SMALL, color=GREEN, va="bottom", linespacing=1.6,
                 transform=axes[1].transAxes)
    fig.suptitle("L*(T) in the thin column: flat where it is resolvable",
                 size=BIG, weight="bold")
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.93])
    footer(fig, "Pre-registration (d) held at rho=0.60 -- five more flat bins, bringing "
                "the project to 20 bins with no resolvable rise. At rho=0.75 the thin "
                "L=4 cell declines monotonically in T and its T=4 point sits below the "
                "threshold, which is the shape a rising L* would make; with n=74 the "
                "confidence interval still contains 0.90, so it is reported as "
                "suggestive and not counted as a rise.")
    deck.savefig(fig)


def page_scorecard(deck, p48, m48, mscan):
    fig = plt.figure(figsize=(16, 9))
    axt = fig.add_axes([0.04, 0.46, 0.92, 0.46])
    rows = [
        ["49565433 (a)", "cosine: val min >= 0.020 better AND beats pool", "SPLIT",
         "val 1.4450 is 0.0138 better, short of 0.020; pool +0.0473 p=3.5e-05"],
        ["49565433 (b)", "4x data gains LESS than cosine", "FALSIFIED",
         "+0.1007 vs +0.0473; B beats A by +0.0533, p=1.6e-07"],
        ["49565433 (c)", "neither cell clears 0.90", "FALSIFIED",
         "cell B 0.9273, Wilson floor 0.9131 above 0.90"],
        ["49565433 (d)", "a clearing cell clears EVERY rho=0.60 bin", "CONFIRMED",
         "cell B clears all 6 (T=4..9) and all 4 at rho=0.75"],
        ["49587538 (a)", "L=8/ff=256 beats the knee, paired p<0.05", "CONFIRMED",
         "+0.0480, 111-39, p=3.4e-09 -- the load-bearing pair"],
        ["49587538 (b)", "L=4/ff=256 beats L=2/ff=1024 by > 0.30", "CONFIRMED",
         "+0.5327, 811-12, p=6.8e-222"],
        ["49587538 (c)", "the thin column's knee is also at L=4", "FALSIFIED",
         "thin knee is L=6; L=4/ff=256 clears no rho=0.60 bin"],
        ["49587538 (d)", "L*(T) stays flat in T at ff=256", "HELD / UNRESOLVED",
         "flat at rho=0.60 (5 bins); rho=0.75 T=4 straddles 0.90 at n=74"],
    ]
    verdict_color = {"CONFIRMED": GREEN, "FALSIFIED": RED, "SPLIT": "#b8860b",
                     "HELD / UNRESOLVED": "#b8860b"}
    # The verdict cell is left blank in the table and overprinted in its own
    # colour; drawing it twice renders one string on top of the other.
    table(axt, ["job", "pre-registered", "verdict", "numbers"],
          [[r[0], r[1], "", r[3]] for r in rows],
          [0.115, 0.335, 0.155, 0.40], fontsize=SMALL - 1)
    for i, r in enumerate(rows):
        axt.text(0.115 + 0.335, 1.0 - 0.13 - i * 0.105, r[2], size=SMALL - 1,
                 weight="bold", color=verdict_color[r[2]], va="top",
                 transform=axt.transAxes)

    thin8 = p48[thin(8)]
    d, won, lost, p = paired(m48[thin(8)],
                             mscan["d0256_h04_L08_ff01024_lr1e-03_pre "
                                   "[sudoku_layer_scan_2026-10-24_n48]"])
    fig.text(0.04, 0.39, "vs the previous best on the SAME eval", size=MED,
             weight="bold", va="top")
    fig.text(0.04, 0.345,
             f"N=48 previous best: L=8/ff=1024, 0.9533, 6.45M params.\n"
             f"New best:  L=8/ff=256, {thin8[0]:.4f}, 3.30M params.\n"
             f"Paired: {d:+.4f}, {won}-{lost}, p={p:.2f} -- a tie on accuracy, at\n"
             f"1.95x fewer parameters. Against L=12/ff=1024 at 9.61M\n"
             f"it is +0.0180, 71-44, p=0.015.\n"
             f"N=96 previous best: 0.8267. New: 0.9273, +0.1007, p=3e-20.\n"
             f"Project best is still 0.9960 at N=24 (job 48912548); the pool\n"
             f"is regenerated per N, so that row shares the protocol only.",
             size=SMALL - 1, va="top", linespacing=1.7)

    fig.text(0.53, 0.39, "provenance", size=MED, weight="bold", va="top")
    fig.text(0.53, 0.345,
             "spin_gnn/ package at commit 38d503a, both runs. seed=0;\n"
             "eval pool seed_start=50000000000, n=1500.\n"
             "49565433: N=96, d_model=256, nhead=4, L=12, ff=1024, pre-norm,\n"
             "  lr=1e-3, batch=32, 150 epochs, degree=5, n_colors=4, rho=0.3,\n"
             "  rho_max=0.95, removal hints. A = cosine; B = 80000 instances.\n"
             "49587538: N=48, same base, ff=256 at L=4/6/8, constant lr, 20000.\n"
             "Checkpoints are the FINAL epoch, not the best val (STATUS.md).\n"
             "Running now: 49639312 (N=96 layer scan at 80000), 49639313\n"
             "  (N=48 data control at 80000, which removes the budget confound).",
             size=SMALL - 1, va="top", linespacing=1.7)
    fig.suptitle("Scorecard: 3 confirmed, 3 falsified, 2 partial", size=BIG,
                 weight="bold")
    footer(fig, "Every falsified line was pre-registered with its falsifier written "
                "before the run, including (b), where the diagnosis I argued for lost.")
    deck.savefig(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_style()

    p96 = pool(f"{N96}/sharedpool_n1500.csv")
    # The control was depth-evaluated in its own job (49524030) and carries the
    # same checkpoint tag as cell B, so re-key it before merging the two CSVs.
    ctrl_rows = read_csv(f"{ROOT}/sudoku_n96_baseline_2026-10-25/depth_n96.csv")
    for r in ctrl_rows:
        r["checkpoint"] = "CONTROL"
    rows96 = read_csv(f"{N96}/depth_n96_plateau.csv") + ctrl_rows
    p48 = pool(f"{FF}/sharedpool_n1500.csv")
    rows48 = read_csv(f"{FF}/depth_by_ff_n48.csv")
    p48wide = pool(f"{N48_POOL}/sharedpool_n1500.csv")
    m48 = masks(f"{FF}/sharedpool_n1500_per_instance.npz")
    mscan = masks(f"{N48_POOL}/sharedpool_n1500_per_instance.npz")
    hists = {}
    for label, d in (("control", f"{ROOT}/sudoku_n96_baseline_2026-10-25/"
                                 "d0256_h04_L12_ff01024_lr1e-03_pre"),
                     ("A", f"{N96}/d0256_h04_L12_ff01024_lr1e-03_pre_cos"),
                     ("B", f"{N96}/d0256_h04_L12_ff01024_lr1e-03_pre")):
        hists[label] = json.load(open(f"{d}/history.json"))

    with PdfPages(PDF) as pdf:
        deck = PdfAndPng(pdf, OUT)
        page_title(deck, p96, p48)
        page_n96_pool(deck, p96)
        page_n96_curves(deck, hists)
        page_n96_depth(deck, rows96)
        page_n48_pairs(deck, p48, m48)
        page_n48_knee(deck, p48, p48wide)
        page_n48_lstar(deck, rows48)
        page_scorecard(deck, p48, m48, mscan)
    print(f"wrote {PDF} ({os.path.getsize(PDF) / 1e6:.2f} MB, {deck.n} pages)")


if __name__ == "__main__":
    main()
