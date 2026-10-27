"""Deck for 49659502 (the leak-free baseline) and 49639313 (the N=48 data control).

Run as: python -m notebooks.make_leakfree_deck
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style
from spin_gnn.sudoku_agg_eval import wilson_interval

GREY, BLUE, RED, GREEN = "0.45", "#1f77b4", "#d62728", "#2ca02c"
SCR = os.environ["SCRATCH"] + "/spin_gnn"
LF = SCR + "/sudoku_n24_leakfree_2026-10-26"
LEAK = SCR + "/sudoku_memorization_2026-10-26"
N48 = SCR + "/sudoku_n48_data_control_2026-10-26"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/sudoku_leakfree_2026-10-26"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def table(ax, header, rows, colw, fontsize=SMALL, row_colors=None, dy=0.115,
          rule_dy=0.09):
    ax.axis("off")
    y = 0.92
    for i, (txt, w) in enumerate(zip(header, colw)):
        ax.text(sum(colw[:i]), y, txt, size=fontsize, weight="bold", va="top")
    ax.plot([0, sum(colw)], [y - rule_dy, y - rule_dy], color="0.6", lw=1.0)
    for r, row in enumerate(rows):
        y -= dy
        c = (row_colors or {}).get(r, "black")
        for i, (txt, w) in enumerate(zip(row, colw)):
            ax.text(sum(colw[:i]), y, txt, size=fontsize, va="top", color=c)
    ax.set_xlim(0, sum(colw))
    ax.set_ylim(0, 1)


def page_title(d):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.5, 0.78, "Taking away the shortcut", ha="center", size=BIG + 10, weight="bold")
    fig.text(0.5, 0.685, "N=24, one change from the 0.9960 record: --shuffle-adj",
             ha="center", size=MED + 3, color=GREY)
    fig.text(0.5, 0.53,
             "It still solves 85% of held-out instances — and for the first time\n"
             "in this project, solve_rate falls with reasoning depth.",
             ha="center", va="center", size=MED + 6, weight="bold", linespacing=1.6)
    fig.text(0.5, 0.30,
             "0.8540 [0.8352, 0.8710] on the same n=1500 pool protocol as the record.\n"
             "Order-insensitive: shuffling, sorting by id, or handing the leak back\n"
             "all tie identity (p = 0.33 / 0.69 / 0.54).  At fixed hint density,\n"
             "solve_rate falls 0.946 → 0.827 over T = 2..5  (r = −0.998).",
             ha="center", va="center", size=MED, linespacing=1.7)
    footer(fig, "Jobs 49659502 (05:41:35) and 49639313 (03:24:52), both ExitCode 0:0. "
                "Package commit b0baa00; training commit 5103e83; seed 0. "
                "Held-out pool seed_start=50000000000, n=1500.")
    d.savefig(fig)


def page_orderings(d):
    """The before/after that defines the whole result."""
    lf = [r for r in load(LF + "/ordering_and_memorization.csv") if r["part"] == "ordering"]
    leak = [r for r in load(LEAK + "/ordering_counterfactual.csv")
            if int(r["n_nodes"]) == 24]
    order5 = ["identity", "within_block", "shuffle", "byid", "bycolor"]

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("Same graphs, different neighbour order", size=MED + 6, weight="bold", y=0.96)
    for k, (rows, title, color) in enumerate([
            (leak, "the 0.9960 record  (trained on the sorted order)", RED),
            (lf, "the leak-free retrain  (trained on a permuted order)", GREEN)]):
        ax = fig.add_subplot(1, 2, k + 1)
        byo = {r["ordering"]: r for r in rows}
        xs = [o for o in order5 if o in byo]
        v = [float(byo[o]["heldout_solve_rate"]) for o in xs]
        lo = [v[i] - float(byo[o]["heldout_wilson_lo"]) for i, o in enumerate(xs)]
        hi = [float(byo[o]["heldout_wilson_hi"]) - v[i] for i, o in enumerate(xs)]
        ax.bar(range(len(xs)), v, 0.62, yerr=[lo, hi], capsize=4, color=color,
               edgecolor="black", linewidth=0.8)
        for i, val in enumerate(v):
            ax.text(i, val + 0.045, f"{val:.3f}", ha="center", size=SMALL - 1)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=30, ha="right", size=SMALL)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, size=SMALL + 2)
        if k == 0:
            ax.set_ylabel("solve_rate  (n=400 held-out, paired)")
        ax.axhline(0.90, color="0.6", ls="--", lw=1.0)
    fig.text(0.5, 0.135,
             "Every ordering preserves the edge set, the colouring, the hints and the DD trace — "
             "the trace is recomputed on the\nreordered adjacency and asserted equal. "
             "`bycolor` re-instates the leak; the leak-free model does not want it (p=0.54).",
             ha="center", va="center", size=SMALL, linespacing=1.6)
    fig.tight_layout(rect=[0.02, 0.20, 0.98, 0.93])
    footer(fig, "Left: job 49655447, n=400, bycolor not yet implemented when it ran. "
                "Right: job 49659502, n=400, same pool. Paired McNemar exact vs identity.")
    d.savefig(fig)


def page_depth(d):
    rows = load(LF + "/depth_n24_leakfree.csv")
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("Where it fails: deeper instances, at identical hint density",
                 size=MED + 6, weight="bold", y=0.96)
    ax = fig.add_axes([0.08, 0.30, 0.52, 0.56])
    for rho, color, mark in [(0.75, BLUE, "o"), (0.6, GREEN, "s")]:
        sel = [r for r in rows if abs(float(r["rho_requested"]) - rho) < 1e-9]
        T = [int(r["T"]) for r in sel]
        # Recover k from the rounded rate in the CSV and re-derive both the point
        # and the interval from it, so the bar is never inconsistent with the rate.
        ns = [int(r["n"]) for r in sel]
        ks = [round(float(r["solve_rate"]) * n) for r, n in zip(sel, ns)]
        v = [k / n for k, n in zip(ks, ns)]
        ci = [wilson_interval(k, n) for k, n in zip(ks, ns)]
        # clipped because at p=1 the Wilson upper limit is 1 only up to rounding
        err = np.clip(np.array([[v[i] - ci[i][0], ci[i][1] - v[i]]
                                for i in range(len(v))]).T, 0, None)
        ax.errorbar(T, v, yerr=err, marker=mark, color=color, lw=2.2, capsize=4,
                    label=f"rho={rho}  (r={-0.970 if rho == 0.75 else -0.998:+.3f})")
    ax.axhline(0.90, color="0.6", ls="--", lw=1.0)
    ax.set_xlabel("T  (DD rounds the instance needs)")
    ax.set_ylabel("solve_rate")
    ax.set_ylim(0.78, 1.03)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=SMALL)

    ax2 = fig.add_axes([0.64, 0.30, 0.33, 0.56])
    table(ax2, ["rho", "T", "n", "realized rho", "solve"],
          [[r["rho_requested"], r["T"], r["n"], f"{float(r['realized_rho']):.4f}",
            f"{float(r['solve_rate']):.3f}"] for r in rows],
          [0.14, 0.10, 0.14, 0.30, 0.20], dy=0.105)

    fig.text(0.5, 0.15,
             "At rho=0.75 the realized density is IDENTICAL in all three bins (0.7500), so the fall "
             "cannot be a density confound.\nFinding 11 measured L*(T) flat in T — but it measured it "
             "on a task where depth did not bind. It does here.",
             ha="center", va="center", size=SMALL + 1, linespacing=1.7)
    footer(fig, "Job 49659502, sudoku_depth_eval, --n-pool 600 --min-per-bin 40. "
                "Bins too thin to keep: 34 instances at rho=0.75, 52 at rho=0.60. Wilson 95%.")
    d.savefig(fig)


def page_memorization(d):
    lf = [r for r in load(LF + "/ordering_and_memorization.csv") if r["part"] == "memorization"]
    leak = load(LEAK + "/memorization_gap.csv")
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("Pre-registration (d), falsified: the honest task is data-hungrier",
                 size=MED + 5, weight="bold", y=0.96)
    ax = fig.add_axes([0.11, 0.36, 0.38, 0.48])
    labels, gaps, ps = [], [], []
    for r in leak:
        nt = int(r["n_train_instances"] or 20000)
        labels.append(f"N={r['n_nodes']}\n@{nt//1000}k\nleaked")
        gaps.append(float(r["gap"]))
        ps.append(float(r["p"]))
    for r in lf:
        labels.append("N=24\n@80k\nLEAK-FREE")
        gaps.append(float(r["gap"]))
        ps.append(float(r["p"]))
    cols = [RED if p < 0.05 else GREY for p in ps]
    cols[-1] = GREEN
    ax.bar(range(len(gaps)), gaps, 0.6, color=cols, edgecolor="black", linewidth=0.8)
    for i, g in enumerate(gaps):
        ax.text(i, g + (0.006 if g > 0 else -0.016), f"{g:+.4f}", ha="center", size=SMALL - 1)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, size=SMALL - 2)
    ax.set_ylabel("train − held-out  solve_rate", size=SMALL + 2)
    ax.set_ylim(-0.04, 0.13)

    ax2 = fig.add_axes([0.55, 0.36, 0.42, 0.48])
    rows = []
    for r in leak:
        nt = int(r["n_train_instances"] or 20000)
        rows.append([f"N={r['n_nodes']} @{nt//1000}k",
                     f"{float(r['train_solve_rate']):.4f}",
                     f"{float(r['heldout_solve_rate']):.4f}",
                     f"{float(r['gap']):+.4f}", f"{float(r['p']):.2g}"])
    r = lf[0]
    rows.append(["N=24 @80k  shuf", f"{float(r['train_solve_rate']):.4f}",
                 f"{float(r['heldout_solve_rate']):.4f}",
                 f"{float(r['gap']):+.4f}", f"{float(r['p']):.2g}"])
    table(ax2, ["run (all L=12)", "train", "held-out", "gap", "p"], rows,
          [0.30, 0.16, 0.19, 0.19, 0.16], row_colors={len(rows) - 1: GREEN}, dy=0.15)

    fig.text(0.5, 0.185,
             "I predicted no gap at 80000, replicating the leaked N=96 result (−0.0133, p=0.355). "
             "Wrong by +0.1000 at p=2.6e-09.\nRemoving the shortcut makes the task harder, so the "
             "same 80000 instances no longer cover it — the obverse of \"data closes it\".\n"
             "Scored on a recoverable prefix of its own training set: index i is build_instances(cfg, 1, i). "
             "Unpaired z-test.",
             ha="center", va="center", size=SMALL, linespacing=1.7)
    footer(fig, "Leaked rows: job 49655447, n=600 per side. Leak-free row: job 49659502, n=600 per side. "
                "Grey = not resolvable at p<0.05.")
    d.savefig(fig)


def page_n48(d):
    rows = load(N48 + "/sharedpool_n1500.csv")
    def get(L, run):
        for r in rows:
            if f"_L{L:02d}_" in r["checkpoint"] and run in r["checkpoint"]:
                return r
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("Second run: 4x data at N=48 does not move the knee down",
                 size=MED + 6, weight="bold", y=0.96)
    ax = fig.add_axes([0.09, 0.40, 0.40, 0.46])
    xs = np.arange(2)
    for i, (L, color) in enumerate([(3, RED), (4, BLUE)]):
        v, lo, hi = [], [], []
        for run in ["layer_scan", "data_control"]:
            r = get(L, run)
            v.append(float(r["solve_rate"]))
            lo.append(v[-1] - float(r["wilson_lo"]))
            hi.append(float(r["wilson_hi"]) - v[-1])
        ax.errorbar(xs + i * 0.04, v, yerr=[lo, hi], marker="o", lw=2.4, capsize=5,
                    color=color, label=f"L={L}")
        for j, val in enumerate(v):
            ax.text(xs[j] + i * 0.04, val + 0.022, f"{val:.4f}", ha="center", size=SMALL - 1)
    ax.axhline(0.90, color="0.6", ls="--", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(["20000", "80000"])
    ax.set_xlabel("n_train_instances", size=SMALL + 2)
    ax.set_ylabel("solve_rate  (n=1500 shared pool)", size=SMALL + 2)
    ax.set_ylim(0.60, 1.02)
    ax.set_xlim(-0.3, 1.3)
    ax.legend(fontsize=SMALL)

    ax2 = fig.add_axes([0.56, 0.44, 0.41, 0.40])
    table(ax2, ["cell", "solve", "vs L=4 @20k", "p"],
          [["L=4  @20000", "0.9093", "baseline", "—"],
           ["L=4  @80000", "0.9553", "+0.0460", "5.1e-09"],
           ["L=3  @20000", "0.8293", "−0.0800", "4.1e-13"],
           ["L=3  @80000", "0.6560", "−0.2533", "3.0e-85"]],
          [0.33, 0.20, 0.30, 0.17],
          row_colors={3: RED}, dy=0.19)

    fig.text(0.5, 0.20,
             "Pre-registrations (a)-(c) all confirmed: L=3 still misses a rho=0.60 bin so L*(48)=4 stands "
             "unconfounded by data budget,\nL=4 improves, and the 4x gain at N=48 (+0.0460) is smaller "
             "than at N=96 (+0.1007).\nNOT predicted, and unexplained: 4x data COSTS the under-depth "
             "model a quarter of its solve_rate. The pool is held out, so this is\nnot pool memorization. "
             "L=3's r(solve_rate, T) is +0.92 at rho=0.75 and +0.83 at rho=0.60 — it does BETTER on\n"
             "deeper instances, which no implementation of DD can do.",
             ha="center", va="center", size=SMALL, linespacing=1.7)
    footer(fig, "Job 49639313, leaked-order run, so these are decoding scores. Paired McNemar exact "
                "against L=4 @20000 on the same 1500 instances.")
    d.savefig(fig)


def page_scorecard(d):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("Pre-registrations, as resolved", size=MED + 6, weight="bold", y=0.96)
    ax = fig.add_axes([0.05, 0.56, 0.90, 0.32])
    table(ax, ["", "prediction (written before the run)", "outcome"],
          [["(a)", "below 0.9960 at p<0.05 but above 0.50 — a cost, not a collapse",
            "CONFIRMED   0.8540 [0.8352, 0.8710]"],
           ["(b)", "order-insensitive across every colour-free ordering",
            "CONFIRMED   p = 0.66 / 0.33 / 0.69"],
           ["(c)", "re-instating the leak (bycolor) does not help it",
            "CONFIRMED   −0.0100, p = 0.54"],
           ["(d)", "no memorization gap at 80000, replicating 49655447",
            "FALSIFIED   +0.1000, p = 2.6e-09"]],
          [0.05, 0.56, 0.39], fontsize=SMALL,
          row_colors={0: GREEN, 1: GREEN, 2: GREEN, 3: RED}, dy=0.22)
    fig.text(0.06, 0.46, "What (c) settles, and it is the expensive question:",
             size=SMALL + 1, weight="bold", va="top")
    fig.text(0.06, 0.405,
             "The model does not reach for the shortcut when it is offered, so the ENCODING is fine: a "
             "set-based adjacency\nrepresentation is not needed. Only the generator had to change, and "
             "it already has (5103e83).",
             size=SMALL, linespacing=1.8, va="top")
    fig.text(0.06, 0.265, "What is still owed:", size=SMALL + 1, weight="bold", va="top")
    fig.text(0.06, 0.21,
             "Every solve_rate in STATUS.md except this one is a decoding score, and shuffle_adj is in "
             "agg_eval's INSTANCE_KEYS,\nso no _shuf cell pools with an existing checkpoint — "
             "re-measuring a result means retraining it. Depth now binds\n(page 3), so the leak-free "
             "L*(T) series at N=24 is the run finding 11 was trying to be.",
             size=SMALL, linespacing=1.8, va="top")
    footer(fig, "My record on mechanism predictions in this project is now 3 for 5. "
                "49702811 is running the SAE on this checkpoint against the leaked record.")
    d.savefig(fig)


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    pdf_path = OUT + "/leakfree_baseline.pdf"
    with PdfPages(pdf_path) as pdf:
        d = PdfAndPng(pdf, OUT)
        page_title(d)
        page_orderings(d)
        page_depth(d)
        page_memorization(d)
        page_n48(d)
        page_scorecard(d)
    print(f"wrote {pdf_path}  ({os.path.getsize(pdf_path)/1e6:.2f} MB, {d.n} pages)")


if __name__ == "__main__":
    main()
