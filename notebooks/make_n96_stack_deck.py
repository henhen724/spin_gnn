"""Slide deck for job 49641051: the N=96 2x2 over data budget x lr schedule.

Run as `python -m notebooks.make_n96_stack_deck` from the repo root.
Carries the tokenization and reasoning-trace pages per hshunt's standing request;
here they are drawn with shuffle_adj=False, i.e. the LEAKED order these four
checkpoints were trained on, so the leak is visible on the page.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

SCRATCH = os.environ["SCRATCH"]
RUN = f"{SCRATCH}/spin_gnn/sudoku_n96_stack_2026-10-26"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sudoku_n96_stack_2026-10-26")

BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "0.45"

# solve_rate on the shared n=1500 pool, and the final train/val from history.json
CELLS = {           # (n_train, schedule): (solve_rate, lo, hi, val_nll, train_nll)
    (20000, "constant"): (0.8267, 0.8067, 0.8450, 1.4621, 1.4332),
    (20000, "cosine"):   (0.8740, 0.8562, 0.8898, 1.4451, 1.4173),
    (80000, "constant"): (0.9273, 0.9131, 0.9394, 1.3948, 1.4187),
    (80000, "cosine"):   (0.9453, 0.9326, 0.9557, 1.4014, 1.4235),
}
EDGES = [  # (label, delta, won, lost, p)
    ("cosine @ 20000", +0.0473, 180, 109, 3.52e-05),
    ("cosine @ 80000", +0.0180, 88, 61, 3.28e-02),
    ("4x data @ constant", +0.1007, 215, 64, 2.97e-20),
    ("4x data @ cosine", +0.0713, 153, 46, 1.26e-14),
]


def table(ax, header, rows, colw, fontsize=SMALL, row_colors=None, dy=0.115,
          rule_dy=0.09):
    ax.axis("off")
    y = 0.92
    for i, txt in enumerate(header):
        ax.text(sum(colw[:i]), y, txt, size=fontsize, weight="bold", va="top")
    ax.plot([0, sum(colw)], [y - rule_dy, y - rule_dy], color="0.6", lw=1.0)
    for r, row in enumerate(rows):
        y -= dy
        c = (row_colors or {}).get(r, "black")
        for i, txt in enumerate(row):
            ax.text(sum(colw[:i]), y, txt, size=fontsize, va="top", color=c)
    ax.set_xlim(0, sum(colw)); ax.set_ylim(0, 1)


def page_title(out):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.78, "Both N=96 levers are real,", size=BIG + 7, weight="bold")
    fig.text(0.06, 0.68, "and they partly substitute", size=BIG + 7, weight="bold")
    fig.text(0.06, 0.55,
             "Stacking cosine lr onto the 80,000-instance recipe gives a new best at N=96:\n"
             "0.9453 [0.9326, 0.9557], up +0.0180 on the same pool, 88-61, p = 0.033.\n"
             "But cosine was worth +0.0473 at 20,000 and only +0.0180 at 80,000.",
             size=MED + 1, va="top")
    fig.text(0.06, 0.30,
             "The 2x2 over {20000, 80000} x {constant, cosine} is now complete on one shared pool of\n"
             "1500 held-out instances, so all four cells are paired-testable against each other.\n"
             "All four edges are significant; the two gains do not add.",
             size=SMALL + 1, va="top", color="0.25")
    footer(fig, "job 49641051, COMPLETED 11:40:56, ExitCode 0:0 | N=96, d=256, L=12, ff=1024, "
                "150 epochs | leaked token order throughout -- see page 2")
    out.savefig(fig)


def page_grid(out):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "The 2x2, on one pool", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "Every cell is one change from its neighbours; every comparison below is paired "
             "McNemar exact on the same 1500 instances.", size=SMALL)

    ax = fig.add_axes([0.07, 0.40, 0.38, 0.42])
    w, x = 0.36, np.arange(2)
    for k, (sched, c) in enumerate((("constant", GREY), ("cosine", BLUE))):
        vals = [CELLS[(n, sched)][0] for n in (20000, 80000)]
        err = [[v - CELLS[(n, sched)][1] for v, n in zip(vals, (20000, 80000))],
               [CELLS[(n, sched)][2] - v for v, n in zip(vals, (20000, 80000))]]
        ax.bar(x + (k - 0.5) * w, vals, w, yerr=err, capsize=4, color=c, label=sched)
    ax.set_xticks(x); ax.set_xticklabels(["20,000", "80,000"])
    ax.set_xlabel("training instances"); ax.set_ylabel("shared-pool solve_rate")
    ax.set_ylim(0.75, 1.0); ax.legend(fontsize=SMALL - 2, loc="upper left")
    ax.axhline(0.90, color=RED, ls=":", lw=1)
    ax.set_title("new best 0.9453", size=MED)

    rows = [[lab, f"{d:+.4f}", f"{w_}-{l}", f"{p:.2g}"] for lab, d, w_, l, p in EDGES]
    ax = fig.add_axes([0.54, 0.44, 0.42, 0.38])
    table(ax, ["comparison", "delta", "disc.", "p"], rows, [0.42, 0.20, 0.20, 0.18],
          dy=0.145, row_colors={0: BLUE, 1: BLUE, 2: "black", 3: "black"})

    fig.text(0.06, 0.30,
             "Cosine is worth 2.6x less once the data budget is 4x larger (+0.0473 -> +0.0180), and "
             "4x data is worth\nless once the schedule is already cosine (+0.1007 -> +0.0713). "
             "Stacking both gives +0.1187 against the\n+0.1480 the single-lever gains predict if they "
             "added: an interaction of -0.0293. Diminishing returns, not\nindependence -- which also "
             "means neither single-lever number generalises to a different budget.",
             size=SMALL, va="top")
    footer(fig, "reference cell for the CSV is 80000/constant (0.9273); the other three deltas are "
                "recomputed pairwise from sharedpool_n1500_per_instance.npz")
    out.savefig(fig)


def page_nll(out):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "val_nll went the wrong way", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "The new best has a HIGHER val_nll than the cell it beats, and beats it anyway. "
             "Finding 4, now with one change isolated.", size=SMALL)

    ax = fig.add_axes([0.07, 0.40, 0.38, 0.42])
    for (n, sched), (sr, lo, hi, val, _tr) in CELLS.items():
        c = BLUE if sched == "cosine" else GREY
        ax.errorbar(val, sr, yerr=[[sr - lo], [hi - sr]], fmt="o", color=c, capsize=4)
        ax.annotate(f"{n//1000}k {sched}", (val, sr), textcoords="offset points",
                    xytext=(6, -12), size=SMALL - 3)
    a, b = CELLS[(80000, "constant")], CELLS[(80000, "cosine")]
    ax.annotate("", xy=(b[3], b[0]), xytext=(a[3], a[0]),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    ax.set_xlabel("final val_nll"); ax.set_ylabel("shared-pool solve_rate")
    ax.set_title("red arrow: val_nll +0.0066, solve_rate +0.0180", size=MED - 2)

    ax = fig.add_axes([0.56, 0.40, 0.38, 0.42])
    d = pd.read_csv(f"{RUN}/depth_n96_stacked.csv")
    d6 = d[d.rho_requested == 0.6]
    ax.plot(d6["T"], d6.solve_rate, "o-", color=BLUE, lw=2)
    for _, r in d6.iterrows():
        ax.annotate(f"n={int(r.n)}", (r["T"], r.solve_rate), textcoords="offset points",
                    xytext=(0, -15), ha="center", va="top", size=SMALL - 4, color=GREY)
    ax.axhline(0.90, color=RED, ls=":", lw=1.2)
    ax.set_xlabel("DD reasoning depth T"); ax.set_ylabel("solve_rate")
    ax.set_ylim(0.85, 1.01); ax.set_title("rho=0.60: every bin clears 0.90", size=MED - 1)

    fig.text(0.06, 0.30,
             "Final val_nll 1.4014 (new best) vs 1.3948 (cell B) -- worse by 0.0066 -- while "
             "solve_rate is better by\n+0.0180 at p = 0.033. The two runs differ in exactly one "
             "thing, the lr schedule, so this is not a\nconfounded cross-model comparison: it is the "
             "cleanest counterexample yet to reading val_nll as progress.\nThe train-val gap is "
             "+0.0221 against cell B's +0.0239, i.e. unchanged -- and per finding 16 that quantity "
             "is\nnot a fit diagnostic anyway.",
             size=SMALL, va="top")
    footer(fig, "depth bins are 1500 fresh instances at rho=0.60, realized density 0.59375 in every "
                "bin; weakest bin is T=5 at 0.932 (n=146)")
    out.savefig(fig)


def page_prereg(out):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Pre-registrations, as resolved", size=BIG, weight="bold")
    fig.text(0.06, 0.875, "Written in the sbatch header before the job ran. Three of four held.",
             size=SMALL)
    rows = [
        ("(a)", "It beats cell B's 0.9273 paired at p < 0.05",
         "CONFIRMED", "+0.0180, 88-61, p = 0.033. New best at N=96."),
        ("(b)", "The gain is SMALLER than cosine's +0.0473\n"
                "at 20000 (falsifier: a gain >= +0.0473)",
         "CONFIRMED", "+0.0180, i.e. 2.6x smaller. Flagged as low confidence\n"
                      "because it rests on the same train/val reasoning that was\n"
                      "wrong by 2.1x in finding 15 -- this time it held."),
        ("(c)", "Every rho=0.60 bin still clears 0.90",
         "CONFIRMED", "0.988 / 0.932 / 0.972 / 0.971 / 0.971 / 0.977 over\n"
                      "T = 4..9. Cell B's weakest bin was T=9 at 0.907."),
        ("(d)", "The train-val gap widens past +0.0239",
         "FALSIFIED", "+0.0221 -- narrower by 0.0018, i.e. unchanged. The\n"
                      "prediction should not have been made: finding 16 had\n"
                      "already shown this quantity is not a fit diagnostic."),
    ]
    y = 0.80
    for tag, claim, verdict, note in rows:
        c = GREEN if verdict == "CONFIRMED" else RED
        fig.text(0.06, y, tag, size=SMALL, weight="bold", va="top")
        fig.text(0.10, y, claim, size=SMALL - 1, va="top")
        fig.text(0.42, y, verdict, size=SMALL - 1, va="top", weight="bold", color=c)
        fig.text(0.54, y, note, size=SMALL - 1, va="top", color="0.25")
        y -= 0.185
    footer(fig, "(b) is the first time the train/val-gap style of reasoning has predicted "
                "correctly in this project; it was still the weakest-grounded of the four")
    out.savefig(fig)


def page_caveats(out):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "What this is worth", size=BIG, weight="bold")
    left = [("IT DOES SHOW", GREEN),
            ("A new best at N=96 on the metric of", "black"),
            ("record, with all four 2x2 cells paired", "black"),
            ("on one pool.", "black"),
            ("", "black"),
            ("That the two levers are sub-additive,", "black"),
            ("so single-lever gains measured at one", "black"),
            ("budget do not transfer to another.", "black")]
    right = [("IT DOES NOT SHOW", RED),
             ("Anything about reasoning. These four", "black"),
             ("checkpoints were trained on the LEAKED", "black"),
             ("token order (finding 16), so every", "black"),
             ("number here is a DECODING score: the", "black"),
             ("N=96 record collapses to 0.0375 under", "black"),
             ("adj-permutation.", "black"),
             ("", "black"),
             ("One seed per cell. The depth bins are", "black"),
             ("from one draw, n = 43..146 per bin.", "black")]
    for x, col in ((0.06, left), (0.53, right)):
        for i, (line, c) in enumerate(col):
            fig.text(x, 0.84 - 0.052 * i, line, size=SMALL,
                     weight="bold" if i == 0 else "normal", color=c, va="top")
    fig.text(0.06, 0.28,
             "WHAT I WOULD DO NEXT. The N=96 optimisation question is now answered well enough; "
             "the open question worth\ncompute is the same recipe trained leak-free, where N=24 "
             "already costs 0.996 -> 0.854 and reasoning depth\nfinally bites (findings 16-17). "
             "A leak-free N=96 run is the honest version of this page.",
             size=SMALL, va="top")
    footer(fig, "49641051 | out dir sudoku_n96_stack_2026-10-26 | the other three cells are "
                "49524030 (20k), 49565433 (cell A cosine, cell B 80k)")
    out.savefig(fig)


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    ex = example_instance(n_nodes=96, degree=5, n_colors=4, shuffle_adj=False, pick="median")
    pdf_path = os.path.join(OUT, "n96_stack.pdf")
    with PdfPages(pdf_path) as pdf:
        out = PdfAndPng(pdf, OUT)
        page_title(out)
        page_tokenization(out, ex)
        page_trace(out, ex)
        page_grid(out)
        page_nll(out)
        page_prereg(out)
        page_caveats(out)
    print(f"{out.n} pages -> {pdf_path} ({os.path.getsize(pdf_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
