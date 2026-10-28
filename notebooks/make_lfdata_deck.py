"""Slide deck for job 49705064: the leak-free N=24 task was DATA-BOUND.

Run as `python -m notebooks.make_lfdata_deck` from the repo root.

Carries the tokenization and reasoning-trace pages (hshunt's standing request),
built at shuffle_adj=True so they describe this run's own leak-free instances.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from notebooks.deck_tokenization import MONO, example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

SCRATCH = os.environ["SCRATCH"]
RUN = f"{SCRATCH}/spin_gnn/sudoku_n24_data_2026-10-26"
PREV = f"{SCRATCH}/spin_gnn/sudoku_n24_leakfree_2026-10-26"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sudoku_lfdata_2026-10-28")

BLUE, RED, GREEN, ORANGE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "0.45"
GATE = 0.90
CELLS = ["baseline", "A", "B"]
COLORS = {"baseline": RED, "A": GREEN, "B": BLUE}
LABEL = {"baseline": "80k x 300ep (finding 17)", "A": "240k x 300ep (cell A)",
         "B": "240k x 100ep (cell B)"}


def load():
    pool = pd.read_csv(f"{RUN}/sharedpool_n1500.csv")
    # Row order is the order the checkpoints were passed: reference first.
    pool["cell"] = CELLS
    depth = {"baseline": pd.read_csv(f"{PREV}/depth_n24_leakfree.csv"),
             "A": pd.read_csv(f"{RUN}/depth_data240k_ep300.csv"),
             "B": pd.read_csv(f"{RUN}/depth_data240k_ep100.csv")}
    for d in depth.values():
        d["k"] = (d.solve_rate * d.n).round().astype(int)
    sweep = {"baseline": pd.read_csv(f"{PREV}/sweep.csv").iloc[0],
             "A": pd.read_csv(f"{RUN}/data240k_ep300/sweep.csv").iloc[0],
             "B": pd.read_csv(f"{RUN}/data240k_ep100/sweep.csv").iloc[0]}
    mem = {"A": pd.read_csv(f"{RUN}/memorization_data240k_ep300.csv").iloc[0],
           "B": pd.read_csv(f"{RUN}/memorization_data240k_ep100.csv").iloc[0]}
    return pool.set_index("cell"), depth, sweep, mem


def table(ax, header, rows, colw, fontsize=SMALL, row_colors=None, dy=0.115, rule_dy=0.09):
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
    ax.set_xlim(0, sum(colw))
    ax.set_ylim(0, 1)


def page_title(out, pool):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.82, "The leak-free task was DATA-BOUND.", size=BIG + 6, weight="bold")
    fig.text(0.06, 0.725, "3x the data takes N=24 from 0.854 to 1.000.",
             size=BIG + 6, weight="bold")
    fig.text(0.06, 0.60,
             f"On the SAME 1500-instance shared pool that scored the leak-free\n"
             f"baseline at {pool.loc['baseline', 'solve_rate']:.4f}: n_train 80000 -> 240000 gives "
             f"{pool.loc['A', 'solve_rate']:.4f}, delta {pool.loc['A', 'delta']:+.4f},\n"
             f"{pool.loc['A', 'discordant_won']} discordant instances won and "
             f"{pool.loc['A', 'discordant_lost']} lost, McNemar exact p = "
             f"{pool.loc['A', 'mcnemar_exact_p']:.1e}.\n"
             f"Every instance in the pool, solved.",
             size=MED + 1, va="top")
    fig.text(0.06, 0.385,
             "This is the prerequisite the project has been blocked on. After finding 16 closed the\n"
             "adjacency-order leak, the clean N=24 cell sat at 0.854 -- BELOW the 0.90 gate that\n"
             "defines L* -- so no layer threshold could be measured on the real task at all. It now\n"
             "clears the gate by 10 points at one change: more distinct instances.",
             size=SMALL + 1, va="top", color="0.25")
    fig.text(0.06, 0.195,
             "And the cheap cell got there too. Cell B trains on the same 240000 instances for 100\n"
             "epochs -- 24M samples, exactly the step count of the 0.854 run -- and also scores 1.0000.\n"
             "So the binding constraint was COVERAGE, not gradient steps, and the baseline's 300\n"
             "epochs over 80000 instances were two thirds wasted compute.",
             size=SMALL, va="top", color=GREY)
    footer(fig, "job 49705064, COMPLETED 23:35:04, ExitCode 0:0 | commit 9781a31 | "
                "scripts/slurm/sudoku2_n24_leakfree_data.sbatch | seed 0 | shuffle_adj=True")
    out.savefig(fig)


def page_metrics(out, pool, sweep, mem):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Metrics of record", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "One pool, 1500 instances at seed_start=5e10, scored by all three checkpoints; "
             "paired McNemar exact.", size=SMALL)

    rows = []
    for c in CELLS:
        p, s = pool.loc[c], sweep[c]
        g = "+0.1000" if c == "baseline" else f"{mem[c]['gap']:+.4f}"
        rows.append([LABEL[c],
                     f"{p.solve_rate:.4f}",
                     f"[{p.wilson_lo:.4f}, {p.wilson_hi:.4f}]",
                     "reference" if c == "baseline" else f"{p.delta:+.4f}",
                     "--" if c == "baseline" else
                     f"{int(p.discordant_won)}/{int(p.discordant_lost)}",
                     "--" if c == "baseline" else f"{p.mcnemar_exact_p:.1e}",
                     f"{s.final_val_nll:.4f}", f"{s.final_val_accuracy:.4f}", g])
    ax = fig.add_axes([0.05, 0.60, 0.92, 0.21])
    table(ax, ["cell", "solve_rate", "95% Wilson", "delta", "won/lost", "McNemar p",
               "val_nll", "val_acc", "mem gap"], rows,
          [0.235, 0.105, 0.185, 0.09, 0.10, 0.105, 0.095, 0.09, 0.09],
          fontsize=SMALL - 1, row_colors={0: RED, 1: GREEN, 2: BLUE},
          dy=0.25, rule_dy=0.16)

    fig.text(0.06, 0.545, "THE COMPARISON IS AS TIGHT AS THIS PROJECT CAN MAKE IT",
             size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.505,
             "- Same instance distribution (N=24, degree 5, q=4, rho 0.30-0.95, removal hints,\n"
             "  shuffle_adj=True), so sudoku_agg_eval's check_comparable allows the pool at all.\n"
             "- Same architecture, optimizer and seed: d_model 256, 4 heads, L=12, ff=1024, pre-norm,\n"
             "  lr=1e-3 constant, batch 32, seed 0. n_train and epochs are the only knobs that moved.\n"
             f"- Cells A and B solve the SAME 1500 instances, so they do not merely tie on the rate --\n"
             f"  they agree instance by instance (0 discordant pairs between them).",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.255, "ONE HONEST ASTERISK ON 'IDENTICAL ARCHITECTURE'",
             size=SMALL, weight="bold", va="top", color=ORANGE)
    fig.text(0.06, 0.215,
             f"max_len is computed over the TRAINING pool, and a 240000-instance pool contains a\n"
             f"slightly longer trace than an 80000-instance one: {int(sweep['baseline'].max_len)} -> "
             f"{int(sweep['A'].max_len)} tokens, so n_params goes\n"
             f"{int(sweep['baseline'].n_params):,} -> {int(sweep['A'].n_params):,} "
             f"(+{int(sweep['A'].n_params - sweep['baseline'].n_params)} = two positional embedding "
             f"rows, 0.005%). It also gives the\ndata cells a 2-token longer generation budget. "
             f"Neither can account for +0.146; truncated_rate is\n0.0000 for all three cells, so "
             f"nothing was ever cut off at either budget.",
             size=SMALL - 1, va="top")
    footer(fig, "python -m spin_gnn.sudoku_agg_eval --n-instances 1500 --seed-start 50000000000; "
                "mem gap from sudoku_memorization_eval --n-memorization 600")
    out.savefig(fig)


def page_depth(out, depth):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "The depth cost was a data artefact", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "Pre-registration (d) was the control I most wanted to survive. It did not.", size=SMALL)

    ax = fig.add_axes([0.07, 0.47, 0.39, 0.35])
    for c in CELLS:
        g = depth[c][depth[c].rho_requested == 0.60].sort_values("T")
        lo, hi = [], []
        for _, r in g.iterrows():
            a, b = stats.beta.interval(0.95, r.k + 0.5, r.n - r.k + 0.5)
            # max(0, .) because a bin at exactly 1.000 gets an upper bound that
            # rounds a hair below 1, and errorbar refuses a negative yerr.
            lo.append(max(0.0, r.solve_rate - a)); hi.append(max(0.0, b - r.solve_rate))
        ax.errorbar(g["T"], g.solve_rate, yerr=[lo, hi], fmt="-o", color=COLORS[c],
                    ms=5, lw=1.8, capsize=2, label=LABEL[c])
    ax.axhline(GATE, color="0.35", ls="--", lw=1.2)
    ax.text(4.6, GATE + 0.006, "0.90 gate", size=SMALL - 3, color="0.35")
    ax.set_xlabel("DD depth T", size=SMALL); ax.set_ylabel("solve_rate", size=SMALL)
    ax.set_title("rho=0.60, realized density 0.583 in every bin", size=SMALL + 1)
    ax.set_ylim(0.78, 1.015); ax.set_xticks([2, 3, 4, 5])
    ax.tick_params(labelsize=SMALL - 2); ax.legend(fontsize=SMALL - 3, loc="lower left")

    rows = []
    short = {"baseline": "80k x 300ep", "A": "240k x 300ep", "B": "240k x 100ep"}
    for c in CELLS:
        g = depth[c][depth[c].rho_requested == 0.60].sort_values("T")
        r = np.corrcoef(g["T"], g.solve_rate)[0, 1] if g.solve_rate.nunique() > 1 else float("nan")
        rows.append([short[c], *[f"{v:.4f}" for v in g.solve_rate],
                     "n/a (flat)" if np.isnan(r) else f"{r:+.3f}"])
    ax2 = fig.add_axes([0.52, 0.60, 0.45, 0.22])
    table(ax2, ["", "T=2", "T=3", "T=4", "T=5", "Pearson r"], rows,
          [0.24, 0.13, 0.13, 0.13, 0.13, 0.17], fontsize=SMALL - 1,
          row_colors={0: RED, 1: GREEN, 2: BLUE}, dy=0.26, rule_dy=0.18)
    fig.text(0.52, 0.555, "n per bin: T=2:92, T=3:195, T=4:163, T=5:98 (548 per row)",
             size=SMALL - 3, va="top", color=GREY)

    fig.text(0.52, 0.505, "WHAT I PREDICTED, AND WHY IT WAS WRONG",
             size=SMALL, weight="bold", va="top", color=RED)
    fig.text(0.52, 0.465,
             "Prediction: more data lifts every bin by about the\n"
             "same amount and the slope persists, because finding 13\n"
             "showed the depth cost is serial computation and no\n"
             "amount of data fixes a model that is too shallow.\n"
             "Falsifier: Pearson r at rho=0.60 above -0.5.\n\n"
             "What happened: the deep bins did not just improve,\n"
             "they SATURATED. L=12 was never too shallow for T<=5;\n"
             "it had not seen enough instances to cover the deep ones.\n"
             "Note the pre-registered STATISTIC is uninformative here:\n"
             "cell B's r = -0.775 is three bins at exactly 1.000 plus\n"
             "ONE missed instance in 548, so it does not clear -0.5\n"
             "even though the cost it was built to detect is gone.",
             size=SMALL - 1, va="top")

    fig.text(0.07, 0.395, "THE SAME THING HAPPENS AT rho=0.75.",
             size=SMALL, weight="bold", va="top")
    fig.text(0.07, 0.355,
             "The baseline fell 1.000 / 0.969 / 0.890 over T=1,2,3;\n"
             "both data cells score 1.000 in all three bins. Cell B's\n"
             "only miss anywhere is that one instance at rho=0.60,\n"
             "T=5, so even the cheap cell is within one instance of a\n"
             "clean sweep of all 1096 depth-eval instances.",
             size=SMALL - 1, va="top")
    fig.text(0.07, 0.105,
             "Consequence: the leak-free L*(T) ladder at N=24 now has NOTHING TO MEASURE -- every\n"
             "cell would sit at 1.000. It has to move to N=48 or N=96, which finding 22 already argued.",
             size=SMALL - 1, va="top", color=RED)
    footer(fig, "sudoku_depth_eval --rho 0.75 0.60 --n-pool 600 --min-per-bin 40, same generator "
                "stream as the baseline")
    out.savefig(fig)


def page_prereg(out, pool, sweep, mem):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Pre-registrations, as resolved", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "Written into scripts/slurm/sudoku2_n24_leakfree_data.sbatch before the job ran.",
             size=SMALL)

    rows = [
        ["(a)", "Cell A clears 0.90 on the shared pool", "CONFIRMED",
         f"{pool.loc['A', 'solve_rate']:.4f} vs 0.8540, "
         f"{pool.loc['A', 'delta']:+.4f}, p={pool.loc['A', 'mcnemar_exact_p']:.1e}"],
        ["(b)", "Cell B also beats 0.8540, BY LESS than A", "FALSIFIED\n   (the flagged half)",
         "B beats it by the same +0.1460; A and B\n   agree on all 1500 instances. The\n"
         "   falsifier I named -- 'B >= A, the extra\n   epochs do nothing' -- is what happened."],
        ["(c)", "The memorization gap shrinks below\n      +0.1000", "CONFIRMED",
         f"{mem['A']['gap']:+.4f} (train {mem['A']['train_solve_rate']:.4f} vs held-out\n"
         f"   {mem['A']['heldout_solve_rate']:.4f}, p={mem['A']['p']:.1f}, n=600), both cells"],
        ["(d)", "The depth cost survives: r at rho=0.60\n      stays below -0.5",
         "FALSIFIED\n   (in substance)",
         "All four bins at 1.0000 in cell A, so the\n   cost is gone. The pre-registered r test\n"
         "   cannot say so: cell B's r=-0.775 is one\n   missed instance in 548 and still 'passes'."],
    ]
    ax = fig.add_axes([0.06, 0.30, 0.90, 0.53])
    table(ax, ["", "claim", "verdict", "evidence"], rows,
          [0.05, 0.37, 0.17, 0.41], fontsize=SMALL - 1,
          row_colors={0: GREEN, 1: ORANGE, 2: GREEN, 3: RED}, dy=0.20, rule_dy=0.09)

    fig.text(0.06, 0.245, "MY RECORD ON DATA-BUDGET PREDICTIONS IS NOW 3 FOR 8, ALL ONE DIRECTION.",
             size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.205,
             "The sbatch header said it before the job: 'I have twice underestimated how much data\n"
             "this task wants.' Three times, now -- (b) and (d) are the same mistake again, and (d) is\n"
             "the expensive one, because finding 17's depth claim was the first clean evidence in this\n"
             "project that reasoning depth is what costs, and at N=24 it is now attributable to\n"
             "coverage instead. The claim is not dead, just unmeasured at a budget that can hold it.",
             size=SMALL - 1, va="top")
    footer(fig, "pool never trained or tuned on: seed_start 5e10 vs training seeds 0-239999")
    out.savefig(fig)


def page_next(out, pool):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "What this changes, and what it does not", size=BIG, weight="bold")

    fig.text(0.06, 0.855, "WHAT IT CHANGES", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.815,
             "- The leak-free N=24 task is SOLVED, not merely improved: 1500/1500 on a held-out pool,\n"
             "  with the adjacency-order shortcut of finding 16 removed from training.\n"
             "- The working budget for every future leak-free run is 240000 instances x 100 epochs.\n"
             "  That is the number scripts/slurm/sudoku2_leakfree_lstar.sbatch has been waiting for.\n"
             "- Finding 17's 0.854 is demoted from 'the cost of removing the leak' to 'the cost of\n"
             "  removing the leak AT 80000 instances'. The leak was worth data, not capability.\n"
             "- Finding 17's depth cost at N=24 is withdrawn (see the previous page).",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.555, "WHAT IT DOES NOT", size=SMALL, weight="bold", va="top", color=RED)
    fig.text(0.06, 0.515,
             "- It says nothing about N=48 or N=96, where the traces are 2-4x longer and 240000\n"
             "  instances is 3-12x the GPU time. The budget transfers as a hypothesis, not a fact.\n"
             "- A ceiling result cannot be ranked. Anything at 1.000 on 1500 instances ties anything\n"
             "  else at 1.000, so this pool can no longer separate N=24 cells from each other and\n"
             "  the metric of record has to move to a harder distribution (lower rho, or larger N).\n"
             "- One seed, as always in this project.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.275, "WHAT I WOULD DO NEXT", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.235,
             "1. Move the leak-free L*(T) ladder to N=48 at 240000 x 100ep -- the budget is now known\n"
             "   and the T range there (roughly 3-7) is wide enough to split, unlike N=24's 2-5.\n"
             "2. Re-measure the N=24 leak-free depth cost at a density where it is not at ceiling\n"
             "   (rho=0.45), which is one depth_eval call against the cell B checkpoint.\n"
             "3. Rerun the SAE and probe analyses on cell B: the interpretability results so far were\n"
             "   all read off a model that only solved 85% of the task.",
             size=SMALL - 1, va="top")
    footer(fig, "cell B is the checkpoint to keep: same 1.0000 at a third of the compute "
                "(d0256_h04_L12_ff01024_lr1e-03_pre_shuf in data240k_ep100)")
    out.savefig(fig)


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    pool, depth, sweep, mem = load()
    # shuffle_adj=True and shuffle_events=False: exactly this run's distribution.
    ex = example_instance(n_nodes=24, shuffle_adj=True, pick="deep")
    pdf_path = os.path.join(OUT, "lfdata.pdf")
    with PdfPages(pdf_path) as pdf:
        out = PdfAndPng(pdf, OUT)
        page_title(out, pool)
        page_tokenization(out, ex)
        page_trace(out, ex)
        page_metrics(out, pool, sweep, mem)
        page_depth(out, depth)
        page_prereg(out, pool, sweep, mem)
        page_next(out, pool)
    print(f"{out.n} pages -> {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
