"""Slide deck summarizing the DD-Sudoku transformer side task to date.

Covers the three model-size levers swept over 2026-10-17..19 (depth/ff,
nhead, d_model), the 900-epoch budget test, and the new random-hint DD
pilot that shows N=24 has no hint-density transition to learn.

Usage: python -m scripts.build_sudoku_slides --out notebooks/sudoku_slides.pdf
"""
import argparse
import csv
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

SCRATCH = os.environ["SCRATCH"]
SWEEP = f"{SCRATCH}/spin_gnn/sudoku_hparam_sweep/sweep.csv"
SWEEP900 = f"{SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-19_ep900/sweep.csv"
PILOT = f"{SCRATCH}/spin_gnn/dd_random_hint_timing.csv"

TITLE_FS, BODY_FS = 26, 17


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k == "checkpoint_dir":
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def textpage(pdf, title, lines, subtitle=None):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.88, title, fontsize=TITLE_FS, fontweight="bold", va="top")
    y = 0.76
    if subtitle:
        fig.text(0.06, y, subtitle, fontsize=BODY_FS + 1, style="italic", va="top", color="#444444")
        y -= 0.09
    for line in lines:
        fig.text(0.06, y, line, fontsize=BODY_FS, va="top", family="monospace" if line.startswith("  ") else None)
        y -= 0.062
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="notebooks/sudoku_slides.pdf")
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows, rows900, pilot = load(SWEEP), load(SWEEP900), load(PILOT)
    best = max(rows900, key=lambda r: r["agg_solve_rate"]) if rows900 else None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with PdfPages(args.out) as pdf:
        # --- 1. title + headline -------------------------------------------
        textpage(
            pdf, "DD-Sudoku transformer: where the ceiling actually is",
            [
                "Headline: three independent capacity levers (depth/FF, head count, head width)",
                "and a 3x training budget ALL leave end-to-end solve_rate pinned at 0.72-0.85.",
                "",
                "New, and it reframes the goal: at the model's current N=24 the DD cascade runs",
                "only ~1.7 rounds and full recovery sets in only above rho~0.7 (the trivial,",
                "nearly-fully-hinted corner). There is no hint-density phase transition at N=24",
                "to find -- the transformer has to scale to much larger N before the question",
                "we care about is even well posed.",
                "",
                f"  branch sudoku-dd @ {commit}   |   q=4, d=5, N=24, seed=0   |   2026-10-19",
            ],
        )

        # --- 2. metrics table ----------------------------------------------
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.text(0.06, 0.92, "Metrics: best config per lever (same eval throughout)", fontsize=TITLE_FS - 2,
                 fontweight="bold", va="top")
        sel = []
        for r in rows:
            key = (int(r["d_model"]), int(r["nhead"]), int(r["num_layers"]), int(r["dim_feedforward"]),
                   int(r["epochs"]))
            sel.append((key, r, "300ep sweep"))
        picks = [
            ("L=2  ff=1024  (shallow)", lambda r: r["num_layers"] == 2 and r["dim_feedforward"] == 1024 and r["epochs"] == 150),
            ("L=8  ff=1024  300ep", lambda r: r["num_layers"] == 8 and r["dim_feedforward"] == 1024 and r["epochs"] == 300),
            ("L=12 ff=1024  300ep  d64", lambda r: r["num_layers"] == 12 and r["dim_feedforward"] == 1024 and r["epochs"] == 300 and r["d_model"] == 64 and r["nhead"] == 4),
            ("  + nhead=16", lambda r: r["nhead"] == 16),
            ("  + d_model=128", lambda r: r["d_model"] == 128),
            ("  + d_model=256", lambda r: r["d_model"] == 256),
        ]
        table = [["config", "params", "val_nll", "mean_o1", "k0", "agg_solve_rate"]]
        for label, pred in picks:
            m = [r for r in rows if pred(r)]
            if m:
                r = m[0]
                table.append([label, f"{int(r['n_params']):,}", f"{r['final_val_nll']:.2f}",
                              f"{r['mean_o1_solve_rate']:.3f}", f"{r['solve_rate_k0']:.2f}",
                              f"{r['agg_solve_rate']:.3f}"])
        for r in rows900:
            table.append([f"900ep  d_model={int(r['d_model'])}", f"{int(r['n_params']):,}",
                          f"{r['final_val_nll']:.2f}", f"{r['mean_o1_solve_rate']:.3f}",
                          f"{r['solve_rate_k0']:.2f}", f"{r['agg_solve_rate']:.3f}"])
        ax = fig.add_axes([0.05, 0.12, 0.9, 0.7])
        ax.axis("off")
        t = ax.table(cellText=table[1:], colLabels=table[0], loc="center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(13)
        t.scale(1, 1.9)
        for j in range(len(table[0])):
            t[0, j].set_text_props(fontweight="bold")
            t[0, j].set_facecolor("#dddddd")
        fig.text(0.06, 0.06, "mean_o1 = solve_rate averaged over k=0..4 uncolored nodes (near-trivial end); "
                             "agg = full rho in [0.3, 0.95].", fontsize=12, color="#444444")
        pdf.savefig(fig)
        plt.close(fig)

        # --- 3. depth is the only lever that ever worked --------------------
        fig, axes = plt.subplots(1, 3, figsize=(13.33, 7.5))
        fig.suptitle("The one lever that worked (depth), and two that did not", fontsize=TITLE_FS - 4,
                     fontweight="bold")
        d64 = [r for r in rows if r["d_model"] == 64 and r["nhead"] == 4 and r["dim_feedforward"] == 1024]
        d64.sort(key=lambda r: r["num_layers"])
        ax = axes[0]
        for ep, mk in ((150, "o"), (300, "s")):
            s = [r for r in d64 if r["epochs"] == ep]
            if s:
                ax.plot([r["num_layers"] for r in s], [r["agg_solve_rate"] for r in s], mk + "-",
                        label=f"{ep} epochs", rasterized=True)
        ax.set_xlabel("num_layers"); ax.set_ylabel("agg_solve_rate")
        ax.set_title("depth (ff=1024, d=64)"); ax.legend(fontsize=11); ax.grid(alpha=0.3)

        ax = axes[1]
        h = [r for r in rows if r["num_layers"] == 12 and r["d_model"] == 64 and r["dim_feedforward"] == 1024
             and r["epochs"] == 300]
        h.sort(key=lambda r: r["nhead"])
        ax.semilogx([r["nhead"] for r in h], [r["agg_solve_rate"] for r in h], "o-", base=2, rasterized=True)
        ax.semilogx([r["nhead"] for r in h], [r["mean_o1_solve_rate"] for r in h], "s--", base=2,
                    label="mean_o1", rasterized=True)
        ax.set_xlabel("nhead (fixed d_model=64)"); ax.set_title("head count: monotonically worse")
        ax.legend(fontsize=11); ax.grid(alpha=0.3)

        ax = axes[2]
        w = [r for r in rows if r["num_layers"] == 12 and r["nhead"] == 4 and r["dim_feedforward"] == 1024
             and r["epochs"] == 300]
        w.sort(key=lambda r: r["d_model"])
        ax.semilogx([r["d_model"] for r in w], [r["agg_solve_rate"] for r in w], "o-", base=2,
                    label="agg_solve_rate", rasterized=True)
        ax.semilogx([r["d_model"] for r in w], [r["solve_rate_k0"] for r in w], "^--", base=2,
                    label="k=0 (pure copy)", rasterized=True)
        ax.set_xlabel("d_model"); ax.set_title("head width: agg up, easy cases collapse")
        ax.legend(fontsize=11); ax.grid(alpha=0.3)
        for a in axes:
            a.set_ylim(0, 1.05)
        fig.tight_layout(rect=[0, 0.02, 1, 0.94])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # --- 4. budget test --------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
        fig.suptitle("3x training budget: refuted as the explanation", fontsize=TITLE_FS - 4, fontweight="bold")
        labels, a300, a900, n300, n900 = [], [], [], [], []
        for r9 in sorted(rows900, key=lambda r: r["d_model"]):
            m = [r for r in rows if r["d_model"] == r9["d_model"] and r["num_layers"] == 12
                 and r["nhead"] == 4 and r["dim_feedforward"] == 1024 and r["epochs"] == 300]
            if m:
                labels.append(f"d={int(r9['d_model'])}")
                a300.append(m[0]["agg_solve_rate"]); a900.append(r9["agg_solve_rate"])
                n300.append(m[0]["final_val_nll"]); n900.append(r9["final_val_nll"])
        x = np.arange(len(labels))
        axes[0].bar(x - 0.2, a300, 0.4, label="300 epochs", rasterized=True)
        axes[0].bar(x + 0.2, a900, 0.4, label="900 epochs", rasterized=True)
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels); axes[0].set_ylim(0, 1.0)
        axes[0].set_ylabel("agg_solve_rate"); axes[0].set_title("solve rate: flat (0.720->0.720, 0.773->0.767)")
        axes[0].legend(fontsize=11); axes[0].grid(alpha=0.3, axis="y")
        axes[1].bar(x - 0.2, n300, 0.4, label="300 epochs", rasterized=True)
        axes[1].bar(x + 0.2, n900, 0.4, label="900 epochs", rasterized=True)
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("final val NLL"); axes[1].set_title("val NLL: worse = pure overfitting")
        axes[1].legend(fontsize=11); axes[1].grid(alpha=0.3, axis="y")
        fig.tight_layout(rect=[0, 0.02, 1, 0.94])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # --- 5. the N=24 problem ---------------------------------------------
        if pilot:
            fig, axes = plt.subplots(1, 3, figsize=(13.33, 7.5))
            fig.suptitle("Why there is no transition to find at N=24 (DD ground truth, random hints)",
                         fontsize=TITLE_FS - 5, fontweight="bold")
            rho = [r["rho"] for r in pilot]
            axes[0].plot(rho, [r["resolved_frac"] for r in pilot], "o-", rasterized=True)
            axes[0].set_title("fully resolved fraction"); axes[0].set_ylabel("P(DD solves all N)")
            axes[1].plot(rho, [r["mean_frac_recovered"] for r in pilot], "s-", color="tab:orange", rasterized=True)
            axes[1].set_title("mean fraction of nodes recovered")
            axes[2].plot(rho, [r["mean_n_rounds"] for r in pilot], "^-", color="tab:green", rasterized=True)
            axes[2].set_title("DD cascade depth (rounds)")
            axes[2].axhline(1.7, ls=":", color="red")
            axes[2].text(0.02, 1.75, "peak 1.7 rounds", color="red", fontsize=12)
            for a in axes:
                a.set_xlabel("hint density rho"); a.grid(alpha=0.3)
            fig.text(0.06, 0.02, "Pilot: 40 instances/bin, q=4 d=5 N=24. Full N-scaling sweep (N=24..384) "
                                 "running as Slurm 48579017.", fontsize=12, color="#444444")
            fig.tight_layout(rect=[0, 0.05, 1, 0.94])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        # --- 6. comparison to previous best + next -----------------------------
        prev = max((r for r in rows if r["epochs"] == 300), key=lambda r: r["agg_solve_rate"], default=None)
        lines = []
        if best and prev:
            lines += [
                "Current best vs. previous best, same eval (100 instances/bin, eval seed 2e10):",
                f"  previous  d={int(prev['d_model'])} 300ep : agg={prev['agg_solve_rate']:.3f}  "
                f"mean_o1={prev['mean_o1_solve_rate']:.3f}  k0={prev['solve_rate_k0']:.2f}",
                f"  current   d={int(best['d_model'])} 900ep : agg={best['agg_solve_rate']:.3f}  "
                f"mean_o1={best['mean_o1_solve_rate']:.3f}  k0={best['solve_rate_k0']:.2f}",
                "  -> agg_solve_rate did NOT improve. The 900ep run's real contribution was",
                "     retracting a false claim: d_model=128's bad easy-case score at 300ep was",
                "     undertraining (mean_o1 0.896 -> 0.968), not a width penalty.",
                "",
                "Running now:",
                "  48579017  DD reference curve, N = 24/48/96/192/384, random hints (CPU array)",
                "  48579019  scratchpad encoding: restate each deduction's evidence next to it,",
                "            testing whether long-range retrieval is the real constraint",
                "",
                "Then: scale N until the transition is visible; ~11*N tokens, so N=192 ~ 1700 tokens.",
            ]
        textpage(pdf, "Comparison to previous best, and what is running", lines)

    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
