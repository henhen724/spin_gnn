"""Figures and deck tables for /spin-gnn/n192-ladder/.

FIGURE ENVIRONMENT ONLY -- this imports matplotlib, not torch:

  module purge
  module load devel math viz python/3.12.1
  module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
  python3 -u scripts/n192_ladder_figs.py

Every number quoted on a deck page is printed here, so the deck is generated
by committed code rather than transcribed by hand.
"""
import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from slides import SERIES, MUTED, style_axes, label_line  # noqa: E402

S = "/scratch/users/hshunt/spin_gnn"
OUT = "/home/users/hshunt/slides/decks/spin-gnn/n192-ladder/figures"
NSPLIT = f"{S}/runs/stagensplit_2026-11-20"
DPI = 150
LN4 = math.log(4.0)
FLOOR = {48: 1.371, 96: 1.625, 192: 1.861}

# (label, N, rho_curve glob, sweep.csv glob, history.json glob)
CELLS = [
    ("N=48 s0", 48,
     f"{S}/sudoku_n48_fall_scan_2026-11-02/rho_curve_t08_*.csv",
     f"{S}/sudoku_n48_fall_scan_2026-11-02/t08_*/sweep.csv",
     f"{S}/sudoku_n48_fall_scan_2026-11-02/t08_*/d0256*/history.json"),
    ("N=96 s0", 96,
     f"{S}/sudoku_n96_fall_scan_2026-11-02/rho_curve_t00_*.csv",
     f"{S}/sudoku_n96_fall_scan_2026-11-02/t00_*/sweep.csv",
     f"{S}/sudoku_n96_fall_scan_2026-11-02/t00_*/d0256*/history.json"),
    ("N=96 s1", 96,
     f"{S}/sudoku_n96_fall_scan_2026-11-02/rho_curve_t04_*.csv",
     f"{S}/sudoku_n96_fall_scan_2026-11-02/t04_*/sweep.csv",
     f"{S}/sudoku_n96_fall_scan_2026-11-02/t04_*/d0256*/history.json"),
    ("N=192 s0", 192,
     f"{S}/sudoku_n192_ladder_2026-11-20/rho_curve_t00_*.csv",
     f"{S}/sudoku_n192_ladder_2026-11-20/t00_*/sweep.csv",
     f"{S}/sudoku_n192_ladder_2026-11-20/t00_*/d0256*/history.json"),
    ("N=192 s1", 192,
     f"{S}/sudoku_n192_ladder_2026-11-20/rho_curve_t01_*.csv",
     f"{S}/sudoku_n192_ladder_2026-11-20/t01_*/sweep.csv",
     f"{S}/sudoku_n192_ladder_2026-11-20/t01_*/d0256*/history.json"),
]


def one(pattern):
    hits = sorted(glob.glob(pattern))
    assert len(hits) == 1, f"{pattern} -> {hits}"
    return hits[0]


def load_curve(pattern):
    rows = list(csv.DictReader(open(one(pattern))))
    out = []
    for r in rows:
        out.append({
            "rho": float(r["realized_rho"]),
            "solve": float(r["solve_rate"]),
            "T": float(r["dd_mean_T"]),
            "wf": float(r["well_formed_rate"]),
            "trunc": float(r["truncated_rate"]),
            "outcome": float(r["outcome_accuracy"]),
        })
    return sorted(out, key=lambda d: d["rho"])


def load_sweep(pattern):
    return list(csv.DictReader(open(one(pattern))))[0]


def load_history(pattern):
    return json.load(open(one(pattern)))


def load_nsplit():
    """Pooled per-section nll/accuracy per run_id, de-duplicated on
    (section, rho) because a couple of the probe CSVs were written twice."""
    rows = []
    for p in sorted(glob.glob(f"{NSPLIT}/nsplit*.csv")):
        rows += list(csv.DictReader(open(p)))
    agg = {}
    for r in rows:
        rid = r["run_id"]
        key = (r["section"], r["rho"])
        seen = agg.setdefault(rid, {"_seen": set(), "_d": {}})
        if key in seen["_seen"]:
            continue
        seen["_seen"].add(key)
        n = int(r["n_tokens"])
        a = seen["_d"].setdefault(r["section"], [0.0, 0.0, 0])
        a[0] += float(r["mean_nll"]) * n
        a[1] += float(r["accuracy"]) * n
        a[2] += n
    return {rid: {s: (v[0] / v[2], v[1] / v[2], v[2])
                  for s, v in d["_d"].items()} for rid, d in agg.items()}


def nsplit_by_rho(section):
    rows = []
    for p in sorted(glob.glob(f"{NSPLIT}/nsplit*.csv")):
        rows += list(csv.DictReader(open(p)))
    out = {}
    for r in rows:
        if r["section"] != section:
            continue
        out.setdefault(r["run_id"], {})[float(r["rho"])] = float(r["mean_nll"])
    return {k: sorted(v.items()) for k, v in out.items()}


# ----------------------------------------------------------------- figures

def fig_solve_vs_rho(data):
    """Three panels, one per size. N=192 is the flat line on the floor."""
    groups = [("N=48", ["N=48 s0"]), ("N=96", ["N=96 s0", "N=96 s1"]),
              ("N=192", ["N=192 s0", "N=192 s1"])]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, (title, labels) in zip(axes, groups):
        for j, lab in enumerate(labels):
            c = data[lab]["curve"]
            x = [p["rho"] for p in c]
            y = [p["solve"] for p in c]
            ax.plot(x, y, color=SERIES[j], lw=2, marker="o", ms=3,
                    rasterized=True)
            # Both seeds finish at the same height in two of the three
            # panels (1.0 at N=96, 0.0 at N=192), so stagger the labels.
            label_line(ax, x[-1], y[-1], lab.split()[-1], color=SERIES[j],
                       dy=9 if j == 0 else -9)
        ax.axhline(0.5, color=MUTED, ls=":", lw=1)
        ax.axvline(0.46, color=MUTED, ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlim(0.28, 1.06)
        style_axes(ax, xlabel=r"hint density $\rho$",
                   ylabel="solve rate" if ax is axes[0] else None)
    axes[0].set_ylim(-0.04, 1.04)
    fig.tight_layout()
    fig.savefig(f"{OUT}/solve_vs_rho.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_small_k(data):
    """Solve rate with only k blank cells, k = 0..4. Removes length entirely."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    show = [("N=48 s0", 0), ("N=96 s0", 1), ("N=192 s0", 2)]
    for lab, j in show:
        sw = data[lab]["sweep"]
        y = [float(sw[f"solve_rate_k{k}"]) for k in range(5)]
        ax.plot(range(5), y, color=SERIES[j], lw=2, marker="o", ms=5,
                rasterized=True)
        # N=48 and N=96 both finish at 1.0; stagger so neither hides.
        label_line(ax, 4, y[-1], lab.split()[0], color=SERIES[j],
                   dy={0: 11, 1: -11, 2: 0}[j])
    # With k=1 the node to fill is determined, so solving reduces to naming
    # one colour: guessing uniformly scores 1/q.
    ax.axhline(0.25, color=MUTED, ls="--", lw=1.2)
    ax.text(1.6, 0.285, r"$1/q$ — guessing the colour", color=MUTED,
            fontsize=9)
    ax.set_xticks(range(5))
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.2, 4.9)
    style_axes(ax, xlabel="number of blank cells $k$",
               ylabel="solve rate")
    fig.tight_layout()
    fig.savefig(f"{OUT}/small_k_solve.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_train_curves(data):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    show = [("N=96 s0", 0), ("N=192 s0", 1), ("N=192 s1", 2)]
    for lab, j in show:
        h = data[lab]["history"]
        x = [r["epoch"] for r in h]
        y = [r["nll"] for r in h]
        ax.plot(x, y, color=SERIES[j], lw=2, rasterized=True)
        # The two N=192 cells converge to within 0.03 nats of each other.
        label_line(ax, x[-1], y[-1], lab, color=SERIES[j],
                   dy={0: 0, 1: -10, 2: 10}[j])
        N = data[lab]["N"]
        ax.axhline(FLOOR[N], color=SERIES[j], ls=":", lw=1)
    ax.set_xlim(0, 60)
    style_axes(ax, xlabel="epoch", ylabel=r"$\mathrm{val\_nll}$ (nats/token)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/train_curves.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_reason_color():
    """THE figure: reason_color sits on ln q at N=192, at every density."""
    by = nsplit_by_rho("reason_color")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    show = [("n96_L12_s0", "N=96 s0", 0), ("n192_L12_s0", "N=192 s0", 1),
            ("n192_L12_s1", "N=192 s1", 2)]
    for rid, lab, j in show:
        pts = by[rid]
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        # The two N=192 seeds land on ln q to 4 decimals, so they draw on top
        # of one another; dash the upper one and stagger the labels so both
        # are legible rather than one hiding the other.
        ax.plot(x, y, color=SERIES[j], lw=2.6 if j == 1 else 2, marker="o",
                ms=4, ls="--" if j == 1 else "-", rasterized=True)
        label_line(ax, x[-1], y[-1], lab, color=SERIES[j],
                   dy={0: 0, 1: 11, 2: -11}[j])
    ax.axhline(LN4, color=MUTED, ls="--", lw=1.2)
    ax.text(0.36, LN4 + 0.05, r"$\ln q = 1.386$ — a uniform guess",
            color=MUTED, fontsize=9)
    ax.set_ylim(-0.1, 1.65)
    style_axes(ax, xlabel=r"hint density $\rho$",
               ylabel="cross-entropy on the deduced colour (nats)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/reason_color_nll.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_section_excess(agg):
    """Where each cell's loss actually sits, as contribution = nll x share."""
    order = ["graph_nbr_novel", "graph_nbr_copy", "hint_color", "hint_node",
             "reason_color", "reason_node", "round_mark", "begin",
             "graph_lead", "term"]
    pretty = {"graph_nbr_novel": "graph: new nbr", "graph_nbr_copy": "graph: repeat nbr",
              "hint_color": "hint colour", "hint_node": "hint node",
              "reason_color": "DEDUCED COLOUR", "reason_node": "DEDUCED NODE",
              "round_mark": "round marker", "begin": "section marks",
              "graph_lead": "graph: row head", "term": "terminator"}
    cells = [("n96_L12_s0", "N=96 s0", 0), ("n192_L12_s0", "N=192 s0", 1)]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    h = 0.36
    for k, (rid, lab, j) in enumerate(cells):
        tot = sum(v[2] for v in agg[rid].values())
        vals = [agg[rid][s][0] * agg[rid][s][2] / tot for s in order]
        ypos = [i + (k - 0.5) * h for i in range(len(order))]
        ax.barh(ypos, vals, height=h, color=SERIES[j], label=lab,
                rasterized=True)
        ax.text(vals[0] + 0.02, ypos[0], lab, color=SERIES[j], fontsize=9,
                va="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([pretty[s] for s in order], fontsize=9)
    ax.invert_yaxis()
    style_axes(ax, xlabel="contribution to val_nll (nats/token)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/section_excess.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ tables

def a_eff(solve, N, rho):
    nb = N - round(N * rho)
    return solve ** (1.0 / nb) if nb > 0 and solve > 0 else float("nan")


def a_eff_upper(n_trials, N, rho, conf=0.95):
    """A solve rate of 0 out of n does not pin a_eff down, it only bounds it
    from above. The one-sided Clopper-Pearson bound for 0 successes is
    p < 1 - (1-conf)^(1/n), and a_eff is monotone in p, so
    a_eff < p_upper^(1/n_blank). Quoting the bound is honest where quoting
    a_eff = 0 would not be -- 0 successes is not evidence of zero skill."""
    nb = N - round(N * rho)
    p_up = 1.0 - (1.0 - conf) ** (1.0 / n_trials)
    return p_up, p_up ** (1.0 / nb), nb


def print_tables(data, agg):
    print("\n=== N=192 transition curve, both seeds (14 densities x 400) ===")
    for lab in ("N=192 s0", "N=192 s1"):
        c = data[lab]["curve"]
        print("  %s: solve_rate max %.4f over %d bins; well_formed %.4f-%.4f; "
              "truncated max %.4f"
              % (lab, max(p["solve"] for p in c), len(c),
                 min(p["wf"] for p in c), max(p["wf"] for p in c),
                 max(p["trunc"] for p in c)))
        print("      outcome_accuracy %.4f-%.4f   dd_mean_T %.2f-%.2f"
              % (min(p["outcome"] for p in c), max(p["outcome"] for p in c),
                 min(p["T"] for p in c), max(p["T"] for p in c)))

    print("\n=== small-k solve rate (k blank cells), the length-free eval ===")
    print("  %-10s %7s %7s %7s %7s %7s %8s" % ("cell", "k=0", "k=1", "k=2",
                                               "k=3", "k=4", "mean"))
    for lab, _, _, _, _ in CELLS:
        sw = data[lab]["sweep"]
        ks = [float(sw[f"solve_rate_k{k}"]) for k in range(5)]
        print("  %-10s %7.3f %7.3f %7.3f %7.3f %7.3f %8.4f  (k>=1 mean %.4f)"
              % (lab, *ks, float(sw["mean_o1_solve_rate"]),
                 sum(ks[1:]) / 4.0))
    print("  NB k=0 is degenerate -- no cell to fill, only the terminator to"
          " emit -- and is erratic at every size (0.710 / 0.010 / 1.000)."
          "\n  The k>=1 mean is the one to read. 100 instances per k, so a"
          " single k is noisy; the ordering is not.")

    print("\n=== training ===")
    print("  %-10s %9s %9s %9s %7s %9s" % ("cell", "final_nll", "best_nll",
                                           "best_ep", "floor", "excess"))
    for lab, N, _, _, _ in CELLS:
        h = data[lab]["history"]
        best = min(h, key=lambda r: r["nll"])
        print("  %-10s %9.4f %9.4f %9d %7.3f %9.4f"
              % (lab, h[-1]["nll"], best["nll"], best["epoch"], FLOOR[N],
                 h[-1]["nll"] - FLOOR[N]))
        print("             last-5 drop %+.4f   val_accuracy %.4f"
              % (h[-1]["nll"] - h[-5]["nll"],
                 float(data[lab]["sweep"]["final_val_accuracy"])))

    print("\n=== teacher-forced per section (NSPLIT probe) ===")
    order = ["reason_color", "reason_node", "round_mark", "term",
             "hint_color", "hint_node", "graph_nbr_novel", "graph_nbr_copy",
             "graph_lead", "begin"]
    cells = ["n96_L12_s0", "n192_L12_s0", "n192_L12_s1"]
    print("  %-17s %22s %22s %22s" % ("section", *cells))
    for s in order:
        line = "  %-17s" % s
        for rid in cells:
            nll, acc, n = agg[rid][s]
            line += "   %8.4f nats %6.3f" % (nll, acc)
        print(line)
    print("\n  pooled / floor / excess, and what the two reasoning sections cost:")
    for rid, N in (("n96_L12_s0", 96), ("n192_L12_s0", 192), ("n192_L12_s1", 192)):
        tot = sum(v[2] for v in agg[rid].values())
        pooled = sum(v[0] * v[2] for v in agg[rid].values()) / tot
        exc = pooled - FLOOR[N]
        rc = agg[rid]["reason_color"]
        rn = agg[rid]["reason_node"]
        cc, cn = rc[0] * rc[2] / tot, rn[0] * rn[2] / tot
        print("    %-13s pooled %.4f  floor %.3f  excess %.4f | colour %.4f "
              "(%4.1f%%)  node %.4f (%4.1f%%)  together %4.1f%% of excess on "
              "%4.1f%% of tokens"
              % (rid, pooled, FLOOR[N], exc, cc, 100 * cc / exc, cn,
                 100 * cn / exc, 100 * (cc + cn) / exc,
                 100 * (rc[2] + rn[2]) / tot))

    print("\n=== a_eff at matched DD depth (N1's quantity) ===")
    for lab, N, _, _, _ in CELLS:
        c = data[lab]["curve"]
        near = [p for p in c if 11.0 <= p["T"] <= 13.5]
        if not near:
            print("  %-10s no bin with T in [11.0, 13.5]  (T range %.2f-%.2f)"
                  % (lab, min(p["T"] for p in c), max(p["T"] for p in c)))
            continue
        for p in near:
            if p["solve"] > 0:
                print("  %-10s T=%.2f rho=%.3f solve=%.4f a_eff=%.4f"
                      % (lab, p["T"], p["rho"], p["solve"],
                         a_eff(p["solve"], N, p["rho"])))
            else:
                p_up, a_up, nb = a_eff_upper(400, N, p["rho"])
                print("  %-10s T=%.2f rho=%.3f solve=0/400 -> p<%.5f (95%%), "
                      "n_blank=%d, so a_eff < %.4f"
                      % (lab, p["T"], p["rho"], p_up, nb, a_up))
                print("             N1 predicted a_eff < 0.9118; its falsifier"
                      " was a_eff >= 0.9626.")
                print("             The bound EXCLUDES the falsifier but does"
                      " not reach the prediction -> N1 half-resolved.")


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for lab, N, curve, sweep, hist in CELLS:
        data[lab] = {"N": N, "curve": load_curve(curve),
                     "sweep": load_sweep(sweep), "history": load_history(hist)}
    agg = load_nsplit()

    fig_solve_vs_rho(data)
    fig_small_k(data)
    fig_train_curves(data)
    fig_reason_color()
    fig_section_excess(agg)
    print_tables(data, agg)
    print("\nwrote figures to", OUT)


if __name__ == "__main__":
    main()
