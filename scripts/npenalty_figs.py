"""Stage NPENALTY -- figures and numbers for where the N-penalty lives.

Analysis only: reads CSVs already on disk, trains nothing, writes no
checkpoint. Registered in `runs/npenalty_2026-11-21/REGISTERED.md`.

Run in the FIGURE environment, never the torch one:

    module purge; module load devel math viz python/3.12.1
    module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
    export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig

Two sources, both the programme's standing evals:
  * `rho_curve_*.csv` -- the 14-point, 400-per-bin rollout eval. Gives
    solve_rate and dd_mean_T, from which a_eff is derived.
  * stage ENTROPY's `entropy_*of*.csv` -- per-section TEACHER-FORCED NLL and
    accuracy on the same density bins.

a_eff is the N=96 reader's definition, not retyped loosely:
    a_eff = solve_rate ** (1 / n_blank),  n_blank = N - round(N * rho)
using the NOMINAL grid rho, because the generator plants exactly round(N*rho)
hints so n_blank is an integer constant inside a bin.
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line

SCRATCH = os.environ.get("SCRATCH", "/scratch/users/hshunt")
S = os.path.join(SCRATCH, "spin_gnn")
OUT = "/home/users/hshunt/slides/decks/spin-gnn/n-penalty/figures"
DPI = 150

CURVES = [
    ("N=48 s0", 48, f"{S}/sudoku_n48_fall_scan_2026-11-02/rho_curve_t08_*.csv"),
    ("N=96 s0", 96, f"{S}/sudoku_n96_fall_scan_2026-11-02/rho_curve_t00_*.csv"),
    ("N=96 s1", 96, f"{S}/sudoku_n96_fall_scan_2026-11-02/rho_curve_t04_*.csv"),
]
# N=96 depth ladder, all seed 0 but the last, all on the same 14-point eval.
DEPTH = [("L08", "n96_L08_s0"), ("L12", "n96_L12_s0"), ("L16", "n96_L16_s0"),
         ("L12 seed 1", "n96_L12_s1")]
DED = ("reason_node", "round_mark", "reason_color")


def load_curve(pat, N):
    """[(rho, dd_mean_T, n_blank, a_eff)] for bins with a nonzero solve rate."""
    rows = list(csv.DictReader(open(sorted(glob.glob(pat))[0])))
    out = []
    for r in rows:
        rho = float(r["rho"])
        T = float(r["dd_mean_T"])
        sr = float(r["solve_rate"])
        nb = N - round(N * rho)
        if sr <= 0 or nb <= 0:
            continue
        out.append((rho, T, nb, sr ** (1.0 / nb)))
    return out


def interp(pts, xi, x):
    p = sorted(pts, key=lambda t: t[xi])
    xs = [t[xi] for t in p]
    ys = [t[3] for t in p]
    if x < xs[0] or x > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if xs[i] >= x:
            if xs[i] == xs[i - 1]:
                return ys[i]
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


def load_entropy():
    rows = []
    for f in sorted(glob.glob(f"{S}/runs/stageentropy_2026-11-20/entropy_*of*.csv")):
        rows += list(csv.DictReader(open(f)))
    return rows


def dd_T_map():
    """(N, rho) -> dd_mean_T, so teacher-forced rows can be placed on the
    same depth axis as the rollout eval."""
    m = {}
    for N, pat in ((48, CURVES[0][2]), (96, CURVES[1][2])):
        for r in csv.DictReader(open(sorted(glob.glob(pat))[0])):
            m[(N, round(float(r["rho"]), 2))] = float(r["dd_mean_T"])
    return m


# ------------------------------------------------------------------ figures

def fig_two_normalisers(data):
    """The same three cells against depth and against blank count. Exactly 3
    series per panel, which is the palette budget."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for ax, xi, xlab in ((axes[0], 1, "DD depth $T$ (rounds)"),
                         (axes[1], 2, r"$n_{\rm blank}$ (cells to emit)")):
        for col, (name, _N, _p) in zip(SERIES, CURVES):
            pts = sorted(data[name], key=lambda t: t[xi])
            xs = [t[xi] for t in pts]
            ys = [t[3] for t in pts]
            ax.plot(xs, ys, "-o", color=col, linewidth=2.0, markersize=4)
            label_line(ax, xs[-1], ys[-1], name, color=col, dx=7, dy=0)
        style_axes(ax, xlabel=xlab, ylabel=r"per-deduction rate $a_{\rm eff}$")
        ax.set_ylim(0.88, 1.005)
    axes[0].set_xlim(0, 27)
    axes[1].set_xlim(0, 80)
    axes[0].set_title("matched on depth: the curves separate", fontsize=12)
    axes[1].set_title("matched on blank count: the gap shrinks ~5x", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "two_normalisers.png"), dpi=DPI)
    plt.close(fig)


def fig_rollout_vs_tf(data, ent, tmap):
    """Rollout a_eff against teacher-forced accuracy on the same bins. If
    exposure bias were the story these would diverge."""
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    sel = [r for r in ent if r["run_id"] == "n96_L12_s0"]
    rhos = sorted({float(r["rho"]) for r in sel}, reverse=True)
    xs, tf = [], []
    for rho in rhos:
        g = [r for r in sel if abs(float(r["rho"]) - rho) < 1e-9
             and r["section"] in ("reason_node", "round_mark", "reason_color",
                                  "term")]
        n = sum(int(r["n_tokens"]) for r in g)
        a = sum(float(r["accuracy"]) * int(r["n_tokens"]) for r in g) / n
        T = tmap.get((96, round(rho, 2)))
        if T:
            xs.append(T)
            tf.append(a)
    ax.plot(xs, tf, "-o", color=SERIES[0], linewidth=2.0, markersize=5)
    label_line(ax, xs[-1], tf[-1], "teacher-forced\nper-token", color=SERIES[0],
               dx=7, dy=0)
    pts = sorted(data["N=96 s0"], key=lambda t: t[1])
    ax.plot([t[1] for t in pts], [t[3] for t in pts], "-o", color=SERIES[1],
            linewidth=2.0, markersize=5)
    label_line(ax, pts[-1][1], pts[-1][3], r"rollout $a_{\rm eff}$",
               color=SERIES[1], dx=7, dy=-6)
    style_axes(ax, xlabel="DD depth $T$ (rounds)", ylabel="accuracy")
    ax.set_xlim(0, 26)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rollout_vs_tf.png"), dpi=DPI)
    plt.close(fig)


def fig_sections(ent, tmap):
    """Which token class carries the loss, at both sizes."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharey=True)
    for ax, sec in zip(axes, DED):
        for col, rid, N, lab in ((SERIES[0], "n48_L12_s0", 48, "N=48"),
                                 (SERIES[1], "n96_L12_s0", 96, "N=96")):
            pts = []
            for r in ent:
                if r["run_id"] != rid or r["section"] != sec:
                    continue
                T = tmap.get((N, round(float(r["rho"]), 2)))
                if T and r["accuracy"]:
                    pts.append((T, float(r["accuracy"])))
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col,
                    linewidth=2.0, markersize=4.5)
            label_line(ax, pts[-1][0], pts[-1][1], lab, color=col, dx=6, dy=0)
        ax.set_title(sec, fontsize=12.5)
        style_axes(ax, xlabel="DD depth $T$")
        ax.set_xlim(0, 26)
        ax.set_ylim(0.74, 1.01)
    axes[0].set_ylabel("teacher-forced accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sections.png"), dpi=DPI)
    plt.close(fig)


def fig_depth_vs_seed(ent):
    """Depth moves reason_node less than the seed does."""
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    labs, vals = [], []
    for lab, rid in DEPTH:
        g = [r for r in ent if r["run_id"] == rid and r["section"] == "reason_node"
             and abs(float(r["rho"]) - 0.45) < 1e-9]
        if g:
            labs.append(lab)
            vals.append(float(g[0]["accuracy"]))
    cols = [MUTED] * len(vals)
    for i, l in enumerate(labs):
        if l == "L16":
            cols[i] = SERIES[0]
        if l == "L12 seed 1":
            cols[i] = SERIES[1]
    ax.bar(range(len(vals)), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, "%.4f" % v, ha="center", fontsize=11)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs)
    ax.set_ylim(0.78, 0.98)
    style_axes(ax, ylabel="reason_node accuracy at $\\rho=0.45$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "depth_vs_seed.png"), dpi=DPI)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {name: load_curve(pat, N) for name, N, pat in CURVES}
    ent = load_entropy()
    tmap = dd_T_map()

    print("=== a_eff at matched DD depth (the registered normaliser) ===")
    for T in (6, 8, 10, 12):
        vs = [interp(data[n], 1, T) for n, _, _ in CURVES]
        gap = vs[0] - vs[1] if vs[0] and vs[1] else None
        print("  T=%-3d  " % T + "  ".join(
            "%s %s" % (CURVES[i][0], "%.4f" % vs[i] if vs[i] else "-")
            for i in range(3))
            + ("   gap(48->96 s0) %+.4f" % gap if gap else ""))

    print("\n=== a_eff at matched n_blank ===")
    for nb in (22, 26, 30, 32):
        vs = [interp(data[n], 2, nb) for n, _, _ in CURVES]
        gap = vs[0] - vs[1] if vs[0] and vs[1] else None
        print("  nb=%-3d " % nb + "  ".join(
            "%s %s" % (CURVES[i][0], "%.4f" % vs[i] if vs[i] else "-")
            for i in range(3))
            + ("   gap(48->96 s0) %+.4f" % gap if gap else ""))

    fig_two_normalisers(data)
    fig_rollout_vs_tf(data, ent, tmap)
    fig_sections(ent, tmap)
    fig_depth_vs_seed(ent)
    print("\nwrote 4 figures to %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
