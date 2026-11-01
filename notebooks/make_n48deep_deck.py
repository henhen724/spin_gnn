"""Slide deck for the N=48 leak-free deep rungs (jobs 50527633, 50527634).

    module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 \
        py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 \
        py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
    python3 ~/spin_gnn/notebooks/make_n48deep_deck.py

Eight landscape pages to $SCRATCH/spin_gnn/notebooks/sudoku_n48deep_2026-10-31/,
PDF plus one PNG per page (nothing on this cluster rasterizes a PDF, so the PNGs
are the only way to read the deck back before sending it).
"""
import glob
import math
import os
import re
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.expanduser("~/spin_gnn"))
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import (BIG, MED, SMALL, PdfAndPng, footer, set_style,
                                  textpage)

S = os.environ["SCRATCH"] + "/spin_gnn"
OUT = f"{S}/notebooks/sudoku_n48deep_2026-10-31"
DIRS = [f"{S}/sudoku_n48_leakfree_lstar_2026-10-29",   # 50099945: L=6, 4, 3
        f"{S}/sudoku_n48_leakfree_lstar_2026-10-31"]   # 50527633: L=12, 50527634: L=8
GATE, LADDER = 0.90, [12, 8, 6, 4, 3]
GREEN, RED, AMBER, GREY = "#276749", "#9b2c2c", "#975a16", "0.35"
COL = {12: "#2b6cb0", 8: "#c53030", 6: "#975a16", 4: "#4a5568", 3: "#718096"}


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def ca_trend(ks, ns, xs):
    ks, ns, xs = np.asarray(ks, float), np.asarray(ns, float), np.asarray(xs, float)
    N, K = ns.sum(), ks.sum()
    pbar, xbar = K / N, (ns * xs).sum() / N
    var = pbar * (1 - pbar) * (ns * (xs - xbar) ** 2).sum()
    return (ks * (xs - xbar)).sum() / np.sqrt(var) if var > 0 else float("nan")


def load():
    lab = lambda c: int(re.search(r"_L(\d+)_", c).group(1))
    solved, seen = {}, {}
    for d in DIRS:
        for p in sorted(glob.glob(f"{d}/sharedpool_n1500_*_per_instance.npz")):
            job = re.search(r"_(\d+)_per_instance", p).group(1)
            z = np.load(p, allow_pickle=True)
            for i, name in enumerate(z["names"]):
                k, v = lab(str(name)), np.asarray(z["solved"][i], bool)
                if k in solved:
                    assert (solved[k] == v).all(), f"L={k} disagrees, {job} vs {seen[k]}"
                else:
                    solved[k], seen[k] = v, job
    rows = []
    for L in LADDER:
        if L not in solved:
            continue
        v = solved[L]
        lo, hi = wilson(int(v.sum()), v.size)
        rows.append(dict(L=L, n=v.size, rate=v.mean(), lo=lo, hi=hi))
    pool = pd.DataFrame(rows)

    bins = []
    for d in DIRS:
        bins += [pd.read_csv(p) for p in sorted(glob.glob(f"{d}/depth_lstar_*.csv"))]
    bins = pd.concat(bins, ignore_index=True)
    bins["L"] = bins.checkpoint.map(lab)
    bins = bins.drop_duplicates(subset=["checkpoint", "rho_requested", "T"])
    return dict(pool=pool, solved=solved, bins=bins)


def clears(bins, rho):
    sub = bins[bins.rho_requested == rho]
    nb = sub["T"].nunique()
    return {int(L): int(((sub.L == L) & (sub.solve_rate >= GATE)).sum())
            for L in sorted(sub.L.unique(), reverse=True)}, nb


def hard_rho(bins):
    """The density with the most depth bins -- the one L* is read on."""
    return bins.groupby("rho_requested")["T"].nunique().idxmax()


def page_verdict(o, D):
    rho = hard_rho(D["bins"])
    cl, nb = clears(D["bins"], rho)
    pool = D["pool"].set_index("L")
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.92, "How many layers does the honest N=48 task need?",
             ha="center", size=BIG + 3, weight="bold")
    fig.text(0.5, 0.862,
             "Leak-free (adjacency shuffled), 240000 instances x 100 epochs, ff=1024.\n"
             "Two weeks ago the answer was \"more than 6, we do not know how many\".",
             ha="center", va="top", size=MED, color="0.3", linespacing=1.5)
    fig.text(0.5, 0.715,
             "L* lies in (8, 12]. The ladder is closed.",
             ha="center", size=BIG + 2, weight="bold", color="#2b6cb0")

    ax = fig.add_axes([0.08, 0.17, 0.40, 0.44])
    ls = sorted(cl)
    ax.bar([str(L) for L in ls], [cl[L] for L in ls],
           color=[COL.get(L, GREY) for L in ls], rasterized=True)
    ax.axhline(nb, color="k", ls="--", lw=2)
    ax.text(-0.45, nb + 0.12, f"all {nb} bins = clears the gate", size=SMALL)
    ax.set_ylim(0, nb + 0.9)
    ax.set_xlabel("num_layers")
    ax.set_ylabel(f"depth bins at or above {GATE:.2f}")
    ax.set_title(f"density {rho:.2f}, {nb} bins of equal density", size=MED)

    ax = fig.add_axes([0.57, 0.17, 0.38, 0.44])
    t = D["pool"].sort_values("L")
    ax.errorbar(t.L, t.rate, yerr=[t.rate - t.lo, t.hi - t.rate], fmt="o-", lw=2.5,
                ms=10, capsize=5, color="#2b6cb0", rasterized=True)
    ax.axhline(GATE, color="k", ls="--", lw=2)
    ax.text(t.L.min() - 0.3, GATE + 0.03, "0.90", size=SMALL)
    ax.set_xticks(sorted(t.L))
    ax.set_xlabel("num_layers")
    ax.set_ylabel("shared-pool solve rate (n=1500)")
    ax.set_ylim(-0.05, 1.12)
    ax.set_title("the metric of record, same 1500 instances", size=MED)

    got = ", ".join(f"L={L} {pool.rate[L]:.3f}" for L in sorted(pool.index, reverse=True))
    footer(fig, f"Jobs 50527633 (L=12) and 50527634 (L=8), commit 4b9c0e5, seed 0; L=6/4/3 are "
                f"finding 33's cells (50099945), re-evaluated inside both jobs and identical "
                f"instance for instance. Shared pool: {got}.")
    o.savefig(fig)


def page_method(o, D):
    rho = hard_rho(D["bins"])
    textpage(
        o,
        "What is being measured, and what changed",
        "One number per cell is not enough: L* is a statement about depth bins, not about an average.",
        [("THE TASK IS THE LEAK-FREE ONE.", GREEN,
          "Finding 16 showed the generator listed each node's neighbours sorted by their true colour, which "
          "let a model read the answer off token ORDER -- every leaked-order solve rate in this project is a "
          "decoding score. --shuffle-adj removes it. These five cells are all leak-free, so the depth they "
          "need is the depth DEDUCTION needs."),
         ("L* IS DEFINED ON BINS AT FIXED DENSITY.", GREY,
          f"Instances are binned by T, the number of DD rounds they actually need, at a requested density of "
          f"{rho:.2f}. A cell clears the gate only if EVERY bin is at or above 0.90. Density is pinned inside "
          "the bins -- the realized spread is printed on the bin page -- so \"deeper\" cannot quietly mean "
          "\"sparser\", which is the confound that invalidated the first version of this measurement."),
         ("WHAT CHANGED SINCE 10-29.", GREY,
          "Finding 33 ran L = 6, 4, 3 and none of them cleared a single bin, so L* was UNDEFINED above 6 -- "
          "the ladder was open at the top and could not be read. These two jobs add L=8 and L=12 going "
          "upward, at the same budget and on the same pool, and each job re-scores the 10-29 cells so that "
          "every contrast is paired on identical instances."),
         ("THE FREE REPRODUCIBILITY CHECK.", GREEN,
          "Because the reference cells are evaluated two or three times across jobs, the loader asserts the "
          "repeats agree instance for instance before pooling anything. They do. A disagreement would have "
          "meant the eval path was not deterministic and nothing below would be safe to pair.")],
        "Pre-registrations were fixed in the sbatch header before either job landed, and the analysis script "
        "was written while they trained, so no threshold here was chosen after seeing a number.")


def page_ladder(o, D):
    rho = hard_rho(D["bins"])
    sub = D["bins"][D["bins"].rho_requested == rho]
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.94, f"The gate, bin by bin, at density {rho:.2f}",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.885,
             "Each line is one cell. Dashed = the 0.90 gate. A cell owns L* only if its whole line is above it.",
             ha="center", size=MED, color="0.3")
    ax = fig.add_axes([0.09, 0.17, 0.56, 0.62])
    for L in sorted(sub.L.unique(), reverse=True):
        r = sub[sub.L == L].sort_values("T")
        ax.plot(r["T"], r.solve_rate, "o-", lw=3, ms=10, color=COL.get(L, GREY),
                label=f"L={L}", rasterized=True)
    ax.axhline(GATE, color="k", ls="--", lw=2.5)
    ax.set_xticks(sorted(sub["T"].unique()))
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel("T, the number of DD rounds the instance needs")
    ax.set_ylabel("solve rate")
    ax.legend(loc="lower left", ncol=2, fontsize=SMALL)

    cl, nb = clears(D["bins"], rho)
    lines = [f"bins cleared, of {nb}", ""]
    for L in sorted(cl, reverse=True):
        mark = "<-- on the gate" if 0 < cl[L] < nb else ("clears" if cl[L] == nb else "")
        lines.append(f"   L={L:<3} {cl[L]}/{nb}   {mark}")
    spread = sub.realized_rho.max() - sub.realized_rho.min()
    lines += ["", f"realized density spread", f"   {spread:.6f} across all bins",
              "   (the control: deeper is not sparser)"]
    fig.text(0.70, 0.74, "\n".join(lines), size=MED - 3, va="top", family="monospace",
             linespacing=1.6)
    footer(fig, "Bins with fewer than ~60 instances are dropped by the eval, so the T range shown is the "
                "range the pool actually supports at this density.")
    o.savefig(fig)


def page_trend(o, D):
    rho = hard_rho(D["bins"])
    sub = D["bins"][D["bins"].rho_requested == rho]
    rows = []
    for L in sorted(sub.L.unique(), reverse=True):
        r = sub[sub.L == L].sort_values("T")
        z = ca_trend((r.solve_rate * r.n).round(), r.n, r["T"])
        hi, lo = r.solve_rate.max(), r.solve_rate.min()
        kind = ("floored at zero" if hi < 0.05 else
                "ON THE GATE" if hi >= GATE > lo else
                "above throughout" if lo >= GATE else "below throughout")
        rows.append((L, hi, lo, z, kind))
    on = [r for r in rows if r[4] == "ON THE GATE"]
    above = [r for r in rows if r[4] == "above throughout"]
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.94, "Does the depth the network needs track the depth the problem needs?",
             ha="center", size=BIG, weight="bold")
    if on:
        head = (f"YES, and it shows up twice. L={on[0][0]} clears the three shallowest bins and misses the "
                f"three deepest, so its gate crosses between T=5 and T=6.")
        if above:
            a = above[0]
            head += (f" And L={a[0]}, which clears every bin, still falls {a[1]:.3f} -> {a[2]:.3f} "
                     f"with z = {a[3]:+.2f}: it is above the line, not flat.")
    else:
        head = "Not answerable here -- no cell sits on the gate."
    fig.text(0.5, 0.875, textwrap.fill(head, width=104), ha="center", size=MED,
             color="#2b6cb0", weight="bold", va="top", linespacing=1.45)

    ax = fig.add_axes([0.09, 0.19, 0.52, 0.58])
    for L, _, _, _, kind in rows:
        r = sub[sub.L == L].sort_values("T")
        ax.plot(r["T"], r.solve_rate, "o-", lw=3.5 if kind == "ON THE GATE" else 1.8,
                ms=11 if kind == "ON THE GATE" else 6, color=COL.get(L, GREY),
                alpha=1.0 if kind == "ON THE GATE" else 0.45,
                label=f"L={L} ({kind.lower()})", rasterized=True)
    ax.axhline(GATE, color="k", ls="--", lw=2.5)
    ax.set_xticks(sorted(sub["T"].unique()))
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel("T, DD rounds needed")
    ax.set_ylabel("solve rate")
    ax.legend(loc="lower left", fontsize=SMALL - 1)

    txt = ["cell      top    bottom   C-A z    where it sits", ""]
    for L, hi, lo, z, kind in rows:
        txt.append(f"L={L:<3}  {hi:5.3f}   {lo:5.3f}   {z:+6.2f}   {kind}")
    txt += ["", "Every cell that is not pinned at zero falls",
            "with T. A cell at the floor CANNOT trend,",
            "whatever the model does -- which is why",
            "findings 11 and 33 could not answer this",
            "and finding 22 could only answer it",
            "marginally (z = -1.92, p = 0.055)."]
    fig.text(0.655, 0.76, "\n".join(txt), size=MED - 4, va="top", family="monospace",
             linespacing=1.55)
    footer(fig, "Cochran-Armitage is a trend test across ordered bins; negative means solve rate falls as T "
                "rises. It is reported for every cell, so the one pinned at the floor can be seen not to move. "
                "L=8's T=7 bin (0.775, n=129) is below its T=8 bin (0.857, n=70): the decline is a trend, "
                "not a monotone staircase.")
    o.savefig(fig)


def page_prev(o, D):
    pool = D["pool"].set_index("L")
    best = int(pool.rate.idxmax())
    x, y = D["solved"][best], D["solved"][6]
    won, lost = int((x & ~y).sum()), int((~x & y).sum())
    p = mcnemar_exact(won, lost)
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.94, "Against the previous best on the same eval",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.885,
             "Same 1500 instances, same protocol, so this is a paired test and not a comparison of two averages.",
             ha="center", size=MED, color="0.3")

    ax = fig.add_axes([0.08, 0.20, 0.40, 0.56])
    names = [f"L=6\n(finding 33, 10-29)", f"L={best}\n(this pair of jobs)"]
    vals = [pool.rate[6], pool.rate[best]]
    los = [pool.lo[6], pool.lo[best]]
    his = [pool.hi[6], pool.hi[best]]
    ax.bar(names, vals, color=[GREY, "#2b6cb0"],
           yerr=[np.array(vals) - los, np.array(his) - vals], capsize=6, rasterized=True)
    ax.axhline(GATE, color="k", ls="--", lw=2)
    ax.text(-0.45, GATE + 0.02, "0.90 gate", size=SMALL)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.04, f"{v:.4f}", ha="center", size=MED - 2, weight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("shared-pool solve rate (n=1500)")
    ax.set_title("previous best vs new best", size=MED)

    lines = [
        f"paired delta      {pool.rate[best] - pool.rate[6]:+.4f}",
        f"discordant        {won} - {lost}",
        f"McNemar exact p   {p:.3g}",
        "",
        "L* at N=48, by task and MLP width",
        "",
        "  leaked order, ff=1024    L* = 4",
        "  leaked order, ff=256     L* = 6",
        "  LEAK-FREE,    ff=1024    L* in (8, 12]",
        "",
        "Removing the adjacency-order leak costs",
        "at least FOUR rungs of depth at N=48,",
        "and that is measured at 3x the data.",
        "",
        "The leaked rows sit on a DIFFERENT pool",
        "(shuffle_adj is in INSTANCE_KEYS), so they",
        "are quoted for context and not tested.",
    ]
    fig.text(0.55, 0.78, "\n".join(lines), size=MED - 3, va="top", family="monospace",
             linespacing=1.6)
    footer(fig, "Finding 33's L=6 is the previous best because it was the top of the old ladder; L=4 and L=3 "
                "score far lower and are shown on the bin page.")
    o.savefig(fig)


def page_limits(o, D):
    pool = D["pool"].set_index("L")
    rho = hard_rho(D["bins"])
    cl, nb = clears(D["bins"], rho)
    l8 = f"{pool.rate[8]:.4f}" if 8 in pool.index else "n/a"
    textpage(
        o,
        "Pre-registrations scored, and where this is not robust",
        "Fixed in the sbatch header before either job landed.",
        [("(a) L=12 CLEARS EVERY BIN, SO L* IS DEFINED.  CONFIRMED.", GREEN,
          f"{cl.get(12, 0)} of {nb} bins at density {rho:.2f}. This is the whole point of the pair of jobs: "
          "finding 33 left the ladder open at the top, and it is now closed."),
         ("(b) L=8 LANDS IN [0.70, 0.84] ON THE POOL.  MISSED, ON THE HIGH SIDE.", RED,
          f"It scored {l8}. The interval came from interpolating finding 33's L=6 toward finding 23's N=24 "
          "ceiling, and that interpolation was too pessimistic. My data-and-depth predictions in this "
          "project remain mostly underestimates, which is worth saying out loud."),
         ("(c) L*(T) RISES AT WHICHEVER CELL SITS ON THE GATE.  CONFIRMED AT L=8.", GREEN,
          "The three bins it misses are the three deepest, so the gate crosses between T=5 and T=6. This is "
          "the first time the leak-free task has produced a cell on the gate at all, and the cleanest "
          "version of the result finding 22 could only get marginally."),
         ("WHERE IT IS NOT ROBUST.", AMBER,
          "ONE SEED per cell -- finding 37 has just shown a second seed moving a layer-indexed readout by "
          "one rung elsewhere in this project. THE BRACKET IS THREE RUNGS WIDE: L=9, 10, 11 are unmeasured, "
          "so one L=10 cell halves it and is the cheapest next job. ONLY ONE CELL STRADDLES THE GATE, so the "
          "crossing point T=5/T=6 rests on L=8's six bins alone, and its T=7 bin (0.775) sits below its T=8 "
          "bin (0.857). FAST PHASE ONLY (finding 34). And everything here is at ff=1024, so the leak-free "
          "width factor at N=48 is unmeasured -- that is the hyper-parameter scan this ladder was built for.")],
        "Notebook: notebooks/sudoku_n48deep_2026-10-31_results.ipynb. Analysis: $SCRATCH/n48_deep_analysis.py, "
        "written while the jobs trained and validated against the 10-29 ladder it reproduces exactly.",
        y0=0.79, width=104)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_style()
    D = load()
    ex = example_instance(n_nodes=48, rho=0.60, pick="deep")
    note = textwrap.fill(
        "T is the x-axis of every plot in this deck: it is the number of DD ROUNDS this instance needs, "
        "counted on the trace above -- not the number of tokens and not the number of nodes. The depth bins "
        "group held-out instances by exactly this number, at fixed density, and L* asks how many layers the "
        "network needs before it clears 0.90 in all of them.", width=122)
    with PdfPages(os.path.join(OUT, "slides.pdf")) as pdf:
        o = PdfAndPng(pdf, OUT, total=8)
        page_verdict(o, D)
        page_method(o, D)
        page_tokenization(o, ex)
        page_trace(o, ex, note=note)
        page_ladder(o, D)
        page_trend(o, D)
        page_prev(o, D)
        page_limits(o, D)
    mb = os.path.getsize(os.path.join(OUT, "slides.pdf")) / 1e6
    print(f"wrote {OUT}/slides.pdf  ({mb:.2f} MB)")


if __name__ == "__main__":
    main()
