"""Slide deck for the N=48 leak-free SEED REPLICATES (50691430 L=10, 50691432
L=8) and the n=6000 depth re-evaluation (50728837).

    module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 \
        py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 \
        py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
    cd ~/spin_gnn && python3 -m notebooks.make_n48_seed1_deck

Eight landscape pages to $SCRATCH/spin_gnn/notebooks/sudoku_n48_seed1_2026-11-01/,
PDF plus one PNG per page (nothing on this cluster rasterizes a PDF, so the PNGs
are the only way to read the deck back before sending it).

The result is negative and it is about the measurement, not the model: at fixed
architecture the seed-to-seed spread in solve_rate is LARGER than the depth gaps
the whole L* ladder is built out of. The second job (n=6000) is the useful
counterpoint -- it shows the SAMPLING error bar is now small, which is what
isolates the seed as the thing that is actually uncontrolled.
"""
import glob
import json
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
OUT = f"{S}/notebooks/sudoku_n48_seed1_2026-11-01"
DIRS = [f"{S}/sudoku_n48_leakfree_lstar_2026-10-29",   # 50099945: L=6, 4, 3
        f"{S}/sudoku_n48_leakfree_lstar_2026-10-31",   # 50527633 L=12, 50527634 L=8
        f"{S}/sudoku_n48_leakfree_L10_2026-10-31",     # 50607549: L=10
        f"{S}/sudoku_n48_leakfree_seed1_2026-11-01"]   # 50691430/50691432: seed 1
BIGPOOL = f"{S}/sudoku_n48_leakfree_bigpool_2026-11-01"
DEPTH_GLOBS = ["depth_lstar_*.csv", "depth_L10_*.csv", "depth_seed1_*.csv"]
GATE, LADDER = 0.90, [12, 10, 8, 6, 4, 3]
GREEN, RED, AMBER, GREY = "#276749", "#9b2c2c", "#975a16", "0.35"
SEEDCOL = {0: "#2b6cb0", 1: "#c53030"}


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def cell_of(ckpt):
    """(num_layers, seed) from a checkpoint label. Seed 0 tags carry no _sNN."""
    name = str(ckpt)
    L = int(re.search(r"_L(\d+)_", name).group(1))
    m = re.search(r"_s(\d+)(?:\b|_|\s|$)", name)
    return L, (int(m.group(1)) if m else 0)


def load():
    """Everything the pages read, keyed on (num_layers, seed)."""
    D = {}

    # --- the metric of record: the shared n=1500 pool -----------------------
    frames = []
    for d in DIRS:
        for p in sorted(glob.glob(f"{d}/sharedpool_n1500_*.csv")):
            frames.append(pd.read_csv(p))
    pool = pd.concat(frames, ignore_index=True)
    pool["L"] = [cell_of(c)[0] for c in pool.checkpoint]
    pool["seed"] = [cell_of(c)[1] for c in pool.checkpoint]
    D["pool"] = pool.drop_duplicates(subset=["L", "seed"]).set_index(["L", "seed"])

    # per-instance matrices, so every contrast below is paired
    solved = {}
    for d in DIRS:
        for p in sorted(glob.glob(f"{d}/sharedpool_n1500_*_per_instance.npz")):
            z = np.load(p, allow_pickle=True)
            for i, nm in enumerate(str(x) for x in z["names"]):
                solved.setdefault(cell_of(nm), z["solved"][i].astype(bool))
    D["solved"] = solved

    # --- depth bins ---------------------------------------------------------
    frames = []
    for d in DIRS:
        for g in DEPTH_GLOBS:
            for p in sorted(glob.glob(f"{d}/{g}")):
                frames.append(pd.read_csv(p))
    dep = pd.concat(frames, ignore_index=True)
    dep["L"] = [cell_of(c)[0] for c in dep.checkpoint]
    dep["seed"] = [cell_of(c)[1] for c in dep.checkpoint]
    D["depth"] = dep.drop_duplicates(subset=["L", "seed", "rho_requested", "T"])

    big = pd.concat([pd.read_csv(p) for p in
                     sorted(glob.glob(f"{BIGPOOL}/depth_bigpool_*.csv"))],
                    ignore_index=True)
    big["L"] = [cell_of(c)[0] for c in big.checkpoint]
    big["seed"] = [cell_of(c)[1] for c in big.checkpoint]
    D["big"] = big.drop_duplicates(subset=["L", "seed", "rho_requested", "T"])

    # --- training histories -------------------------------------------------
    hist = {}
    for d in DIRS:
        for h in sorted(glob.glob(f"{d}/L*/*/history.json")):
            L, seed = cell_of(os.path.basename(os.path.dirname(h)))
            rows = json.load(open(h))
            hist[(L, seed)] = dict(val_nll=rows[-1]["nll"], acc=rows[-1]["accuracy"],
                                   epoch=rows[-1]["epoch"],
                                   curve=[(r["epoch"], r["nll"]) for r in rows])
    D["hist"] = hist
    D["cells"] = sorted(D["pool"].index.tolist())
    return D


def paired(D, a, b):
    """delta, a-wins, b-wins, McNemar p for two cells on the same 1500."""
    x, y = D["solved"][a], D["solved"][b]
    won, lost = int((x & ~y).sum()), int((~x & y).sum())
    return float(x.mean() - y.mean()), won, lost, mcnemar_exact(won, lost)


def spread(D, L):
    """|pool(L, seed 1) - pool(L, seed 0)| and the paired test behind it."""
    if (L, 1) not in D["solved"] or (L, 0) not in D["solved"]:
        return None
    d, w, l, p = paired(D, (L, 1), (L, 0))
    return dict(L=L, delta=d, spread=abs(d), won=w, lost=l, p=p)


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def page_verdict(o, D):
    sp = [s for s in (spread(D, L) for L in (8, 10)) if s]
    smax = max(s["spread"] for s in sp) if sp else float("nan")
    g128 = paired(D, (12, 0), (8, 0))[0]
    g810 = paired(D, (8, 0), (10, 0))[0]
    d1, w1, l1, p1 = paired(D, (10, 1), (8, 1))
    line = "  ".join(f"L={s['L']}: {s['spread']:.4f} (p={s['p']:.1e})" for s in sp)

    blocks = [
        ("The question", GREY,
         "Finding 40 put L=10 (0.8007) BELOW L=8 (0.8660) on the shared n=1500 pool, paired "
         "p=5.4e-09. That test compares two CHECKPOINTS on the same instances; it says nothing "
         "about two TRAINING PROCEDURES, and no rung of this ladder had ever been run twice. "
         "H1 said the inversion was optimization noise. These two jobs retrain L=8 and L=10 with "
         "--seed 1 and change nothing else."),
        ("The answer: H1, and much more of it than predicted", RED,
         f"At FIXED architecture the two seeds differ by {line}. The pre-registration called "
         f"anything at or above 0.065 the alarm case, and said it was the outcome I did not "
         f"expect. The measured spread is {smax:.4f} -- larger than the L=12 minus L=8 gap "
         f"({abs(g128):.4f}) that finding 38's headline rests on, and {smax/abs(g810):.1f}x the "
         f"inversion ({abs(g810):.4f}) it was invoked to explain."),
        ("So the ladder cannot be read rung by rung -- and finding 40 is withdrawn", RED,
         "Every L* in this file (findings 11, 14, 22, 33, 38, 40) is a single seed per rung. A "
         "ladder whose rungs carry an unmeasured spread bigger than its steps cannot locate a "
         "knee, and the MLP-depth scan that was to be centred on L=10 cannot start until each "
         "rung is an average over seeds rather than a draw. The inversion itself did not survive: "
         f"run within seed 1 the same comparison goes the OTHER way, L=10 - L=8 = {d1:+.4f} "
         f"({w1}-{l1}, p={p1:.1e}) against {-g810:+.4f} at seed 0. Finding 40 was a draw, not a "
         "property of depth 10."),
        ("The counterpoint: the SAMPLING error bar is now small", GREEN,
         "Job 50728837 re-evaluated the same six seed-0 checkpoints on a 4x larger depth pool "
         "(n_pool 6000, nested in the old 1500). L*(48, rho=0.60)=12's binding bin went n=70 -> "
         "289 and its simultaneous 95% lower bound 0.8422 -> 0.9434, so it clears the gate at 95% "
         "for the first time. Sampling was never the big error bar here. The seed is."),
    ]
    textpage(o, "A seed is worth more than two layers",
             "N=48 leak-free, ff=1024, 240000 x 100ep -- 50691430 / 50691432 / 50728837",
             blocks,
             "Metric of record: sudoku_agg_eval, n=1500 at seed_start 5e10, disjoint from every "
             "training set, paired McNemar exact. `seed` is not in INSTANCE_KEYS, so seed-1 cells "
             "score on exactly the same instances as seed-0 cells.", width=124)


def page_table(o, D):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, "The ladder, now with a second draw at two rungs",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.875, "shared pool n=1500; Wilson 95%; val_nll from each cell's own history.json",
             ha="center", size=MED, color="0.3")

    hdr = (f"{'cell':>9}  {'solve':>7}  {'95% CI':>18}  {'val_nll':>8}  {'tok acc':>8}  "
           f"{'trunc':>7}  {'out acc':>8}")
    rows = [hdr, "-" * len(hdr)]
    for L in LADDER:
        for seed in (0, 1):
            if (L, seed) not in D["pool"].index:
                continue
            r = D["pool"].loc[(L, seed)]
            lo, hi = wilson(int(round(r.solve_rate * r.n_instances)), int(r.n_instances))
            h = D["hist"].get((L, seed), {})
            rows.append(f"{f'L={L} s{seed}':>9}  {r.solve_rate:7.4f}  [{lo:.4f}, {hi:.4f}]  "
                        f"{h.get('val_nll', float('nan')):8.4f}  "
                        f"{h.get('acc', float('nan')):8.4f}  "
                        f"{r.truncated_rate:7.4f}  {r.outcome_accuracy:8.4f}")
    fig.text(0.06, 0.80, "\n".join(rows), size=MED - 3, va="top", family="monospace",
             linespacing=1.6)

    # the contrasts that matter, every one of them paired on the same 1500
    con = [("seed spread at L=8", (8, 1), (8, 0), "noise"),
           ("seed spread at L=10", (10, 1), (10, 0), "noise"),
           ("depth gap, finding 38", (12, 0), (8, 0), "signal"),
           ("finding-40 inversion", (10, 0), (8, 0), "signal"),
           ("depth, within seed 1", (10, 1), (8, 1), "signal")]
    txt = [f"{'contrast':<23}{'cells':>11}{'kind':>8}{'delta':>10}{'disc':>10}{'McNemar p':>12}",
           "-" * 74]
    for label, a, b, kind in con:
        cells = f"{a[0]}s{a[1]}-{b[0]}s{b[1]}"
        if a not in D["solved"] or b not in D["solved"]:
            txt.append(f"{label:<23}{cells:>11}{kind:>8}{'pending':>10}")
            continue
        d, w, l, p = paired(D, a, b)
        txt.append(f"{label:<23}{cells:>11}{kind:>8}{d:+10.4f}{f'{w}-{l}':>10}{p:12.2e}")
    fig.text(0.06, 0.46, "\n".join(txt), size=MED - 3, va="top", family="monospace",
             linespacing=1.6)

    footer(fig, "Every row is the same 1500 held-out instances, so each delta is a paired "
                "within-instance contrast, not two independent rates. 17 repeat evaluations of "
                "the seed-0 reference cells across four jobs agree instance for instance.")
    o.savefig(fig)


def page_spread(o, D):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Reseeding one rung moves it further than changing its depth",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.898, "shared n=1500 pool, Wilson 95%; the only difference between a red "
                         "and a blue point at the same L is --seed",
             ha="center", size=MED, color="0.3")
    ax1 = fig.add_axes([0.07, 0.16, 0.42, 0.63])
    ax2 = fig.add_axes([0.60, 0.16, 0.35, 0.63])

    for seed in (0, 1):
        xs = [L for L in LADDER if (L, seed) in D["pool"].index]
        if not xs:
            continue
        ys = [D["pool"].loc[(L, seed)].solve_rate for L in xs]
        errs = np.array([[y - wilson(int(round(y * 1500)), 1500)[0],
                          wilson(int(round(y * 1500)), 1500)[1] - y] for y in ys]).T
        ax1.errorbar(xs, ys, yerr=errs, marker="o", ms=9, lw=2, capsize=4,
                     color=SEEDCOL[seed], label=f"seed {seed}",
                     ls="-" if seed == 0 else "--", rasterized=True)
    ax1.axhline(GATE, color="0.6", ls=":", lw=1.5)
    ax1.text(3.1, GATE + 0.012, "0.90 gate", size=SMALL, color="0.45")
    ax1.set_xlabel("num_layers")
    ax1.set_ylabel("solve_rate, shared pool n=1500")
    ax1.set_title("The same ladder, two draws")
    ax1.set_xticks(LADDER)
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # the comparison that makes the point: spread vs the steps of the ladder
    bars, cols = [], []
    for L in (8, 10):
        s = spread(D, L)
        if s:
            bars.append((f"seed spread\nat L={L}", s["spread"]))
            cols.append(RED)
    for lab, a, b in [("depth step\nL=12 vs L=8\n(finding 38)", (12, 0), (8, 0)),
                      ("depth step\nL=10 vs L=8\n(finding 40)", (10, 0), (8, 0))]:
        bars.append((lab, abs(paired(D, a, b)[0])))
        cols.append("#2b6cb0")
    ax2.bar(range(len(bars)), [v for _, v in bars], color=cols, rasterized=True)
    ax2.set_xticks(range(len(bars)))
    ax2.set_xticklabels([k for k, _ in bars], size=SMALL - 2)
    ax2.set_ylabel("|delta solve_rate|")
    ax2.set_title("Noise (red) against signal (blue)")
    ax2.grid(alpha=0.3, axis="y")
    for i, (_, v) in enumerate(bars):
        ax2.text(i, v + 0.004, f"{v:.4f}", ha="center", size=SMALL - 1)

    footer(fig, "Red bars change NOTHING but the random seed; blue bars are the depth effects "
                "this ladder was built to measure. When the red bars are the taller ones, a "
                "single-seed rung is not a measurement of its architecture.")
    o.savefig(fig)


def page_depth(o, D):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, "The seed acts where L* is read: the deep bins",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.875, "rho=0.60, realized density pinned at 0.5836 in every bin "
                         "(spread < 0.0005), so 'deeper' is not 'sparser'",
             ha="center", size=MED, color="0.3")
    dep = D["depth"]
    axes = [fig.add_axes([0.07, 0.17, 0.40, 0.60]), fig.add_axes([0.56, 0.17, 0.40, 0.60])]
    for ax, L in zip(axes, (8, 10)):
        cleared = []
        for seed in (0, 1):
            r = dep[(dep.L == L) & (dep.seed == seed) & (dep.rho_requested == 0.60)]
            if r.empty:
                continue
            r = r.sort_values("T")
            ax.plot(r["T"], r.solve_rate, marker="o", ms=8, lw=2,
                    color=SEEDCOL[seed], ls="-" if seed == 0 else "--",
                    label=f"seed {seed}", rasterized=True)
            # L* is a COUNT of bins over the gate, so quote the count the gate
            # actually reads rather than leaving it to be eyeballed off the line.
            cleared.append((seed, int((r.solve_rate >= GATE).sum()), len(r)))
        if cleared:
            ax.text(0.97, 0.05, "bins over the 0.90 gate:  "
                    + ",  ".join(f"seed {s} {c}/{n}" for s, c, n in cleared),
                    transform=ax.transAxes, size=SMALL - 1, color="0.25", ha="right")
        ax.axhline(GATE, color="0.6", ls=":", lw=1.5)
        ax.set_title(f"L={L}")
        ax.set_xlabel("T (DD rounds required)")
        ax.set_ylabel("solve_rate in bin")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
    footer(fig, "Note the two panels point opposite ways -- seed 1 is the worse draw at L=8 and "
                "the better one at L=10 -- but in both the seeds agree at T=3 and separate as T "
                "grows. The seed's effect is concentrated on exactly the axis L* is read off, "
                "which is why it can move L* by more than a layer can, in either direction.")
    o.savefig(fig)


def page_bigpool(o, D):
    """The other job: sampling error, now measured, and the L* audit it feeds."""
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "The other error bar: sampling, and why it was not the problem",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.898, "50728837 -- same six seed-0 checkpoints, depth pool 1500 -> 6000, "
                         "no retraining", ha="center", size=MED, color="0.3")

    ax = fig.add_axes([0.06, 0.17, 0.38, 0.63])
    old = D["depth"][(D["depth"].seed == 0) & (D["depth"].rho_requested == 0.60)]
    new = D["big"][(D["big"].seed == 0) & (D["big"].rho_requested == 0.60)]
    for L in LADDER:
        a = old[old.L == L].sort_values("T")
        b = new[new.L == L].sort_values("T")
        if a.empty or b.empty:
            continue
        ax.plot(a["T"], a.solve_rate, marker="o", ms=5, lw=1, alpha=0.45, color="0.5",
                rasterized=True)
        ax.plot(b["T"], b.solve_rate, marker="s", ms=6, lw=2, color=SEEDCOL[0],
                rasterized=True)
        ax.annotate(f"L={L}", (b["T"].iloc[-1], b.solve_rate.iloc[-1]),
                    textcoords="offset points", xytext=(8, -3), size=SMALL - 1,
                    color=SEEDCOL[0])
    ax.set_xlim(right=float(new["T"].max()) + 0.9)
    ax.axhline(GATE, color="0.6", ls=":", lw=1.5)
    ax.set_xlabel("T (DD rounds required)")
    ax.set_ylabel("solve_rate in bin")
    ax.set_title("grey = n=1500, blue = n=6000 (nested)", size=MED)
    ax.grid(alpha=0.3)

    lines = [
        "L*(48, leak-free, rho=0.60) = 12, binding bin T=8",
        "",
        f"{'':<13}{'n':>6}{'rate':>9}{'95% lo':>9}   gate at 95%",
        f"{'n_pool 1500':<13}{70:>6}{0.9571:>9.4f}{0.8422:>9.4f}   no",
        f"{'n_pool 6000':<13}{289:>6}{0.9792:>9.4f}{0.9434:>9.4f}   YES",
        "",
        "Bonferroni-corrected Wilson over the six gated bins,",
        "worst bin -- not the bin with the lowest point rate.",
        "Nesting control, pre-registered: bin-size ratios",
        "3.82-4.45 vs an expected 4.00, no bin's solve_rate",
        "moved more than 2.54 SE. The pools nest exactly.",
        "",
        "SAME AUDIT, EVERY L* IN STATUS (lstar_fragility.py,",
        "CPU-only, reads CSVs already on disk):",
        "  of the five L* values quoted flat, ONE held before",
        "  tonight -- N=48 rho=0.75, L*=8 (lo 0.9455).",
        "  N=96 rho=0.75 ff=1024 L*=4 clears 155/172 = 0.9012;",
        "  at 154/172 it does not. ONE INSTANCE sets it, and",
        "  finding 26's 'ff matters more at larger N' is 2.0x",
        "  on points but 1.0x at 95%.",
        "  Finding 31's L=7 misses by 0.0029 = 0.08 bin SE.",
        "",
        "Why this hid: every other number here is a PAIRED",
        "contrast on shared instances (finding 40: p=5.4e-09",
        "on a 0.065 gap). L* is an ABSOLUTE threshold on ONE",
        "cell, so it gets none of that power -- bin size is",
        "all it has, and the deep bins were n=43-70.",
    ]
    fig.text(0.47, 0.84, "\n".join(lines), size=SMALL - 1, va="top", family="monospace",
             linespacing=1.42)
    footer(fig, "Design rule from inverting the bound: at a simultaneous 95% over six bins, a "
                "cell whose true rate is 0.94 needs n=373, 0.96 needs n=161, 0.98 needs n=99. "
                "Cells at the knee sit at 0.93-0.96, so n=70 could never have cleared.",
           width=130)
    o.savefig(fig)


def page_scorecard(o, D):
    sp8, sp10 = spread(D, 8), spread(D, 10)
    smax = max(s["spread"] for s in (sp8, sp10) if s)
    hb = D["hist"]

    def nll(L, s):
        return hb.get((L, s), {}).get("val_nll", float("nan"))

    if sp10:
        d1, w1, l1, p1 = paired(D, (10, 1), (8, 1))
        d0 = paired(D, (10, 0), (8, 0))[0]
        b_txt = (f"D_1 = pool(L=10,s1) - pool(L=8,s1) = {d1:+.4f} ({w1}-{l1}, p={p1:.1e}), "
                 f"against D_0 = {d0:+.4f} at seed 0. Predicted D_1 > 0: "
                 f"{'CONFIRMED' if d1 > 0 else 'FALSIFIED'} -- so finding 40's L=10 < L=8 "
                 "inversion does not replicate, it changes sign. One seed of it was an artifact.")
        dn = nll(10, 1) - nll(8, 1)
        c_txt = (f"Within seed 1, val_nll(L=10) - val_nll(L=8) = {dn:+.4f} and the pool delta is "
                 f"{d1:+.4f}; signs {'AGREE' if (dn < 0) == (d1 > 0) else 'DISAGREE'}. "
                 f"Implied sensitivity {abs(d1)/abs(dn) if dn else float('nan'):.1f} solve_rate "
                 f"per unit nll, against the pre-registered band [10, 60] and the seed-0 pairs' "
                 f"25-28.")
    else:
        b_txt = "L=10 seed 1 had not finished when this page was rendered."
        c_txt = b_txt

    blocks = [
        ("(a) seed spread >= 0.040  ::  CONFIRMED, in the alarm branch", RED,
         f"max spread {smax:.4f} over L=8 ({sp8['spread']:.4f}, {sp8['won']}-{sp8['lost']}, "
         f"p={sp8['p']:.1e})" + (f" and L=10 ({sp10['spread']:.4f}, {sp10['won']}-{sp10['lost']}, "
         f"p={sp10['p']:.1e})" if sp10 else "") +
         ". The header named >= 0.065 the case in which every single-seed L* in STATUS becomes "
         "uninterpretable without replicates, and said it was not the one I expected. "
         "(b) ORDERING: " + b_txt),
        ("(c) MECHANISM -- val_nll concordance, sensitivity in [10, 60]  ::  (d) CONTROLS CONFIRMED", GREY,
         c_txt + f" The seed-0 pair gives 25.2, inside the band; the seeds differ by "
         f"{abs(nll(8,1)-nll(8,0)):.4f} val_nll at L=8, predicting "
         f"~{25*abs(nll(8,1)-nll(8,0)):.2f} of solve rate against {sp8['spread']:.4f} measured. "
         f"Controls: realized density 0.5834-0.5839 vs 0.5836 +- 0.001; truncated_rate <= 0.0007 "
         f"and outcome_accuracy >= 0.9953 everywhere, so these are deduction scores and not "
         f"formatting; 17 repeat evals agree instance for instance."),
        ("What it costs the plan, and the cheapest fix", AMBER,
         "The MLP-depth scan has L* as its dependent variable, and L* now carries two error bars: "
         "sampling, measured and controlled tonight (+-0.03 at the binding bin), and seed, "
         "measured at 0.16 and not controlled. The scan cannot resolve a one-rung effect against "
         "the second. Fix: three seeds at L=8 and L=12 only, the cells that bracket L* at "
         "rho=0.60 -- SE ~0.09 per rung, 4 jobs at 7-11 h."),
        ("What is NOT in doubt, and the honest negative", GREEN,
         "The seed moves a rung, not the project: L=12 (0.9780) beats L=6 (0.6153) by 0.36 and "
         "L=3 (0.0093) by 0.97, and findings 27/28/29 (the DD verdict) are interventions on fixed "
         "checkpoints, untouched. Nothing improved tonight: the best N=48 leak-free cell is still "
         "L=12 seed 0 at 0.9780 [0.9693, 0.9843], unchanged since finding 38. The deliverable is "
         "an error bar, not a number."),
    ]
    textpage(o, "Pre-registration scorecard, and what to do about it",
             "scored against scripts/slurm/sudoku2_n48_seed1_replicate.sbatch, written before either cell was submitted",
             blocks,
             "Commit at submit a10a098, seed 1, N=48 leak-free ff=1024, 240000 x 100ep, "
             "--shuffle-adj; the only config difference from the seed-0 cells is --seed. "
             "Previous best on this eval: L=12 seed 0, 0.9780 (finding 38, job 50527633).",
             width=132)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_style()
    D = load()
    print("cells:", D["cells"])

    # pick="deep": this deck's whole subject is reasoning depth, and a T=2 trace
    # does not show why a network needs layers.
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.60, rho_max=0.95,
                          shuffle_adj=True, pick="deep")
    pages = 8
    with PdfPages(f"{OUT}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, OUT, total=pages)
        page_verdict(o, D)
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=48, degree 5, 4 colours, --shuffle-adj: the leak-free task. "
                               "Every cell in this deck trains on exactly this encoding.")
        page_table(o, D)
        page_spread(o, D)
        page_depth(o, D)
        page_bigpool(o, D)
        page_scorecard(o, D)
    print(f"wrote {OUT}/slides.pdf ({o.n} pages)")


if __name__ == "__main__":
    main()
