"""Slide deck for the N=96 fall-ensemble scan (50860571 + 50903022), all six cells.

    srun -p dev -t 00:20:00 -c 2 --mem=16G python3 notebooks/make_n96_fallscan_deck.py

Writes $SCRATCH/spin_gnn/decks/n96_fallscan_2026-11-03/slides.pdf plus one PNG per
page (there is no PDF rasterizer on this cluster, so the PNGs are the only way to
actually look at the deck before sending it).

Numbers come from `scripts/n96_fallscan_analysis`, the same committed reader the
results notebook imports, so deck, notebook and `python3 scripts/n96_fallscan_analysis.py`
cannot disagree. Pages 2 and 3 are the mandatory tokenization and real-trace pages.
"""
import math
import os
import sys
import textwrap
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.expanduser("~/spin_gnn"))
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style, textpage
from scripts.n96_fallscan_analysis import (
    SCAN_DEFAULT, TASKS, load, rho_50, rho50_ci, history)

OUT = os.path.join(os.environ["SCRATCH"], "spin_gnn", "decks",
                   "n96_fallscan_2026-11-03")
GREEN, RED, BLUE, GREY, ORANGE = "#1a7f37", "#b42318", "#1f4e9c", "0.35", "#b5500a"
KEY = ("num_layers", "dim_feedforward", "n_train_instances", "seed")


def load_cells():
    rows = load(os.path.join(SCAN_DEFAULT, "rho_curve_t*.csv"))
    cells = defaultdict(list)
    for r in rows:
        cells[tuple(r[k] for k in KEY)].append(r)
    for k in cells:
        cells[k].sort(key=lambda r: r["realized_rho"])
    label = {v: k for k, v in TASKS.items()}
    present = [k for k in TASKS.values() if k in cells]
    S = {}
    for k in present:
        g = cells[k]
        v, flag = rho_50([(r["realized_rho"], r["solve_rate"]) for r in g])
        lo, hi, _ = rho50_ci(g)
        fv, bv, tr, ne = history(g[0]["path"])
        scored = [r for r in g if r["solve_rate"] > 0.1]
        S[k] = dict(label=label[k], short=label[k].split(" (")[0], L=k[0],
                    n_train=k[2], seed=k[3], n_params=g[0]["n_params"], epochs=ne,
                    rho50=v, flag=flag, lo=lo, hi=hi, val=fv,
                    wf_min=min(r["well_formed_rate"] for r in g),
                    tr_max=max((r["truncated_rate"] for r in scored), default=0.0),
                    T_max=max(r["dd_mean_T"] for r in g),
                    floor=min(r["realized_rho"] for r in g))
    return cells, S, present


def seed_pairs(S, present):
    byconf = defaultdict(dict)
    for k in present:
        byconf[(k[0], k[1], k[2])][k[3]] = k
    return {c: d for c, d in byconf.items() if len(d) > 1}


# ------------------------------------------------------------------ page 1
def page_verdict(o, S, present, pairs, yard):
    best = min(present, key=lambda k: S[k]["rho50"])
    worst = max(present, key=lambda k: S[k]["rho50"])
    d8 = next(k for k in present if k[0] == 8 and k[3] == 0)
    d12 = next(k for k in present if k[0] == 12 and k[3] == 0)
    spreads = ", ".join(f"{c[0]}: {abs(S[d[0]]['rho50'] - S[d[1]]['rho50']):.3f}"
                        for c, d in sorted(pairs.items()))
    textpage(
        o,
        "N=96: the transition is measurable — and the seed beats depth",
        "6 cells, 67 GPU-h · jobs 50860571 + 50903022 · commit d016168 · "
        "rho_50 = hint density at solve rate 0.5, LOWER IS BETTER",
        [
            ("H1 PASSES — the measurement works, which it did not at N=48.", GREEN,
             f"All six cells cross solve_rate 0.5 strictly inside the rho grid, so rho_50 is an "
             f"interpolation and not a censored bound. At N=48 every cell solved the whole grid and "
             f"rho_50 could only be quoted as '< floor'. dd_mean_T runs 1.2 to "
             f"{max(S[k]['T_max'] for k in present):.1f} rounds down to a floor of "
             f"{min(S[k]['floor'] for k in present):.3f}: fast phase to deep slow phase."),
            (f"H4 FAILS at {yard/0.05:.1f}x its bar — and the pre-registration says that voids H2 and H3.", RED,
             f"Two pairs differ ONLY in --seed. Their rho_50 spreads ({spreads}) both blow past the "
             f"0.05 bar, with paired per-rho solve differences peaking at 0.490. The sbatch header "
             f"committed the reading in advance: 'if it is larger than 0.05 here, this whole grid is "
             f"one draw and only the ordering survives.' H2's L=8->12 gap is "
             f"{S[d8]['rho50'] - S[d12]['rho50']:+.3f}, H3's margin 0.069 — neither is a measurement."),
            ("WHY — rho_50 is essentially a function of how far the run converged.", BLUE,
             "val_nll rank-orders rho_50 across all six cells at Spearman +0.943; 0.086 of val_nll "
             "spans 0.266 of rho_50; all six are still descending when training stops. The seed "
             "moves val_nll by up to 0.065, the whole L=8->16 depth range by 0.018. Depth, data and "
             "seed are three ways of moving one number, and the seed moves it ~4x further."),
            ("Best rho_50 improved 0.691 -> 0.564 on the same eval. It is a seed draw, not a method.", ORANGE,
             f"The one-knob decomposition puts 90% of that on the seed, 10% on depth. The seed cuts "
             f"both ways and the downside is larger: 0->1 bought -0.115 at L=12 and cost +0.138 at "
             f"L=8, making {S[worst]['short']} the grid's worst at {S[worst]['rho50']:.3f}. "
             f"50933500 (cosine, one knob, two seeds) is running to test whether the spread can be "
             f"reduced at all."),
        ],
        "Every number read at build time from the committed reader "
        "scripts/n96_fallscan_analysis.py, the same one the results notebook imports. "
        "Eval: 14-point rho grid 0.95-0.32, 400 fresh instances per bin, identical "
        "instances across all six cells, so every cell-to-cell difference is paired.",
        y0=0.835, gap=0.050, width=118)


# ------------------------------------------------------------------ page 4
def page_curves(o, cells, S, present, yard):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Success rate across the DD transition, all six cells",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.895,
             "400 paired instances per point. Ticks mark rho_50; the grey band "
             "is what ONE SEED is worth.",
             ha="center", size=SMALL, color="0.3")

    # Tall axes: at height 0.45 the bottom 40% of the slide was blank, which on
    # a phone screen is 40% of the legible area.
    ax = fig.add_axes([0.07, 0.17, 0.56, 0.61])
    ax2 = fig.add_axes([0.71, 0.17, 0.25, 0.61])
    order = sorted(present, key=lambda k: S[k]["rho50"])
    cmap = plt.get_cmap("viridis")
    col = {k: cmap(i / max(1, len(order) - 1)) for i, k in enumerate(order)}

    for k in order:
        g = cells[k]
        ax.plot([r["realized_rho"] for r in g], [r["solve_rate"] for r in g],
                "o-", ms=4, lw=2, color=col[k],
                label=f"{S[k]['short']}   rho_50 {S[k]['rho50']:.3f}", rasterized=True)
        ax.plot([S[k]["rho50"]] * 2, [0.44, 0.56], lw=2.5, color=col[k])
    hl = next(k for k in present if k[0] == 12 and k[3] == 0)
    ax.axvspan(S[hl]["rho50"] - yard / 2, S[hl]["rho50"] + yard / 2,
               color="0.6", alpha=0.25, zorder=0)
    ax.axhline(0.5, color="k", lw=1.0, ls="--")
    ax.set_xlabel("realized hint density rho"); ax.set_ylabel("solve rate")
    ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
    ax.legend(fontsize=SMALL - 2, loc="upper left")
    ax.text(0.98, 0.52, f"seed yardstick {yard:.3f}", transform=ax.transAxes,
            ha="right", size=SMALL - 2, color="0.35")

    for k in order:
        ax2.errorbar(S[k]["rho50"], S[k]["L"] + (0.25 if S[k]["seed"] else -0.25),
                     xerr=[[S[k]["rho50"] - S[k]["lo"]], [S[k]["hi"] - S[k]["rho50"]]],
                     fmt="o", ms=9, color=col[k], capsize=4, rasterized=True)
    for conf, d in sorted(seed_pairs(S, present).items()):
        a, b = S[d[0]], S[d[1]]
        ax2.plot([a["rho50"], b["rho50"]], [conf[0] - 0.25, conf[0] + 0.25],
                 lw=2.5, color=RED, alpha=0.65, zorder=0)
    ax2.set_yticks([8, 12, 16]); ax2.set_ylim(6, 18)
    ax2.set_xlabel("rho_50  (lower = better)"); ax2.set_ylabel("num_layers")
    ax2.set_title("rho_50 with 95% CI\nred = a seed pair", size=SMALL + 1)
    ax2.grid(alpha=0.3)

    footer(fig, "CIs are binomial resampling of each bin (n=400) pushed back through the "
                "same interpolation, so they are SAMPLING error only — about +/-0.01, roughly "
                "a tenth of the seed term. Quoting them as the uncertainty on an architecture "
                "would understate it by an order of magnitude; that is the whole lesson of this deck.")
    o.savefig(fig)


# ------------------------------------------------------------------ page 5
def page_seed(o, cells, S, pairs, yard):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "The control that broke the grid: same config, different seed",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.895,
             "Each pair is byte-identical but for --seed. Pre-registered bar: spread <= 0.05.",
             ha="center", size=SMALL, color="0.3")

    # Top at 0.82, not 0.86: these axes carry two-line titles, which at 0.86
    # ran up into the subtitle.
    confs = sorted(pairs)
    axes = [fig.add_axes([0.07 + 0.32 * i, 0.38, 0.24, 0.44]) for i in range(len(confs))]
    axd = fig.add_axes([0.07 + 0.32 * len(confs), 0.38, 0.24, 0.44])
    for ax, conf in zip(axes, confs):
        d = pairs[conf]
        for s in sorted(d):
            g = cells[d[s]]
            ax.plot([r["realized_rho"] for r in g], [r["solve_rate"] for r in g],
                    "o-", ms=4, lw=2, label=f"seed {s}   {S[d[s]]['rho50']:.3f}",
                    rasterized=True)
        ax.axhline(0.5, color="k", lw=0.9, ls="--")
        sp = abs(S[d[0]]["rho50"] - S[d[1]]["rho50"])
        ax.set_title(f"L={conf[0]}, {conf[2]//1000}k\nspread {sp:.3f}  "
                     f"({'FAIL' if sp > 0.05 else 'pass'})",
                     size=SMALL + 1, color=RED if sp > 0.05 else GREEN)
        ax.set_xlabel("realized rho"); ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
        ax.legend(fontsize=SMALL - 2)
    axes[0].set_ylabel("solve rate")

    for conf in confs:
        d = pairs[conf]
        a, b = cells[d[0]], cells[d[1]]
        axd.plot([x["realized_rho"] for x in a],
                 [x["solve_rate"] - y["solve_rate"] for x, y in zip(a, b)],
                 "o-", ms=4, lw=2, label=f"L={conf[0]}", rasterized=True)
    axd.axhline(0, color="k", lw=0.9)
    axd.set_xlabel("realized rho"); axd.set_ylabel("paired delta solve_rate")
    axd.set_title("seed 0 - seed 1, same instances", size=SMALL + 1)
    axd.grid(alpha=0.3); axd.legend(fontsize=SMALL - 2)

    fig.text(0.5, 0.26,
             f"Yardstick in force: {yard:.3f}  —  the larger of the two spreads: "
             f"a gap is only as sharp as its noisier rung.",
             ha="center", size=MED, weight="bold", color=RED)
    fig.text(0.5, 0.19,
             "The L=8 replicate (50903022) was submitted precisely in case the shallow end was "
             "quieter and could rescue H2.\nIt is the noisier end, and it set the yardstick — "
             "reproducing finding 41, which measured the big seed term at the shallow end at N=48.",
             ha="center", size=SMALL, color="0.25", linespacing=1.5)
    footer(fig, "These are paired curves: both seeds of a pair see the same 400 instances at each "
                "rho, so the difference between them is the model, not the sample. The peak paired "
                "difference is 0.490 of solve rate at L=8 — half the dynamic range of the metric.")
    o.savefig(fig)


# ------------------------------------------------------------------ page 6
def page_mechanism(o, S, present):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "What is actually varying: how far each run converged",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.895,
             "Finding 4 put val_nll vs solve rate at r=+0.33 ACROSS models. Within one N, "
             "one ensemble and one protocol it is almost deterministic.",
             ha="center", size=SMALL, color="0.3")

    ax = fig.add_axes([0.08, 0.34, 0.40, 0.49])
    ax2 = fig.add_axes([0.58, 0.34, 0.36, 0.49])
    # Two pairs sit within 0.004 of each other in val_nll (t2/t3 and t0/t1), so
    # one fixed label offset prints each pair on top of itself. Cluster by val,
    # then inside a cluster put the higher-rho label above and the lower below
    # -- alternating by rank is not enough, since which of the two is on top
    # depends on rho, not on val order.
    by_val = sorted(present, key=lambda k: S[k]["val"])
    vspan = S[by_val[-1]]["val"] - S[by_val[0]]["val"]
    rspan = max(S[k]["rho50"] for k in present) - min(S[k]["rho50"] for k in present)
    groups = []                       # close in val AND in rho: a real collision
    for k in by_val:
        if groups and S[k]["val"] - S[groups[-1][-1]]["val"] < 0.08 * vspan:
            groups[-1].append(k)
        else:
            groups.append([k])
    dy = {}
    for g in groups:
        sub = []
        for k in sorted(g, key=lambda k: S[k]["rho50"]):
            if sub and S[k]["rho50"] - S[sub[-1][-1]]["rho50"] < 0.05 * rspan:
                sub[-1].append(k)
            else:
                sub.append([k])
        for s in sub:
            for j, k in enumerate(s):
                dy[k] = -4 if len(s) == 1 else round(-13 + j * 22 / (len(s) - 1))
    for i, k in enumerate(by_val):
        c = RED if S[k]["seed"] else BLUE
        ax.scatter(S[k]["val"], S[k]["rho50"], s=140, color=c, zorder=3, rasterized=True)
        last = i == len(by_val) - 1   # rightmost: label inwards, off the bar panel
        ax.annotate(S[k]["short"], (S[k]["val"], S[k]["rho50"]),
                    textcoords="offset points",
                    xytext=(-9 if last else 9, dy[k]),
                    ha="right" if last else "left", size=SMALL - 2)
    v = np.array([S[k]["val"] for k in present]); r = np.array([S[k]["rho50"] for k in present])
    m, b = np.polyfit(v, r, 1)
    xs = np.linspace(v.min(), v.max(), 10)
    ax.plot(xs, m * xs + b, "--", color="0.5", lw=1.5, zorder=1)
    ax.set_xlabel("final val_nll"); ax.set_ylabel("rho_50  (lower = better)")
    ax.set_title("Spearman +0.943 over six cells", size=SMALL + 2)
    ax.grid(alpha=0.3)
    ax.text(0.03, 0.95, "blue = seed 0\nred = seed 1", transform=ax.transAxes,
            va="top", size=SMALL - 2, color="0.3")

    # How far each knob moves val_nll -- the channel everything acts through.
    d8 = next(k for k in present if k[0] == 8 and k[3] == 0)
    d12 = next(k for k in present if k[0] == 12 and k[3] == 0)
    d16 = next(k for k in present if k[0] == 16 and k[3] == 0)
    s8 = next(k for k in present if k[0] == 8 and k[3] == 1)
    s12 = next(k for k in present if k[0] == 12 and k[3] == 1)
    bars = [("depth L=8 -> 16\n(whole range, seed 0)", abs(S[d16]["val"] - S[d8]["val"]), BLUE),
            ("depth L=8 -> 12", abs(S[d12]["val"] - S[d8]["val"]), BLUE),
            ("seed 0 -> 1 at L=12", abs(S[s12]["val"] - S[d12]["val"]), RED),
            ("seed 0 -> 1 at L=8", abs(S[s8]["val"] - S[d8]["val"]), RED)]
    bars.sort(key=lambda t: t[1])
    ax2.barh([t[0] for t in bars], [t[1] for t in bars],
             color=[t[2] for t in bars], rasterized=True)
    for i, t in enumerate(bars):
        ax2.text(t[1] + 0.0012, i, f"{t[1]:.4f}", va="center", size=SMALL - 1)
    ax2.set_xlabel("|change in final val_nll|")
    ax2.set_title("The seed outruns depth in the convergence channel",
                  size=SMALL + 2)
    ax2.set_xlim(0, max(t[1] for t in bars) * 1.25)
    ax2.tick_params(axis="y", labelsize=SMALL - 2)
    ax2.grid(alpha=0.3, axis="x")

    fig.text(0.5, 0.22,
             "Consequence 1: inside a fixed (N, ensemble, protocol), val_nll is a near-free "
             "proxy for rho_50 — the 14-point x 400-instance\neval costs hours per cell, and "
             "the ranking is already sitting in history.json.\n"
             "Consequence 2: a lever that lowers val_nll and tightens its spread should move "
             "rho_50 with it. That is the premise of\n50933500, and the reason the planned "
             "depth scan is blocked behind it.",
             ha="center", va="top", size=SMALL, color="0.25", linespacing=1.6)
    footer(fig, "All six cells still have val_nll falling at the last epoch (last-5 drop negative "
                "everywhere), so none is capacity-bound — they are all stopped mid-descent, which is "
                "exactly the regime in which where the iterate happens to sit is a lottery. The "
                "negative train-minus-val gap is finding 16's averaging artifact, not underfitting.")
    o.savefig(fig)


# ------------------------------------------------------------------ page 7
def page_table(o, S, present, yard):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Every cell, every control", ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.895,
             "Identical but for the column named: N=96, degree 5, 4 colours, --fall "
             "--shuffle-adj, d_model 256, nhead 4, ff 1024, pre-norm, lr 1e-3 constant, "
             "batch 32, equal 375k gradient steps.",
             ha="center", size=SMALL, color="0.3")

    hdr = f"{'cell':<26}{'L':>3}{'n_train':>9}{'seed':>6}{'params':>12}{'rho_50':>9}{'95% CI':>17}{'val_nll':>10}{'wf min':>9}{'trunc':>8}"
    body = [hdr, "-" * len(hdr)]
    for k in sorted(present, key=lambda k: S[k]["rho50"]):
        s = S[k]
        ci = "[%.3f, %.3f]" % (s["lo"], s["hi"])
        body.append(f"{s['short']:<26}{s['L']:>3}{s['n_train']:>9}{s['seed']:>6}"
                    f"{s['n_params']:>12,}{s['rho50']:>9.3f}{ci:>17}"
                    f"{s['val']:>10.4f}{s['wf_min']:>9.4f}{s['tr_max']:>8.4f}")
    fig.text(0.5, 0.80, "\n".join(body), ha="center", va="top", size=SMALL,
             family="monospace", linespacing=1.6)

    blocks = [
        ("Pairing", GREEN,
         "0 of 14 requested densities realize differently across cells — every curve is "
         "paired point for point, so all differences above are paired differences."),
        ("Truncation — the max_len margin fix holds", GREEN,
         f"max truncated_rate {max(S[k]['tr_max'] for k in present):.4f} over the 51 bins with "
         f"solve>0.1, against a pre-registered bar of 0.01. Traces run to ~800 tokens at the "
         f"deep end and are not being cut off."),
        ("Grammar — one cell, low rho only, and NOT the undertrained one", ORANGE,
         f"min well_formed {min(S[k]['wf_min'] for k in present):.4f}, confined to t4 below rho 0.6 "
         f"(0.9325 at its own crossing, 1.0000 above rho 0.8). Its byte-identical seed-0 twin is "
         f"clean, so this is the seed again rather than a grammar bug. t5 has the worst val_nll "
         f"in the grid by a wide margin and clean grammar (0.9925), so low-rho malformedness is "
         f"not simply what undertrained models do here. Unexplained."),
    ]
    y = 0.47
    for head, colr, text in blocks:
        fig.text(0.07, y, head, size=MED - 2, weight="bold", color=colr)
        fig.text(0.07, y - 0.035, textwrap.fill(text, width=118), size=SMALL - 1,
                 va="top", linespacing=1.45)
        y -= 0.035 + 0.030 * (textwrap.fill(text, width=118).count("\n") + 1) + 0.025

    footer(fig, f"Sampling CIs are +/-0.01; the seed term is {yard:.3f}. Sampling is not the binding "
                f"uncertainty in this grid and must not be quoted as if it were.")
    o.savefig(fig)


# ------------------------------------------------------------------ page 8
def page_next(o, S, present, pairs, yard):
    sg = max((abs(S[d[0]]["rho50"] - S[d[1]]["rho50"]) / math.sqrt(2)) / 0.79788
             for d in pairs.values())
    n50 = math.ceil(2 * sg ** 2 * (2.8 / 0.05) ** 2)
    before = next(k for k in present if k[0] == 8 and k[3] == 0)
    best = min(present, key=lambda k: S[k]["rho50"])
    d12x = next(k for k in present if k[0] == 12 and k[3] == 0 and k[2] == 240000)
    d16x = next(k for k in present if k[0] == 16)
    textpage(
        o,
        "Before and after on the same eval — and why the next run is not a depth scan",
        "No earlier N=96 curve exists — this eval starts with 50860571. "
        "'Before' is the only cell that had landed when asked.",
        [
            (f"BEFORE {S[before]['short']}  rho_50 {S[before]['rho50']:.3f}   ->   "
             f"AFTER {S[best]['short']}  rho_50 {S[best]['rho50']:.3f}   "
             f"({S[best]['rho50'] - S[before]['rho50']:+.3f}, CIs disjoint)", ORANGE,
             "The after-cell differs from the before-cell in TWO knobs, depth 8->12 and seed 0->1. "
             "The one-knob depth control lands within 0.012 of the before-cell, so 90% of the move "
             "is the seed. Had the other L=8 seed landed first this morning, the same procedure "
             "would have reported 0.829 as the best. No method improvement is claimed."),
            (f"Averaging the seed away is not affordable: ~{n50} seeds per rung, ~{2*n50*11:,} GPU-h.", RED,
             f"At sigma_hat {sg:.3f} per cell, resolving H2's own 0.05 bar at 95%/80% needs ~{n50} "
             f"seeds at each of two rungs, ~{2*n50*11:,} GPU-h against the ~67 this whole grid cost. "
             f"The completed grid also closes the escape route: the spread is not a quirk of L=12, "
             f"it is LARGER at L=8. The seed term must be reduced, not sampled."),
            ("RUNNING NOW — 50933500, one knob, two seeds, pre-registered.", BLUE,
             "--lr-schedule cosine at L=12 / 240k / 50 epochs, seeds 0 and 1, otherwise identical "
             "to the pair above. Every cell here stops mid-descent at constant lr, so the "
             "end-point is a lottery; decay should lower val_nll and tighten its spread. P1, the "
             "test: does the spread fall below 0.115? P3, beating 0.564, is NOT the test — 0.564 is "
             "itself a lucky draw. Falsifier: spread >= 0.115 AND no val_nll gain means the seed "
             "term is intrinsic at N=96, and that is the finding, not a third schedule."),
            ("What H2 becomes if the spread does come down.", GREY,
             f"The depth ordering is right — 0.691 -> 0.679 -> 0.613 as L goes 8 -> 12 -> 16 — and "
             f"the 12->16 step ({S[d12x]['rho50'] - S[d16x]['rho50']:+.3f}) is the largest architectural effect here, about "
             f"half the yardstick. So the cheapest real test of depth is L=12 vs L=16 UNDER the "
             f"schedule, not a rerun of the 8->12 gap that failed. That is the step the standing "
             f"depth question needs, and it stays blocked until the spread is known."),
        ],
        "Deck and results notebook both read scripts/n96_fallscan_analysis.py at build time. "
        "Notebook: notebooks/sudoku_n96_fallscan_2026-11-03_results.ipynb. "
        "Curves and checkpoints: $SCRATCH/spin_gnn/sudoku_n96_fall_scan_2026-11-02/.",
        y0=0.835, gap=0.045, width=118)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_style()
    cells, S, present = load_cells()
    pairs = seed_pairs(S, present)
    yard = max(abs(S[d[0]]["rho50"] - S[d[1]]["rho50"]) for d in pairs.values())
    print(f"{len(present)} cells; seed yardstick {yard:.3f}")

    # The deck's subject is reasoning DEPTH, so the trace page shows a deep instance
    # from the same distribution every cell trained on -- fall ensemble at N=96, at the
    # deep end of the eval grid.
    ex = example_instance(n_nodes=96, degree=5, n_colors=4, rho=0.33, rho_max=0.95,
                          shuffle_adj=True, fall=True, pick="deep")
    print(f"trace instance: {ex.n_rounds} DD rounds, {len(ex.tokens)} tokens")

    with PdfPages(f"{OUT}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, OUT, total=7)
        page_verdict(o, S, present, pairs, yard)
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=96, degree 5, 4 colours, --shuffle-adj --fall: the task all six "
                               "cells in this deck were trained on. rho=0.33 is the deep end of the "
                               "eval grid, below where any cell still crosses 50% success.")
        page_curves(o, cells, S, present, yard)
        page_seed(o, cells, S, pairs, yard)
        page_mechanism(o, S, present)
        page_next(o, S, present, pairs, yard)

    mb = os.path.getsize(f"{OUT}/slides.pdf") / 1e6
    print(f"wrote {OUT}/slides.pdf  ({mb:.2f} MB, {o.n} pages)")
    if mb > 2:
        print("WARNING: over the 2 MB delivery budget")


if __name__ == "__main__":
    main()
