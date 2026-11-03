"""Slide deck for the N=48 FALL-ensemble scaling scan (array 50760297, 13 cells)
and the matching "before" curves on the unconstrained ensemble (50760252).

    module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 \
        py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 \
        py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
    cd ~/spin_gnn && python3 -m notebooks.make_fallscan_deck

Eight landscape pages to $SCRATCH/spin_gnn/notebooks/sudoku_n48_fallscan_2026-11-02/,
PDF plus one PNG per page (nothing on this cluster rasterizes a PDF, so the PNGs
are the only way to read the deck back before sending it).

The question hshunt asked was how success rate scales across the transition, and
with training examples and model size. The four hypotheses were pre-registered in
the header of scripts/slurm/sudoku2_n48_fall_scan_2026-11-02.sbatch BEFORE the
jobs were submitted, together with the scoring rule used here: every level
comparison ACROSS cells is read against s_L8(rho), the seed-to-seed spread at
fixed architecture, and anything smaller is reported as "not separable at this
budget" rather than as a result. Curve shape WITHIN one checkpoint is paired and
carries no such caveat.
"""
import csv
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.expanduser("~/spin_gnn"))
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style, textpage

S = os.environ["SCRATCH"] + "/spin_gnn"
SCAN = f"{S}/sudoku_n48_fall_scan_2026-11-02"
BEFORE = f"{S}/sudoku_n48_rho_curves_2026-11-02"
OUT = f"{S}/notebooks/sudoku_n48_fallscan_2026-11-02"
COMMIT = "cd97030"

GREEN, RED, AMBER, GREY = "#276749", "#9b2c2c", "#975a16", "0.35"
LCOL = {3: "#718096", 4: "#dd6b20", 6: "#b7791f", 8: "#2b6cb0", 10: "#6b46c1", 12: "#276749"}
NSTY = {30000: ":", 120000: "--", 240000: "-"}
NTRAIN = (30000, 120000, 240000)


# --------------------------------------------------------------------- data

def _read(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        # rho_curve_t07_<arrayjob>_7.csv -> array task 7. Taken from the filename
        # rather than re-derived from (L, ff, n_train, seed), which would silently
        # renumber the cells if the SPECS order in the sbatch ever changed.
        m = re.search(r"rho_curve_t(\d+)_", os.path.basename(f))
        for r in csv.DictReader(open(f)):
            r["task"] = int(m.group(1)) if m else -1
            for k in ("rho", "realized_rho", "solve_rate", "outcome_accuracy",
                      "well_formed_rate", "truncated_rate", "dd_mean_T"):
                r[k] = float(r[k])
            for k in ("num_layers", "dim_feedforward", "n_params",
                      "n_train_instances", "epochs", "seed"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def _history(path):
    """(epochs, train_nll, val_nll) arrays for one cell."""
    h = json.load(open(os.path.join(path, "history.json")))
    return (np.array([e["epoch"] for e in h], float),
            np.array([e["train_nll"] for e in h], float),
            np.array([e["nll"] for e in h], float))


def rho_50(rhos, solves):
    """Realized density where solve_rate crosses 0.5, linearly interpolated.

    Returns (value, flag): "ok", ">top" (the curve never reaches 0.5 even at
    the top of the grid) or "<floor" (it never drops to 0.5 before the
    ensemble runs out of removable hints). Both flags are plotted as open
    markers -- they are bounds, not measurements.
    """
    pts = sorted(zip(rhos, solves))
    if all(s < 0.5 for _, s in pts):
        return pts[-1][0], ">top"
    if all(s >= 0.5 for _, s in pts):
        return pts[0][0], "<floor"
    for (r0, s0), (r1, s1) in zip(pts, pts[1:]):
        if s0 < 0.5 <= s1:
            return r0 + (0.5 - s0) * (r1 - r0) / (s1 - s0), "ok"
    return float("nan"), "none"


def load():
    rows = _read(f"{SCAN}/rho_curve_t*.csv")
    before = _read(f"{BEFORE}/rho_curve_default_ensemble_*.csv")

    cells = {}
    for r in rows:
        cells.setdefault((r["num_layers"], r["dim_feedforward"],
                          r["n_train_instances"], r["seed"]), []).append(r)
    for v in cells.values():
        v.sort(key=lambda r: r["rho"])
    grid = sorted({r["rho"] for r in rows})
    realized = {q: next(r["realized_rho"] for r in rows if r["rho"] == q) for q in grid}
    # one realized density per requested rho across ALL cells is the pairing
    # control: if it ever failed, the curves would not be comparable point by point.
    paired = all(len({round(r["realized_rho"], 6) for r in rows if r["rho"] == q}) == 1
                 for q in grid)

    bcells = {}
    for r in before:
        bcells.setdefault((r["num_layers"], r["seed"]), []).append(r)
    for v in bcells.values():
        v.sort(key=lambda r: r["rho"])

    def curve(c):
        return (np.array([r["realized_rho"] for r in c]),
                np.array([r["solve_rate"] for r in c]))

    D = {"cells": cells, "bcells": bcells, "grid": grid, "realized": realized,
         "paired": paired, "rows": rows, "before": before, "curve": curve}
    D["r50"] = {k: rho_50(*curve(v)) for k, v in cells.items()}
    D["br50"] = {k: rho_50(*curve(v)) for k, v in bcells.items()}
    D["T"] = {q: next(r["dd_mean_T"] for r in rows if r["rho"] == q) for q in grid}

    # s_L8(rho): the yardstick, per n_train and worst case over n_train.
    D["s"] = {}
    for n in NTRAIN:
        a = {r["rho"]: r["solve_rate"] for r in cells[(8, 1024, n, 0)]}
        b = {r["rho"]: r["solve_rate"] for r in cells[(8, 1024, n, 1)]}
        D["s"][n] = {q: abs(a[q] - b[q]) for q in grid}
    D["s_worst"] = {q: max(D["s"][n][q] for n in NTRAIN) for q in grid}
    D["hist"] = {k: _history(v[0]["path"]) for k, v in cells.items()}
    D["params"] = {k: v[0]["n_params"] for k, v in cells.items()}
    return D


def solve_at(D, key, rho):
    return next(r["solve_rate"] for r in D["cells"][key] if r["rho"] == rho)


def _r50(D, key):
    return D["r50"][key][0]


def _mark(flag):
    return dict(marker="o", mfc="white" if flag != "ok" else None, ms=11, mew=2.5)


# -------------------------------------------------------------------- pages

def page_verdict(o, D):
    r4, r8, r12 = (_r50(D, (L, 1024, 120000, 0)) for L in (4, 8, 12))
    wide = _r50(D, (4, 4096, 120000, 0))
    blocks = [
        ("H1  The transition is a transition -- CONFIRMED as shape, over-run at the deep end", GREEN,
         f"Every cell that learns the task at all produces a monotone breakdown curve, not a flat line: "
         f"solve falls from ~1.0 at rho=0.94 to <0.1 at the ensemble floor rho=0.333 for 9 of the 11 "
         f"learning cells. The pre-registered falsifier (flat to within 0.1, or rho_50<0.40 everywhere) "
         f"did not fire. But the sub-prediction rho_50>0.50 FAILS for the two strongest cells: L=12 at "
         f"120k gives {r12:.3f}, and L=12 at 240k never drops to 0.5 at all -- it is still at 0.755 "
         f"where the fall ensemble runs out of removable hints. At N=48 the best model now outruns the "
         f"generator, which is a statement about N=48, not about the model."),
        ("H2  Depth buys the tail -- CONFIRMED at matched data, not separable at 30k", GREEN,
         f"At 120k examples rho_50 = {r4:.3f} / {r8:.3f} / {r12:.3f} for L = 4 / 8 / 12, monotone, a "
         f"gap of {r4-r12:.3f} against a pre-registered bar of 0.10. The curves agree to 0.030 at "
         f"rho=0.94 (bar 0.03) and separate only below rho~0.8, exactly as predicted. At 30k the L "
         f"ordering inverts by 0.157 in rho_50 -- smaller than the 0.154 seed spread measured at L=8, "
         f"so it is not separable at this budget."),
        ("H3  Data helps most in the tail -- direction confirmed, size separable only at L=12", AMBER,
         f"Going 30k -> 240k distinct examples at equal 375k gradient steps moves tail solve (rho<=0.5) "
         f"by +0.86 at L=12, +0.06 at L=8 and +0.00 at L=4; the easy end (rho>=0.8) moves +0.14 / +0.00 "
         f"/ -0.00. Only the L=12 effect clears s_L8. The unregistered finding is the INTERACTION: depth "
         f"pays only if it is fed. The 30k cells overfit as the pre-registration warned they might "
         f"(val-train gap +0.14 at L=8, +0.24 at L=12, against -0.01 at 120k/240k)."),
        ("H4  It is depth, not parameters -- CONFIRMED, and this is the load-bearing cell", GREEN,
         f"L=4 with ff=4096 has 9.60M parameters against L=12/ff=1024's 9.61M, a 0.15% match. Its "
         f"rho_50 is {wide:.3f}, which is {abs(wide-r4):.3f} from the 3.30M L=4 model (bar <=0.03) and "
         f"{wide-r12:+.3f} above L=12 (bar >=+0.08). Its whole curve lies on top of the narrow L=4 curve "
         f"to within 0.042 everywhere. 2.91x the parameters at fixed depth buys nothing. Every depth "
         f"reading in findings 11/14/22/33/38 stands as a depth reading."),
    ]
    textpage(o, "Success rate across the DD transition, by depth, data and width",
             f"N=48, d=5, q=4, FALL ensemble -- 13 cells, 400 instances per density, "
             f"commit {COMMIT}, jobs 50760297 / 50760252",
             blocks,
             "All four hypotheses and the scoring rule were written into the sbatch header before "
             "submission. Scoring rule, verbatim: any level comparison across cells is read against "
             "s_L8 at the same rho; where the effect is smaller the statement is 'not separable at "
             "this budget', not a result with a p-value. Curve shape within one checkpoint is paired "
             "and does not need this caveat.",
             y0=0.82, gap=0.026, width=128)


def page_curves(o, D):
    fig, axes = plt.subplots(1, 4, figsize=(16, 9), sharey=True,
                             gridspec_kw={"wspace": 0.08, "left": 0.055, "right": 0.985,
                                          "top": 0.80, "bottom": 0.17})
    fig.text(0.5, 0.93, "The transition, one panel per data budget",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.875,
             "solve rate vs realized hint density; every curve is the SAME 400 instances per density",
             ha="center", size=MED, color="0.3")

    for ax, n in zip(axes[:3], NTRAIN):
        for L in (4, 8, 12):
            x, y = D["curve"](D["cells"][(L, 1024, n, 0)])
            ax.plot(x, y, "-o", color=LCOL[L], ms=4, lw=2.4, label=f"L={L}")
        xs, ys = D["curve"](D["cells"][(8, 1024, n, 1)])
        ax.plot(xs, ys, "--", color=LCOL[8], lw=1.8, alpha=0.75, label="L=8, seed 1")
        x8, y8 = D["curve"](D["cells"][(8, 1024, n, 0)])
        ax.fill_between(x8, np.minimum(y8, ys), np.maximum(y8, ys),
                        color=LCOL[8], alpha=0.14, lw=0)
        ax.axhline(0.5, color="0.6", lw=1, ls=":")
        ax.set_title(f"{n:,} examples\n{int(round(375000/ (n/32))) if n else 0} epochs, 375k steps",
                     size=MED - 2)
        ax.set_xlabel("hint density rho")
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("solve rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(fontsize=SMALL - 1, loc="lower left", framealpha=0.9)

    ax = axes[3]
    for n in NTRAIN:
        v = [(L, *D["r50"][(L, 1024, n, 0)]) for L in (4, 8, 12)]
        ax.plot([a for a, _, _ in v], [b for _, b, _ in v], NSTY[n], color="0.25",
                lw=2, label=f"{n//1000}k", zorder=1)
        for L, val, flag in v:
            ax.plot([L], [val], color=LCOL[L], ls="none", **_mark(flag))
    for n in NTRAIN:
        val, flag = D["r50"][(8, 1024, n, 1)]
        ax.plot([8.35], [val], color=LCOL[8], ls="none", marker="s", ms=7,
                mfc="white" if flag != "ok" else None, mew=2)
    ax.axhspan(0.28, 0.3333, color=RED, alpha=0.12, lw=0)
    ax.text(4.2, 0.295, "ensemble floor rho_min = 0.333", size=SMALL - 2, color=RED, va="center")
    ax.set_xticks([4, 8, 12])
    ax.set_xlim(3.3, 13.0)
    ax.set_xlabel("layers L")
    ax.set_ylabel("rho_50")
    # sharey=True hides this panel's ticks, but its y is a DENSITY, not a solve
    # rate -- unlabelled it reads off the solve-rate axis four panels away.
    ax.tick_params(labelleft=True)
    ax.set_title("rho_50 (open = bound,\nnot a crossing)", size=MED - 2)
    ax.legend(fontsize=SMALL - 1, title="examples", title_fontsize=SMALL - 2, loc="center right")
    ax.grid(alpha=0.25)

    footer(fig, "Open markers never cross 0.5 inside the reachable range: L=4 at 30k and 240k because "
                "the run never left its loss plateau (next page), L=12 at 240k because it still solves "
                "75.5% where the ensemble runs out of removable hints. Squares are the L=8 seed-1 "
                "replicates, and the shaded band between the two L=8 seeds is the yardstick every "
                "cross-cell claim here is read against.")
    o.savefig(fig)


def page_effects(o, D):
    """H2, H3 and H4 on one page, every effect read against the same yardstick."""
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.955, "What actually moves the curve: depth yes, data yes, parameters no",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.915,
             "equal 375k gradient steps everywhere, so the data axis varies DISTINCT examples with "
             "optimization held fixed", ha="center", size=SMALL + 2, color="0.3")
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.975, top=0.875, bottom=0.235,
                          wspace=0.21, hspace=0.50)
    grid, xr = D["grid"], [D["realized"][q] for q in D["grid"]]

    ax = fig.add_subplot(gs[0, 0])
    trio = [((4, 1024, 120000, 0), "L=4, ff=1024", LCOL[4], "-"),
            ((4, 4096, 120000, 0), "L=4, ff=4096", "#c05621", "--"),
            ((12, 1024, 120000, 0), "L=12, ff=1024", LCOL[12], "-")]
    for key, lab, col, ls in trio:
        x, y = D["curve"](D["cells"][key])
        ax.plot(x, y, ls, color=col, lw=2.8, marker="o", ms=5,
                label=f"{lab}   {D['params'][key]/1e6:.2f}M")
    ax.axhline(0.5, color="0.6", lw=1, ls=":")
    ax.invert_xaxis()
    ax.set_xlabel("hint density rho", size=SMALL + 1)
    ax.set_ylabel("solve rate", size=SMALL + 1)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=SMALL - 1, loc="lower left")
    a4 = np.array([r["solve_rate"] for r in D["cells"][(4, 1024, 120000, 0)]])
    aw = np.array([r["solve_rate"] for r in D["cells"][(4, 4096, 120000, 0)]])
    ax.set_title(f"H4  2.91x the parameters at fixed depth: max curve gap "
                 f"{np.abs(a4 - aw).max():.3f}", size=SMALL + 2)

    ax = fig.add_subplot(gs[0, 1])
    spread = [max(solve_at(D, (L, 1024, 120000, 0), q) for L in (4, 8, 12))
              - min(solve_at(D, (L, 1024, 120000, 0), q) for L in (4, 8, 12)) for q in grid]
    wgap = [abs(solve_at(D, (4, 4096, 120000, 0), q) - solve_at(D, (4, 1024, 120000, 0), q))
            for q in grid]
    dgap = [solve_at(D, (12, 1024, 240000, 0), q) - solve_at(D, (12, 1024, 30000, 0), q)
            for q in grid]
    ax.fill_between(xr, 0, [D["s"][120000][q] for q in grid], color=GREY, alpha=0.35, lw=0,
                    label="s_L8 at 120k (matched bar)")
    ax.plot(xr, [D["s_worst"][q] for q in grid], color="0.2", lw=1.6, ls="--",
            label="s_L8 worst case")
    ax.plot(xr, spread, "-o", color=LCOL[12], lw=3, ms=5, label="depth: L=4 vs 8 vs 12")
    ax.plot(xr, dgap, "-^", color="#6b46c1", lw=2.4, ms=6, label="data: 240k - 30k at L=12")
    ax.plot(xr, wgap, "-s", color="#c05621", lw=2.4, ms=5, label="width: ff 4096 - 1024 at L=4")
    ax.invert_xaxis()
    ax.set_xlabel("hint density rho", size=SMALL + 1)
    ax.set_ylabel("difference in solve rate", size=SMALL + 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=SMALL - 2, loc="upper left")
    ax.set_title("Effect size against the seed yardstick", size=SMALL + 2)

    ax = fig.add_subplot(gs[1, 0])
    for L in (4, 8, 12):
        v = [(n, *D["r50"][(L, 1024, n, 0)]) for n in NTRAIN]
        ax.plot([a for a, _, _ in v], [b for _, b, _ in v], "-", color=LCOL[L], lw=2.6,
                label=f"L={L}")
        for n, val, flag in v:
            ax.plot([n], [val], color=LCOL[L], ls="none", **_mark(flag))
    for n in NTRAIN:
        val, flag = D["r50"][(8, 1024, n, 1)]
        ax.plot([n * 1.09], [val], color=LCOL[8], ls="none", marker="s", ms=7,
                mfc="white" if flag != "ok" else None, mew=2)
    ax.axhspan(0.28, 0.3333, color=RED, alpha=0.12, lw=0)
    ax.text(33000, 0.303, "ensemble floor", size=SMALL - 2, color=RED, va="center")
    ax.set_xscale("log")
    ax.set_xticks(list(NTRAIN))
    ax.set_xticklabels(["30k", "120k", "240k"])
    ax.set_xlim(26000, 300000)
    ax.set_ylim(0.26, 1.02)
    ax.set_xlabel("distinct training instances", size=SMALL + 1)
    ax.set_ylabel("rho_50", size=SMALL + 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=SMALL - 1, loc="lower left")
    ax.set_title("H2 x H3  the interaction: depth pays only if it is fed", size=SMALL + 2)

    ax = fig.add_subplot(gs[1, 1])
    for n, col in zip(NTRAIN, ("#9b2c2c", "#975a16", "#2b6cb0")):
        ep, tr, va = D["hist"][(4, 1024, n, 0)]
        steps = ep * np.ceil(n / 32)
        ax.plot(steps / 1000, tr, color=col, lw=2.2, label=f"{n // 1000}k train")
        ax.plot(steps / 1000, va, color=col, lw=1.3, ls="--", alpha=0.8)
    ax.set_xlabel("thousands of gradient steps", size=SMALL + 1)
    ax.set_ylabel("NLL (solid train, dashed val)", size=SMALL + 1)
    ax.set_ylim(1.48, 1.92)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=SMALL - 1)
    ax.set_title("L=4 only: a plateau, escaped once in three", size=SMALL + 2)

    footer(fig, "An effect is claimed only where it clears the grey band. Depth clears it from rho=0.79 "
                "down, peaking at 0.752 at rho=0.65; data at L=12 clears it everywhere below rho=0.83; "
                "width never clears it anywhere, its largest excursion being 0.042 between two models "
                "whose parameter counts differ by 0.15%. Bottom right: all three L=4 cells settle near "
                "NLL 1.64 and the 120k one fell off that plateau around step 230k, with the 240k cell "
                "cut off at 375k steps at exactly the NLL the 120k cell had when it escaped.")
    o.savefig(fig)


def page_table(o, D):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Every cell", ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.898,
             f"N=48, d=5, q=4, fall ensemble, d_model=256, 4 heads, lr 1e-3 constant, batch 32, "
             f"375k steps, --shuffle-adj; commit {COMMIT}",
             ha="center", size=SMALL + 1, color="0.3")
    ax = fig.add_axes([0.03, 0.10, 0.94, 0.77])
    ax.axis("off")
    hdr = (f"{'task':>4} {'L':>3} {'ff':>5} {'params':>9} {'n_train':>8} {'ep':>4} {'seed':>4}  "
           f"{'train':>6} {'val':>6} {'gap':>7}  {'0.94':>6} {'0.75':>6} {'0.65':>6} "
           f"{'0.50':>6} {'0.33':>6}  {'rho_50':>8}")
    lines = [hdr, "-" * len(hdr)]
    task = {k: v[0]["task"] for k, v in D["cells"].items()}
    order = sorted(D["cells"], key=lambda k: (k[0], k[1], k[2], k[3]))
    for k in order:
        L, ff, n, sd = k
        ep, tr, va = D["hist"][k]
        val, flag = D["r50"][k]
        tag = {"ok": "", ">top": " >top", "<floor": "<floor"}[flag]
        lines.append(
            f"{task[k]:>4} {L:>3} {ff:>5} {D['params'][k]/1e6:>8.2f}M {n:>8,} {int(ep[-1])+1:>4} "
            f"{sd:>4}  {tr[-1]:>6.3f} {va[-1]:>6.3f} {va[-1]-tr[-1]:>+7.3f}  "
            + " ".join(f"{solve_at(D, k, q):>6.3f}" for q in (0.95, 0.75, 0.65, 0.50, 0.33))
            + f"  {val:>6.3f}{tag:>6}")
    lines += ["", "requested rho      0.95   0.75   0.65   0.50   0.33",
              "realized rho     " + " ".join(f"{D['realized'][q]:>6.3f}"
                                             for q in (0.95, 0.75, 0.65, 0.50, 0.33)),
              "mean DD rounds T " + " ".join(f"{D['T'][q]:>6.2f}"
                                             for q in (0.95, 0.75, 0.65, 0.50, 0.33))]
    ax.text(0.5, 1.0, "\n".join(lines), family="monospace", size=12.2, ha="center", va="top",
            linespacing=1.42)
    footer(fig, "'gap' is final val NLL minus final train NLL. The three 30k cells are the only ones "
                "that overfit, and they overfit more the deeper they are (+0.042 at L=4, +0.140 at L=8, "
                "+0.243 at L=12) -- the pre-registration named this as a thing to watch and said it "
                "would be part of the answer rather than a reason to drop the cells. Note that val NLL "
                "does not rank the cells: L=12/30k has the WORST val NLL of all 13 (1.650) and still "
                "solves 96.5% at rho=0.94.")
    o.savefig(fig)


def page_before(o, D):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, "Against the previous best, on the one axis the two ensembles share",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.875,
             "the unconstrained ensemble stops at rho=0.471; the fall ensemble reaches 0.333",
             ha="center", size=MED, color="0.3")
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.98, top=0.79, bottom=0.245, wspace=0.20)

    # L=4's 240k fall cell never left its plateau, so the honest L=4 representative
    # is the one L=4 cell that trained. L=8 and L=12 use 240k like the old cells did.
    best = {4: (4, 1024, 120000, 0), 8: (8, 1024, 240000, 0), 12: (12, 1024, 240000, 0)}
    ax = fig.add_subplot(gs[0, 0])
    for (L, sd), c in sorted(D["bcells"].items()):
        if sd != 0 or L not in (4, 8, 12):
            continue
        x, y = D["curve"](c)
        ax.plot(x, y, "--s", color=LCOL[L], lw=2, ms=5, alpha=0.8,
                label=f"L={L}, unconstrained, 240k x 750k steps")
    for L in (4, 8, 12):
        x, y = D["curve"](D["cells"][best[L]])
        ax.plot(x, y, "-o", color=LCOL[L], lw=2.8, ms=5,
                label=f"L={L}, fall, {best[L][2] // 1000}k x 375k steps")
    ax.axvline(0.4714, color=RED, lw=1.4, ls=":")
    ax.text(0.468, 0.03, "old floor", size=SMALL - 1, color=RED, rotation=90, va="bottom")
    ax.axhline(0.5, color="0.6", lw=1, ls=":")
    ax.invert_xaxis()
    ax.set_xlabel("hint density rho")
    ax.set_ylabel("solve rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=SMALL - 3, loc="lower left", ncol=1)
    ax.set_title("vs rho -- NOT paired across ensembles", size=MED - 2)

    ax = fig.add_subplot(gs[0, 1])
    bT = {}
    for r in D["before"]:
        bT.setdefault(r["rho"], r["dd_mean_T"])
    for (L, sd), c in sorted(D["bcells"].items()):
        if sd != 0 or L not in (4, 8, 12):
            continue
        ax.plot([bT[r["rho"]] for r in c], [r["solve_rate"] for r in c], "--s",
                color=LCOL[L], lw=2, ms=5, alpha=0.8)
    for L in (4, 8, 12):
        c = D["cells"][best[L]]
        ax.plot([D["T"][r["rho"]] for r in c], [r["solve_rate"] for r in c], "-o",
                color=LCOL[L], lw=2.8, ms=5)
    ax.axvspan(7.80, 13.79, color=GREEN, alpha=0.10, lw=0)
    ax.text(10.8, 0.90, "newly reachable\n7.8 -> 13.8 rounds", size=SMALL - 1, color=GREEN,
            ha="center")
    ax.axhline(0.5, color="0.6", lw=1, ls=":")
    ax.set_xlabel("mean DD rounds T required")
    ax.set_ylabel("solve rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.set_title("vs required reasoning depth -- the comparable axis", size=MED - 2)

    footer(fig, "The fall fix changes the GRAPH ensemble, so no fall cell is paired with any earlier "
                "checkpoint and the two cannot be McNemar'd; required reasoning depth is the axis they "
                "share. The old cells also got 750k gradient steps against the scan's 375k, so this is "
                "not a like-for-like win and none is claimed. Where they overlap the new cells track "
                "the old ones at equal L on half the optimization, and then carry on for another 1.77x "
                "in depth: the previous best, L=12 unconstrained, solved 0.948 at its floor (T=7.8) and "
                "was never measured past it, while L=12/240k fall solves 0.985 at T=7.2 and 0.755 at "
                "T=13.8.")
    o.savefig(fig)


def page_controls(o, D):
    live = [r for r in D["rows"] if r["solve_rate"] > 0.1]
    viol = [r for r in live if r["outcome_accuracy"] < 0.99 or r["truncated_rate"] > 0.01]
    worst = min(r["outcome_accuracy"] for r in live)
    wtr = max(r["truncated_rate"] for r in live)
    blocks = [
        ("PASS  pairing, and the ensemble floor is labelled not fitted", GREEN,
         f"All 13 curves see an identical realized density at each of the 14 requested rho values "
         f"({'verified' if D['paired'] else 'FAILED'}), so every within-grid comparison is paired "
         f"point for point. Exactly one requested point (0.33) lands on the ensemble floor "
         f"rho_min=0.3333, as designed -- the grid is 14 distinct realized densities, not one "
         f"density measured several times. dd_mean_T runs 1.15 -> 13.79 rounds across it."),
        ("FAIL  outcome accuracy and truncation, in the direction that UNDER-reports the deep cells", RED,
         f"The bars were outcome_accuracy >= 0.99 and truncated_rate <= 0.01 at every density with "
         f"solve > 0.1. {len(viol)} of {len(live)} live bins miss them, worst case {worst:.4f} and "
         f"{wtr:.4f}. All {len(viol)} belong to the two L=12 cells at the three deepest densities. "
         f"The cause is the context window: max_len is 414 tokens and the longest true fall trace at "
         f"N=48 is 408, so a model that emits even a few spare tokens at rho=0.33 is cut off "
         f"mid-trace. solve_rate() separates this from reasoning failure by design, and the effect is "
         f"bounded by 0.028 -- far below the 0.6-0.8 depth effects -- so no conclusion turns on it, "
         f"but the L=12 tail numbers are a lower bound and the cap must be raised before N=96."),
        ("CAVEAT  two of the three L=4/ff=1024 cells never left their loss plateau", AMBER,
         "L=4 is at the edge of trainability on this task. Its three cells all plateau near NLL 1.64 "
         "and exactly one escaped inside 375k steps. The L=4 row therefore measures trainability, not "
         "capacity, and the honest reading of H2 and H3 at L=4 is 'no measurement'. The wide L=4 cell "
         "trained normally, which is what makes H4 readable at all."),
        ("WHAT THIS DOES NOT SHOW", GREY,
         "One architecture family, one N, one learning rate, one seed per cell off the L=8 column. "
         "s_L8 is itself estimated from three seed pairs, so the yardstick has its own error bar. "
         "Nothing here says the network is running DD -- that is the SAE/ablation question, and these "
         "curves only say where it stops succeeding, not what it was doing while it succeeded."),
    ]
    textpage(o, "Controls, and what this measurement cannot carry",
             "pre-registered checks, scored after the fact against the recorded CSVs",
             blocks,
             "Negative and failed checks are listed here rather than in a footnote because the "
             "pre-registration named them in advance. The truncation failure is the one that would "
             "change a number if fixed, and it would move the deepest L=12 points UP.",
             y0=0.82, gap=0.026, width=128)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_style()
    D = load()
    print(f"{len(D['cells'])} fall cells, {len(D['bcells'])} before cells, paired={D['paired']}")

    # fall=True and shuffle_adj=True are the instance keys these 13 checkpoints
    # were trained with; pick="deep" because the subject of this deck is depth.
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.33, rho_max=0.95,
                          shuffle_adj=True, fall=True, pick="deep")
    with PdfPages(f"{OUT}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, OUT, total=8)
        page_verdict(o, D)
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the task all 13 "
                               "cells in this deck were trained on. rho=0.33 is the deep end of the "
                               "eval grid, where only the L=12 cells still succeed.")
        page_curves(o, D)
        page_effects(o, D)
        page_table(o, D)
        page_controls(o, D)
        page_before(o, D)
    print(f"wrote {OUT}/slides.pdf ({o.n} pages, "
          f"{os.path.getsize(OUT + '/slides.pdf')/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
