#!/usr/bin/env python3
"""Stage HOLDOUT — test stage GATE's post-hoc exponent trend on held-out density.

Registered in `runs/stageholdout_2026-11-17/REGISTERED.md`, committed before this
file existed.

WHAT IS BEING TESTED
--------------------
Stage GATE (99) reported, post hoc, that the cost of demanding DD-faithfulness

    ratio = log S_ddf / log S

falls monotonically as the task hardens: 3.456 -> 2.668 across rho 0.95 -> 0.50,
Spearman against rho +0.988.  Nobody predicted it before seeing it, and STATUS.md
flags it as not to be believed until re-registered on held-out data.

`stageexp_fit.py:49` sets FIT_RHOS = GRID[:10], and every fit script in the
project takes its density list from there.  So rho = 0.45, 0.40, 0.35 and 0.33
have been evaluated on all 79 runs and have NEVER entered a ratio, an exponent,
a prefactor or a regime fit.  They are held out with respect to this claim.

WHY THE CLAIM IS SUSPECT, WHICH IS THE POINT
--------------------------------------------
The ratio is undefined at S = 1 (denominator zero) and at S_ddf = 0 (numerator
divergent), and those two exclusions act at OPPOSITE ends of the grid: at high
rho runs drop out for solving everything, at low rho for solving nothing.
Dropping S_ddf = 0 removes exactly the cells with the largest numerator, biasing
the surviving median downward, and it bites harder as rho falls.  That alone
could manufacture the whole +0.988.  So H3 — a fixed population, and a paired
within-run comparison — is the test that decides, not the extrapolation.

Reads `runs/stagebound_2026-11-17/bound_cells.csv` and nothing else.  No
training, no GPU, no new instances, nothing submitted.  Login node, seconds.
"""

import collections
import csv as _csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stageexp_fit import med  # noqa: E402
from stagegate_fit import GRID, FIT_RHOS, spearman  # noqa: E402

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import (SERIES, MUTED, style_axes, label_line,  # noqa: E402
                    series_colors)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = "/scratch/users/hshunt/spin_gnn/runs"
SRC = os.path.join(RUNS, "stagebound_2026-11-17", "bound_cells.csv")
OUT = os.path.join(RUNS, "stageholdout_2026-11-17")
FIGS = os.path.join(OUT, "figures")
DPI = 150

HELD = [0.45, 0.40, 0.35, 0.33]          # never in any fit
# Published in stage GATE's VERDICT, deck 25 and STATUS.md.  D1 must reproduce
# these to three decimal places or the stage stops.
GATE_MEDIANS = {0.95: 3.456, 0.90: 3.574, 0.85: 3.271, 0.80: 3.130,
                0.75: 3.074, 0.70: 2.968, 0.65: 2.911, 0.60: 2.798,
                0.55: 2.726, 0.50: 2.668}
# The six points the registered extrapolation was fitted on.
OLS_RHOS = [0.75, 0.70, 0.65, 0.60, 0.55, 0.50]


def fig(name):
    f, ax = plt.subplots(figsize=(7.2, 4.3))
    return f, ax, os.path.join(FIGS, name)


def save(f, path):
    f.tight_layout()
    f.savefig(path, dpi=DPI)
    plt.close(f)
    print("  wrote", os.path.basename(path))


def load():
    """Per (run, rho) cell with the ratio and the two per-index hazards."""
    cells = []
    with open(SRC) as fh:
        for r in _csv.DictReader(fh):
            c = {"run_id": r["run_id"], "cell": r["cell"],
                 "rho": round(float(r["rho"]), 4),
                 "params": int(float(r["params"])),
                 "S": float(r["S"]), "S_ddf": float(r["S_ddf"]),
                 "S_perm": float(r["S_perm"]),
                 "njs": int(float(r["njs"]))}
            # GATE's definedness condition, copied verbatim so D1 can bite.
            c["ratio"] = (np.log(c["S_ddf"]) / np.log(c["S"])
                          if 0 < c["S"] < 1 and 0 < c["S_ddf"] < 1
                          else float("nan"))
            # H4: the per-index geometric-mean hazards.  By construction
            # ratio = log(1 - h_ddf) / log(1 - h) exactly, so this is a
            # decomposition of the ratio, not an independent measurement of it.
            if c["njs"] > 0:
                c["h"] = (1.0 - c["S"] ** (1.0 / c["njs"])
                          if c["S"] > 0 else 1.0)
                c["h_ddf"] = (1.0 - c["S_ddf"] ** (1.0 / c["njs"])
                              if c["S_ddf"] > 0 else 1.0)
            else:
                c["h"] = c["h_ddf"] = float("nan")
            cells.append(c)
    return cells


def cause(c):
    """Why a cell has no ratio.  Order matters: S == 1 and S_ddf == 0 can
    never co-occur (S_ddf <= S), but S == 0 forces S_ddf == 0."""
    if c["S"] >= 1.0:
        return "S==1"
    if c["S"] <= 0.0:
        return "S==0"
    if c["S_ddf"] >= 1.0:
        return "Sddf==1"
    if c["S_ddf"] <= 0.0:
        return "Sddf==0"
    return "ok"


def ols(xs, ys):
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    sl = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return float(sl), float(y.mean() - sl * x.mean())


# ----------------------------------------------------------------- figures

def h1_extrapolation(by_rho, slope, icept):
    f, ax, path = fig("H1_extrapolation.png")
    c = series_colors(2, highlight=0)
    fit_x = [r for r in FIT_RHOS if by_rho.get(r)]
    fit_y = [med([q["ratio"] for q in by_rho[r]]) for r in fit_x]
    hel_x = [r for r in HELD if by_rho.get(r)]
    hel_y = [med([q["ratio"] for q in by_rho[r]]) for r in hel_x]

    ex = np.linspace(0.78, 0.31, 50)
    ax.plot(ex, slope * ex + icept, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.plot(fit_x, fit_y, color=c[0], lw=2.0, marker="o", ms=5, zorder=3)
    ax.plot(hel_x, hel_y, color=SERIES[1], lw=2.0, marker="o", ms=7,
            mfc="white", mew=2.0, zorder=4)
    # The two densities the censoring ledger disqualifies from the headline.
    for r, y in zip(hel_x, hel_y):
        if r <= 0.35:
            ax.plot([r], [y], marker="x", ms=11, color=MUTED, mew=2.0,
                    zorder=5)
    ax.axvline(0.475, color=MUTED, lw=1.0, ls=":")
    style_axes(ax, xlabel=r"hint density $\rho$",
               ylabel=r"$\log S_{\rm DD}\,/\,\log S$")
    ax.set_xlim(0.975, 0.305)
    lo = min(list(fit_y) + list(hel_y)) - 0.38
    ax.set_ylim(lo, max(fit_y) + 0.30)
    # Not at xs[0]: the grid descends, so the first point sits at the left
    # edge and a label there runs straight over the rho = 0.90 peak.
    i = fit_x.index(0.70)
    label_line(ax, fit_x[i], fit_y[i], "fitted densities", color=c[0], dy=15)
    if hel_x:
        label_line(ax, hel_x[0], hel_y[0], "held out", color=SERIES[1],
                   dy=-19)
    top = ax.get_ylim()[1]
    ax.text(0.52, top - 0.06, "never used in any fit  " + r"$\rightarrow$",
            fontsize=8, color=MUTED, ha="right", va="top")
    ax.text(0.96, lo + 0.26,
            "dashed: least squares on " + r"$\rho\in[0.75,0.50]$"
            + ", extrapolated", fontsize=8, color=MUTED, ha="left",
            va="bottom")
    ax.text(0.96, lo + 0.06,
            r"$\times$ marks densities the censoring ledger disqualifies",
            fontsize=8, color=MUTED, ha="left", va="bottom")
    save(f, path)


def h3a_population(by_rho, fixed_ids):
    f, ax, path = fig("H3a_population.png")
    c = series_colors(2, highlight=1)
    xs = [r for r in GRID if by_rho.get(r)]
    ya = [med([q["ratio"] for q in by_rho[r]]) for r in xs]
    # Only over the ten fitted densities: C10 is defined as measurable at all
    # ten, so below the grid its members start dropping out too and it stops
    # being a fixed population.  Drawing it there would claim a control the
    # curve no longer has.
    xf = [r for r in FIT_RHOS if by_rho.get(r)]
    yf = [med([q["ratio"] for q in by_rho[r] if q["run_id"] in fixed_ids])
          for r in xf]
    ax.plot(xs, ya, color=c[0], lw=2.0, marker="o", ms=4)
    ax.plot(xf, yf, color=c[1], lw=2.4, marker="s", ms=4)
    ax.axvline(0.475, color=MUTED, lw=1.0, ls=":")
    style_axes(ax, xlabel=r"hint density $\rho$",
               ylabel=r"median $\log S_{\rm DD}\,/\,\log S$")
    ax.set_xlim(0.975, 0.305)
    ax.set_ylim(min(ya) - 0.22, max(ya + yf) + 0.10)
    # In the held-out stretch, where the blue curve is absent and nothing
    # can collide with it.
    i = xs.index(0.45)
    label_line(ax, xs[i], ya[i], "all usable cells", color=c[0], dy=14)
    j = xf.index(0.70)
    label_line(ax, xf[j], yf[j], f"fixed population, n={len(fixed_ids)}",
               color=c[1], dy=14)
    ax.text(0.52, ax.get_ylim()[1] - 0.02,
            "held out  " + r"$\rightarrow$", fontsize=8, color=MUTED,
            ha="right", va="top")
    ax.text(0.96, ax.get_ylim()[0] + 0.02,
            "the fixed population is the runs measurable at every fitted\n"
            "density, so it is drawn only where it really is fixed",
            fontsize=8, color=MUTED, ha="left", va="bottom")
    save(f, path)


def h3b_paired(diffs):
    f, ax, path = fig("H3b_paired.png")
    d = sorted(diffs)
    x = np.arange(len(d))
    neg = sum(1 for v in d if v < 0)
    col = [SERIES[0] if v < 0 else SERIES[1] for v in d]
    ax.bar(x, d, color=col, width=0.9, lw=0)
    ax.axhline(0.0, color="#333333", lw=1.0)
    ax.axhline(med(d), color=MUTED, lw=1.2, ls="--")
    style_axes(ax, xlabel="run, sorted",
               ylabel=r"ratio at $\rho=0.50$ $-$ ratio at $\rho=0.95$")
    ax.text(len(d) * 0.60, med(d), f"median {med(d):+.2f}", fontsize=8,
            color=MUTED, ha="left", va="bottom")
    ax.text(len(d) - 0.5, ax.get_ylim()[1] * 0.92,
            f"{neg} of {len(d)} runs fall ({100.0 * neg / len(d):.0f}%)",
            fontsize=9, color=MUTED, ha="right", va="top")
    ax.text(len(d) * 0.22, ax.get_ylim()[0] * 0.95,
            "each bar is one run compared with itself, so no change of\n"
            "population can produce this",
            fontsize=8, color=MUTED, ha="left", va="bottom")
    save(f, path)


def h4_hazards(by_rho):
    f, ax, path = fig("H4_hazards.png")
    xs = [r for r in FIT_RHOS if by_rho.get(r)]
    hh = [med([q["h"] for q in by_rho[r]]) for r in xs]
    hd = [med([q["h_ddf"] for q in by_rho[r]]) for r in xs]
    rh = [v / hh[0] for v in hh]
    rd = [v / hd[0] for v in hd]
    c = series_colors(2, highlight=0)
    ax.plot(xs, rh, color=c[0], lw=2.2, marker="o", ms=4)
    ax.plot(xs, rd, color=c[1], lw=2.2, marker="s", ms=4)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=":")
    style_axes(ax, xlabel=r"hint density $\rho$",
               ylabel=r"per-index hazard, relative to $\rho=0.95$")
    ax.set_xlim(0.975, 0.455)
    # Both curves start at 1.0 by construction, so a label at xs[0] lands on
    # top of the other one; separate them where the curves have separated.
    i = xs.index(0.65)
    label_line(ax, xs[i], rh[i], r"colour hazard $\bar h$", color=c[0], dy=15)
    j = xs.index(0.60)
    label_line(ax, xs[j], rd[j], r"faithfulness hazard $\bar h_{\rm DD}$",
               color=c[1], dy=-18)
    ax.text(0.95, ax.get_ylim()[1] * 0.97,
            f"over this span " + r"$\bar h$" + f" grows {rh[-1]:.2f}x and "
            + r"$\bar h_{\rm DD}$" + f" grows {rd[-1]:.2f}x",
            fontsize=8, color=MUTED, ha="left", va="top")
    save(f, path)


def d2_censoring(by_rho_all):
    f, ax, path = fig("D2_censoring.png")
    xs = [r for r in GRID]
    keys = ["S==1", "Sddf==1", "S==0", "Sddf==0"]
    cols = [MUTED, "#c9c6bf", SERIES[1], SERIES[0]]
    bot = np.zeros(len(xs))
    w = np.arange(len(xs))
    for k, col in zip(keys, cols):
        v = np.array([sum(1 for c in by_rho_all[r] if cause(c) == k)
                      for r in xs], float)
        ax.bar(w, v, bottom=bot, color=col, width=0.76, lw=0)
        bot += v
    ax.set_xticks(w)
    ax.set_xticklabels([f"{r:.2f}" for r in xs], fontsize=8)
    style_axes(ax, xlabel=r"hint density $\rho$",
               ylabel="cells with no ratio, of 79")
    ax.set_ylim(0, 79)
    ax.axhline(39.5, color=MUTED, lw=1.0, ls="--")
    ax.text(9.2, 40.5, "half the runs", fontsize=8, color=MUTED, ha="right",
            va="bottom")
    ax.axvline(9.5, color="#333333", lw=1.0, ls=":")
    ax.text(9.65, 77, "held out", fontsize=8, color=MUTED, ha="left",
            va="top")
    ax.text(-0.35, 77,
            "grey: " + r"$S=1$" + ", solved every instance, so "
            + r"$\log S=0$" + "\n"
            "orange: " + r"$S=0$" + ", solved none of them\n"
            "blue: " + r"$S_{\rm DD}=0$" + ", never once DD-faithful",
            fontsize=8, color=MUTED, ha="left", va="top")
    save(f, path)


# ------------------------------------------------------------------- main

def main():
    os.makedirs(FIGS, exist_ok=True)
    cells = load()
    by_rho_all = collections.defaultdict(list)
    for c in cells:
        by_rho_all[c["rho"]].append(c)
    by_rho = collections.defaultdict(list)
    for c in cells:
        if c["ratio"] == c["ratio"]:
            by_rho[c["rho"]].append(c)

    out, a = [], lambda s="": out.append(s)
    a("STAGE HOLDOUT — the post-hoc exponent trend, on densities never fitted")
    a("=" * 78)
    runs = {c["run_id"] for c in cells}
    a(f"cells {len(cells)}   runs {len(runs)}   densities "
      f"{len(by_rho_all)}   source {os.path.basename(SRC)}")
    a("")
    a("The ratio log S_DD / log S is defined only where 0 < S < 1 and")
    a("0 < S_ddf < 1, exactly as stage GATE defined it.  Those two exclusions")
    a("act at opposite ends of the grid, which is the reason for H3.")

    # ---------------------------------------------------------------- D3
    a("")
    a("-" * 78)
    a("D3 — MONOTONICITY AT THE NEW DENSITIES, AND WHY IT MISSES")
    bad = [c for c in cells if c["S_ddf"] > c["S"] + 1e-12]
    lowbad = [c for c in bad if c["rho"] < 0.50]
    a(f"  S_ddf > S on {len(bad)} of {len(cells)} cells "
      f"({len(lowbad)} of them below the fit grid)")
    a(f"  registered: 0 — 'any violation is a scorer bug'  ->  "
      f"{'PASS' if not bad else 'MISS'}")
    for c in bad:
        a(f"    {c['cell']}  run {c['run_id']}  rho {c['rho']:.2f}:  "
          f"S {c['S']:.6f}  S_ddf {c['S_ddf']:.6f}  "
          f"excess {c['S_ddf'] - c['S']:+.6f}")
    if bad:
        a("  THE PREDICTION WAS ILL-FOUNDED, AND THE VIOLATION IS NOT A BUG.")
        a("  S_ddf <= S is not guaranteed by anything.  S_perm = S exactly at")
        a("  that cell, so this is not product-versus-measurement slack: the")
        a("  two criteria are simply NOT NESTED.  The permissive hazard counts")
        a("  rollouts that got a colour wrong; the DD-faithful hazard counts")
        a("  rollouts that named a cell direct deduction had not yet forced.")
        a("  A rollout can name an unforced cell and still guess the right")
        a("  colour — penalised by h_ddf, not by h — and can get a colour")
        a("  wrong on a cell that WAS forced — penalised by h, not by h_ddf.")
        a("  Neither set contains the other, so neither solve rate dominates.")
        a("  The inequality that does hold by construction is the one stage")
        a("  BOUND registered, S_joint <= min(S_ddf, S_perm), because the")
        a("  joint criterion is the UNION of the two failure sets:")
        jb = [c for c in cells if c["S_ddf"] > c["S_perm"] + 1e-12]
        a(f"    S_ddf > S_perm on {len(jb)} of {len(cells)} cells — the same")
        a("    cell, and the excess is one wrong colour on a forced cell.")
        a("  CONSEQUENCE: stage GATE's rho_50_ddf >= rho_50_published is an")
        a("  EMPIRICAL fact on 1105 of 1106 cells, not a logical one, and")
        a("  should stop being described as if the criterion were strictly")
        a("  stronger.  It is stronger in one direction only.")
        a("")
        a("  AND THIS WAS ALREADY KNOWN.  Finding 99 records it, in STATUS.md,")
        a("  naming THIS CELL: 'the criteria are NOT NESTED ... 1 of 1106")
        a("  cells has S_ddf > S (r68 at rho 0.95, +0.0012)'.  So D3 is not a")
        a("  discovery and not an ill-founded guess either — it is a")
        a("  prediction I registered against a fact my own status file")
        a("  already stated.  The failure was not reading it before writing")
        a("  the registration.  Scored MISS, and the lesson is narrower and")
        a("  worse than 'the inequality does not hold': CHECK THE CLAIM")
        a("  AGAINST STATUS BEFORE REGISTERING IT.")

    # ---------------------------------------------------------------- D1
    a("")
    a("-" * 78)
    a("D1 — NO LOADER DRIFT: GATE'S TEN MEDIANS, RECOMPUTED HERE")
    a("     rho   GATE published   recomputed      diff      n")
    worst = 0.0
    for r in FIT_RHOS:
        v = [c["ratio"] for c in by_rho[r]]
        m = med(v)
        d = abs(m - GATE_MEDIANS[r])
        worst = max(worst, d)
        a(f"    {r:.2f}        {GATE_MEDIANS[r]:9.3f}    {m:9.3f}  "
          f"{d:8.4f}  {len(v):5d}")
    d1 = worst < 0.0005
    a(f"  worst absolute difference {worst:.5f}")
    a(f"  registered: reproduce to three decimal places  ->  "
      f"{'PASS' if d1 else 'FAIL'}")
    if not d1:
        a("  THE STAGE STOPS HERE AS REGISTERED: H1-H4 are not scored, because")
        a("  a loader difference means the held-out numbers are not comparable")
        a("  with the published ones they are being tested against.")

    # ---------------------------------------------------------------- D2
    a("")
    a("-" * 78)
    a("D2 — THE CENSORING LEDGER, PUBLISHED NOT HIDDEN")
    a("     rho   usable    S==1  Sddf==1    S==0  Sddf==0   median ratio")
    dens_rows = []
    for r in GRID:
        cnt = collections.Counter(cause(c) for c in by_rho_all[r])
        v = [c["ratio"] for c in by_rho[r]]
        tag = "  <- held out" if r in HELD else ""
        a(f"    {r:.2f}   {cnt['ok']:6d}  {cnt['S==1']:6d}  "
          f"{cnt['Sddf==1']:7d}  {cnt['S==0']:6d}  {cnt['Sddf==0']:7d}   "
          f"{med(v):12.4f}{tag}")
        dens_rows.append({
            "rho": r, "n_usable": cnt["ok"], "n_S_eq_1": cnt["S==1"],
            "n_Sddf_eq_1": cnt["Sddf==1"], "n_S_eq_0": cnt["S==0"],
            "n_Sddf_eq_0": cnt["Sddf==0"],
            "median_ratio": f"{med(v):.6f}",
            "p25": f"{np.percentile(v, 25):.6f}" if v else "",
            "p75": f"{np.percentile(v, 75):.6f}" if v else "",
            "median_h": f"{med([c['h'] for c in by_rho[r]]):.6f}" if v else "",
            "median_h_ddf": (f"{med([c['h_ddf'] for c in by_rho[r]]):.6f}"
                             if v else ""),
            "held_out": int(r in HELD)})
    a("  at rho 0.35 and 0.33 more than half the runs solve nothing under the")
    a("  DD-faithful criterion, so their medians are taken over the survivors")
    a("  and are reported as CENSORED, not used to decide H1 or H2.")

    # ---------------------------------------------------------------- H1
    slope, icept = ols(OLS_RHOS, [GATE_MEDIANS[r] for r in OLS_RHOS])
    a("")
    a("-" * 78)
    a("H1 — DOES THE TREND CONTINUE BELOW THE FIT GRID")
    a(f"  least squares on the six points rho 0.75..0.50 of the PUBLISHED")
    a(f"  medians: slope {slope:+.3f} per unit rho, intercept {icept:.3f}")
    a("     rho   predicted   measured   registered band      verdict")
    bands = {0.45: (2.40, 2.70), 0.40: (2.25, 2.65)}
    h1 = True
    for r in [0.45, 0.40]:
        m = med([c["ratio"] for c in by_rho[r]])
        pred = slope * r + icept
        lo, hi = bands[r]
        ok = lo <= m <= hi
        h1 = h1 and ok
        a(f"    {r:.2f}   {pred:9.3f}  {m:9.3f}   [{lo:.2f}, {hi:.2f}]"
          f"            {'in' if ok else 'OUT'}")
    stopped = any(med([c["ratio"] for c in by_rho[r]]) >= 2.668
                  for r in [0.45, 0.40])
    a(f"  falsifier: either median reaches 2.668 (the rho=0.50 value) -> "
      f"{'TRIPPED' if stopped else 'not tripped'}")
    a(f"  registered: both in band  ->  {'HIT' if h1 else 'MISS'}")
    # Where the sequence stops descending.  Reported whether or not H1 hits,
    # because "in band" and "still monotone" are different claims and the
    # deck must not let the first stand in for the second.
    seq = [(r, med([c["ratio"] for c in by_rho[r]])) for r in GRID
           if by_rho[r]]
    inv = [(seq[i - 1][0], seq[i][0], seq[i][1] - seq[i - 1][1])
           for i in range(1, len(seq)) if seq[i][1] > seq[i - 1][1]]
    a(f"  steps where the median RISES as rho falls: {len(inv)} of "
      f"{len(seq) - 1}")
    for lo_r, hi_r_, dv in inv:
        a(f"    {lo_r:.2f} -> {hi_r_:.2f}: {dv:+.4f}")
    a("  the ten fitted densities descend monotonically apart from the")
    a("  0.95 -> 0.90 step, which was published that way.  Below the grid the")
    a("  sequence FLATTENS: the fall from 0.50 to 0.33 is only "
      f"{seq[9][1] - seq[-1][1]:.3f}, against {seq[0][1] - seq[9][1]:.3f}")
    a("  across the ten fitted densities, so the linear extrapolation")
    a("  over-predicts the decline at the far end even though both held-out")
    a("  medians land inside the registered bands.")
    for r in [0.35, 0.33]:
        m = med([c["ratio"] for c in by_rho[r]])
        a(f"  for the record, censored: rho {r:.2f} median {m:.3f} over "
          f"{len(by_rho[r])} survivors (predicted {slope * r + icept:.3f})")

    # ---------------------------------------------------------------- H2
    a("")
    a("-" * 78)
    a("H2 — THE RANK CORRELATION OVER ALL FOURTEEN DENSITIES")
    xs14 = [r for r in GRID if by_rho[r]]
    y14 = [med([c["ratio"] for c in by_rho[r]]) for r in xs14]
    sp14 = spearman(y14, xs14)
    xs12 = [r for r in xs14 if r > 0.35]
    y12 = [med([c["ratio"] for c in by_rho[r]]) for r in xs12]
    sp12 = spearman(y12, xs12)
    a(f"  Spearman(median ratio, rho) over {len(xs14)} densities = {sp14:+.4f}")
    a(f"  over the {len(xs12)} uncensored densities (rho >= 0.40) = {sp12:+.4f}")
    a(f"  GATE published +0.988 over the ten fitted densities")
    h2 = sp14 >= 0.90
    a(f"  registered: >= +0.90, falsified below +0.60  ->  "
      f"{'HIT' if h2 else ('MISS' if sp14 >= 0.60 else 'FALSIFIED')}")

    # --------------------------------------------------------------- H3a
    a("")
    a("-" * 78)
    a("H3a — A FIXED POPULATION: IS THE TREND A CENSORING ARTEFACT")
    ok_by_run = collections.defaultdict(set)
    for c in cells:
        if c["ratio"] == c["ratio"]:
            ok_by_run[c["run_id"]].add(c["rho"])
    fit_set, all_set = set(FIT_RHOS), set(GRID)
    c10 = {k for k, v in ok_by_run.items() if fit_set <= v}
    c14 = {k for k, v in ok_by_run.items() if all_set <= v}
    a(f"  C10 = runs measurable at all ten fitted densities: {len(c10)} of "
      f"{len(runs)}")
    a(f"  C14 = runs measurable at all fourteen:              {len(c14)} of "
      f"{len(runs)}")
    a("     rho   all cells   fixed pop   difference")
    worst_gap = 0.0
    for r in FIT_RHOS:
        ma = med([c["ratio"] for c in by_rho[r]])
        mf = med([c["ratio"] for c in by_rho[r] if c["run_id"] in c10])
        worst_gap = max(worst_gap, abs(ma - mf))
        a(f"    {r:.2f}   {ma:9.3f}   {mf:9.3f}   {ma - mf:+10.3f}")
    yf = [med([c["ratio"] for c in by_rho[r] if c["run_id"] in c10])
          for r in FIT_RHOS]
    sp_c10 = spearman(yf, list(FIT_RHOS))
    h3a = sp_c10 >= 0.90 and worst_gap <= 0.50
    a(f"  Spearman over the fixed population = {sp_c10:+.4f}   "
      f"largest gap to all-cells {worst_gap:.3f}")
    a(f"  registered: Spearman >= +0.90 AND every gap <= 0.50  ->  "
      f"{'HIT' if h3a else ('MISS' if sp_c10 >= 0.60 else 'FALSIFIED')}")
    if len(c14) < 20:
        a(f"  C14 has {len(c14)} runs: the fourteen-density version is "
          f"UNDERPOWERED as registered and decides nothing.")
    else:
        y14f = [med([c["ratio"] for c in by_rho[r] if c["run_id"] in c14])
                for r in GRID if by_rho[r]]
        a(f"  C14 version, {len(c14)} runs: Spearman "
          f"{spearman(y14f, [r for r in GRID if by_rho[r]]):+.4f}")

    # --------------------------------------------------------------- H3b
    a("")
    a("-" * 78)
    a("H3b — PAIRED WITHIN RUN, IMMUNE TO ANY CHANGE OF POPULATION")
    hi_r, lo_r = {}, {}
    for c in by_rho[0.95]:
        hi_r[c["run_id"]] = c["ratio"]
    for c in by_rho[0.50]:
        lo_r[c["run_id"]] = c["ratio"]
    pairs = [(k, hi_r[k], lo_r[k]) for k in sorted(hi_r) if k in lo_r]
    diffs = [b - a_ for _, a_, b in pairs]
    neg = sum(1 for d in diffs if d < 0)
    frac = neg / len(diffs) if diffs else float("nan")
    h3b = frac >= 0.70 and med(diffs) < -0.30
    a(f"  runs measurable at both rho 0.95 and rho 0.50: {len(pairs)}")
    a(f"  ratio falls on {neg} of them = {100.0 * frac:.1f}%")
    a(f"  median paired change {med(diffs):+.4f}   "
      f"quartiles {np.percentile(diffs, 25):+.3f} / "
      f"{np.percentile(diffs, 75):+.3f}")
    a(f"  registered: >= 70% negative AND median < -0.30  ->  "
      f"{'HIT' if h3b else ('MISS' if frac >= 0.55 else 'FALSIFIED')}")

    # ---------------------------------------------------------------- H4
    a("")
    a("-" * 78)
    a("H4 — WHICH HAZARD MOVES, SINCE ratio = log(1-h_DD)/log(1-h) EXACTLY")
    a("     rho   median h   median h_DD   h / h(0.95)   h_DD / h_DD(0.95)")
    h0 = med([c["h"] for c in by_rho[0.95]])
    hd0 = med([c["h_ddf"] for c in by_rho[0.95]])
    for r in FIT_RHOS:
        mh = med([c["h"] for c in by_rho[r]])
        mhd = med([c["h_ddf"] for c in by_rho[r]])
        a(f"    {r:.2f}   {mh:8.5f}   {mhd:11.5f}   {mh / h0:11.3f}   "
          f"{mhd / hd0:17.3f}")
    hN = med([c["h"] for c in by_rho[0.50]])
    hdN = med([c["h_ddf"] for c in by_rho[0.50]])
    gh, gd = hN / h0, hdN / hd0
    h4 = hdN > hd0 and gh > gd
    a(f"  colour hazard grows {gh:.3f}x, faithfulness hazard grows {gd:.3f}x")
    a(f"  registered: h_DD rises AND h rises by the larger factor  ->  "
      f"{'HIT' if h4 else 'MISS'}")
    a("  reading: the ratio falls because premature naming becomes relatively")
    a("  less dominant, not because the model grows more DD-faithful." if h4
      else "  reading: NOT the mechanism registered — see the table above.")

    # ------------------------------------------------------------ figures
    print("figures:")
    h1_extrapolation(by_rho, slope, icept)
    h3a_population(by_rho, c10)
    h3b_paired(diffs)
    h4_hazards(by_rho)
    d2_censoring(by_rho_all)

    # ----------------------------------------------------------- scorecard
    a("")
    a("=" * 78)
    a("SCORECARD")
    for lab, v in [("H1 trend continues below the grid", h1),
                   ("H2 Spearman over all 14 densities", h2),
                   ("H3a fixed population", h3a),
                   ("H3b paired within run", h3b),
                   ("H4 which hazard moves", h4),
                   ("D1 no loader drift", d1),
                   ("D3 monotonicity", not bad)]:
        a(f"  {lab:38s} {'HIT' if v else 'MISS'}")

    with open(os.path.join(OUT, "holdout_density.csv"), "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(dens_rows[0].keys()))
        w.writeheader()
        for r in dens_rows:
            w.writerow(r)
    with open(os.path.join(OUT, "holdout_paired.csv"), "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["run_id", "ratio_rho095", "ratio_rho050", "diff"])
        for k, a_, b in pairs:
            w.writerow([k, f"{a_:.6f}", f"{b:.6f}", f"{b - a_:.6f}"])
    with open(os.path.join(OUT, "VERDICT.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("  wrote holdout_density.csv, holdout_paired.csv, VERDICT.txt")


if __name__ == "__main__":
    main()
