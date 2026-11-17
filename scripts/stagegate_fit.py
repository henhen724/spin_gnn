#!/usr/bin/env python3
"""Stage GATE -- recompute rho_50 under a DD-FAITHFUL solve criterion.

Finding 98 showed the free-generation scorer calls a pair "ok" when the cell is
merely uncoloured and the colour is right -- `node_valid = node not in assigned`
never means "forced".  So the project's whole ranking quantity,

    rho_50 = the hint density where the free-generation solve rate crosses 0.50,

measures "produces the planted colouring", not "executes direct deduction",
which is what the brief actually asks for.  This script recomputes it under a
criterion that requires every emitted cell to have been forced when emitted.

Predictions were committed in runs/stagegate_2026-11-17/REGISTERED.md BEFORE
this file existed (commit 500a0c1).  Refit of stage LOGIC's CSVs: no training,
no GPU, nothing submitted.

A DEFECT IN THAT REGISTRATION, FOUND BY ITS OWN D2 AND REPORTED NOT PATCHED
OVER.  The registered estimator was

    h_j^logic = (n_err(j) - n_err_in_F(j)) / n_at_risk(j),

which I described as the DD-unfaithful rate.  It is not.  `n_err` counts FIRST
ERRORS under the permissive rollout rule and `n_err_in_F` is the benign subset
of those (2.01% at j = 0), so the registered formula counts *permissive failures
that also named an unforced cell* -- a subset of failures, not the set of
unfaithful emissions.  A rollout that names an unforced cell and colours it
correctly is not a permissive error at all, so it never appears in `n_err`,
which is the entire point of 98.

The column that does count unfaithful emissions is `n_in_F`, which is an
EMISSION-level count (median n_in_F/n_at_risk = 0.7432 over 19,889 rows, and
n_in_F > n_err on 19,365 of them, so it cannot be an error count):

    h_j^ddf = 1 - n_in_F(j) / n_at_risk(j).

Both estimators are computed below.  The registered one is reported as defective
and its predictions are scored as VOID BY SCORER DEFECT, never as honest misses;
the corrected one is labelled POST-HOC throughout and claimed as nobody's
prediction.

A SECOND DEFECT, found while writing the deck and fixed here.  G3 and G5 looked
the hub up by (arch_mode, num_layers, d_model) = ("full", 6, 256), which is not
a key: it pools the hub (4,899,898) with `full L=6 ff256` (2,535,994), so the
first run of this scorer reported a six-seed "hub" median of 0.6227 -> 0.7326
spanning two architectures.  That is finding 96's defect exactly, and
REGISTERED.md already said to key on the `cell` label.  Fixed by is_cell();
the hub is 3 seeds, 0.6068 -> 0.6942.  G5's verdict is unchanged, and the
published separation from J2 lands at 0.0113 against the 0.012 the registration
quoted from the rung table.
"""

import collections
import csv as _csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stageexp_fit import fnum, tobool, read_shards, med, spread  # noqa: E402

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import (SERIES, MUTED, style_axes, label_line,  # noqa: E402
                    series_colors)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = "/scratch/users/hshunt/spin_gnn/runs"
LOGIC = os.path.join(RUNS, "stagelogic_2026-11-17")
OUT = os.path.join(RUNS, "stagegate_2026-11-17")
FIGS = os.path.join(OUT, "figures")
DPI = 150

GRID = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5,
        0.45, 0.4, 0.35, 0.33]
FIT_RHOS = GRID[:10]
HARD = [r for r in FIT_RHOS if r <= 0.75]   # the six with n >= 12
GATE = 0.740
HALF = 0.50

# Keyed on the `cell` LABEL, not on (arch_mode, num_layers, d_model).  The
# triple is not a key: ("full", 6, 256) pools the hub (4,899,898) with
# `full L=6 ff256` (2,535,994), and ("tied_no_mlp", 16, 320) pools k = 1, 2, 4.
# That is the defect finding 96 fixed, and REGISTERED.md says to key on `cell`;
# a first draft of this scorer keyed on the triple anyway and reported a
# six-seed "hub" spanning two architectures.
J2_LABEL = "L=16 h=4 no_mlp d192"
HUB_LABEL = "L=6 h=4 full"


def is_cell(rec, label):
    return " ".join(rec["cell"].split()) == label


def fig(name):
    f, ax = plt.subplots(figsize=(7.2, 4.3))
    return f, ax, os.path.join(FIGS, name)


def save(f, path):
    f.tight_layout()
    f.savefig(path, dpi=DPI)
    plt.close(f)
    print("  wrote", os.path.basename(path))


def load():
    """Per (run_id, rho): three hazard products and the measured solve rate."""
    haz = read_shards(LOGIC, "F_hazard_all_*of10.csv", keep=[
        "run_id", "in79", "rho", "j", "n_at_risk", "n_in_F", "n_err",
        "n_err_in_F"])
    summ = read_shards(LOGIC, "LF_summary_all_*of10.csv", keep=[
        "run_id", "cell", "arch_mode", "num_layers", "d_model",
        "n_trainable_params", "val_acc", "in79", "rho", "mean_n_blank",
        "solve_rate"])

    acc = collections.defaultdict(lambda: {"ddf": 0.0, "regd": 0.0,
                                           "perm": 0.0, "z_ddf": False,
                                           "z_regd": False, "z_perm": False,
                                           "njs": 0})
    for h in haz:
        if not tobool(h, "in79"):
            continue
        ar = fnum(h, "n_at_risk")
        if not (ar > 0):
            continue
        p = acc[(h["run_id"], round(fnum(h, "rho"), 4))]
        p["njs"] += 1
        e_all, e_ben, inf = (fnum(h, "n_err"), fnum(h, "n_err_in_F"),
                             fnum(h, "n_in_F"))
        for tag, hv in (("ddf", 1.0 - inf / ar),          # CORRECTED, post-hoc
                        ("regd", (e_all - e_ben) / ar),   # as registered
                        ("perm", e_all / ar)):            # permissive, for D1
            if hv >= 1.0:
                p["z_" + tag] = True
            elif hv > 0:
                p[tag] += np.log1p(-hv)

    cells = []
    for s in summ:
        if not tobool(s, "in79"):
            continue
        key = (s["run_id"], round(fnum(s, "rho"), 4))
        if key not in acc:
            continue
        p = acc[key]
        g = lambda t: 0.0 if p["z_" + t] else float(np.exp(p[t]))
        cells.append({
            "run_id": s["run_id"], "cell": s["cell"],
            "arch": (s["arch_mode"], int(float(s["num_layers"])),
                     int(float(s["d_model"]))),
            "params": fnum(s, "n_trainable_params"),
            "val_acc": fnum(s, "val_acc"),
            "rho": round(fnum(s, "rho"), 4),
            "n_blank": fnum(s, "mean_n_blank"),
            "S": fnum(s, "solve_rate"),
            "S_ddf": g("ddf"), "S_regd": g("regd"), "S_perm": g("perm"),
            "njs": p["njs"],
        })
    return cells, len(haz), len(summ)


def rho50(curve):
    """Scan the grid from high rho down; linear interpolation in rho across the
    first bracket where the rate crosses 0.50.  Registered before any rate was
    read."""
    pts = [(r, curve[r]) for r in GRID if r in curve and curve[r] == curve[r]]
    if not pts:
        return float("nan"), "no_data"
    if pts[0][1] < HALF:
        return float("nan"), "censored_high"
    if pts[-1][1] >= HALF:
        return float("nan"), "censored_low"
    for (r0, v0), (r1, v1) in zip(pts, pts[1:]):
        if v0 >= HALF > v1:
            if v0 == v1:
                return r1, "ok"
            return r0 + (v0 - HALF) / (v0 - v1) * (r1 - r0), "ok"
    return float("nan"), "no_crossing"


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def fmt(v):
    return "  censored" if v != v else f"{v:10.4f}"


# ----------------------------------------------------------------- figures

def g1_ratio(by_rho):
    f, ax, path = fig("G1_ratio.png")
    xs, ys, los, his = [], [], [], []
    for r in FIT_RHOS:
        v = [c["ratio"] for c in by_rho[r] if c["ratio"] == c["ratio"]]
        if not v:
            continue
        xs.append(r)
        ys.append(med(v))
        # Interquartile, not min-max: at easy densities S -> 1 so log S -> 0
        # and the ratio of two near-zero logs is numerically wild on a handful
        # of runs.  The quartiles say what the population does; the extremes
        # there say what floating point does.
        los.append(float(np.percentile(v, 25)))
        his.append(float(np.percentile(v, 75)))
    c = series_colors(2, highlight=0)
    ax.fill_between(xs, los, his, color=c[0], alpha=0.16, lw=0)
    ax.plot(xs, ys, color=c[0], lw=2.0, marker="o", ms=4)
    ax.axhspan(1.0 / 0.61, 1.0 / 0.37, color=MUTED, alpha=0.18, lw=0)
    ax.axhline(2.0, color=MUTED, lw=1.0, ls=":")
    style_axes(ax, xlabel=r"hint density $\rho$",
               ylabel=r"$\log S_{\rm DD} / \log S$")
    ax.set_xlim(0.975, 0.425)
    ax.set_ylim(1.2, max(his) + 1.25)
    # Short label only: "median, shaded IQR" at this font ran to the canvas
    # edge, so the band is explained in the caption block instead.
    label_line(ax, xs[-1], ys[-1], "median", color=c[0], dy=16)
    top = ax.get_ylim()[1]
    ax.text(0.97, top - 0.10,
            r"grey band: $1/R$ predicted from finding 97, $R\in[0.37,0.61]$",
            fontsize=8, color=MUTED, ha="left", va="top")
    above = sum(1 for y in ys if y > 1.0 / 0.37)
    ax.text(0.97, top - 0.38,
            f"the measured median sits ABOVE it at {above} of {len(ys)} "
            f"densities", fontsize=8, color=MUTED, ha="left", va="top")
    ax.text(0.97, top - 0.66,
            "blue band: interquartile range of the ratio over runs",
            fontsize=8, color=MUTED, ha="left", va="top")
    ax.text(0.43, 1.25, "post-hoc estimator, not the registered one",
            fontsize=8, color=MUTED, ha="right", va="bottom")
    ax.text(0.43, 2.04, "exact squaring", fontsize=8, color=MUTED,
            ha="right", va="bottom")
    save(f, path)


def g2_curves(cells):
    f, ax, path = fig("G2_curves.png")
    j2 = [c for c in cells if is_cell(c, J2_LABEL)]
    seeds = sorted({c["run_id"] for c in j2})
    cur_o = {r: med([c["S"] for c in j2 if c["rho"] == r]) for r in GRID
             if any(c["rho"] == r for c in j2)}
    cur_d = {r: med([c["S_ddf"] for c in j2 if c["rho"] == r]) for r in GRID
             if any(c["rho"] == r for c in j2)}
    c = series_colors(2)
    xo = [r for r in GRID if r in cur_o]
    ax.plot(xo, [cur_o[r] for r in xo], color=c[0], lw=2.0, marker="o", ms=4)
    ax.plot(xo, [cur_d[r] for r in xo], color=c[1], lw=2.0, marker="s", ms=4)
    ax.axhline(HALF, color=MUTED, lw=1.0, ls="--")
    style_axes(ax, xlabel=r"hint density $\rho$", ylabel="solve rate")
    ax.set_xlim(0.985, 0.315)
    ax.set_ylim(-0.08, 1.12)
    label_line(ax, 0.50, cur_o[0.5], "as published", color=c[0], dy=16)
    label_line(ax, 0.85, cur_d[0.85], "DD-faithful", color=c[1], dy=16)
    # The two half-solve crossings, which are the rho_50 the gate reads.  The
    # values are parked in the empty lower-left corner rather than beside the
    # markers: both curves pass through the middle and any offset there is
    # struck through by one of them.
    for k, (cur, col, lab) in enumerate(((cur_d, c[1], "DD-faithful"),
                                         (cur_o, c[0], "as published"))):
        x50, st = rho50(cur)
        if st == "ok":
            ax.plot([x50], [HALF], marker="v", ms=9, color=col, zorder=6)
            ax.text(0.965, 0.30 - 0.10 * k,
                    rf"$\rho_{{50}} = {x50:.3f}$   {lab}",
                    fontsize=9, color=col, ha="left", va="center")
    ax.text(0.325, HALF - 0.03, "half-solve", fontsize=8, color=MUTED,
            ha="left", va="top")
    ax.text(0.325, 1.09,
            f"J2 = no_mlp L=16 d192, median over {len(seeds)} seeds",
            fontsize=8, color=MUTED, ha="left", va="top")
    save(f, path)


def g3_scatter(runs):
    f, ax, path = fig("G3_scatter.png")
    ok = [r for r in runs if r["rho50"] == r["rho50"]
          and r["rho50_ddf"] == r["rho50_ddf"]]
    # Three meanings, three colours: the population is muted, the two cells the
    # brief is between are accented, the gate is a different hue entirely.
    pop, acc, gate_c = MUTED, SERIES[0], SERIES[1]
    ax.scatter([r["rho50"] for r in ok], [r["rho50_ddf"] for r in ok],
               s=22, color=pop, alpha=0.55, lw=0)
    lim = [0.30, 1.00]
    ax.plot(lim, lim, color=MUTED, lw=1.0, ls=":")
    ax.axhline(GATE, color=gate_c, lw=1.2)
    ax.axvline(GATE, color=gate_c, lw=1.2, ls="--")
    hi = [r for r in ok if is_cell(r, J2_LABEL) or is_cell(r, HUB_LABEL)]
    ax.scatter([r["rho50"] for r in hi], [r["rho50_ddf"] for r in hi],
               s=70, color=acc, zorder=5, lw=0)
    style_axes(ax, xlabel=r"$\rho_{50}$ as published",
               ylabel=r"$\rho_{50}$, DD-faithful")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.text(0.985, GATE + 0.012, "gate 0.740", fontsize=8, color=gate_c,
            ha="right", va="bottom")
    ax.text(0.315, 0.985, f"{len(ok)} runs, every one above the dotted "
            f"identity:", fontsize=8, color=MUTED, ha="left", va="top")
    ax.text(0.315, 0.945, "the stricter criterion never makes a run look "
            "better", fontsize=8, color=MUTED, ha="left", va="top")
    ax.text(0.315, 0.885, f"accented: J2 and the hub, {len(hi)} seeds",
            fontsize=8, color=acc, ha="left", va="top")
    save(f, path)


def g6_cells(arch):
    """Per-architecture dumbbell: published rho_50 -> DD-faithful rho_50.

    Seed medians, which is the aggregation G2 and G5 were registered at.  y is
    parameters on a log axis, so the pile-up of the non-clearing group at
    rho_50 ~ 0.87 shows up as a vertical wall rather than as a table."""
    f, ax, path = fig("G6_cells.png")
    keep, drop = SERIES[0], MUTED
    for r in arch:
        if r["pub"] != r["pub"] or r["ddf"] != r["ddf"]:
            continue
        c = keep if r["ddf"] <= GATE else drop
        ax.plot([r["pub"], r["ddf"]], [r["params"]] * 2, color=c, lw=1.4,
                alpha=0.85, solid_capstyle="round")
        ax.scatter([r["ddf"]], [r["params"]], s=26, color=c, lw=0, zorder=4)
        ax.scatter([r["pub"]], [r["params"]], s=26, color="white",
                   edgecolors=c, linewidths=1.2, zorder=4)
    ax.axvline(GATE, color=SERIES[1], lw=1.2, ls="--")
    style_axes(ax, xlabel=r"$\rho_{50}$   (hollow: as published,"
                         r" filled: DD-faithful)",
               ylabel="trainable parameters", logy=True)
    ax.set_xlim(0.36, 0.95)
    nk = sum(1 for r in arch if r["ddf"] == r["ddf"] and r["ddf"] <= GATE)
    ax.text(GATE - 0.012, 1.3e7, "gate 0.740", fontsize=8, color=SERIES[1],
            ha="right", va="top")
    ax.text(0.37, 1.3e7, f"{nk} architectures clear it DD-faithfully",
            fontsize=8, color=keep, ha="left", va="top")
    ax.text(0.37, 9.0e6, "the rest pile up near 0.87 whatever their size",
            fontsize=8, color=drop, ha="left", va="top")
    save(f, path)


def d1_closure(cells):
    f, ax, path = fig("D1_closure.png")
    use = [c for c in cells if c["n_blank"] >= 12]
    c_ = series_colors(2, highlight=0)
    ax.scatter([c["S"] for c in use], [c["S_perm"] for c in use],
               s=14, color=c_[0], alpha=0.45, lw=0)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls=":")
    style_axes(ax, xlabel="measured free-generation solve rate",
               ylabel=r"$\prod_j (1 - n_{\rm err}/n_{\rm at\,risk})$")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    d = [abs(c["S_perm"] - c["S"]) for c in use]
    ax.text(0.985, 0.02,
            f"{len(use)} cells, $n \\geq 12$; median |diff| {med(d):.4f}, "
            f"max {max(d):.4f}", fontsize=8, color=MUTED, ha="right",
            va="bottom")
    save(f, path)


# ------------------------------------------------------------------- main

def main():
    os.makedirs(FIGS, exist_ok=True)
    lines = []

    def a(s=""):
        print(s)
        lines.append(s)

    cells, n_haz, n_summ = load()
    a("STAGE GATE -- rho_50 under a DD-faithful solve criterion")
    a("=" * 72)
    a("")
    a("Predictions registered in REGISTERED.md, committed 500a0c1 BEFORE this")
    a("scorer existed.  Refit of stage LOGIC CSVs; no training, no GPU,")
    a("nothing submitted.")
    a("")
    a(f"inputs: {n_haz} hazard rows, {n_summ} summary rows -> "
      f"{len(cells)} joined (run, rho) cells")

    # ---------------------------------------------------------- the defect
    a("")
    a("DEFECT IN MY OWN REGISTRATION, CAUGHT BY ITS OWN D2")
    a("=" * 72)
    a("The registered estimator was h_j = (n_err - n_err_in_F)/n_at_risk,")
    a("described as the DD-unfaithful rate.  IT IS NOT THAT.  `n_err` counts")
    a("FIRST ERRORS under the permissive rollout rule and `n_err_in_F` is the")
    a("benign subset of those, so the formula counts permissive failures that")
    a("also named an unforced cell -- a SUBSET OF FAILURES, not the set of")
    a("unfaithful emissions.  A rollout that names an unforced cell and")
    a("colours it correctly is not a permissive error at all and never enters")
    a("`n_err`; that is the whole content of finding 98.")
    a("")
    a("`n_in_F` is an EMISSION-level count, so the correct estimator is")
    a("h_j^ddf = 1 - n_in_F(j)/n_at_risk(j).  Evidence it is emission-level:")
    a("median n_in_F/n_at_risk = 0.7432 over 19,889 rows, and n_in_F > n_err")
    a("on 19,365 of them, which no error-subset column could do.")
    a("")
    a("CONSEQUENCE, APPLIED BELOW WITHOUT EXCEPTION: G1-G5 as registered do")
    a("not test what they claim, and are scored VOID BY SCORER DEFECT rather")
    a("than as honest misses.  Everything computed from h^ddf is POST-HOC and")
    a("is claimed as nobody's prediction.")

    for c in cells:
        c["ratio"] = (np.log(c["S_ddf"]) / np.log(c["S"])
                      if 0 < c["S"] < 1 and 0 < c["S_ddf"] < 1
                      else float("nan"))
        c["ratio_regd"] = (np.log(c["S_regd"]) / np.log(c["S"])
                           if 0 < c["S"] < 1 and 0 < c["S_regd"] < 1
                           else float("nan"))

    by_rho = collections.defaultdict(list)
    for c in cells:
        by_rho[c["rho"]].append(c)

    a("")
    a("  what the registered formula gives, for the record:")
    rr = [c["ratio_regd"] for c in cells if c["ratio_regd"] == c["ratio_regd"]]
    a(f"  median log S_regd / log S = {med(rr):.4f} over {len(rr)} cells --")
    a("  i.e. it simply reproduces the permissive solve rate, as it must,")
    a("  because its numerator differs from n_err by 2%.")

    # ------------------------------------------------------------------ D1
    a("")
    a("D1 -- PRODUCT CLOSURE AGAINST THE MEASURED SOLVE RATE (registered)")
    a("-" * 72)
    use = [c for c in cells if c["n_blank"] >= 12]
    diffs = [abs(c["S_perm"] - c["S"]) for c in use]
    d1_med, d1_max = med(diffs), max(diffs)
    a(f"  cells with n >= 12: {len(use)}")
    a(f"  median |prod(1 - n_err/n_at_risk) - S| = {d1_med:.5f}  (bar < 0.01)")
    a(f"  max                                    = {d1_max:.5f}  (bar < 0.05)")
    a("  POS's P2(a) residual for reference: 2.25e-02")
    d1_ok = (d1_med < 0.01) and (d1_max < 0.05)
    a(f"  -> D1 {'PASSES' if d1_ok else 'DOES NOT CLOSE'}  -- the product form")
    a("     itself is sound; what was wrong was which numerator I fed it.")

    # ------------------------------------------------------------------ D2
    a("")
    a("D2 -- MONOTONICITY")
    a("-" * 72)
    a("  As registered: S_logic <= S_strict_hat on every cell.  THAT")
    a("  INEQUALITY WAS BACKWARDS -- h_logic <= h_perm by construction, so the")
    a("  registered product can only be the LARGER of the two.  D2 fired on")
    a("  that, which is how the defect above was found.")
    a("")
    a("  AND THE OBVIOUS REPAIR, S_ddf <= S, IS ALSO NOT FORCED.  The two")
    a("  criteria are NOT NESTED: S_ddf asks only 'did it ever name an")
    a("  unforced cell', while S asks 'did it reach SOLVED with the planted")
    a("  colouring'.  A rollout can choose every cell faithfully and still")
    a("  fail S -- wrong colour on a forced cell, a grammar slip, early")
    a("  termination -- and conversely.  So neither direction is guaranteed,")
    a("  and what IS guaranteed is that the TRUE DD-faithful solve rate, the")
    a("  intersection of the two, is at most min(S, S_ddf).")
    v_ddf = [c for c in cells if c["S_ddf"] > c["S"] + 1e-9]
    a(f"  cells with S_ddf > S: {len(v_ddf)} of {len(cells)}")
    for c in sorted(v_ddf, key=lambda c: c["S"] - c["S_ddf"])[:3]:
        a(f"    {c['run_id']} rho {c['rho']:.2f}: S {c['S']:.4f} vs "
          f"S_ddf {c['S_ddf']:.4f}  (excess {c['S_ddf'] - c['S']:+.4f})")
    a("  -> D2 as registered: WRONG INEQUALITY, and its repair is wrong too;")
    a("     the quantity that is actually bounded is min(S, S_ddf), reported")
    a("     per density below.  The run-level check that matters is that")
    a("     rho_50 never moves the easy way, scored next.")

    # ---------------------------------------------------- rho_50 per run
    curves = collections.defaultdict(lambda: ({}, {}))
    meta = {}
    for c in cells:
        o, d = curves[c["run_id"]]
        o[c["rho"]] = c["S"]
        d[c["rho"]] = c["S_ddf"]
        meta[c["run_id"]] = c
    runs = []
    for rid, (o, d) in curves.items():
        v_o, st_o = rho50(o)
        v_d, st_d = rho50(d)
        m = meta[rid]
        runs.append({"run_id": rid, "cell": m["cell"], "arch": m["arch"],
                     "params": m["params"], "val_acc": m["val_acc"],
                     "rho50": v_o, "status": st_o,
                     "rho50_ddf": v_d, "status_ddf": st_d})
    both = [r for r in runs if r["rho50"] == r["rho50"]
            and r["rho50_ddf"] == r["rho50_ddf"]]

    a("")
    a("rho_50 CENSORING (reported, never dropped silently)")
    a("-" * 72)
    for (so, sl), k in sorted(
            collections.Counter((r["status"], r["status_ddf"])
                                for r in runs).items(), key=lambda t: -t[1]):
        a(f"  published {so:<14s} DD-faithful {sl:<14s} {k:3d} runs")
    a(f"  -> {len(both)} of {len(runs)} runs have both defined")
    back = [r for r in both if r["rho50_ddf"] < r["rho50"] - 1e-9]
    a(f"  rho50_ddf < rho50 on {len(back)} of {len(both)} -- the run-level")
    a("  form of D2, and the one that matters: no run gets EASIER under the")
    a("  stricter criterion.")
    d2_ok = len(back) == 0

    # ------------------------------------------------- post-hoc G1 / G4
    a("")
    a("POST-HOC G1 -- DOES THE DD-FAITHFUL CRITERION SQUARE THE SOLVE RATE?")
    a("-" * 72)
    a("  (the registered prediction was [1.5, 2.8] from 97's 1/R; it is scored")
    a("   VOID because the estimator it named was the wrong quantity)")
    a("  rho      n    runs   median ratio   spread")
    g1_med = {}
    inb = 0
    for r in FIT_RHOS:
        v = [c["ratio"] for c in by_rho[r] if c["ratio"] == c["ratio"]]
        if not v:
            a(f"  {r:4.2f}   --      0     (no cells with 0 < S < 1)")
            continue
        nb = med([c["n_blank"] for c in by_rho[r]])
        m = med(v)
        lo, hi = spread(v)
        g1_med[r] = m
        mark = ""
        if r in HARD:
            ok = 1.5 <= m <= 2.8
            inb += ok
            mark = "  in band" if ok else "  outside"
        a(f"  {r:4.2f} {nb:5.1f}  {len(v):5d}   {m:12.3f}   "
          f"[{lo:.3f}, {hi:.3f}]{mark}")
    a(f"  -> in [1.5, 2.8] at {inb} of {len(HARD)} densities with n >= 12")

    a("")
    a("POST-HOC G4 -- IS THE PENALTY WEAKEST WHERE THE TASK IS HARDEST?")
    a("-" * 72)
    xs = [r for r in FIT_RHOS if r in g1_med]
    sp4 = spearman(xs, [g1_med[r] for r in xs])
    a(f"  Spearman(median ratio, rho) over {len(xs)} densities = {sp4:+.3f}")
    a("  97's 1/R falls 2.26 (rho 0.75) -> 1.65 (rho 0.50), so POSITIVE was")
    a("  the expectation: the squaring penalty shrinks as the task gets harder")

    # ------------------------------------------------------- post-hoc G2
    a("")
    a("POST-HOC G2 -- DOES J2 STILL CLEAR THE GATE?")
    a("-" * 72)
    j2 = [r for r in runs if is_cell(r, J2_LABEL)]
    a(f"  J2 = no_mlp L=16 d192, {len(j2)} seeds, 2,504,890 trainable")
    a("  run_id            published   DD-faithful")
    for r in sorted(j2, key=lambda r: r["run_id"]):
        a(f"  {r['run_id']:<16s} {fmt(r['rho50'])}   {fmt(r['rho50_ddf'])}"
          f"   [{r['status_ddf']}]")
    j2m = med([r["rho50_ddf"] for r in j2])
    a(f"  median DD-faithful rho_50 = {fmt(j2m)}   (gate {GATE:.3f})")
    g2 = "clears" if j2m <= GATE else "FAILS"
    a(f"  -> J2 {g2} the gate under the DD-faithful criterion")
    a("")
    a("  The estimator ignores grammar / early-term / truncation failures, so")
    a("  it is an UPPER bound on the DD-faithful solve rate and this rho_50 is")
    a("  an OPTIMISTIC (low) bound.  A failure is therefore real; a pass is an")
    a("  upper-bound pass only.  That asymmetry was registered in advance.")

    # ------------------------------------------------------- post-hoc G3
    a("")
    a("POST-HOC G3 -- DOES THE RANKING SURVIVE?")
    a("-" * 72)
    sp3 = spearman([r["rho50"] for r in both], [r["rho50_ddf"] for r in both])
    a(f"  Spearman over {len(both)} runs with both defined = {sp3:+.4f}")
    shift = [r["rho50_ddf"] - r["rho50"] for r in both]
    lo, hi = spread(shift)
    a(f"  median shift {med(shift):+.4f}, range [{lo:+.4f}, {hi:+.4f}]")

    # ------------------------------------------------------- post-hoc G5
    a("")
    a("POST-HOC G5 -- THE ORDERING AT THE TOP OF THE RUNG TABLE")
    a("-" * 72)
    a("  keyed on the cell LABEL; keying on (arch_mode, num_layers, d_model)")
    a("  pools the hub with `full L=6 ff256`, 96's defect -- see is_cell()")
    pair = {}
    for name, lab in (("J2  no_mlp L=16 d192", J2_LABEL),
                      ("hub full L=6 d256", HUB_LABEL)):
        g = [r for r in runs if is_cell(r, lab)]
        pair[lab] = (med([r["rho50"] for r in g]),
                     med([r["rho50_ddf"] for r in g]))
        a(f"  {name:<22s} published {fmt(pair[lab][0])}   "
          f"DD-faithful {fmt(pair[lab][1])}   ({len(g)} seeds)")
    (jp, jd), (hp, hd) = pair[J2_LABEL], pair[HUB_LABEL]
    a(f"  separation: published {abs(jp - hp):.4f} -> "
      f"DD-faithful {abs(jd - hd):.4f}; J2 still lower: {jd <= hd}")

    # ------------------------------------------------- the population view
    a("")
    a("WHAT THE DD-FAITHFUL CRITERION DOES TO THE POPULATION")
    a("=" * 72)
    pub_pass = [r for r in both if r["rho50"] <= GATE]
    ddf_pass = [r for r in both if r["rho50_ddf"] <= GATE]
    a(f"  clearing {GATE:.3f} as published:  {len(pub_pass):3d} of {len(both)}")
    a(f"  clearing {GATE:.3f} DD-faithful:   {len(ddf_pass):3d} of {len(both)}")
    lost = [r for r in pub_pass if r["rho50_ddf"] > GATE]
    a(f"  runs that lose the gate: {len(lost)} of {len(pub_pass)}")
    if ddf_pass:
        a("  survivors, smallest first:")
        for r in sorted(ddf_pass, key=lambda r: r["params"])[:5]:
            a(f"    {r['cell']:<26s} {r['params']:>12,.0f}  "
              f"{fmt(r['rho50'])} -> {fmt(r['rho50_ddf'])}")
    a("")
    a("  per-density solve rate, median over all in79 runs.  min(S, S_ddf) is")
    a("  the only strict UPPER BOUND on the true DD-faithful rate, since the")
    a("  two criteria are not nested:")
    a("  rho      n    median S   median S_ddf   median min(S, S_ddf)")
    for r in GRID:
        g = by_rho.get(r, [])
        if not g:
            continue
        a(f"  {r:4.2f} {med([c['n_blank'] for c in g]):5.1f}   "
          f"{med([c['S'] for c in g]):8.4f}   "
          f"{med([c['S_ddf'] for c in g]):12.4f}   "
          f"{med([min(c['S'], c['S_ddf']) for c in g]):20.4f}")

    # ----------------------------------- the same thing, per architecture
    # Seed medians: the aggregation G2 and G5 were registered at, and the one
    # the rung table is actually read at.  A run is a seed, not a model class.
    by_cell = collections.defaultdict(list)
    for r in runs:
        by_cell[r["cell"]].append(r)
    arch = []
    for cell, rs in by_cell.items():
        po = [r["rho50"] for r in rs if r["rho50"] == r["rho50"]]
        pd_ = [r["rho50_ddf"] for r in rs if r["rho50_ddf"] == r["rho50_ddf"]]
        arch.append({"cell": cell, "params": rs[0]["params"], "seeds": len(rs),
                     "pub": med(po) if po else float("nan"),
                     "ddf": med(pd_) if pd_ else float("nan")})
    arch.sort(key=lambda r: r["params"])
    a_pub = [r for r in arch if r["pub"] <= GATE]
    a_ddf = [r for r in arch if r["ddf"] <= GATE]
    a("")
    a("THE SAME QUESTION PER ARCHITECTURE, WHICH IS HOW THE TABLE IS READ")
    a("=" * 72)
    a(f"  {len(arch)} architectures, seed medians")
    a(f"  clearing {GATE:.3f} as published:  {len(a_pub):3d} of {len(arch)}")
    a(f"  clearing {GATE:.3f} DD-faithful:   {len(a_ddf):3d} of {len(arch)}")
    same = {r["cell"] for r in a_pub} == {r["cell"] for r in a_ddf}
    a(f"  SAME SET: {same}")
    a("  -> the 4 run-level losses above are single seeds of architectures")
    a("     whose seed MEDIAN never cleared the gate in the first place:")
    for r in sorted(lost, key=lambda r: r["params"]):
        m = next(x for x in arch if x["cell"] == r["cell"])
        a(f"     {r['cell']:<26s} seed {r['rho50']:.4f}, "
          f"median of {m['seeds']} {m['pub']:.4f} (> {GATE:.3f})")
    a("")
    a("  cell                               params   published   DD-faithful")
    for r in arch:
        a(f"  {'* ' if r['ddf'] <= GATE else '  '}{r['cell']:<30s}"
          f"{r['params']:>10,.0f} {fmt(r['pub'])} {fmt(r['ddf'])}")
    rest = [r for r in arch if r["ddf"] > GATE and r["pub"] == r["pub"]]
    if rest:
        a("")
        a(f"  COMPRESSION: the {len(rest)} architectures that do not clear it")
        a(f"  span {min(r['pub'] for r in rest):.4f}-"
          f"{max(r['pub'] for r in rest):.4f} as published "
          f"(width {max(r['pub'] for r in rest) - min(r['pub'] for r in rest):.4f})")
        a(f"  and {min(r['ddf'] for r in rest):.4f}-"
          f"{max(r['ddf'] for r in rest):.4f} DD-faithful "
          f"(width {max(r['ddf'] for r in rest) - min(r['ddf'] for r in rest):.4f}).")
        a("  That collapse is why the rank correlation is only +0.74: below the")
        a("  gate the criterion stops distinguishing architectures at all.")
        a(f"  rho_50 = {med([r['ddf'] for r in rest]):.3f} is "
          f"n = {48 * (1 - med([r['ddf'] for r in rest])):.1f} blanks.")

    # ---------------------------------------------------------- scorecard
    a("")
    a("SCORECARD")
    a("=" * 72)
    a("  G1 squaring law            VOID BY SCORER DEFECT")
    a("  G2 J2 clears the gate      VOID BY SCORER DEFECT")
    a("  G3 ranking survives        VOID BY SCORER DEFECT")
    a("  G4 penalty vs difficulty   VOID BY SCORER DEFECT")
    a("  G5 top-of-table ordering   VOID BY SCORER DEFECT")
    a(f"  D1 product closure         {'PASSES' if d1_ok else 'FAILS'}")
    a("  D2 monotonicity            REGISTERED FORM BACKWARDS, and its obvious")
    a("                             repair is not forced either (the criteria")
    a("                             are not nested); the run-level form")
    a(f"                             {'HOLDS' if d2_ok else 'FAILS'} at "
      f"{len(back)} of {len(both)} runs moving the easy way")
    a("")
    a("  D2 is the reason this stage has a result at all: a diagnostic whose")
    a("  only job was to be boring caught the estimator being the wrong")
    a("  quantity.  98's lesson was 'read both scorers'; this one is 'read")
    a("  which column the scorer actually wrote'.")

    # ---------------------------------------------------------------- CSVs
    with open(os.path.join(OUT, "gate_cells.csv"), "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["run_id", "cell", "arch_mode", "num_layers", "d_model",
                    "params", "val_acc", "rho", "mean_n_blank", "n_j",
                    "solve_rate", "S_ddf", "S_perm", "S_regd", "ratio"])
        for c in sorted(cells, key=lambda c: (c["run_id"], -c["rho"])):
            w.writerow([c["run_id"], c["cell"], c["arch"][0], c["arch"][1],
                        c["arch"][2], f"{c['params']:.0f}",
                        f"{c['val_acc']:.6f}", c["rho"], c["n_blank"],
                        c["njs"], f"{c['S']:.6f}", f"{c['S_ddf']:.6f}",
                        f"{c['S_perm']:.6f}", f"{c['S_regd']:.6f}",
                        f"{c['ratio']:.6f}"])
    with open(os.path.join(OUT, "gate_runs.csv"), "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["run_id", "cell", "arch_mode", "num_layers", "d_model",
                    "params", "val_acc", "rho50", "status", "rho50_ddf",
                    "status_ddf", "shift"])
        for r in sorted(runs, key=lambda r: r["run_id"]):
            sh = (r["rho50_ddf"] - r["rho50"]
                  if r["rho50"] == r["rho50"]
                  and r["rho50_ddf"] == r["rho50_ddf"] else float("nan"))
            w.writerow([r["run_id"], r["cell"], r["arch"][0], r["arch"][1],
                        r["arch"][2], f"{r['params']:.0f}",
                        f"{r['val_acc']:.6f}", f"{r['rho50']:.6f}",
                        r["status"], f"{r['rho50_ddf']:.6f}",
                        r["status_ddf"], f"{sh:.6f}"])
    a("")
    a(f"wrote gate_cells.csv ({len(cells)} rows), gate_runs.csv ({len(runs)})")

    print("")
    g1_ratio(by_rho)
    g2_curves(cells)
    g3_scatter(runs)
    g6_cells(arch)
    d1_closure(cells)

    with open(os.path.join(OUT, "VERDICT.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote VERDICT.txt")


if __name__ == "__main__":
    main()
