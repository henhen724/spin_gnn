#!/usr/bin/env python3
"""Stage LOGIC: merge the shards, draw the ten registered figures, score L1-L5.

Reads `L_position_*.csv` / `F_hazard_*.csv` / `F_errors_*.csv` /
`LF_summary_*.csv` from LOGIC_OUT (skipping `*smoke*`), plus stage POS's
`B_summary_*.csv` from POS_OUT for the same-eval check, and writes figures to
LOGIC_OUT/figures/ and a plain-text verdict.

THE FIGURE LIST IS THE ONE REGISTERED IN runs/stagelogic_2026-11-17/
REGISTERED.md BEFORE THE JOB RAN. Nothing is added here that was not
registered there, and nothing registered there is quietly dropped: a figure
with no data prints "SKIPPED (no rows)" instead of disappearing.

Definitions, repeated so a reader of this file alone cannot get them wrong:

  F              nodes DD forces in the current round, from the hints plus all
                 COMPLETED earlier rounds, minus what this round already
                 emitted. Same-round siblings are not evidence.
  p_logic        teacher-forced argmax accuracy over the reasoning section
                 with a node scored correct iff it is in F
  p_strict       the same positions, argmax == the token the trace wrote
                 (stage POS's measurement A rule)
  premature      a node-position argmax naming an UNCOLOURED node outside F
                 -- a guard failure
  n              48 * (1 - realized_rho)           blank cells
  k              the exponent of S = C p^k, per density, fitted by OLS of
                 log S on log p over the 79 in79 runs with 0 < S < 1

Run with the figure module set:
  module purge
  module load devel math viz python/3.12.1 py-numpy/1.26.3_py312 \\
      py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  python3 scripts/stagelogic_fit.py
"""

import collections
import csv
import glob
import math
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line, series_colors  # noqa: E402

OUT = os.environ.get("LOGIC_OUT") or "/scratch/users/hshunt/spin_gnn/runs/stagelogic_2026-11-17"
POS = os.environ.get("POS_OUT") or "/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17"
INST = os.environ.get("LOGIC_INST") or os.path.join(POS, "instances")
FIGS = os.path.join(OUT, "figures")
DPI = 150
N_NODES = 48

PANEL_RHOS = [0.75, 0.60, 0.50]
OVERLAY_RHO = 0.60
FIT_RHOS = [r for r in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)]
MIN_FIT_POINTS = 8
L1_BAND = (0.85, 1.20)      # predicted slope of k_logic against n
L1_FALSIFY = (0.70, 1.40)
L1_PREFACTOR = (0.5, 2.0)   # predicted median C over rho >= 0.65
L3_BAND = (1.1, 1.6)        # predicted median k_logic / k_strict
REF_COEF = 0.73             # hshunt's k_strict ~ 0.73 n


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def read_shards(directory, pattern):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        if "smoke" in os.path.basename(path):
            continue
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def f(r, k):
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return float("nan")


def i(r, k):
    try:
        return int(float(r[k]))
    except (TypeError, ValueError, KeyError):
        return -1


def b(r, k):
    """Every CSV cell is a string and booleans arrive as 'True'."""
    return str(r.get(k, "")).strip() == "True"


def run_table(rows):
    runs = {}
    for r in rows:
        rid = r.get("run_id")
        if rid and rid not in runs:
            runs[rid] = {"run_id": rid, "cell": r.get("cell", ""),
                         "arch_mode": r.get("arch_mode", ""),
                         "num_layers": i(r, "num_layers"),
                         "d_model": i(r, "d_model"),
                         "val_acc": f(r, "val_acc"), "in79": b(r, "in79"),
                         "in_seven": b(r, "in_seven")}
    return runs


def blanks(realized_rho):
    return N_NODES * (1.0 - realized_rho)


def availability(inst_dir, rhos):
    """Mean share of nodes 'available to be premature' at each node position:
    (N - already coloured - |F|) / N, over the cached eval instances.

    NO MODEL ENTERS THIS. It is a property of the eval set alone, and it is
    exactly the L5 confound registered before the run: as rho falls there are
    simply more uncoloured nodes outside F to name, so a premature rate that
    rises with sparsity is partly 'more ways to be premature'.

    Read off the trace by bookkeeping rather than by recomputing DD, which is
    licensed by the forced-set identity `_stagelogic_dry_check` verifies: round
    r IS the set DD forces at the start of round r, so |F| at a node position
    is the rest of the current round. Needs numpy only -- no torch -- so the
    figure module set is enough.
    """
    out = {}
    for rho in rhos:
        # glob rather than rebuild the name: instance_cache hashes the seed as
        # `seed_start % 1000000`, which is 0 here, and a copy of that arithmetic
        # would be a silent way to read the wrong bin.
        hits = glob.glob(os.path.join(inst_dir, "inst_rho%05d_n*_s*.pkl"
                                      % round(rho * 10000)))
        if not hits:
            continue
        with open(sorted(hits)[0], "rb") as fh:
            inst = pickle.load(fh)
        per_j = collections.defaultdict(lambda: [0.0, 0])
        tot = [0.0, 0]
        for adj, _colors, hints, trace in inst:
            N = adj.shape[0]
            colored, j = len(hints), 0
            for ev in trace["rounds"]:
                remaining = len(ev)
                for _nd, _c in ev:
                    frac = (N - colored - remaining) / N
                    per_j[j][0] += frac
                    per_j[j][1] += 1
                    tot[0] += frac
                    tot[1] += 1
                    colored += 1
                    remaining -= 1
                    j += 1
        out[round(rho, 4)] = (tot[0] / tot[1] if tot[1] else float("nan"),
                              {j: v[0] / v[1] for j, v in per_j.items()})
    return out


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def powerlaw(p, s):
    """OLS of log s on log p. Returns k, se, C, R2, npts.

    S = C p^k, so the slope IS the exponent and exp(intercept) the prefactor.
    Only 0 < s < 1 and 0 < p < 1 enter: a run that never solves carries no
    information about the exponent and a saturated one carries none either.
    """
    p = np.asarray(p, float)
    s = np.asarray(s, float)
    m = (p > 0) & (p < 1) & (s > 0) & (s < 1) & np.isfinite(p) & np.isfinite(s)
    if m.sum() < 3 or len(set(np.round(p[m], 10))) < 3:
        return (float("nan"),) * 4 + (int(m.sum()),)
    lr = stats.linregress(np.log(p[m]), np.log(s[m]))
    return lr.slope, lr.stderr, math.exp(lr.intercept), lr.rvalue ** 2, int(m.sum())


def through_origin(x, y):
    """Slope a of y = a x and its standard error. hshunt's prediction is
    k ~ n, i.e. a ~ 1, so the intercept is not a free parameter here; the free
    fit is reported alongside it, never instead of it."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan"), float("nan"), 0
    a = float(x @ y / (x @ x))
    resid = y - a * x
    se = float(math.sqrt((resid @ resid) / max(len(x) - 1, 1) / (x @ x)))
    return a, se, len(x)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig(name, w=7.4, h=4.4):
    f_, ax = plt.subplots(figsize=(w, h))
    return f_, ax, os.path.join(FIGS, name)


def save(f_, path):
    f_.tight_layout()
    f_.savefig(path, dpi=DPI)
    plt.close(f_)
    print("  wrote %s" % os.path.basename(path))


def panels(name, n=3, w=11.0, h=3.9):
    f_, axes = plt.subplots(1, n, figsize=(w, h), sharey=True)
    return f_, axes, os.path.join(FIGS, name)


def l1_logic_vs_strict(cells):
    f_, axes, path = panels("L1_logic_vs_strict.png")
    for ax, rho in zip(axes, PANEL_RHOS):
        sel = [c for c in cells.values() if abs(c["rho"] - rho) < 1e-9 and c["in79"]]
        if not sel:
            ax.set_title("rho %.2f: no rows" % rho)
            continue
        x = [c["p_strict"] for c in sel]
        y = [c["p_logic"] for c in sel]
        lo, hi = min(x + y), max(x + y)
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, zorder=1)
        ax.scatter(x, y, s=14, color=SERIES[0], alpha=0.75, zorder=2,
                   rasterized=True)
        gain = np.median(np.array(y) - np.array(x))
        ax.set_title("rho %.2f   median gain %+.4f" % (rho, gain), fontsize=10)
        style_axes(ax, xlabel="strict token accuracy", ylabel=None)
    style_axes(axes[0], xlabel="strict token accuracy", ylabel="logical token accuracy")
    save(f_, path)


def l2_fit_panels(cells, fits):
    f_, axes, path = panels("L2_logS_vs_logp.png")
    for ax, rho in zip(axes, PANEL_RHOS):
        sel = [c for c in cells.values()
               if abs(c["rho"] - rho) < 1e-9 and c["in79"]
               and 0 < c["solve_rate"] < 1 and 0 < c["p_logic"] < 1]
        if not sel:
            ax.set_title("rho %.2f: no rows" % rho)
            continue
        x = np.array([c["p_logic"] for c in sel])
        y = np.array([c["solve_rate"] for c in sel])
        ax.scatter(x, y, s=14, color=SERIES[0], alpha=0.8, rasterized=True)
        fit = fits.get(round(rho, 4))
        if fit and fit["k_logic"] == fit["k_logic"]:
            xs = np.linspace(x.min(), x.max(), 60)
            ax.plot(xs, fit["C_logic"] * xs ** fit["k_logic"], color=SERIES[1], lw=1.6)
            ax.set_title("rho %.2f   k = %.1f, R2 %.2f, n = %.1f"
                         % (rho, fit["k_logic"], fit["r2_logic"], fit["n"]), fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        style_axes(ax, xlabel="logical token accuracy", ylabel=None)
    style_axes(axes[0], xlabel="logical token accuracy", ylabel="solve rate")
    save(f_, path)


def l3_k_vs_n(fits, a_logic, a_strict):
    f_, ax, path = fig("L3_exponent_vs_n.png")
    rows = [v for v in fits.values() if v["k_logic"] == v["k_logic"]]
    rows.sort(key=lambda v: v["n"])
    n = np.array([v["n"] for v in rows])
    kl = np.array([v["k_logic"] for v in rows])
    ks = np.array([v["k_strict"] for v in rows])
    el = np.array([v["se_logic"] for v in rows])
    es = np.array([v["se_strict"] for v in rows])
    ax.plot(n, n, color=MUTED, lw=1.2, ls="--")
    label_line(ax, n[-1], n[-1], "k = n", color=MUTED)
    ax.plot(n, REF_COEF * n, color=MUTED, lw=1.0, ls=":")
    label_line(ax, n[-1], REF_COEF * n[-1], "k = 0.73n", color=MUTED)
    ax.errorbar(n, kl, yerr=el, color=SERIES[0], lw=1.8, marker="o", ms=4, capsize=2)
    ax.errorbar(n, ks, yerr=es, color=SERIES[1], lw=1.4, marker="s", ms=3.5, capsize=2)
    label_line(ax, n[-1], kl[-1], "logical  (a = %.2f)" % a_logic, color=SERIES[0])
    label_line(ax, n[-1], ks[-1], "strict  (a = %.2f)" % a_strict, color=SERIES[1])
    style_axes(ax, xlabel="blank cells  n = 48(1 - realized rho)",
               ylabel="fitted exponent k")
    save(f_, path)


def l4_prefactor(fits):
    f_, ax, path = fig("L4_prefactor_vs_n.png")
    rows = sorted((v for v in fits.values() if v["C_logic"] == v["C_logic"]),
                  key=lambda v: v["n"])
    n = [v["n"] for v in rows]
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.plot(n, [v["C_logic"] for v in rows], color=SERIES[0], lw=1.8, marker="o", ms=4)
    ax.plot(n, [v["C_strict"] for v in rows], color=SERIES[1], lw=1.4, marker="s", ms=3.5)
    label_line(ax, n[-1], rows[-1]["C_logic"], "logical", color=SERIES[0])
    label_line(ax, n[-1], rows[-1]["C_strict"], "strict", color=SERIES[1])
    ax.set_yscale("log")
    style_axes(ax, xlabel="blank cells  n", ylabel="prefactor C of  S = C p^k")
    save(f_, path)


def l5_r2(fits):
    f_, ax, path = fig("L5_r2_vs_n.png")
    rows = sorted((v for v in fits.values() if v["r2_logic"] == v["r2_logic"]),
                  key=lambda v: v["n"])
    n = [v["n"] for v in rows]
    ax.plot(n, [v["r2_logic"] for v in rows], color=SERIES[0], lw=1.8, marker="o", ms=4)
    ax.plot(n, [v["r2_strict"] for v in rows], color=SERIES[1], lw=1.4, marker="s", ms=3.5)
    label_line(ax, n[-1], rows[-1]["r2_logic"], "logical", color=SERIES[0])
    label_line(ax, n[-1], rows[-1]["r2_strict"], "strict", color=SERIES[1])
    ax.set_ylim(0, 1.02)
    style_axes(ax, xlabel="blank cells  n", ylabel="R2 of the power-law fit")
    save(f_, path)


def l6_acc_vs_j(jrows, runs):
    f_, axes, path = panels("L6_node_acc_vs_j.png")
    for ax, rho in zip(axes, PANEL_RHOS):
        series = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))
        for r in jrows:
            if abs(f(r, "rho") - rho) > 1e-9 or r.get("tok_type") != "node":
                continue
            c = series[r["run_id"]][i(r, "index")]
            c[0] += i(r, "n_logic_correct")
            c[1] += i(r, "n_strict_correct")
            c[2] += i(r, "n_total")
        if not series:
            ax.set_title("rho %.2f: no rows" % rho)
            continue
        for rid, per_j in series.items():
            js = sorted(k for k in per_j if per_j[k][2] >= 100)
            if len(js) < 3:
                continue
            in79 = runs.get(rid, {}).get("in79", True)
            ax.plot(js, [per_j[j][0] / per_j[j][2] for j in js],
                    color=SERIES[0] if in79 else MUTED, lw=0.7, alpha=0.45)
            ax.plot(js, [per_j[j][1] / per_j[j][2] for j in js],
                    color=SERIES[1], lw=0.7, alpha=0.30)
        ax.set_title("rho %.2f" % rho, fontsize=10)
        style_axes(ax, xlabel="deduction index j", ylabel=None)
    style_axes(axes[0], xlabel="deduction index j",
               ylabel="node accuracy (blue logical, orange strict)")
    save(f_, path)


def l7_classes(jrows):
    f_, ax, path = fig("L7_node_classes_vs_j.png")
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in jrows:
        if abs(f(r, "rho") - OVERLAY_RHO) > 1e-9 or r.get("tok_type") != "node":
            continue
        a = agg[i(r, "index")]
        for k in ("n_in_F", "n_premature", "n_recolor", "n_nonnode", "n_total"):
            a[k] += i(r, k)
    js = sorted(j for j in agg if agg[j]["n_total"] >= 1000)
    if not js:
        print("  L7 SKIPPED (no rows)")
        plt.close(f_)
        return
    tot = np.array([agg[j]["n_total"] for j in js], float)
    stack = [np.array([agg[j][k] for j in js], float) / tot
             for k in ("n_in_F", "n_premature", "n_recolor", "n_nonnode")]
    cols = [SERIES[0], SERIES[1], SERIES[2], MUTED]
    names = ["in F (logical hit)", "premature (guard failure)",
             "recolor", "non-node"]
    ax.stackplot(js, *stack, colors=cols, labels=names)
    bot = np.zeros(len(js))
    for s, c, nm in zip(stack, cols, names):
        label_line(ax, js[-1], bot[-1] + s[-1] / 2, nm, color=c)
        bot = bot + s
    ax.set_ylim(0, 1)
    style_axes(ax, xlabel="deduction index j  (pooled over all 80 runs, rho %.2f)"
               % OVERLAY_RHO, ylabel="share of node-position argmaxes")
    save(f_, path)


def f1_unforced(cells, runs):
    f_, ax, path = fig("F1_unforced_fraction.png")
    by_run = collections.defaultdict(list)
    for c in cells.values():
        if c["n_err_node"] >= 50 and c["frac_unforced"] == c["frac_unforced"]:
            by_run[c["run_id"]].append((c["rho"], c["frac_unforced"]))
    if not by_run:
        print("  F1 SKIPPED (no rows)")
        plt.close(f_)
        return
    for rid, pts in by_run.items():
        pts.sort()
        in79 = runs.get(rid, {}).get("in79", True)
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=SERIES[0] if in79 else SERIES[1], lw=0.8, alpha=0.45)
    ax.axhline(0.5, color=MUTED, lw=1.2, ls="--")
    label_line(ax, 0.95, 0.5, "half", color=MUTED)
    ax.set_ylim(0, 1.02)
    ax.invert_xaxis()
    style_axes(ax, xlabel="hint density rho  (sparser to the right)",
               ylabel="fraction of first errors naming an unforced node")
    save(f_, path)


def f2_in_f_rate(haz):
    f_, axes, path = panels("F2_inF_rate_vs_j.png")
    for ax, rho in zip(axes, PANEL_RHOS):
        agg = collections.defaultdict(lambda: [0, 0])
        for r in haz:
            if abs(f(r, "rho") - rho) > 1e-9:
                continue
            c = agg[i(r, "j")]
            c[0] += i(r, "n_in_F")
            c[1] += i(r, "n_at_risk")
        js = sorted(j for j in agg if agg[j][1] >= 1000)
        if not js:
            ax.set_title("rho %.2f: no rows" % rho)
            continue
        ax.plot(js, [agg[j][0] / agg[j][1] for j in js], color=SERIES[0], lw=1.6)
        ax.set_ylim(0, 1.02)
        ax.set_title("rho %.2f" % rho, fontsize=10)
        style_axes(ax, xlabel="pair index j", ylabel=None)
    style_axes(axes[0], xlabel="pair index j",
               ylabel="share of emitted nodes that DD forces")
    save(f_, path)


def f3_same_eval(cells, pos_summary):
    f_, ax, path = fig("F3_solve_agreement.png", w=5.2, h=4.8)
    xs, ys = [], []
    for key, c in cells.items():
        p = pos_summary.get(key)
        if p is None or c["solve_rate"] != c["solve_rate"]:
            continue
        xs.append(p)
        ys.append(c["solve_rate"])
    if not xs:
        print("  F3 SKIPPED (no rows)")
        plt.close(f_)
        return float("nan"), float("nan"), 0
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1)
    ax.scatter(xs, ys, s=10, color=SERIES[0], alpha=0.6, rasterized=True)
    d = np.abs(np.array(xs) - np.array(ys))
    ax.set_title("max |difference| %.4f on %d cells" % (d.max(), len(xs)), fontsize=10)
    style_axes(ax, xlabel="stage POS measurement B solve rate",
               ylabel="stage LOGIC replayed solve rate")
    save(f_, path)
    r = stats.pearsonr(xs, ys).statistic if len(xs) > 2 else float("nan")
    return float(d.max()), float(r), len(xs)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    os.makedirs(FIGS, exist_ok=True)
    lrows = read_shards(OUT, "L_position_*.csv")
    srows = read_shards(OUT, "LF_summary_*.csv")
    haz = read_shards(OUT, "F_hazard_*.csv")
    erows = read_shards(OUT, "F_errors_*.csv")
    posrows = read_shards(POS, "B_summary_*.csv")
    print("L rows %d   summary rows %d   hazard rows %d   error rows %d"
          % (len(lrows), len(srows), len(haz), len(erows)))
    runs = run_table(srows or lrows)
    print("runs present %d  (in79 %d)"
          % (len(runs), sum(r["in79"] for r in runs.values())))

    cells = {}
    for r in srows:
        key = (r["run_id"], round(f(r, "rho"), 4))
        cells[key] = {
            "run_id": r["run_id"], "rho": f(r, "rho"),
            "realized_rho": f(r, "realized_rho"), "n": blanks(f(r, "realized_rho")),
            "in79": b(r, "in79"), "p_logic": f(r, "p_logic"),
            "p_strict": f(r, "p_strict"), "p_logic_node": f(r, "p_logic_node"),
            "p_strict_node": f(r, "p_strict_node"),
            "solve_rate": f(r, "solve_rate"), "frac_unforced": f(r, "frac_unforced"),
            "n_err_node": i(r, "n_err_node"), "n_first_err": i(r, "n_first_err"),
            "n_err_unforced": i(r, "n_err_unforced"),
            "n_err_unforced_now": i(r, "n_err_unforced_now"),
            "node_premature": i(r, "node_premature"), "node_in_F": i(r, "node_in_F"),
            "node_recolor": i(r, "node_recolor"), "node_nonnode": i(r, "node_nonnode"),
        }
    pos_summary = {(r["run_id"], round(f(r, "rho"), 4)): f(r, "solve_rate")
                   for r in posrows}
    avail = availability(INST, sorted({round(c["rho"], 4) for c in cells.values()}))
    print("availability computed on %d densities from %s" % (len(avail), INST))

    # ---- the fit table ----------------------------------------------------
    fits = {}
    for rho in FIT_RHOS:
        sel = [c for c in cells.values() if abs(c["rho"] - rho) < 1e-9 and c["in79"]]
        if len(sel) < MIN_FIT_POINTS:
            continue
        s = [c["solve_rate"] for c in sel]
        kl, el, cl, r2l, nl = powerlaw([c["p_logic"] for c in sel], s)
        ks, es, cs, r2s, ns = powerlaw([c["p_strict"] for c in sel], s)
        fits[round(rho, 4)] = dict(
            rho=rho, realized=np.mean([c["realized_rho"] for c in sel]),
            n=np.mean([c["n"] for c in sel]), n_runs=len(sel),
            k_logic=kl, se_logic=el, C_logic=cl, r2_logic=r2l, npts_logic=nl,
            k_strict=ks, se_strict=es, C_strict=cs, r2_strict=r2s, npts_strict=ns)

    ok = [v for v in fits.values() if v["k_logic"] == v["k_logic"]]
    a_logic, a_logic_se, _ = through_origin([v["n"] for v in ok],
                                            [v["k_logic"] for v in ok])
    a_strict, a_strict_se, _ = through_origin([v["n"] for v in ok],
                                              [v["k_strict"] for v in ok])
    free_logic = powerlaw([v["n"] / N_NODES for v in ok], [v["k_logic"] for v in ok]) \
        if len(ok) >= 3 else None

    print("\nfigures:")
    if cells:
        l1_logic_vs_strict(cells)
        l2_fit_panels(cells, fits)
    if ok:
        l3_k_vs_n(fits, a_logic, a_strict)
        l4_prefactor(fits)
        l5_r2(fits)
    jrows = [r for r in lrows if r.get("scheme") == "j"]
    if jrows:
        l6_acc_vs_j(jrows, runs)
        l7_classes(jrows)
    if cells:
        f1_unforced(cells, runs)
    if haz:
        f2_in_f_rate(haz)
    d_max, d_r, d_n = f3_same_eval(cells, pos_summary) if pos_summary else (float("nan"),) * 2 + (0,)

    # ---- the verdict ------------------------------------------------------
    L = []
    w = L.append
    w("Stage LOGIC -- registered predictions scored against the measurement.")
    w("Population on disk: %d runs (%d of the 79)."
      % (len(runs), sum(r["in79"] for r in runs.values())))
    w("Nothing below was chosen after seeing the numbers; the predictions and the")
    w("figure list are in REGISTERED.md, committed before the job was submitted.")
    w("")
    w("THE FIT TABLE.  S = C p^k per density, OLS of log S on log p over the in79")
    w("runs with 0 < S < 1.  n = 48(1 - realized rho).")
    w("")
    w("  rho   real_rho     n  runs |  k_logic    +-se   C_log   R2_log | "
      "k_strict    +-se  C_str  R2_str |  k_log/n")
    for rho in sorted(fits, reverse=True):
        v = fits[rho]
        w("  %.2f    %.4f  %5.1f  %4d |  %7.2f %7.2f  %6.3f   %5.3f | "
          "%7.2f %7.2f %6.3f   %5.3f |  %6.3f"
          % (v["rho"], v["realized"], v["n"], v["npts_logic"], v["k_logic"],
             v["se_logic"], v["C_logic"], v["r2_logic"], v["k_strict"],
             v["se_strict"], v["C_strict"], v["r2_strict"],
             v["k_logic"] / v["n"] if v["n"] else float("nan")))
    w("")

    w("L1  k_logic against n, forced through the origin: a = %.3f +- %.3f on %d "
      "densities." % (a_logic, a_logic_se, len(ok)))
    w("    strict, the same fit: a = %.3f +- %.3f  (hshunt's background value 0.73)."
      % (a_strict, a_strict_se))
    if free_logic and free_logic[0] == free_logic[0]:
        w("    free power fit k_logic ~ (n/48)^%.2f +- %.2f, R2 %.3f."
          % (free_logic[0], free_logic[1], free_logic[3]))
    hi = [v["C_logic"] for v in ok if v["rho"] >= 0.65 - 1e-9]
    medC = float(np.median(hi)) if hi else float("nan")
    w("    median prefactor over rho >= 0.65: C = %.3f on %d densities."
      % (medC, len(hi)))
    verdict = ("HIT" if L1_BAND[0] <= a_logic <= L1_BAND[1] else
               "MISS, FALSIFIED" if not (L1_FALSIFY[0] <= a_logic <= L1_FALSIFY[1])
               else "MISS, not falsified")
    note = ("" if L1_PREFACTOR[0] <= medC <= L1_PREFACTOR[1]
            else "  (prefactor OUTSIDE its band)")
    w("    predicted a in [%.2f, %.2f], falsified outside [%.2f, %.2f], prefactor "
      "in [%.1f, %.1f]  -> %s%s"
      % (L1_BAND + L1_FALSIFY + L1_PREFACTOR + (verdict, note)))
    w("")

    bad = [(c["run_id"], c["rho"], c["p_logic"] - c["p_strict"])
           for c in cells.values() if c["p_logic"] < c["p_strict"] - 1e-12]
    w("L2  p_logic >= p_strict on %d of %d (run, rho) cells."
      % (len(cells) - len(bad), len(cells)))
    if bad:
        w("    %d VIOLATIONS -- this is an arithmetic identity, so it is a BUG, not"
          " a finding. Worst: %s rho %.2f, %.2e"
          % (len(bad), bad[0][0], bad[0][1], min(b[2] for b in bad)))
    else:
        gains = [c["p_logic"] - c["p_strict"] for c in cells.values()]
        w("    No violation. Median gain %+.4f, max %+.4f -- that is the share of"
          % (float(np.median(gains)), float(np.max(gains))))
        w("    measurement A's errors that were order bookkeeping, not reasoning.")
    w("")

    ratios = [v["k_logic"] / v["k_strict"] for v in ok
              if v["rho"] >= 0.50 - 1e-9 and v["k_strict"] not in (0.0,)
              and v["k_strict"] == v["k_strict"]]
    medr = float(np.median(ratios)) if ratios else float("nan")
    w("L3  median k_logic / k_strict over rho >= 0.50: %.3f on %d densities."
      % (medr, len(ratios)))
    w("    predicted [%.1f, %.1f], falsified below 1.0 or above 2.0  -> %s"
      % (L3_BAND[0], L3_BAND[1],
         "HIT" if L3_BAND[0] <= medr <= L3_BAND[1] else
         "MISS, FALSIFIED" if not (1.0 <= medr <= 2.0) else "MISS, not falsified"))
    w("")

    fr = [c["frac_unforced"] for c in cells.values()
          if c["in79"] and 0.50 - 1e-9 <= c["rho"] <= 0.75 + 1e-9
          and c["n_err_node"] >= 50 and c["frac_unforced"] == c["frac_unforced"]]
    medf = float(np.median(fr)) if fr else float("nan")
    w("L4  fraction of first errors naming an UNFORCED node, rho 0.50..0.75, in79 "
      "runs with >= 50 scored errors:")
    w("    median %.3f over %d (run, rho) cells; quartiles %.3f / %.3f."
      % (medf, len(fr),
         float(np.percentile(fr, 25)) if fr else float("nan"),
         float(np.percentile(fr, 75)) if fr else float("nan")))
    w("    predicted > 0.5, falsified below  -> %s"
      % ("HIT" if medf > 0.5 else "MISS, FALSIFIED"))
    tot_e = sum(c["n_err_node"] for c in cells.values())
    tot_u = sum(c["n_err_unforced"] for c in cells.values())
    tot_n = sum(c["n_err_unforced_now"] for c in cells.values())
    w("    pooled over every density and run: %d of %d scored first errors (%.3f%%)"
      % (tot_u, tot_e, 100.0 * tot_u / tot_e if tot_e else float("nan")))
    w("    name a node DD had not yet forced; the %d exception%s named a forced node"
      % (tot_e - tot_u, "" if tot_e - tot_u == 1 else "s"))
    w("    and mis-coloured it. Against the model's OWN current state (same-round")
    w("    siblings counted, the second registered column): %d of %d (%.3f%%)."
      % (tot_n, tot_e, 100.0 * tot_n / tot_e if tot_e else float("nan")))
    w("")

    w("L5  the premature-rate confound, reported raw AND against availability.")
    w("    availability = mean over node positions of (N - coloured - |F|)/N, a")
    w("    property of the eval set with no model in it: how much of the graph was")
    w("    even available to be named prematurely. The normalised column is the raw")
    w("    rate divided by it -- 1.0 would mean 'names an available node at chance'.")
    w("")
    w("    rho      n | premature  in_F  recolor nonnode |   raw   avail   norm")
    for rho in sorted({round(c["rho"], 4) for c in cells.values()}, reverse=True):
        sel = [c for c in cells.values() if abs(c["rho"] - rho) < 1e-9 and c["in79"]]
        if not sel:
            continue
        tot = sum(c["node_premature"] + c["node_in_F"] + c["node_recolor"]
                  + c["node_nonnode"] for c in sel)
        if not tot:
            continue
        pr = sum(c["node_premature"] for c in sel) / tot
        av = avail.get(round(rho, 4), (float("nan"), {}))[0]
        w("    %.2f  %5.1f |  %8.4f %5.3f  %7.4f %7.4f | %5.3f  %5.3f  %5.3f"
          % (rho, np.mean([c["n"] for c in sel]), pr,
             sum(c["node_in_F"] for c in sel) / tot,
             sum(c["node_recolor"] for c in sel) / tot,
             sum(c["node_nonnode"] for c in sel) / tot,
             pr, av, pr / av if av == av and av > 0 else float("nan")))
    raws, norms = [], []
    for rho in sorted({round(c["rho"], 4) for c in cells.values()}):
        sel = [c for c in cells.values() if abs(c["rho"] - rho) < 1e-9 and c["in79"]]
        tot = sum(c["node_premature"] + c["node_in_F"] + c["node_recolor"]
                  + c["node_nonnode"] for c in sel)
        av = avail.get(round(rho, 4), (float("nan"), {}))[0]
        if tot and av == av and av > 0:
            raws.append((rho, sum(c["node_premature"] for c in sel) / tot))
            norms.append((rho, raws[-1][1] / av))
    if len(raws) >= 4:
        tr = stats.spearmanr([r[0] for r in raws], [r[1] for r in raws]).statistic
        tn = stats.spearmanr([r[0] for r in norms], [r[1] for r in norms]).statistic
        w("")
        w("    Spearman of the rate against rho: raw %+.3f, normalised %+.3f over %d"
          % (tr, tn, len(raws)))
        w("    densities. If only the raw version trends, the honest statement is")
        w("    'there are more ways to be premature', not 'the guard is weaker'.")
        w("    -> %s" % ("both trend: the guard really does weaken as rho falls"
                         if tn < -0.5 else
                         "raw only: AVAILABILITY, not a weaker guard" if tr < -0.5
                         else "neither trends"))
        thin = [r for r, _ in raws if avail.get(round(r, 4), (1.0, {}))[0] < 0.025]
        if thin:
            w("    CAVEAT, A MAGNITUDE CHECKED AT A DEGENERATE POINT IS NOT CHECKED:")
            w("    at rho %s availability is below 0.025, i.e. ABOUT ONE node of 48"
              % ", ".join("%.2f" % r for r in sorted(thin, reverse=True)))
            w("    is even available to name prematurely, so the normalised column")
            w("    there divides by nearly nothing. Read it from rho 0.80 down.")
    w("")

    w("SAME-EVAL CHECK (figure F3). The replayed rollouts must be stage POS's own.")
    w("  max |difference| in solve rate over %d (run, rho) cells: %.4f, Pearson r "
      "%.4f." % (d_n, d_max, d_r))
    w("  A nonzero maximum means the replay is NOT the same rollout and every F")
    w("  number above is about a different decode.")

    text = "\n".join(L)
    print("\npredictions:\n")
    print(text)
    path = os.path.join(OUT, "VERDICT.txt")
    with open(path, "w") as fh:
        fh.write(text + "\n")
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
