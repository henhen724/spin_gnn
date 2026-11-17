#!/usr/bin/env python3
"""Stage POS: merge the shards, draw the ten registered figures, score P1-P5.

Reads whatever `A_position_*.csv` / `B_hazard_*.csv` / `B_instances_*.csv` /
`B_summary_*.csv` shards exist in POS_OUT (skipping `*smoke*`) and writes
figures to POS_OUT/figures/ plus a plain-text verdict file.

THE FIGURE LIST IS THE ONE REGISTERED IN runs/stagepos_2026-11-17/REGISTERED.md
BEFORE THE JOB RAN. Nothing is added here that was not registered there, and
nothing registered there is quietly dropped: a figure with no data prints
"SKIPPED (no rows)" instead of disappearing.

Definitions, repeated here so a reader of this file alone cannot get them wrong:

  hazard(j)      = (n_err + n_stop) / n_at_risk      both failure modes
  hazard_err(j)  = n_err / n_at_risk                 bad pairs only
  survival S(j)  = prod_{i<j} (1 - hazard(i))
  n              = 48 * (1 - realized_rho)           blank cells

`n_stop` is early SOLVED, UNRESOLVED on a resolvable instance, or running out
of budget. It is kept separate so a model that quits early cannot look
error-free, and pooled into `hazard` because quitting early IS a failure to
solve.

Run with the figure module set:
  module purge
  module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 \\
      py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  POS_OUT=$SCRATCH/spin_gnn/runs/stagepos_2026-11-17 python3 scripts/stagepos_figs.py
"""

import collections
import csv
import glob
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line, series_colors  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from stagepos_cells import ALL_DIRS, RHO_GRID  # noqa: E402
from minarch_analysis import load_all, cell_key  # noqa: E402

OUT = os.environ.get("POS_OUT") or "/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17"
FIGS = os.path.join(OUT, "figures")
DPI = 150
N_NODES = 48

# the three densities the registered small multiples use
PANEL_RHOS = [0.75, 0.60, 0.50]
OVERLAY_RHO = 0.60
MIN_AT_RISK = 100          # P1's floor: a hazard on <100 instances is noise
P2_BAND = (1.5, 2.3)       # predicted
P2_FALSIFY = (1.3, 2.5)    # falsified outside
REF_EXPONENT = 1.84        # hshunt's k ~ n^1.84


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def read_shards(pattern):
    """Every shard matching the pattern, smoke excluded, as dicts of str."""
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, pattern))):
        if "smoke" in os.path.basename(path):
            continue
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                r["_shard"] = os.path.basename(path)
                rows.append(r)
    return rows


def f(r, k):
    v = r.get(k, "")
    try:
        return float(v)
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
    """run_id -> the run's identity columns, from whichever CSV has them."""
    runs = {}
    for r in rows:
        rid = r.get("run_id")
        if rid and rid not in runs:
            runs[rid] = {
                "run_id": rid, "cell": r.get("cell", ""), "key": r.get("key", ""),
                "seed": i(r, "seed"), "arch_mode": r.get("arch_mode", ""),
                "num_layers": i(r, "num_layers"), "d_model": i(r, "d_model"),
                "n_trainable_params": i(r, "n_trainable_params"),
                "val_acc": f(r, "val_acc"), "solve50": f(r, "solve50"),
                "in79": b(r, "in79"), "in_seven": b(r, "in_seven"),
            }
    return runs


# --------------------------------------------------------------------------
# hazard assembly
# --------------------------------------------------------------------------

def hazard_series(hrows):
    """(run_id, rho) -> arrays j, hazard, hazard_err, n_at_risk, realized_rho."""
    by = collections.defaultdict(list)
    for r in hrows:
        by[(r["run_id"], round(f(r, "rho"), 4))].append(r)
    out = {}
    for key, rs in by.items():
        rs.sort(key=lambda r: i(r, "j"))
        j = np.array([i(r, "j") for r in rs], float)
        risk = np.array([f(r, "n_at_risk") for r in rs])
        err = np.array([f(r, "n_err") for r in rs])
        stop = np.array([f(r, "n_stop") for r in rs])
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(risk > 0, (err + stop) / risk, np.nan)
            he = np.where(risk > 0, err / risk, np.nan)
        out[key] = {
            "j": j, "h": h, "h_err": he, "risk": risk,
            "realized_rho": f(rs[0], "realized_rho"),
            "n": N_NODES * (1.0 - f(rs[0], "realized_rho")),
            "dd_mean_T": f(rs[0], "dd_mean_T"),
            "round_mean": np.array([f(r, "n_round_sum") / max(f(r, "n_emitted"), 1)
                                    for r in rs]),
        }
    return out


def survival(h):
    s = np.ones_like(h)
    acc = 1.0
    for k, hv in enumerate(h):
        s[k] = acc
        if hv == hv:
            acc *= (1.0 - hv)
    return s


def integrated_hazard(d):
    """sum_j h(j) over the j where the hazard is actually measured."""
    m = (d["risk"] >= 1) & np.isfinite(d["h"])
    return float(np.nansum(d["h"][m]))


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def powerlaw_fit(x, y):
    """Least squares on log-log. Returns slope, stderr, r2, npts."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 3:
        return float("nan"), float("nan"), float("nan"), int(m.sum())
    res = stats.linregress(np.log(x[m]), np.log(y[m]))
    return float(res.slope), float(res.stderr), float(res.rvalue ** 2), int(m.sum())


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def _grid(n_panels):
    ncol = min(4, max(1, n_panels))
    nrow = int(math.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow),
                             squeeze=False)
    return fig, [axes[r][c] for r in range(nrow) for c in range(ncol)], axes


def _finish(fig, axes_flat, used, path):
    for ax in axes_flat[used:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % os.path.basename(path))


def a_panels(arows, runs, scheme, tok_types, fname, xlabel, title_note):
    """A1/A2/A3: teacher-forced accuracy, one panel per named run."""
    named = [r for r in sorted(runs) if runs[r]["in_seven"]]
    if not named:
        print("  SKIPPED %s (no named runs)" % fname)
        return
    sel = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in arows:
        if r.get("scheme") != scheme or r.get("tok_type") not in tok_types:
            continue
        if not b(r, "in_seven"):
            continue
        sel[r["run_id"]][round(f(r, "rho"), 4)].append(r)
    if not sel:
        print("  SKIPPED %s (no rows)" % fname)
        return
    fig, flat, _ = _grid(len(named))
    for k, rid in enumerate(named):
        ax = flat[k]
        for c, rho in enumerate(PANEL_RHOS):
            rs = sel.get(rid, {}).get(round(rho, 4), [])
            if not rs:
                continue
            agg = collections.defaultdict(lambda: [0.0, 0.0])
            for r in rs:
                agg[i(r, "index")][0] += f(r, "n_correct")
                agg[i(r, "index")][1] += f(r, "n_total")
            xs = sorted(x for x in agg if x >= 0 and agg[x][1] >= 20)
            if not xs:
                continue
            ys = [agg[x][0] / agg[x][1] for x in xs]
            ax.plot(xs, ys, color=SERIES[c], lw=1.4)
            label_line(ax, xs[-1], ys[-1], "%.2f" % rho, color=SERIES[c])
        style_axes(ax, xlabel=xlabel, ylabel="next-token accuracy")
        ax.set_ylim(0, 1.02)
        ax.set_title("%s s%d" % (runs[rid]["cell"], runs[rid]["seed"]),
                     fontsize=8, color=MUTED)
    fig.suptitle(title_note, fontsize=9, color=MUTED)
    _finish(fig, flat, len(named), os.path.join(FIGS, fname))


def b1_panels(haz, runs):
    named = [r for r in sorted(runs) if runs[r]["in_seven"]]
    if not named:
        print("  SKIPPED B1_hazard_named.png (no named runs)")
        return
    fig, flat, _ = _grid(len(named))
    for k, rid in enumerate(named):
        ax = flat[k]
        drawn = 0
        for c, rho in enumerate(PANEL_RHOS):
            d = haz.get((rid, round(rho, 4)))
            if d is None:
                continue
            m = d["risk"] >= MIN_AT_RISK
            if m.sum() < 2:
                continue
            ax.plot(d["j"][m], d["h"][m], color=SERIES[c], lw=1.4)
            label_line(ax, d["j"][m][-1], d["h"][m][-1], "%.2f" % rho, color=SERIES[c])
            drawn += 1
        style_axes(ax, xlabel="deduction index $j$", ylabel="hazard $h(j)$")
        ax.set_ylim(bottom=0)
        ax.set_title("%s s%d" % (runs[rid]["cell"], runs[rid]["seed"]),
                     fontsize=8, color=MUTED)
        if not drawn:
            ax.text(0.5, 0.5, "no bin with $\\geq$%d at risk" % MIN_AT_RISK,
                    ha="center", va="center", fontsize=7, color=MUTED,
                    transform=ax.transAxes)
    fig.suptitle("hazard against deduction index, by density", fontsize=9, color=MUTED)
    _finish(fig, flat, len(named), os.path.join(FIGS, "B1_hazard_named.png"))


def _acc_colors(vals):
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    span = max(hi - lo, 1e-9)
    cmap = plt.get_cmap("viridis")
    return cmap, lo, hi, [cmap(0.12 + 0.8 * (v - lo) / span) for v in vals]


def overlay(haz, runs, kind, fname, ylabel, title_note):
    """B2/B3: one line per run at rho=OVERLAY_RHO, coloured by val accuracy."""
    keys = [(rid, d) for (rid, rho), d in haz.items()
            if abs(rho - OVERLAY_RHO) < 1e-9 and runs.get(rid, {}).get("in79")]
    if not keys:
        print("  SKIPPED %s (no rows)" % fname)
        return
    keys.sort(key=lambda kd: runs[kd[0]]["val_acc"])
    accs = np.array([runs[rid]["val_acc"] for rid, _ in keys])
    cmap, lo, hi, cols = _acc_colors(accs)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for (rid, d), col in zip(keys, cols):
        m = d["risk"] >= MIN_AT_RISK
        if m.sum() < 2:
            continue
        y = d["h"][m] if kind == "hazard" else survival(d["h"])[m]
        ax.plot(d["j"][m], y, color=col, lw=0.9, alpha=0.85, rasterized=True)
    style_axes(ax, xlabel="deduction index $j$", ylabel=ylabel)
    ax.set_ylim(bottom=0)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(lo, hi))
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label("final validation token accuracy", color=MUTED, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title(title_note, fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % fname)


def b4(haz, runs, verdict):
    """THE QUANTITATIVE TEST: sum_j h(j) against n, log-log, over rho >= 0.50."""
    xs, ys, accs = [], [], []
    for (rid, rho), d in haz.items():
        if not runs.get(rid, {}).get("in79") or rho < 0.4999:
            continue
        H = integrated_hazard(d)
        if d["n"] > 0 and H > 0:
            xs.append(d["n"])
            ys.append(H)
            accs.append(runs[rid]["val_acc"])
    if len(xs) < 3:
        print("  SKIPPED B4_integrated_hazard.png (no rows)")
        return
    xs, ys, accs = np.array(xs), np.array(ys), np.array(accs)
    slope, se, r2, npts = powerlaw_fit(xs, ys)
    cmap, lo, hi, cols = _acc_colors(accs)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.scatter(xs, ys, s=10, c=cols, alpha=0.8, linewidths=0, rasterized=True)
    gx = np.linspace(max(xs.min(), 1e-6), xs.max(), 50)
    anchor = float(np.nanmedian(ys / xs ** REF_EXPONENT))
    ax.plot(gx, anchor * gx ** REF_EXPONENT, color=MUTED, lw=1.4, ls="--")
    label_line(ax, gx[-1], anchor * gx[-1] ** REF_EXPONENT,
               "$n^{%.2f}$" % REF_EXPONENT, color=MUTED)
    if slope == slope:
        anchor2 = float(np.nanmedian(ys / xs ** slope))
        ax.plot(gx, anchor2 * gx ** slope, color=SERIES[1], lw=1.6)
        label_line(ax, gx[-1], anchor2 * gx[-1] ** slope,
                   "fit $n^{%.2f}$" % slope, color=SERIES[1])
    style_axes(ax, xlabel="blank cells $n = 48(1-\\rho_{\\rm real})$",
               ylabel="$\\sum_j h(j)$", logx=True, logy=True)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(lo, hi))
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label("final validation token accuracy", color=MUTED, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "B4_integrated_hazard.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote B4_integrated_hazard.png")
    verdict.append(
        "P2(b) integrated hazard vs n, rho >= 0.50, %d points: slope %.3f "
        "+- %.3f, R2 %.3f.  predicted [%.1f, %.1f], falsified outside "
        "[%.1f, %.1f]  -> %s"
        % (npts, slope, se, r2, P2_BAND[0], P2_BAND[1], P2_FALSIFY[0],
           P2_FALSIFY[1],
           "HIT" if P2_BAND[0] <= slope <= P2_BAND[1] else
           ("MISS, FALSIFIED" if not (P2_FALSIFY[0] <= slope <= P2_FALSIFY[1])
            else "between the band and the falsifier")))


def c1(haz, runs):
    """P4's confound: hazard against the mean round at each j."""
    named = [r for r in sorted(runs) if runs[r]["in_seven"]]
    if not named:
        print("  SKIPPED C1_hazard_vs_round.png (no named runs)")
        return
    fig, flat, _ = _grid(len(named))
    for k, rid in enumerate(named):
        ax = flat[k]
        for c, rho in enumerate(PANEL_RHOS):
            d = haz.get((rid, round(rho, 4)))
            if d is None:
                continue
            m = (d["risk"] >= MIN_AT_RISK) & np.isfinite(d["round_mean"])
            if m.sum() < 2:
                continue
            ax.plot(d["round_mean"][m], d["h"][m], color=SERIES[c], lw=1.2,
                    marker="o", ms=2.5)
            label_line(ax, d["round_mean"][m][-1], d["h"][m][-1], "%.2f" % rho,
                       color=SERIES[c])
        style_axes(ax, xlabel="mean DD round at $j$", ylabel="hazard $h(j)$")
        ax.set_ylim(bottom=0)
        ax.set_title("%s s%d" % (runs[rid]["cell"], runs[rid]["seed"]),
                     fontsize=8, color=MUTED)
    fig.suptitle("the round/position confound: hazard against DD depth",
                 fontsize=9, color=MUTED)
    _finish(fig, flat, len(named), os.path.join(FIGS, "C1_hazard_vs_round.png"))


def c2(haz, runs):
    """P5's confound: hazard against j/n."""
    keys = [(rid, d) for (rid, rho), d in haz.items()
            if abs(rho - OVERLAY_RHO) < 1e-9 and runs.get(rid, {}).get("in79")]
    if not keys:
        print("  SKIPPED C2_hazard_vs_jovern.png (no rows)")
        return
    accs = np.array([runs[rid]["val_acc"] for rid, _ in keys])
    cmap, lo, hi, cols = _acc_colors(accs)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for (rid, d), col in zip(keys, cols):
        m = d["risk"] >= MIN_AT_RISK
        if m.sum() < 2 or d["n"] <= 0:
            continue
        ax.plot(d["j"][m] / d["n"], d["h"][m], color=col, lw=0.9, alpha=0.85,
                rasterized=True)
    style_axes(ax, xlabel="fraction of the trace completed, $j/n$",
               ylabel="hazard $h(j)$")
    ax.set_ylim(bottom=0)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(lo, hi))
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label("final validation token accuracy", color=MUTED, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title("hazard against $j/n$, $\\rho=%.2f$" % OVERLAY_RHO,
                 fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "C2_hazard_vs_jovern.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote C2_hazard_vs_jovern.png")


def d1(srows, runs, verdict):
    """The same-eval comparison: this run's solve rate vs the published curve."""
    pub = {}
    for r in load_all(ALL_DIRS):
        pub[(cell_key(r), round(float(r["rho"]), 4))] = float(r["solve_rate"])
    xs, ys = [], []
    for r in srows:
        rid = r["run_id"]
        key = runs.get(rid, {}).get("key", "")
        p = pub.get((tuple(_unkey(key)), round(f(r, "rho"), 4)))
        s = f(r, "solve_rate")
        if p is None or s != s:
            continue
        xs.append(p)
        ys.append(s)
    if len(xs) < 3:
        verdict.append("D1 SKIPPED: could not match any (cell, rho) to a "
                       "published rho-curve solve rate.")
        print("  SKIPPED D1_solve_agreement.png (no matched rows)")
        return
    xs, ys = np.array(xs), np.array(ys)
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls="--")
    ax.scatter(xs, ys, s=10, color=SERIES[0], alpha=0.7, linewidths=0, rasterized=True)
    style_axes(ax, xlabel="published rho-curve solve rate",
               ylabel="this run's solve rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "D1_solve_agreement.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote D1_solve_agreement.png")
    d = ys - xs
    verdict.append("D1 same-eval agreement on %d (run, rho) cells: mean "
                   "difference %+.4f, max |difference| %.4f, Pearson r %.4f."
                   % (len(xs), float(d.mean()), float(np.abs(d).max()),
                      float(np.corrcoef(xs, ys)[0, 1])))


def _unkey(s):
    """'a|b|c|0' back to the tuple cell_key produced, ints where they were ints."""
    out = []
    for part in s.split("|"):
        try:
            out.append(int(part))
        except ValueError:
            try:
                out.append(float(part))
            except ValueError:
                out.append(part)
    return out


# --------------------------------------------------------------------------
# predictions
# --------------------------------------------------------------------------

def score_p1(haz, runs, verdict):
    """Spearman rho_s(j, h(j)) over the bins with n_at_risk >= MIN_AT_RISK."""
    per_rho = collections.defaultdict(list)
    for (rid, rho), d in haz.items():
        if not runs.get(rid, {}).get("in79") or not (0.4999 <= rho <= 0.7501):
            continue
        m = (d["risk"] >= MIN_AT_RISK) & np.isfinite(d["h"])
        if m.sum() < 4:
            continue
        rs = stats.spearmanr(d["j"][m], d["h"][m]).statistic
        if rs == rs:
            per_rho[rho].append((rid, float(rs)))
    if not per_rho:
        verdict.append("P1 SKIPPED: no (run, rho) cell had 4 bins with "
                       "n_at_risk >= %d." % MIN_AT_RISK)
        return
    lines = ["P1 hazard rises with j.  Spearman rho_s(j, h(j)), bins with "
             "n_at_risk >= %d:" % MIN_AT_RISK]
    for rho in sorted(per_rho, reverse=True):
        vals = np.array([v for _, v in per_rho[rho]])
        pos = int((vals > 0).sum())
        seven = [v for rid, v in per_rho[rho] if runs[rid]["in_seven"]]
        lines.append(
            "   rho %.2f: %d/%d runs positive, median rho_s %+.3f; "
            "named runs %d/%d positive"
            % (rho, pos, len(vals), float(np.median(vals)),
               int(sum(1 for v in seven if v > 0)), len(seven)))
    allv = np.array([v for rho in per_rho for _, v in per_rho[rho]])
    maj = float((allv > 0).mean())
    flat = float((np.abs(allv) < 0.3).mean())
    lines.append("   pooled: %.1f%% of (run, rho) cells positive, %.1f%% flat "
                 "(|rho_s| < 0.3)  -> %s"
                 % (100 * maj, 100 * flat,
                    "HIT" if maj > 0.5 and flat < 0.5 else "MISS, FALSIFIED"))
    verdict.append("\n".join(lines))


def score_p2a(haz, srows, verdict):
    """prod_j (1 - h(j)) must reproduce the measured solve rate. An identity."""
    meas = {(r["run_id"], round(f(r, "rho"), 4)): f(r, "solve_rate") for r in srows}
    worst, n = 0.0, 0
    for key, d in haz.items():
        if key not in meas or meas[key] != meas[key]:
            continue
        pred = float(np.prod(1.0 - np.nan_to_num(d["h"], nan=0.0)))
        worst = max(worst, abs(pred - meas[key]))
        n += 1
    if not n:
        verdict.append("P2(a) SKIPPED: no (run, rho) cell had both a hazard "
                       "series and a solve rate.")
        return
    verdict.append(
        "P2(a) arithmetic identity on %d (run, rho) cells: max |prod(1-h) - "
        "solve rate| = %.2e  -> %s"
        % (n, worst, "HIT" if worst < 1e-9 else
           "MISMATCH -- THIS IS A BUG, NOT A FINDING"))


def score_p3(arows, runs, verdict):
    """Teacher-forced colour accuracy at the last deduction vs the first."""
    sel = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0]))
    for r in arows:
        if r.get("scheme") != "j" or r.get("tok_type") != "color":
            continue
        if abs(f(r, "rho") - 0.60) > 1e-9:
            continue
        idx = i(r, "index")
        if idx < 0:
            continue
        sel[r["run_id"]][idx][0] += f(r, "n_correct")
        sel[r["run_id"]][idx][1] += f(r, "n_total")
    if not sel:
        verdict.append("P3 SKIPPED: no teacher-forced colour rows at rho=0.60.")
        return
    falls, named_falls, named_n, deltas = 0, 0, 0, []
    for rid, per_j in sel.items():
        xs = sorted(j for j in per_j if per_j[j][1] >= 20)
        if len(xs) < 4:
            continue
        first = np.mean([per_j[j][0] / per_j[j][1] for j in xs[:3]])
        last = np.mean([per_j[j][0] / per_j[j][1] for j in xs[-3:]])
        deltas.append(last - first)
        if last < first:
            falls += 1
            if runs.get(rid, {}).get("in_seven"):
                named_falls += 1
        if runs.get(rid, {}).get("in_seven"):
            named_n += 1
    if not deltas:
        verdict.append("P3 SKIPPED: no run had 4 usable deduction indices.")
        return
    verdict.append(
        "P3 teacher-forced colour accuracy falls with position (rho=0.60): "
        "%d/%d runs fall, named %d/%d, median (last - first) %+.4f.\n"
        "   Registered as EXPECTED TO MISS. A flat A with a rising B means the "
        "position dependence is error ACCUMULATION, not intrinsic difficulty. "
        "-> %s"
        % (falls, len(deltas), named_falls, named_n, float(np.median(deltas)),
           "P3 HIT (>=5 of the named runs fall)" if named_falls >= 5 else
           "P3 MISSED as expected" if named_n else "P3 undecided"))


def score_p4(haz, runs, verdict):
    """Does the trend survive within a round, or is it only across rounds?"""
    within, across = [], []
    for (rid, rho), d in haz.items():
        if not runs.get(rid, {}).get("in79") or not (0.4999 <= rho <= 0.7501):
            continue
        m = (d["risk"] >= MIN_AT_RISK) & np.isfinite(d["h"]) & np.isfinite(d["round_mean"])
        if m.sum() < 6:
            continue
        j, h, rm = d["j"][m], d["h"][m], d["round_mean"][m]
        rs_across = stats.spearmanr(rm, h).statistic
        # partial: regress out the round, correlate the residual hazard with j
        A = np.vstack([np.ones_like(rm), rm]).T
        beta, *_ = np.linalg.lstsq(A, h, rcond=None)
        resid = h - A @ beta
        rs_within = stats.spearmanr(j, resid).statistic
        if rs_across == rs_across:
            across.append(float(rs_across))
        if rs_within == rs_within:
            within.append(float(rs_within))
    if not across:
        verdict.append("P4 SKIPPED: no cell had 6 usable bins.")
        return
    verdict.append(
        "P4 the round/position confound, reported either way: hazard vs mean "
        "DD round, median Spearman %+.3f over %d cells (%.0f%% positive); "
        "hazard with the round regressed out, vs j, median Spearman %+.3f "
        "(%.0f%% positive).\n"
        "   If only the across-round trend survives, the finding is 'error "
        "grows with DD depth', NOT 'with token position'."
        % (float(np.median(across)), len(across),
           100 * float(np.mean(np.array(across) > 0)),
           float(np.median(within)) if within else float("nan"),
           100 * float(np.mean(np.array(within) > 0)) if within else float("nan")))


def score_p5(haz, runs, verdict):
    """Does the hazard collapse on j/n, or does it still rise in j/n?"""
    rs_j, rs_frac = [], []
    for (rid, rho), d in haz.items():
        if not runs.get(rid, {}).get("in79") or not (0.4999 <= rho <= 0.7501):
            continue
        m = (d["risk"] >= MIN_AT_RISK) & np.isfinite(d["h"])
        if m.sum() < 4 or d["n"] <= 0:
            continue
        rs_j.append(float(stats.spearmanr(d["j"][m], d["h"][m]).statistic))
        rs_frac.append(float(stats.spearmanr(d["j"][m] / d["n"], d["h"][m]).statistic))
    if not rs_j:
        verdict.append("P5 SKIPPED: no usable cell.")
        return
    verdict.append(
        "P5 the fewer-blanks-left confound: hazard vs j, median Spearman "
        "%+.3f; hazard vs j/n, median Spearman %+.3f over %d cells. "
        "Monotone rescaling within a cell cannot change Spearman, so these "
        "agree by construction -- what j/n buys is COMPARABILITY ACROSS "
        "densities, which is figure C2, not this number."
        % (float(np.median(rs_j)), float(np.median(rs_frac)), len(rs_j)))


# --------------------------------------------------------------------------

def main():
    os.makedirs(FIGS, exist_ok=True)
    arows = read_shards("A_position_*.csv")
    hrows = read_shards("B_hazard_*.csv")
    srows = read_shards("B_summary_*.csv")
    irows = read_shards("B_instances_*.csv")
    runs = run_table(arows + hrows + srows + irows)
    n79 = sum(1 for r in runs.values() if r["in79"])
    print("A rows %d   hazard rows %d   summary rows %d   instance rows %d"
          % (len(arows), len(hrows), len(srows), len(irows)))
    print("runs present %d  (in79 %d, named %d)"
          % (len(runs), n79, sum(1 for r in runs.values() if r["in_seven"])))
    if not runs:
        raise SystemExit("no CSVs in %s -- has the job finished?" % OUT)
    if n79 != 79:
        print("NOTE: %d of the 79 successful runs are present, not 79. Figures "
              "and fits below cover only what is on disk." % n79)

    haz = hazard_series(hrows)
    verdict = []

    print("\nfigures:")
    a_panels(arows, runs, "offset", {"node", "color", "round", "term"},
             "A1_acc_vs_offset.png", "token offset from BEGIN_REASONING",
             "teacher-forced accuracy against token offset, by density")
    a_panels(arows, runs, "j", {"color"}, "A2_acc_vs_j_color.png",
             "deduction index $j$",
             "teacher-forced accuracy, COLOUR tokens only")
    a_panels(arows, runs, "j", {"node"}, "A2_acc_vs_j_node.png",
             "deduction index $j$",
             "teacher-forced accuracy, NODE tokens only")
    a_panels(arows, runs, "round", {"node", "color", "round", "term"},
             "A3_acc_vs_round.png", "DD round index",
             "teacher-forced accuracy against DD round, by density")
    b1_panels(haz, runs)
    overlay(haz, runs, "hazard", "B2_hazard_all.png", "hazard $h(j)$",
            "hazard against $j$, $\\rho=%.2f$, one line per run" % OVERLAY_RHO)
    overlay(haz, runs, "survival", "B3_survival_all.png",
            "survival $S(j) = \\prod_{i<j}(1-h(i))$",
            "survival against $j$, $\\rho=%.2f$, one line per run" % OVERLAY_RHO)
    b4(haz, runs, verdict)
    c1(haz, runs)
    c2(haz, runs)
    d1(srows, runs, verdict)

    print("\npredictions:")
    score_p1(haz, runs, verdict)
    score_p2a(haz, srows, verdict)
    score_p3(arows, runs, verdict)
    score_p4(haz, runs, verdict)
    score_p5(haz, runs, verdict)

    text = ("Stage POS -- registered predictions scored against the measurement.\n"
            "Population on disk: %d runs (%d of the 79).\n"
            "Nothing below was chosen after seeing the numbers; the predictions "
            "and the figure list are in REGISTERED.md, committed before the job "
            "was submitted.\n\n" % (len(runs), n79)) + "\n\n".join(verdict) + "\n"
    path = os.path.join(OUT, "VERDICT.txt")
    with open(path, "w") as fh:
        fh.write(text)
    print()
    print(text)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
