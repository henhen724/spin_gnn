#!/usr/bin/env python3
"""Is the solve-rate power law universal? The same plot for the SAT runs.

hshunt's sudoku result: at fixed density, solve rate is close to a power law in
final validation token accuracy, S ~ acc^k, and k grows with the length of the
reasoning trace (k ~ n^1.84 in blank cells). Call it the acc-to-the-effective-
length model. This script asks whether the SAT-solving transformers in this
project show the same thing.

NOTHING IS TRAINED AND NOTHING IS RE-EVALUATED. Every number here is read off
files that were already on disk.

WHAT EXISTS, which is the first result and not a preamble:

  sat_hparam_sweep   51 scored runs, aggregated.csv. Validation token accuracy
                     is `final_accuracy`. The end-to-end metric is
                     `final_val_satisfying_completion_rate` -- a WEAK probe: it
                     teacher-forces the trace up to the second-to-last token
                     and checks whether the single greedy final token yields an
                     assignment satisfying all clauses. It is not free
                     generation.
  sat2_hparam_sweep  103 scored runs. 72 are in aggregated.csv, which was last
                     written 2026-09-29; the other 31 finished 2026-10-02 and
                     have their scores in their own config.json. Validation
                     token accuracy is `final_val_accuracy` (or history[-1]
                     ["accuracy"]); the end-to-end metric is `solve_rate`, a
                     REAL free-generation eval. This is the family that can
                     answer the question.
  scale_sweep        0 files. Also not the SAT model -- it is a spin/Ising
                     scaling sweep.
  sat_checkpoints    0 files.

  The aggregated.csv on disk is NOT overwritten. The 31 missing runs are read
  from their own config.json instead, so hshunt's file is left exactly as it
  was and this script is pure read + new output.

THE ONE THING THAT CANNOT BE DONE FROM DISK. Both sweeps are pinned at a single
difficulty -- every run in sat_hparam_sweep is n_vars=24, k=3, alpha=4.2, and
every run in sat2_hparam_sweep is n_vars=16, k=3, alpha=1.25, m_clauses=20,
n_hints=0 (verified field by field, not assumed). They sweep architecture and
learning rate only. So there is no difficulty axis to fit the exponent against,
and step 3 of the request -- exponent per difficulty bin, compared against mean
trace length -- has no data behind it. The cost of the job that would create it
is reported at the end rather than the job being submitted.
"""

import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line  # noqa: E402

S = "/scratch/users/hshunt/spin_gnn"
OUT = os.path.join(S, "runs", "sat_powerlaw_2026-11-17")
FIGS = os.path.join(OUT, "figures")
DPI = 150

# the sudoku exponents hshunt fitted, for the comparison page
SUDOKU_K = {0.95: 2.0, 0.90: 4.6, 0.80: 19.9, 0.70: 38.0, 0.60: 69.0, 0.50: 92.0}


def _f(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------

def collect_sat1():
    """51 rows: (acc, end-to-end rate). The weak single-step metric."""
    path = os.path.join(S, "sat_hparam_sweep", "aggregated.csv")
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append({
                "family": "sat_hparam_sweep",
                "run": os.path.basename(r["checkpoint_dir"].rstrip("/")),
                "acc": _f(r["final_accuracy"]),
                "baseline": _f(r["baseline_accuracy"]),
                "end2end": _f(r["final_val_satisfying_completion_rate"]),
                "metric": "val_satisfying_completion_rate",
                "d_model": _f(r["d_model"]), "nhead": _f(r["nhead"]),
                "num_layers": _f(r["num_layers"]),
                "n_vars": _f(r["n_vars"]), "alpha": _f(r["alpha"]),
                "m_clauses": float("nan"),
                "n_val": _f(r["n_val_instances"]),
                "source": "aggregated.csv",
            })
    return out


def collect_sat2():
    """103 rows: (acc, solve_rate). aggregated.csv first, config.json for the rest."""
    out, seen = [], set()
    agg = os.path.join(S, "sat2_hparam_sweep", "aggregated.csv")
    if os.path.exists(agg):
        with open(agg, newline="") as fh:
            for r in csv.DictReader(fh):
                run = os.path.basename(r["checkpoint_dir"].rstrip("/"))
                seen.add(run)
                out.append({
                    "family": "sat2_hparam_sweep", "run": run,
                    "acc": _f(r["final_val_accuracy"]),
                    "baseline": _f(r["baseline_accuracy"]),
                    "end2end": _f(r["solve_rate"]),
                    "outcome_acc": _f(r["outcome_accuracy"]),
                    "well_formed": _f(r["well_formed_rate"]),
                    "metric": "solve_rate",
                    "d_model": _f(r["d_model"]), "nhead": _f(r["nhead"]),
                    "num_layers": _f(r["num_layers"]),
                    "n_vars": _f(r["n_vars"]), "alpha": _f(r["alpha"]),
                    "m_clauses": _f(r["m_clauses"]),
                    "n_val": _f(r["n_val_instances"]),
                    "source": "aggregated.csv",
                })
    # the runs that finished after the aggregate was last written
    for cpath in sorted(glob.glob(os.path.join(S, "sat2_hparam_sweep", "*", "config.json"))):
        run = os.path.basename(os.path.dirname(cpath))
        if run in seen:
            continue
        cfg = json.load(open(cpath))
        sm = cfg.get("solve_metrics") or {}
        hpath = os.path.join(os.path.dirname(cpath), "history.json")
        acc = float("nan")
        if os.path.exists(hpath):
            hist = json.load(open(hpath))
            if hist:
                acc = _f(hist[-1].get("accuracy"))
        out.append({
            "family": "sat2_hparam_sweep", "run": run, "acc": acc,
            "baseline": _f(cfg.get("baseline_accuracy")),
            "end2end": _f(sm.get("solve_rate")),
            "outcome_acc": _f(sm.get("outcome_accuracy")),
            "well_formed": _f(sm.get("well_formed_rate")),
            "metric": "solve_rate",
            "d_model": _f(cfg.get("d_model")), "nhead": _f(cfg.get("nhead")),
            "num_layers": _f(cfg.get("num_layers")),
            "n_vars": _f(cfg.get("n_vars")), "alpha": _f(cfg.get("alpha")),
            "m_clauses": _f(cfg.get("m_clauses")),
            "n_val": _f(sm.get("n_instances")),
            "source": "config.json (not in aggregated.csv)",
        })
    return out


def pinned(rows, fields):
    """Which difficulty fields are constant. The reason step 3 cannot run."""
    out = {}
    for k in fields:
        vals = sorted({r.get(k) for r in rows if r.get(k) == r.get(k)})
        out[k] = vals
    return out


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------

def fit(rows, label):
    """log S against log acc on the runs that solve anything at all."""
    acc = np.array([r["acc"] for r in rows], float)
    s = np.array([r["end2end"] for r in rows], float)
    ok = np.isfinite(acc) & np.isfinite(s)
    solves = ok & (s > 0) & (acc > 0)
    never = ok & (s <= 0)
    res = None
    if solves.sum() >= 3:
        res = stats.linregress(np.log(acc[solves]), np.log(s[solves]))
    return {
        "label": label, "n": int(ok.sum()), "n_solve": int(solves.sum()),
        "n_never": int(never.sum()), "acc": acc, "s": s,
        "solves": solves, "never": never,
        "k": float(res.slope) if res else float("nan"),
        "se": float(res.stderr) if res else float("nan"),
        "r2": float(res.rvalue ** 2) if res else float("nan"),
        "intercept": float(res.intercept) if res else float("nan"),
        "p": float(res.pvalue) if res else float("nan"),
        "acc_range": (float(np.nanmin(acc[solves])), float(np.nanmax(acc[solves])))
        if solves.sum() else (float("nan"), float("nan")),
    }


def plot_family(fitres, rows, fname, title, ylabel):
    """The log-log fit, windowed on the runs that actually solve.

    The x window is set from the SOLVERS. A single never-solver sitting three
    decades down in accuracy would otherwise compress every real point into a
    vertical sliver and make a shallow fit look vertical; the ones outside the
    window are counted in the title instead of being silently dropped.
    """
    f = fitres
    if not f["solves"].sum():
        print("  SKIPPED %s (nothing solves)" % fname)
        return
    lo, hi = f["acc_range"]
    xlo, xhi = lo / 1.12, hi * 1.12
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    floor = float(np.nanmin(f["s"][f["solves"]])) / 2.5
    nev = f["never"] & np.isfinite(f["acc"])
    inwin = nev & (f["acc"] >= xlo)
    if inwin.sum():
        ax.scatter(f["acc"][inwin], np.full(int(inwin.sum()), floor),
                   s=26, marker="x", color=MUTED, linewidths=1.1, rasterized=True)
    ax.scatter(f["acc"][f["solves"]], f["s"][f["solves"]], s=26,
               color=SERIES[0], alpha=0.85, linewidths=0, rasterized=True)
    gx = np.linspace(lo, hi, 40)
    ax.plot(gx, np.exp(f["intercept"]) * gx ** f["k"], color=SERIES[1], lw=1.8)
    label_line(ax, gx[-1], np.exp(f["intercept"]) * gx[-1] ** f["k"],
               "$\\mathrm{acc}^{%.1f}$" % f["k"], color=SERIES[1])
    style_axes(ax, xlabel="final validation token accuracy", ylabel=ylabel,
               logx=True, logy=True)
    ax.set_xlim(xlo, xhi)
    off = int((nev & (f["acc"] < xlo)).sum())
    sub = "%d solve, %d never solve (x, at the floor)" % (f["n_solve"], int(nev.sum()))
    if off:
        sub += "; %d of those are off-scale to the left" % off
    ax.set_title("%s\n%s" % (title, sub), fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % fname)


def plot_threshold(fitres, fname, title, ylabel):
    """The full accuracy range, linear y, zeros included: is there a floor?"""
    f = fitres
    ok = np.isfinite(f["acc"]) & np.isfinite(f["s"])
    if ok.sum() < 3:
        print("  SKIPPED %s (no rows)" % fname)
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(f["acc"][f["never"]], f["s"][f["never"]], s=26, marker="x",
               color=MUTED, linewidths=1.1, rasterized=True)
    ax.scatter(f["acc"][f["solves"]], f["s"][f["solves"]], s=26,
               color=SERIES[0], alpha=0.85, linewidths=0, rasterized=True)
    if f["solves"].sum():
        edge = f["acc_range"][0]
        ax.axvline(edge, color=SERIES[1], lw=1.3, ls="--")
        label_line(ax, edge, float(np.nanmax(f["s"][ok])) * 0.92,
                   "lowest solver: %.3f" % edge, color=SERIES[1])
    style_axes(ax, xlabel="final validation token accuracy", ylabel=ylabel)
    ax.set_title(title, fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % fname)


def plot_exponent_context(fits, fname):
    """The SAT exponent next to the sudoku exponents it is supposed to join."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    rhos = sorted(SUDOKU_K)
    ns = [48 * (1 - r) for r in rhos]
    ks = [SUDOKU_K[r] for r in rhos]
    ax.scatter(ns, ks, s=30, color=SERIES[0], linewidths=0)
    # rhos are ascending, so the LAST entry is the smallest n; label the
    # right-hand end of the curve, which is the largest n
    imax = int(np.argmax(ns))
    label_line(ax, ns[imax], ks[imax], "sudoku DD", color=SERIES[0], dy=-10)
    gx = np.linspace(min(ns), max(ns), 40)
    anchor = float(np.median(np.array(ks) / np.array(ns) ** 1.84))
    ax.plot(gx, anchor * gx ** 1.84, color=MUTED, lw=1.2, ls="--")
    label_line(ax, gx[-1], anchor * gx[-1] ** 1.84, "$n^{1.84}$", color=MUTED)
    ylo, yhi = min(ks) / 3.0, max(ks) * 3.0
    for c, f in enumerate(fits):
        # a fit consistent with no relationship has no exponent worth drawing,
        # and a NEGATIVE one cannot go on a log axis at all
        if f["k"] != f["k"] or f["k"] <= 0 or f["p"] > 0.05:
            continue
        col = SERIES[(c + 1) % 3]
        ax.axhspan(max(f["k"] - f["se"], ylo), min(f["k"] + f["se"], yhi),
                   color=col, alpha=0.15, lw=0)
        ax.axhline(f["k"], color=col, lw=1.3, ls=":")
        label_line(ax, gx[-1], f["k"], f["label"], color=col)
    style_axes(ax, xlabel="blank cells $n$ (sudoku axis only)",
               ylabel="fitted exponent $k$", logx=True, logy=True)
    ax.set_ylim(ylo, yhi)
    ax.set_title("the SAT exponents have no difficulty axis to sit on",
                 fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % fname)


# --------------------------------------------------------------------------

def main():
    os.makedirs(FIGS, exist_ok=True)
    sat1, sat2 = collect_sat1(), collect_sat2()
    report = []

    for name, rows in (("sat_hparam_sweep", sat1), ("sat2_hparam_sweep", sat2)):
        nsrc = sum(1 for r in rows if r["source"].startswith("config"))
        report.append("%s: %d runs (%d read from per-run config.json because "
                      "aggregated.csv predates them)." % (name, len(rows), nsrc))
    for fam in (S + "/scale_sweep", S + "/sat_checkpoints"):
        n = sum(len(fs) for _, _, fs in os.walk(fam)) if os.path.isdir(fam) else 0
        report.append("%s: %d files on disk -- nothing to score."
                      % (os.path.basename(fam), n))

    # one CSV with every point, so the plot can be re-made without this script
    allrows = sat1 + sat2
    cols = ["family", "run", "metric", "acc", "baseline", "end2end",
            "outcome_acc", "well_formed", "d_model", "nhead", "num_layers",
            "n_vars", "alpha", "m_clauses", "n_val", "source"]
    cpath = os.path.join(OUT, "sat_points.csv")
    with open(cpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in allrows:
            w.writerow(r)
    print("wrote %s (%d rows)" % (cpath, len(allrows)))

    f1 = fit(sat1, "sat_hparam (weak probe)")
    f2 = fit(sat2, "sat2 (free generation)")

    print("\nfigures:")
    plot_family(f1, sat1, "S1_sat_hparam.png",
                "$n_{\\rm vars}=24$, $\\alpha=4.2$; single-step satisfying-completion probe",
                "satisfying-completion rate")
    plot_family(f2, sat2, "S2_sat2.png",
                "$n_{\\rm vars}=16$, $\\alpha=1.25$, $m=20$; free-generation solve rate",
                "solve rate")
    plot_threshold(f1, "S1b_sat_hparam_threshold.png",
                   "sat_hparam: solvers and never-solvers overlap completely",
                   "satisfying-completion rate")
    plot_threshold(f2, "S2b_sat2_threshold.png",
                   "sat2: nothing below 0.51 token accuracy ever solves",
                   "solve rate")
    plot_exponent_context([f1, f2], "S3_exponent_context.png")

    report.append("")
    for f in (f1, f2):
        report.append(
            "%s: %d runs, %d solve something, %d never solve at all. "
            "Power-law fit on the %d that do: S ~ acc^%.2f, standard error "
            "%.2f, R2 %.3f, p %.2g, over accuracy %.4f to %.4f."
            % (f["label"], f["n"], f["n_solve"], f["n_never"], f["n_solve"],
               f["k"], f["se"], f["r2"], f["p"], f["acc_range"][0], f["acc_range"][1]))
        nz = f["never"] & np.isfinite(f["acc"])
        if nz.sum():
            report.append(
                "   the %d never-solvers span accuracy %.4f to %.4f; the "
                "solvers start at %.4f."
                % (int(nz.sum()), float(np.nanmin(f["acc"][nz])),
                   float(np.nanmax(f["acc"][nz])), f["acc_range"][0]))
        report.append(
            "   sudoku's fit on the same kind of plot has R2 ~ 0.92. This one "
            "has R2 %.3f, so %s"
            % (f["r2"],
               "the power law does NOT transfer." if f["r2"] < 0.5
               else "the power law does transfer."))

    # the difficulty axes, checked rather than assumed
    report.append("")
    report.append("DIFFICULTY AXES, checked field by field:")
    for name, rows, fields in (("sat_hparam_sweep", sat1, ["n_vars", "alpha"]),
                               ("sat2_hparam_sweep", sat2,
                                ["n_vars", "alpha", "m_clauses"])):
        p = pinned(rows, fields)
        report.append("  %s: %s" % (name, "; ".join(
            "%s = %s" % (k, v[0] if len(v) == 1 else "VARIES %s" % v)
            for k, v in p.items())))
    report.append("  Every difficulty field is PINNED in both sweeps. There is "
                  "no bin to fit a separate exponent in, so step 3 of the "
                  "request -- exponent per difficulty bin against mean trace "
                  "length -- has no data behind it and no job was submitted "
                  "for it.")
    report.append("")
    report.append(
        "WHAT STEP 3 WOULD COST, since it was asked for and cannot be done "
        "from disk. The sudoku exponent-vs-length curve needed 14 difficulty "
        "bins x 400 instances scored by free generation, per run. For SAT the "
        "difficulty axis would be alpha (clause ratio) or n_vars, and the "
        "existing checkpoints can be re-scored without retraining: "
        "%d sat2 checkpoints x (say) 8 alpha bins x 300 instances of greedy "
        "rollout. The sudoku equivalent -- 7 models x 14 bins x 400 -- took "
        "8m55s on one GPU, so this is of order 1-2 GPU-hours plus a generator "
        "that makes k-SAT instances at a swept alpha, which does not exist "
        "yet and is the real cost. The 51 sat_hparam checkpoints are a worse "
        "bet: 12 of them have been deleted, and the metric that family "
        "records is the weak single-step probe, not free generation."
        % len([r for r in sat2 if r["end2end"] == r["end2end"]]))

    text = "\n".join(report) + "\n"
    rpath = os.path.join(OUT, "REPORT.txt")
    with open(rpath, "w") as fh:
        fh.write(text)
    print("\n" + text)
    print("wrote %s" % rpath)


if __name__ == "__main__":
    main()
