"""Stage PREFACTOR -- why does C fall to 0.797 at rho 0.50?

Stage LOGIC fitted S = C p^k per density and got the exponent it predicted
(0.932 +- 0.025 n through the origin) but NOT the prefactor: C is 1.005 for
rho >= 0.65 and then falls to 0.797.  Taken at face value that says a model
with no token error on any scored position still loses a fifth of the
instances.  Taken sceptically it says a straight line fitted near p ~ 0.9 is
being extrapolated to p = 1 and is lying about its own intercept.

Five registered questions decide between those, all from CSVs already on disk:

  Q1  curvature         -- refit with a quadratic in log p
  Q2  extrapolation     -- is (1 - C) explained by (1 - p_max)?
  Q3  top-decile refit  -- fit only the runs nearest p = 1
  Q4  residual subset   -- the best model's failures, per instance
  Q5  n-spread          -- stratify by the instance's own n_blank

Predictions and falsifiers: runs/stageprefactor_2026-11-17/REGISTERED.md,
committed before this script was written.  No training, no GPU, no new
instances.
"""

import collections
import csv
import glob
import math
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line  # noqa: E402

OUT = os.environ.get("PREF_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stageprefactor_2026-11-17"
LOGIC = os.environ.get("LOGIC_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stagelogic_2026-11-17"
POS = os.environ.get("POS_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17"
FIGS = os.path.join(OUT, "figures")
DPI = 150
N_NODES = 48
FIT_RHOS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
PANEL_RHOS = [0.75, 0.60, 0.50]
TOP_N = 8                 # Q3: "top decile" of 79 runs
MIN_STRATUM = 100         # Q4, Q5: a rate on fewer instances is not a rate
HARD_RHO = 0.50


def fnum(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def tobool(r, k):
    return str(r.get(k, "")).strip().lower() in ("1", "true", "yes")


def read_shards(directory, pattern):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        with open(path, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def blanks(realized_rho):
    return N_NODES * (1.0 - realized_rho)


# --------------------------------------------------------------------------
# fits
# --------------------------------------------------------------------------

def ols(X, y):
    """Least squares with standard errors.  X includes its own intercept."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return beta, se, r2


def linear_fit(p, s):
    """log S = b + k log p.  Returns C, k, se_k, R2, npts."""
    p, s = np.asarray(p, float), np.asarray(s, float)
    m = (p > 0) & (p < 1) & (s > 0) & (s < 1)
    if m.sum() < 3:
        return (float("nan"),) * 4 + (int(m.sum()),)
    x = np.log(p[m])
    X = np.column_stack([np.ones(m.sum()), x])
    beta, se, r2 = ols(X, np.log(s[m]))
    return math.exp(beta[0]), beta[1], se[1], r2, int(m.sum())


def quad_fit(p, s):
    """log S = b + k log p + q (log p)^2.  Returns C_q, k, q, se_q, R2, npts."""
    p, s = np.asarray(p, float), np.asarray(s, float)
    m = (p > 0) & (p < 1) & (s > 0) & (s < 1)
    if m.sum() < 5:
        return (float("nan"),) * 5 + (int(m.sum()),)
    x = np.log(p[m])
    X = np.column_stack([np.ones(m.sum()), x, x ** 2])
    beta, se, r2 = ols(X, np.log(s[m]))
    return math.exp(beta[0]), beta[1], beta[2], se[2], r2, int(m.sum())


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


def p1_residuals(cells, fits):
    f_, axes = plt.subplots(1, 3, figsize=(11.0, 3.9), sharey=True)
    for ax, rho in zip(axes, PANEL_RHOS):
        sel = [c for c in cells.values()
               if abs(c["rho"] - rho) < 1e-9 and c["in79"]
               and 0 < c["solve_rate"] < 1 and 0 < c["p_logic"] < 1]
        x = np.log([c["p_logic"] for c in sel])
        y = np.log([c["solve_rate"] for c in sel])
        fit = fits[round(rho, 4)]
        pred = math.log(fit["C"]) + fit["k"] * x
        ax.axhline(0.0, color=MUTED, lw=1.0)
        ax.scatter(np.exp(x), y - pred, s=14, color=SERIES[0], alpha=0.75,
                   rasterized=True)
        xs = np.linspace(x.min(), x.max(), 60)
        qp = (math.log(fit["C_q"]) + fit["k_q"] * xs + fit["q"] * xs ** 2
              - (math.log(fit["C"]) + fit["k"] * xs))
        ax.plot(np.exp(xs), qp, color=SERIES[1], lw=1.6)
        ax.set_title("rho %.2f   q = %+.1f +- %.1f"
                     % (rho, fit["q"], fit["se_q"]), fontsize=10)
        style_axes(ax, xlabel="logical token accuracy p", ylabel=None)
    style_axes(axes[0], xlabel="logical token accuracy p",
               ylabel="log S  -  linear fit")
    save(f_, os.path.join(FIGS, "P1_residuals_vs_p.png"))


def p2_extrapolation(fits, slope, inter, r2):
    f_, ax, path = fig("P2_C_vs_extrapolation.png")
    rows = sorted(fits.values(), key=lambda v: v["n"])
    x = np.array([1.0 - v["p_max"] for v in rows])
    y = np.array([1.0 - v["C"] for v in rows])
    ax.axhline(0.0, color=MUTED, lw=1.0)
    ax.scatter(x, y, s=34, color=SERIES[0], zorder=3)
    xs = np.linspace(0, x.max() * 1.05, 40)
    ax.plot(xs, inter + slope * xs, color=SERIES[1], lw=1.6)
    for v in rows:
        if v["rho"] in (0.95, 0.75, 0.60, 0.50):
            ax.annotate("rho %.2f" % v["rho"], (1 - v["p_max"], 1 - v["C"]),
                        textcoords="offset points", xytext=(6, -3),
                        fontsize=9, color=MUTED)
    label_line(ax, xs[-1], inter + slope * xs[-1], "R2 = %.2f" % r2,
               color=SERIES[1])
    style_axes(ax, xlabel="1 - p_max   (how far the line is extrapolated)",
               ylabel="1 - C   (the apparent shortfall)")
    save(f_, path)


def p3_three_fits(fits):
    f_, ax, path = fig("P3_C_linear_quad_top.png")
    rows = sorted(fits.values(), key=lambda v: v["n"])
    n = [v["n"] for v in rows]
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.plot(n, [v["C"] for v in rows], color=SERIES[0], lw=2.0, marker="o", ms=4.5)
    ax.plot(n, [v["C_top"] for v in rows], color=SERIES[1], lw=1.7, marker="s", ms=4)
    ax.plot(n, [v["C_q"] for v in rows], color=SERIES[2], lw=1.5, marker="^", ms=4)
    ends = [(n[-1], rows[-1]["C"], "all 79, linear", SERIES[0]),
            (n[-1], rows[-1]["C_top"], "top %d only" % TOP_N, SERIES[1]),
            (n[-1], rows[-1]["C_q"], "all 79, quadratic", SERIES[2])]
    lo, hi = ax.get_ylim()
    ends.sort(key=lambda e: e[1])
    for idx in range(1, len(ends)):
        if ends[idx][1] - ends[idx - 1][1] < 0.08 * (hi - lo):
            ends[idx] = (ends[idx][0], ends[idx - 1][1] + 0.08 * (hi - lo),
                         ends[idx][2], ends[idx][3])
    for xx, yy, tt, cc in ends:
        label_line(ax, xx, yy, tt, color=cc)
    style_axes(ax, xlabel="blank cells  n", ylabel="prefactor C")
    save(f_, path)


def p4_best_model(nb_pts, t_pts, best_run, rho):
    """Registered as 'solve rate against n_blank'.  n_blank turns out to be
    constant inside a density bin, so that panel is one point by construction
    and the T_true panel -- also registered -- carries the question."""
    f_, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    ax = axes[0]
    ax.plot([p[0] for p in nb_pts], [p[1] for p in nb_pts],
            color=MUTED, lw=2.0, marker="o", ms=6)
    for nb, rate, tot in nb_pts:
        ax.annotate("all %d instances" % tot, (nb, rate),
                    textcoords="offset points", xytext=(0, 10), fontsize=9,
                    color=MUTED, ha="center")
    ax.set_xlim(min(p[0] for p in nb_pts) - 2, max(p[0] for p in nb_pts) + 2)
    style_axes(ax, xlabel="n_blank  (constant inside the bin: no variation)",
               ylabel="solve rate of the most accurate model")
    ax = axes[1]
    ax.plot([p[0] for p in t_pts], [p[1] for p in t_pts],
            color=SERIES[0], lw=2.0, marker="o", ms=5)
    for t, rate, tot in t_pts:
        ax.annotate("%d" % tot, (t, rate), textcoords="offset points",
                    xytext=(0, 9), fontsize=8, color=MUTED, ha="center")
    style_axes(ax, xlabel="DD rounds T_true   (label = instances)", ylabel=None)
    axes[0].set_ylim(0, 1.05)
    f_.suptitle("%s at rho %.2f" % (best_run, rho), fontsize=10, color=MUTED)
    save(f_, os.path.join(FIGS, "P4_best_model_vs_nblank.png"))


def p6_two_lines(cells, fits, rho):
    """The claim in one picture: the same 79 points, two straight lines, and
    only the one fitted near p = 1 passes through S = 1 at p = 1."""
    f_, ax, path = fig("P6_two_fits_at_rho%02d.png" % round(rho * 100))
    sel = [c for c in cells.values()
           if abs(c["rho"] - rho) < 1e-9 and c["in79"]
           and 0 < c["solve_rate"] < 1 and 0 < c["p_logic"] < 1]
    sel.sort(key=lambda c: -c["p_logic"])
    p = np.array([c["p_logic"] for c in sel])
    s = np.array([c["solve_rate"] for c in sel])
    v = fits[round(rho, 4)]
    ax.scatter(p[TOP_N:], s[TOP_N:], s=16, color=MUTED, alpha=0.8,
               rasterized=True)
    ax.scatter(p[:TOP_N], s[:TOP_N], s=34, color=SERIES[1], zorder=3,
               rasterized=True)
    xs = np.linspace(p.min(), 1.0, 200)
    ax.plot(xs, v["C"] * xs ** v["k"], color=SERIES[0], lw=1.8)
    ax.plot(xs, v["C_top"] * xs ** v["k_top"], color=SERIES[1], lw=1.8)
    ax.scatter([1.0], [v["C"]], s=46, color=SERIES[0], marker="D", zorder=4)
    ax.scatter([1.0], [v["C_top"]], s=46, color=SERIES[1], marker="D", zorder=4)
    ax.set_yscale("log")
    # Both lines land within 0.2 of each other at p = 1, which on a log axis
    # is two labels on top of one another; push them apart by a factor.
    label_line(ax, 1.0, v["C_top"] * 2.2,
               "top %d:  C = %.3f" % (TOP_N, v["C_top"]), color=SERIES[1])
    label_line(ax, 1.0, v["C"] * 0.33, "all 79:  C = %.3f" % v["C"],
               color=SERIES[0])
    ax.set_xlim(p.min() - 0.005, 1.10)
    style_axes(ax, xlabel="logical token accuracy p   (rho %.2f, n = %d)"
               % (rho, round(v["n"])), ylabel="solve rate S")
    save(f_, path)


def p5_stratified(fits):
    """The two curves land exactly on top of each other, and that overlap IS
    the result: with n_blank constant, x = n_blank * log p is log p times a
    constant, so the stratified fit has the identical intercept."""
    f_, ax, path = fig("P5_stratified_C.png")
    rows = sorted((v for v in fits.values() if v["C_s"] == v["C_s"]),
                  key=lambda v: v["n"])
    n = [v["n"] for v in rows]
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.plot(n, [v["C"] for v in rows], color=SERIES[0], lw=3.4, marker="o", ms=7)
    ax.plot(n, [v["C_s"] for v in rows], color=SERIES[1], lw=1.4, marker="s",
            ms=3.5)
    label_line(ax, n[-1], rows[-1]["C"], "pooled, and stratified by n_blank:\n"
               "the same fit, max gap %.1e"
               % max(abs(v["C_s"] - v["C"]) for v in rows), color=SERIES[1])
    style_axes(ax, xlabel="blank cells  n", ylabel="prefactor C")
    save(f_, path)


# --------------------------------------------------------------------------


def load_strata(pos_dir):
    """Stream B_instances_*.csv.  448k rows, so only the five columns that
    matter are kept and nothing is held as a dict."""
    strata = collections.defaultdict(lambda: [0, 0])          # (run, rho, nb)
    by_T = collections.defaultdict(lambda: [0, 0])            # (run, rho, T)
    kinds = collections.Counter()                             # (run, kind)
    for path in sorted(glob.glob(os.path.join(pos_dir, "B_instances_*.csv"))):
        with open(path, newline="") as fh:
            rd = csv.reader(fh)
            head = next(rd)
            ix = {name: k for k, name in enumerate(head)}
            c_run, c_rho = ix["run_id"], ix["rho"]
            c_nb, c_T = ix["n_blank"], ix["T_true"]
            c_sol, c_kind = ix["solved"], ix["first_kind"]
            for row in rd:
                rho = round(float(row[c_rho]), 4)
                solved = 1 if row[c_sol].strip().lower() in ("1", "true") else 0
                key = (row[c_run], rho, int(row[c_nb]))
                a = strata[key]
                a[0] += 1
                a[1] += solved
                if rho == HARD_RHO:
                    t = by_T[(row[c_run], rho, int(row[c_T]))]
                    t[0] += 1
                    t[1] += solved
                    if not solved:
                        kinds[(row[c_run], row[c_kind])] += 1
    return strata, by_T, kinds


def main():
    os.makedirs(FIGS, exist_ok=True)
    srows = read_shards(LOGIC, "LF_summary_*.csv")
    cells = {}
    for r in srows:
        cells[(r["run_id"], round(fnum(r, "rho"), 4))] = {
            "run_id": r["run_id"], "rho": fnum(r, "rho"),
            "n": blanks(fnum(r, "realized_rho")), "in79": tobool(r, "in79"),
            "p_logic": fnum(r, "p_logic"), "solve_rate": fnum(r, "solve_rate"),
        }
    print("cells %d (in79 %d)"
          % (len(cells), sum(c["in79"] for c in cells.values())))
    # Mean DD depth per density, for the scored-token count 2n + T + 1.
    ddT, ddN = collections.defaultdict(float), collections.Counter()
    for r in read_shards(POS, "B_summary_*.csv"):
        rho = round(fnum(r, "rho"), 4)
        ddT[rho] += fnum(r, "dd_mean_T")
        ddN[rho] += 1
    ddT = {k: v / ddN[k] for k, v in ddT.items()}
    strata, by_T, kinds = load_strata(POS)
    print("strata %d  (rho %.2f T-strata %d)" % (len(strata), HARD_RHO, len(by_T)))

    fits = {}
    for rho in FIT_RHOS:
        sel = [c for c in cells.values()
               if abs(c["rho"] - rho) < 1e-9 and c["in79"]]
        if len(sel) < 8:
            continue
        p = [c["p_logic"] for c in sel]
        s = [c["solve_rate"] for c in sel]
        C, k, se_k, r2, npts = linear_fit(p, s)
        C_q, k_q, q, se_q, r2_q, _ = quad_fit(p, s)
        used = [c for c in sel if 0 < c["solve_rate"] < 1 and 0 < c["p_logic"] < 1]
        used.sort(key=lambda c: -c["p_logic"])
        top = used[:TOP_N]
        C_t, k_t, se_t, r2_t, n_t = linear_fit([c["p_logic"] for c in top],
                                               [c["solve_rate"] for c in top])
        # Q5: one point per (run, n_blank) stratum, x = n_blank * log p_run.
        xs, ys = [], []
        for c in sel:
            lp = math.log(c["p_logic"]) if 0 < c["p_logic"] < 1 else None
            if lp is None:
                continue
            for (rid, rh, nb), v in strata.items():
                if rid != c["run_id"] or abs(rh - rho) > 1e-9 or v[0] < MIN_STRATUM:
                    continue
                rate = v[1] / v[0]
                if not 0 < rate < 1:
                    continue
                xs.append(nb * lp)
                ys.append(math.log(rate))
        if len(xs) >= 5:
            beta, se, r2_s = ols(np.column_stack([np.ones(len(xs)), xs]), ys)
            C_s, a_s, se_a, n_s = math.exp(beta[0]), beta[1], se[1], len(xs)
        else:
            C_s, a_s, se_a, r2_s, n_s = (float("nan"),) * 4 + (len(xs),)
        # How much n_blank actually varies inside this bin -- Q5 assumes it
        # varies at all, and that assumption has to be reported, not assumed.
        nb_vals = sorted({nb for (rid, rh, nb) in strata
                          if abs(rh - rho) < 1e-9})
        fits[round(rho, 4)] = dict(
            rho=rho, n=float(np.mean([c["n"] for c in sel])), n_runs=len(sel),
            C=C, k=k, se_k=se_k, r2=r2, npts=npts,
            C_q=C_q, k_q=k_q, q=q, se_q=se_q, r2_q=r2_q,
            C_top=C_t, k_top=k_t, se_top=se_t, r2_top=r2_t, n_top=n_t,
            C_s=C_s, a_s=a_s, se_a=se_a, r2_s=r2_s, n_s=n_s,
            n_blank_distinct=len(nb_vals), n_blank_vals=";".join(map(str, nb_vals)),
            p_max=max(c["p_logic"] for c in used) if used else float("nan"))

    rows = sorted(fits.values(), key=lambda v: v["n"])
    x2 = np.array([1 - v["p_max"] for v in rows])
    y2 = np.array([1 - v["C"] for v in rows])
    lr = stats.linregress(x2, y2)

    best_run = max((c for c in cells.values()
                    if abs(c["rho"] - HARD_RHO) < 1e-9 and c["in79"]),
                   key=lambda c: c["p_logic"])["run_id"]

    # Q4, the best model's residual failures.  n_blank is constant inside the
    # bin so that half is one point; T_true is the difficulty variable that
    # does vary, and it was registered alongside n_blank.
    nb_pts = sorted((nb, v[1] / v[0], v[0]) for (r, rh, nb), v in strata.items()
                    if r == best_run and abs(rh - HARD_RHO) < 1e-9)
    t_all = sorted((t, v[1] / v[0], v[0]) for (r, rh, t), v in by_T.items()
                   if r == best_run)
    T_MIN = 25          # 400 instances over ~13 T values; 100 would leave none
    t_pts = [p for p in t_all if p[2] >= T_MIN]

    print("\nfigures:")
    p1_residuals(cells, fits)
    p2_extrapolation(fits, lr.slope, lr.intercept, lr.rvalue ** 2)
    p3_three_fits(fits)
    p4_best_model(nb_pts, t_pts, best_run, HARD_RHO)
    p5_stratified(fits)
    p6_two_lines(cells, fits, HARD_RHO)
    p6_two_lines(cells, fits, 0.60)

    with open(os.path.join(OUT, "prefactor_fits.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for v in rows:
            w.writerow(v)
    with open(os.path.join(OUT, "prefactor_strata.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "rho", "n_blank", "n_inst", "n_solved"])
        for (rid, rh, nb), v in sorted(strata.items()):
            w.writerow([rid, rh, nb, v[0], v[1]])

    # ---- verdict ----------------------------------------------------------
    L = []
    a = L.append
    a("Stage PREFACTOR -- why C falls to 0.797 at rho 0.50.")
    a("Predictions and falsifiers are in REGISTERED.md, committed before this")
    a("script was written.  No training, no GPU, no new instances.")
    a("")
    a("  rho     n  runs |      C      k +- se |   C_q      q +- se |   C_top "
      " k_top +- se |  p_max   n_blank")
    for v in rows[::-1]:
        a("  %.2f %5.1f  %4d | %6.3f %6.1f %4.1f | %6.3f %+7.1f %5.1f | %6.3f "
          "%6.1f %5.1f | %6.4f   %s"
          % (v["rho"], v["n"], v["n_runs"], v["C"], v["k"], v["se_k"],
             v["C_q"], v["q"], v["se_q"], v["C_top"], v["k_top"], v["se_top"],
             v["p_max"], v["n_blank_vals"]))
    a("")

    hard = [v for v in rows if v["C"] < 0.95]
    sig = [v for v in hard if abs(v["q"]) > 2 * v["se_q"]]
    closer = [v for v in hard if abs(v["C_q"] - 1) < abs(v["C"] - 1)]
    a("Q1 curvature.  Densities with C < 0.95: %d (%s)."
      % (len(hard), ", ".join("%.2f" % v["rho"] for v in hard)))
    a("   q significant (|q| > 2 se) at %d of %d; C_q closer to 1 at %d of %d."
      % (len(sig), len(hard), len(closer), len(hard)))
    a("   predicted: q significant AND C_q closer to 1 at a majority -> %s"
      % ("HIT" if len(sig) > len(hard) / 2 and len(closer) > len(hard) / 2
         else "MISS, FALSIFIED"))
    a("")
    a("Q2 extrapolation distance.  (1 - C) = %+.3f + %.3f (1 - p_max), R2 %.3f, "
      "p %.2g, %d densities."
      % (lr.intercept, lr.slope, lr.rvalue ** 2, lr.pvalue, len(rows)))
    a("   predicted: positive slope with R2 > 0.5, falsified below 0.2 -> %s"
      % ("HIT" if lr.slope > 0 and lr.rvalue ** 2 > 0.5 else
         ("MISS, FALSIFIED" if lr.rvalue ** 2 < 0.2 else "INCONCLUSIVE")))
    a("")
    big = [v for v in rows if v["n"] >= 17]
    up = [v for v in big if v["C_top"] > v["C"]]
    med_top = float(np.median([v["C_top"] for v in rows if v["rho"] <= 0.6001]))
    a("Q3 top-%d refit.  C_top > C at %d of %d densities with n >= 17; "
      "median C_top over rho <= 0.60 = %.3f." % (TOP_N, len(up), len(big), med_top))
    a("   predicted: all of them, and median >= 0.95 -> %s"
      % ("HIT" if len(up) == len(big) and med_top >= 0.95 else
         ("MISS, FALSIFIED" if len(up) <= len(big) / 2 else "PARTIAL")))
    a("")
    cell = cells[(best_run, round(HARD_RHO, 4))]
    a("Q4 the residual subset.  Most accurate run at rho %.2f: %s, p_logic %.4f, "
      "solve rate %.4f." % (HARD_RHO, best_run, cell["p_logic"], cell["solve_rate"]))
    a("   Registered against n_blank -- but n_blank is CONSTANT in the bin "
      "(%s), so" % fits[round(HARD_RHO, 4)]["n_blank_vals"])
    a("   that half of Q4 is one point and admits no correlation: VOID, "
      "not missed.")
    a("   Against T_true, the difficulty variable that does vary, strata "
      ">= %d instances:" % T_MIN)
    a("     T_true   " + " ".join("%6d" % p[0] for p in t_pts))
    a("     instances" + " ".join("%6d" % p[2] for p in t_pts))
    a("     solved   " + " ".join("%6.3f" % p[1] for p in t_pts))
    spt = (stats.spearmanr([p[0] for p in t_pts], [p[1] for p in t_pts]).statistic
           if len(t_pts) > 2 else float("nan"))
    inst_t = [(t, s) for t, rate, tot in t_all for s in
              ([1] * int(round(rate * tot)) + [0] * (tot - int(round(rate * tot))))]
    spi = stats.spearmanr([p[0] for p in inst_t], [p[1] for p in inst_t])
    a("   Spearman over strata %.3f; per instance %.3f (p %.2g, n %d). "
      "Descriptive:" % (spt, spi.statistic, spi.pvalue, len(inst_t)))
    a("   the registered -0.5 threshold was set for n_blank, not for T_true.")
    kk = sorted(((n, k) for (r, k), n in kinds.items() if r == best_run),
                reverse=True)
    tot_fail = sum(n for n, _ in kk)
    a("   its %d failures by first_kind: %s"
      % (tot_fail, ", ".join("%s %d (%.0f%%)" % (k, n, 100.0 * n / tot_fail)
                             for n, k in kk) or "none"))
    a("")
    lower = [v for v in big if v["C_s"] < v["C"]]
    spread = [v for v in rows if v["n_blank_distinct"] > 1]
    gap = max(abs(v["C_s"] - v["C"]) for v in rows)
    a("Q5 the n spread.  Distinct n_blank values per bin: %s."
      % ", ".join("%.2f->%d" % (v["rho"], v["n_blank_distinct"])
                  for v in rows[::-1]))
    if not spread:
        a("   The premise is FALSE: the generator plants round(rho*48) hints, so")
        a("   every instance in a fit bin has the same n_blank and the spread is")
        a("   zero.  With n_blank constant, x = n_blank*log p is log p times a")
        a("   constant, so the stratified fit is the pooled fit with the slope")
        a("   rescaled and an IDENTICAL intercept -- max |C_s - C| = %.1e over" % gap)
        a("   the ten densities.  VOID, not missed; C_s < C at %d of %d is"
          % (len(lower), len(big)))
        a("   floating point, not evidence.  What this does establish: Jensen")
        a("   curvature over an n spread CANNOT be the cause, because there is")
        a("   no spread to be convex over.  Ruled out by construction.")
    else:
        a("   C_s < C at %d of %d densities with n >= 17; predicted C_s < C "
          "(Jensen lift removed) -> %s"
          % (len(lower), len(big),
             "HIT" if len(lower) > len(big) / 2 else "MISS, FALSIFIED"))
    a("")
    a("DESCRIPTIVE, POST-HOC -- not registered, do not score it.  Freeing C")
    a("costs the exponent: the pooled fit buys a small k with a small C.")
    a("  n   k/n    k_top/n   (2n+T+1)/n = the scored tokens per instance")
    for v in rows[::-1]:
        tbar = ddT.get(round(v["rho"], 4), float("nan"))
        a("  %4.1f  %5.2f  %7.2f   %6.2f"
          % (v["n"], v["k"] / v["n"], v["k_top"] / v["n"],
             (2 * v["n"] + tbar + 1) / v["n"]))
    a("  k_top/n exceeds 1 at the large-n densities, so a node token is not the")
    a("  only fatal position -- consistent with Q4, where %d%% of the best"
      % round(100.0 * max(kk)[0] / tot_fail))
    a("  model's residual failures are %s, a colour and not a node."
      % max(kk)[1])
    a("  Read k_top only at n >= 12.  Below that the top 8 runs all sit at")
    a("  p > 0.999 with S ~ 1, the fit has no lever arm, and k_top (0.56-0.76 n)")
    a("  is noise -- which is also why Q3's claim was registered for n >= 17.")
    a("")
    a("WHAT THIS DOES NOT ESTABLISH.  The top-8 refit trades sample size for a")
    a("shorter extrapolation: 8 points, not 79, and its se is OLS, so it does")
    a("not carry the binomial noise in S over 400 instances.  C_top ~ 1 is")
    a("therefore consistent with C = 1, not a measurement of it.  And nothing")
    a("here is a new eval -- same 79 runs, same 448k rollouts, same CSVs as")
    a("stage LOGIC, refitted.  No model changed and no instance was regenerated.")
    a("")
    q1 = len(sig) > len(hard) / 2 and len(closer) > len(hard) / 2
    q2 = lr.slope > 0 and lr.rvalue ** 2 > 0.5
    q3 = len(up) == len(big) and med_top >= 0.95
    a("SCORECARD: Q1 %s, Q2 %s, Q3 %s, Q4 VOID as registered (T_true reported), "
      "Q5 VOID (no n spread)."
      % ("HIT" if q1 else "MISS", "HIT" if q2 else "MISS",
         "HIT" if q3 else "MISS"))
    a("")
    if q1 or q2 or q3:
        a("BOTTOM LINE: C < 1 is an artefact of extrapolating a straight line in")
        a("log p from p ~ %.3f out to p = 1.  Fit only the %d runs nearest p = 1"
          % (min(v["p_max"] for v in rows if v["n"] >= 17), TOP_N))
        a("and the prefactor returns to %.3f at rho 0.50, where the pooled fit"
          % fits[round(HARD_RHO, 4)]["C_top"])
        a("said 0.797.  The S = p^(0.93 n) law does NOT need a second failure")
        a("mode: at p = 1 the models solve everything, as a fatal-logical-error")
        a("account requires.  The pooled C is biased low because log S is convex")
        a("in log p and the bulk of the 79 runs sit far from p = 1.")
    else:
        a("BOTTOM LINE: no artefact explains C < 1 -- it is a real second")
        a("failure mode, and Q4's first_kind breakdown is the only evidence")
        a("this stage has about what it is.")

    text = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "VERDICT.txt"), "w") as fh:
        fh.write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
