"""Stage EXPONENT -- where do the k fatal positions live?

Finding 95 closed the prefactor: S = p_logic^k with no prefactor, once k is
fitted where the extrapolation to p = 1 is short.  What it left open is k.
Fitted that way k_top/n runs 1.11-1.26 at n >= 17 -- above n, below the
P = 2n + T + 1 positions the reasoning section actually holds.

Three accounts, and this script separates them:
  (A) colour positions are independently fatal          -> k -> 2n
  (B) round markers / the terminal are fatal            -> k ~ n + T + 1
  (C) k is not a position count.  It is
      (-log S)/(-log p_logic), and p_logic averages over ALL P positions
      including ones whose errors never kill a rollout, so non-fatal error
      mass in the denominator inflates k above the number of fatal positions.

Predictions X1-X6 + D1 are in REGISTERED.md, committed at 30937f9 before this
file existed.  No training, no GPU, no new instances: a refit of stage LOGIC's
and stage POS's CSVs.  Runs on a login node in a few minutes.

Everything is per-position LOGICAL error, e_c = 1 - p_logic_c, so finding 94's
mis-scoring of legal within-round reorderings is already removed and no result
here can be bought by it.
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
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line  # noqa: E402

OUT = os.environ.get("EXP_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stageexp_2026-11-17"
LOGIC = os.environ.get("LOGIC_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stagelogic_2026-11-17"
POS = os.environ.get("POS_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17"
FIGS = os.path.join(OUT, "figures")
DPI = 150
FIT_RHOS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
HARD = [r for r in FIT_RHOS if r <= 0.75]          # the n >= 12 densities
CLASSES = ["node", "color", "round", "term"]
CCOL = {"node": SERIES[0], "color": SERIES[1], "round": SERIES[2],
        "term": MUTED}


def fnum(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def tobool(r, k):
    return str(r.get(k, "")).strip().lower() in ("1", "true", "yes")


def read_shards(directory, pattern, keep=None):
    """Stream the shards, keeping only `keep` columns.  B_instances is 448,000
    rows x 26 columns; holding all of it as dicts is what would make this need
    a job instead of a login node."""
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({k: r[k] for k in keep} if keep else dict(r))
    return rows


def med(xs):
    xs = [x for x in xs if x == x]
    return float(np.median(xs)) if xs else float("nan")


def spread(xs):
    xs = [x for x in xs if x == x]
    return (min(xs), max(xs)) if xs else (float("nan"), float("nan"))


def ols0(x, y):
    """OLS through the origin.  Returns slope, its standard error, the
    CENTERED R^2 and the uncentered one.  Both are reported because a
    no-intercept fit's uncentered R^2 is flattering by construction, and 95 is
    the standing lesson that a number read off a fitted line can be a property
    of the fit; the centered one is what the verdict scores."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3 or not np.any(x):
        return (float("nan"),) * 4 + (len(x),)
    a = float(np.sum(x * y) / np.sum(x * x))
    res = y - a * x
    dof = max(len(x) - 1, 1)
    se = float(math.sqrt(np.sum(res ** 2) / dof / np.sum(x * x)))
    ss_res = float(np.sum(res ** 2))
    r2_c = 1.0 - ss_res / float(np.sum((y - y.mean()) ** 2)) \
        if np.sum((y - y.mean()) ** 2) > 0 else float("nan")
    r2_u = 1.0 - ss_res / float(np.sum(y ** 2)) \
        if np.sum(y ** 2) > 0 else float("nan")
    return a, se, r2_c, r2_u, len(x)


def wilson(k, n, z=1.96):
    """95% Wilson interval.  X5's node_invalid bar is cleared on 235 rollouts;
    a bar cleared inside its own interval is not cleared."""
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fig(name, w=7.4, h=4.4):
    f_, ax = plt.subplots(figsize=(w, h))
    return f_, ax, os.path.join(FIGS, name)


def save(f_, path):
    f_.tight_layout()
    f_.savefig(path, dpi=DPI)
    plt.close(f_)
    print("  wrote %s" % os.path.basename(path))


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------

def load_cells():
    """One record per (run, rho): the class error rates, the class position
    counts PER INSTANCE, the masses n_c e_c, P, and -log S."""
    summ = read_shards(LOGIC, "LF_summary_all_*of10.csv")
    pos = read_shards(LOGIC, "L_position_all_*of10.csv", keep=[
        "run_id", "rho", "scheme", "index", "tok_type",
        "n_total", "n_logic_correct"])

    # class totals live in the scheme="all", index=-1 rows
    byclass = collections.defaultdict(dict)
    for r in pos:
        if r["scheme"] != "all" or r["index"].strip() != "-1":
            continue
        key = (r["run_id"], round(float(r["rho"]), 4))
        byclass[key][r["tok_type"]] = (fnum(r, "n_total"),
                                       fnum(r, "n_logic_correct"))

    cells, skipped = [], collections.Counter()
    for r in summ:
        rho = round(fnum(r, "rho"), 4)
        if rho not in [round(x, 4) for x in FIT_RHOS]:
            skipped["rho_not_in_fit_grid"] += 1
            continue
        if not tobool(r, "in79"):
            skipped["not_in79"] += 1
            continue
        key = (r["run_id"], rho)
        if key not in byclass or "any" not in byclass[key]:
            skipped["no_L_position_row"] += 1
            continue
        cl = byclass[key]
        ninst = fnum(r, "n_instances")
        n_any, c_any = cl["any"]
        P = n_any / ninst
        c = {
            "run_id": r["run_id"], "cell": r["cell"], "seed": int(fnum(r, "seed")),
            "arch_mode": r["arch_mode"], "params": int(fnum(r, "n_trainable_params")),
            "rho": rho, "realized_rho": fnum(r, "realized_rho"),
            "n": fnum(r, "mean_n_blank"), "T": fnum(r, "dd_mean_T"),
            "n_instances": ninst, "P": P,
            "p_logic": fnum(r, "p_logic"),
            "p_logic_node_lf": fnum(r, "p_logic_node"),
            "p_logic_color_lf": fnum(r, "p_logic_color"),
            "solve_rate": fnum(r, "solve_rate"),
        }
        # per-position error rate and per-instance position count, by class
        tot_mass = 0.0
        for k in CLASSES:
            nt, nc = cl.get(k, (0.0, 0.0))
            c["n_" + k] = nt / ninst
            c["e_" + k] = (1.0 - nc / nt) if nt else float("nan")
            m = (nt - nc) / ninst
            c["m_" + k] = m                      # = n_k * e_k, per instance
            tot_mass += m
        c["m_all"] = tot_mass
        c["P_sum"] = sum(c["n_" + k] for k in CLASSES)
        c["P_expect"] = 2.0 * c["n"] + c["T"] + 1.0
        # the quantity the whole stage turns on
        S = c["solve_rate"]
        c["usable"] = 0.0 < S < 1.0
        c["neg_log_S"] = -math.log(S) if c["usable"] else float("nan")
        c["ratio_node"] = (c["neg_log_S"] / c["m_node"]
                           if c["usable"] and c["m_node"] > 0 else float("nan"))
        c["k_emp"] = (c["neg_log_S"] / -math.log(c["p_logic"])
                      if c["usable"] and 0 < c["p_logic"] < 1 else float("nan"))
        # How much worse a node position is than the AVERAGE position.  This
        # is the dilution factor of account (C): p_logic is an average over
        # all P positions, and classes that are never wrong drag it up.
        c["dilute"] = (c["e_node"] / -math.log(c["p_logic"])
                       if 0 < c["p_logic"] < 1 else float("nan"))
        cells.append(c)
    return cells, skipped


def load_free():
    """Per-rollout free generation (stage POS) joined to the forced-set
    classification (stage LOGIC).  The two were written by different scorers;
    X6 is the claim that they agree about what the failure was."""
    inst = read_shards(POS, "B_instances_seven.csv", keep=[
        "run_id", "in79", "rho", "inst", "first_kind", "solved"])
    inst += read_shards(POS, "B_instances_rest_*of10.csv", keep=[
        "run_id", "in79", "rho", "inst", "first_kind", "solved"])
    ferr = read_shards(LOGIC, "F_errors_all_*of10.csv", keep=[
        "run_id", "rho", "inst", "first_kind", "err_in_F", "err_in_F_now"])
    fmap = {(r["run_id"], round(float(r["rho"]), 4), r["inst"]): r
            for r in ferr}
    return inst, fmap


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def rho_ticks(ax):
    """Label all ten densities, the way x2_mass's categorical axis already
    does.  Matplotlib's auto locator put an irregular 0.90 / 0.80 / 0.75 / ...
    axis under the line plots, which reads as non-uniform spacing when the
    densities are in fact evenly spaced by 0.05."""
    ax.set_xticks(list(FIT_RHOS))
    ax.set_xticklabels(["%.2f" % r for r in FIT_RHOS], fontsize=9)


def x1_ratio(cells):
    f_, ax, path = fig("X1_ratio_by_density.png", w=7.4, h=4.6)
    xs, ys = [], []
    for c in cells:
        if c["usable"]:
            xs.append(c["rho"])
            ys.append(c["ratio_node"])
    ax.scatter(xs, ys, s=9, color=MUTED, alpha=0.35, linewidths=0,
               zorder=2)
    mr = [r for r in FIT_RHOS]
    mv = [med([c["ratio_node"] for c in cells
               if c["rho"] == round(r, 4) and c["usable"]]) for r in mr]
    ax.plot(mr, mv, color=SERIES[0], lw=2.2, marker="o", ms=4.5, zorder=4)
    label_line(ax, mr[-1], mv[-1], "median", color=SERIES[0])
    ax.axhline(1.0, color=SERIES[1], lw=1.4, ls="--", zorder=3)
    ax.text(0.945, 1.03, "ratio = 1:  one fatal event per node error",
            color=SERIES[1], fontsize=9, va="bottom")
    ax.invert_xaxis()
    ax.set_ylim(0, 1.6)
    rho_ticks(ax)
    style_axes(ax, xlabel="hint density rho   (harder to the right)",
               ylabel="(-log S) / (n e_node)")
    save(f_, path)


def x2_mass(cells):
    f_, ax, path = fig("X2_class_mass.png", w=7.4, h=4.6)
    bot = np.zeros(len(FIT_RHOS))
    xs = np.arange(len(FIT_RHOS))
    for k in CLASSES:
        v = np.array([med([c["m_" + k] for c in cells
                           if c["rho"] == round(r, 4)]) for r in FIT_RHOS])
        ax.bar(xs, v, bottom=bot, width=0.72, color=CCOL[k],
               label=k, edgecolor="white", linewidth=0.6)
        bot = bot + v
    for i, k in enumerate(CLASSES):
        # color and term have no visible area: they are EXACTLY zero at the
        # median run, at every density.  Saying so on the figure, because a
        # legend entry with nothing next to it reads as a plotting bug.
        zero = k in ("color", "term")
        ax.text(1.02, 0.86 - 0.09 * i, k + ("  = 0" if zero else ""),
                transform=ax.transAxes, color=CCOL[k], fontsize=10,
                va="center")
    ax.text(0.02, 0.93, "colour and terminal positions are exactly 0 at the\n"
                        "median run at every density -- they are measured,\n"
                        "present (n and 1 per rollout), and simply never wrong",
            transform=ax.transAxes, color=MUTED, fontsize=9, va="top")
    ax.set_xticks(xs)
    ax.set_xticklabels(["%.2f" % r for r in FIT_RHOS])
    style_axes(ax, xlabel="hint density rho   (harder to the right)",
               ylabel="logical errors per rollout, by position class")
    save(f_, path)


def x3_fit(cells, fits):
    f_, ax, path = fig("X3_fit.png", w=7.4, h=4.6)
    show = [0.60, 0.55, 0.50]
    for i, r in enumerate(show):
        pts = [(c["m_node"], c["neg_log_S"]) for c in cells
               if c["rho"] == round(r, 4) and c["usable"]]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=16,
                   color=SERIES[i], alpha=0.75, linewidths=0, zorder=3)
        xm = max(p[0] for p in pts)
        a = fits[round(r, 4)][0]
        ax.plot([0, xm], [0, a * xm], color=SERIES[i], lw=1.6, zorder=2)
        label_line(ax, xm, a * xm, "rho %.2f" % r, color=SERIES[i])
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1.2, ls="--", zorder=1)
    # below the line, not on it: at 0.56*lim the text sat across the dashes
    ax.text(lim * 0.46, lim * 0.70, "y = x   (one fatal event per node error)",
            color=MUTED, fontsize=9, ha="left", va="center")
    style_axes(ax, xlabel="n e_node   (logical node errors per rollout)",
               ylabel="-log S")
    save(f_, path)


def ph_factors(rows):
    """Post-hoc: the two factors whose product is k/n.  Neither is a second
    fatal position class, which is what accounts (A) and (B) proposed."""
    f_, ax, path = fig("PH_factors.png", w=7.4, h=4.6)
    rs = [d["rho"] for d in rows]
    D = [d["dilute"] for d in rows]
    R = [d["ratio_node"] for d in rows]
    K = [d["k_emp"] / d["n"] for d in rows]
    ax.plot(rs, D, color=SERIES[0], lw=2.2, marker="o", ms=4)
    ax.plot(rs, R, color=SERIES[1], lw=2.2, marker="o", ms=4)
    ax.plot(rs, K, color=SERIES[2], lw=2.2, marker="o", ms=4)
    label_line(ax, rs[-1], D[-1], "D  dilution", color=SERIES[0])
    label_line(ax, rs[-1], R[-1], "R  fatal share", color=SERIES[1])
    label_line(ax, rs[-1], K[-1], "k/n = D x R", color=SERIES[2])
    ax.axhline(1.0, color=MUTED, lw=1.1, ls="--")
    ax.invert_xaxis()
    ax.set_ylim(0, 2.3)
    rho_ticks(ax)
    style_axes(ax, xlabel="hint density rho   (harder to the right)",
               ylabel="dimensionless")
    save(f_, path)


def x5_fatality(kinds):
    f_, ax, path = fig("X5_fatality.png", w=7.4, h=4.3)
    order = [k for k in ["color_wrong", "node_invalid", "early_term",
                         "truncated", "grammar"] if kinds.get(k, (0, 0))[1]]
    vals = [kinds[k][0] / kinds[k][1] for k in order]
    cols = [SERIES[1] if k == "color_wrong" else
            SERIES[0] if k == "node_invalid" else MUTED for k in order]
    ax.barh(np.arange(len(order)), vals, color=cols, height=0.62)
    for i, k in enumerate(order):
        ax.text(max(vals) * 0.015 + vals[i], i,
                "  %.4f   (n = %s)" % (vals[i], f"{kinds[k][1]:,}"),
                va="center", fontsize=9, color="#333333")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(0, max(max(vals) * 1.9, 0.02))
    style_axes(ax, xlabel="solve rate among rollouts whose FIRST error was of "
                          "this class")
    save(f_, path)


def x6_forced(byrho):
    f_, ax, path = fig("X6_forced.png", w=7.4, h=4.4)
    rs = [r for r in FIT_RHOS if byrho.get(round(r, 4), (0, 0))[1]]
    v = [1.0 - byrho[round(r, 4)][0] / byrho[round(r, 4)][1] for r in rs]
    ax.plot(rs, v, color=SERIES[0], lw=2.2, marker="o", ms=4.5)
    label_line(ax, rs[-1], v[-1], "unforced", color=SERIES[0])
    ax.axhline(0.90, color=SERIES[1], lw=1.4, ls="--")
    ax.text(0.945, 0.903, "registered floor 0.90", color=SERIES[1],
            fontsize=9, va="bottom")
    ax.invert_xaxis()
    ax.set_ylim(0, 1.04)
    rho_ticks(ax)
    style_axes(ax, xlabel="hint density rho   (harder to the right)",
               ylabel="share of color_wrong first errors\nat an UNFORCED node")
    save(f_, path)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    os.makedirs(FIGS, exist_ok=True)
    cells, skipped = load_cells()
    print("cells: %d   skipped: %s" % (len(cells), dict(skipped)))

    # ---------------- D1, the identity -----------------------------------
    d_P, d_sum, d_ident, d_node, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
    for c in cells:
        d_P = max(d_P, abs(c["P"] - c["P_expect"]))
        d_sum = max(d_sum, abs(c["P"] - c["P_sum"]))
        d_ident = max(d_ident, abs((1.0 - c["p_logic"]) - c["m_all"] / c["P"]))
        d_node = max(d_node, abs(c["e_node"] - (1.0 - c["p_logic_node_lf"])))
        d_color = max(d_color, abs(c["e_color"] - (1.0 - c["p_logic_color_lf"])))

    # ---------------- per-density summaries ------------------------------
    rows, fits = [], {}
    for r in FIT_RHOS:
        rr = round(r, 4)
        here = [c for c in cells if c["rho"] == rr]
        use = [c for c in here if c["usable"]]
        fits[rr] = ols0([c["m_node"] for c in use],
                        [c["neg_log_S"] for c in use])
        rows.append({
            "rho": rr,
            "n": med([c["n"] for c in here]),
            "T": med([c["T"] for c in here]),
            "P": med([c["P"] for c in here]),
            "n_runs": len(here), "n_used": len(use),
            "n_drop_S0": sum(1 for c in here if c["solve_rate"] <= 0),
            "n_drop_S1": sum(1 for c in here if c["solve_rate"] >= 1),
            "e_node": med([c["e_node"] for c in here]),
            "e_color": med([c["e_color"] for c in here]),
            "e_round": med([c["e_round"] for c in here]),
            "e_term": med([c["e_term"] for c in here]),
            "m_node": med([c["m_node"] for c in here]),
            "m_color": med([c["m_color"] for c in here]),
            "m_round": med([c["m_round"] for c in here]),
            "m_term": med([c["m_term"] for c in here]),
            "ratio_node": med([c["ratio_node"] for c in use]),
            "ratio_lo": spread([c["ratio_node"] for c in use])[0],
            "ratio_hi": spread([c["ratio_node"] for c in use])[1],
            "k_emp": med([c["k_emp"] for c in use]),
            "dilute": med([c["dilute"] for c in use]),
            "color_over_node": med([c["m_color"] / c["m_node"] for c in here
                                    if c["m_node"] > 0]),
            "rt_over_node": med([(c["m_round"] + c["m_term"]) / c["m_node"]
                                 for c in here if c["m_node"] > 0]),
            "slope": fits[rr][0], "slope_se": fits[rr][1],
            "r2_centered": fits[rr][2], "r2_uncentered": fits[rr][3],
        })

    # ---------------- X5 / X6, the per-rollout tables ---------------------
    inst, fmap = load_free()
    fitset = {round(x, 4) for x in FIT_RHOS}
    kinds = collections.defaultdict(lambda: [0, 0])        # [n_solved, n]
    cw_forced = collections.defaultdict(lambda: [0, 0])    # [n_in_F_now, n]
    unmatched, matched = 0, 0
    for r in inst:
        if str(r["in79"]).strip().lower() not in ("1", "true", "yes"):
            continue
        rho = round(float(r["rho"]), 4)
        if rho not in fitset:
            continue
        k = r["first_kind"]
        if k == "none":
            continue
        kinds[k][1] += 1
        kinds[k][0] += int(float(r["solved"]))
        if k != "color_wrong":
            continue
        f = fmap.get((r["run_id"], rho, r["inst"]))
        if f is None:
            unmatched += 1
            continue
        matched += 1
        cw_forced[rho][1] += 1
        cw_forced[rho][0] += int(float(f["err_in_F_now"]))
    kinds = {k: tuple(v) for k, v in kinds.items()}
    cw_all = [sum(v[0] for v in cw_forced.values()),
              sum(v[1] for v in cw_forced.values())]
    cw_hard = [sum(cw_forced[round(r, 4)][0] for r in HARD),
               sum(cw_forced[round(r, 4)][1] for r in HARD)]

    # ---------------- scoring --------------------------------------------
    hard = [d for d in rows if round(d["rho"], 4) in {round(x, 4) for x in HARD}]
    n_cnz = sum(1 for c in cells if c["e_color"] > 0)
    max_ec = max(c["e_color"] for c in cells)
    n_tnz = sum(1 for c in cells if c["e_term"] > 0)
    max_et = max(c["e_term"] for c in cells)

    x1_in = sum(1 for d in hard if 0.8 <= d["ratio_node"] <= 1.3)
    x1_wide = sum(1 for d in hard if 0.5 <= d["ratio_node"] <= 2.0)
    x1 = "HIT" if x1_in > len(hard) / 2 else (
        "MISS" if x1_wide <= len(hard) / 2 else "MISS")
    x2a = sum(1 for d in hard if d["color_over_node"] > 0.20)
    x2b = cw_hard[0] / cw_hard[1] if cw_hard[1] else float("nan")
    x2 = "HIT" if (x2a > len(hard) / 2 and x2b < 0.05) else "MISS"
    x3a = sum(1 for d in hard if 0.8 <= d["slope"] <= 1.3)
    x3b = sum(1 for d in hard if d["r2_centered"] > 0.80)
    x3 = "HIT" if (x3a > len(hard) / 2 and x3b > len(hard) / 2) else "MISS"
    x4n = sum(1 for d in hard if d["rt_over_node"] < 0.10)
    x4 = "HIT" if x4n > len(hard) / 2 else "MISS"
    p_cw = kinds.get("color_wrong", (0, 1))
    p_ni = kinds.get("node_invalid", (0, 1))
    r_cw = p_cw[0] / p_cw[1] if p_cw[1] else float("nan")
    r_ni = p_ni[0] / p_ni[1] if p_ni[1] else float("nan")
    x5 = "HIT" if (r_cw < 0.02 and r_ni < 0.10) else "MISS"
    x6v = 1.0 - cw_all[0] / cw_all[1] if cw_all[1] else float("nan")
    x6 = "HIT" if x6v > 0.90 else "MISS"

    # ---------------- figures --------------------------------------------
    x1_ratio(cells)
    x2_mass(cells)
    x3_fit(cells, fits)
    x5_fatality(kinds)
    x6_forced(cw_forced)
    ph_factors(rows)

    # ---------------- csv -------------------------------------------------
    with open(os.path.join(OUT, "exp_cells.csv"), "w", newline="") as fh:
        keys = list(cells[0].keys())
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for c in cells:
            w.writerow(c)
    with open(os.path.join(OUT, "exp_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for d in rows:
            w.writerow(d)

    # ---------------- verdict ---------------------------------------------
    L = []
    a = L.append
    a("Stage EXPONENT -- where do the k fatal positions live?")
    a("k_top/n is 1.11-1.26 at n >= 17 (finding 95), above n and below the")
    a("P = 2n + T + 1 positions the reasoning section holds.  (A) colour")
    a("positions fatal -> 2n.  (B) round/terminal fatal -> n + T + 1.  (C) k is")
    a("not a position count: p_logic averages over all P positions, so error")
    a("mass at positions that never kill a rollout inflates it.")
    a("Predictions in REGISTERED.md, committed 30937f9 before this script.")
    a("No training, no GPU, no new instances -- a refit of CSVs on disk.")
    a("")
    a("%d (run, rho) cells over the ten fit densities; skipped %s"
      % (len(cells), dict(skipped)))
    a("")
    a("  rho    n     P  runs used | e_node  e_color e_round e_term | "
      "n*e_node  -logS  ratio    k_emp")
    for d, rr in zip(rows, FIT_RHOS):
        here = [c for c in cells if c["rho"] == round(rr, 4) and c["usable"]]
        a("  %.2f %5.1f %5.1f  %3d %4d | %.4f  %.4f  %.4f  %.4f | "
          "%7.3f %6.3f  %.3f  %7.2f"
          % (d["rho"], d["n"], d["P"], d["n_runs"], d["n_used"],
             d["e_node"], d["e_color"], d["e_round"], d["e_term"],
             d["m_node"], med([c["neg_log_S"] for c in here]),
             d["ratio_node"], d["k_emp"]))
    a("  runs dropped because S = 0: %s"
      % ", ".join("%.2f:%d" % (d["rho"], d["n_drop_S0"]) for d in rows))
    a("  runs dropped because S = 1: %s"
      % ", ".join("%.2f:%d" % (d["rho"], d["n_drop_S1"]) for d in rows))
    a("  e_color and e_term print as 0.0000 because the MEDIAN run makes no")
    a("  colour or terminal error at any density -- NOT because they are")
    a("  identically zero.  Over all %d cells: e_color is nonzero in %d"
      % (len(cells), n_cnz))
    a("  (max %.4f), e_term in %d (max %.4f).  Every cell has exactly one"
      % (max_ec, n_tnz, max_et))
    a("  terminal position and exactly n colour positions, so the classes are")
    a("  present and measured; the model simply gets them right.")
    a("")
    a("X1 the fatal mass is the node class, and k = n.")
    a("   median (-log S)/(n e_node) over the %d densities with n >= 12:"
      % len(hard))
    a("   " + "  ".join("%.2f:%.3f" % (d["rho"], d["ratio_node"])
                        for d in hard))
    a("   run min-max at rho 0.50: %.3f to %.3f"
      % (rows[-1]["ratio_lo"], rows[-1]["ratio_hi"]))
    a("   in [0.8, 1.3] at %d of %d; in [0.5, 2.0] at %d of %d"
      % (x1_in, len(hard), x1_wide, len(hard)))
    a("   predicted: median in [0.8, 1.3] at a majority -> %s" % x1)
    a("")
    a("X2 the colour class carries error mass but not fatal mass.")
    a("   median (n e_color)/(n e_node) at n >= 12: "
      + "  ".join("%.2f:%.3f" % (d["rho"], d["color_over_node"]) for d in hard))
    a("   above 0.20 at %d of %d" % (x2a, len(hard)))
    a("   share of color_wrong first errors at a node IN F_now, n >= 12:")
    a("     %d of %d = %.4f" % (cw_hard[0], cw_hard[1], x2b))
    a("   predicted: mass ratio > 0.20 at a majority AND forced share < 0.05"
      " -> %s" % x2)
    a("")
    a("X3 a zero-parameter node-only law.  OLS of -log S on n e_node through")
    a("   the origin, per density.  R2 is CENTERED (the uncentered one is")
    a("   flattering for a no-intercept fit); both are in exp_summary.csv.")
    for d in hard:
        a("     rho %.2f  slope %.3f +/- %.3f   R2_c %.3f  R2_u %.3f  (n=%d)"
          % (d["rho"], d["slope"], d["slope_se"], d["r2_centered"],
             d["r2_uncentered"], d["n_used"]))
    a("   slope in [0.8, 1.3] at %d of %d; R2_c > 0.80 at %d of %d"
      % (x3a, len(hard), x3b, len(hard)))
    a("   predicted: both at a majority -> %s" % x3)
    a("")
    a("X4 round markers and the terminal are not where the exponent lives.")
    a("   median (T e_round + e_term)/(n e_node) at n >= 12: "
      + "  ".join("%.2f:%.4f" % (d["rho"], d["rt_over_node"]) for d in hard))
    a("   below 0.10 at %d of %d" % (x4n, len(hard)))
    a("   predicted: below 0.10 at a majority -> %s" % x4)
    a("")
    a("X5 every free-generation first error is fatal, whatever its class.")
    a("   solve rate among rollouts whose first error was of this class,")
    a("   pooled over the ten fit densities, in79 runs:")
    for k in sorted(kinds, key=lambda z: -kinds[z][1]):
        a("     %-12s %8d rollouts   solve rate %.5f"
          % (k, kinds[k][1], kinds[k][0] / kinds[k][1]))
    a("   predicted: color_wrong < 0.02 and node_invalid < 0.10 -> %s" % x5)
    a("")
    a("X6 the color_wrong label is a GUARD failure, not a colour failure.")
    a("   join B_instances (free generation) to F_errors_all (forced set) on")
    a("   (run_id, rho, inst): %d matched, %d unmatched." % (matched, unmatched))
    a("   share of color_wrong first errors at an UNFORCED node:")
    a("     pooled %d of %d = %.4f" % (cw_all[1] - cw_all[0], cw_all[1], x6v))
    a("     by density: " + "  ".join(
        "%.2f:%.3f" % (r, 1.0 - cw_forced[round(r, 4)][0]
                       / cw_forced[round(r, 4)][1])
        for r in FIT_RHOS if cw_forced[round(r, 4)][1]))
    a("   predicted: above 0.90 -> %s" % x6)
    a("")
    a("D1 the class decomposition is an identity, not a model.")
    a("   max |P_data - (2n + T + 1)|            = %.3e" % d_P)
    a("   max |P_data - sum_c n_c|               = %.3e" % d_sum)
    a("   max |(1 - p_logic) - sum_c n_c e_c / P| = %.3e" % d_ident)
    a("   max |e_node  - (1 - p_logic_node)|     = %.3e  (two sources)" % d_node)
    a("   max |e_color - (1 - p_logic_color)|    = %.3e  (two sources)" % d_color)
    a("   -> %s" % ("PASSES" if max(d_sum, d_ident, d_node, d_color) < 1e-6
                    and d_P < 0.1 else "FAILS -- X1-X4 are invalid"))
    a("")
    a("WHY X2 IS SCORED A MISS AND NOT A SPLIT.  Its second half HELD -- the")
    a("forced share is 0.0094, inside the registered 0.05.  But the first half")
    a("names an explicit falsification criterion, 'falsified if the mass ratio")
    a("is below 0.05', and the measured ratio is 0.000.  A registered")
    a("falsifier that fires is a MISS even when the reason it fired supports")
    a("the conclusion; reading it as a split would be reinterpreting after the")
    a("fact.  What failed is my premise that there would be colour error mass")
    a("to dilute with -- there is essentially none.")
    a("")
    a("POST-HOC, NOT REGISTERED -- the two factors of k/n.  Since")
    a("R = (-log S)/(n e_node) and D = e_node/(-log p_logic), their product is")
    a("k_emp/n BY CONSTRUCTION; this table is an IDENTITY, not a fit, and its")
    a("whole content is the SIZE of the two factors and the fact that they")
    a("pull in opposite directions and nearly cancel.")
    a("  rho     n | D = e_node/(-log p)  R = fatal share | D*R   k_emp/n")
    for d in rows:
        a("  %.2f %5.1f |        %.3f          |    %.3f      | %.3f   %.3f"
          % (d["rho"], d["n"], d["dilute"], d["ratio_node"],
             d["dilute"] * d["ratio_node"], d["k_emp"] / d["n"]))
    a("  D and R are per-cell medians, so D*R and k_emp/n agree to a couple")
    a("  of parts in a hundred rather than exactly -- a median of products")
    a("  is not a product of medians.  Per cell the identity is exact.")
    a("  D > 1 says a node position is that many times more error-prone than")
    a("  the AVERAGE position, because colour and terminal positions are")
    a("  essentially never wrong and round markers are only moderately so.")
    a("  Alone it would put k at 1.8n.  R < 1 pulls it back.  Neither factor")
    a("  is a second fatal position class, which is what (A) and (B) proposed.")
    a("")
    a("WHAT R < 1 MEANS, AND WHAT THIS STAGE DOES NOT ESTABLISH.  R is the")
    a("ratio of the free-generation fatal hazard per rollout to the")
    a("teacher-forced logical node-error count per rollout.  R ~ 0.5 says")
    a("free generation makes about half as many fatal node errors as teacher")
    a("forcing makes node errors -- teacher forcing is the HARDER regime")
    a("here.  Why is NOT established by this stage.  Candidates, none tested:")
    a("  (i) under teacher forcing the model is carried along the reference")
    a("      prefix and scored at all n node positions whether or not its own")
    a("      policy would have reached them;")
    a("  (ii) in free generation it emits cells in an order of its own")
    a("      choosing, and stage POS already found the per-step hazard FALLS")
    a("      with j (P1, falsified at 31.1% positive), so its later choices")
    a("      are its safer ones;")
    a("  (iii) R rises monotonically with difficulty, 0.443 at rho 0.75 to")
    a("      0.607 at rho 0.50, so whatever it is weakens as n grows.")
    a("Separating these needs a measurement this stage does not contain.")
    a("")
    a("BOTTOM LINE.  The excess of k over n is NOT a second fatal position.")
    a("A colour position is essentially never independently wrong -- the")
    a("median run's colour error rate is exactly 0 at all ten densities and")
    a("only %d of %d cells are nonzero -- so account (A) is dead: there is ONE"
      % (n_cnz, len(cells)))
    a("fatal choice per blank cell, WHICH cell to fill, and the colour is free")
    a("once the cell is right.  The 99.7% of free-generation failures")
    a("labelled color_wrong are 98.8% of them at an UNFORCED node (X6), i.e.")
    a("a guard failure the scorer names after the colour it had to guess.")
    a("Account (B) fails too, but less cleanly: round markers carry 16-20% of")
    a("the node class's error mass and are never fatal, and that dilution is")
    a("most of the D = 1.8 factor above.  Account (C) is what survives.")
    a("")
    a("SCORECARD: X1 %s, X2 %s, X3 %s, X4 %s, X5 %s, X6 %s." %
      (x1, x2, x3, x4, x5, x6))
    a("X1 and X3 missed in the SAME direction and AGREE with each other")
    a("(median %.3f vs slope %.3f at rho 0.50), which is what the registration"
      % (rows[-1]["ratio_node"], rows[-1]["slope"]))
    a("asked for: one robust statistic and one fit had to agree, and they do.")
    a("X5's node_invalid bar was cleared at %.4f against a 0.10 ceiling on"
      % r_ni)
    a("only %d rollouts (%d solved) -- a 95%% Wilson interval of [%.3f, %.3f],"
      % (p_ni[1], p_ni[0], *wilson(p_ni[0], p_ni[1])))
    a("which straddles the bar.  Read it as suggestive, not as a clearance.")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "VERDICT.txt"), "w") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
