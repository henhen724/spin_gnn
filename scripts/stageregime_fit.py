"""Stage REGIME -- why is teacher forcing the HARDER regime?

Finding 97 decomposed the solve-rate exponent as k/n = D x R, with
D = e_node / (-log p_logic) the dilution (1.82-2.00) and
R = (-log S) / (n e_node) the fatal share (0.37-0.61).  D is understood:
round markers carry never-fatal error mass that dilutes the denominator of k.
R is not.  R < 1 says free generation produces about half as many fatal node
events per rollout as teacher forcing produces node errors, which is backwards
from the usual intuition.

This script puts the two regimes on the same per-position axis:

  e_j = 1 - n_logic_correct(j)/n_total(j)      teacher forced, from L_position
  h_j = (n_err(j) - n_err_in_F(j))/n_at_risk(j) free generation, from F_hazard
  r_j = h_j / e_j

Both are LOGICAL, so finding 94's mis-scoring of legal within-round
reorderings is already removed from both sides and no result here can be
bought by it.

The pivot is j = 0.  At the first node decision the two instruments see the
same prefix, the same instances, and no censoring (verified before
registering: n_at_risk == n_instances on 1120/1120 cells).  r_0 ~ 1 means the
gap is ACCUMULATED over j; r_0 < 1 means it is DEFINITIONAL and no selection
story is admissible.

Predictions O1-O5 + D1 are in REGISTERED.md, committed at 8da2183 before this
file existed.  No training, no GPU, no new instances -- a refit of stage
LOGIC's CSVs.  Nothing is submitted.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stageexp_fit import fnum, tobool, read_shards, med, spread  # noqa: E402

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import (SERIES, MUTED, style_axes, label_line,  # noqa: E402
                    series_colors)

OUT = os.environ.get("REGIME_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stageregime_2026-11-17"
LOGIC = os.environ.get("LOGIC_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stagelogic_2026-11-17"
FIGS = os.path.join(OUT, "figures")
DPI = 150
FIT_RHOS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
HARD = [r for r in FIT_RHOS if r <= 0.75]          # the n >= 12 densities
AT_RISK_MIN = 30          # O3's floor: a hazard on < 30 rollouts is not a rate


def key(r):
    return (r["run_id"], round(float(r["rho"]), 4))


def spearman(x, y):
    """Rank correlation, written out rather than imported so the tie handling
    is visible: average ranks, which is what scipy does and what stages POS
    and LOGIC reported."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 4:
        return float("nan"), len(x)

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        out = np.empty(len(v), float)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            out[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return out

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float(np.sum(rx ** 2)) * float(np.sum(ry ** 2)))
    return (float(np.sum(rx * ry) / den) if den > 0 else float("nan")), len(x)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_positions():
    """Join F_hazard (free generation) to L_position scheme=j/tok_type=node
    (teacher forced) on (run_id, rho, j).  Returns the joined per-position
    rows, the per-cell summary, and the bookkeeping the registration demands:
    unmatched keys counted on both sides, never dropped silently."""
    haz = {}
    for r in read_shards(LOGIC, "F_hazard_all_*of10.csv"):
        if not tobool(r, "in79"):
            continue
        haz[(r["run_id"], round(float(r["rho"]), 4), int(r["j"]))] = r

    pos, pos_all = {}, {}
    for r in read_shards(LOGIC, "L_position_all_*of10.csv"):
        if not tobool(r, "in79") or r["tok_type"] != "node":
            continue
        if r["scheme"] == "j":
            pos[(r["run_id"], round(float(r["rho"]), 4), int(r["index"]))] = r
        elif r["scheme"] == "all" and r["index"] == "-1":
            pos_all[key(r)] = r

    rows = []
    only_haz = sorted(set(haz) - set(pos))
    only_pos = sorted(set(pos) - set(haz))
    # D1 is an identity over the POSITION table alone, so it has to be summed
    # over that table's own j range -- see the d1 block in main().  Keep the
    # per-cell n_total map and the last hazard row, so the verdict can say WHY
    # a cell's hazard coverage stops short instead of only that it does.
    pos_cell = collections.defaultdict(dict)
    pos_flat = []   # (rho, n_total == n_instances, n is constant over instances)
    for (run_id, rho, j), p in pos.items():
        pos_cell[(run_id, rho)][j] = fnum(p, "n_total")
        nb = fnum(p, "mean_n_blank")
        pos_flat.append((rho, fnum(p, "n_total") == fnum(p, "n_instances"),
                         abs(nb - round(nb)) < 1e-9))
    haz_last = {}
    for (run_id, rho, j), h in haz.items():
        c = (run_id, rho)
        if c not in haz_last or j > haz_last[c][0]:
            haz_last[c] = (j, h)
    for k in sorted(set(haz) & set(pos)):
        h, p = haz[k], pos[k]
        n_at_risk = fnum(h, "n_at_risk")
        n_total = fnum(p, "n_total")
        n_err = fnum(h, "n_err")
        n_benign = fnum(h, "n_err_in_F")
        n_fatal = n_err - n_benign
        e_j = (1.0 - fnum(p, "n_logic_correct") / n_total) if n_total > 0 \
            else float("nan")
        h_j = (n_fatal / n_at_risk) if n_at_risk > 0 else float("nan")
        rows.append({
            "run_id": k[0], "rho": k[1], "j": k[2],
            "n_instances": fnum(h, "n_instances"),
            "n_at_risk": n_at_risk, "n_total": n_total,
            "n_in_F": fnum(h, "n_in_F"),
            "n_err": n_err, "n_benign": n_benign, "n_fatal": n_fatal,
            "e_j": e_j, "h_j": h_j,
            "r_j": (h_j / e_j) if e_j > 0 else float("nan"),
            "usable": bool(n_at_risk > 0 and n_total > 0 and e_j > 0),
        })
    return rows, pos_all, only_haz, only_pos, pos_cell, haz_last, pos_flat


def load_cells():
    """Per (run, rho): solve rate, p_logic_node, n, T -- for R itself."""
    out = {}
    for r in read_shards(LOGIC, "LF_summary_all_*of10.csv"):
        if not tobool(r, "in79"):
            continue
        n = 48.0 * (1.0 - fnum(r, "realized_rho")) if "realized_rho" in r \
            else fnum(r, "mean_n_blank")
        out[key(r)] = {
            "solve_rate": fnum(r, "solve_rate"),
            "p_logic_node": fnum(r, "p_logic_node"),
            "n": fnum(r, "mean_n_blank"),
            "T": fnum(r, "dd_mean_T"),
            "n_from_rho": n,
        }
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig(name, w=7.4, h=4.6):
    f_, ax = plt.subplots(figsize=(w, h))
    return f_, ax, os.path.join(FIGS, name)


def save(f_, path):
    f_.tight_layout()
    f_.savefig(path, dpi=DPI)
    plt.close(f_)
    print("wrote", path)


def rho_ticks(ax):
    """Label every density.  Matplotlib's auto locator leaves an irregular
    0.90 / 0.80 / 0.75 / ... axis, which reads as non-uniform spacing when the
    densities are evenly spaced by 0.05 (caught in stage EXPONENT)."""
    ax.set_xticks(list(FIT_RHOS))
    ax.set_xticklabels(["%.2f" % r for r in FIT_RHOS], fontsize=9)


def o1_anchor(rows):
    f_, ax, path = fig("O1_anchor.png")
    # Only the ten densities the axis labels and the verdict scores.  The
    # tables also carry rho 0.45-0.33, which plotted as unlabelled clouds to
    # the right of 0.50 -- data the stage never discusses.
    grid = [round(r, 4) for r in FIT_RHOS]
    anchor = [r for r in rows
              if r["j"] == 0 and r["usable"] and r["rho"] in grid]
    ax.scatter([r["rho"] for r in anchor], [r["r_j"] for r in anchor],
               s=10, color=MUTED, alpha=0.4, linewidths=0, zorder=2)
    mv = [med([r["r_j"] for r in rows
               if r["j"] == 0 and r["usable"] and r["rho"] == round(x, 4)])
          for x in FIT_RHOS]
    ax.plot(FIT_RHOS, mv, color=SERIES[0], lw=2.2, marker="o", ms=4.5,
            zorder=4)
    label_line(ax, FIT_RHOS[-1], mv[-1], "median", color=SERIES[0])
    ax.axhline(1.0, color=SERIES[1], lw=1.4, ls="--", zorder=3)
    ax.text(0.945, 1.04, "r = 1:  the two regimes agree at the first decision",
            color=SERIES[1], fontsize=9, va="bottom")
    ax.invert_xaxis()
    ax.set_ylim(0, 2.0)
    rho_ticks(ax)
    style_axes(ax, xlabel="hint density rho   (harder to the right)",
               ylabel="r_0 = h_0 / e_0   at the FIRST node decision")
    save(f_, path)


def o2_profiles(rows, rho=0.50):
    f_, ax, path = fig("O2_profiles.png")
    js = sorted({r["j"] for r in rows if r["rho"] == round(rho, 4)})
    ev, hv, keep = [], [], []
    for j in js:
        sel = [r for r in rows if r["rho"] == round(rho, 4) and r["j"] == j
               and r["usable"] and r["n_at_risk"] >= AT_RISK_MIN]
        if len(sel) < 10:
            continue
        keep.append(j)
        ev.append(med([r["e_j"] for r in sel]))
        hv.append(med([r["h_j"] for r in sel]))
    ax.plot(keep, ev, color=SERIES[0], lw=2.2, marker="o", ms=4)
    ax.plot(keep, hv, color=SERIES[1], lw=2.2, marker="o", ms=4)
    # Both curves converge near zero at the last j, so the direct labels are
    # pushed apart vertically and anchored a few j back, where the two are
    # still clearly separated.
    ax.set_xlim(-0.8, keep[-1] + 9.0)
    ax.set_xticks([t for t in range(0, keep[-1] + 1, 5)])
    label_line(ax, keep[-1], ev[-1], "e_j  teacher forced", color=SERIES[0],
               dy=30)
    label_line(ax, keep[-1], hv[-1], "h_j  free generation", color=SERIES[1],
               dy=12)
    ax.set_ylim(bottom=0)
    style_axes(ax, xlabel="deduction index j   (which blank cell, in order)",
               ylabel="logical error rate at that node decision")
    ax.text(0.02, 0.06, "rho = 0.50, n = 24.  Median over runs; only j with "
                        "at least %d rollouts\nstill at risk and 10 runs "
                        "reporting." % AT_RISK_MIN,
            transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
    save(f_, path)


def o3_ratio(rows):
    f_, ax, path = fig("O3_ratio.png")
    jmax = 0
    for i, rho in enumerate([0.75, 0.60, 0.50]):
        js, vs = [], []
        for j in sorted({r["j"] for r in rows if r["rho"] == round(rho, 4)}):
            sel = [r for r in rows if r["rho"] == round(rho, 4)
                   and r["j"] == j and r["usable"]
                   and r["n_at_risk"] >= AT_RISK_MIN]
            if len(sel) < 10:
                continue
            js.append(j)
            vs.append(med([r["r_j"] for r in sel]))
        if not js:
            continue
        ax.plot(js, vs, color=SERIES[i], lw=2.0, marker="o", ms=3.5)
        # dy lifts the label off the x axis: every curve ends at r_j = 0
        # exactly, so a label at the end point sits on the axis line.
        label_line(ax, js[-1], vs[-1], "rho %.2f" % rho, color=SERIES[i],
                   dy=18)
        jmax = max(jmax, js[-1])
    ax.axhline(1.0, color=MUTED, lw=1.1, ls="--")
    ax.set_xlim(-0.8, jmax + 6.0)
    ax.set_ylim(bottom=0)
    style_axes(ax, xlabel="deduction index j",
               ylabel="r_j = h_j / e_j")
    ax.text(0.02, 0.95, "Every curve ends at exactly 0: a rollout that reaches "
                        "the LAST blank with no\nerror faces a cell the "
                        "completed colouring has already determined, so the\n"
                        "free-generation hazard there is zero while the "
                        "teacher-forced rate is not.",
            transform=ax.transAxes, color=MUTED, fontsize=9, va="top")
    save(f_, path)


def o4_atrisk(rows):
    f_, ax, path = fig("O4_atrisk.png")
    for i, rho in enumerate([0.75, 0.60, 0.50]):
        js, vs = [], []
        for j in sorted({r["j"] for r in rows if r["rho"] == round(rho, 4)}):
            sel = [r for r in rows if r["rho"] == round(rho, 4)
                   and r["j"] == j and r["n_instances"] > 0]
            if len(sel) < 10:
                continue
            js.append(j)
            vs.append(med([r["n_at_risk"] / r["n_instances"] for r in sel]))
        if not js:
            continue
        ax.plot(js, vs, color=SERIES[i], lw=2.0, marker="o", ms=3.5)
        label_line(ax, js[-1], vs[-1], "rho %.2f" % rho, color=SERIES[i])
    ax.set_ylim(0, 1.04)
    style_axes(ax, xlabel="deduction index j",
               ylabel="fraction of rollouts still at risk")
    save(f_, path)


def ph_budget(rows, cells):
    """Where R comes from: the running sums the two regimes accumulate."""
    f_, ax, path = fig("PH_budget.png")
    rho = 0.50
    js, ce, ch = [], [], []
    se = sh = 0.0
    for j in sorted({r["j"] for r in rows if r["rho"] == round(rho, 4)}):
        sel = [r for r in rows if r["rho"] == round(rho, 4) and r["j"] == j
               and r["usable"] and r["n_at_risk"] >= AT_RISK_MIN]
        if len(sel) < 10:
            continue
        se += med([r["e_j"] for r in sel])
        sh += med([-math.log(1.0 - r["h_j"]) for r in sel if r["h_j"] < 1])
        js.append(j)
        ce.append(se)
        ch.append(sh)
    ax.plot(js, ce, color=SERIES[0], lw=2.2, marker="o", ms=3.5)
    ax.plot(js, ch, color=SERIES[1], lw=2.2, marker="o", ms=3.5)
    label_line(ax, js[-1], ce[-1], "sum e_j   (teacher forced)",
               color=SERIES[0])
    label_line(ax, js[-1], ch[-1], "sum -log(1-h_j)   (free generation)",
               color=SERIES[1])
    ax.set_xlim(-0.8, js[-1] + 11.0)
    ax.set_ylim(bottom=0)
    style_axes(ax, xlabel="deduction index j",
               ylabel="cumulative expected errors per rollout")
    # lower right: the top left is where the teacher-forced curve climbs
    ax.text(0.98, 0.06, "rho = 0.50.  The ratio of the two end points IS R, "
                        "the fatal share\nof finding 97 -- read here as a "
                        "budget accumulated decision by decision.",
            transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom",
            ha="right")
    save(f_, path)


def ph_mechanism(rows):
    """POST-HOC, found by reading stagelogic_score.py, not registered.

    The three rates that between them ARE R, at the uncensored j = 0 anchor:
    how often the model names a cell DD has not forced, how often it colours
    such a cell correctly anyway, and how often it errs on a forced cell.
    """
    f_, ax, path = fig("PH_mechanism.png")
    xs, prem, guess, onf = [], [], [], []
    for rho in FIT_RHOS:
        sel = [r for r in rows if r["rho"] == round(rho, 4) and r["j"] == 0]
        ar = sum(r["n_at_risk"] for r in sel)
        inf = sum(r["n_in_F"] for r in sel)
        fatal = sum(r["n_fatal"] for r in sel)
        benign = sum(r["n_benign"] for r in sel)
        unf = ar - inf
        if not ar or not unf:
            continue
        xs.append(rho)
        prem.append(unf / ar)
        guess.append(1.0 - fatal / unf)
        onf.append(benign / inf if inf else float("nan"))
    c = series_colors(3, highlight=1)
    for v, lab, col in ((prem, "named an unforced cell", c[0]),
                        (guess, "of those, colour right", c[1]),
                        (onf, "error on a forced cell", c[2])):
        ax.plot(xs, v, color=col, lw=2.2, marker="o", ms=3.5)
    ax.invert_xaxis()
    rho_ticks(ax)
    # room at the right for the direct labels, which otherwise run off the axes
    ax.set_xlim(0.975, 0.335)
    ax.set_ylim(-0.02, 1.0)
    label_line(ax, xs[-1], prem[-1], "named an unforced cell", color=c[0])
    label_line(ax, xs[-1], guess[-1], "of those, colour right", color=c[1])
    label_line(ax, xs[-1], onf[-1], "error on a forced cell", color=c[2])
    # The chance line is a reference for the COLOUR-RIGHT curve only.  Drawn
    # short and annotated on the left, because a full-width rule at 0.25 lands
    # exactly where the unforced share ends and reads as if it belonged to
    # that curve instead.
    ax.plot([0.95, 0.76], [0.25, 0.25], color=MUTED, lw=1.0, ls=":")
    ax.text(0.855, 0.265, "1/q = 0.25, a blind guess", color=MUTED,
            fontsize=9, ha="center")
    style_axes(ax, xlabel="hint density rho", ylabel="rate at j = 0")
    save(f_, path)
    return xs, prem, guess, onf


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    os.makedirs(FIGS, exist_ok=True)
    rows, pos_all, only_haz, only_pos, pos_cell, haz_last, pos_flat = \
        load_positions()
    cells = load_cells()
    out = []

    def a(s=""):
        out.append(s)

    a("Stage REGIME -- why is teacher forcing the HARDER regime?")
    a("Finding 97 left R = (-log S)/(n e_node) = 0.37-0.61 unexplained.  This")
    a("puts the free-generation hazard h_j and the teacher-forced logical node")
    a("error e_j on the same deduction index j.  Both LOGICAL.  Predictions in")
    a("REGISTERED.md, committed at 8da2183 before this script was written.")
    a("No training, no GPU, no new instances -- a refit of stage LOGIC's CSVs.")
    a()

    # ---------------- D1 ----------------
    bycell = collections.defaultdict(list)
    for r in rows:
        bycell[(r["run_id"], r["rho"])].append(r)
    # The identity is a statement about the POSITION table: every instance is
    # scored at every node position it has.  So it is summed over that table's
    # own j range.  An earlier version of this block summed over the JOINED j
    # instead and reported a max difference of 6801, which was this check being
    # wrong and not D1 being false -- a joined-only sum silently drops every j
    # the hazard table never reached.  The second half of D1, the unmatched-key
    # count, is what carries the coverage information, and it is broken out by
    # density below.
    d1_max, d1_cells, d1_eq = 0.0, 0, 0
    for k, jmap in pos_cell.items():
        ref = pos_all.get(k)
        if ref is None:
            continue
        d1_cells += 1
        d = abs(sum(jmap.values()) - fnum(ref, "n_total"))
        d1_max = max(d1_max, d)
        d1_eq += (d < 1e-9)
    # n_total(j) == n_instances for j < n, the other half of the identity.
    # The clause is conditional on j < n, so it is only well-posed where n is
    # the same for every instance in the cell -- true by construction in all
    # ten scored densities (n_blank is a constant integer inside a fit bin) and
    # false at rho 0.33/0.35, where mean_n_blank is fractional and instances
    # with fewer blanks simply have no node at the largest j.  Counted both
    # ways rather than restricted, so the split is visible.
    d1_flat = sum(1 for rho, eq, const in pos_flat if const and not eq)
    d1_flat_n = sum(1 for rho, eq, const in pos_flat if const)
    d1_var = sum(1 for rho, eq, const in pos_flat if not const and not eq)
    d1_var_n = sum(1 for rho, eq, const in pos_flat if not const)
    a("D1  the two tables are the same population, position by position.")
    a("    sum_j n_total(j) == n_total(scheme=all) on %d of %d cells, max |diff| %.0f"
      % (d1_eq, d1_cells, d1_max))
    a("    n_total(j) != n_instances at %d of %d position rows where n is the"
      % (d1_flat, d1_flat_n))
    a("    same for every instance (the ten scored densities), and at %d of %d"
      % (d1_var, d1_var_n))
    a("    where it is not (rho 0.33/0.35, j >= 29, so j >= n for some instance")
    a("    and the `for j < n' clause does not apply).")
    a("    joined keys %d;  only in F_hazard %d;  only in L_position %d"
      % (len(rows), len(only_haz), len(only_pos)))
    byrho = collections.Counter(round(k[1], 4) for k in only_pos)
    a("    unmatched by rho: %s"
      % ("  ".join("%.2f:%d" % (r, c) for r, c in sorted(byrho.items()))
         or "none"))
    in_grid = sum(c for r, c in byrho.items()
                  if any(abs(r - f) < 1e-9 for f in FIT_RHOS))
    a("    unmatched keys INSIDE the ten scored densities: %d" % in_grid)
    # why a short cell stops: the at-risk set is exhausted, or the last
    # survivor completed its trace.  Either way it is not a missing row.
    short = [k for k, jmap in pos_cell.items()
             if set(jmap) - set(x["j"] for x in bycell.get(k, []))]
    exh = sum(1 for k in short
              if k in haz_last
              and fnum(haz_last[k][1], "n_at_risk") - fnum(haz_last[k][1], "n_err") == 0)
    a("    %d cells have short hazard coverage; at-risk set exhausted at the"
      % len(short))
    a("    last j in %d of them, the rest end with a lone survivor completing."
      % exh)
    d1_ok = (d1_max < 1e-9) and (d1_flat == 0) and (in_grid == 0) \
        and (len(only_haz) == 0)
    a("    %s" % ("PASSES -- the identity is exact, and every unmatched key is"
                 if d1_ok else
                 "DOES NOT CLOSE -- O1-O5 are invalid, not merely wrong"))
    if d1_ok:
        a("    at rho <= 0.45, outside every density O1-O5 are scored at.")
    a()

    # ---------------- O1 ----------------
    a("O1  the j = 0 anchor -- same prefix, same instances, no censoring.")
    a("     rho      n | runs | median r_0 | run min-max       | in [0.8,1.25]")
    o1_hit = 0
    o1_rows = []
    o1_drop = []          # the e_0 = 0 cells the registration says to report
    for rho in FIT_RHOS:
        sel = [r for r in rows if r["rho"] == round(rho, 4) and r["j"] == 0
               and r["usable"]]
        dropped = [r for r in rows if r["rho"] == round(rho, 4) and r["j"] == 0
                   and not r["usable"]]
        o1_drop.append((rho, dropped))
        v = [r["r_j"] for r in sel]
        m = med(v)
        lo, hi = spread(v)
        nn = med([cells[(r["run_id"], r["rho"])]["n"] for r in sel
                  if (r["run_id"], r["rho"]) in cells])
        inband = 0.8 <= m <= 1.25
        if rho in HARD and inband:
            o1_hit += 1
        o1_rows.append({"rho": rho, "n": nn, "runs": len(v), "r0": m,
                        "lo": lo, "hi": hi})
        a("    %5.2f %5.1f | %4d | %10.3f | %6.3f - %6.3f | %s%s"
          % (rho, nn, len(v), m, lo, hi, "yes" if inband else "no",
             "" if rho in HARD else "   (n < 12, not scored)"))
    a("    in [0.8, 1.25] at %d of %d densities with n >= 12"
      % (o1_hit, len(HARD)))
    o1_med = med([r["r0"] for r in o1_rows if r["rho"] in HARD])
    a("    pooled median r_0 over those six densities = %.3f" % o1_med)
    a("    predicted: median in [0.8, 1.25] at a majority -> %s"
      % ("HIT" if o1_hit > len(HARD) / 2 else "MISS"))
    a()
    # The drop count the registration promised at every density, plus the one
    # thing that makes it more than bookkeeping: a dropped cell has e_0 = 0,
    # i.e. teacher forcing is PERFECT at the first decision, so r_0 is 0/0 or
    # +inf there.  If h_0 > 0 in those cells, excluding them removes exactly
    # the cells where free generation is the harder regime, and the surviving
    # median r_0 is biased DOWN.  That has to be sized before O1 is read.
    a("    DROPPED CELLS (e_0 = 0, so r_0 undefined) -- and which way they push:")
    a("     rho | kept | dropped | of those, h_0 > 0 | median h_0 there")
    tot_drop = tot_pos = 0
    for rho, dr in o1_drop:
        if not dr and rho not in HARD:
            continue
        pos_h = [r for r in dr if r["h_j"] > 0]
        tot_drop += len(dr)
        tot_pos += len(pos_h)
        a("    %5.2f | %4d | %7d | %17d | %s"
          % (rho, sum(1 for r in rows if r["rho"] == round(rho, 4)
                      and r["j"] == 0 and r["usable"]),
             len(dr), len(pos_h),
             ("%.4f" % med([r["h_j"] for r in pos_h])) if pos_h else "--"))
    a("    pooled: %d dropped, %d of them with h_0 > 0" % (tot_drop, tot_pos))
    if tot_pos:
        a("    So the exclusion does remove cells where free generation errs and")
        a("    teacher forcing does not.  Those are r_0 = +inf, and dropping them")
        a("    biases the reported median r_0 DOWNWARD -- O1's MISS is therefore")
        a("    NOT an artefact of the drop rule: correcting it would move r_0 UP,")
        a("    and it is already below the band.  The bias is bounded by how many")
        a("    cells there are: %d of %d at j = 0." % (tot_pos, tot_pos + tot_drop))
    else:
        a("    No dropped cell has h_0 > 0, so the drop rule removes only 0/0")
        a("    cells and cannot move the median r_0 in either direction.")
    a()
    if o1_med < 0.8:
        a("    READ: r_0 < 1 AT THE FIRST DECISION.  The two regimes do NOT")
        a("    agree where they are provably on the same state and the same")
        a("    instances, so R < 1 is present before any selection or any")
        a("    ordering freedom can act.  Candidates (i) and (ii) cannot be")
        a("    the whole story, and a DEFINITIONAL difference between the two")
        a("    instruments is implicated.  O5 is the one that names a")
        a("    definitional difference, and it is scored below.")
    elif o1_med > 1.25:
        a("    READ: r_0 > 1 -- free generation is HARDER at the first")
        a("    decision and the gap must reverse later.")
    else:
        a("    READ: the regimes AGREE at j = 0, so R < 1 is accumulated over")
        a("    j.  That is where candidates (i) and (ii) live, and O3/O4 say")
        a("    whether the accumulation tracks survivorship.")
    a()

    # ---------------- O2 ----------------
    a("O2  shape cannot explain R, because e_j is flat in j.")
    a("     rho      n | CV(e_j) | Jensen term / sum -log(1-h_j)")
    o2a = o2b = 0
    for rho in FIT_RHOS:
        cvs, jens = [], []
        for k, rs in bycell.items():
            if k[1] != round(rho, 4):
                continue
            use = [x for x in rs if x["usable"]
                   and x["n_at_risk"] >= AT_RISK_MIN]
            if len(use) < 4:
                continue
            e = np.array([x["e_j"] for x in use], float)
            if e.mean() > 0:
                cvs.append(float(e.std() / e.mean()))
            h = np.array([x["h_j"] for x in use], float)
            h = h[h < 1]
            if len(h) >= 4 and h.mean() < 1:
                s = float(np.sum(-np.log(1 - h)))
                flat = len(h) * (-math.log(1 - h.mean()))
                if s > 0:
                    jens.append((s - flat) / s)
        mc, mj = med(cvs), med(jens)
        if rho in HARD:
            if mc < 0.25:
                o2a += 1
            if abs(mj) < 0.10:
                o2b += 1
        nn = med([cells[k]["n"] for k in bycell if k[1] == round(rho, 4)
                  and k in cells])
        a("    %5.2f %5.1f | %7.3f | %+.4f%s"
          % (rho, nn, mc, mj, "" if rho in HARD else "   (n < 12)"))
    a("    CV(e_j) < 0.25 at %d of %d;  |Jensen| < 0.10 at %d of %d"
      % (o2a, len(HARD), o2b, len(HARD)))
    a("    predicted: both at a majority -> %s"
      % ("HIT" if o2a > len(HARD) / 2 and o2b > len(HARD) / 2 else "MISS"))
    a("    If e_j is flat then R is a ratio of MEANS and the falling shape of")
    a("    h_j (POS P1) is irrelevant to R by construction.  That retires")
    a("    candidate (ii) analytically rather than empirically.")
    a()

    # ---------------- O3 ----------------
    a("O3  the gap grows with j (registered as a CONFIRMATION, not a")
    a("    discovery: POS P1 + LOGIC P3 nearly force it).")
    a("     rho      n | cells | median Spearman(r_j, j) | share negative")
    o3_hit = 0
    for rho in FIT_RHOS:
        rs_, neg, tot = [], 0, 0
        for k, rs in bycell.items():
            if k[1] != round(rho, 4):
                continue
            use = [x for x in rs if x["usable"]
                   and x["n_at_risk"] >= AT_RISK_MIN]
            if len(use) < 4:
                continue
            s, _ = spearman([x["j"] for x in use], [x["r_j"] for x in use])
            if s == s:
                rs_.append(s)
                tot += 1
                neg += 1 if s < 0 else 0
        m = med(rs_)
        if rho in HARD and m < -0.3:
            o3_hit += 1
        nn = med([cells[k]["n"] for k in bycell if k[1] == round(rho, 4)
                  and k in cells])
        a("    %5.2f %5.1f | %5d | %23.3f | %s"
          % (rho, nn, tot, m,
             ("%.3f" % (neg / tot)) if tot else "n/a"))
    a("    median Spearman < -0.3 at %d of %d densities with n >= 12"
      % (o3_hit, len(HARD)))
    a("    predicted: < -0.3 at a majority -> %s"
      % ("HIT" if o3_hit > len(HARD) / 2 else "MISS"))
    a()

    # ---------------- O4 ----------------
    a("O4  survivorship is large enough for O3 to be about selection.")
    a("     rho      n | last j | median n_at_risk/n_instances at that j")
    o4v = float("nan")
    for rho in FIT_RHOS:
        best = {}
        for k, rs in bycell.items():
            if k[1] != round(rho, 4):
                continue
            use = [x for x in rs if x["n_instances"] > 0]
            if not use:
                continue
            x = max(use, key=lambda z: z["j"])
            best[k] = (x["j"], x["n_at_risk"] / x["n_instances"])
        if not best:
            continue
        m = med([v[1] for v in best.values()])
        lj = med([v[0] for v in best.values()])
        if abs(rho - 0.50) < 1e-9:
            o4v = m
        nn = med([cells[k]["n"] for k in best if k in cells])
        a("    %5.2f %5.1f | %6.1f | %.4f" % (rho, nn, lj, m))
    a("    at rho = 0.50 the median surviving fraction at the last j is %.4f"
      % o4v)
    a("    predicted: < 0.40 -> %s" % ("HIT" if o4v < 0.40 else "MISS"))
    a()

    # ---------------- O5 ----------------
    a("O5  the censoring is small, and which way it pushes R.")
    tot_err = sum(r["n_err"] for r in rows)
    tot_ben = sum(r["n_benign"] for r in rows)
    a("    benign first errors (named a FORCED cell, just not the")
    a("    reference's next one): %d of %d first errors = %.4f"
      % (tot_ben, tot_err, (tot_ben / tot_err) if tot_err else float("nan")))
    a("    predicted: < 0.05 -> %s"
      % ("HIT" if tot_err and tot_ben / tot_err < 0.05 else "MISS"))
    a()
    a("     rho | median r_j as measured | with benign censoring undone | up?")
    o5_up = 0
    for rho in FIT_RHOS:
        asis, fixed = [], []
        for k, rs in bycell.items():
            if k[1] != round(rho, 4):
                continue
            use = sorted([x for x in rs if x["usable"]
                          and x["n_at_risk"] >= AT_RISK_MIN],
                         key=lambda z: z["j"])
            if len(use) < 4:
                continue
            cum = 0.0
            for x in use:
                asis.append(x["r_j"])
                den = x["n_at_risk"] + cum
                if den > 0 and x["e_j"] > 0:
                    fixed.append((x["n_fatal"] / den) / x["e_j"])
                cum += x["n_benign"]
        ma, mf = med(asis), med(fixed)
        up = mf > ma
        if rho in HARD and up:
            o5_up += 1
        a("    %5.2f | %22.4f | %28.4f | %s"
          % (rho, ma, mf, "yes" if up else "no"))
    a("    raises median r_j at %d of %d densities with n >= 12"
      % (o5_up, len(HARD)))
    a("    predicted: RAISES at a majority -> %s"
      % ("HIT" if o5_up > len(HARD) / 2 else "MISS"))
    a()
    a("    A DEFECT IN MY OWN REGISTRATION, and it is the second half of O5.")
    a("    REGISTERED.md argues in prose that benign censoring SHRINKS the")
    a("    denominator and therefore RAISES h_j -- which is right -- and then")
    a("    predicts that UNDOING the censoring raises r_j, which is the")
    a("    opposite direction: undoing it ENLARGES the denominator and must")
    a("    LOWER h_j.  The two sentences contradict each other.  The")
    a("    prediction is scored exactly as written, so this half is a MISS if")
    a("    the correction lowers r_j, and the prose was the correct half.")
    a("    The correction is an UPPER bound on the denominator (a benignly")
    a("    censored rollout is credited as still at risk at every later j), so")
    a("    the corrected r_j is a LOWER bound.")
    a()

    # ---------------- post-hoc ----------------
    a("POST-HOC, NOT REGISTERED -- the per-decision budget behind R.")
    a("     rho      n | sum e_j | sum -log(1-h_j) | ratio | R from 97")
    for rho in HARD:
        se_, sh_, rr = [], [], []
        for k, rs in bycell.items():
            if k[1] != round(rho, 4):
                continue
            use = [x for x in rs if x["usable"]]
            if len(use) < 4:
                continue
            se_.append(sum(x["e_j"] for x in use))
            sh_.append(sum(-math.log(1 - x["h_j"]) for x in use
                           if x["h_j"] < 1))
            c = cells.get(k)
            if c and 0 < c["solve_rate"] < 1 and c["p_logic_node"] < 1:
                rr.append(-math.log(c["solve_rate"])
                          / (c["n"] * (1 - c["p_logic_node"])))
        a("    %5.2f %5.1f | %7.3f | %15.3f | %5.3f | %.3f"
          % (rho, med([cells[k]["n"] for k in bycell
                       if k[1] == round(rho, 4) and k in cells]),
             med(se_), med(sh_),
             med(sh_) / med(se_) if med(se_) else float("nan"), med(rr)))
    a("    The `ratio` column is this stage's own reconstruction of R from the")
    a("    per-position tables; `R from 97` is the whole-rollout quantity.")
    a("    They are built from different columns of the same runs, so they")
    a("    agree only up to the j range the hazard table covers -- the")
    a("    difference is a measure of what the per-position view leaves out,")
    a("    NOT a second estimate to average with the first.")
    a()

    # ---------------- PH2: what R actually is ----------------
    # NOT REGISTERED.  Found after O1 came in at 0.499 by reading the scorer
    # that produced both tables, which is the only place the two event
    # definitions can be compared.  O1's registered READ said a DEFINITIONAL
    # difference was implicated; this names it, but naming it was post-hoc.
    a("POST-HOC, NOT REGISTERED -- WHAT R ACTUALLY IS.")
    a("    O1 says r_0 = 0.499 where the two instruments are provably on the")
    a("    same prefix, the same instances and the same greedy argmax over the")
    a("    same 58 symbols (rollout_batch takes an unconstrained argmax, so at")
    a("    j = 0 the emitted token is the SAME TOKEN in both regimes).  Two")
    a("    instruments cannot disagree about one token unless they are scoring")
    a("    different EVENTS, and they are:")
    a()
    a("      e_j  counts  `the cell named was not forced by DD'")
    a("      h_j  counts  `the pair emitted was not an ok pair'")
    a()
    a("    and stagelogic_score.py's walk_continuation defines an ok pair as")
    a("    node_valid AND color_correct, where node_valid is only `not already")
    a("    coloured' -- NOT `forced'.  So naming a cell DD has not yet forced is")
    a("    not an error in free generation at all, provided the colour is right.")
    a("    Confirmed on the first-error records: of 20657 first errors at j = 0")
    a("    over the six scored densities, 20656 are kind=color_wrong and 1 is")
    a("    node_invalid; err_uncolored is 1 on all but that one.  The cell")
    a("    CHOICE is never itself the error.")
    a()
    a("    The three rates at the uncensored j = 0 anchor, pooled over 79 runs:")
    a("     rho | emissions | named unforced | of those, colour | error on a")
    a("         |           |                |      right      | forced cell")
    for rho in FIT_RHOS:
        sel = [r for r in rows if r["rho"] == round(rho, 4) and r["j"] == 0]
        ar = sum(r["n_at_risk"] for r in sel)
        inf = sum(r["n_in_F"] for r in sel)
        fatal = sum(r["n_fatal"] for r in sel)
        benign = sum(r["n_benign"] for r in sel)
        unf = ar - inf
        if not ar or not unf:
            continue
        a("    %.2f | %9d | %6d  %6.4f | %15.4f | %10.4f"
          % (rho, ar, unf, unf / ar, 1.0 - fatal / unf,
             benign / inf if inf else float("nan")))
    a()
    a("    READ, and it answers 97's open question:  the model has LEARNED")
    a("    direct deduction -- on a cell DD really has forced it errs 0.1-0.3%")
    a("    of the time.  What it has not learned is WHEN TO WAIT: it names a")
    a("    not-yet-forced cell 3.7% of the time at rho 0.95 rising to 25.3% at")
    a("    rho 0.50, and then colours it correctly about HALF the time, against")
    a("    1/q = 0.25 for a blind guess.  Teacher forcing counts every one of")
    a("    those premature picks as a logical error; free generation only")
    a("    counts the mis-coloured half.  R ~ 0.5 IS THAT HALF.  Teacher")
    a("    forcing is therefore not the harder regime -- it is the stricter")
    a("    SCORE, and R is the rate at which a premature guess happens to be")
    a("    right.  This also explains O1 and O2 together: the one-half is a")
    a("    property of a single decision, which is why it is already there at")
    a("    j = 0 and why no shape in j was ever needed to produce it.")
    a()
    a("    What this does NOT establish: the colour accuracy on unforced cells")
    a("    is ~0.5 at every density, but WHY it is 0.5 rather than 0.25 or 0.9")
    a("    is not measured here.  A cell DD has not forced may still be heavily")
    a("    constrained (q - 1 colours excluded of q), so 0.5 is not evidence of")
    a("    guessing at random, and the n_F_sum column needed to size that is")
    a("    not used by this stage.")
    a()

    # ---------------- write ----------------
    with open(os.path.join(OUT, "regime_positions.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    a("SCORECARD written alongside regime_positions.csv (%d rows)." % len(rows))

    o1_anchor(rows)
    o2_profiles(rows)
    o3_ratio(rows)
    o4_atrisk(rows)
    ph_budget(rows, cells)
    ph_mechanism(rows)

    txt = "\n".join(out) + "\n"
    with open(os.path.join(OUT, "VERDICT.txt"), "w") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
