#!/usr/bin/env python3
"""Stage HEADS (105) — the scoring half.

Registered in `runs/stageheads_2026-11-19/REGISTERED.md`, committed before this
file existed; the measurement in `scripts/stageheads_probe.py` was committed
before this file existed too.  The split is forced: on this cluster loading
`py-matplotlib` unloads `py-pytorch`, so a scorer that draws cannot also run a
model.

WHAT IS SCORED
--------------
H1  hop 1 exists and tracks the gate   median S_row(pass) / median S_row(fail) >= 1.5
H2  and it is not only depth           J2 beats the other eight depth-16 cells, 3 of 3
H3  hop 2 exists                       pooled S_hint >= 1.5
H4  the join is complete in EFFORT     pooled R_dc in [0.90, 1.10]
H5  blanks are discriminated           pooled R_ab >= 1.3
H6  the circuit is small               median n_50 <= 4 of L x H

E1  the query is the right position    argmax == true colour, and is a COLOR token
E2  the capture is a distribution      every (l,h) row sums to 1, nothing above q
E3  the layout mirrors the encoder     token_positions agrees with encode_instance

POOLING.  The registration says masses are summed over every (l,h) and over
instances BEFORE any ratio is taken, "so each is mass-weighted and no ratio of
ratios appears".  Carried to the run axis, a pooled quantity sums masses and
counts over the 18 gate-passing runs and divides once.  That weights a run by
its depth x heads, because each (l,h) row of attention sums to 1.  The
unweighted median over the same 18 runs is printed beside every pooled number
so the weighting can be seen rather than trusted; where the two disagree about
a band, VERDICT says so.

Per-cell values are the median over that cell's seeds.  Runs are keyed on the
`cell` LABEL, never on (arch_mode, num_layers, d_model) — finding 96's defect.

No training, no GPU, no new instances, nothing submitted.  Login node, seconds.
"""

import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stageexp_fit import fnum, tobool, read_shards, med, spread  # noqa: E402
from stagegate_fit import GATE, J2_LABEL, is_cell  # noqa: E402

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import (SERIES, MUTED, style_axes, label_line,  # noqa: E402
                    series_colors)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = "/scratch/users/hshunt/spin_gnn/runs"
OUT = os.path.join(RUNS, "stageheads_2026-11-19")
BOUND = os.path.join(RUNS, "stagebound_2026-11-17", "bound_runs.csv")
FIGS = os.path.join(OUT, "figures")
DPI = 150

RHOS = [0.90, 0.75, 0.60]
MAIN = 0.75                      # the density every band is registered at
DEPTH16 = 16

H1_BAR, H1_FALSIFIER = 1.5, 1.0
H3_BAR, H3_FALSIFIER = 1.5, 1.0
H4_BAND, H4_FALSIFIER = (0.90, 1.10), (0.80, 1.25)
H5_BAR, H5_FALSIFIER = 1.3, 1.0
H6_BAR, H6_FALSIFIER = 4, 12
E1_RIGHT, E1_COLOR = 0.90, 0.99


def fig(name):
    f, ax = plt.subplots(figsize=(7.2, 4.3))
    return f, ax, os.path.join(FIGS, name)


def save(f, path):
    f.tight_layout()
    f.savefig(path, dpi=DPI)
    plt.close(f)
    print("  wrote", os.path.basename(path))


def ratio(num, den):
    return num / den if den > 0 else float("nan")


# ------------------------------------------------------------------ loading

def load_gate():
    """Per-cell median rho50_pub, and the gate verdict that follows from it.

    The gate is stage GATE's 0.740 = 0.592 + s*, unchanged.  A censored rho50
    is dropped from the median rather than imputed; a cell whose seeds are all
    censored has no rho50 and cannot pass.
    """
    by_cell = collections.defaultdict(list)
    by_run = {}
    with open(BOUND, newline="") as fh:
        for r in csv.DictReader(fh):
            cell = " ".join(r["cell"].split())
            v = fnum(r, "rho50_pub")
            by_run[r["run_id"]] = v
            by_cell[cell].append(v)
    cell_rho50 = {c: med(v) for c, v in by_cell.items()}
    passing = {c for c, v in cell_rho50.items() if v == v and v <= GATE}
    return cell_rho50, passing, by_run


def derive(r):
    """The five registered quantities for one (run, rho), plus diagnostics."""
    k_tot, n_hint = fnum(r, "k_tot"), fnum(r, "n_hint_tok")
    n_q = fnum(r, "n_q")
    d = {
        "run_id": r["run_id"],
        "cell": " ".join(r["cell"].split()),
        "rho": round(fnum(r, "rho"), 4),
        "depth": int(fnum(r, "depth")),
        "heads": int(fnum(r, "heads")),
        "d_model": int(fnum(r, "d_model")),
        "arch_mode": r["arch_mode"],
        "params": fnum(r, "params"),
        "val_acc": fnum(r, "val_acc"),
        "in79": tobool(r, "in79"),
        "n_q": n_q,
        # masses, kept so the pool can re-sum rather than average averages
        "m_own_row": fnum(r, "m_own_row"), "m_own_hdr": fnum(r, "m_own_hdr"),
        "m_own_nbr": fnum(r, "m_own_nbr"), "m_rows": fnum(r, "m_rows"),
        "m_nbrhint": fnum(r, "m_nbrhint"), "m_hints": fnum(r, "m_hints"),
        "k_tot": k_tot, "n_hint_tok": n_hint,
        "n50": fnum(r, "n50"), "top1_frac": fnum(r, "top1_frac"),
        "rank1_frac": ratio(fnum(r, "rank1"), n_q),
        "mean_rank": ratio(fnum(r, "rank_sum"), n_q),
        "e1_right_frac": ratio(fnum(r, "e1_color_right"), n_q),
        "e1_color_frac": ratio(fnum(r, "e1_is_color"), n_q),
        "e2_bad": fnum(r, "e2_bad"), "e3_bad": fnum(r, "e3_bad"),
    }
    for k in ("crit_dc", "drop_dc", "assigned_ab", "blank_ab",
              "crit_all", "drop_all", "blank_all"):
        d["mass_" + k] = fnum(r, "mass_" + k)
        d["n_" + k] = fnum(r, "n_" + k)

    d["S_row"] = 48.0 * ratio(d["m_own_row"], d["m_rows"])
    d["S_hint"] = ratio(ratio(d["m_nbrhint"], k_tot),
                        ratio(d["m_hints"], n_hint))
    d["R_dc"] = ratio(ratio(d["mass_crit_dc"], d["n_crit_dc"]),
                      ratio(d["mass_drop_dc"], d["n_drop_dc"]))
    d["R_ab"] = ratio(ratio(d["mass_assigned_ab"], d["n_assigned_ab"]),
                      ratio(d["mass_blank_ab"], d["n_blank_ab"]))
    # the pair-count the circuit size is a fraction of
    d["n_pairs"] = d["depth"] * d["heads"]
    d["n50_frac"] = ratio(d["n50"], d["n_pairs"])
    return d


def pooled(rows, kind):
    """Sum masses and counts over runs, then divide once."""
    s = lambda k: sum(r[k] for r in rows)            # noqa: E731
    if kind == "S_row":
        return 48.0 * ratio(s("m_own_row"), s("m_rows"))
    if kind == "S_hint":
        return ratio(ratio(s("m_nbrhint"), s("k_tot")),
                     ratio(s("m_hints"), s("n_hint_tok")))
    if kind == "R_dc":
        return ratio(ratio(s("mass_crit_dc"), s("n_crit_dc")),
                     ratio(s("mass_drop_dc"), s("n_drop_dc")))
    if kind == "R_ab":
        return ratio(ratio(s("mass_assigned_ab"), s("n_assigned_ab")),
                     ratio(s("mass_blank_ab"), s("n_blank_ab")))
    raise KeyError(kind)


def cell_values(rows, key):
    """cell -> median over that cell's seeds."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cell"]].append(r[key])
    return {c: med(v) for c, v in by.items()}


# ----------------------------------------------------------------- figures

def f1_gate(rows75, cell_rho50, passing):
    """S_row against the architecture's rho_50, one point per cell."""
    f, ax, path = fig("H1_srow_vs_rho50.png")
    sr = cell_values(rows75, "S_row")
    cells = [c for c in sr if cell_rho50.get(c, float("nan")) ==
             cell_rho50.get(c, float("nan"))]
    xs = [cell_rho50[c] for c in cells]
    ys = [sr[c] for c in cells]
    cols = [SERIES[0] if c in passing else MUTED for c in cells]
    ax.scatter(xs, ys, s=38, c=cols, zorder=3, rasterized=True)
    ax.axvline(GATE, color=SERIES[1], lw=1.2, ls="--", zorder=2)
    ax.annotate("gate 0.740", xy=(GATE, max(ys) * 0.96), xytext=(4, 0),
                textcoords="offset points", color=SERIES[1], fontsize=9)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.annotate("no preference", xy=(max(xs), 1.0), xytext=(-78, 5),
                textcoords="offset points", color=MUTED, fontsize=9)
    style_axes(ax, xlabel=r"architecture's half-solve density $\rho_{50}$"
                          " (lower is better)",
               ylabel=r"own-row selectivity $S_{\rm row}$ at $\rho=0.75$")
    save(f, path)


def f2_depth16(rows75):
    """The nine depth-16 cells, one dot per seed, J2 accented."""
    f, ax, path = fig("H2_depth16.png")
    d16 = [r for r in rows75 if r["depth"] == DEPTH16]
    cells = sorted({r["cell"] for r in d16},
                   key=lambda c: med([r["S_row"] for r in d16
                                      if r["cell"] == c]))
    for i, c in enumerate(cells):
        ys = [r["S_row"] for r in d16 if r["cell"] == c]
        col = SERIES[0] if c == J2_LABEL else MUTED
        ax.scatter([i] * len(ys), ys, s=34, c=col, zorder=3)
        ax.scatter([i], [med(ys)], s=90, marker="_", c=col, zorder=4)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c.replace("L=16 h=4 ", "") for c in cells],
                       rotation=38, ha="right", fontsize=8)
    style_axes(ax, ylabel=r"$S_{\rm row}$ at $\rho=0.75$")
    save(f, path)


def f3_shint(rows, passing):
    """Pooled hop-2 selectivity across the three densities."""
    f, ax, path = fig("H3_shint_vs_rho.png")
    for i, (name, sel) in enumerate((("gate-passing", True),
                                     ("below gate", False))):
        xs, ys = [], []
        for rho in sorted(RHOS):
            sub = [r for r in rows if round(r["rho"], 4) == rho
                   and (r["cell"] in passing) == sel]
            if sub:
                xs.append(rho)
                ys.append(pooled(sub, "S_hint"))
        ax.plot(xs, ys, "-o", ms=4, color=SERIES[i], rasterized=True)
        label_line(ax, xs[-1], ys[-1], name, color=SERIES[i])
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=":")
    style_axes(ax, xlabel=r"hint density $\rho$",
               ylabel=r"hop-2 selectivity $S_{\rm hint}$")
    ax.set_xlim(0.55, 1.00)
    save(f, path)


def f4_slots(rows75, passing):
    """Mean attention per neighbour slot, by class, pooled over the gate."""
    f, ax, path = fig("H4_slot_classes.png")
    sub = [r for r in rows75 if r["cell"] in passing]
    names = ["critical", "droppable", "blank"]
    keys = ["crit_all", "drop_all", "blank_all"]
    ys = [ratio(sum(r["mass_" + k] for r in sub),
                sum(r["n_" + k] for r in sub)) for k in keys]
    ax.bar(names, ys, width=0.54, color=series_colors(3, highlight=1),
           rasterized=True)
    for x, y in zip(names, ys):
        ax.annotate("%.4f" % y, xy=(x, y), xytext=(0, 4), ha="center",
                    textcoords="offset points", fontsize=9)
    style_axes(ax, ylabel="mean attention mass per neighbour token")
    save(f, path)


def f5_n50(rows75, passing):
    """How many (layer, head) pairs carry half the own-row mass."""
    f, ax, path = fig("H6_n50.png")
    vals = [r["n50"] for r in rows75 if r["cell"] in passing]
    hi = int(max(vals)) if vals else 1
    edges = np.arange(0.5, hi + 1.5, 1.0)
    ax.hist(vals, bins=edges, color=SERIES[0], rasterized=True)
    ax.axvline(H6_BAR + 0.5, color=SERIES[1], lw=1.2, ls="--")
    ax.annotate("registered bar $\\leq 4$", xy=(H6_BAR + 0.5, 0),
                xytext=(5, 10), textcoords="offset points",
                color=SERIES[1], fontsize=9)
    ax.set_xticks(range(1, hi + 1))
    style_axes(ax, xlabel=r"$n_{50}$: pairs holding half the own-row mass",
               ylabel="gate-passing runs")
    save(f, path)


def f6_headmap(byhead, run_id, title_cell):
    """Own-row mass per (layer, head) for one run — the circuit, drawn."""
    f, ax, path = fig("H6_headmap.png")
    sub = [b for b in byhead if b["run_id"] == run_id
           and abs(fnum(b, "rho") - MAIN) < 1e-9]
    if not sub:
        plt.close(f)
        return
    L = int(max(fnum(b, "layer") for b in sub)) + 1
    H = int(max(fnum(b, "head") for b in sub)) + 1
    m = np.zeros((H, L))
    nq = max(fnum(b, "n_q") for b in sub)
    for b in sub:
        m[int(fnum(b, "head")), int(fnum(b, "layer"))] = \
            fnum(b, "own_row") / nq
    im = ax.imshow(m, aspect="auto", origin="lower", cmap="magma",
                   vmin=0.0, rasterized=True)
    cb = f.colorbar(im, ax=ax)
    cb.set_label("own-row attention mass", fontsize=9)
    ax.set_xticks(range(L))
    ax.set_yticks(range(H))
    style_axes(ax, xlabel="block application (depth)", ylabel="head")
    ax.grid(False)
    ax.set_title(title_cell, fontsize=9, color=MUTED)
    save(f, path)


def f7_layer_profile(byhead, pairs):
    """Own-row mass by depth for a passing run and a failing one."""
    f, ax, path = fig("H1_layer_profile.png")
    for i, (run_id, name) in enumerate(pairs):
        sub = [b for b in byhead if b["run_id"] == run_id
               and abs(fnum(b, "rho") - MAIN) < 1e-9]
        if not sub:
            continue
        nq = max(fnum(b, "n_q") for b in sub)
        by_l = collections.defaultdict(float)
        nh = len({int(fnum(b, "head")) for b in sub})
        for b in sub:
            by_l[int(fnum(b, "layer"))] += fnum(b, "own_row") / nq / nh
        xs = sorted(by_l)
        ys = [by_l[x] for x in xs]
        ax.plot([x + 1 for x in xs], ys, "-o", ms=3.5, color=SERIES[i],
                rasterized=True)
        label_line(ax, xs[-1] + 1, ys[-1], name, color=SERIES[i])
    ax.axhline(6.0 / 288.0, color=MUTED, lw=1.0, ls=":")
    ax.annotate("uniform over the 48 rows", xy=(1, 6.0 / 288.0),
                xytext=(4, 5), textcoords="offset points",
                color=MUTED, fontsize=9)
    style_axes(ax, xlabel="block application (depth)",
               ylabel="mean own-row mass per head")
    save(f, path)


# ------------------------------------------------------------------ scoring

def main():
    os.makedirs(FIGS, exist_ok=True)
    raw = read_shards(OUT, "heads_runs_*of*.csv")
    byhead = read_shards(OUT, "heads_byhead_*of*.csv")
    if not raw:
        print("no shards in", OUT)
        return
    rows_all = [derive(r) for r in raw]
    cell_rho50, passing, _ = load_gate()

    out = []
    rows = [r for r in rows_all if r["in79"]]
    excluded = [r for r in rows_all if not r["in79"]]
    n_runs = len({r["run_id"] for r in rows})
    rows75 = [r for r in rows if abs(r["rho"] - MAIN) < 1e-9]
    pass75 = [r for r in rows75 if r["cell"] in passing]
    fail75 = [r for r in rows75 if r["cell"] not in passing]

    out.append("STAGE HEADS (105) — where the colour step looks")
    out.append("=" * 72)
    out.append("")
    out.append("Registered: runs/stageheads_2026-11-19/REGISTERED.md "
               "(commit 3acf78e, before the probe and before this scorer).")
    out.append("%d runs over %d cells, rho in %s, %d queries per (run, rho)."
               % (n_runs, len({r['cell'] for r in rows}), RHOS,
                  int(med([r["n_q"] for r in rows]))))
    out.append("Gate-passing: %d cells, %d runs at rho %.2f.  Below gate: %d."
               % (len({r['cell'] for r in pass75}), len(pass75), MAIN,
                  len(fail75)))
    out.append("")

    # ---------------------------------------------------------------- E2/E3
    e2 = sum(r["e2_bad"] for r in rows_all)
    e3 = sum(r["e3_bad"] for r in rows_all)
    out.append("DIAGNOSTICS")
    out.append("-" * 72)
    out.append("E2  distribution        %s  %d queries with a bad row sum or "
               "mass above q" % ("PASS" if e2 == 0 else "FAIL", e2))
    out.append("E3  layout agreement    %s  %d queries where token_positions "
               "and encode_instance disagree" % ("PASS" if e3 == 0 else
                                                 "FAIL", e3))
    col_frac = [r["e1_color_frac"] for r in rows_all]
    right_pass = [r["e1_right_frac"] for r in rows_all
                  if r["cell"] in passing and r["in79"]]
    e1a = min(col_frac) >= E1_COLOR
    e1b = min(right_pass) >= E1_RIGHT if right_pass else False
    out.append("E1  argmax is a COLOR   %s  min %.4f over all %d (run, rho) "
               "(bar %.2f)" % ("PASS" if e1a else "FAIL", min(col_frac),
                               len(col_frac), E1_COLOR))
    out.append("E1  argmax is CORRECT   %s  min %.4f over the gate-passing "
               "(run, rho) (bar %.2f)" % ("PASS" if e1b else "FAIL",
                                          min(right_pass) if right_pass
                                          else float("nan"), E1_RIGHT))
    out.append("")

    # ------------------------------------------------------------------- H1
    sr_cell = cell_values(rows75, "S_row")
    p_vals = [v for c, v in sr_cell.items() if c in passing]
    f_vals = [v for c, v in sr_cell.items() if c not in passing]
    h1 = ratio(med(p_vals), med(f_vals))
    v1 = ("HIT" if h1 >= H1_BAR else
          "MISS" if h1 <= H1_FALSIFIER else "SPLIT")
    out.append("PREDICTIONS")
    out.append("-" * 72)
    out.append("H1  hop 1 tracks the gate      %-5s  ratio %.3f  "
               "(bar >= %.1f, falsifier <= %.1f)" % (v1, h1, H1_BAR,
                                                     H1_FALSIFIER))
    out.append("      median S_row  gate-passing %.3f over %d cells  "
               "[%.3f, %.3f]" % (med(p_vals), len(p_vals), *spread(p_vals)))
    out.append("      median S_row  below gate   %.3f over %d cells  "
               "[%.3f, %.3f]" % (med(f_vals), len(f_vals), *spread(f_vals)))
    out.append("      SIGN WAS DISCLOSED as near-certain in the registration: "
               "a 3-instance smoke had already")
    out.append("      seen ~1 for L=3 full and ~5 for L=12 full.  Only the "
               "magnitude over %d cells was live." % len(sr_cell))

    # ------------------------------------------------------------------- H2
    d16 = [r for r in rows75 if r["depth"] == DEPTH16]
    d16_cells = sorted({r["cell"] for r in d16})
    j2_seeds = [r["S_row"] for r in d16 if is_cell(r, J2_LABEL)]
    other = [med([r["S_row"] for r in d16 if r["cell"] == c])
             for c in d16_cells if c != J2_LABEL]
    bar2 = med(other)
    beats = sum(1 for v in j2_seeds if v > bar2)
    h2 = ("HIT" if beats == len(j2_seeds) and j2_seeds else
          "MISS" if beats == 0 else "SPLIT")
    out.append("")
    out.append("H2  and it is not only depth   %-5s  %d of %d seeds beat the "
               "other depth-16 cells" % (h2, beats, len(j2_seeds)))
    out.append("      %s seeds: %s" % (J2_LABEL,
                                       ", ".join("%.3f" % v
                                                 for v in sorted(j2_seeds))))
    out.append("      median of the other %d depth-16 cells: %.3f  "
               "[%.3f, %.3f]" % (len(other), bar2, *spread(other)))
    if v1 in ("HIT", "SPLIT") and h2 == "MISS":
        out.append("      THE REGISTERED READING APPLIES: H1 is confounded "
                   "with depth and H2 is the control, so")
        out.append("      hop-1 selectivity is a DEPTH effect and this stage "
                   "has not shown it tracks competence.")

    # ---------------------------------------------------------- H3, H4, H5
    for name, kind, band, fals, line in (
            ("H3  hop 2 exists", "S_hint", (H3_BAR, None),
             (H3_FALSIFIER, None), "pooled S_hint"),
            ("H4  complete in EFFORT", "R_dc", H4_BAND, H4_FALSIFIER,
             "pooled R_dc"),
            ("H5  blanks discriminated", "R_ab", (H5_BAR, None),
             (H5_FALSIFIER, None), "pooled R_ab")):
        p = pooled(pass75, kind)
        m = med([r[kind] for r in pass75])
        if band[1] is None:
            v = "HIT" if p >= band[0] else "MISS" if p <= fals[0] else "SPLIT"
            bandtxt = "bar >= %.2f, falsifier <= %.2f" % (band[0], fals[0])
        else:
            v = ("HIT" if band[0] <= p <= band[1] else
                 "MISS" if (p <= fals[0] or p >= fals[1]) else "SPLIT")
            bandtxt = ("band [%.2f, %.2f], falsifier <= %.2f or >= %.2f"
                       % (band[0], band[1], fals[0], fals[1]))
        out.append("")
        out.append("%-30s %-5s  %s %.3f  (%s)"
                   % (name, v, line, p, bandtxt))
        out.append("      unweighted median over the same %d runs: %.3f  "
                   "[%.3f, %.3f]" % (len(pass75), m,
                                     *spread([r[kind] for r in pass75])))
        pf = pooled(fail75, kind)
        out.append("      below the gate, pooled: %.3f  (not a registered "
                   "prediction; printed for contrast)" % pf)
        if band[1] is None:
            agree = (m >= band[0]) == (p >= band[0])
        else:
            agree = (band[0] <= m <= band[1]) == (band[0] <= p <= band[1])
        if not agree:
            out.append("      WEIGHTING MATTERS HERE: the mass-weighted pool "
                       "and the unweighted median fall on")
            out.append("      opposite sides of the band.  The registered "
                       "quantity is the pooled one and that is")
            out.append("      what is scored, but this verdict should be read "
                       "as weaker than its label.")

    # ------------------------------------------------------------------- H6
    n50s = [r["n50"] for r in pass75]
    m6 = med(n50s)
    h6 = ("HIT" if m6 <= H6_BAR else "MISS" if m6 >= H6_FALSIFIER
          else "SPLIT")
    out.append("")
    out.append("H6  the circuit is small       %-5s  median n_50 %.1f of "
               "L x H (bar <= %d, falsifier >= %d)" % (h6, m6, H6_BAR,
                                                       H6_FALSIFIER))
    out.append("      over %d gate-passing runs, range [%.0f, %.0f]; "
               "L x H median %d" % (len(n50s), *spread(n50s),
                                    int(med([r["n_pairs"]
                                             for r in pass75]))))
    out.append("      top single pair holds %.3f of the own-row mass "
               "(median over the same runs)"
               % med([r["top1_frac"] for r in pass75]))

    # ------------------------------------------------------- context, not H
    out.append("")
    out.append("NOT PREDICTIONS — printed because they are the same measurement")
    out.append("-" * 72)
    out.append("named cell's row is the TOP row of 48:  %.3f of queries "
               "(gate-passing), %.3f (below)"
               % (med([r["rank1_frac"] for r in pass75]),
                  med([r["rank1_frac"] for r in fail75])))
    out.append("mean rank of that row among 48:          %.2f "
               "(gate-passing), %.2f (below)"
               % (med([r["mean_rank"] for r in pass75]),
                  med([r["mean_rank"] for r in fail75])))
    for rho in sorted(RHOS, reverse=True):
        sub = [r for r in rows if abs(r["rho"] - rho) < 1e-9
               and r["cell"] in passing]
        if sub:
            out.append("rho %.2f pooled:  S_row %6.3f   S_hint %5.3f   "
                       "R_dc %5.3f   R_ab %5.3f"
                       % (rho, pooled(sub, "S_row"), pooled(sub, "S_hint"),
                          pooled(sub, "R_dc"), pooled(sub, "R_ab")))

    if excluded:
        out.append("")
        out.append("THE 80th RUN, reported separately as every stage has done")
        out.append("-" * 72)
        for r in sorted(excluded, key=lambda r: -r["rho"]):
            out.append("  %s  rho %.2f  S_row %.3f  S_hint %.3f  R_dc %.3f  "
                       "R_ab %.3f  n50 %.0f"
                       % (r["cell"], r["rho"], r["S_row"], r["S_hint"],
                          r["R_dc"], r["R_ab"], r["n50"]))

    out.append("")
    out.append("SCOPE, from the registration — none of this is weakened by a hit")
    out.append("-" * 72)
    out.append("Attention is not causation: mass on OWNROW says where the "
               "model looks, not that it uses it.")
    out.append("It does not separate adjacency-local inhibition from "
               "candidate-set computation; 104 disclosed")
    out.append("those as the same function under two names.")
    out.append("It is the COLOUR step on FORCED cells only (c = 1); nothing "
               "here touches the cell step.")
    out.append("%d queries per (run, rho) resolves no per-instance rate below "
               "~0.5%%." % int(med([r["n_q"] for r in rows])))

    text = "\n".join(out) + "\n"
    with open(os.path.join(OUT, "VERDICT.txt"), "w") as fh:
        fh.write(text)
    print(text)

    # ------------------------------------------------------------- the CSVs
    keys = sorted(rows_all[0].keys())
    with open(os.path.join(OUT, "heads_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rho50_pub_cell"] + keys)
        w.writeheader()
        for r in sorted(rows_all, key=lambda r: (r["cell"], r["rho"])):
            row = dict(r)
            row["rho50_pub_cell"] = cell_rho50.get(r["cell"], float("nan"))
            w.writerow(row)
    print("  wrote heads_runs.csv (%d rows)" % len(rows_all))

    # -------------------------------------------------------------- figures
    best = min(pass75, key=lambda r: cell_rho50.get(r["cell"], 9.0))
    worst = max(fail75, key=lambda r: cell_rho50.get(r["cell"], -1.0))
    f1_gate(rows75, cell_rho50, passing)
    f7_layer_profile(byhead, [(best["run_id"], best["cell"]),
                              (worst["run_id"], worst["cell"])])
    f2_depth16(rows75)
    f3_shint(rows, passing)
    f4_slots(rows75, passing)
    f5_n50(rows75, passing)
    f6_headmap(byhead, best["run_id"], best["cell"] + r"  at $\rho=0.75$")
    print("  figures in", FIGS)


if __name__ == "__main__":
    main()
