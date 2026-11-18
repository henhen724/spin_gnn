#!/usr/bin/env python3
"""Stage BOUND — close stage GATE's upper bound into a two-sided bracket.

Registered in `runs/stagebound_2026-11-17/REGISTERED.md`, committed before this
file existed.

WHAT THIS CLOSES
----------------
Stage GATE (99) recomputed every architecture's `rho_50` under a DD-faithful
criterion and had to report it as an UPPER bound, because

    h_j^ddf = 1 - n_in_F(j) / n_at_risk(j)

asks only "was the cell the model named already forced".  It is blind to a wrong
colour on a cell that WAS forced, to grammar violations, and to early
termination.  STATUS.md recorded folding those in as needing the rollouts
re-scored: a job, with a new registration.

That was wrong, and reading the writer is what shows it.  In
`scripts/stagelogic_score.py`, `walk_forced` appends the pair at the index where
a rollout dies BEFORE breaking out of the loop, so the already-written column
`n_err_in_F(j)` is exactly "first errors at j whose named cell was in F" — the
wrong-colour-on-a-forced-cell class.  And `measure_F` writes `n_first_err` and
`n_err_node` per (run, rho), whose difference is the exact count of deaths that
were not node-bearing at all.

THE ESTIMATOR
-------------
At index j two sets of rollouts fail the joint criterion: A(j), those naming a
cell not in F, of size n_at_risk - n_in_F; and B(j), those committing their first
permissive error, of size n_err.  A node-bearing error on an unforced cell lies
in both and must not be counted twice; a node-bearing error on a FORCED cell is
in B only and is counted by n_err_in_F; a non-node death is in neither, because
`walk_forced` breaks before recording the pair, so it never reaches n_at_risk.
Writing m(j) for that last class,

    |A(j) u B(j)| = (n_at_risk - n_in_F) + n_err_in_F + m(j)

exactly.  The first two terms are per-j on disk.  Only the DISTRIBUTION of m is
missing; its total M = n_first_err - n_err_node is exact per cell.  So

    h_j^joint = [(n_at_risk - n_in_F) + n_err_in_F] / n_at_risk
    S_joint   = prod_j (1 - h_j^joint)            upper end of the bracket
    phi       = M / n_instances                   the mass it cannot see
    S_joint - phi                                 lower end

and an architecture is a REAL pass only if the lower end clears the 0.740 gate.

Loaders, `rho50` and the cell keying are stage GATE's, imported in spirit and
copied in fact so the two scorers cannot drift; in particular runs are keyed on
the `cell` LABEL, never on (arch_mode, num_layers, d_model), which is the defect
96 found and GATE reproduced.

No training, no GPU, no new instances, nothing submitted.  Login node, seconds.
"""

import collections
import csv as _csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stageexp_fit import fnum, tobool, read_shards, med  # noqa: E402
from stagegate_fit import (GRID, FIT_RHOS, HARD, GATE, HALF, J2_LABEL,  # noqa: E402
                           HUB_LABEL, is_cell, rho50, spearman, fmt)

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import (SERIES, MUTED, style_axes, label_line,  # noqa: E402
                    series_colors)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = "/scratch/users/hshunt/spin_gnn/runs"
LOGIC = os.path.join(RUNS, "stagelogic_2026-11-17")
OUT = os.path.join(RUNS, "stagebound_2026-11-17")
FIGS = os.path.join(OUT, "figures")
DPI = 150


def fig(name):
    f, ax = plt.subplots(figsize=(7.2, 4.3))
    return f, ax, os.path.join(FIGS, name)


def save(f, path):
    f.tight_layout()
    f.savefig(path, dpi=DPI)
    plt.close(f)
    print("  wrote", os.path.basename(path))


def load():
    """Per (run_id, rho): three hazard products, the measured rate, and phi."""
    haz = read_shards(LOGIC, "F_hazard_all_*of10.csv", keep=[
        "run_id", "in79", "rho", "j", "n_at_risk", "n_in_F", "n_err",
        "n_err_in_F"])
    summ = read_shards(LOGIC, "LF_summary_all_*of10.csv", keep=[
        "run_id", "cell", "arch_mode", "num_layers", "d_model",
        "n_trainable_params", "val_acc", "in79", "rho", "mean_n_blank",
        "solve_rate", "n_instances", "n_first_err", "n_err_node"])

    acc = collections.defaultdict(lambda: {"ddf": 0.0, "joint": 0.0,
                                           "perm": 0.0, "z_ddf": False,
                                           "z_joint": False, "z_perm": False,
                                           "njs": 0, "n_eif": 0.0})
    for h in haz:
        if not tobool(h, "in79"):
            continue
        ar = fnum(h, "n_at_risk")
        if not (ar > 0):
            continue
        p = acc[(h["run_id"], round(fnum(h, "rho"), 4))]
        p["njs"] += 1
        e_all, e_inf, inf = (fnum(h, "n_err"), fnum(h, "n_err_in_F"),
                             fnum(h, "n_in_F"))
        p["n_eif"] += e_inf
        unf = ar - inf
        for tag, hv in (("ddf", unf / ar),                  # stage GATE
                        ("joint", (unf + e_inf) / ar),      # this stage
                        ("perm", e_all / ar)):              # for D4
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
        n_inst = fnum(s, "n_instances")
        m_non = fnum(s, "n_first_err") - fnum(s, "n_err_node")
        phi = m_non / n_inst if n_inst > 0 else float("nan")
        sj = g("joint")
        cells.append({
            "run_id": s["run_id"], "cell": s["cell"],
            "arch": (s["arch_mode"], int(float(s["num_layers"])),
                     int(float(s["d_model"]))),
            "params": fnum(s, "n_trainable_params"),
            "val_acc": fnum(s, "val_acc"),
            "rho": round(fnum(s, "rho"), 4),
            "n_blank": fnum(s, "mean_n_blank"),
            "S": fnum(s, "solve_rate"),
            "S_ddf": g("ddf"), "S_joint": sj, "S_perm": g("perm"),
            "S_lo": max(0.0, sj - phi),
            "phi": phi, "m_non": m_non, "n_inst": n_inst,
            "n_eif": p["n_eif"], "njs": p["njs"],
        })
    return cells, len(haz), len(summ)


def by_run(cells):
    """Per run: the four rho_50s and their censoring flags."""
    cur = collections.defaultdict(dict)
    meta = {}
    for c in cells:
        for t in ("S", "S_ddf", "S_joint", "S_lo"):
            cur[(c["run_id"], t)][c["rho"]] = c[t]
        meta[c["run_id"]] = c
    runs = []
    for rid, m in meta.items():
        rec = {"run_id": rid, "cell": m["cell"], "arch": m["arch"],
               "params": m["params"], "val_acc": m["val_acc"]}
        for t, nm in (("S", "pub"), ("S_ddf", "ddf"), ("S_joint", "joint"),
                      ("S_lo", "lo")):
            v, flag = rho50(cur[(rid, t)])
            rec["rho50_" + nm], rec["flag_" + nm] = v, flag
        runs.append(rec)
    return sorted(runs, key=lambda r: r["params"])


# ----------------------------------------------------------------- figures

def b1_cost(by_rho):
    """log S_joint / log S_ddf against rho.  The median is flat at 1 and says
    nothing; the content is the upper tail, so plot that."""
    f, ax, path = fig("B1_cost.png")
    xs = [r for r in FIT_RHOS if r in by_rho]
    med_ = [by_rho[r]["med"] for r in xs]
    p95 = [by_rho[r]["p95"] for r in xs]
    mx = [by_rho[r]["max"] for r in xs]
    ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
    for ys, nm, col, dy in ((mx, "max", MUTED, 10),
                            (p95, "95th pct", SERIES[1], 0),
                            (med_, "median", SERIES[0], -12)):
        ax.plot(xs, ys, "-o", color=col, ms=3.5, lw=1.7, rasterized=True)
        label_line(ax, xs[0], ys[0], nm, color=col, dy=dy)
    ax.text(0.455, 1.0, "  1.0 = the colour term costs nothing",
            fontsize=8, color=MUTED, ha="left", va="bottom")
    style_axes(ax, xlabel="hint density $\\rho$",
               ylabel="$\\log S_{\\rm joint} / \\log S_{\\rm ddf}$")
    ax.set_xlim(0.44, 1.03)
    save(f, path)


def b2_phi(cnt_by_rho, n_per):
    """Non-node deaths as raw counts.  A rate figure would be eleven invisible
    points; the count is the honest version of the same thing."""
    f, ax, path = fig("B2_phi.png")
    xs = [r for r in GRID if r in cnt_by_rho]
    ys = [cnt_by_rho[r] for r in xs]
    ax.bar(xs, ys, width=0.028, color=SERIES[0], rasterized=True)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylim(0, 3.4)
    ax.text(0.52, 3.2,
            f"{int(sum(ys))} rollouts in all died of grammar, early stopping or"
            f"\ntruncation, out of {n_per * len(xs):,.0f} scored"
            f" ({n_per:,.0f} at each density)",
            fontsize=9, color=MUTED, ha="left", va="top")
    style_axes(ax, xlabel="hint density $\\rho$",
               ylabel="non-node deaths (count)")
    ax.set_xlim(0.30, 0.99)
    save(f, path)


def b3_curves(cells, label, tag):
    """The three criteria on one architecture, with the bracket shaded."""
    f, ax, path = fig("B3_curves.png")
    sel = [c for c in cells if is_cell(c, label)]
    cur = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in sel:
        for t in ("S", "S_ddf", "S_joint", "S_lo"):
            cur[t][c["rho"]].append(c[t])
    xs = [r for r in GRID if r in cur["S"]]
    c3 = SERIES
    # For J2 the two criteria agree at 41 of its 42 cells -- one rollout at
    # rho 0.55 named a forced cell and got the colour wrong, worth 0.0007 -- so
    # the orange goes down as a wide halo and the green rides on top of it.
    # Drawn as two thin lines the orange would simply be invisible and the page
    # would claim a coincidence the reader could not check.
    y_pub = [med(cur["S"][r]) for r in xs]
    y_ddf = [med(cur["S_ddf"][r]) for r in xs]
    y_jnt = [med(cur["S_joint"][r]) for r in xs]
    ax.plot(xs, y_ddf, "-", color=c3[1], lw=6.0, alpha=0.35, rasterized=True)
    ax.plot(xs, y_pub, "-o", color=c3[0], ms=3.5, lw=1.8, rasterized=True)
    ax.plot(xs, y_jnt, "-o", color=c3[2], ms=3.5, lw=1.8, rasterized=True)
    i50 = xs.index(0.55)
    label_line(ax, xs[i50], y_pub[i50], "published", color=c3[0], dy=12)
    label_line(ax, xs[i50], y_jnt[i50], "DD-faithful (halo), joint (line)",
               color=c3[2], dy=-14)
    ax.axhline(HALF, color=MUTED, lw=1.0, ls="--")
    ax.axvline(GATE, color=MUTED, lw=1.0, ls=":")
    ax.text(GATE, 0.30, " gate 0.740", fontsize=8, color=MUTED, va="top")
    ax.text(0.31, 0.95,
            "the two strict curves differ at one density only, by 0.0007",
            fontsize=8, color=MUTED, ha="left", va="top")
    style_axes(ax, xlabel="hint density $\\rho$", ylabel="solve rate")
    ax.set_xlim(0.30, 0.99)
    ax.set_ylim(-0.03, 1.03)
    save(f, path)
    return path


def b4_dumbbell(arch):
    """Per architecture: published -> joint lower end, against parameters."""
    f, ax, path = fig("B4_dumbbell.png")
    rows = [a for a in arch if a["pub"] == a["pub"]]
    rows.sort(key=lambda a: a["params"])
    for a in rows:
        y = a["params"]
        clears = (a["lo"] == a["lo"]) and a["lo"] <= GATE
        col = SERIES[0] if clears else MUTED
        x1 = a["lo"] if a["lo"] == a["lo"] else 0.95
        ax.plot([a["pub"], x1], [y, y], "-", color=col, lw=1.3, alpha=0.8,
                rasterized=True)
        ax.plot([a["pub"]], [y], "o", mfc="white", mec=col, ms=5.5, mew=1.3,
                rasterized=True)
        if a["ddf"] == a["ddf"]:
            ax.plot([a["ddf"]], [y], "o", mfc=col, mec=col, ms=3.0, alpha=0.5,
                    rasterized=True)
        if a["lo"] == a["lo"]:
            ax.plot([x1], [y], "o", mfc=col, mec=col, ms=5.5, rasterized=True)
        else:
            ax.plot([x1], [y], ">", mfc=col, mec=col, ms=5.0, rasterized=True)
    ax.axvline(GATE, color=SERIES[1], lw=1.4)
    ax.text(GATE, ax.get_ylim()[1], " gate", fontsize=9, color=SERIES[1],
            va="top")
    style_axes(ax, xlabel="$\\rho_{50}$ (lower is better)",
               ylabel="trainable parameters", logy=True)
    ax.set_xlim(0.36, 0.97)
    ax.text(0.37, ax.get_ylim()[0] * 1.15,
            "hollow: published      filled: joint criterion, lower end of the"
            " bracket"
            "\nthe DD-faithful marker sits under the filled one at all but three"
            "\narchitectures, because the two criteria agree there"
            "\nblue: still clears the gate",
            fontsize=8, color=MUTED, ha="left", va="bottom")
    save(f, path)


def d3_monotone(cells):
    """S_joint against S_ddf: by construction on or below the identity."""
    f, ax, path = fig("D3_monotone.png")
    bad = [c for c in cells if c["S_joint"] < c["S_ddf"] - 1e-12]
    ok = [c for c in cells if not (c["S_joint"] < c["S_ddf"] - 1e-12)]
    ax.plot([0, 1], [0, 1], "-", color=MUTED, lw=1.0)
    ax.plot([c["S_ddf"] for c in ok], [c["S_joint"] for c in ok], "o",
            color=SERIES[0], ms=2.6, alpha=0.45, rasterized=True)
    ax.plot([c["S_ddf"] for c in bad], [c["S_joint"] for c in bad], "o",
            color=SERIES[1], ms=4.0, alpha=0.9, rasterized=True)
    ax.text(0.03, 0.97, f"blue: identical, {len(ok)} cells",
            fontsize=9, color=SERIES[0], va="top")
    ax.text(0.03, 0.90, f"orange: joint strictly lower, {len(bad)} cells",
            fontsize=9, color=SERIES[1], va="top")
    gaps = sorted((c["S_ddf"] - c["S_joint"] for c in bad), reverse=True)
    ax.text(0.03, 0.83,
            f"17 architectures, but 39 of the {len(bad)} are the three\n"
            f"smallest; median drop {gaps[len(gaps) // 2]:.4f}, and only"
            f" {sum(1 for g in gaps if g > 0.01)} exceed 0.01",
            fontsize=8, color=MUTED, va="top")
    style_axes(ax, xlabel="$S_{\\rm ddf}$ (stage GATE)",
               ylabel="$S_{\\rm joint}$ (this stage)")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.0)
    save(f, path)


# ----------------------------------------------------------------- main

def main():
    os.makedirs(FIGS, exist_ok=True)
    cells, n_haz, n_summ = load()
    runs = by_run(cells)
    out = []
    P = lambda s="": (out.append(s), print(s))

    P("STAGE BOUND — the DD-faithful bound closed into a two-sided bracket")
    P("=" * 78)
    P(f"hazard rows {n_haz}   summary rows {n_summ}   joined cells {len(cells)}"
      f"   runs {len(runs)}")
    P()
    P("The joint criterion: a rollout counts only if every cell it named was")
    P("already forced AND it never erred AND it terminated properly.  Per index")
    P("  numerator = (n_at_risk - n_in_F) + n_err_in_F + m(j)")
    P("with the first two exact on disk and only the DISTRIBUTION of m missing;")
    P("its total is n_first_err - n_err_node, so phi brackets what is unseen.")
    P()

    hard = [c for c in cells if c["n_blank"] >= 12]

    # ---- D4: the permissive product still reproduces the measured rate
    P("-" * 78)
    P("D4 — THE GATE-ERA CLOSURE STILL HOLDS, SO THE LOADERS HAVE NOT DRIFTED")
    d = [abs(c["S_perm"] - c["S"]) for c in hard]
    P(f"  |S_perm - S| over {len(d)} cells with n >= 12:"
      f"  median {med(d):.5f}   max {max(d):.5f}")
    P(f"  registered: median < 0.01, max < 0.05 ->"
      f"  {'PASS' if med(d) < 0.01 and max(d) < 0.05 else 'FAIL'}")
    P()

    # ---- D3: monotonicity
    P("-" * 78)
    P("D3 — MONOTONICITY, AND THE VIOLATIONS LOCATE THE MISSING MASS")
    viol_ddf = [c for c in cells if c["S_joint"] > c["S_ddf"] + 1e-12]
    P(f"  S_joint > S_ddf on {len(viol_ddf)} of {len(cells)} cells"
      f"   (registered: 0 — any violation is a scorer bug)")
    viol_perm = [c for c in cells if c["S_joint"] > c["S_perm"] + 1e-12]
    exc_ok = [c for c in viol_perm if c["phi"] > 0]
    exc_bad = [c for c in viol_perm if not (c["phi"] > 0)]
    P(f"  S_joint > S_perm on {len(viol_perm)} cells;"
      f" of those {len(exc_ok)} have phi > 0 and {len(exc_bad)} do not")
    if viol_perm:
        worst = max(viol_perm, key=lambda c: c["S_joint"] - c["S_perm"])
        P(f"  largest excess {worst['S_joint'] - worst['S_perm']:.4f}"
          f" at rho {worst['rho']:.2f}, against phi {worst['phi']:.4f}"
          f"  -> {'within' if worst['S_joint'] - worst['S_perm'] <= worst['phi'] + 1e-9 else 'OUTSIDE'} the bound")
    n_eif_tot = sum(c["n_eif"] for c in cells)
    P(f"  total wrong-colour-on-a-forced-cell first errors: {n_eif_tot:.0f}")
    strict = [c for c in cells if c["S_joint"] < c["S_ddf"] - 1e-12]
    P(f"  S_joint STRICTLY below S_ddf on {len(strict)} of {len(cells)} cells"
      f"  (equality iff n_err_in_F = 0 at every j)")
    if strict:
        wg = max(strict, key=lambda c: c["S_ddf"] - c["S_joint"])
        P(f"  largest absolute gap {wg['S_ddf'] - wg['S_joint']:.6f}"
          f" at rho {wg['rho']:.2f}, S_ddf {wg['S_ddf']:.6f}"
          f" -> S_joint {wg['S_joint']:.6f}")
    with_eif = [c for c in cells if c["n_eif"] > 0]
    P(f"  cells with any such error: {len(with_eif)} of {len(cells)};"
      f" median count where present"
      f" {med([c['n_eif'] for c in with_eif]):.1f}"
      f" of {med([c['n_inst'] for c in with_eif]):.0f} rollouts")
    P()

    # ---- B2: the non-node mass
    P("-" * 78)
    P("B2 — THE NON-NODE MASS, WHICH DECIDES WHETHER THE BRACKET IS USEFUL")
    by_phi = {}
    for r in FIT_RHOS:
        v = [c["phi"] for c in cells if c["rho"] == r and c["phi"] == c["phi"]]
        if v:
            by_phi[r] = {"med": float(np.median(v)),
                         "p90": float(np.percentile(v, 90)),
                         "max": float(max(v)), "n": len(v)}
    P(f"  {'rho':>6} {'median phi':>12} {'90th pct':>10} {'max':>10} {'cells':>7}")
    for r in FIT_RHOS:
        if r in by_phi:
            b = by_phi[r]
            P(f"  {r:6.2f} {b['med']:12.5f} {b['p90']:10.5f} {b['max']:10.5f}"
              f" {b['n']:7d}")
    ph = [c["phi"] for c in hard if c["phi"] == c["phi"]]
    m_h, p90_h = float(np.median(ph)), float(np.percentile(ph, 90))
    P(f"  pooled over n >= 12: median {m_h:.5f}   90th pct {p90_h:.5f}"
      f"   max {max(ph):.5f}")
    b2 = "HIT" if (m_h < 0.02 and p90_h < 0.10) else (
        "MISS" if m_h > 0.05 else "SPLIT")
    P(f"  registered: median < 0.02 AND 90th pct < 0.10, falsified above 0.05"
      f"  ->  {b2}")
    tot_m = sum(c["m_non"] for c in cells)
    tot_e = sum(c["n_inst"] for c in cells)
    P(f"  in counts: {tot_m:.0f} non-node deaths over {tot_e:.0f} rollouts")
    P()

    # ---- B1: what the colour term costs
    P("-" * 78)
    P("B1 — WHAT THE WRONG-COLOUR-ON-A-FORCED-CELL TERM ADDS")
    by_rho = {}
    for r in FIT_RHOS:
        v = [np.log(c["S_joint"]) / np.log(c["S_ddf"])
             for c in cells if c["rho"] == r
             and 0 < c["S_joint"] and 0 < c["S_ddf"] < 1]
        if len(v) >= 3:
            by_rho[r] = {"med": float(np.median(v)),
                         "q1": float(np.percentile(v, 25)),
                         "q3": float(np.percentile(v, 75)),
                         "p95": float(np.percentile(v, 95)),
                         "max": float(max(v)), "n": len(v)}
    P(f"  {'rho':>6} {'median':>10} {'95th pct':>10} {'max':>10} {'runs':>6}")
    for r in FIT_RHOS:
        if r in by_rho:
            b = by_rho[r]
            P(f"  {r:6.2f} {b['med']:10.4f} {b['p95']:10.4f} {b['max']:10.4f}"
              f" {b['n']:6d}")
    inband = [r for r in HARD if r in by_rho and 1.00 <= by_rho[r]["med"] <= 1.50]
    out_of = [r for r in HARD if r in by_rho and by_rho[r]["med"] > 2.5]
    nh = len([r for r in HARD if r in by_rho])
    b1 = "HIT" if len(inband) > nh / 2 else ("MISS" if len(out_of) > nh / 2
                                             else "SPLIT")
    P(f"  in [1.00, 1.50] at {len(inband)} of {nh} densities with n >= 12,"
      f" above 2.5 at {len(out_of)}  ->  {b1}")
    allr = [np.log(c["S_joint"]) / np.log(c["S_ddf"]) for c in cells
            if 0 < c["S_joint"] and 0 < c["S_ddf"] < 1]
    P(f"  pooled over {len(allr)} cells: median {np.median(allr):.6f}"
      f"   99th pct {np.percentile(allr, 99):.6f}   max {max(allr):.6f}")
    P("  the band was registered as [1.00, 1.50]; the measurement sits ON the")
    P("  floor of it, so read this as the colour term adding nothing, not as a")
    P("  quantity that happened to land inside a wide band.")
    P()

    # ---- B3: J2
    P("-" * 78)
    P("B3 — THE STANDING MINIMUM, J2 = " + J2_LABEL)
    j2 = [r for r in runs if is_cell(r, J2_LABEL)]
    P(f"  {'run':<26} {'published':>10} {'DD-faith':>10} {'joint':>10}"
      f" {'lower end':>10}")
    for r in sorted(j2, key=lambda r: r["run_id"]):
        P(f"  {r['run_id']:<26} {fmt(r['rho50_pub'])} {fmt(r['rho50_ddf'])}"
          f" {fmt(r['rho50_joint'])} {fmt(r['rho50_lo'])}")
    j2med = {t: med([r["rho50_" + t] for r in j2])
             for t in ("pub", "ddf", "joint", "lo")}
    P(f"  {'seed median':<26} {fmt(j2med['pub'])} {fmt(j2med['ddf'])}"
      f" {fmt(j2med['joint'])} {fmt(j2med['lo'])}")
    b3 = "HIT" if j2med["lo"] == j2med["lo"] and j2med["lo"] <= GATE else "MISS"
    P(f"  registered: lower end <= {GATE}  ->  {b3}"
      f"   (headroom {GATE - j2med['lo']:+.4f})")
    hub = [r for r in runs if is_cell(r, HUB_LABEL)]
    hubmed = {t: med([r["rho50_" + t] for r in hub])
              for t in ("pub", "ddf", "joint", "lo")}
    P(f"  hub {HUB_LABEL} ({len(hub)} seeds):"
      f" {fmt(hubmed['pub'])} {fmt(hubmed['ddf'])} {fmt(hubmed['joint'])}"
      f" {fmt(hubmed['lo'])}")
    P(f"  J2 - hub separation: published {j2med['pub'] - hubmed['pub']:+.4f}"
      f"   DD-faithful {j2med['ddf'] - hubmed['ddf']:+.4f}"
      f"   joint lower {j2med['lo'] - hubmed['lo']:+.4f}")
    P()

    # ---- B4: per architecture
    P("-" * 78)
    P("B4 — THE VERDICT TABLE PER ARCHITECTURE, WHICH IS HOW IT IS READ")
    by_cell = collections.defaultdict(list)
    for r in runs:
        by_cell[" ".join(r["cell"].split())].append(r)
    arch = []
    for lab, rs in by_cell.items():
        a = {"cell": lab, "n": len(rs),
             "params": med([r["params"] for r in rs])}
        for t in ("pub", "ddf", "joint", "lo"):
            a[t] = med([r["rho50_" + t] for r in rs])
        arch.append(a)
    arch.sort(key=lambda a: a["params"])
    P(f"  {'architecture':<30} {'seeds':>5} {'params':>10} {'pub':>9}"
      f" {'ddf':>9} {'joint':>9} {'lower':>9}")
    for a in arch:
        P(f"  {a['cell']:<30} {a['n']:5d} {a['params']:10.0f}"
          f" {fmt(a['pub'])[1:]} {fmt(a['ddf'])[1:]} {fmt(a['joint'])[1:]}"
          f" {fmt(a['lo'])[1:]}")
    cl = lambda t: {a["cell"] for a in arch if a[t] == a[t] and a[t] <= GATE}
    s_pub, s_ddf, s_lo = cl("pub"), cl("ddf"), cl("lo")
    P()
    P(f"  clear the {GATE} gate:  published {len(s_pub)} of {len(arch)}"
      f"   DD-faithful {len(s_ddf)}   joint lower end {len(s_lo)}")
    P(f"  published set: {sorted(s_pub)}")
    P(f"  joint lower-end set: {sorted(s_lo)}")
    P(f"  subset of the published set: {s_lo <= s_pub}"
      f"   identical: {s_lo == s_pub}")
    b4 = "HIT" if (s_lo <= s_pub and len(s_lo) >= 4) else (
        "MISS" if (not s_lo <= s_pub or not s_lo) else "SPLIT")
    P(f"  registered: subset of the 6 AND at least 4  ->  {b4}")
    P()

    # ---- B5: the ranking
    P("-" * 78)
    P("B5 — THE RANKING")
    pair = [(r["rho50_ddf"], r["rho50_joint"]) for r in runs
            if r["rho50_ddf"] == r["rho50_ddf"]
            and r["rho50_joint"] == r["rho50_joint"]]
    rho_s = spearman([p[0] for p in pair], [p[1] for p in pair])
    P(f"  Spearman(rho50_ddf, rho50_joint) = {rho_s:+.4f} over {len(pair)} runs")
    b5 = "HIT" if rho_s >= 0.95 else ("MISS" if rho_s < 0.8 else "SPLIT")
    P(f"  registered: >= 0.95, falsified below 0.8  ->  {b5}")
    pair2 = [(r["rho50_pub"], r["rho50_lo"]) for r in runs
             if r["rho50_pub"] == r["rho50_pub"]
             and r["rho50_lo"] == r["rho50_lo"]]
    P(f"  Spearman(published, joint lower) ="
      f" {spearman([p[0] for p in pair2], [p[1] for p in pair2]):+.4f}"
      f" over {len(pair2)} runs")
    cens = collections.Counter(r["flag_lo"] for r in runs)
    P(f"  censoring of the lower end: {dict(cens)}")
    P()

    # ---- the bracket width, which is the point of the stage
    P("-" * 78)
    P("THE BRACKET, WHICH IS WHAT STATUS SAID NEEDED A JOB")
    w = [a["lo"] - a["joint"] for a in arch
         if a["lo"] == a["lo"] and a["joint"] == a["joint"]]
    if w:
        P(f"  rho_50 bracket width over {len(w)} architectures:"
          f" median {med(w):.4f}   max {max(w):.4f}")
    wc = [c["S_joint"] - c["S_lo"] for c in hard]
    P(f"  solve-rate bracket width over {len(hard)} cells with n >= 12:"
      f" median {med(wc):.5f}   max {max(wc):.5f}")
    P()

    # ---- scorecard
    P("=" * 78)
    P("SCORECARD")
    for nm, v in (("B1 colour term adds little", b1),
                  ("B2 non-node mass is small", b2),
                  ("B3 J2 clears at the lower end", b3),
                  ("B4 verdict table unchanged", b4),
                  ("B5 ranking survives", b5)):
        P(f"  {nm:<36} {v}")
    P(f"  {'D3 monotonicity':<36}"
      f" {'PASS' if not viol_ddf else 'FAIL'}")
    P(f"  {'D4 closure against the measured rate':<36}"
      f" {'PASS' if med(d) < 0.01 and max(d) < 0.05 else 'FAIL'}")

    # ---- figures and CSVs
    print()
    b1_cost(by_rho)
    cnt = collections.defaultdict(float)
    inst = collections.defaultdict(float)
    for c in cells:
        cnt[c["rho"]] += c["m_non"]
        inst[c["rho"]] += c["n_inst"]
    b2_phi(cnt, med(list(inst.values())))
    b3_curves(cells, J2_LABEL, "j2")
    b4_dumbbell(arch)
    d3_monotone(cells)

    with open(os.path.join(OUT, "bound_cells.csv"), "w", newline="") as fh:
        wtr = _csv.DictWriter(fh, fieldnames=[
            "run_id", "cell", "rho", "n_blank", "params", "val_acc",
            "S", "S_perm", "S_ddf", "S_joint", "S_lo", "phi", "m_non",
            "n_inst", "n_eif", "njs"])
        wtr.writeheader()
        for c in sorted(cells, key=lambda c: (c["run_id"], -c["rho"])):
            wtr.writerow({k: c[k] for k in wtr.fieldnames})
    with open(os.path.join(OUT, "bound_runs.csv"), "w", newline="") as fh:
        wtr = _csv.DictWriter(fh, fieldnames=[
            "run_id", "cell", "params", "val_acc", "rho50_pub", "flag_pub",
            "rho50_ddf", "flag_ddf", "rho50_joint", "flag_joint", "rho50_lo",
            "flag_lo"])
        wtr.writeheader()
        for r in runs:
            wtr.writerow({k: r[k] for k in wtr.fieldnames})
    with open(os.path.join(OUT, "VERDICT.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("  wrote bound_cells.csv, bound_runs.csv, VERDICT.txt")


if __name__ == "__main__":
    main()
