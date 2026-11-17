"""Presentation figures for the stage LOGIC deck.

The registered figures in stagelogic_fit.py are built to be scored against
REGISTERED.md: three-panel comparisons, every series on one axis, titles that
carry the fit statistics.  A deck page wants the opposite -- one claim, one
picture, labels that do not collide.  This script redraws the eight that the
deck actually shows, reusing stagelogic_fit's loaders so there is exactly one
definition of every quantity.

Every number printed at the end must match VERDICT.txt.  Where a figure here
disagreed with the registered number, the registered one won and the
difference is explained in a comment -- see stagepos_deckfigs.py for the two
cases where that happened.

Run after stagelogic_fit.py, same module set.
"""

import collections
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line  # noqa: E402
from stagelogic_fit import (  # noqa: E402
    OUT, POS, INST, DPI, read_shards, run_table, blanks, availability,
    powerlaw, through_origin, f, i, b, FIT_RHOS, OVERLAY_RHO)

DECK = os.path.join(OUT, "deckfigs")


def stagger(ax, ends, gap=0.07):
    """Right-hand-end labels, pushed apart so two curves that finish at the
    same height do not print on top of each other."""
    lo, hi = ax.get_ylim()
    sep = gap * (hi - lo)
    ends = sorted(ends, key=lambda e: e[1])
    for k in range(1, len(ends)):
        x, y, text, col = ends[k]
        if y - ends[k - 1][1] < sep:
            ends[k] = (x, ends[k - 1][1] + sep, text, col)
    for x, y, text, col in ends:
        label_line(ax, x, y, text, color=col)


def fig(name, w=7.4, h=4.4):
    f_, ax = plt.subplots(figsize=(w, h))
    return f_, ax, os.path.join(DECK, name)


def save(f_, path):
    f_.tight_layout()
    f_.savefig(path, dpi=DPI)
    plt.close(f_)
    print("  wrote %s" % os.path.basename(path))


# --------------------------------------------------------------------------


def d1_exponent(fits, a_logic, a_strict):
    """THE headline: logical scoring lifts the exponent from 0.79n to 0.93n."""
    f_, ax, path = fig("log_d1_exponent_vs_n.png")
    rows = sorted((v for v in fits.values() if v["k_logic"] == v["k_logic"]),
                  key=lambda v: v["n"])
    n = np.array([v["n"] for v in rows])
    kl = np.array([v["k_logic"] for v in rows])
    ks = np.array([v["k_strict"] for v in rows])
    ax.plot(n, n, color=MUTED, lw=1.2, ls="--")
    ax.plot(n, 0.73 * n, color=MUTED, lw=1.0, ls=":")
    ax.errorbar(n, kl, yerr=[v["se_logic"] for v in rows],
                color=SERIES[0], lw=2.0, marker="o", ms=4.5, capsize=2)
    ax.errorbar(n, ks, yerr=[v["se_strict"] for v in rows],
                color=SERIES[1], lw=1.6, marker="s", ms=4, capsize=2)
    stagger(ax, [(n[-1], n[-1], "one error per deduction", MUTED),
                 (n[-1], kl[-1], "logical:  k = %.2f n" % a_logic, SERIES[0]),
                 (n[-1], ks[-1], "strict:  k = %.2f n" % a_strict, SERIES[1]),
                 (n[-1], 0.73 * n[-1], "k = 0.73 n, the old reading", MUTED)],
            gap=0.085)
    style_axes(ax, xlabel="blank cells  n = 48(1 - realized rho)",
               ylabel="fitted exponent k  of  S = C p^k")
    save(f_, path)


def d2_powerlaw(cells, fits, rho=0.60):
    """One density, log-log: the power law is a straight line, not a fit shape
    chosen to look good."""
    f_, ax, path = fig("log_d2_powerlaw_rho%02d.png" % round(rho * 100))
    sel = [c for c in cells.values()
           if abs(c["rho"] - rho) < 1e-9 and c["in79"]
           and 0 < c["solve_rate"] < 1 and 0 < c["p_logic"] < 1]
    x = np.array([c["p_logic"] for c in sel])
    y = np.array([c["solve_rate"] for c in sel])
    ax.scatter(x, y, s=20, color=SERIES[0], alpha=0.8, rasterized=True)
    fit = fits[round(rho, 4)]
    xs = np.linspace(x.min(), x.max(), 80)
    ax.plot(xs, fit["C_logic"] * xs ** fit["k_logic"], color=SERIES[1], lw=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    label_line(ax, xs[-1], fit["C_logic"] * xs[-1] ** fit["k_logic"],
               "k = %.1f,  R2 = %.2f" % (fit["k_logic"], fit["r2_logic"]),
               color=SERIES[1])
    style_axes(ax, xlabel="logical token accuracy  p  (rho %.2f, n = %.0f)"
               % (rho, fit["n"]), ylabel="free-generation solve rate  S")
    save(f_, path)
    return len(sel), fit


def d3_prefactor(fits):
    """C stays at 1 while the puzzle is easy and sags once it is not."""
    f_, ax, path = fig("log_d3_prefactor_vs_n.png")
    rows = sorted((v for v in fits.values() if v["C_logic"] == v["C_logic"]),
                  key=lambda v: v["n"])
    n = [v["n"] for v in rows]
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.plot(n, [v["C_logic"] for v in rows], color=SERIES[0], lw=2.0,
            marker="o", ms=4.5)
    ax.set_ylim(0.70, 1.10)
    stagger(ax, [(n[-1], rows[-1]["C_logic"], "logical", SERIES[0]),
                 (n[-1], 1.0, "C = 1: a perfect model solves everything", MUTED)],
            gap=0.10)
    style_axes(ax, xlabel="blank cells  n",
               ylabel="prefactor C of  S = C p^k")
    save(f_, path)


def d4_gain(cells):
    """How much of measurement A's error was order bookkeeping."""
    f_, ax, path = fig("log_d4_gain_hist.png")
    g = np.array([c["p_logic"] - c["p_strict"] for c in cells.values()
                  if c["p_logic"] == c["p_logic"]])
    ax.hist(g, bins=50, color=SERIES[0], alpha=0.85)
    ax.axvline(0.0, color=MUTED, lw=1.2, ls="--")
    ax.axvline(float(np.median(g)), color=SERIES[1], lw=1.6)
    label_line(ax, float(np.median(g)), ax.get_ylim()[1] * 0.80,
               "median %+.4f" % np.median(g), color=SERIES[1])
    style_axes(ax, xlabel="p_logic - p_strict, one point per (model, density)",
               ylabel="cells")
    save(f_, path)
    return len(g), float(np.median(g)), float(g.max()), int((g < 0).sum())


def d5_classes(jrows):
    """What the wrong node predictions are: premature, never a recolour."""
    f_, ax, path = fig("log_d5_classes_vs_j.png")
    agg = collections.defaultdict(collections.Counter)
    for r in jrows:
        # in79 only.  The registered L7 figure pools all 80; the registered L5
        # TABLE drops the flagged L=2 run, and the deck quotes the table, so
        # both have to use the table's population or the page contradicts
        # itself by 0.0015.
        if (abs(f(r, "rho") - OVERLAY_RHO) > 1e-9
                or r.get("tok_type") != "node" or not b(r, "in79")):
            continue
        a = agg[i(r, "index")]
        for k in ("n_in_F", "n_premature", "n_recolor", "n_nonnode", "n_total"):
            a[k] += i(r, k)
    js = sorted(j for j in agg if agg[j]["n_total"] >= 1000)
    tot = np.array([agg[j]["n_total"] for j in js], float)
    stack = [np.array([agg[j][k] for j in js], float) / tot
             for k in ("n_in_F", "n_premature", "n_recolor", "n_nonnode")]
    ax.stackplot(js, *stack, colors=[SERIES[0], SERIES[1], SERIES[2], MUTED])
    mid = js[len(js) // 2]
    m = js.index(mid)
    # Labels go INSIDE the bands.  The registered version put all four at the
    # right-hand edge, where recolour (0.00%) and non-node (0.9%) printed on
    # top of each other.
    ax.text(mid, stack[0][m] / 2, "in F: a legally forced node",
            color="white", fontsize=11, ha="center", va="center")
    ax.text(mid, stack[0][m] + stack[1][m] / 2, "premature",
            color="white", fontsize=11, ha="center", va="center")
    ax.set_ylim(0, 1)
    style_axes(ax, xlabel="deduction index j   (the 79 solving models, rho %.2f)"
               % OVERLAY_RHO, ylabel="share of node-position argmaxes")
    save(f_, path)
    share = {k: float(sum(agg[j][k] for j in js) / tot.sum())
             for k in ("n_in_F", "n_premature", "n_recolor", "n_nonnode")}
    return share


def d6_premature(cells, avail):
    """Raw premature rate against the share of the graph that could be named
    prematurely at all."""
    f_, ax, path = fig("log_d6_premature_vs_rho.png")
    agg = collections.defaultdict(lambda: [0, 0])
    for c in cells.values():
        if not c["in79"]:
            continue                      # the registered L5 table's population
        tot = (c["node_in_F"] + c["node_premature"]
               + c["node_recolor"] + c["node_nonnode"])
        a = agg[round(c["rho"], 4)]
        a[0] += c["node_premature"]
        a[1] += tot
    rhos = sorted(agg)
    raw = [agg[r][0] / agg[r][1] for r in rhos]
    # availability() returns (mean, per-j dict) per density; the deck wants the
    # mean only.
    av = [avail[r][0] if r in avail else float("nan") for r in rhos]
    ax.plot(rhos, raw, color=SERIES[0], lw=2.0, marker="o", ms=4.5)
    ax.plot(rhos, av, color=SERIES[1], lw=1.8, marker="s", ms=4, ls="--")
    ax.invert_xaxis()
    ax.set_ylim(0, 0.33)
    stagger(ax, [(rhos[0], raw[0], "premature rate", SERIES[0]),
                 (rhos[0], av[0], "nodes available to name prematurely", SERIES[1])],
            gap=0.10)
    style_axes(ax, xlabel="hint density rho   (sparser to the right)",
               ylabel="share of node positions")
    save(f_, path)
    return list(zip(rhos, raw, av))


def d7_unforced(cells):
    """Free generation: when a rollout dies, was the node it named forced?"""
    f_, ax, path = fig("log_d7_unforced_hist.png")
    v = np.array([c["frac_unforced"] for c in cells.values()
                  if c["in79"] and 0.4999 <= c["rho"] <= 0.7501
                  and c["n_err_node"] >= 50
                  and c["frac_unforced"] == c["frac_unforced"]])
    ax.hist(v, bins=np.linspace(0, 1, 51), color=SERIES[0], alpha=0.85)
    ax.axvline(0.5, color=MUTED, lw=1.2, ls="--")
    ax.set_yscale("log")
    label_line(ax, 0.5, ax.get_ylim()[1] * 0.25, "half", color=MUTED)
    style_axes(ax, xlabel="fraction of first errors naming an UNFORCED node",
               ylabel="cells  (log scale)")
    save(f_, path)
    return len(v), float(np.median(v)), float(v.min())


def d8_same_eval(cells, pos_summary):
    f_, ax, path = fig("log_d8_solve_agreement.png", w=5.4, h=4.8)
    xs, ys = [], []
    for key, c in cells.items():
        p = pos_summary.get(key)
        if p is None or c["solve_rate"] != c["solve_rate"]:
            continue
        xs.append(p)
        ys.append(c["solve_rate"])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1)
    ax.scatter(xs, ys, s=12, color=SERIES[0], alpha=0.6, rasterized=True)
    d = np.abs(np.array(xs) - np.array(ys))
    style_axes(ax, xlabel="stage POS solve rate",
               ylabel="stage LOGIC replayed solve rate")
    save(f_, path)
    return len(xs), float(d.max())


def main():
    os.makedirs(DECK, exist_ok=True)
    lrows = read_shards(OUT, "L_position_*.csv")
    srows = read_shards(OUT, "LF_summary_*.csv")
    posrows = read_shards(POS, "B_summary_*.csv")
    runs = run_table(srows or lrows)

    cells = {}
    for r in srows:
        cells[(r["run_id"], round(f(r, "rho"), 4))] = {
            "run_id": r["run_id"], "rho": f(r, "rho"),
            "realized_rho": f(r, "realized_rho"), "n": blanks(f(r, "realized_rho")),
            "in79": b(r, "in79"), "p_logic": f(r, "p_logic"),
            "p_strict": f(r, "p_strict"), "solve_rate": f(r, "solve_rate"),
            "frac_unforced": f(r, "frac_unforced"),
            "n_err_node": i(r, "n_err_node"),
            "node_premature": i(r, "node_premature"), "node_in_F": i(r, "node_in_F"),
            "node_recolor": i(r, "node_recolor"), "node_nonnode": i(r, "node_nonnode"),
        }
    pos_summary = {(r["run_id"], round(f(r, "rho"), 4)): f(r, "solve_rate")
                   for r in posrows}
    avail = availability(INST, sorted({round(c["rho"], 4) for c in cells.values()}))

    fits = {}
    for rho in FIT_RHOS:
        sel = [c for c in cells.values() if abs(c["rho"] - rho) < 1e-9 and c["in79"]]
        if len(sel) < 8:
            continue
        s = [c["solve_rate"] for c in sel]
        kl, el, cl, r2l, _ = powerlaw([c["p_logic"] for c in sel], s)
        ks, es, cs, r2s, _ = powerlaw([c["p_strict"] for c in sel], s)
        fits[round(rho, 4)] = dict(
            rho=rho, n=np.mean([c["n"] for c in sel]), n_runs=len(sel),
            k_logic=kl, se_logic=el, C_logic=cl, r2_logic=r2l,
            k_strict=ks, se_strict=es, C_strict=cs, r2_strict=r2s)

    ok = [v for v in fits.values() if v["k_logic"] == v["k_logic"]]
    a_logic, a_logic_se, _ = through_origin([v["n"] for v in ok],
                                            [v["k_logic"] for v in ok])
    a_strict, a_strict_se, _ = through_origin([v["n"] for v in ok],
                                              [v["k_strict"] for v in ok])

    print("figures:")
    d1_exponent(fits, a_logic, a_strict)
    npts, fit60 = d2_powerlaw(cells, fits)
    d3_prefactor(fits)
    ncell, med_gain, max_gain, n_neg = d4_gain(cells)
    jrows = [r for r in lrows if r.get("scheme") == "j"]
    share = d5_classes(jrows)
    prem = d6_premature(cells, avail)
    n_unf, med_unf, min_unf = d7_unforced(cells)
    n_same, d_max = d8_same_eval(cells, pos_summary)

    print("\nnumbers the deck quotes -- every one must match VERDICT.txt:")
    print("  runs present %d (in79 %d)"
          % (len(runs), sum(r["in79"] for r in runs.values())))
    print("  L1  a_logic  = %.3f +- %.3f" % (a_logic, a_logic_se))
    print("  L1  a_strict = %.3f +- %.3f" % (a_strict, a_strict_se))
    print("  L1  median prefactor rho >= 0.65 = %.3f"
          % np.median([v["C_logic"] for v in ok if v["rho"] >= 0.6499]))
    print("  L3  median k_logic/k_strict = %.3f"
          % np.median([v["k_logic"] / v["k_strict"] for v in ok]))
    print("  L2  gain over %d cells: median %+.4f, max %+.4f, negatives %d"
          % (ncell, med_gain, max_gain, n_neg))
    print("  rho 0.60 fit: %d runs, k = %.2f, C = %.3f, R2 = %.3f"
          % (npts, fit60["k_logic"], fit60["C_logic"], fit60["r2_logic"]))
    print("  node classes at rho %.2f: in_F %.4f  premature %.4f  recolor %.6f"
          "  nonnode %.4f"
          % (OVERLAY_RHO, share["n_in_F"], share["n_premature"],
             share["n_recolor"], share["n_nonnode"]))
    print("  L4  frac_unforced over %d cells: median %.3f, min %.3f"
          % (n_unf, med_unf, min_unf))
    print("  F3  same-eval: %d cells, max |difference| %.4f" % (n_same, d_max))
    for r, raw, av in prem:
        print("      rho %.2f  premature %.4f  availability %.4f" % (r, raw, av))


if __name__ == "__main__":
    main()
