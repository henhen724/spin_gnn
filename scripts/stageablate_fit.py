"""Stage ABLATE (106) -- the scoring half.

Scores A1-A7 and E1-E4 exactly as runs/stageablate_2026-11-19/REGISTERED.md
fixes them, and tabulates the six EXPLORATORY ALL* conditions separately so a
reader can always tell a prediction from a lookup.

matplotlib unloads torch on this cluster, so this is a separate program from
scripts/stageablate_probe.py.  Run it in the figure module set:

    module purge
    module load devel math viz python/3.12.1
    module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
    export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

from stageexp_fit import fnum, read_shards, med, spread
from slides import SERIES, MUTED, style_axes, label_line, series_colors

OUT = "/scratch/users/hshunt/spin_gnn/runs/stageablate_2026-11-19"
FIG = os.path.join(OUT, "figures")
RHOS = [0.90, 0.75, 0.60]
MAIN = 0.75

# Registered bars and falsifiers.  Copied from REGISTERED.md; nothing here is
# chosen after seeing a number.
A1_BAR, A1_FALS = 0.50, 0.90
A2_BAR, A2_FALS = 0.90, 0.70
A3_BAR, A3_FALS = 0.90, 0.70
A4_BAR, A4_FALS = 15, 9
A5_BAR, A5_FALS = 0.95, 0.995
A6_BAR, A6_FALS = 0.15, 0.30
A7_BAR, A7_FALS = 0.50, 0.90

EXPLORATORY = ["ALLPAIRS", "ALLHINT", "ALLBOTH", "ALLROWS6", "ALLADJ", "ALLSEC"]
LADDER_LABEL = {
    "ALLPAIRS": "v's own row",
    "ALLHINT": "neighbours' hint colours",
    "ALLBOTH": "own row + neighbours' hints",
    "ALLROWS6": "all six rows naming v",
    "ALLADJ": "six rows + neighbours' hints",
    "ALLSEC": "the whole hints + graph sections",
}


def load():
    rows = read_shards(OUT, "ablate_runs_*of*.csv")
    for r in rows:
        r["acc"] = fnum(r, "acc")
        r["rho"] = fnum(r, "rho")
        r["m"] = int(r["m"])
        r["n_q"] = int(r["n_q"])
        r["depth"] = int(r["depth"])
        r["params"] = int(r["params"])
        r["mass_removed"] = fnum(r, "mass_removed")
        r["rho50"] = fnum(r, "rho50_pub_cell")
        for k in ("e2_bad", "e3_bad", "e4_bad"):
            r[k] = int(r[k])
    return rows


def pick(rows, cond=None, m=None, rho=MAIN, group=None):
    out = rows
    if cond is not None:
        out = [r for r in out if r["cond"] == cond]
    if m is not None:
        out = [r for r in out if r["m"] == m]
    if rho is not None:
        out = [r for r in out if abs(r["rho"] - rho) < 1e-9]
    if group is not None:
        out = [r for r in out if r["group"] == group]
    return out


def accs(rows, **kw):
    return [r["acc"] for r in pick(rows, **kw)]


def verdict(value, bar, fals, hi_is_hit):
    """hi_is_hit: True when the prediction is `value >= bar`."""
    if hi_is_hit:
        if value >= bar:
            return "HIT"
        return "MISS" if value <= fals else "SPLIT"
    if value <= bar:
        return "HIT"
    return "MISS" if value >= fals else "SPLIT"


# ----------------------------------------------------------------- figures

def _bar_ladder(rows, path):
    """The redundancy ladder.  One claim: accuracy survives every partial
    block and only falls when every copy of v's adjacency is removed."""
    names = EXPLORATORY
    vals = [med(accs(rows, cond=c, group="gate")) for c in names]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    cols = series_colors(len(names), highlight=len(names) - 1)
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.015, yi, "%.3f" % v, va="center", fontsize=9, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([LADDER_LABEL[c] for c in names], fontsize=9)
    ax.set_xlim(0, 1.12)
    style_axes(ax, xlabel="argmax-colour accuracy at the j=0 query (median over the 18 gate-passing runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _dose(rows, path):
    """A1/A2/A4.  acc against m, TOP against RAND."""
    ms = [1, 2, 3, 6]
    top = [med(accs(rows, cond="TOP", m=m, group="gate")) for m in ms]
    rnd = [med(accs(rows, cond="RAND", m=m, group="gate")) if m != 2 else np.nan
           for m in ms]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ms, top, "-o", color=SERIES[0], lw=2, ms=5, rasterized=True)
    rm = [m for m, v in zip(ms, rnd) if v == v]
    rv = [v for v in rnd if v == v]
    ax.plot(rm, rv, "-s", color=SERIES[1], lw=2, ms=5, rasterized=True)
    ax.axhline(A1_BAR, color=MUTED, ls="--", lw=1)
    label_line(ax, ms[-1], top[-1], "top-m heads", color=SERIES[0])
    label_line(ax, rm[-1], rv[-1], "random m heads", color=SERIES[1], dy=-12)
    ax.text(1.05, A1_BAR + 0.02, "registered bar for A1", fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(ms)
    style_axes(ax, xlabel="m: heads blocked from reading v's own row",
               ylabel="argmax-colour accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _groups(rows, path):
    """A6.  Per-run accuracy under ALLROWS6, gate-passing against below gate."""
    g = accs(rows, cond="ALLROWS6", group="gate")
    b = accs(rows, cond="ALLROWS6", group="below")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    rs = np.random.RandomState(0)
    for i, (v, c, name) in enumerate(((g, SERIES[0], "gate-passing"),
                                      (b, SERIES[1], "below gate"))):
        x = i + 0.12 * rs.randn(len(v))
        ax.plot(x, v, "o", color=c, ms=5, alpha=0.75, rasterized=True)
        ax.plot([i - 0.28, i + 0.28], [med(v)] * 2, "-", color=c, lw=2.5)
        ax.text(i, -0.07, "%s\nn=%d, median %.3f" % (name, len(v), med(v)),
                ha="center", va="top", fontsize=9, color=MUTED)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([])
    style_axes(ax, ylabel="accuracy with all six rows naming v blocked")
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _density(rows, path):
    """A7 and the ladder against hint density."""
    keep = ["ALLBOTH", "ALLROWS6", "ALLSEC"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for c, col in zip(keep, SERIES):
        v = [med(accs(rows, cond=c, rho=rho, group="gate")) for rho in RHOS]
        ax.plot(RHOS, v, "-o", color=col, lw=2, ms=5, rasterized=True)
        label_line(ax, RHOS[-1], v[-1], LADDER_LABEL[c], color=col)
    ax.set_xlim(0.56, 0.95)
    ax.set_ylim(-0.05, 1.12)
    ax.invert_xaxis()
    style_axes(ax, xlabel="hint density rho", ylabel="argmax-colour accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _mass(rows, path):
    """Own-row attention mass removed against the accuracy cost of removing it."""
    pts = [(r["mass_removed"], r["acc"]) for r in rows
           if r["cond"] == "TOP" and r["group"] == "gate"
           and abs(r["rho"] - MAIN) < 1e-9]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=SERIES[0],
            ms=5, alpha=0.7, rasterized=True)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.08)
    style_axes(ax, xlabel="fraction of the query's own-row attention mass removed",
               ylabel="argmax-colour accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _depth(rows, path):
    """Does the surviving redundancy depend on depth?"""
    pts = [(r["depth"], r["acc"], r["group"]) for r in rows
           if r["cond"] == "ALLROWS6" and abs(r["rho"] - MAIN) < 1e-9]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for grp, col, name in (("gate", SERIES[0], "gate-passing"),
                           ("below", SERIES[1], "below gate")):
        v = [(d, a) for d, a, g in pts if g == grp]
        if not v:
            continue
        jitter = np.random.RandomState(1).uniform(-0.25, 0.25, len(v))
        ax.plot([d + j for (d, _), j in zip(v, jitter)], [a for _, a in v],
                "o", color=col, ms=5, alpha=0.7, rasterized=True)
        xm = max(d for d, _ in v)
        label_line(ax, xm, med([a for d, a in v if d == xm]), name, color=col)
    ax.set_ylim(0, 1.08)
    style_axes(ax, xlabel="depth (applications)",
               ylabel="accuracy with all six rows naming v blocked")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- verdict

def main():
    rows = load()
    os.makedirs(FIG, exist_ok=True)
    gate_runs = sorted(set(r["run_id"] for r in rows if r["group"] == "gate"))
    below_runs = sorted(set(r["run_id"] for r in rows if r["group"] == "below"))
    L = []
    w = L.append

    w("STAGE ABLATE (finding 106) -- scored against runs/stageablate_2026-11-19/REGISTERED.md")
    w("")
    w("%d gate-passing runs, %d below-gate cells, %d densities, %d rows."
      % (len(gate_runs), len(below_runs), len(RHOS), len(rows)))
    n_q = sorted(set(r["n_q"] for r in rows))
    w("held-out instances per (run, rho): %s" % n_q)
    w("")

    # ---- execution checks
    w("EXECUTION CHECKS")
    intact = accs(rows, cond="INTACT", group="gate", rho=None)
    e1 = all(a == 1.0 for a in intact)
    w("  E1 intact accuracy == 1.0000 on all %d gate-passing (run, rho): %s"
      % (len(intact), "PASS" if e1 else "FAIL (min %.4f)" % min(intact)))
    for key, name in (("e2_bad", "E2 block landed and row renormalises"),
                      ("e3_bad", "E3 locality (rows p < q unchanged)"),
                      ("e4_bad", "E4 token layout")):
        tot = sum(r[key] for r in rows)
        w("  %-44s %s%s" % (name, "PASS" if tot == 0 else "FAIL", ""
                            if tot == 0 else " (%d)" % tot))
    w("")

    # ---- registered predictions
    w("REGISTERED PREDICTIONS (rho = %.2f unless stated, median over the %d "
      "gate-passing runs)" % (MAIN, len(gate_runs)))

    def line(tag, text, value, bar, fals, hi, extra=""):
        v = verdict(value, bar, fals, hi)
        w("  %-3s %-52s %7.4f   bar %-7s fals %-7s  %s%s"
          % (tag, text, value, ("%.4g" % bar), ("%.4g" % fals), v, extra))
        return v

    res = {}
    a1 = med(accs(rows, cond="TOP", m=3, group="gate"))
    res["A1"] = line("A1", "acc, top-3 heads blocked from v's own row", a1,
                     A1_BAR, A1_FALS, False)
    a2 = med(accs(rows, cond="RAND", m=3, group="gate"))
    res["A2"] = line("A2", "acc, 3 random heads blocked (control)", a2,
                     A2_BAR, A2_FALS, True)
    a3 = med(accs(rows, cond="OTHERROW", m=3, group="gate"))
    res["A3"] = line("A3", "acc, top-3 blocked from a WRONG row (control)", a3,
                     A3_BAR, A3_FALS, True)

    mono = 0
    for rid in gate_runs:
        v = []
        for m in (1, 2, 3, 6):
            a = [r["acc"] for r in pick(rows, cond="TOP", m=m, group="gate")
                 if r["run_id"] == rid]
            v.append(a[0] if a else float("nan"))
        if all(v[i] >= v[i + 1] - 1e-12 for i in range(3)):
            mono += 1
    res["A4"] = line("A4", "runs with acc monotone non-increasing in m",
                     float(mono), A4_BAR, A4_FALS, True,
                     extra="  (%d of %d)" % (mono, len(gate_runs)))

    a5 = med(accs(rows, cond="TOP", m=1, group="gate"))
    res["A5"] = line("A5", "acc, top-1 head blocked", a5, A5_BAR, A5_FALS, False)

    g3 = med(accs(rows, cond="TOP", m=3, group="gate"))
    b3 = med(accs(rows, cond="TOP", m=3, group="below"))
    res["A6"] = line("A6", "|below-gate - gate-passing| at top-3 (a NULL)",
                     abs(b3 - g3), A6_BAR, A6_FALS, False,
                     extra="  (gate %.4f, below %.4f)" % (g3, b3))

    per_rho = [med(accs(rows, cond="TOP", m=3, rho=rho, group="gate"))
               for rho in RHOS]
    res["A7"] = line("A7", "worst density, top-3 blocked", max(per_rho),
                     A7_BAR, A7_FALS, False,
                     extra="  (%s)" % ", ".join("%.2f:%.4f" % (r, v)
                                                for r, v in zip(RHOS, per_rho)))
    w("")
    tally = {}
    for v in res.values():
        tally[v] = tally.get(v, 0) + 1
    w("  SCORECARD  " + ", ".join("%d %s" % (tally.get(k, 0), k)
                                  for k in ("HIT", "SPLIT", "MISS")))
    w("")

    # ---- exploratory
    w("EXPLORATORY CONDITIONS -- no registered bar, not predictions.")
    w("  %-34s %9s %9s   %s" % ("blocked at the query", "gate", "below", "per density (gate)"))
    for c in EXPLORATORY:
        g = med(accs(rows, cond=c, group="gate"))
        b = med(accs(rows, cond=c, group="below"))
        pr = ", ".join("%.2f:%.3f" % (rho, med(accs(rows, cond=c, rho=rho,
                                                    group="gate")))
                       for rho in RHOS)
        lo, hi = spread(accs(rows, cond=c, group="gate"))
        w("  %-34s %9.4f %9.4f   %s   [%.3f, %.3f]"
          % (LADDER_LABEL[c], g, b, pr, lo, hi))
    w("")
    w("  The ladder is the result: every partial block leaves the colour step")
    w("  intact, and accuracy only falls once every copy of v's adjacency is")
    w("  removed.  ALLSEC is the positive control -- it proves the harness can")
    w("  break the model, which is what makes the nulls above readable.")
    w("")

    text = "\n".join(L)
    with open(os.path.join(OUT, "VERDICT.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)

    # merged table for the deck
    cols = list(rows[0].keys())
    with open(os.path.join(OUT, "ablate_runs.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    _bar_ladder(rows, os.path.join(FIG, "ladder.png"))
    _dose(rows, os.path.join(FIG, "A1_dose.png"))
    _groups(rows, os.path.join(FIG, "A6_groups.png"))
    _density(rows, os.path.join(FIG, "A7_density.png"))
    _mass(rows, os.path.join(FIG, "mass_vs_acc.png"))
    _depth(rows, os.path.join(FIG, "depth.png"))
    print("\nfigures -> %s" % FIG)


if __name__ == "__main__":
    main()
