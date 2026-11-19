"""Stage BYPASS (107) -- the scoring half.

Scores B1, B1b, B2, B3, B4, B5, B6 and the four E-checks against
runs/stagebypass_2026-11-19/REGISTERED.md, writes VERDICT.txt, the merged
bypass_runs.csv, and the deck figures.

Runs in the MATPLOTLIB module set, which unloads torch; the measurement half
is scripts/stagebypass_probe.py and the two cannot be one program.
"""
import csv
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.stageexp_fit import fnum, med, read_shards, spread
from slides import MUTED, SERIES, label_line, series_colors, style_axes

OUT = "/scratch/users/hshunt/spin_gnn/runs/stagebypass_2026-11-19"
ABLATE = "/scratch/users/hshunt/spin_gnn/runs/stageablate_2026-11-19/ablate_runs.csv"
FIG = os.path.join(OUT, "figures")
MAIN = 0.75
RHOS = [0.90, 0.75, 0.60]
GATE = 0.740
CONDS = ["INTACT", "ALLPAIRS", "ALLHINT", "ALLROWS6", "ALLSEC"]
NICE = {"INTACT": "nothing blocked",
        "ALLPAIRS": "own graph row",
        "ALLHINT": "neighbours' hint colours",
        "ALLROWS6": "all six rows naming the cell",
        "ALLSEC": "whole hints + graph sections"}

# registered bars
B1_HI, B1_LO = 0.80, 0.40
B1B_HI_N, B1B_LO_N = 3, 8          # >=3 of 4 gate, <=8 of 32 others, at 0.70
B1B_THR = 0.70
B2_HI, B2_LO = 0.80, 0.40
B2_PAIRS = [("r76", "r77", "r75"), ("r74", "r73", "r72")]   # held hi, held lo, seen
B3_BAR = -0.40
B4_MARGIN = 0.30
B5_BAR = 0.40
B6_MED, B6_SEP = 0.15, 0.20


def load():
    rows = read_shards(OUT, "bypass_runs_*of*.csv")
    for r in rows:
        r["acc"] = fnum(r, "acc")
        r["rho"] = fnum(r, "rho")
        r["rho50"] = fnum(r, "rho50")
        r["mass_removed"] = fnum(r, "mass_removed")
    path = os.path.join(OUT, "bypass_runs.csv")
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def fam(cell):
    for f in ("tied_no_mlp", "no_mlp", "rf_mlp", "linear", "full"):
        if f in cell:
            return f
    return "?"


def at(rows, cond, rho=MAIN):
    """run_id -> acc for one condition at one density."""
    return {r["run_id"]: r["acc"] for r in rows
            if r["cond"] == cond and abs(r["rho"] - rho) < 1e-9}


def verdict(value, bar, fals, hi_is_hit):
    ok = value >= bar if hi_is_hit else value <= bar
    bad = value <= fals if hi_is_hit else value >= fals
    return "HIT" if ok else ("MISS (falsifier tripped)" if bad else "MISS")


# ----------------------------------------------------------------- figures
def _ladder(rows, path):
    """The five conditions as a bar ladder, gate-passing against the rest."""
    g = [c for c in CONDS]
    gate = [med([r["acc"] for r in rows if r["cond"] == c
                 and abs(r["rho"] - MAIN) < 1e-9 and r["group"] == "gate"])
            for c in g]
    below = [med([r["acc"] for r in rows if r["cond"] == c
                  and abs(r["rho"] - MAIN) < 1e-9 and r["group"] == "below"])
             for c in g]
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.barh(y + 0.19, gate, 0.36, color=SERIES[0], label="gate")
    ax.barh(y - 0.19, below, 0.36, color=MUTED, label="below")
    for i, (a, b) in enumerate(zip(gate, below)):
        ax.text(a + 0.012, i + 0.19, "%.3f" % a, va="center", fontsize=9)
        ax.text(b + 0.012, i - 0.19, "%.3f" % b, va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([NICE[c] for c in g])
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("argmax-colour accuracy (median over runs, $\\rho=0.75$)")
    ax.text(1.02, len(g) - 0.81, "4 gate-passing", color=SERIES[0], fontsize=9)
    ax.text(1.02, len(g) - 1.19, "32 below", color=MUTED, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _dissoc(rows, path, seen):
    """B2: the two tied cells, three seeds each, architecture held exactly
    fixed.  The seen seed is the filled square; the two held-out seeds are
    circles.  If ALLHINT were a family property all six would sit level."""
    a = at(rows, "ALLHINT")
    r50 = {r["run_id"]: r["rho50"] for r in rows}
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    cols = series_colors(2)
    for i, (hi, lo, sn) in enumerate(B2_PAIRS):
        xs, ys, mk = [], [], []
        for rid in (hi, lo):
            if rid in a:
                xs.append(r50[rid]); ys.append(a[rid]); mk.append("o")
        if sn in seen:
            xs.append(seen[sn][0]); ys.append(seen[sn][1]); mk.append("s")
        order = np.argsort(xs)
        xs = [xs[j] for j in order]; ys = [ys[j] for j in order]
        mk = [mk[j] for j in order]
        ax.plot(xs, ys, "-", color=cols[i], lw=2.0, zorder=1)
        for x, y, m in zip(xs, ys, mk):
            ax.plot([x], [y], m, color=cols[i], ms=11 if m == "s" else 9,
                    mfc=cols[i] if m == "s" else "white", mew=2.0, zorder=2)
    ax.axvline(GATE, color=MUTED, ls=":", lw=1.4)
    ax.text(GATE - 0.004, 0.52, "gate", color=MUTED, fontsize=9,
            rotation=90, ha="right", va="center")
    label_line(ax, 0.60, 0.98, "d320 k4", color=cols[0])
    label_line(ax, 0.62, 0.90, "d320 k2", color=cols[1])
    ax.set_xlabel(r"$\rho_{50}$ of that seed (lower is a better model)")
    ax.set_ylabel("accuracy, neighbours' hint colours blocked")
    ax.set_ylim(-0.05, 1.12)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter(rows, path, cond, ylabel):
    """B3/B4: acc against per-run rho50 inside no_mlp, the only held-out family
    with both a usable n and a usable outcome range."""
    a = at(rows, cond)
    pts = [(r["rho50"], a[r["run_id"]], fam(r["cell"]))
           for r in rows if r["cond"] == cond
           and abs(r["rho"] - MAIN) < 1e-9 and r["run_id"] in a]
    nm = [(x, y) for x, y, f in pts if f == "no_mlp"]
    other = [(x, y) for x, y, f in pts if f != "no_mlp"]
    rho_s = spearmanr([p[0] for p in nm], [p[1] for p in nm]).correlation
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot([p[0] for p in other], [p[1] for p in other], "o",
            color=MUTED, ms=6, mfc="white", mew=1.4)
    ax.plot([p[0] for p in nm], [p[1] for p in nm], "o",
            color=SERIES[0], ms=8)
    ax.text(0.03, 0.06, r"$\rho_s = %+.3f$  (no_mlp, $n=%d$)" % (rho_s, len(nm)),
            transform=ax.transAxes, color=SERIES[0], fontsize=11)
    ax.axvline(GATE, color=MUTED, ls=":", lw=1.4)
    ax.set_xlabel(r"$\rho_{50}$ of that run (lower is a better model)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.12)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rho_s, len(nm)


def _density(rows, path):
    """ALLHINT against density, gate-passing against the rest."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, grp in enumerate(["gate", "below"]):
        ys = [med([r["acc"] for r in rows if r["cond"] == "ALLHINT"
                   and abs(r["rho"] - p) < 1e-9 and r["group"] == grp])
              for p in sorted(RHOS)]
        c = SERIES[0] if grp == "gate" else MUTED
        ax.plot(sorted(RHOS), ys, "-o", color=c, lw=2.4, ms=7)
        label_line(ax, sorted(RHOS)[-1], ys[-1],
                   "4 gate-passing" if grp == "gate" else "32 below", color=c)
    ax.set_xlim(0.55, 0.99)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel(r"hint density $\rho$")
    ax.set_ylabel("accuracy, neighbours' hint colours blocked")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pooled(rows, path, seen_pts):
    """Descriptive only, never scored: the 36 held-out runs laid over the 43
    stage ABLATE already measured, to show how little outcome range the
    held-out population has."""
    a = at(rows, "ALLHINT")
    held = [(r["rho50"], a[r["run_id"]]) for r in rows if r["cond"] == "ALLHINT"
            and abs(r["rho"] - MAIN) < 1e-9 and r["run_id"] in a]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot([p[0] for p in seen_pts], [p[1] for p in seen_pts], "o",
            color=MUTED, ms=6, mfc="white", mew=1.4)
    ax.plot([p[0] for p in held], [p[1] for p in held], "o",
            color=SERIES[0], ms=8)
    ax.axvline(GATE, color=MUTED, ls=":", lw=1.4)
    label_line(ax, 0.80, 0.30, "36 held out", color=SERIES[0])
    label_line(ax, 0.45, 0.88, "43 already probed", color=MUTED)
    ax.set_xlabel(r"$\rho_{50}$ of that run (lower is a better model)")
    ax.set_ylabel("accuracy, neighbours' hint colours blocked")
    ax.set_ylim(-0.05, 1.12)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _rank(rows, path):
    """The one-sided result: runs ordered by competence, ALLHINT on y.  Rank
    rather than rho50 on x because 32 of the 36 rho50 values lie inside 0.035
    of each other, so the raw axis hides the ordering the claim is about."""
    a = at(rows, "ALLHINT")
    r50 = {r["run_id"]: r["rho50"] for r in rows}
    order = sorted(a, key=lambda k: r50[k])
    ys = [a[k] for k in order]
    xs = np.arange(1, len(order) + 1)
    cols = [SERIES[0] if y >= B1B_THR else MUTED for y in ys]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(xs, ys, 0.74, color=cols)
    ax.axhline(B1B_THR, color=SERIES[1], ls="--", lw=1.6)
    ax.text(36.4, B1B_THR + 0.02, "0.70", color=SERIES[1], fontsize=9, ha="right")
    ax.set_xlim(0.2, 36.8)
    ax.set_ylim(0, 1.14)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32, 36])
    ax.set_xlabel(r"run, ordered by $\rho_{50}$ (1 = most competent of the 36)")
    ax.set_ylabel("accuracy, hint colours blocked")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def seen_allhint():
    """The 43 ABLATE runs at ALLHINT / rho=0.75, keyed by run_id, with the
    per-run rho50 joined from bound_runs.csv."""
    B = "/scratch/users/hshunt/spin_gnn/runs/stagebound_2026-11-17/bound_runs.csv"
    with open(B) as fh:
        r50 = {r["run_id"]: float(r["rho50_pub"]) for r in csv.DictReader(fh)}
    out = {}
    with open(ABLATE) as fh:
        for r in csv.DictReader(fh):
            if r["cond"] == "ALLHINT" and abs(float(r["rho"]) - MAIN) < 1e-9:
                out[r["run_id"]] = (r50[r["run_id"]], float(r["acc"]))
    return out


# ------------------------------------------------------------------ main
def main():
    os.makedirs(FIG, exist_ok=True)
    rows = load()
    if not rows:
        print("no shards yet")
        return
    runs = sorted({r["run_id"] for r in rows})
    L = []
    P = L.append
    P("STAGE BYPASS (107) -- VERDICT")
    P("bars and falsifiers: runs/stagebypass_2026-11-19/REGISTERED.md")
    P("%d runs, %d rows, densities %s, query j=0, n_q=200" %
      (len(runs), len(rows), sorted({r["rho"] for r in rows})))
    P("")

    a = at(rows, "ALLHINT")
    r50 = {r["run_id"]: r["rho50"] for r in rows}
    cell = {r["run_id"]: r["cell"] for r in rows}
    gate = [k for k in a if r50[k] < GATE]
    rest = [k for k in a if r50[k] >= GATE]

    # ---- E-checks first: a failure voids the stage
    P("E-CHECKS")
    intact = at(rows, "INTACT")
    e1n = sum(1 for v in intact.values() if v >= 0.99995)
    P("  E1  acc(INTACT)=1.0000 on %d of %d runs           %s"
      % (e1n, len(intact), "PASS" if e1n >= 34 else "FAIL"))
    zero = [(r["run_id"], r["cond"]) for r in rows
            if r["cond"] != "INTACT" and r["mass_removed"] <= 0]
    P("  E2  every ablation removes >0 mass: %d violations  %s"
      % (len(zero), "PASS" if not zero else "FAIL"))
    e2 = sum(int(r["e2_bad"]) for r in rows)
    e3 = sum(int(r["e3_bad"]) for r in rows)
    e4 = sum(int(r["e4_bad"]) for r in rows)
    P("  E3  locality + mask integrity: e2_bad=%d e3_bad=%d   %s"
      % (e2, e3, "PASS" if e2 == 0 and e3 == 0 else "FAIL"))
    want_rows = len(runs) * len(RHOS) * len(CONDS)
    nq = {int(r["n_q"]) for r in rows}
    P("  E4  %d rows (want %d), n_q in %s, e4_bad=%d        %s"
      % (len(rows), want_rows, sorted(nq), e4,
         "PASS" if len(rows) == want_rows and nq == {200} and e4 == 0 else "FAIL"))
    P("")

    P("REGISTERED PREDICTIONS")
    # ---- B1
    mg, mr = med([a[k] for k in gate]), med([a[k] for k in rest])
    v1 = "HIT" if (mg >= B1_HI and mr <= B1_LO) else (
        "MISS (falsifier tripped)" if mg <= B1_LO else "MISS")
    P("  B1  median ALLHINT: gate(n=%d)=%.3f (bar >=%.2f), rest(n=%d)=%.3f "
      "(bar <=%.2f)  %s" % (len(gate), mg, B1_HI, len(rest), mr, B1_LO, v1))
    # ---- B1b
    ng = sum(1 for k in gate if a[k] >= B1B_THR)
    nr = sum(1 for k in rest if a[k] >= B1B_THR)
    v1b = "HIT" if (ng >= B1B_HI_N and nr <= B1B_LO_N) else (
        "MISS (falsifier tripped)" if (ng <= 1 or nr >= 16) else "MISS")
    P("  B1b ALLHINT>=%.2f on %d of %d gate (bar >=%d) and %d of %d rest "
      "(bar <=%d)  %s" % (B1B_THR, ng, len(gate), B1B_HI_N, nr, len(rest),
                          B1B_LO_N, v1b))
    # ---- B2, the within-cell dissociation
    seen = seen_allhint()
    hi_ok = lo_ok = 0
    for hi, lo, sn in B2_PAIRS:
        P("      %s rho50=%.3f ALLHINT=%.3f | sibling %s rho50=%.3f "
          "ALLHINT=%.3f | seen %s rho50=%.3f ALLHINT=%.3f"
          % (hi, r50.get(hi, float('nan')), a.get(hi, float('nan')),
             lo, r50.get(lo, float('nan')), a.get(lo, float('nan')),
             sn, seen[sn][0], seen[sn][1]))
        hi_ok += int(a.get(hi, -1) >= B2_HI)
        lo_ok += int(a.get(lo, 2) <= B2_LO)
    v2 = ("HIT" if hi_ok == 2 and lo_ok == 2 else
          "MISS (falsifier tripped)" if hi_ok == 0 else "SPLIT")
    P("  B2  %d of 2 held-out tied runs >=%.2f, %d of 2 siblings <=%.2f  %s"
      % (hi_ok, B2_HI, lo_ok, B2_LO, v2))
    # ---- B3 / B4
    s3, n3 = _scatter(rows, os.path.join(FIG, "scatter_allhint.png"),
                      "ALLHINT", "accuracy, neighbours' hint colours blocked")
    s4, _ = _scatter(rows, os.path.join(FIG, "scatter_allrows6.png"),
                     "ALLROWS6", "accuracy, all six rows naming the cell blocked")
    v3 = verdict(s3, B3_BAR, 0.0, hi_is_hit=False)
    P("  B3  Spearman(ALLHINT, rho50) in no_mlp (n=%d) = %+.3f (bar <=%.2f)  %s"
      % (n3, s3, B3_BAR, v3))
    margin = abs(s3) - abs(s4)
    v4 = "HIT" if margin >= B4_MARGIN else (
        "MISS (falsifier tripped)" if abs(s4) >= abs(s3) else "MISS")
    P("  B4  |ALLHINT|-|ALLROWS6| = %.3f-%.3f = %.3f (bar >=%.2f)  %s"
      % (abs(s3), abs(s4), margin, B4_MARGIN, v4))
    # ---- B5
    ap = at(rows, "ALLPAIRS")
    d5 = med(list(ap.values())) - med(list(a.values()))
    v5 = verdict(d5, B5_BAR, 0.0, hi_is_hit=True)
    P("  B5  median ALLPAIRS - median ALLHINT = %.3f - %.3f = %.3f "
      "(bar >=%.2f)  %s" % (med(list(ap.values())), med(list(a.values())),
                            d5, B5_BAR, v5))
    # ---- B6
    sec = at(rows, "ALLSEC")
    m6 = med(list(sec.values()))
    sep6 = abs(med([sec[k] for k in gate]) - med([sec[k] for k in rest]))
    v6 = "HIT" if (m6 <= B6_MED and sep6 <= B6_SEP) else (
        "MISS (falsifier tripped)" if (m6 > 0.30 or sep6 > 0.40) else "MISS")
    P("  B6  median ALLSEC = %.3f (bar <=%.2f), gate-rest separation = %.3f "
      "(bar <=%.2f)  %s" % (m6, B6_MED, sep6, B6_SEP, v6))
    P("")

    # ---- declared unscoreable in advance
    P("DESCRIPTIVE, DECLARED UNSCOREABLE IN ADVANCE (REGISTERED.md section 4)")
    pooled = spearmanr([r50[k] for k in a], [a[k] for k in a]).correlation
    lo, hi = spread([r50[k] for k in a])
    P("  pooled Spearman over all %d = %+.3f -- NOT SCORED, rho50 range "
      "%.3f-%.3f, IQR 0.018 against the seen 0.170" % (len(a), pooled, lo, hi))
    for f in ("full", "linear", "rf_mlp", "tied_no_mlp", "no_mlp"):
        ks = [k for k in a if fam(cell[k]) == f]
        if not ks:
            continue
        flo, fhi = spread([r50[k] for k in ks])
        tag = "" if f in ("no_mlp",) else "   (not scored)"
        P("  %-12s n=%2d  median ALLHINT=%.3f  rho50 %.3f-%.3f%s"
          % (f, len(ks), med([a[k] for k in ks]), flo, fhi, tag))
    P("")
    P("SCORECARD  %s" % "  ".join(
        "%s=%s" % (n, v.split()[0]) for n, v in
        [("B1", v1), ("B1b", v1b), ("B2", v2), ("B3", v3),
         ("B4", v4), ("B5", v5), ("B6", v6)]))

    _ladder(rows, os.path.join(FIG, "ladder.png"))
    _dissoc(rows, os.path.join(FIG, "dissociation.png"), seen)
    _density(rows, os.path.join(FIG, "density.png"))
    _pooled(rows, os.path.join(FIG, "pooled.png"), list(seen.values()))
    _rank(rows, os.path.join(FIG, "rank.png"))

    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(OUT, "VERDICT.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
