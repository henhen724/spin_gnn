"""Stage TIED -- is S1's deficit a deduction deficit or a scoring artefact?

S1 (`tied_no_mlp T=16 d320`, 613,818 params) is one shared block applied 16
times: an iterated map, and the most analytically tractable architecture that
gets anywhere near direct deduction.  It is held out of the gate by 0.017 of
rho_50, which finding 62's 16.09 slope converts to about 0.0011 of token
accuracy; the measured val-accuracy gap to J2 (`no_mlp L=16 d192`, 2,504,890
params) is 0.011, ten times larger.  REGISTERED.md quotes 0.011 for both and
is wrong on the first: the gate needs 0.0011.  STATUS records the gap as open
with every registered training knob closed.

Finding 94 gives a reason to re-examine the measurement rather than try an
eighth knob: validation token accuracy is STRICT, scoring the argmax against a
reference trace that writes each round in sorted node order even though DD
fixes no order within a round.  Stage LOGIC put that mis-scoring at a median
+0.0106 -- the size of the whole S1-J2 gap, and ten times what the gate needs.

Hypothesis: weight tying costs the capacity to memorise an arbitrary output
ORDERING CONVENTION while leaving the deduction intact.

Predictions and falsifiers: runs/stagetied_2026-11-17/REGISTERED.md, committed
before this script was written.  No training, no GPU, no new instances.
"""

import collections
import csv
import glob
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line  # noqa: E402

OUT = os.environ.get("TIED_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stagetied_2026-11-17"
LOGIC = os.environ.get("LOGIC_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stagelogic_2026-11-17"
FIGS = os.path.join(OUT, "figures")
DPI = 150
N_NODES = 48
FIT_RHOS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]

# The two architectures the brief turns on.  S1 is the tractable candidate,
# J2 the standing gate-passing minimum it is measured against.
GROUPS = {
    "S1": ("tied_no_mlp", "16", "320"),
    "J2": ("no_mlp", "16", "192"),
}
COL = {"S1": SERIES[1], "J2": SERIES[0]}


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


def med(xs):
    xs = [x for x in xs if x == x]
    return float(np.median(xs)) if xs else float("nan")


def spread(xs):
    """min-max over seeds.  The non-negotiable is to quote the spread, never
    a bare mean."""
    xs = [x for x in xs if x == x]
    return (min(xs), max(xs)) if xs else (float("nan"), float("nan"))


def fig(name, w=7.4, h=4.4):
    f_, ax = plt.subplots(figsize=(w, h))
    return f_, ax, os.path.join(FIGS, name)


def save(f_, path):
    f_.tight_layout()
    f_.savefig(path, dpi=DPI)
    plt.close(f_)
    print("  wrote %s" % os.path.basename(path))


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def t1_gain(summ):
    f_, ax, path = fig("T1_gain_by_density.png")
    for g in ("J2", "S1"):
        rhos = [s["rho"] for s in summ if s["group"] == g]
        y = [s["gain"] for s in summ if s["group"] == g]
        lo = [s["gain_lo"] for s in summ if s["group"] == g]
        hi = [s["gain_hi"] for s in summ if s["group"] == g]
        ax.fill_between(rhos, lo, hi, color=COL[g], alpha=0.18, lw=0)
        ax.plot(rhos, y, color=COL[g], lw=2.0, marker="o", ms=4.5)
        label_line(ax, rhos[-1], y[-1], g, color=COL[g])
    ax.invert_xaxis()
    style_axes(ax, xlabel="hint density rho   (harder to the right)",
               ylabel="p_logic - p_strict   (band = seed min-max)")
    save(f_, path)


def t2_closure(summ):
    f_, ax, path = fig("T2_gap_closure.png")
    byr = {s["rho"]: s for s in summ if s["group"] == "gap"}
    rhos = sorted(byr, reverse=True)
    n = [byr[r]["n"] for r in rhos]
    ax.axhline(0.0, color=MUTED, lw=1.0)
    ax.plot(n, [byr[r]["d_strict"] for r in rhos], color=SERIES[0], lw=2.0,
            marker="o", ms=4.5)
    ax.plot(n, [byr[r]["d_logic"] for r in rhos], color=SERIES[2], lw=2.0,
            marker="s", ms=4.5)
    label_line(ax, n[-1], byr[rhos[-1]]["d_strict"], "strict", color=SERIES[0])
    label_line(ax, n[-1], byr[rhos[-1]]["d_logic"], "logical", color=SERIES[2])
    style_axes(ax, xlabel="blank cells  n",
               ylabel="J2 minus S1 token accuracy")
    save(f_, path)


def t3_premature(summ):
    f_, ax, path = fig("T3_premature.png")
    for g in ("J2", "S1"):
        rhos = [s["rho"] for s in summ if s["group"] == g]
        y = [s["premature"] for s in summ if s["group"] == g]
        lo = [s["prem_lo"] for s in summ if s["group"] == g]
        hi = [s["prem_hi"] for s in summ if s["group"] == g]
        ax.fill_between(rhos, lo, hi, color=COL[g], alpha=0.18, lw=0)
        ax.plot(rhos, y, color=COL[g], lw=2.0, marker="o", ms=4.5)
        label_line(ax, rhos[-1], y[-1], g, color=COL[g])
    ax.invert_xaxis()
    style_axes(ax, xlabel="hint density rho   (harder to the right)",
               ylabel="premature rate: unforced-node steps / all node steps")
    save(f_, path)


def t4_acc_vs_solve(cells):
    f_, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
    for ax, key, title in ((axes[0], "p_strict", "strict scoring"),
                           (axes[1], "p_logic", "logical scoring")):
        for g in ("J2", "S1"):
            xs = [c[key] for c in cells if c["group"] == g and c["solve_rate"] > 0]
            ys = [c["solve_rate"] for c in cells
                  if c["group"] == g and c["solve_rate"] > 0]
            ax.scatter(xs, ys, s=20, color=COL[g], alpha=0.8, rasterized=True)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10, color=MUTED)
        style_axes(ax, xlabel="token accuracy", ylabel=None)
    style_axes(axes[0], xlabel="token accuracy", ylabel="solve rate S")
    # One legend-free label per group, placed on the right-hand panel.
    for g, yy in (("J2", 0.55), ("S1", 0.25)):
        axes[1].text(0.03, yy, g, transform=axes[1].transAxes, color=COL[g],
                     fontsize=12, fontweight="bold")
    save(f_, os.path.join(FIGS, "T4_acc_vs_solve.png"))


def ph_modes(cells, hi_runs):
    """Post-hoc: one line per training run, so the bimodality is visible."""
    f_, ax, path = fig("PH_modes.png", w=7.4, h=4.6)
    byrun = collections.defaultdict(list)
    for c in cells:
        byrun[(c["group"], c["run_id"])].append((c["rho"], c["solve_rate"]))
    for (g, rid), pts in sorted(byrun.items()):
        pts.sort(reverse=True)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        hi = rid in hi_runs
        ax.plot(xs, ys, color=COL[g], lw=2.0 if hi else 1.1,
                alpha=0.95 if hi else 0.45,
                ls="-" if hi else "--", marker="o", ms=3.5)
    ax.invert_xaxis()
    style_axes(ax, xlabel="hint density rho   (harder to the right)",
               ylabel="solve rate S")
    # Not SERIES[0]: blue already means "untied" in this figure.
    ax.text(0.04, 0.17, "SOLVED-IT (solid): 2 tied, both k>1;  2 untied",
            transform=ax.transAxes, color="#333333", fontsize=9)
    ax.text(0.04, 0.09, "STUCK (dashed): 7 tied, incl. all 3 k=1;  1 untied",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    ax.text(0.04, 0.01, "orange = tied,  blue = untied;  no run in between",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    save(f_, path)


# --------------------------------------------------------------------------


def main():
    os.makedirs(FIGS, exist_ok=True)
    rows = read_shards(LOGIC, "LF_summary_all_*.csv")

    def group_of(r):
        key = (r["arch_mode"], r["num_layers"], r["d_model"])
        for g, want in GROUPS.items():
            if key == want:
                return g
        return None

    cells = []
    for r in rows:
        g = group_of(r)
        if g is None or not tobool(r, "in79"):
            continue
        rho = round(fnum(r, "rho"), 4)
        if rho not in [round(x, 4) for x in FIT_RHOS]:
            continue
        cells.append(dict(
            group=g, run_id=r["run_id"], seed=r.get("seed", "?"), rho=rho,
            n=N_NODES * (1.0 - fnum(r, "realized_rho")),
            val_acc=fnum(r, "val_acc"),
            params=int(fnum(r, "n_trainable_params")), cell=r["cell"],
            p_logic=fnum(r, "p_logic"), p_strict=fnum(r, "p_strict"),
            p_logic_node=fnum(r, "p_logic_node"),
            p_strict_node=fnum(r, "p_strict_node"),
            p_logic_color=fnum(r, "p_logic_color"),
            p_strict_color=fnum(r, "p_strict_color"),
            in_F=fnum(r, "node_in_F"), premature=fnum(r, "node_premature"),
            recolor=fnum(r, "node_recolor"), nonnode=fnum(r, "node_nonnode"),
            solve_rate=fnum(r, "solve_rate")))
        cells[-1]["gain"] = cells[-1]["p_logic"] - cells[-1]["p_strict"]
        # node_in_F/premature/recolor/nonnode arrive as COUNTS of node steps,
        # not fractions -- checked below, not assumed.  Divide by their sum to
        # get a rate, and name the denominator in words: every node position
        # the model was scored at, at this density, for this run.
        c = cells[-1]
        c["node_steps"] = c["in_F"] + c["premature"] + c["recolor"] + c["nonnode"]
        c["premature_rate"] = (c["premature"] / c["node_steps"]
                               if c["node_steps"] else float("nan"))
        c["in_F_rate"] = (c["in_F"] / c["node_steps"]
                          if c["node_steps"] else float("nan"))

    inv = collections.defaultdict(set)
    for c in cells:
        inv[c["group"]].add((c["run_id"], c["seed"]))
    print("population (in79, the ten fit densities):")
    for g in ("S1", "J2"):
        runs = sorted(inv[g])
        print("  %s = %-12s L=%s d=%s : %d runs, %d cells  %s"
              % (g, GROUPS[g][0], GROUPS[g][1], GROUPS[g][2], len(runs),
                 sum(1 for c in cells if c["group"] == g),
                 ", ".join("%s(s%s)" % t for t in runs)))

    # The class columns must be fractions of node steps, not counts, or the
    # premature comparison in T4 is meaningless.  Check, do not assume.
    sums = [c["node_steps"] for c in cells]
    print("node-class columns sum to %.0f-%.0f across %d cells -- these are "
          "COUNTS of node" % (min(sums), max(sums), len(sums)))
    print("steps, not fractions, so every class is divided by that sum before "
          "use.")
    # The two groups are evaluated on the SAME instances, so the denominator
    # must match density by density, or a ratio of counts would not be a ratio
    # of rates.  Verify rather than trust it.
    bad = []
    for rho in FIT_RHOS:
        rho = round(rho, 4)
        per = {g: {c["node_steps"] for c in cells
                   if c["group"] == g and c["rho"] == rho} for g in GROUPS}
        if per["S1"] != per["J2"] or len(per["S1"]) != 1:
            bad.append((rho, per))
    print("denominator matches between S1 and J2 at all ten densities: %s"
          % ("yes" if not bad else "NO -- %s" % bad))

    summ = []
    for rho in FIT_RHOS:
        rho = round(rho, 4)
        at = {g: [c for c in cells if c["group"] == g and c["rho"] == rho]
              for g in GROUPS}
        if not all(at[g] for g in GROUPS):
            continue
        for g in GROUPS:
            glo, ghi = spread([c["gain"] for c in at[g]])
            plo, phi = spread([c["premature_rate"] for c in at[g]])
            summ.append(dict(
                group=g, rho=rho, n=med([c["n"] for c in at[g]]),
                n_runs=len(at[g]),
                p_logic=med([c["p_logic"] for c in at[g]]),
                p_strict=med([c["p_strict"] for c in at[g]]),
                gain=med([c["gain"] for c in at[g]]), gain_lo=glo, gain_hi=ghi,
                premature=med([c["premature_rate"] for c in at[g]]),
                prem_lo=plo, prem_hi=phi,
                solve_rate=med([c["solve_rate"] for c in at[g]])))
        sj = [s for s in summ if s["rho"] == rho and s["group"] == "J2"][0]
        ss = [s for s in summ if s["rho"] == rho and s["group"] == "S1"][0]
        summ.append(dict(
            group="gap", rho=rho, n=sj["n"], n_runs=0,
            d_strict=sj["p_strict"] - ss["p_strict"],
            d_logic=sj["p_logic"] - ss["p_logic"],
            gain=ss["gain"] - sj["gain"],
            prem_ratio=(ss["premature"] / sj["premature"]
                        if sj["premature"] else float("nan")),
            d_solve=sj["solve_rate"] - ss["solve_rate"]))

    print("\nfigures:")
    t1_gain(summ)
    t2_closure(summ)
    t3_premature(summ)
    t4_acc_vs_solve(cells)

    with open(os.path.join(OUT, "tied_cells.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0].keys()))
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c["group"], -c["rho"], c["run_id"])):
            w.writerow(c)
    keys = sorted({k for s in summ for k in s})
    with open(os.path.join(OUT, "tied_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for s in sorted(summ, key=lambda s: (-s["rho"], s["group"])):
            w.writerow({k: s.get(k, "") for k in keys})

    # ---- verdict ----------------------------------------------------------
    gaps = [s for s in summ if s["group"] == "gap"]
    big = [s for s in gaps if s["n"] >= 12]

    # ---- post-hoc mode split ---------------------------------------------
    # Printing the per-run table showed the runs are bimodal, not spread.  The
    # cut is placed at the LARGEST gap in the sorted rho=0.95 accuracies rather
    # than at a number chosen by hand, and the gap is reported next to the
    # within-cluster spread so the reader can see it is not a close call.
    easy = round(max(FIT_RHOS), 4)
    hard = round(min(FIT_RHOS), 4)
    at95 = sorted(((c["p_strict"], c["run_id"]) for c in cells
                   if c["rho"] == easy), reverse=True)
    jumps = [(at95[i][0] - at95[i + 1][0], i) for i in range(len(at95) - 1)]
    _, icut = max(jumps)
    hi_runs = {r for _, r in at95[:icut + 1]}
    hi_p95 = [v for v, _ in at95[:icut + 1]]
    lo_p95 = [v for v, _ in at95[icut + 1:]]
    runmodes, n_hi, n_run = [], {g: 0 for g in GROUPS}, {g: 0 for g in GROUPS}
    for c in sorted((c for c in cells if c["rho"] == hard),
                    key=lambda c: (c["group"], c["run_id"])):
        hi = c["run_id"] in hi_runs
        runmodes.append(dict(run_id=c["run_id"], group=c["group"],
                             seed=c["seed"], val_acc=c["val_acc"],
                             params=c["params"], cell=c["cell"],
                             mode="SOLVED-IT" if hi else "STUCK",
                             p_strict_50=c["p_strict"],
                             solve_50=c["solve_rate"]))
        n_run[c["group"]] += 1
        n_hi[c["group"]] += 1 if hi else 0
    variants = {}
    for r in runmodes:
        if r["group"] != "S1":
            continue
        v = variants.setdefault(r["params"], dict(n=0, hi=0, runs=[],
                                                  cell=r["cell"]))
        v["n"] += 1
        v["hi"] += 1 if r["mode"] == "SOLVED-IT" else 0
        v["runs"].append("%s/s%s%s" % (r["run_id"], r["seed"],
                                       "*" if r["mode"] == "SOLVED-IT" else ""))
    fisher_p = float(stats.fisher_exact(
        [[n_hi["S1"], n_run["S1"] - n_hi["S1"]],
         [n_hi["J2"], n_run["J2"] - n_hi["J2"]]])[1])

    cond = {}
    for s in gaps:
        sel = {g: [c for c in cells if c["rho"] == s["rho"]
                   and c["group"] == g and c["run_id"] in hi_runs]
               for g in GROUPS}
        pj, ps = sel["J2"], sel["S1"]
        cond[s["rho"]] = dict(
            n_S1=len(ps), n_J2=len(pj),
            d_strict=med([c["p_strict"] for c in pj])
            - med([c["p_strict"] for c in ps]),
            d_logic=med([c["p_logic"] for c in pj])
            - med([c["p_logic"] for c in ps]),
            d_solve=med([c["solve_rate"] for c in pj])
            - med([c["solve_rate"] for c in ps]),
            prem_ratio=(med([c["premature_rate"] for c in ps])
                        / med([c["premature_rate"] for c in pj])))
    ph_modes(cells, hi_runs)
    L = []
    a = L.append
    a("Stage TIED -- is S1's deficit a deduction deficit or a scoring artefact?")
    a("S1 = tied_no_mlp L=16 d=320 (613,818 params, ONE shared block applied 16")
    a("times).  J2 = no_mlp L=16 d=192 (2,504,890 params), the standing minimum.")
    a("Predictions in REGISTERED.md, committed before this script was written.")
    a("No training, no GPU, no new instances -- a refit of stage LOGIC's CSVs.")
    a("")
    a("  rho     n |  p_strict S1/J2  |  p_logic S1/J2   |   gain S1/J2    | "
      "d_strict d_logic")
    for s in gaps:
        ss = [x for x in summ if x["rho"] == s["rho"] and x["group"] == "S1"][0]
        sj = [x for x in summ if x["rho"] == s["rho"] and x["group"] == "J2"][0]
        a("  %.2f %5.1f | %6.4f / %6.4f | %6.4f / %6.4f | %+6.4f/%+6.4f | "
          "%+7.4f %+7.4f"
          % (s["rho"], s["n"], ss["p_strict"], sj["p_strict"],
             ss["p_logic"], sj["p_logic"], ss["gain"], sj["gain"],
             s["d_strict"], s["d_logic"]))
    a("")

    t1_win = [s for s in gaps if s["gain"] > 0]
    t1_med = med([s["gain"] for s in gaps])
    a("T1 tying is penalised more by strict scoring.")
    a("   median gain(S1) - gain(J2) over the ten densities = %+.4f; "
      "S1 > J2 at %d of %d." % (t1_med, len(t1_win), len(gaps)))
    t1 = t1_med > 0.002 and len(t1_win) > len(gaps) / 2
    a("   predicted: median > +0.002 and a majority -> %s"
      % ("HIT" if t1 else "MISS, FALSIFIED"))
    a("")
    half = [s for s in big if s["d_logic"] < 0.5 * s["d_strict"]]
    ninety = [s for s in big if s["d_logic"] >= 0.9 * s["d_strict"]]
    a("T2 the gap narrows under logical scoring.  Densities with n >= 12: %d."
      % len(big))
    a("   d_logic < 0.5 d_strict at %d of %d; d_logic >= 0.9 d_strict at %d."
      % (len(half), len(big), len(ninety)))
    ratios = [s["d_logic"] / s["d_strict"] for s in big
              if s["d_strict"] not in (0.0,) and s["d_strict"] == s["d_strict"]]
    a("   median d_logic/d_strict = %.3f" % med(ratios))
    t2 = len(half) > len(big) / 2
    a("   predicted: narrows by more than half at a majority -> %s"
      % ("HIT" if t2 else
         ("MISS, FALSIFIED" if len(ninety) > len(big) / 2 else "MISS")))
    a("")
    still = [s for s in big if s["d_logic"] > 0]
    t3 = len(still) > len(big) / 2
    a("T3 it does not fully rescue S1.")
    a("   S1 still behind J2 under logical scoring at %d of %d densities with "
      "n >= 12." % (len(still), len(big)))
    a("   predicted: still behind at a majority -> %s"
      % ("HIT" if t3 else "MISS -- S1 reaches or passes J2 once mis-scoring "
         "is removed"))
    a("")
    prem_up = [s for s in gaps if s["prem_ratio"] > 1.0]
    prem_med = med([s["prem_ratio"] for s in gaps])
    a("T4 the residual deficit is the GUARD, not the rule.")
    a("   node_premature(S1)/node_premature(J2): median %.3f, above 1 at %d "
      "of %d." % (prem_med, len(prem_up), len(gaps)))
    t4 = prem_med > 1.2 and len(prem_up) > len(gaps) / 2
    a("   predicted: median ratio > 1.2 and a majority -> %s"
      % ("HIT" if t4 else
         ("MISS, FALSIFIED" if prem_med < 1.0 else "MISS")))
    a("")
    a("   absolute premature rates (unforced-node steps / all node steps):")
    a("     rho    " + " ".join("%7.2f" % s["rho"] for s in gaps))
    for g in ("S1", "J2"):
        a("     %s     " % g + " ".join(
            "%7.4f" % [x for x in summ
                       if x["rho"] == s["rho"] and x["group"] == g][0]["premature"]
            for s in gaps))
    a("   Same denominator for both groups at every density (verified: they are")
    a("   scored on the same instances), so a ratio of counts IS a ratio of")
    a("   rates -- but the rates are printed because a ratio alone hides that")
    a("   J2's guard is not perfect either.")
    a("")
    a("T5 the gate does not move, and this stage cannot move it.")
    a("   solve_rate is read straight from stage LOGIC and is scored against")
    a("   the planted colouring, so it is ALREADY invariant to within-round")
    a("   emission order.  Nothing here recomputes it.  J2 minus S1 solve rate:")
    a("     " + "  ".join("%.2f:%+.3f" % (s["rho"], s["d_solve"]) for s in gaps))
    a("   rho_50 and the gate verdict for S1 are UNCHANGED.  Nothing in this")
    a("   stage clears the gate or may be reported as having cleared it.")
    a("")
    dmax = max(abs(c["p_logic_color"] - c["p_strict_color"]) for c in cells)
    a("D1 colour positions, definitional check.")
    a("   max |p_logic_color - p_strict_color| over %d cells = %.2e -> %s"
      % (len(cells), dmax, "PASSES" if dmax < 1e-9 else "FAILS, T1-T2 INVALID"))
    a("")
    a("POST-HOC, NOT REGISTERED -- found by printing the per-run table, and it")
    a("changes how every number above should be read.  Runs do not sit on a")
    a("continuum: at rho = 0.95 every run scores either >= %.3f or <= %.3f with"
      % (min(hi_p95), max(lo_p95)))
    a("nothing in between, so ANY cut in that interval gives this same split.")
    a("Call the first mode SOLVED-IT and the second STUCK.")
    a("   run grp seed    params  mode      | p_strict, solve rate at rho=0.50")
    for r in runmodes:
        a("   %-3s %-3s  %s  %8d  %-9s |   %6.4f      %.3f"
          % (r["run_id"], r["group"], r["seed"], r["params"], r["mode"],
             r["p_strict_50"], r["solve_50"]))
    a("   SOLVED-IT rate: S1 %d of %d, J2 %d of %d.  Fisher exact two-sided "
      "p = %.3f" % (n_hi["S1"], n_run["S1"], n_hi["J2"], n_run["J2"], fisher_p))
    a("   -- NOT significant at 12 runs.  The mode RATE is suggestive only.")
    a("")
    a("   A DEFECT IN MY OWN REGISTERED GROUP DEFINITION, found the same way.")
    a("   REGISTERED.md defines S1 by (arch_mode, num_layers, d_model) -- and")
    a("   that pools THREE variants, distinguished by the number k of DISTINCT")
    a("   shared blocks cycled over the 16 applications.  k is in the CSV's")
    a("   `cell` label, which I did not key on.  Only k=1 is S1 as STATUS")
    a("   names it, at 613,818 params:")
    for p in sorted(variants):
        v = variants[p]
        a("     %-32s %8d params: %d runs, %d SOLVED-IT  (%s)"
          % (v["cell"], p, v["n"], v["hi"], " ".join(v["runs"])))
    a("   The steps are +%d and +%d params, and one no_mlp block at d=320 is"
      % (sorted(variants)[1] - sorted(variants)[0],
         sorted(variants)[2] - sorted(variants)[0]))
    a("   4*320^2 + 4*320 + 2*2*320 = 412,160 exactly, so the labels and the")
    a("   parameter counts agree on k = 1, 2, 4.")
    a("   Every k=1 run -- the actual minimal model -- is STUCK, 0 of 3.  Both")
    a("   tied runs that reached the untied mode have k > 1, 2 of 6, and at")
    a("   different seeds (k=2 won on seed 2, k=4 on seed 1).  T1-T5 above are")
    a("   scored on the pooled group because that is what was registered, and")
    a("   they are not re-scored here; but the pooled group was the wrong")
    a("   object and the next registration must key on the `cell` label.")
    a("   What is not in doubt: J2's one STUCK run (r67, 2.5M params, untied) is")
    a("   indistinguishable from the seven STUCK tied runs -- p_strict 0.872 vs")
    a("   0.853-0.866, premature rate 0.205 vs 0.225-0.235, solve 0.04 vs")
    a("   0.02-0.04.  Being STUCK is not something tying does to a model.")
    a("")
    a("   Re-reading T2 and T4 within the SOLVED-IT mode only (2 runs vs 2):")
    a("   rho     n | d_strict all / SOLVED-IT | prem ratio all / SOLVED-IT")
    for s in gaps:
        c = cond[s["rho"]]
        a("   %.2f %5.1f |  %+7.4f  /  %+7.4f    |  %6.2f  /  %6.2f"
          % (s["rho"], s["n"], s["d_strict"], c["d_strict"],
             s["prem_ratio"], c["prem_ratio"]))
    a("   At rho 0.50 the token gap falls from %+.4f to %+.4f (%.1fx smaller)"
      % (gaps[-1]["d_strict"], cond[gaps[-1]["rho"]]["d_strict"],
         gaps[-1]["d_strict"] / cond[gaps[-1]["rho"]]["d_strict"]))
    a("   and T4's premature ratio from %.2f to %.2f.  Across all ten densities"
      % (gaps[-1]["prem_ratio"], cond[gaps[-1]["rho"]]["prem_ratio"]))
    a("   the median premature ratio falls from %.2f to %.2f -- still above"
      % (med([s["prem_ratio"] for s in gaps]),
         med([cond[s["rho"]]["prem_ratio"] for s in gaps])))
    a("   T4's registered 1.2, so the guard deficit survives the conditioning,")
    a("   but at ~30% worse rather than ~700%.  The registered scores")
    a("   above stand as registered -- they were group medians and that is what")
    a("   was predicted -- but a median over a bimodal population mostly reports")
    a("   WHICH MODE THE RUNS ARE IN, not how good the architecture is.")
    a("")
    a("DESCRIPTIVE, POST-HOC -- not registered, do not score it.  Why a token")
    a("gap of 0.08 becomes a solve-rate gap of 0.48: finding 95 says")
    a("S = p_logic^k with no prefactor and k ~ n, so a token-accuracy ratio is")
    a("raised to the n-th power.  Predicted vs observed solve-rate ratio:")
    a("   rho     n | p_J2/p_S1 | (ratio)^n predicted | S_J2/S_S1 observed")
    for s in gaps:
        ss = [x for x in summ if x["rho"] == s["rho"] and x["group"] == "S1"][0]
        sj = [x for x in summ if x["rho"] == s["rho"] and x["group"] == "J2"][0]
        n = s["n"]
        rat = sj["p_logic"] / ss["p_logic"] if ss["p_logic"] else float("nan")
        obs = (sj["solve_rate"] / ss["solve_rate"]
               if ss["solve_rate"] else float("nan"))
        a("   %.2f %5.1f |  %7.5f  |  %17.1f  | %17.1f"
          % (s["rho"], n, rat, rat ** n, obs))
    a("   Order-of-magnitude agreement with no free parameter is the point;")
    a("   exact agreement is not expected, because k is only approximately n")
    a("   and the two groups differ in WHICH positions they get wrong.")
    a("")
    a("SCORECARD: T1 %s, T2 %s, T3 %s, T4 %s, T5 as registered, D1 %s."
      % ("HIT" if t1 else "MISS", "HIT" if t2 else "MISS",
         "HIT" if t3 else "MISS", "HIT" if t4 else "MISS",
         "pass" if dmax < 1e-9 else "FAIL"))
    a("")
    if t1 and t2:
        a("BOTTOM LINE: part of the gap S1 is held out by is the SCORER, not the")
        a("model.  Strict scoring charges the tied block for choosing a legal")
        a("node the reference trace happens to name later.")
    elif t1:
        a("BOTTOM LINE: tying is penalised more by strict scoring, but not by")
        a("enough to account for the gap -- mis-scoring buys back ~14%, so the")
        a("rest of the measured deficit is not an ordering artefact.  The")
        a("post-hoc split says what it is instead: not a lower ceiling but a")
        a("lower hit rate.  2 of 9 tied runs reach the untied mode, and once")
        a("there the token gap is %+.4f rather than %+.4f."
          % (cond[gaps[-1]["rho"]]["d_strict"], gaps[-1]["d_strict"]))
    else:
        a("BOTTOM LINE: the mis-scoring hypothesis is dead.  S1's deficit is")
        a("not an artefact of within-round ordering, and the 0.011 gap stands")
        a("as a real deduction gap.")
    a("")
    a("WHAT THIS DOES NOT ESTABLISH.  The mode split is POST-HOC and rests on")
    a("12 runs, 2 of them in the SOLVED-IT mode per group; Fisher gives p = "
      "%.3f" % fisher_p)
    a("for the 2-of-9 against 2-of-3 rate, which is no evidence of a rate")
    a("difference at all.  The conditional gap is a median over two runs each.")
    a("It is a hypothesis for a seeded sweep, not a measurement.  p_logic is a")
    a("teacher-forced quantity and solve rate is free generation; T5 is the")
    a("reminder that improving the former cannot move the latter here.  And")
    a("nothing was retrained -- this is the same 1,120-cell table as stage")
    a("LOGIC, read two ways.")

    text = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "VERDICT.txt"), "w") as fh:
        fh.write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
