"""Stage LOCAL -- how much of the adjacency join does the colour step do?

103 proved the colour step is exact: 77 of 79 runs emit zero colours that
contradict a visible neighbour, in 40,449 chances.  Its verdict then conceded
that exactness alone cannot tell "computes the candidate set" from "is inhibited
by the neighbour tokens in context", and STATUS now says that needs internals.

That sentence bundles two accounts.  Adjacency-LOCAL inhibition really is the
same function as computing the candidate set and this stage does not claim to
separate it.  But GLOBAL inhibition, and every PARTIAL version of the join, make
different quantitative predictions, and the instance cache settles them with no
model loaded.

The observable is Pi(D): the properness of an emitter that joins against only
k - D of the named cell's k assigned neighbours.  It interpolates exactly
between Pi(0) = 1 and the full-drop rung = 103's Q_null = 0.3379, and the model
sits at the D = 0 end.  Predictions L1-L6 + E1-E2 are in REGISTERED.md,
committed at d6bcc3a before this file existed.

Login node, seconds, nothing submitted.
"""

import collections
import csv
import itertools
import math
import os
import pickle
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line, series_colors  # noqa: E402
from scripts.stagegate_fit import GRID, FIT_RHOS  # noqa: E402
from scripts.stagecand_fit import (hint_asg, cand_and_degree,  # noqa: E402
                                   load_errors0, load_hazard0, INST,
                                   N_PER_BIN, SEED_START_TAG, Q)
from scripts.stageproper_fit import load_bhaz0  # noqa: E402

OUT = os.environ.get("LOCAL_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stagelocal_2026-11-19"
FIGS = os.path.join(OUT, "figures")
DPI = 150
DMAX = 4                      # rungs reported as a fixed-D ladder
KVALS = [1, 2, 3, 4, 5]
KMIN_STRAT = 200              # a k-stratum below this is not reported as one


def instance_adj(rho):
    """Per instance at this density: adjacency, the HINT assignment (which is
    the j = 0 pre-state exactly), and c/k from CAND's own arithmetic so the
    three stages cannot drift."""
    path = os.path.join(INST, "inst_rho%05d_n%d_s%d.pkl"
                        % (round(rho * 10000), N_PER_BIN, SEED_START_TAG))
    with open(path, "rb") as fh:
        inst = pickle.load(fh)
    out = []
    for adj, colors, hints, trace in inst:
        asg = hint_asg(adj.shape[0], hints)
        c, k = cand_and_degree(adj, asg)
        out.append((adj, asg, c, k, asg < 0))
    return out


def cell_ladder(nbr_cols, c):
    """For one named cell: Pi(D) for every D = 0..k, exactly, by enumerating
    the C(k, k-D) retained subsets.  `nbr_cols` is the MULTISET of colours on
    the assigned neighbours, so duplicates are what make a neighbour droppable.

    An emitter retaining subset A sees S_A, avoids the planted colour and draws
    uniformly from the rest, so it is proper with probability
    (c-1)/(q - |S_A| - 1).  Unforced means |S| <= q-2, hence |S_A| <= q-2 and
    the denominator is at least 1."""
    k = len(nbr_cols)
    out = []
    for D in range(k + 1):
        m = k - D
        tot, n = 0.0, 0
        for A in itertools.combinations(range(k), m):
            sa = len({nbr_cols[i] for i in A})
            den = Q - sa - 1
            tot += (c - 1.0) / den
            n += 1
        out.append(tot / n)
    return out


def main():
    os.makedirs(FIGS, exist_ok=True)
    errs = load_errors0()
    bh = load_bhaz0()
    fh = load_hazard0()

    # ---- the two carrier runs 103 found, recomputed its way exactly --------
    # improper-and-wrong-and-unforced = n_emit - n_proper - n_err_in_F, the
    # numerator of 1 - Q_u.  Anything looser counts forced-cell errors and
    # finds carriers 103 did not have.
    imp_by_run = collections.Counter()
    for key, b in bh.items():
        if key[1] in FIT_RHOS and key in fh:
            imp_by_run[key[0]] += (b["n_emit"] - b["n_proper"]
                                   - fh[key]["n_err_in_F"])
    carriers = {rid for rid, v in imp_by_run.items() if v > 0}

    by_rho = collections.defaultdict(list)
    for e in errs:
        by_rho[e[1]].append(e)

    L = []

    def A(s=""):
        L.append(s)
        print(s)

    A("STAGE LOCAL -- how much of the adjacency join does the colour step do?")
    A("=" * 78)
    A("j=0 error records %d   runs with any improper draw (fit densities): %d"
      % (len(errs), len(carriers)))
    A("")
    A("Pi(D) = properness of an emitter joining against only k-D of the named")
    A("        cell's k assigned neighbours, averaged over which D are dropped.")
    A("        Pi(0) = 1 identically; the full-drop rung is 103's Q_null")
    A("        rescaled to unforced cells -- see E1, which caught that.")
    A("")

    rows, strat = [], []
    e2_bad = 0
    pooled_num = collections.Counter()   # D -> summed Pi
    pooled_den = collections.Counter()   # D -> contributing cells
    pooled_full_num, pooled_full_den = 0.0, 0
    pooled_qn_num, pooled_qn_den = 0.0, 0   # 103's population: forced INCLUDED
    pooled_red_num, pooled_red_den = 0, 0
    e2_k = 0
    pooled_k_num = collections.Counter()  # k -> summed Pi(1)
    pooled_k_den = collections.Counter()
    pooled77_num, pooled77_den = 0.0, 0   # Pi(1) on the 77 non-carriers
    skipped_colored = 0

    for rho in GRID:
        tab = instance_adj(rho)
        num = collections.Counter()
        den = collections.Counter()
        full_num, full_den = 0.0, 0
        qn_num, qn_den = 0.0, 0
        red_num, red_den = 0, 0
        allq_hit, allq_tot = 0, 0
        for ii in range(len(tab)):
            asg = tab[ii][1]
            allq_tot += 1
            allq_hit += int(len({int(x) for x in asg if x >= 0}) == Q)

        for (rid, r, inst, node, in_F, kind) in by_rho[rho]:
            adj, asg, c, k, free = tab[inst]
            if not free[node]:
                skipped_colored += 1
                continue
            # 103's Q_null population: every FREE named cell, forced ones
            # contributing (c-1)/3 = 0.  Reproduced here so E1 can compare.
            qn_num += (int(c[node]) - 1) / 3.0
            qn_den += 1
            if c[node] < 2:                       # forced: (c-1)/... is 0/0
                continue
            nbr_cols = [int(x) for x in asg[adj[node]] if x >= 0]
            kk = len(nbr_cols)
            if kk != int(k[node]):
                e2_k += 1
            lad = cell_ladder(nbr_cols, int(c[node]))
            if abs(lad[0] - 1.0) > 1e-12:
                e2_bad += 1
            for D in range(min(DMAX, kk) + 1):
                num[D] += lad[D]
                den[D] += 1
            full_num += lad[kk]
            full_den += 1
            red_num += int(kk > len(set(nbr_cols)))
            red_den += 1
            if kk >= 1 and rho in FIT_RHOS:
                pooled_k_num[kk] += lad[1]
                pooled_k_den[kk] += 1
                if rid not in carriers:
                    pooled77_num += lad[1]
                    pooled77_den += 1

        row = dict(rho=rho, held_out=int(rho not in FIT_RHOS),
                   n_cells=den[0], n_named=qn_den,
                   allq=allq_hit / allq_tot if allq_tot else float("nan"),
                   full=full_num / full_den if full_den else float("nan"),
                   qnull=qn_num / qn_den if qn_den else float("nan"),
                   red=red_num / red_den if red_den else float("nan"))
        for D in range(DMAX + 1):
            row["pi%d" % D] = num[D] / den[D] if den[D] else float("nan")
            row["n%d" % D] = den[D]
        rows.append(row)
        if rho in FIT_RHOS:
            for D in range(DMAX + 1):
                pooled_num[D] += num[D]
                pooled_den[D] += den[D]
            pooled_full_num += full_num
            pooled_full_den += full_den
            pooled_qn_num += qn_num
            pooled_qn_den += qn_den
            pooled_red_num += red_num
            pooled_red_den += red_den

    fit = [r for r in rows if not r["held_out"]]
    pi = {D: pooled_num[D] / pooled_den[D] if pooled_den[D] else float("nan")
          for D in range(DMAX + 1)}
    pi_full = pooled_full_num / pooled_full_den
    qnull_repro = pooled_qn_num / pooled_qn_den
    frac_unf = pooled_full_den / pooled_qn_den
    red = pooled_red_num / pooled_red_den

    # ---------------------------------------------------------------- E1, E2
    qnull_103 = 0.3379
    A("-" * 78)
    A("E1 -- THE FULL-DROP RUNG AGAINST 103's Q_null")
    A("")
    A("  THE REGISTRATION GOT THIS WRONG AND E1 IS WHY IT EXISTS.  It asserted")
    A("  Pi(k) = Q_null 'to the digit'.  It is not: 103's Q_null averaged over")
    A("  every FREE named cell, with forced ones (c = 1) contributing 0, while")
    A("  the ladder is defined only on UNFORCED cells because c = 1 makes")
    A("  Pi(0) = 0/0.  The two differ by exactly the unforced share.")
    A("")
    A("  103's Q_null recomputed here on 103's own population: %.4f" % qnull_repro)
    A("  103's published Q_null:                               %.4f" % qnull_103)
    e1a = abs(qnull_repro - qnull_103) <= 0.0005
    A("    |difference| %.6f  (bar 0.0005)  ->  %s"
      % (abs(qnull_repro - qnull_103), "PASS" if e1a else "FAIL"))
    A("")
    A("  full-drop rung on the LADDER's population (unforced only): %.4f" % pi_full)
    A("  unforced share of named free cells:                        %.4f" % frac_unf)
    A("  identity Q_null = (unforced share) x (full-drop rung):     %.4f"
      % (frac_unf * pi_full))
    e1b = abs(frac_unf * pi_full - qnull_repro) <= 0.0005
    A("    |difference from Q_null| %.6f  (bar 0.0005)  ->  %s"
      % (abs(frac_unf * pi_full - qnull_repro), "PASS" if e1b else "FAIL"))
    e1_ok = e1a and e1b
    A("  E1 overall: %s" % ("PASS" if e1_ok else "FAIL"))
    A("")
    A("E2 -- Pi(0) = 1 ON EVERY CELL, AND k AGREES WITH CAND's")
    A("  cells checked %d   Pi(0) violations %d   k mismatches %d  ->  %s"
      % (pooled_den[0], e2_bad, e2_k,
         "PASS" if not (e2_bad or e2_k) else "FAIL"))
    A("  DOMAIN: E1 over the ten fit densities on the matched named-cell")
    A("  population; E2 over every named cell read, forced cells excluded")
    A("  because c = 1 makes the ratio 0/0.  %d records naming an ALREADY-"
      % skipped_colored)
    A("  COLOURED cell were skipped, as in 103.")
    A("")

    # ------------------------------------------------------------------- L1
    allq_min = min(r["allq"] for r in fit)
    l1_hit = all(r["allq"] >= 0.99 for r in fit)
    A("-" * 78)
    A("L1 -- DOES GLOBAL COLOUR PRESENCE CARRY ANY INFORMATION (check)")
    A("  fraction of instances in which all 4 colours appear among the hints:")
    A("   " + "  ".join("%.2f:%.4f" % (r["rho"], r["allq"]) for r in fit[:5]))
    A("   " + "  ".join("%.2f:%.4f" % (r["rho"], r["allq"]) for r in fit[5:]))
    A("  minimum over the ten fit densities %.4f  (bar >= 0.99)  ->  %s"
      % (allq_min, "HIT" if l1_hit else "MISS"))
    A("  DISCLOSED AS NEAR-CERTAIN IN ADVANCE.  What it buys is account (B1):")
    A("  if every colour is always visible in context, an emitter inhibited by")
    A("  GLOBAL colour presence has no basis to choose and cannot reach 1.0000.")
    A("")

    # --------------------------------------------------------------- L2, L3
    l2_hit = 0.70 <= pi[1] <= 0.90
    l2_fals = (pi[1] >= 0.95) or (pi[1] <= 0.55)
    l3_hit = 0.40 <= pi[2] <= 0.70
    mono_bad = sum(1 for r in fit
                   if not all(r["pi%d" % D] > r["pi%d" % (D + 1)]
                              for D in range(DMAX)
                              if r["pi%d" % (D + 1)] == r["pi%d" % (D + 1)]))
    l3_fals = (pi[2] >= 0.80) or (mono_bad >= 3)
    A("-" * 78)
    A("L2/L3 -- THE LADDER")
    A("    D   Pi(D)   cells contributing")
    for D in range(DMAX + 1):
        A("   %2d  %.4f   %d" % (D, pi[D], pooled_den[D]))
    A("   full %.4f   %d   <- D = k per cell; x %.4f unforced share = Q_null"
      % (pi_full, pooled_full_den, frac_unf))
    A("  L2: Pi(1) = %.4f in [0.70, 0.90]  ->  %s   falsifier %s"
      % (pi[1], "HIT" if l2_hit else "MISS",
         "TRIPPED" if l2_fals else "not tripped"))
    A("  L3: Pi(2) = %.4f in [0.40, 0.70]  ->  %s   falsifier %s"
      % (pi[2], "HIT" if l3_hit else "MISS",
         "TRIPPED" if l3_fals else "not tripped"))
    A("      strictly decreasing at %d of 10 fit densities" % (len(fit) - mono_bad))
    A("  NOTE ON THE DENOMINATOR: Pi(D) averages only over cells with k >= D,")
    A("  so the population shrinks as D grows and the far rungs are not a")
    A("  fixed-population comparison.  Counts are printed above for that reason.")
    A("")

    # ------------------------------------------------------------------- L4
    l4_hit = all(r["red"] >= 0.50 for r in fit)
    l4_fals = sum(1 for r in fit if r["red"] < 0.30) >= 4
    A("-" * 78)
    A("L4 -- HOW MUCH SLACK THE JOIN HAS")
    A("  named cells with at least one DROPPABLE neighbour (its colour also")
    A("  appears on another assigned neighbour, so S is unchanged):")
    A("   " + "  ".join("%.2f:%.3f" % (r["rho"], r["red"]) for r in fit[:5]))
    A("   " + "  ".join("%.2f:%.3f" % (r["rho"], r["red"]) for r in fit[5:]))
    A("  pooled %.4f   minimum %.4f  (bar >= 0.50 at every fit density)  ->  %s"
      % (red, min(r["red"] for r in fit), "HIT" if l4_hit else "MISS"))
    A("  falsifier (< 0.30 at >= 4 of 10): %s"
      % ("TRIPPED" if l4_fals else "not tripped"))
    A("")

    # ------------------------------------------------------------------- L5
    pik = {k: pooled_k_num[k] / pooled_k_den[k] if pooled_k_den[k] else float("nan")
           for k in KVALS}
    kpop = [k for k in KVALS if pooled_k_den[k] >= KMIN_STRAT]
    A("-" * 78)
    A("L5 -- Pi(1) BY k, THE NUMBER OF ASSIGNED NEIGHBOURS (direction a check)")
    A("    k   Pi(1|k)   cells")
    for k in KVALS:
        A("   %2d   %s   %d"
          % (k, ("%.4f" % pik[k]) if pooled_k_den[k] else "  n/a ",
             pooled_k_den[k]))
        strat.append(dict(k=k, pi1=pik[k], n=pooled_k_den[k]))
    A("")
    A("  L5 IS VOID, AND THE REASON IS THE INTERESTING PART.  It was registered")
    A("  as Pi(1|k=2) <= Pi(1|k=5) - 0.10, but THERE ARE NO k = 5 CELLS: %d of"
      % pooled_k_den[5])
    A("  %d.  A named cell with all five neighbours already coloured is unforced"
      % sum(pooled_k_den[k] for k in KVALS))
    A("  only if those five carry at most two distinct colours, and at q = 4 on")
    A("  a 5-regular graph that never happened.  The registered comparison has")
    A("  an empty arm, so it is scored VOID rather than MISS -- the bar was")
    A("  unmeetable when it was written and I did not check the support first.")
    A("")
    gap4 = pik[4] - pik[2]
    l5_hit = False
    l5_fals = False
    A("  REPORTED INSTEAD, AND NOT AS A HIT: the largest populated stratum is")
    A("  k = 4, and Pi(1|k=4) - Pi(1|k=2) = %+.4f, which clears the registered"
      % gap4)
    A("  +0.10 margin.  This is a SUBSTITUTED comparison chosen after seeing the")
    A("  supports, so it scores nothing; it is here because the monotone trend")
    A("  across k = 1..4 is what L5 was trying to look at.")
    A("  THE SIGN WAS DISCLOSED AS PIGEONHOLE; only the magnitude was live.")
    A("")

    # ------------------------------------------------------------------- L6
    n_emis = 24008            # 103: wrong unforced emissions, 77 runs, fit rhos
    n_obs = 0                 # 103: improper draws among them
    pi1_77 = pooled77_num / pooled77_den if pooled77_den else float("nan")
    miss_cost = 1.0 - pi1_77
    bound = 1.0 - 3.0 / (n_emis * miss_cost) if miss_cost > 0 else float("nan")
    l6_hit = bound > 0.999
    A("-" * 78)
    A("L6 -- THE HEADLINE: HOW OFTEN MUST THE JOIN COMPLETE")
    A("  Pi(1) on the matched cells of the 77 NON-CARRIER runs: %.4f" % pi1_77)
    A("  so an emission that drops one neighbour is improper with prob %.4f"
      % miss_cost)
    A("  103 observed %d improper draws in %d wrong unforced emissions there."
      % (n_obs, n_emis))
    A("  Rule of three: the 95% upper bound on the rate of such emissions is")
    A("  3/%d = %.3e, so the join-completion rate is bounded below by"
      % (n_emis, 3.0 / n_emis))
    A("     pi >= 1 - 3/(%d * %.4f) = %.6f" % (n_emis, miss_cost, bound))
    A("  registered: > 0.999, falsified below 0.99  ->  %s"
      % ("HIT" if l6_hit else "MISS"))
    A("  THE HIT IS MARGINAL AND I AM NOT GOING TO DRESS IT UP: 0.999004")
    A("  against a bar of 0.999000.  The bar was set by the arithmetic of the")
    A("  rule of three on a count I already knew (24,008), so writing 0.999")
    A("  was close to writing the answer; a fourth digit either way would have")
    A("  flipped it.  The robust statement is the one below it -- ~%.0f expected"
      % (n_emis * miss_cost))
    A("  violations against 0 observed -- which does not depend on the bar.")
    A("  expected improper draws if the join dropped one neighbour EVERY time:")
    A("     %.0f.  Observed: 0." % (n_emis * miss_cost))
    A("")

    # ------------------------------------------------------------- scorecard
    A("=" * 78)
    A("SCORECARD")
    for name, hit in [("L1 global presence uninformative (check)", l1_hit),
                      ("L2 dropping one neighbour is expensive", l2_hit),
                      ("L3 the ladder keeps falling", l3_hit),
                      ("L4 the join has slack", l4_hit),
                      ("L6 the join completes essentially always", l6_hit)]:
        A("  %-44s %s" % (name, "HIT" if hit else "MISS"))
    A("  %-44s %s" % ("L5 small-k cells are unforgiving (check)",
                      "VOID -- no k=5 cells exist"))
    A("  %-44s %s" % ("E1 full-drop rung == 103's Q_null",
                      "PASS" if e1_ok else "FAIL"))
    A("  %-44s %s" % ("E2 Pi(0) == 1", "PASS" if not e2_bad else "FAIL"))
    A("")
    A("=" * 78)
    A("WHAT THIS ANSWERS, AND WHAT IT LEAVES OPEN")
    A("")
    A("  103 showed the colour step never writes an illegal colour.  This")
    A("  stage asks what computation that requires, and the answer is: the")
    A("  COMPLETE join, on essentially every emission.")
    A("")
    A("  An emitter that joined against all but one of the named cell's")
    A("  assigned neighbours would be improper %.1f%% of the time.  Over the"
      % (100 * miss_cost))
    A("  24,008 wrong unforced emissions 103 measured on the 77 exact runs")
    A("  that is ~%.0f expected violations.  Zero were observed, which bounds"
      % (n_emis * miss_cost))
    A("  the join-completion rate at >= %.4f with 95%% confidence." % bound)
    A("")
    A("  ACCOUNT (B1), GLOBAL INHIBITION, IS DEAD.  All four colours appear")
    A("  among the hints in %.2f%% of instances, so 'this colour is visible in"
      % (100 * allq_min))
    A("  context' is true of every colour nearly always and cannot select the")
    A("  1-2 the model actually suppresses.  The suppressed set is cell-")
    A("  specific, and identifying it requires reading the named node's row of")
    A("  BEGIN_GRAPH and joining it against BEGIN_HINTS.")
    A("")
    A("  ACCOUNT (B2), ADJACENCY-LOCAL INHIBITION, SURVIVES -- AND THAT WAS")
    A("  DISCLOSED.  Suppressing exactly the colours on this cell's neighbours")
    A("  IS computing the candidate set; the two are the same function and no")
    A("  output-only measurement can separate them.  What has changed is that")
    A("  the surviving account is now a specific, named, two-hop computation")
    A("  rather than 'some mechanism that agrees'.")
    A("")
    A("  AND THE SLACK IS THE ARCHITECTURAL FINDING.  %.1f%% of named cells"
      % (100 * red))
    A("  have at least one neighbour whose colour is duplicated on another")
    A("  neighbour, so a correct emitter does not need equal attention on all")
    A("  k.  For the smallest-analyzable-architecture brief that is the useful")
    A("  number: the join must be complete in EFFECT, not in EFFORT.")

    txt = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "VERDICT.txt"), "w") as f:
        f.write(txt)

    with open(os.path.join(OUT, "local_density.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(OUT, "local_strata.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "pi1", "n"])
        w.writeheader()
        for r in strat:
            w.writerow(r)

    # ----------------------------------------------------------------- figures
    Ds = list(range(DMAX + 1))

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(Ds, [pi[D] for D in Ds], "o-", color=SERIES[0], ms=6,
            rasterized=True)
    ax.axhline(pi_full, color=MUTED, ls=":", lw=1.2)
    ax.text(0.03, 0.16, "dotted: drop every neighbour, %.4f\n"
            "(103's colour-blind null, rescaled to unforced cells)" % pi_full,
            transform=ax.transAxes, fontsize=8, color=MUTED, ha="left")
    ax.plot([0], [1.0], "*", color=SERIES[1], ms=16, rasterized=True,
            clip_on=False)
    ax.text(0.97, 0.93, "the model sits at $D=0$:\n0 illegal colours in 24,008",
            transform=ax.transAxes, fontsize=9, color=SERIES[1], ha="right",
            va="top")
    ax.set_ylim(0.25, 1.06)
    style_axes(ax, xlabel="assigned neighbours dropped from the join, $D$",
               ylabel=r"$\Pi(D)$: chance a wrong colour is still legal")
    ax.set_xticks(Ds)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "L2_ladder.png"), dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = [r["rho"] for r in rows]
    cols = series_colors(2)
    for i, D in enumerate([1, 2]):
        ax.plot(xs, [r["pi%d" % D] for r in rows], "o-", color=cols[i], ms=4,
                rasterized=True)
        label_line(ax, xs[-1], rows[-1]["pi%d" % D], "$D=%d$" % D, color=cols[i])
    style_axes(ax, xlabel=r"hint density $\rho$  (harder $\rightarrow$)",
               ylabel=r"$\Pi(D)$")
    ax.invert_xaxis()
    ax.axvline(0.475, color=MUTED, lw=0.8, ls="--")
    ax.text(0.465, 0.98, "extension\n(decides nothing)", transform=
            ax.get_xaxis_transform(), fontsize=7.5, color=MUTED, ha="left",
            va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "L3_density.png"), dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(xs, [r["red"] for r in rows], "o-", color=SERIES[0], ms=5,
            rasterized=True)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.annotate("registered floor 0.50", (xs[0], 0.5), textcoords="offset points",
                xytext=(8, 6), fontsize=8, color=MUTED)
    style_axes(ax, xlabel=r"hint density $\rho$  (harder $\rightarrow$)",
               ylabel="named cells with a droppable neighbour")
    ax.invert_xaxis()
    ax.axvline(0.475, color=MUTED, lw=0.8, ls="--")
    ax.text(0.465, 0.98, "extension\n(decides nothing)", transform=
            ax.get_xaxis_transform(), fontsize=7.5, color=MUTED, ha="left",
            va="top")
    ax.set_ylim(0, 1.20)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "L4_redundancy.png"), dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(kpop, [pik[k] for k in kpop], color=SERIES[0], width=0.6,
           rasterized=True)
    for k in kpop:
        ax.annotate("%.3f\nn=%s" % (pik[k], format(pooled_k_den[k], ",")),
                    (k, pik[k]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8, color=MUTED)
    ax.text(0.98, 0.99, "no $k=5$ cells: whenever all five\nneighbours were "
            "coloured they carried\n$\\geq 3$ colours, so the cell was forced",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=SERIES[1])
    ax.text(0.02, 0.99, "$k=1$ dropped:\nonly %d cells" % pooled_k_den[1],
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color=MUTED)
    style_axes(ax, xlabel="$k$, assigned neighbours of the named cell",
               ylabel=r"$\Pi(1)$")
    ax.set_xticks(KVALS)
    ax.set_ylim(0, 1.55)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "L5_strata.png"), dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    exp = [n_emis * (1.0 - pi[D]) for D in Ds]
    ax.bar(Ds[1:], exp[1:], color=SERIES[0], width=0.6, rasterized=True)
    ax.set_yscale("symlog", linthresh=1)
    ax.axhline(0, color="k", lw=0.8)
    for D in Ds[1:]:
        ax.annotate(format(int(round(exp[D])), ","), (D, exp[D]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, color=MUTED)
    ax.plot([0], [0], "*", color=SERIES[1], ms=16, rasterized=True,
            clip_on=False)
    ax.set_ylim(0, 4e5)
    ax.text(0.5, 0.97, "$D=0$ is the only rung that predicts 0 "
            "-- and 103 observed 0 in 24,008 chances",
            transform=ax.transAxes, fontsize=9, color=SERIES[1], va="top",
            ha="center")
    style_axes(ax, xlabel="assigned neighbours dropped from the join, $D$",
               ylabel="expected illegal colours in 24,008 emissions")
    ax.set_xticks(Ds)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "L6_bound.png"), dpi=DPI)
    plt.close(fig)

    print("")
    print("figures -> %s" % FIGS)


if __name__ == "__main__":
    main()
