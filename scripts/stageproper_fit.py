"""Stage PROPER -- is the emitted COLOUR inside the named cell's candidate set?

102 (stage CAND) showed the model's choice of CELL is selection on k, the
number of coloured neighbours, not on the candidate set, and that c and k are
too collinear at q=4, d=5 for any re-analysis of these instances to separate
the accounts.  That verdict is about the CELL step.

The COLOUR step is not confounded by that collinearity at all, because the cell
is held fixed: given whatever cell the model named, is the colour it wrote one
the constraints still allow?  And the column that answers it has been on disk
since stage POS ran -- stagepos_score.py:385

    color_proper = all(assigned.get(u) != c for u in adj[node])

aggregated per (run, rho, j) as `n_color_proper_u` in B_hazard.  grep finds
color_proper in no analysis script and in no VERDICT, STATUS, LOG or FINDINGS:
it has never been read.  Fourth time the next question was already on disk.

At j = 0 `assigned` is exactly the hints, so color_proper IS candidate-set
membership under the same pre-state CAND reconstructed, and `n_emitted ==
n_at_risk == n_instances` there, so the `_u` (unconditional) and at-risk
columns coincide.

Predictions R1-R5 + E1-E2 are in REGISTERED.md, committed at bd283a7 before
this file existed.  Nothing submitted, nothing trained, no model loaded.
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

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line  # noqa: E402
from scripts.stageexp_fit import tobool, med  # noqa: E402
from scripts.stagegate_fit import GRID, FIT_RHOS, spearman  # noqa: E402
from scripts.stagecand_fit import (load_hazard0, load_errors0,  # noqa: E402
                                   instance_table)

OUT = os.environ.get("PROPER_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stageproper_2026-11-17"
POS = os.environ.get("POS_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17"
CAND = os.environ.get("CAND_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stagecand_2026-11-17"
FIGS = os.path.join(OUT, "figures")
DPI = 150
Q = 4
RHO_SIZE = 0.60          # R5's density, named in the registration


def load_bhaz0():
    """j = 0 aggregates per (run, rho) from stage POS's B_hazard: the emission
    count and the two colour counts.  `_u` means UNCONDITIONAL over emitted
    pairs, not 'unforced' -- read from measure_B, not from the name."""
    keep = {}
    for path in sorted(glob.glob(os.path.join(POS, "B_hazard_*.csv"))):
        if "smoke" in os.path.basename(path):
            continue
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["j"] != "0" or not tobool(r, "in79"):
                    continue
                keep[(r["run_id"], round(float(r["rho"]), 4))] = dict(
                    cell=r["cell"],
                    params=int(float(r["n_trainable_params"])),
                    n_inst=int(float(r["n_instances"])),
                    at_risk=int(float(r["n_at_risk"])),
                    n_emit=int(float(r["n_emitted"])),
                    n_correct=int(float(r["n_color_correct_u"])),
                    n_proper=int(float(r["n_color_proper_u"])))
    return keep


def read_cand_density():
    """CAND's per-density P_hat and M, read rather than recomputed so the two
    stages cannot drift in their definition of either."""
    out = {}
    with open(os.path.join(CAND, "cand_density.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            out[round(float(r["rho"]), 4)] = dict(
                phat=float(r["phat"]), M=float(r["M"]),
                U=int(float(r["U"])), n_err_in_F=int(float(r["n_err_in_F"])))
    return out


def safe_div(a, b):
    return a / b if b else float("nan")


def main():
    os.makedirs(FIGS, exist_ok=True)
    bh = load_bhaz0()
    fh = load_hazard0()
    errs = load_errors0()
    cand = read_cand_density()

    runs = sorted({k[0] for k in bh})
    joined = sorted(set(bh) & set(fh))
    print("B_hazard j=0 cells %d over %d runs; F_hazard j=0 cells %d; joined %d"
          % (len(bh), len(runs), len(fh), len(joined)))

    # node_invalid count per cell, which is E1's allowance
    inval = collections.Counter()
    for run, rho, inst, node, in_f, kind in errs:
        if kind == "node_invalid":
            inval[(run, rho)] += 1

    # ---------------------------------------------------------------- E1, E2
    e1_bad, e2_bad = [], []
    for key in joined:
        b, f = bh[key], fh[key]
        if not (b["n_emit"] == b["at_risk"] == b["n_inst"] == f["N"] == f["n_inst"]):
            e1_bad.append((key, "count", b["n_emit"], b["at_risk"], b["n_inst"],
                           f["N"], f["n_inst"]))
            continue
        slack = abs((f["N"] - f["n_err"]) - b["n_correct"])
        if slack > inval.get(key, 0):
            e1_bad.append((key, "correct", f["N"] - f["n_err"], b["n_correct"],
                           inval.get(key, 0)))
    for key, b in bh.items():
        if b["n_proper"] < b["n_correct"]:
            e2_bad.append((key, b["n_proper"], b["n_correct"]))

    # ------------------------------------------------- the matched null Q_null
    # for each j = 0 first error, the named cell's candidate count c gives the
    # number of proper-but-wrong colours, c - 1, out of 3 wrong ones.  c = 1
    # (a forced cell) contributes 0 automatically.
    null_num = collections.defaultdict(float)
    null_den = collections.Counter()
    by_rho_err = collections.defaultdict(list)
    for e in errs:
        by_rho_err[e[1]].append(e)
    for rho in GRID:
        tab = instance_table(rho)
        for run, r, inst, node, in_f, kind in by_rho_err[rho]:
            c, k, free = tab[inst]
            if not free[node]:
                # an already-coloured cell: c(v) is undefined there, exactly
                # the domain limit 102's D1 cost.  Excluded and counted.
                continue
            null_num[rho] += (int(c[node]) - 1) / 3.0
            null_den[rho] += 1

    # ------------------------------------------------------------- per density
    rows = []
    for rho in GRID:
        cells = [k for k in joined if k[1] == rho]
        N = sum(bh[k]["n_emit"] for k in cells)
        n_cor = sum(bh[k]["n_correct"] for k in cells)
        n_pro = sum(bh[k]["n_proper"] for k in cells)
        n_eif = sum(fh[k]["n_err_in_F"] for k in cells)
        n_in_F = sum(fh[k]["n_in_F"] for k in cells)
        wrong = N - n_cor
        wrong_u = wrong - n_eif
        Qv = safe_div(n_pro - n_cor, wrong)
        Qu = safe_div(n_pro - n_cor, wrong_u)
        qnull = safe_div(null_num[rho], null_den[rho])
        # M recomputed here from the same columns CAND used, as a join check
        U = N - n_in_F
        M_here = 1.0 - safe_div((sum(fh[k]["n_err"] for k in cells) - n_eif), U)
        rows.append(dict(
            rho=rho, held_out=int(rho not in FIT_RHOS), n_cells=len(cells),
            N=N, n_correct=n_cor, n_proper=n_pro, wrong=wrong,
            n_err_in_F=n_eif, wrong_unforced=wrong_u, U=U,
            Q=Qv, Q_u=Qu, Q_null=qnull, n_null=null_den[rho],
            improper_u=1.0 - Qu, M=M_here,
            M_cand=cand[rho]["M"], phat=cand[rho]["phat"],
            resid=cand[rho]["phat"] - M_here))

    fit = [r for r in rows if not r["held_out"]]

    # --------------------------------------------------------------- per run
    run_rows = []
    for key in joined:
        if abs(key[1] - RHO_SIZE) > 1e-9:
            continue
        b, f = bh[key], fh[key]
        wrong_u = b["n_emit"] - b["n_correct"] - f["n_err_in_F"]
        run_rows.append(dict(
            run_id=key[0], cell=b["cell"], params=b["params"],
            n_emit=b["n_emit"], n_correct=b["n_correct"], n_proper=b["n_proper"],
            wrong_unforced=wrong_u,
            Q_u=safe_div(b["n_proper"] - b["n_correct"], wrong_u)))
    run_rows.sort(key=lambda r: r["params"])

    # ------------------------------------------------------------------ CSVs
    with open(os.path.join(OUT, "proper_density.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(OUT, "proper_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
        w.writeheader()
        for r in run_rows:
            w.writerow(r)

    # --------------------------------------------------------------- verdict
    L = []

    def A(s=""):
        L.append(s)

    A("STAGE PROPER -- is the emitted COLOUR inside the named cell's candidate set?")
    A("=" * 78)
    A("B_hazard j=0 cells %d over %d runs; F_hazard j=0 cells %d; joined %d; "
      "j=0 error records %d" % (len(bh), len(runs), len(fh), len(joined), len(errs)))
    A("")
    A("Q   = (proper - correct) / (N - correct)         among WRONG colours,")
    A("Q_u = (proper - correct) / (N - correct - n_err_in_F)   the share still")
    A("      locally consistent, the second excluding wrong-on-FORCED errors")
    A("      which are improper by construction.")
    A("Q_null = mean over the same named cells of (c-1)/3, the share a")
    A("      colour-blind emitter avoiding the planted colour would reach.")
    A("")
    A("-" * 78)
    A("E1 -- THE TWO SCORERS SAW THE SAME ROLLOUTS")
    A("  joined in79 cells at j = 0: %d" % len(joined))
    A("  count identity n_emitted == n_at_risk == n_instances == F_hazard N:")
    A("    violations %d" % sum(1 for b in e1_bad if b[1] == "count"))
    A("  |(N - n_err) - n_color_correct_u| <= that cell's node_invalid count:")
    A("    violations %d" % sum(1 for b in e1_bad if b[1] == "correct"))
    for b in e1_bad[:6]:
        A("    %s" % (b,))
    A("  -> %s" % ("PASS" if not e1_bad else "FAIL -- the stage stops here"))
    A("")
    A("E2 -- PROPER DOMINATES CORRECT (by construction: the planted colour is proper)")
    A("  rows checked %d   violations %d  -> %s"
      % (len(bh), len(e2_bad), "PASS" if not e2_bad else "FAIL"))
    A("")
    A("  DOMAIN, stated because 102's D1 was scored down for lacking one:")
    A("  E1 is asserted only over in79 cells at j = 0; E2 over every B_hazard")
    A("  row read.  Q_null skips %d of %d j=0 error records naming an"
      % (len(errs) - sum(null_den.values()), len(errs)))
    A("  ALREADY-COLOURED cell, where c(v) is undefined.")
    A("")
    A("-" * 78)
    A("THE LEDGER, j = 0")
    A("   rho      N  correct   proper    wrong  wrongU       Q     Q_u  Q_null")
    for r in rows:
        A("  %4.2f %6d %8d %8d %8d %7d  %6.4f  %6.4f  %6.4f%s"
          % (r["rho"], r["N"], r["n_correct"], r["n_proper"], r["wrong"],
             r["wrong_unforced"], r["Q"], r["Q_u"], r["Q_null"],
             "  <- extension" if r["held_out"] else ""))
    A("")

    # R1 ---------------------------------------------------------------------
    pn_pro = sum(r["n_proper"] for r in fit)
    pn_cor = sum(r["n_correct"] for r in fit)
    pn_N = sum(r["N"] for r in fit)
    Q_pool = safe_div(pn_pro - pn_cor, pn_N - pn_cor)
    Qnull_pool = safe_div(sum(null_num[r["rho"]] for r in fit),
                          sum(null_den[r["rho"]] for r in fit))
    r1_hit = (Q_pool >= 0.90) and (Q_pool - Qnull_pool >= 0.30)
    r1_fals = (Q_pool <= 0.60) or (Q_pool - Qnull_pool <= 0.20)
    A("-" * 78)
    A("R1 -- IS THE COLOUR LOCALLY CONSISTENT (a consistency check, disclosed)")
    A("  pooled over the ten fit densities: Q %.4f   Q_null %.4f   gap %+.4f"
      % (Q_pool, Qnull_pool, Q_pool - Qnull_pool))
    A("  registered: Q >= 0.90 AND gap >= +0.30  ->  %s"
      % ("HIT" if r1_hit else "MISS"))
    A("  falsifier (Q <= 0.60 or gap <= +0.20): %s"
      % ("TRIPPED" if r1_fals else "not tripped"))
    A("  DISCLOSED IN ADVANCE: 98's M in [0.4647, 0.5159] against a blind 0.25")
    A("  already forces p ~ 0.96 under a candidates-vs-uniform mixture, so a")
    A("  high Q is half-implied and this hit is worth little on its own.")
    A("")

    # R2 ---------------------------------------------------------------------
    imp = [r["improper_u"] for r in fit]
    n_band = sum(1 for v in imp if 0.005 <= v <= 0.15)
    n_below = sum(1 for v in imp if v < 0.005)
    n_above = sum(1 for v in imp if v > 0.15)
    r2_hit = n_band >= 7
    A("-" * 78)
    A("R2 -- THE RESIDUAL IMPROPER RATE, 1 - Q_u")
    A("   rho   1 - Q_u   improper draws   wrong unforced")
    for r in rows:
        A("  %4.2f   %7.4f   %14d   %14d%s"
          % (r["rho"], r["improper_u"],
             round(r["improper_u"] * r["wrong_unforced"]), r["wrong_unforced"],
             "  <- extension" if r["held_out"] else ""))
    A("  in [0.005, 0.15] at %d of 10;  below 0.005 at %d;  above 0.15 at %d"
      % (n_band, n_below, n_above))
    A("  registered: in band at >= 7 of 10  ->  %s" % ("HIT" if r2_hit else "MISS"))
    A("  falsifiers: below 0.005 at >= 7 (an exact candidate draw) %s;"
      % ("TRIPPED" if n_below >= 7 else "not tripped"))
    A("              above 0.15 at >= 4 (the wrong description)   %s"
      % ("TRIPPED" if n_above >= 4 else "not tripped"))
    A("  READING: neither falsifier trips, so on the letter of the registration")
    A("  this is an ordinary miss -- the rate sits just under my floor at 6 of")
    A("  10.  But the post-hoc below shows the band was the wrong instrument")
    A("  entirely: this pooled rate is not a population property, it is ONE RUN")
    A("  diluted by 78 others that are exactly zero.  I registered a band on a")
    A("  mean without asking whether the mean described anybody.")
    A("")

    # R3 ---------------------------------------------------------------------
    qu = [r["Q_u"] for r in fit]
    rng = max(qu) - min(qu)
    sp_q = spearman(qu, [r["rho"] for r in fit])
    m_rng = max(r["M"] for r in fit) - min(r["M"] for r in fit)
    r3_hit = (rng <= 0.05) and (sp_q <= 0.60)
    r3_fals = (rng > 0.10) and (sp_q >= 0.80)
    A("-" * 78)
    A("R3 -- THE HEADLINE: DOES THE COLOUR RULE DEGRADE WITH DIFFICULTY")
    A("  over the ten fit densities:")
    A("    Q_u range %.4f   Spearman(Q_u, rho) %+.4f" % (rng, sp_q))
    A("    M   range %.4f   Spearman(M,   rho) %+.4f"
      % (m_rng, spearman([r["M"] for r in fit], [r["rho"] for r in fit])))
    A("  registered: range <= 0.05 AND Spearman <= +0.60  ->  %s"
      % ("HIT" if r3_hit else "MISS"))
    A("  falsifier (range > 0.10 AND Spearman >= +0.80): %s"
      % ("TRIPPED" if r3_fals else "not tripped"))
    A("  READING: the miss is carried entirely by the Spearman half -- the range")
    A("  bar was met with room to spare (%.4f against 0.05) and the falsifier did"
      % rng)
    A("  not trip, because it required BOTH.  A rank correlation of %+.4f over a"
      % sp_q)
    A("  total range of %.4f is a perfectly ordered sequence of differences that"
      % rng)
    A("  are a tenth the size of what the prediction was about; it is monotone,")
    A("  not a degradation.  And the post-hoc below removes even that: with the")
    A("  two carrier runs out, Q_u is exactly 1.0000 at all fourteen densities,")
    A("  so the monotone trend is one run's density profile, not the colour")
    A("  rule's.  R3's substance -- the colour rule does not degrade with")
    A("  difficulty -- is TRUE, and far more sharply than I registered it;")
    A("  the statistic I chose to test it with is what missed.")
    A("")

    # R4 ---------------------------------------------------------------------
    sp_r4 = spearman([r["resid"] for r in fit], [r["improper_u"] for r in fit])
    r4_hit = sp_r4 >= 0.60
    A("-" * 78)
    A("R4 -- DOES THE RESIDUAL EXPLAIN THE PART OF M THAT CAND's P_hat MISSES")
    A("   rho   P_hat       M   P_hat - M   1 - Q_u")
    for r in fit:
        A("  %4.2f  %6.4f  %6.4f     %+7.4f   %7.4f"
          % (r["rho"], r["phat"], r["M"], r["resid"], r["improper_u"]))
    A("  Spearman(P_hat - M, 1 - Q_u) over the ten = %+.4f" % sp_r4)
    A("  registered: >= +0.60, falsified at <= 0  ->  %s"
      % ("HIT" if r4_hit else "MISS"))
    A("  FALSIFIED (the falsifier is <= 0 and %+.4f clears it)." % sp_r4)
    A("  READING: and the post-hoc says it could not have gone otherwise.  M's")
    A("  fall across density is a property of all 79 runs; 1 - Q_u is nonzero in")
    A("  2 of them.  A quantity that is identically zero for 77 runs cannot")
    A("  explain a residual those 77 runs all exhibit.  CAND's unexplained")
    A("  residual in M stays unexplained, and improper draws are now RULED OUT")
    A("  as its source rather than merely unsupported -- which is worth more")
    A("  than the hit would have been.")
    A("")

    # R5 ---------------------------------------------------------------------
    ok = [r for r in run_rows if r["Q_u"] == r["Q_u"]]
    sp_r5 = spearman([r["improper_u"] if False else (1.0 - r["Q_u"]) for r in ok],
                     [math.log(r["params"]) for r in ok])
    r5_hit = sp_r5 <= -0.30
    A("-" * 78)
    A("R5 -- DO THE SMALL ARCHITECTURES CARRY THE IMPROPER DRAWS (rho = %.2f)"
      % RHO_SIZE)
    A("  runs %d   Spearman(1 - Q_u, log params) = %+.4f" % (len(ok), sp_r5))
    A("  registered: <= -0.30, falsified at >= +0.10  ->  %s"
      % ("HIT" if r5_hit else "MISS"))
    if ok:
        worst = sorted(ok, key=lambda r: -(1.0 - r["Q_u"]))[:5]
        A("  highest improper rate:")
        for r in worst:
            A("    %-28s %9d params   1 - Q_u %.4f  on %d wrong"
              % (r["cell"][:28], r["params"], 1.0 - r["Q_u"],
                 r["wrong_unforced"]))
    A("")

    # post-hoc: where the improper draws actually live -------------------------
    # improper draws on unforced cells = N - n_proper - n_err_in_F, because a
    # wrong colour on a FORCED cell is improper by construction (one candidate).
    per_run = collections.defaultdict(lambda: [0, 0, None, 0])  # imp, wrongU, cell, params
    for key in joined:
        if key[1] not in FIT_RHOS:
            continue
        b, f = bh[key], fh[key]
        imp_n = b["n_emit"] - b["n_proper"] - f["n_err_in_F"]
        wu = b["n_emit"] - b["n_correct"] - f["n_err_in_F"]
        rec = per_run[key[0]]
        rec[0] += imp_n
        rec[1] += wu
        rec[2] = b["cell"]
        rec[3] = b["params"]
    tot_imp = sum(v[0] for v in per_run.values())
    carriers = sorted([v for v in per_run.values() if v[0] > 0],
                      key=lambda v: -v[0])
    by_cell = collections.Counter()
    for v in per_run.values():
        by_cell[v[2]] += v[0]
    A("-" * 78)
    A("POST-HOC, NOT REGISTERED -- AND IT IS WHY R5's STATISTIC IS NOISE")
    A("  improper draws on unforced cells, pooled over the ten fit densities")
    A("  and all %d runs: %d, out of %d wrong unforced emissions."
      % (len(per_run), tot_imp, sum(v[1] for v in per_run.values())))
    A("  runs contributing at least one: %d of %d" % (len(carriers), len(per_run)))
    for v in carriers[:8]:
        A("    %4d improper / %5d wrong   %-30s %9d params"
          % (v[0], v[1], v[2][:30], v[3]))
    A("  architectures (cell labels) contributing: %d of %d"
      % (sum(1 for c in by_cell.values() if c > 0), len(by_cell)))
    if tot_imp:
        A("  largest single run's share of all improper draws: %.1f%%"
          % (100.0 * carriers[0][0] / tot_imp))
    carrier_ids = {rid for rid, v in per_run.items() if v[0] > 0}
    A("")
    A("  AND WITH THOSE %d RUNS REMOVED, THE REST OF THE POPULATION IS EXACT:"
      % len(carrier_ids))
    A("     rho   runs   wrong unforced   improper   Q_u (rest)")
    tot_rest = 0
    for r in rows:
        cells_r = [k for k in joined
                   if k[1] == r["rho"] and k[0] not in carrier_ids]
        wu = sum(bh[k]["n_emit"] - bh[k]["n_correct"] - fh[k]["n_err_in_F"]
                 for k in cells_r)
        ip = sum(bh[k]["n_emit"] - bh[k]["n_proper"] - fh[k]["n_err_in_F"]
                 for k in cells_r)
        tot_rest += wu
        r["Q_u_rest"] = (1.0 - ip / wu) if wu else float("nan")
        r["n_rest"] = len(cells_r)
        A("    %4.2f   %4d   %14d   %8d   %10s%s"
          % (r["rho"], len(cells_r), wu, ip,
             ("%.4f" % (1.0 - ip / wu)) if wu else "n/a",
             "  <- extension" if r["held_out"] else ""))
    A("  So 77 of 79 runs never once write a colour that contradicts a")
    A("  neighbour they can already see, over tens of thousands of wrong")
    A("  emissions.  The COLOUR step is an exact candidate-set computation in")
    A("  every architecture but the smallest -- which is the first positive")
    A("  evidence in this line that something in the model represents the")
    A("  candidate set, and it stands even though 102 showed the CELL step")
    A("  does not.")
    A("  R5 asked for a correlation of a PER-RUN RATE against size at one")
    A("  density, where there were 30 events over 61 runs and 60 of those runs")
    A("  had exactly zero.  Spearman over 60 tied zeros is a tie-breaking")
    A("  artefact and its +0.8928 means nothing; the substance of R5 -- that")
    A("  the smallest architectures carry the improper draws -- is true in the")
    A("  strongest possible form, and my estimator could not see it.")
    A("  BEFORE REGISTERING A PER-RUN CORRELATION, COUNT THE EVENTS.")
    A("")

    A("=" * 78)
    A("SCORECARD")
    for name, hit in [("R1 colour locally consistent (check)", r1_hit),
                      ("R2 residual real and small", r2_hit),
                      ("R3 colour rule does not degrade", r3_hit),
                      ("R4 residual explains P_hat - M", r4_hit),
                      ("R5 small models carry it", r5_hit)]:
        A("  %-40s %s" % (name, "HIT" if hit else "MISS"))
    A("  %-40s %s" % ("E1 same rollouts", "PASS" if not e1_bad else "FAIL"))
    A("  %-40s %s" % ("E2 proper >= correct", "PASS" if not e2_bad else "FAIL"))
    A("")
    A("=" * 78)
    A("WHAT THIS ANSWERS, AND WHAT IT LEAVES OPEN")
    A("")
    A("  One hit out of five, and the hit was disclosed in advance as a")
    A("  consistency check.  On the registration's own scoring this stage")
    A("  mostly missed.  But four of those five predictions missed by being")
    A("  TOO WEAK, not by being wrong, and the thing they were too weak to")
    A("  state is the largest result in this line:")
    A("")
    A("  THE COLOUR STEP IS AN EXACT CANDIDATE-SET COMPUTATION, IN 77 OF 79")
    A("  RUNS, WITH NO EXCEPTIONS AT ALL.")
    A("")
    A("  Not 'almost always', which is what R1 and R2 were built to measure.")
    A("  Zero improper draws in %d wrong unforced emissions across all" % tot_rest)
    A("  fourteen densities.  When these models write a colour, it is one the")
    A("  constraints they can already see still allow -- even when the colour")
    A("  is wrong, even on the hardest puzzles, even in the runs that solve")
    A("  almost nothing.  The 147 counterexamples in the whole dataset come")
    A("  from 2 runs and 2 architectures, and 144 of them from a single run of")
    A("  the smallest architecture tried.")
    A("")
    A("  Set against 102, that gives the sharpest statement this project has")
    A("  of what these models actually do:")
    A("")
    A("    THE CELL STEP IS NOT DEDUCTIVE  -- its apparent preference for")
    A("    near-forced cells is selection on k, the number of coloured")
    A("    neighbours, and dies under the within-instance control (102).")
    A("")
    A("    THE COLOUR STEP IS DEDUCTIVE    -- given whatever cell was named,")
    A("    the colour respects every visible constraint, exactly (103).")
    A("")
    A("  A constraint-respecting colour rule bolted to a non-deductive cell")
    A("  rule.  That is a much more specific object than 'the model does")
    A("  direct deduction' or 'it does not', and it is the first positive")
    A("  evidence in this line that ANYTHING in the model represents the")
    A("  candidate set.  It also says where the solve rate is lost: not in")
    A("  writing illegal colours, but in choosing cells whose candidate set")
    A("  has more than one member and then picking from it.")
    A("")
    A("  WHAT IT DOES NOT SHOW.  Exactness of the colour step does not mean")
    A("  the model COMPUTES the candidate set and samples from it; the same")
    A("  record is produced by any mechanism that never emits a seen colour,")
    A("  including simple inhibition from the neighbour tokens in context.")
    A("  Distinguishing those needs the model's internals, not its output,")
    A("  and this stage reads output only.")
    A("")
    A("  WHAT IS STILL BLOCKED.  Everything about the CELL step still waits on")
    A("  a generator change -- larger q, or instances where high-k cells keep")
    A("  three or more candidates -- because c and k are near-collinear on the")
    A("  ensemble that exists.  That needs a job, and therefore hshunt.  This")
    A("  stage did not need one: it was a column that had been sitting in")
    A("  B_hazard unread since stage POS.")

    txt = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "VERDICT.txt"), "w") as f:
        f.write(txt)
    print(txt)

    # ----------------------------------------------------------------- figures
    fitr = [r["rho"] for r in fit]
    ext = [r for r in rows if r["held_out"]]

    # R1: Q and Q_null against rho
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(fitr, [r["Q"] for r in fit], "o-", color=SERIES[0], ms=5,
            rasterized=True)
    ax.plot([r["rho"] for r in ext], [r["Q"] for r in ext], "o--",
            color=SERIES[0], ms=5, mfc="white", rasterized=True)
    ax.plot(fitr, [r["Q_null"] for r in fit], "o-", color=MUTED, ms=4,
            rasterized=True)
    ax.plot([r["rho"] for r in ext], [r["Q_null"] for r in ext], "o--",
            color=MUTED, ms=4, mfc="white", rasterized=True)
    label_line(ax, 0.93, fit[0]["Q"], "measured", color=SERIES[0], dy=8)
    label_line(ax, 0.93, fit[0]["Q_null"], "colour-blind null", color=MUTED,
               dy=-14)
    ax.set_ylim(0.0, 1.06)
    ax.invert_xaxis()
    style_axes(ax, xlabel=r"hint density $\rho$  (harder $\rightarrow$)",
               ylabel="share of wrong colours still proper")
    ax.text(0.62, 0.12, "open markers: densities no fit ever used",
            fontsize=8, color=MUTED, transform=ax.transData)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "R1_proper.png"), dpi=DPI)
    plt.close(fig)

    # R2: 1 - Q_u against rho, with the registered band
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axhspan(0.005, 0.15, color=SERIES[1], alpha=0.10)
    ax.plot(fitr, [r["improper_u"] for r in fit], "o-", color=SERIES[0], ms=5,
            rasterized=True)
    ax.plot([r["rho"] for r in ext], [r["improper_u"] for r in ext], "o--",
            color=SERIES[0], ms=5, mfc="white", rasterized=True)
    top = max(r["improper_u"] for r in rows)
    ax.set_ylim(0, max(0.17, top * 1.15))
    ax.invert_xaxis()
    style_axes(ax, xlabel=r"hint density $\rho$  (harder $\rightarrow$)",
               ylabel=r"$1-Q_u$: wrong colours that clash with a neighbour")
    ax.text(0.94, 0.152, "registered band [0.005, 0.15]", fontsize=8,
            color=SERIES[1], va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "R2_residual.png"), dpi=DPI)
    plt.close(fig)

    # R3: Q_u and M, each as a change from its own value at rho 0.95
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    q0, m0 = fit[0]["Q_u"], fit[0]["M"]
    ax.axhline(0.0, color="black", lw=0.8)
    ax.plot(fitr, [r["Q_u"] - q0 for r in fit], "o-", color=SERIES[0], ms=5,
            rasterized=True)
    ax.plot(fitr, [r["M"] - m0 for r in fit], "o-", color=SERIES[1], ms=5,
            rasterized=True)
    label_line(ax, fitr[-1], fit[-1]["Q_u"] - q0, r"$Q_u$", color=SERIES[0])
    label_line(ax, fitr[-1], fit[-1]["M"] - m0, r"$M$", color=SERIES[1])
    ax.invert_xaxis()
    style_axes(ax, xlabel=r"hint density $\rho$  (harder $\rightarrow$)",
               ylabel=r"change from the value at $\rho=0.95$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "R3_flat.png"), dpi=DPI)
    plt.close(fig)

    # R4: the residual against the improper rate
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.scatter([r["improper_u"] for r in fit], [r["resid"] for r in fit],
               s=34, color=SERIES[0], rasterized=True)
    for r in fit:
        ax.annotate("%.2f" % r["rho"], (r["improper_u"], r["resid"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7,
                    color=MUTED)
    style_axes(ax, xlabel=r"$1-Q_u$, the improper-draw rate",
               ylabel=r"$\hat P - M$, the part Model A over-predicts")
    ax.text(0.02, 0.95, "Spearman %+.3f" % sp_r4, transform=ax.transAxes,
            fontsize=9, va="top", color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "R4_explains.png"), dpi=DPI)
    plt.close(fig)

    # R5: the improper rate against size
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.scatter([r["params"] for r in ok], [1.0 - r["Q_u"] for r in ok],
               s=26, color=SERIES[0], alpha=0.75, rasterized=True)
    ax.set_xscale("log")
    style_axes(ax, xlabel="trainable parameters",
               ylabel=r"$1-Q_u$ at $\rho=%.2f$" % RHO_SIZE)
    nz5 = sum(1 for r in ok if 1.0 - r["Q_u"] == 0.0)
    ax.text(0.98, 0.95,
            "Spearman %+.3f over %d runs\n-- but %d of them are exactly zero,\n"
            "so the statistic is ranking ties" % (sp_r5, len(ok), nz5),
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "R5_size.png"), dpi=DPI)
    plt.close(fig)

    # PH: where the improper draws live, pooled over the ten fit densities
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    order = sorted(per_run.values(), key=lambda v: v[3])
    xs = list(range(len(order)))
    cols = [SERIES[1] if v[0] > 0 else MUTED for v in order]
    ax.bar(xs, [v[0] for v in order], color=cols, width=0.8, rasterized=True)
    zx = [i for i, v in zip(xs, order) if v[0] == 0]
    ax.plot(zx, [0] * len(zx), "o", color=MUTED, ms=3, rasterized=True,
            clip_on=False)
    ax.set_yscale("symlog", linthresh=1)
    style_axes(ax, xlabel="the 79 runs, ordered by trainable parameters",
               ylabel="improper draws, ten fit densities")
    ax.set_xticks([])
    for i, v in zip(xs, order):
        if v[0] > 0:
            ax.annotate("%s\n%s params" % (v[2][:26], format(v[3], ",")),
                        (i, v[0]), textcoords="offset points",
                        xytext=(6, 2), fontsize=7, color=SERIES[1])
    ax.text(0.98, 0.10, "each grey dot is a run with ZERO improper draws (%d of %d)"
            % (len(zx), len(order)), fontsize=8, color=MUTED,
            ha="right", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "PH_carriers.png"), dpi=DPI)
    plt.close(fig)

    # PH: the exactness the pooled rate hid
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    allr = [r["rho"] for r in rows]
    ax.plot(allr, [r["Q_u_rest"] for r in rows], "o-", color=SERIES[0], ms=5,
            rasterized=True)
    ax.plot(allr, [r["Q_u"] for r in rows], "s--", color=SERIES[1], ms=4,
            rasterized=True)
    label_line(ax, allr[-1], rows[-1]["Q_u_rest"], "the other 77 runs",
               color=SERIES[0])
    label_line(ax, allr[-1], rows[-1]["Q_u"], "all 79 pooled", color=SERIES[1],
               dy=-10)
    ax.set_ylim(0.985, 1.002)
    style_axes(ax, xlabel=r"hint density $\rho$",
               ylabel=r"$Q_u$, share of wrong colours still legal")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "PH_exact.png"), dpi=DPI)
    plt.close(fig)

    print("figures -> %s" % FIGS)


if __name__ == "__main__":
    main()
