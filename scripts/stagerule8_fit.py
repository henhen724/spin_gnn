#!/usr/bin/env python3
"""Stage RULE8 -- the INTERVAL on h_rule for the governing cell, not its value.

Eighth refit of stage LOGIC's hazard shards. Nothing is trained and nothing is
submitted. Predictions E1-E5 and diagnostics D1-D5 are registered in
runs/stagerule8_2026-11-24/REGISTERED.md and were written before this ran.

THE OBJECT. The 2026-11-23 brief selects `L=16 h=4 tied_no_mlp d320` -- one
iterated operator h_{t+1} = F(h_t), 613,818 parameters, 3 seeds -- and its metric
is

    h_rule = sum_j n_err_in_F / sum_j n_in_F = Pr(colour wrong | named a forced cell)

pooled over all 14 densities and all j, which is stage RULE2's eval. On that eval
it reads 2 rule errors in 80,956 forced emissions.

THE POINT ESTIMATE CANNOT MOVE. Same eval, same shards, same two counts; E5
requires exactly that. What is new is a DELETE-ONE-RUN jackknife over the cell's
3 seeds, each run dropped whole, and the question is whether resampling the seed
tightens the interval or widens it.

Nothing here is a density threshold and h_rule is never converted into one.
"""

import collections
import math
import os
import sys

from scipy.stats import beta, chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stageexp_fit import wilson                                  # noqa: E402
from stagegate_fit import GRID, FIT_RHOS, J2_LABEL               # noqa: E402
from stagerule_fit import RUNS, line, load, per_cell             # noqa: E402
from stagerule3_fit import sequences                             # noqa: E402

OUT = os.path.join(RUNS, "stagerule8_2026-11-24")

# The cell the brief selects. Resolved by LABEL: the triple
# ('tied_no_mlp', 16, 320) covers THREE cells on this roster -- d320, d320 k2 and
# d320 k4 (RULE7 D5) -- so keying on the triple would pool nine runs into a fake
# three-seed cell. D1 checks this.
CELL = "L=16 h=4 tied_no_mlp d320"
SIBLINGS = ("L=16 h=4 tied_no_mlp d320 k2", "L=16 h=4 tied_no_mlp d320 k4")

# Published figures this stage must reproduce (STATUS finding 122).
PUB_ERR = 2
PUB_F14 = 80956
PUB_F10 = 63213

Z = 1.96                      # the project's existing normal multiplier
T_DF2 = 4.302652729911275     # Student t, 95% two-sided, df = n - 1 = 2

# Registered bars.
E1_BAR = 1.0        # jackknife width / Wilson width: >= 1 means it WIDENED
E4_LO, E4_HI = 0.5, 2.0
TOL = 1e-12


# --------------------------------------------------------------- resampling

def jackknife(per_unit):
    """Delete-one-unit jackknife on the ratio h = sum(E) / sum(F).

    per_unit maps a unit key -> (E, F). The unit is dropped WHOLE, so anything
    shared inside it -- for a run: the seed, the checkpoint, the data order --
    sits inside the resampled unit instead of being assumed away.

    Returns the plug-in estimate, the replicates, the jackknife standard error
    and the bias-corrected estimate. The interval is centred on the PLUG-IN
    estimate, matching stagerule4_fit.ratio_jackknife, so the two stages'
    intervals mean the same thing; the bias-corrected value is reported beside
    it rather than substituted for it.
    """
    keys = sorted(per_unit)
    E = sum(per_unit[k][0] for k in keys)
    F = sum(per_unit[k][1] for k in keys)
    full = E / F if F > 0 else float("nan")
    reps = []
    for k in keys:
        e, f = per_unit[k]
        if F - f > 0:
            reps.append((k, (E - e) / (F - f)))
    n = len(reps)
    if n < 2:
        return full, reps, float("nan"), float("nan"), n
    mean = sum(v for _, v in reps) / n
    se = math.sqrt((n - 1.0) / n * sum((v - mean) ** 2 for _, v in reps))
    bc = n * full - (n - 1.0) * mean
    return full, reps, se, bc, n


def poisson_exact(k, F):
    """Exact Poisson interval on a rate k/F, and the one-sided 95% upper bound.

    A two-sided interval on 2 events is wide whatever the method; the one-sided
    upper bound is the form this line actually uses, because the claim being
    made is 'h_rule stays below the 0.01 bar', not 'h_rule equals 0.00002'.
    """
    lo = chi2.ppf(0.025, 2 * k) / 2.0 / F if k > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * (k + 1)) / 2.0 / F
    up1 = chi2.ppf(0.95, 2 * (k + 1)) / 2.0 / F
    return lo, hi, up1


def clopper_pearson(k, n):
    """Exact binomial interval, as the thing Wilson approximates."""
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


# -------------------------------------------------------------- diagnostics

def d1(fh, arch, cells):
    """The cell is resolved by its LABEL, never by (mode, L, d)."""
    a = next((x for x in arch if x["cell"] == CELL), None)
    runs = sorted({c["run_id"] for c in cells if c["cell"] == CELL})
    sib = sorted({c["run_id"] for c in cells if c["cell"] in SIBLINGS})
    triple = sorted({c["cell"] for c in cells
                     if (c["mode"], c["L"], c["d"]) == ("tied_no_mlp", 16, 320)})
    ok = a is not None and len(runs) == 3 and not (set(runs) & set(sib))
    line(fh, "D1 %s: cell %r resolved by label -> %d run(s) %s"
         % ("PASS" if ok else "FAIL", CELL, len(runs), runs))
    line(fh, "    the triple ('tied_no_mlp', 16, 320) covers %d cells: %s"
         % (len(triple), ", ".join(triple)))
    line(fh, "    sibling runs excluded: %d %s" % (len(sib), sib))
    return ok, a, runs


def d2(fh, cells, runs, a):
    """Per-seed counts sum to the pooled counts."""
    E = sum(c["err_in_F"] for c in cells
            if c["cell"] == CELL and c["rho"] in set(round(r, 4) for r in GRID))
    F = sum(c["in_F"] for c in cells
            if c["cell"] == CELL and c["rho"] in set(round(r, 4) for r in GRID))
    # a["in_F"] is per_cell's own pooling over the window it was given.
    dE, dF = abs(E - a["err_in_F"]), abs(F - a["in_F"])
    ok = dE < 1e-9 and dF < 1e-9
    line(fh, "D2 %s: per-(run,rho) sums vs per_cell: err %d vs %d (resid %.1e), "
             "forced %d vs %d (resid %.1e)"
         % ("PASS" if ok else "FAIL", int(E), int(a["err_in_F"]), dE,
            int(F), int(a["in_F"]), dF))
    return ok


def d3(fh, cells, runs):
    """No (seed, density) pair has an empty forced-emission denominator."""
    grid = set(round(r, 4) for r in GRID)
    seen = {(c["run_id"], c["rho"]): c["in_F"] for c in cells
            if c["cell"] == CELL and c["rho"] in grid}
    empty = [k for k, v in seen.items() if not v > 0]
    missing = [(r, g) for r in runs for g in sorted(grid)
               if (r, g) not in seen]
    ok = not empty and not missing and len(seen) == len(runs) * len(grid)
    line(fh, "D3 %s: %d of %d (seed, density) pairs present, %d with an empty "
             "denominator, %d missing"
         % ("PASS" if ok else "FAIL", len(seen), len(runs) * len(grid),
            len(empty), len(missing)))
    return ok


def d4(fh, seq, runs):
    """n_err_in_F <= n_err on every row of this cell (finding 124)."""
    bad = tot = 0
    for (run, _rho), byj in seq.items():
        if run not in set(runs):
            continue
        for r in byj.values():
            tot += 1
            if r["err_in_F"] > r["err"]:
                bad += 1
    ok = bad == 0 and tot > 0
    line(fh, "D4 %s: n_err_in_F > n_err on %d of %d rows of this cell"
         % ("PASS" if ok else "FAIL", bad, tot))
    return ok


def d5(fh, cells):
    """The pooling window is the one being claimed, and both windows reproduce."""
    a14 = next(x for x in per_cell(cells, rhos=GRID) if x["cell"] == CELL)
    a10 = next(x for x in per_cell(cells, rhos=FIT_RHOS) if x["cell"] == CELL)
    ok = (len(GRID) == 14 and len(FIT_RHOS) == 10
          and int(a14["in_F"]) == PUB_F14 and int(a10["in_F"]) == PUB_F10
          and int(a14["err_in_F"]) == PUB_ERR
          and int(a10["err_in_F"]) == PUB_ERR)
    line(fh, "D5 %s: GRID %d densities, FIT_RHOS %d; forced %d (pub %d) over 14 "
             "and %d (pub %d) over 10; errors %d and %d (pub %d)"
         % ("PASS" if ok else "FAIL", len(GRID), len(FIT_RHOS),
            int(a14["in_F"]), PUB_F14, int(a10["in_F"]), PUB_F10,
            int(a14["err_in_F"]), int(a10["err_in_F"]), PUB_ERR))
    return ok, a14, a10


# ------------------------------------------------------------------ scoring

def intervals(a):
    """Everything the interval question needs, on the published counts."""
    k, F = int(a["err_in_F"]), int(a["in_F"])
    h = k / F
    w_lo, w_hi = wilson(k, F)
    cp_lo, cp_hi = clopper_pearson(k, F)
    p_lo, p_hi, p_up1 = poisson_exact(k, F)
    se_pois = math.sqrt(k) / F
    return {"k": k, "F": F, "h": h, "w": (w_lo, w_hi), "cp": (cp_lo, cp_hi),
            "pois": (p_lo, p_hi), "up1": p_up1, "se_pois": se_pois}


def per_run_counts(cells, runs, rhos):
    win = set(round(r, 4) for r in rhos)
    out = collections.defaultdict(lambda: [0.0, 0.0])
    for c in cells:
        if c["cell"] == CELL and c["rho"] in win:
            out[c["run_id"]][0] += c["err_in_F"]
            out[c["run_id"]][1] += c["in_F"]
    return {k: tuple(v) for k, v in out.items()}


def score(fh, cells, runs, a14, a10, j2):
    iv = intervals(a14)
    per_run = per_run_counts(cells, runs, GRID)
    full, reps, se, bc, n = jackknife(per_run)

    line(fh, "=" * 74)
    line(fh, "THE NUMBER AND ITS COUNTS -- the same eval before and after")
    line(fh, "=" * 74)
    line(fh, "cell            %s" % CELL)
    line(fh, "params          613818   seeds %d   runs %s" % (n, sorted(per_run)))
    line(fh, "eval            all 14 densities, all j, 400 instances per bin")
    line(fh, "h_rule          %.6e  = %d rule errors / %d forced emissions"
         % (iv["h"], iv["k"], iv["F"]))
    line(fh, "                rounded as this line quotes it: %.5f" % iv["h"])
    line(fh, "per seed        " + "   ".join(
        "%s %d/%d" % (r, int(per_run[r][0]), int(per_run[r][1]))
        for r in sorted(per_run)))
    line(fh)
    line(fh, "BEFORE -- Wilson 95% on the same counts")
    line(fh, "   h_rule %.6e  [%.6e, %.6e]  width %.6e"
         % (iv["h"], iv["w"][0], iv["w"][1], iv["w"][1] - iv["w"][0]))
    line(fh, "AFTER  -- delete-one-run jackknife over the %d seeds, Z = %.2f"
         % (n, Z))
    jk_lo, jk_hi = full - Z * se, full + Z * se
    line(fh, "   h_rule %.6e  [%.6e, %.6e]  width %.6e"
         % (full, jk_lo, jk_hi, jk_hi - jk_lo))
    line(fh, "   jackknife SE %.6e, bias-corrected estimate %.6e" % (se, bc))
    line(fh, "   replicates (run dropped -> h_rule on the other two):")
    for k_, v in reps:
        line(fh, "     drop %-6s -> %.6e   (%d/%d removed)"
             % (k_, v, int(per_run[k_][0]), int(per_run[k_][1])))
    line(fh, "   with the honest multiplier for %d replicates, t(df=%d) = %.3f:"
         % (n, n - 1, T_DF2))
    line(fh, "     [%.6e, %.6e]  width %.6e"
         % (full - T_DF2 * se, full + T_DF2 * se, 2 * T_DF2 * se))
    line(fh)

    line(fh, "=" * 74)
    line(fh, "SCORED PREDICTIONS")
    line(fh, "=" * 74)

    w_width = iv["w"][1] - iv["w"][0]
    j_width = jk_hi - jk_lo
    ratio = j_width / w_width if w_width > 0 else float("nan")
    e1 = ratio >= E1_BAR
    line(fh, "E1 %s: jackknife width / Wilson width = %.4f (bar >= %.1f, i.e. "
             "HIT means it WIDENED)" % ("HIT" if e1 else "MISS", ratio, E1_BAR))
    line(fh, "   Wilson    [%.3e, %.3e]  width %.3e"
         % (iv["w"][0], iv["w"][1], w_width))
    line(fh, "   jackknife [%.3e, %.3e]  width %.3e"
         % (jk_lo, jk_hi, j_width))
    line(fh, "   on t(df=2) instead of Z the jackknife width is %.3e, a factor "
             "of %.2f on Wilson" % (2 * T_DF2 * se, 2 * T_DF2 * se / w_width))

    e2 = jk_lo <= 0 < iv["w"][0]
    line(fh, "E2 %s: jackknife lower bound %+.3e, Wilson's %+.3e -- the bar "
             "wanted the first <= 0 and the second > 0"
         % ("HIT" if e2 else "MISS", jk_lo, iv["w"][0]))
    line(fh, "   on t(df=%d) the jackknife lower bound is %+.3e instead, so "
             "this prediction turns entirely on the multiplier too"
         % (n - 1, full - T_DF2 * se))
    line(fh, "   the lower bound is the load-bearing one here: it is what "
             "separates 'keeps DD's rule' from 'makes no rule errors at all'.")

    zero = [r for r in sorted(per_run) if per_run[r][0] == 0]
    e3 = len(zero) >= 1
    line(fh, "E3 %s: %d of %d seeds carry no rule error at all (%s)"
         % ("HIT" if e3 else "MISS", len(zero), n,
            ", ".join(zero) if zero else "none"))
    carriers = [(r, int(per_run[r][0])) for r in sorted(per_run)
                if per_run[r][0] > 0]
    line(fh, "   carriers: %s"
         % ", ".join("%s %d error(s)" % c for c in carriers))

    r_se = se / iv["se_pois"] if iv["se_pois"] > 0 else float("nan")
    e4 = E4_LO <= r_se <= E4_HI
    line(fh, "E4 %s: SE_jack / SE_Poisson = %.4f (bar %.1f to %.1f)"
         % ("HIT" if e4 else "MISS", r_se, E4_LO, E4_HI))
    line(fh, "   SE_jack %.3e over %d runs, SE_Poisson sqrt(%d)/%d = %.3e"
         % (se, n, iv["k"], iv["F"], iv["se_pois"]))
    line(fh, "   a ratio near 1 means the 3 seeds add NO dispersion beyond "
             "counting noise: this is a COUNT problem, not a SEED problem.")

    e5 = (iv["k"] == PUB_ERR and iv["F"] == PUB_F14
          and int(a10["in_F"]) == PUB_F10 and int(a10["err_in_F"]) == PUB_ERR
          and abs(iv["h"] - iv["k"] / iv["F"]) < TOL)
    line(fh, "E5 %s: reproduces finding 122 -- %d errors, %d forced over 14 "
             "(pub %d), %d forced over 10 (pub %d), residual %.1e"
         % ("HIT" if e5 else "MISS", iv["k"], iv["F"], PUB_F14,
            int(a10["in_F"]), PUB_F10, abs(iv["h"] - iv["k"] / iv["F"])))
    line(fh, "   ten-density h_rule %.6e, fourteen-density %.6e -- the SAME "
             "two errors, a bigger denominator."
         % (a10["err_in_F"] / a10["in_F"], iv["h"]))
    line(fh)
    return iv, (full, reps, se, bc, n, jk_lo, jk_hi), [e1, e2, e3, e4, e5]


# ----------------------------------------------------------------- post-hoc

def posthoc(fh, cells, runs, seq, iv, jk, j2, a14):
    full, reps, se, bc, n, jk_lo, jk_hi = jk
    line(fh, "=" * 74)
    line(fh, "POST-HOC -- read after the scored predictions, never a test")
    line(fh, "=" * 74)

    # (a) the other resampling unit
    grid = set(round(r, 4) for r in GRID)
    per_rho = collections.defaultdict(lambda: [0.0, 0.0])
    for c in cells:
        if c["cell"] == CELL and c["rho"] in grid:
            per_rho[c["rho"]][0] += c["err_in_F"]
            per_rho[c["rho"]][1] += c["in_F"]
    f2, r2, se2, bc2, n2 = jackknife({k: tuple(v) for k, v in per_rho.items()})
    line(fh, "(a) the OTHER resampling unit: delete-one-DENSITY, %d replicates" % n2)
    line(fh, "    SE %.3e, interval [%.3e, %.3e] on Z, width %.3e"
         % (se2, f2 - Z * se2, f2 + Z * se2, 2 * Z * se2))
    line(fh, "    %d replicates instead of %d, and the unit is a BIN not a run, "
             "so this answers a different question -- whether one density "
             "carries the rate. It is reported because 3 replicates is the "
             "whole difficulty with the registered procedure." % (n2, n))

    # (b) where the errors actually sit
    line(fh, "(b) where the %d rule errors sit, by (seed, density, j)" % iv["k"])
    hits = []
    for (run, rho), byj in seq.items():
        if run not in set(runs):
            continue
        for j, r in sorted(byj.items()):
            if r["err_in_F"] > 0:
                hits.append((run, rho, j, int(r["err_in_F"]),
                             int(r["in_F"]), int(r["at_risk"])))
    for run, rho, j, e, inf, ar in sorted(hits):
        line(fh, "    %-6s rho %.2f  j %-3d  %d error(s)  (n_in_F %d, "
                 "n_at_risk %d)" % (run, rho, j, e, inf, ar))
    if not hits:
        line(fh, "    none -- which would contradict E5 and is reported as such")

    # (c) the bound the line actually uses
    line(fh, "(c) the ONE-SIDED bound, which is the form the claim takes")
    line(fh, "    h_rule < %.3e at 95%% one-sided (exact Poisson on %d events "
             "in %d forced emissions)" % (iv["up1"], iv["k"], iv["F"]))
    line(fh, "    that is %.0fx under the 0.01 'has DD's rule' bar."
         % (0.01 / iv["up1"]))
    line(fh, "    two-sided, for comparison: Wilson [%.3e, %.3e], "
             "Clopper-Pearson [%.3e, %.3e], exact Poisson [%.3e, %.3e]"
         % (iv["w"][0], iv["w"][1], iv["cp"][0], iv["cp"][1],
            iv["pois"][0], iv["pois"][1]))

    # (d) what tightening would actually cost
    line(fh, "(d) what it would take to tighten it, since resampling cannot")
    # The 2 errors are already observed and cannot be un-observed by adding
    # seeds, so the numerator of the one-sided bound stays at k = 2 -- chi2(0.95,
    # 2(k+1))/2 = 6.2958 -- and only F grows. Using the k = 0 constant 2.9957
    # instead would describe a cell that never erred at all, which this is not,
    # and would understate the cost by 2.1x.
    c2 = chi2.ppf(0.95, 2 * (iv["k"] + 1)) / 2.0
    need = c2 / 1e-5
    per_seed = iv["F"] / len(runs)
    line(fh, "    the %d errors already observed stay in the numerator, so with "
             "ZERO further errors the" % iv["k"])
    line(fh, "    one-sided bound is %.4f/F and reaching 1e-05 needs F >= %.0f "
             "forced emissions" % (c2, need))
    line(fh, "    this cell has %d over %d seeds = %.0f per seed, so that is "
             "~%.1f seeds at zero further errors -- i.e. %d more TRAINING runs."
         % (iv["F"], len(runs), per_seed, need / per_seed,
            math.ceil(need / per_seed) - len(runs)))
    line(fh, "    (the k = 0 constant 2.9957/F would need only %.0f, but it "
             "describes a cell that never"
         % (chi2.ppf(0.95, 2) / 2.0 / 1e-5))
    line(fh, "    erred at all; these two errors cannot be un-observed by "
             "adding seeds.)")
    line(fh, "    the eval is already exhausted: all 14 densities, all j, all "
             "400 instances per bin are in the denominator, so no further "
             "rollouts exist to add without generating them.")

    # (e) the comparison the deck has to end on
    line(fh, "(e) against the standing reference J2 on the same eval")
    jk2, jF2 = int(j2["err_in_F"]), int(j2["in_F"])
    jlo, jhi = wilson(jk2, jF2)
    _, _, jup = poisson_exact(jk2, jF2)
    line(fh, "    %-30s %d/%d  h_rule %.3e  Wilson [%.3e, %.3e]  1-sided < %.3e"
         % (J2_LABEL, jk2, jF2, jk2 / jF2, jlo, jhi, jup))
    line(fh, "    %-30s %d/%d  h_rule %.3e  Wilson [%.3e, %.3e]  1-sided < %.3e"
         % (CELL, iv["k"], iv["F"], iv["h"], iv["w"][0], iv["w"][1], iv["up1"]))
    line(fh, "    J2 is %.2fx the parameters (2504890 vs 613818) and its "
             "interval OVERLAPS this cell's, so neither is resolved from the "
             "other on the rule arm." % (2504890 / 613818.0))
    line(fh)
    # the READING quotes (a) and (b), so they are handed back rather than retyped
    return hits, se2, n2


def reading(fh, iv, jk, res, per_run, hits, se_rho, n_rho):
    full, reps, se, bc, n, jk_lo, jk_hi = jk
    e1, e2, e3, e4, e5 = res
    tag = lambda b: "HIT" if b else "MISS"
    w_width = iv["w"][1] - iv["w"][0]
    t_lo, t_hi = full - T_DF2 * se, full + T_DF2 * se
    t_width = t_hi - t_lo
    line(fh, "=" * 74)
    line(fh, "READING")
    line(fh, "=" * 74)
    line(fh, "SCORE: E1 %s E2 %s E3 %s E4 %s E5 %s"
         % (tag(e1), tag(e2), tag(e3), tag(e4), tag(e5)))
    line(fh)
    line(fh, "1. E1 AND E2 BOTH MISSED, AND BOTH TURN ON THE MULTIPLIER RATHER "
             "THAN ON THE DATA.")
    line(fh, "   I registered that the jackknife would WIDEN this interval. On "
             "the project's")
    line(fh, "   existing Z = %.2f convention it did the opposite: width %.3e "
             "against Wilson's" % (Z, jk_hi - jk_lo))
    line(fh, "   %.3e, a factor of %.2f, so the interval is %.2fx NARROWER and "
             "E1 is a MISS on its"
         % (w_width, (jk_hi - jk_lo) / w_width, w_width / (jk_hi - jk_lo)))
    line(fh, "   own stated bar. E2 misses with it: the Z interval's lower "
             "bound is %+.3e," % jk_lo)
    line(fh, "   above zero, where I predicted it would fall below.")
    line(fh)
    line(fh, "2. BUT THE NARROWING IS THE MULTIPLIER, NOT INFORMATION, AND THE "
             "CORRECTED VERSION IS")
    line(fh, "   WIDER THAN WHAT IT REPLACED. %d replicates carry %d degrees of "
             "freedom, so the" % (n, n - 1))
    line(fh, "   reference distribution is t(df=%d) = %.3f, not the normal. The "
             "same standard error"
         % (n - 1, T_DF2))
    line(fh, "   read against t gives [%+.3e, %+.3e], width %.3e -- %.2fx "
             "WIDER than Wilson, and a"
         % (t_lo, t_hi, t_width, t_width / w_width))
    line(fh, "   lower bound below zero. Both registered statements hold on the "
             "honest multiplier")
    line(fh, "   and fail on the registered one; the bars were adjudicated as "
             "registered and the")
    line(fh, "   t version is reported beside them, not substituted for them. "
             "The transferable")
    line(fh, "   lesson is to REGISTER THE MULTIPLIER WITH THE RESAMPLING UNIT: "
             "a Z interval on a")
    line(fh, "   3-replicate jackknife reports a tightening that the degrees of "
             "freedom do not support.")
    line(fh)
    line(fh, "3. THE POINT ESTIMATE DID NOT MOVE AND COULD NOT. %.6e = %d rule "
             "errors / %d forced" % (iv["h"], iv["k"], iv["F"]))
    line(fh, "   emissions, before and after, on the same eval -- 0.00002 as "
             "this line quotes it.")
    line(fh, "   NO IMPROVEMENT IS CLAIMED and none was available without "
             "generating rollouts.")
    line(fh)
    line(fh, "4. E3 HIT, AND IT IS WHY THE JACKKNIFE SE CAME OUT SMALL: the two "
             "errors sit ONE EACH")
    line(fh, "   in two different seeds, %s, with %s clean."
         % (", ".join("%s at rho %.2f j %d" % (r, rho, j)
                      for r, rho, j, _e, _f, _a in sorted(hits)),
            ", ".join(r for r in sorted(per_run) if per_run[r][0] == 0)))
    line(fh, "   That is the most EVEN arrangement %d events over %d seeds can "
             "take, so no replicate" % (iv["k"], n))
    line(fh, "   ever deletes both errors and the replicate spread (%s)"
         % " / ".join("%.3e" % v for _, v in reps))
    line(fh, "   is as small as it can be. Had both landed in one seed the "
             "replicates would have run")
    line(fh, "   from 0 to about %.3e and the SE would have been roughly twice "
             "this. THE JACKKNIFE SE"
         % (iv["k"] / (iv["F"] - iv["F"] / n)))
    line(fh, "   HERE IS A DRAW FROM A 3-POINT DISTRIBUTION, not a stable "
             "quantity.")
    line(fh)
    line(fh, "5. E4 HIT at %.3f, and it is BELOW 1: the seeds look LESS "
             "dispersed than Poisson" % (se / iv["se_pois"]))
    line(fh, "   counting noise on the same %d events (SE_jack %.3e vs "
             "SE_Poisson %.3e), which is" % (iv["k"], se, iv["se_pois"]))
    line(fh, "   what an even split produces. Seed is NOT an extra variance "
             "component on this cell,")
    line(fh, "   so the interval is a COUNT problem. On a near-zero rate, seeds "
             "buy an interval")
    line(fh, "   through the DENOMINATOR they add and never through their "
             "spread.")
    line(fh)
    line(fh, "6. THE FORM THE CLAIM SHOULD TAKE IS ONE-SIDED: h_rule < %.3e at "
             "95%%, which is" % iv["up1"])
    line(fh, "   %.0fx under the 0.01 bar. A two-sided interval on %d events "
             "spends half its coverage" % (0.01 / iv["up1"], iv["k"]))
    line(fh, "   on a lower bound no claim on this line uses. If a resampling "
             "interval is wanted on")
    line(fh, "   this cell, the DENSITY is the only unit with enough replicates "
             "to support one (%d," % n_rho)
    line(fh, "   SE %.3e, within %.0f%% of Poisson) -- but it resamples bins, "
             "so it answers a"
         % (se_rho, 100 * abs(se_rho / iv["se_pois"] - 1)))
    line(fh, "   different question than the seed does.")
    line(fh)
    line(fh, "LIMITS. Everything is this one cell on stage RULE2's eval. h_rule "
             "conditions on naming")
    line(fh, "a forced cell, a density-dependent share (finding 126), so it is "
             "not a density trend and")
    line(fh, "is never read as one. The jackknife resamples RUNS, so it carries "
             "no statement about")
    line(fh, "instances -- all 3 seeds see the same 400 per bin and the shards "
             "have no per-instance")
    line(fh, "breakdown. A zero on a seed is a bound, never a demonstration of "
             "perfection.")


def main():
    os.makedirs(OUT, exist_ok=True)
    cells, viol, n_haz, n_summ = load()
    seq, _, _ = sequences()
    arch = per_cell(cells, rhos=GRID)
    j2 = next(x for x in arch if x["cell"] == J2_LABEL)

    path = os.path.join(OUT, "VERDICT.txt")
    with open(path, "w") as fh:
        line(fh, "STAGE RULE8 -- THE INTERVAL ON h_rule FOR THE GOVERNING CELL")
        line(fh, "refit of runs/stagelogic_2026-11-17, %d hazard rows, %d "
                 "summary rows" % (n_haz, n_summ))
        line(fh, "registered in %s/REGISTERED.md before this scorer existed" % OUT)
        line(fh, "nothing trained, nothing submitted, no closed knob reopened")
        line(fh)
        line(fh, "=" * 74)
        line(fh, "DIAGNOSTICS -- any failure stops the stage")
        line(fh, "=" * 74)
        ok1, a, runs = d1(fh, arch, cells)
        if not ok1:
            line(fh, "D1 FAILED -- the stage stops here, as registered.")
            print("\nwrote", path)
            return 1
        ok2 = d2(fh, cells, runs, a)
        ok3 = d3(fh, cells, runs)
        ok4 = d4(fh, seq, runs)
        ok5, a14, a10 = d5(fh, cells)
        oks = [ok1, ok2, ok3, ok4, ok5]
        line(fh)
        line(fh, "diagnostics: %d of %d PASS" % (sum(oks), len(oks)))
        line(fh)
        if not all(oks):
            line(fh, "DIAGNOSTIC FAILURE -- no prediction is scored.")
            print("\nwrote", path)
            return 1
        iv, jk, res = score(fh, cells, runs, a14, a10, j2)
        hits, se_rho, n_rho = posthoc(fh, cells, runs, seq, iv, jk, j2, a14)
        reading(fh, iv, jk, res, per_run_counts(cells, runs, GRID),
                hits, se_rho, n_rho)
    print("\nwrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
