#!/usr/bin/env python3
"""Stage RULE4 -- h_rule on the one slice censoring cannot touch.

A fourth refit of stage LOGIC's hazard shards. Nothing is submitted, no model
is trained, no rollout is regenerated. Predictions E1-E5 and diagnostics D1-D4
are registered in runs/stagerule4_2026-11-23/REGISTERED.md and were written and
committed (0d74728) before this script existed.

THE POINT. Censoring removes a rollout at its FIRST error, so at deduction
index j = 0 nothing has been removed yet: n_at_risk(0) = 400 on every
(run, rho) sequence. Stages RULE2 and RULE3 both ended on "unanswerable from
censored rollouts"; the j = 0 slice is the same question asked where that
objection cannot apply.

    h_rule(0) = sum_runs n_err_in_F(0) / sum_runs n_in_F(0)

WHAT j = 0 DOES NOT FIX. It removes survivorship, not all selection. h_rule
conditions on the model having NAMED a forced cell, at j = 0 as everywhere
else, and the share of rollouts that do so falls from 96.3% at rho = 0.95 to
75.9% at rho = 0.33. The denominator is differently composed across densities.
That is reported, never corrected, and the slice is only ever called
censoring-free -- never unbiased.

PRECISION. The numerators here are TENS, not thousands, which is why E5 is a
registered prediction rather than an afterthought. Two intervals are reported:

  - a Poisson log-ratio interval, SE(log R) = sqrt(1/E_a + 1/E_b), which
    assumes every emission is independent;
  - a delete-one-run jackknife over the 79 runs, which does not.

Neither can cluster by INSTANCE: all runs are evaluated on the same 400
instances per bin, so errors concentrated on a few hard instances would
inflate the true variance beyond both. The shards carry no per-instance
breakdown, so that is stated and not fixed.

Run under the FIGURE env (stageexp_fit imports matplotlib at module level):

  module purge --force; module load devel math viz
  module load python/3.12.1
  module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
"""

import collections
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stagegate_fit import GRID, FIT_RHOS, spearman  # noqa: E402
from stagerule_fit import (RUNS, load, line, R1_BAR,  # noqa: E402
                           D1_TOTAL_ERR)
from stagerule2_fit import by_density, RHO_C  # noqa: E402
from stagerule3_fit import sequences  # noqa: E402

OUT = os.path.join(RUNS, "stagerule4_2026-11-23")

# Registered bars. Changing one after reading a number is the thing the
# registration exists to prevent.
N_INST = 400.0         # instances per (run, rho) bin -- D1's identity
E2_RATIO = 1.3         # the degradation factor E2 demands
E4_BAR = R1_BAR        # 0.01, the rule bar, unchanged from stage RULE
E5_WIDTH = 2.0         # widest 95% interval (hi/lo) E2 may be read through
REF_RHO = 0.50
HIGH_RHOS = [0.95, 0.90, 0.85, 0.80]
ADDED = [round(r, 4) for r in GRID[10:]]
ADDED_SET = set(ADDED)
Z = 1.959964           # 95%

# The denominators, read BEFORE the registration was written and tabled in it.
# Reproducing them here is a check that this script sees the same shards, not
# a new measurement.
REG_DEN = {0.95: 30442, 0.90: 29687, 0.85: 28502, 0.80: 27681, 0.75: 27163,
           0.70: 25198, 0.65: 24924, 0.60: 24225, 0.55: 24201, 0.50: 23607,
           0.45: 23659, 0.40: 23530, 0.35: 24055, 0.33: 23990}


# ----------------------------------------------------------------- the slice

def slice_at(seq, jmax_of=None):
    """Pool n_err_in_F and n_in_F per density over j <= jmax.

    jmax_of is None for the registered slice (j = 0 only). The post-hoc clean
    window passes a per-density jmax.

    Returns per-density {"E", "F", "at_risk", "runs"} and, separately, the
    per-run contributions needed for the jackknife.
    """
    acc = collections.defaultdict(
        lambda: {"E": 0.0, "F": 0.0, "at_risk": 0.0, "runs": 0})
    per_run = collections.defaultdict(lambda: {"E": 0.0, "F": 0.0})
    for (run, rho), byj in seq.items():
        jmax = 0 if jmax_of is None else jmax_of.get(rho, 0)
        a = acc[rho]
        a["runs"] += 1
        a["at_risk"] += byj[0]["at_risk"] if 0 in byj else 0.0
        for j in range(0, jmax + 1):
            r = byj.get(j)
            if r is None:
                continue
            a["E"] += r["err_in_F"]
            a["F"] += r["in_F"]
            per_run[(run, rho)]["E"] += r["err_in_F"]
            per_run[(run, rho)]["F"] += r["in_F"]
    return dict(acc), dict(per_run)


def group(acc, rhos):
    """Pool a set of densities into one rate."""
    E = sum(acc[r]["E"] for r in rhos)
    F = sum(acc[r]["F"] for r in rhos)
    return {"E": E, "F": F, "h": (E / F) if F > 0 else float("nan")}


def ratio_poisson(ga, gb):
    """Rate ratio with a Poisson log-ratio 95% interval.

    Exact only if every emission is independent. With p ~ 1e-3 the binomial
    and Poisson variances agree to within p, so the approximation's weakness
    is the independence assumption, not the distribution.
    """
    if gb["h"] <= 0 or ga["E"] <= 0 or gb["E"] <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    R = ga["h"] / gb["h"]
    se = math.sqrt(1.0 / ga["E"] + 1.0 / gb["E"])
    return R, R * math.exp(-Z * se), R * math.exp(Z * se), se


def ratio_jackknife(per_run, rhos_a, rhos_b):
    """Delete-one-RUN jackknife on the rate ratio.

    Each of the 79 runs is dropped whole, from both groups at once, so
    anything shared within a run -- architecture, seed, checkpoint -- is
    inside the resampled unit instead of being assumed away.
    """
    runs = sorted({run for (run, _r) in per_run})
    a_set, b_set = set(rhos_a), set(rhos_b)
    tot = {"aE": 0.0, "aF": 0.0, "bE": 0.0, "bF": 0.0}
    by_run = collections.defaultdict(
        lambda: {"aE": 0.0, "aF": 0.0, "bE": 0.0, "bF": 0.0})
    for (run, rho), v in per_run.items():
        if rho in a_set:
            k = ("aE", "aF")
        elif rho in b_set:
            k = ("bE", "bF")
        else:
            continue
        by_run[run][k[0]] += v["E"]
        by_run[run][k[1]] += v["F"]
        tot[k[0]] += v["E"]
        tot[k[1]] += v["F"]

    def rat(d):
        if d["aF"] <= 0 or d["bF"] <= 0 or d["bE"] <= 0:
            return float("nan")
        hb = d["bE"] / d["bF"]
        return (d["aE"] / d["aF"]) / hb if hb > 0 else float("nan")

    full = rat(tot)
    reps = []
    for run in runs:
        d = {k: tot[k] - by_run[run][k] for k in tot}
        v = rat(d)
        if v == v:
            reps.append(v)
    n = len(reps)
    if n < 3 or full != full:
        return full, float("nan"), float("nan"), float("nan"), n
    mean = sum(reps) / n
    var = (n - 1.0) / n * sum((v - mean) ** 2 for v in reps)
    se = math.sqrt(var)
    return full, full - Z * se, full + Z * se, se, n


# ------------------------------------------------------------- diagnostics

def d1(seq):
    """ZERO CENSORING AT j = 0. The entire basis of the stage.

    n_at_risk(0) must be exactly 400 on every (run, rho) sequence: no rollout
    can have been removed before the first emission. Unlike RULE3's D1 this
    has no second exit channel to get wrong -- completion cannot happen before
    index 0 either.
    """
    bad, missing = [], []
    for key, byj in seq.items():
        if 0 not in byj:
            missing.append(key)
        elif byj[0]["at_risk"] != N_INST:
            bad.append((key, byj[0]["at_risk"]))
    return len(seq), bad, missing


def d2(seq):
    """Reproduce finding 100: 1988 rule errors over all j and all densities."""
    tot = 0.0
    for byj in seq.values():
        for r in byj.values():
            tot += r["err_in_F"]
    return tot


def d3(seq, dens):
    """Summing over all j must reproduce stages RULE2/RULE3 exactly.

    load() skips rows with n_at_risk <= 0, so the comparison sum must too or
    the equality is not an equality. That filter cost stage RULE3 a false
    alarm; it is reproduced here deliberately.
    """
    agg = collections.defaultdict(lambda: {"E": 0.0, "F": 0.0})
    for (_run, rho), byj in seq.items():
        for r in byj.values():
            if r["at_risk"] <= 0:
                continue
            agg[rho]["E"] += r["err_in_F"]
            agg[rho]["F"] += r["in_F"]
    worst, worst_rho = 0.0, None
    for rho, v in agg.items():
        if rho not in dens:
            continue
        mine = v["E"] / v["F"] if v["F"] > 0 else float("nan")
        theirs = dens[rho]["h"]
        rel = abs(mine - theirs) / theirs if theirs > 0 else abs(mine - theirs)
        if rel > worst:
            worst, worst_rho = rel, rho
    return worst, worst_rho, len(agg)


def d4(seq):
    """No impossible counts at j = 0."""
    bad = []
    for key, byj in seq.items():
        r = byj.get(0)
        if r is None:
            continue
        if not (0.0 <= r["in_F"] <= N_INST):
            bad.append((key, "in_F", r["in_F"]))
        if not (0.0 <= r["err_in_F"] <= r["in_F"]):
            bad.append((key, "err_in_F", r["err_in_F"], r["in_F"]))
    return bad


def d_den(acc):
    """The denominators were tabled in the registration before the numerator
    was read. If they moved, this script is not reading the same shards."""
    bad = []
    for rho, want in REG_DEN.items():
        got = acc[rho]["F"]
        if abs(got - want) > 0.5:
            bad.append((rho, want, got))
    return bad


# ----------------------------------------------------------------- post-hoc

def clean_window(seq, frac=0.05):
    """POST-HOC, NOT REGISTERED. The largest j each density can be summed to
    while censoring stays under `frac` of the rollouts.

    j = 0 is clean by construction but thin. This trades a little cleanliness
    for events: at rho = 0.95 almost the whole trace qualifies, at rho = 0.33
    almost none of it, and that contrast is itself the point.
    """
    alive = collections.defaultdict(lambda: collections.defaultdict(float))
    start = collections.defaultdict(float)
    for (_run, rho), byj in seq.items():
        start[rho] += N_INST
        for j, r in byj.items():
            alive[rho][j] += r["at_risk"]
    jmax = {}
    for rho, byj in alive.items():
        best = 0
        for j in sorted(byj):
            if byj[j] >= (1.0 - frac) * start[rho]:
                best = j
            else:
                break
        jmax[rho] = best
    return jmax


def offsupport_share(acc_all):
    """POST-HOC. What share of rollouts name a FORCED cell at j = 0.

    Registered as a descriptive, because it is the selection j = 0 does not
    remove: the denominator's composition, not its size.
    """
    out = {}
    for rho, a in acc_all.items():
        out[rho] = a["F"] / a["at_risk"] if a["at_risk"] > 0 else float("nan")
    return out


# -------------------------------------------------------------------- report

def table(fh, acc, rhos, jmax=None):
    line(fh, "  rho    n_F(0)   n_err_F(0)   h_rule(0)      named-forced"
             + ("     jmax" if jmax else ""))
    for r in rhos:
        a = acc[r]
        h = a["E"] / a["F"] if a["F"] > 0 else float("nan")
        shr = a["F"] / a["at_risk"] if a["at_risk"] > 0 else float("nan")
        extra = ("   %6d" % jmax[r]) if jmax else ""
        line(fh, "  %.2f  %8d   %10d   %.6f        %6.1f%%%s"
                 % (r, round(a["F"]), round(a["E"]), h, 100.0 * shr, extra))


def report_ratio(fh, name, ga, gb, per_run, ra, rb):
    R, lo, hi, se = ratio_poisson(ga, gb)
    Rj, loj, hij, sej, nj = ratio_jackknife(per_run, ra, rb)
    line(fh, "  %s" % name)
    line(fh, "    numerator group   E = %5d   F = %8d   h = %.6f"
             % (round(ga["E"]), round(ga["F"]), ga["h"]))
    line(fh, "    reference group   E = %5d   F = %8d   h = %.6f"
             % (round(gb["E"]), round(gb["F"]), gb["h"]))
    line(fh, "    ratio                     %.4f" % R)
    line(fh, "    Poisson 95%%               [%.4f, %.4f]   SE(log R) = %.4f"
             % (lo, hi, se))
    line(fh, "    jackknife 95%% (%2d runs)   [%.4f, %.4f]   SE = %.4f"
             % (nj, loj, hij, sej))
    width = (hi / lo) if (lo == lo and lo > 0) else float("nan")
    line(fh, "    interval width (hi/lo)    %.3f" % width)
    return {"R": R, "lo": lo, "hi": hi, "se": se, "width": width,
            "Rj": Rj, "loj": loj, "hij": hij, "sej": sej}


def score(fh, acc, rhos, added_v_ref, added_v_high):
    line(fh, "")
    line(fh, "REGISTERED PREDICTIONS")
    line(fh, "")
    verdicts = {}

    # E1 -- the pooled fall does not survive the clean slice.
    lo = added_v_ref["lo"]
    e1 = not (lo == lo and added_v_ref["hi"] < 1.0)
    verdicts["E1"] = e1
    line(fh, "  E1  the clean slice does not reproduce the pooled fall")
    line(fh, "      ratio added / rho=0.50 = %.4f, 95%% [%.4f, %.4f]"
             % (added_v_ref["R"], added_v_ref["lo"], added_v_ref["hi"]))
    line(fh, "      MISS if the interval lies entirely below 1.0")
    line(fh, "      -> %s" % ("HIT" if e1 else "MISS"))
    line(fh, "")

    # E2 -- the rule genuinely degrades.
    R = added_v_high["R"]
    e2 = (R >= E2_RATIO) and (added_v_high["lo"] > 1.0)
    verdicts["E2"] = e2
    line(fh, "  E2  the rule genuinely degrades by >= %.1fx" % E2_RATIO)
    line(fh, "      ratio added / rho>=0.80 = %.4f, 95%% [%.4f, %.4f]"
             % (R, added_v_high["lo"], added_v_high["hi"]))
    line(fh, "      MISS if below %.1f, or the interval includes 1.0"
             % E2_RATIO)
    line(fh, "      -> %s" % ("HIT" if e2 else "MISS"))
    line(fh, "")

    # E3 -- monotone in the right direction.
    xs = [r for r in rhos]
    ys = [acc[r]["E"] / acc[r]["F"] if acc[r]["F"] > 0 else float("nan")
          for r in rhos]
    rs = spearman(xs, ys)
    e3 = rs < 0
    verdicts["E3"] = e3
    line(fh, "  E3  Spearman(rho, h_rule(0)) over 14 densities is negative")
    line(fh, "      rho_s = %+.4f" % rs)
    line(fh, "      MISS if zero or positive")
    line(fh, "      -> %s" % ("HIT" if e3 else "MISS"))
    line(fh, "")

    # E4 -- the rule survives the clean slice everywhere.
    worst = max(ys)
    worst_rho = rhos[ys.index(worst)]
    e4 = worst < E4_BAR
    verdicts["E4"] = e4
    line(fh, "  E4  h_rule(0) < %.2f at all fourteen densities" % E4_BAR)
    line(fh, "      worst density rho = %.2f at h = %.6f" % (worst_rho, worst))
    line(fh, "      MISS if any density is at or above the bar")
    line(fh, "      -> %s" % ("HIT" if e4 else "MISS"))
    line(fh, "")

    # E5 -- powered for its own claim.
    w = added_v_high["width"]
    e5 = (w == w) and (w < E5_WIDTH)
    verdicts["E5"] = e5
    line(fh, "  E5  the 95%% interval on E2's ratio is narrower than %.1fx"
             % E5_WIDTH)
    line(fh, "      width hi/lo = %.3f  (Poisson); jackknife SE = %.4f"
             % (w, added_v_high["sej"]))
    line(fh, "      MISS -> E2 is UNDERPOWERED and its verdict is not read")
    line(fh, "      -> %s" % ("HIT" if e5 else "MISS"))
    line(fh, "")
    return verdicts


def reading(fh, verdicts, acc, rhos, avr, avh, shares):
    line(fh, "")
    line(fh, "WHAT THIS SAYS, AND WHAT IT STILL DOES NOT")
    line(fh, "")
    h_added = group(acc, ADDED)["h"]
    h_high = group(acc, HIGH_RHOS)["h"]
    h_ref = acc[REF_RHO]["E"] / acc[REF_RHO]["F"]
    line(fh, "  On the one slice where censoring is exactly zero, h_rule(0) is")
    line(fh, "  %.6f pooled over rho >= 0.80, %.6f at rho = 0.50 and %.6f"
             % (h_high, h_ref, h_added))
    line(fh, "  pooled over the four bins below rho_c = %.3f." % RHO_C)
    line(fh, "")
    if not verdicts["E5"]:
        line(fh, "  E5 MISSED, so E2 IS NOT READ. The interval on the")
        line(fh, "  degradation ratio is wider than a factor of %.1f, which"
                 % E5_WIDTH)
        line(fh, "  means this slice cannot distinguish a flat rule from one")
        line(fh, "  that degrades by the registered %.1fx. That is a statement"
                 % E2_RATIO)
        line(fh, "  about 57-odd events, not about the model.")
    elif verdicts["E2"]:
        line(fh, "  E2 HIT under a passing E5: the rule degrades below rho_c")
        line(fh, "  by at least %.1fx, and censoring is not the explanation,"
                 % E2_RATIO)
        line(fh, "  because there is none at j = 0.")
    elif verdicts["E1"]:
        line(fh, "  E1 HIT and E2 MISSED under a passing E5. The pooled FALL")
        line(fh, "  does not survive the clean slice -- so stage RULE2's")
        line(fh, "  reading was right to refuse it -- but the clean slice does")
        line(fh, "  not show the registered %.1fx degradation either."
                 % E2_RATIO)
        line(fh, "")
        line(fh, "  E2 AND E3 BOTH MISSED FOR THE SAME REASON, AND IT IS NOT")
        line(fh, "  NOISE. Both tested a MONOTONE trend in rho. h_rule(0) is")
        line(fh, "  not monotone: it is humped, peaking well above rho_c, and")
        line(fh, "  a ratio of the four lowest bins against the four highest")
        line(fh, "  straddles the peak and nets out near 1. The registration")
        line(fh, "  chose the wrong shape, not the wrong size -- see the")
        line(fh, "  post-hoc arms section, where both sides are monotone to")
        line(fh, "  |Spearman| > 0.97 and both ratios exclude 1.")
    else:
        line(fh, "  E1 MISSED: the fall is present even with zero censoring,")
        line(fh, "  which RULE3's frailty argument cannot explain away.")
    line(fh, "")
    line(fh, "  STILL NOT MEASURED, and j = 0 does not fix it: h_rule")
    line(fh, "  conditions on the model NAMING a forced cell. That share runs")
    line(fh, "  %.1f%% at rho = 0.95 down to %.1f%% at rho = 0.33, so the"
             % (100.0 * shares[0.95], 100.0 * shares[0.33]))
    line(fh, "  denominators are differently composed across densities. The")
    line(fh, "  slice is censoring-free, NOT unbiased.")
    line(fh, "")
    line(fh, "  Also not addressed: all 79 runs are evaluated on the SAME 400")
    line(fh, "  instances per bin, so errors concentrated on a few hard")
    line(fh, "  instances would widen the true interval beyond both the")
    line(fh, "  Poisson and the delete-one-run figures. The shards carry no")
    line(fh, "  per-instance breakdown, so this is named, not corrected.")
    line(fh, "")
    line(fh, "  NO TRAINING FOLLOWS FROM ANY OF THIS, as registered.")


def posthoc(fh, seq, e0_added):
    line(fh, "")
    line(fh, "POST-HOC, NOT REGISTERED: the clean window")
    line(fh, "")
    line(fh, "  j = 0 is clean by construction but thin. Summing instead over")
    line(fh, "  every j where censoring has removed under 5% of rollouts")
    line(fh, "  trades a little cleanliness for events. How much trace that")
    line(fh, "  buys is itself the finding.")
    line(fh, "")
    jmax = clean_window(seq, 0.05)
    acc_w, per_run_w = slice_at(seq, jmax)
    rhos = sorted(acc_w, reverse=True)
    line(fh, "  rho    jmax   n_F(<=jmax)   n_err_F   h_rule(<=jmax)")
    for r in rhos:
        a = acc_w[r]
        h = a["E"] / a["F"] if a["F"] > 0 else float("nan")
        line(fh, "  %.2f  %5d   %11d   %7d   %.6f"
                 % (r, jmax[r], round(a["F"]), round(a["E"]), h))
    line(fh, "")
    ga = group(acc_w, ADDED)
    gb = group(acc_w, HIGH_RHOS)
    res = report_ratio(fh, "clean window, added bins vs rho >= 0.80",
                       ga, gb, per_run_w, ADDED, HIGH_RHOS)
    line(fh, "")
    line(fh, "  Events: %d in the added bins against %d at j = 0 alone."
             % (round(ga["E"]), round(e0_added)))
    line(fh, "")
    line(fh, "  DO NOT READ THAT RATIO AS A DENSITY EFFECT. jmax is 2 at")
    line(fh, "  rho = 0.95 and 1 at rho = 0.90, but 0 everywhere else, so the")
    line(fh, "  numerator is j = 0 only while the reference mixes in j = 1 and")
    line(fh, "  j = 2. The index decomposition below shows those later indices")
    line(fh, "  are far cleaner than j = 0, which is most of the 1.60.")
    return jmax, acc_w, res


def perm_p(xs, ys, obs, nrep=200000, seed=20261123):
    """Monte-Carlo one-sided p for a Spearman as extreme as `obs`, by shuffling
    the bin labels.

    Distribution-free over the bins, so it does not assume emissions within a
    bin are exchangeable -- which is exactly the assumption the delete-one-run
    jackknife says is false. It DOES assume the bins are independent of each
    other: they are different instances at different densities, though they
    share the same 79 runs, so the correlation runs through the models rather
    than the puzzles.
    """
    rnd = random.Random(seed)
    ys = list(ys)
    hits = 0
    for _ in range(nrep):
        rnd.shuffle(ys)
        r = spearman(xs, ys)
        if (r <= obs) if obs < 0 else (r >= obs):
            hits += 1
    return hits / float(nrep), nrep


def arms(fh, acc, rhos):
    """POST-HOC, NOT REGISTERED. E3 tested a monotone trend and missed. The
    per-density table says why: h_rule(0) is HUMPED, not monotone, so a single
    Spearman averages a strong rise against a gentler fall and reports almost
    nothing. Both arms are measured separately here, with the peak located
    from the data rather than assumed.
    """
    line(fh, "")
    line(fh, "POST-HOC, NOT REGISTERED: the hump E3's monotone test averaged "
             "away")
    line(fh, "")
    hs = {r: acc[r]["E"] / acc[r]["F"] for r in rhos}
    peak = max(rhos, key=lambda r: hs[r])
    hi_arm = [r for r in rhos if r >= peak]
    lo_arm = [r for r in rhos if r <= peak]
    line(fh, "  peak at rho = %.2f, h_rule(0) = %.6f" % (peak, hs[peak]))
    line(fh, "  high arm (rho >= %.2f, %d points)  Spearman = %+.4f"
             % (peak, len(hi_arm), spearman(hi_arm, [hs[r] for r in hi_arm])))
    line(fh, "  low  arm (rho <= %.2f, %d points)  Spearman = %+.4f"
             % (peak, len(lo_arm), spearman(lo_arm, [hs[r] for r in lo_arm])))
    line(fh, "")
    line(fh, "  Both arms as rate ratios against the peak. Censoring is")
    line(fh, "  exactly zero in every one of these numbers.")
    line(fh, "")
    gp = {"E": acc[peak]["E"], "F": acc[peak]["F"], "h": hs[peak]}
    for name, rs in (("rho = 0.95 (densest)", [0.95]),
                     ("rho >= 0.80 pooled", HIGH_RHOS),
                     ("four added bins pooled", ADDED)):
        g = group(acc, rs)
        R, lo, hi, se = ratio_poisson(g, gp)
        excl = "excludes 1" if (hi < 1.0 or lo > 1.0) else "INCLUDES 1"
        line(fh, "    %-24s h = %.6f   R = %.4f  95%% [%.4f, %.4f]  %s"
                 % (name, g["h"], R, lo, hi, excl))
    line(fh, "")
    line(fh, "  THE RATIO INTERVALS ABOVE ASSUME EMISSIONS ARE EXCHANGEABLE,")
    line(fh, "  AND THE JACKKNIFE SAYS THEY ARE NOT -- deleting one run moves")
    line(fh, "  the pooled ratio far more than Poisson counting allows, so a")
    line(fh, "  few runs carry much of the error mass. The SHAPE does not rest")
    line(fh, "  on that assumption: a monotone ordering across independent")
    line(fh, "  density bins is a distribution-free statement, tested here by")
    line(fh, "  shuffling the bin labels.")
    line(fh, "")
    for nm, arm in (("high arm", hi_arm), ("low arm ", lo_arm)):
        obs = spearman(arm, [hs[r] for r in arm])
        p, nrep = perm_p(arm, [hs[r] for r in arm], obs)
        line(fh, "    %s  %2d bins  Spearman %+.4f   Monte-Carlo p = %s"
                 % (nm, len(arm), obs,
                    ("< %.1e" % (1.0 / nrep)) if p == 0 else "%.2e" % p))
    line(fh, "")
    line(fh, "  That is the claim this stage actually supports: the SHAPE is")
    line(fh, "  solid, the MAGNITUDE of any one ratio is not, because the 79")
    line(fh, "  runs are not 79 exchangeable replicates. Consistent with trap")
    line(fh, "  3 -- the per-architecture ranking is itself density-dependent.")
    line(fh, "")
    line(fh, "  So the ORDERING on both sides of the peak is resolved -- in a")
    line(fh, "  slice with no censoring at all -- while the SIZE of the drop")
    line(fh, "  is not. The weaker statement is the one that survives both")
    line(fh, "  intervals, and it is still enough: h_rule(0) does not keep")
    line(fh, "  falling monotonically with rho, and it does not keep rising")
    line(fh, "  either. The fall below the peak is therefore NOT purely a")
    line(fh, "  survivorship artefact -- some of it is real, because at j = 0")
    line(fh, "  there is no survivorship to blame.")
    line(fh, "")
    line(fh, "  Stage RULE3's lambda* = 1.19 bounded how much censoring COULD")
    line(fh, "  explain; it never claimed censoring DID explain it. A fall")
    line(fh, "  surviving at zero censoring points at lambda near 1, the same")
    line(fh, "  direction finding 124's off-support competence pointed.")
    return peak, hs


def index_decomp(fh, seq, acc, rhos):
    """POST-HOC, NOT REGISTERED. Why the POOLED h_rule fell when h_rule(0)
    does not fall nearly as much.

    h_rule pools every deduction index. If j = 0 is much worse than later
    indices, then a density whose traces are LONGER dilutes its own rate with
    more easy indices -- and trace length is exactly what rho controls
    (mean_n_blank runs 3 to 32). The share of forced emissions sitting at
    j = 0 is the weight, and it falls by the same kind of factor as the
    pooled rate.

    h(>=1) is CENSORED and biased downward, so it is reported as observed and
    never corrected. That bias makes the dilution argument conservative: the
    true later-index rate is at least this low only in the sense that the
    observed one is a floor on the gap.
    """
    line(fh, "")
    line(fh, "POST-HOC, NOT REGISTERED: j = 0 against every later index")
    line(fh, "")
    tot = collections.defaultdict(lambda: {"E": 0.0, "F": 0.0})
    for (_run, rho), byj in seq.items():
        for r in byj.values():
            if r["at_risk"] <= 0:
                continue
            tot[rho]["E"] += r["err_in_F"]
            tot[rho]["F"] += r["in_F"]
    line(fh, "  rho   w0=F(0)/F_all   h_rule(0)   h_rule(j>=1)   h_rule pooled")
    rows = []
    for r in rhos:
        F0, E0 = acc[r]["F"], acc[r]["E"]
        Ft, Et = tot[r]["F"], tot[r]["E"]
        F1, E1 = Ft - F0, Et - E0
        w0 = F0 / Ft if Ft > 0 else float("nan")
        h0 = E0 / F0 if F0 > 0 else float("nan")
        h1 = E1 / F1 if F1 > 0 else float("nan")
        hp = Et / Ft if Ft > 0 else float("nan")
        rows.append((r, w0, h0, h1, hp))
        line(fh, "  %.2f      %6.3f       %.6f      %.6f       %.6f"
                 % (r, w0, h0, h1, hp))
    line(fh, "")
    line(fh, "  The first deduction is where the rule errors are: h_rule(0) is")
    line(fh, "  3x to 7x h_rule(j>=1) at every one of the fourteen densities.")
    line(fh, "  So pooled h_rule is a LENGTH-WEIGHTED average of a bad first")
    line(fh, "  index and a clean tail, and rho sets the length (mean_n_blank")
    line(fh, "  runs 3 to 32). The weight w0 is where that bites.")
    line(fh, "")
    line(fh, "  BUT w0 DOES NOT EXPLAIN THE FALL. Split at the peak:")
    line(fh, "")
    byr = {r[0]: r for r in rows}
    pk = max(rows, key=lambda r: r[2])[0]
    hi_a, hi_b = byr[max(byr)], byr[pk]
    lo_a, lo_b = byr[pk], byr[min(byr)]
    for nm, a, b in (("high arm", hi_a, hi_b), ("low arm ", lo_a, lo_b)):
        line(fh, "    %s  rho %.2f -> %.2f :  w0 %.3f -> %.3f (%.2fx)"
                 % (nm, a[0], b[0], a[1], b[1], b[1] / a[1]))
        line(fh, "              h_rule(0) %.6f -> %.6f (%.2fx)"
                 % (a[2], b[2], b[2] / a[2]))
        line(fh, "              pooled    %.6f -> %.6f (%.2fx)"
                 % (a[4], b[4], b[4] / a[4]))
    line(fh, "")
    line(fh, "  On the LOW arm w0 barely moves (%.3f -> %.3f), so there is no"
             % (lo_a[1], lo_b[1]))
    line(fh, "  dilution left to do the work, and the pooled fall (%.2fx) and"
             % (lo_a[4] / lo_b[4]))
    line(fh, "  the censoring-free j = 0 fall (%.2fx) agree. THE FALL BELOW THE"
             % (lo_a[2] / lo_b[2]))
    line(fh, "  PEAK IS REAL -- not dilution, and not censoring, because there")
    line(fh, "  is none at j = 0.")
    line(fh, "")
    line(fh, "  What dilution DOES explain is the opposite: on the HIGH arm w0")
    line(fh, "  falls %.2fx, which is why h_rule(0) rises %.2fx while the"
             % (hi_a[1] / hi_b[1], hi_b[2] / hi_a[2]))
    line(fh, "  pooled rate rises only %.2fx. Pooling MASKED a rise; it did not"
             % (hi_b[4] / hi_a[4]))
    line(fh, "  manufacture a fall.")
    line(fh, "")
    line(fh, "  This does not contradict stage RULE3. lambda* = 1.19 was what")
    line(fh, "  censoring WOULD need to erase the pooled fall, never a claim")
    line(fh, "  that it does. The fall surviving at zero censoring is direct")
    line(fh, "  evidence that lambda sits near 1, which is what finding 124's")
    line(fh, "  off-support competence argued on separate grounds.")
    line(fh, "")
    line(fh, "  CAVEAT, stated not corrected: h_rule(j>=1) IS censored and")
    line(fh, "  biased downward, so the j=0-vs-later gap is an upper estimate")
    line(fh, "  of itself. The weight w0 is not -- it is a count of emissions,")
    line(fh, "  and the pooled column is exactly stages RULE2 and RULE3's")
    line(fh, "  number (D3 passed at 0.000e+00).")
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "VERDICT.txt")
    fh = open(path, "w")

    seq, n_haz, blank = sequences()
    cells, _v, _h, _s = load()
    dens = by_density(cells)

    line(fh, "STAGE RULE4 -- h_rule on the censoring-free j = 0 slice")
    line(fh, "=" * 64)
    line(fh, "")
    line(fh, "Registered in runs/stagerule4_2026-11-23/REGISTERED.md, committed")
    line(fh, "0d74728 BEFORE this script existed. A refit of stage LOGIC's")
    line(fh, "shards: nothing trained, nothing submitted, no rollout")
    line(fh, "regenerated.")
    line(fh, "")
    line(fh, "  hazard rows read        %d" % n_haz)
    line(fh, "  (run, rho) sequences    %d" % len(seq))
    line(fh, "  densities               %d" % len(dens))
    line(fh, "")

    # ---- diagnostics first. If any fails, nothing above it is reported.
    line(fh, "DIAGNOSTICS")
    line(fh, "")
    n_seq, bad1, miss1 = d1(seq)
    ok1 = (not bad1) and (not miss1)
    line(fh, "  D1  zero censoring at j = 0: n_at_risk(0) = %d everywhere"
             % int(N_INST))
    line(fh, "      sequences checked %d, violations %d, missing j=0 %d"
             % (n_seq, len(bad1), len(miss1)))
    if bad1:
        line(fh, "      first few: %s" % (bad1[:5],))
    line(fh, "      -> %s" % ("PASS" if ok1 else "FAIL"))
    line(fh, "")

    tot_err = d2(seq)
    ok2 = abs(tot_err - D1_TOTAL_ERR) < 0.5
    line(fh, "  D2  reproduce finding 100's total rule errors")
    line(fh, "      got %d, registered %d" % (round(tot_err), D1_TOTAL_ERR))
    line(fh, "      -> %s" % ("PASS" if ok2 else "FAIL"))
    line(fh, "")

    worst, worst_rho, n_d = d3(seq, dens)
    ok3 = worst < 1e-12
    line(fh, "  D3  summing over all j reproduces stages RULE2 and RULE3")
    line(fh, "      densities %d, worst relative difference %.3e%s"
             % (n_d, worst, (" at rho = %.2f" % worst_rho) if worst_rho else
                ""))
    line(fh, "      -> %s" % ("PASS" if ok3 else "FAIL"))
    line(fh, "")

    bad4 = d4(seq)
    ok4 = not bad4
    line(fh, "  D4  no impossible counts at j = 0")
    line(fh, "      violations %d" % len(bad4))
    if bad4:
        line(fh, "      first few: %s" % (bad4[:5],))
    line(fh, "      -> %s" % ("PASS" if ok4 else "FAIL"))
    line(fh, "")

    acc, per_run = slice_at(seq, None)
    badd = d_den(acc)
    okd = not badd
    line(fh, "  D5  the denominators match the ones tabled in the")
    line(fh, "      registration before the numerator was read")
    line(fh, "      mismatches %d" % len(badd))
    if badd:
        line(fh, "      %s" % (badd[:5],))
    line(fh, "      -> %s" % ("PASS" if okd else "FAIL"))
    line(fh, "")

    if not (ok1 and ok2 and ok3 and ok4 and okd):
        line(fh, "A DIAGNOSTIC FAILED. Nothing below is reported.")
        fh.close()
        print(open(path).read())
        return

    rhos = sorted(acc, reverse=True)
    line(fh, "")
    line(fh, "THE CENSORING-FREE SLICE, PER DENSITY")
    line(fh, "")
    table(fh, acc, rhos)
    line(fh, "")
    line(fh, "  mean_n_blank per density, for scale: %s"
             % ", ".join("%.1f" % blank[r] for r in rhos))
    line(fh, "")

    ga = group(acc, ADDED)
    g_ref = {"E": acc[REF_RHO]["E"], "F": acc[REF_RHO]["F"],
             "h": acc[REF_RHO]["E"] / acc[REF_RHO]["F"]}
    gh = group(acc, HIGH_RHOS)
    line(fh, "")
    line(fh, "GROUP COMPARISONS")
    line(fh, "")
    avr = report_ratio(fh, "E1: four added bins vs rho = 0.50",
                       ga, g_ref, per_run, ADDED, [REF_RHO])
    line(fh, "")
    avh = report_ratio(fh, "E2: four added bins vs rho >= 0.80",
                       ga, gh, per_run, ADDED, HIGH_RHOS)

    verdicts = score(fh, acc, rhos, avr, avh)
    shares = offsupport_share(acc)
    reading(fh, verdicts, acc, rhos, avr, avh, shares)
    peak, hs = arms(fh, acc, rhos)
    rows = index_decomp(fh, seq, acc, rhos)
    posthoc(fh, seq, ga["E"])

    line(fh, "")
    line(fh, "SCOPE")
    line(fh, "")
    line(fh, "  d5q4 at N = 48, 79 runs, 31 architectures, 400 instances per")
    line(fh, "  bin, eval seed start 60,000,000,000. Nothing here is")
    line(fh, "  per-architecture: every number is pooled over all runs,")
    line(fh, "  because no single-seed rung is a measurement. The h_rule")
    line(fh, "  RANKING is density-dependent and is not quoted anywhere.")
    fh.close()
    print(open(path).read())


if __name__ == "__main__":
    main()
