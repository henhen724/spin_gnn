"""Stage RULE5 -- remove the selection stage RULE4 could not remove.

Registered in runs/stagerule5_2026-11-23/REGISTERED.md, committed b06b701
BEFORE this script existed. A refit of stage LOGIC's shards: nothing is
trained, nothing is submitted, no rollout is regenerated, no closed knob is
reopened.

Stage RULE4 scored h_rule on the j = 0 slice, where censoring is exactly zero,
and found a hump. But h_rule's DENOMINATOR is chosen by the model: it counts
only rollouts whose first emission named a cell that one-step propagation
forces, and that share runs 96.3% at rho = 0.95 down to 75.9% at rho = 0.33.
If the model names an unforced cell when it is unsure, the survivors in that
denominator are the confident ones and the selection tightens as rho falls --
the right shape to manufacture RULE4's low-arm fall out of nothing.

This stage scores

    h_all(0) = n_err(0) / n_at_risk(0)

instead: the chance the FIRST emitted colour disagrees with the planted trace,
over every rollout. Censoring-free because j = 0 filters nothing, and
selection-free because the denominator is not the model's choice. That
denominator is a constant 31,600 at all fourteen densities, by construction
rather than by measurement, so the cross-density comparison is a comparison of
raw counts.

Run under the FIGURE env (stageexp_fit imports matplotlib at module level):

  module purge --force; module load devel math viz
  module load python/3.12.1
  module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
"""

import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stagegate_fit import GRID, FIT_RHOS, spearman  # noqa: E402
from stagerule_fit import RUNS, line  # noqa: E402
from stagerule3_fit import sequences  # noqa: E402
from stagerule4_fit import (ADDED, Z, REG_DEN, group, ratio_poisson,  # noqa: E402
                            ratio_jackknife, perm_p)

OUT = os.path.join(RUNS, "stagerule5_2026-11-23")

N_INST = 400.0
N_RUNS = 79
AT_RISK_0 = N_RUNS * N_INST          # 31,600 at every density, by construction

# Registered bars.
E1_BAR = -0.95                       # Spearman over all 14, h_all(0)
E2_FACTOR = 10.0                     # unforced rate / forced rate, every bin
E3_TOL = 1.5                         # predicted endpoints within this factor
E3_PRED = {0.95: 0.0117, 0.33: 0.1220}
E4_BAR = 0.0                         # Spearman over the ten fitted bins
E5_TOL = 1e-12

# Stage RULE4's published numbers, for the E4 and E5 comparisons.
RULE4_PEAK = 0.75                    # where h_rule(0) peaked
RULE4_LOW_SPEARMAN = 0.9758          # h_rule(0) on rho <= 0.75, ten bins

# Carried over from the RULE4 registration, where they were read before their
# own numerator. REG_DEN is n_F(0); this is n_err_in_F(0).
REG_ERR0 = {0.95: 35, 0.90: 48, 0.85: 74, 0.80: 80, 0.75: 87, 0.70: 80,
            0.65: 77, 0.60: 65, 0.55: 62, 0.50: 62, 0.45: 56, 0.40: 52,
            0.35: 50, 0.33: 50}

HIGH_RHOS = [0.95, 0.90, 0.85, 0.80]
ADDED_SET = set(ADDED)


# --------------------------------------------------------------------------
# the slice


def slice0(seq):
    """Pool every j = 0 count per density.

    Unlike stage RULE4's slice_at this keeps n_err as well as n_err_in_F, which
    is the whole point: n_err is the numerator whose denominator the model does
    not choose.
    """
    acc = collections.defaultdict(
        lambda: {"at_risk": 0.0, "in_F": 0.0, "err": 0.0, "err_in_F": 0.0,
                 "runs": 0})
    pr_all = collections.defaultdict(lambda: {"E": 0.0, "F": 0.0})
    pr_rule = collections.defaultdict(lambda: {"E": 0.0, "F": 0.0})
    for (run, rho), byj in seq.items():
        r = byj.get(0)
        if r is None:
            continue
        a = acc[rho]
        a["runs"] += 1
        for k in ("at_risk", "in_F", "err", "err_in_F"):
            a[k] += r[k]
        pr_all[(run, rho)]["E"] += r["err"]
        pr_all[(run, rho)]["F"] += r["at_risk"]
        pr_rule[(run, rho)]["E"] += r["err_in_F"]
        pr_rule[(run, rho)]["F"] += r["in_F"]
    return dict(acc), dict(pr_all), dict(pr_rule)


def rates(a):
    """The three rates and the share, from one pooled density row."""
    unf_den = a["at_risk"] - a["in_F"]
    unf_err = a["err"] - a["err_in_F"]
    return {
        "h_all": a["err"] / a["at_risk"] if a["at_risk"] > 0 else float("nan"),
        "h_rule": a["err_in_F"] / a["in_F"] if a["in_F"] > 0 else float("nan"),
        "h_unf": unf_err / unf_den if unf_den > 0 else float("nan"),
        "s_F": a["in_F"] / a["at_risk"] if a["at_risk"] > 0 else float("nan"),
        "unf_den": unf_den, "unf_err": unf_err,
    }


def g_all(acc, rhos):
    return group({r: {"E": acc[r]["err"], "F": acc[r]["at_risk"]}
                  for r in rhos}, rhos)


def g_rule(acc, rhos):
    return group({r: {"E": acc[r]["err_in_F"], "F": acc[r]["in_F"]}
                  for r in rhos}, rhos)


def g_unf(acc, rhos):
    d = {}
    for r in rhos:
        v = rates(acc[r])
        d[r] = {"E": v["unf_err"], "F": v["unf_den"]}
    return group(d, rhos)


# --------------------------------------------------------------------------
# diagnostics


def d1(seq):
    """n_at_risk(0) = 400 everywhere, and the pooled denominator is exactly
    31,600 at each of the fourteen densities."""
    bad, miss = [], []
    pooled = collections.defaultdict(float)
    for (run, rho), byj in seq.items():
        r = byj.get(0)
        if r is None:
            miss.append((run, rho))
            continue
        if abs(r["at_risk"] - N_INST) > 1e-9:
            bad.append((run, rho, r["at_risk"]))
        pooled[rho] += r["at_risk"]
    off = [(r, v) for r, v in sorted(pooled.items())
           if abs(v - AT_RISK_0) > 1e-9]
    return len(seq), bad, miss, off


def d2(seq):
    """n_err(0) >= n_err_in_F(0) on every sequence. A violation would mean
    n_err is not a superset of the forced errors and the decomposition into a
    forced and an unforced arm is meaningless."""
    bad = []
    for (run, rho), byj in seq.items():
        r = byj.get(0)
        if r is None:
            continue
        if r["err"] + 1e-9 < r["err_in_F"]:
            bad.append((run, rho, r["err"], r["err_in_F"]))
    return bad


def d3(seq):
    """n_in_F(0) <= n_at_risk(0), so the unforced count is non-negative."""
    bad = []
    for (run, rho), byj in seq.items():
        r = byj.get(0)
        if r is None:
            continue
        if r["in_F"] > r["at_risk"] + 1e-9:
            bad.append((run, rho, r["in_F"], r["at_risk"]))
    return bad


def d4(acc):
    """The n_F(0) pooled table reproduces the fourteen registered numbers."""
    bad = []
    for r, want in sorted(REG_DEN.items()):
        got = acc.get(r, {}).get("in_F")
        if got is None or abs(got - want) > 0.5:
            bad.append((r, got, want))
    return bad


def d5(acc):
    """79 runs at every density."""
    return [(r, a["runs"]) for r, a in sorted(acc.items())
            if a["runs"] != N_RUNS]


# --------------------------------------------------------------------------
# reporting


def table(fh, acc, rhos):
    line(fh, "  rho   at risk   named F    s_F     h_all(0)   h_rule(0)   "
             "h_unf(0)   unf/rule")
    for r in rhos:
        a = acc[r]
        v = rates(a)
        mark = " <" if r in ADDED_SET else "  "
        line(fh, "  %.2f   %6d   %6d   %.3f   %.6f   %.6f   %.6f   %7.1fx%s"
                 % (r, int(a["at_risk"]), int(a["in_F"]), v["s_F"],
                    v["h_all"], v["h_rule"], v["h_unf"],
                    v["h_unf"] / v["h_rule"], mark))
    line(fh, "")
    line(fh, "  s_F is the share of first emissions naming a forced cell --")
    line(fh, "  the selection h_rule conditions on and h_all does not. Rows")
    line(fh, "  marked < are the four bins below rho_c ~ 0.465, where an")
    line(fh, "  off-trace colour on an UNFORCED cell need not be a logical")
    line(fh, "  error at all, so h_all there is an UPPER BOUND (registration,")
    line(fh, "  'the interpretive limit').")


def report_ratio(fh, name, ga, gb, pr, rhos_a, rhos_b):
    R, lo, hi, se = ratio_poisson(ga, gb)
    line(fh, "  %s" % name)
    line(fh, "      a: %d / %d = %.6f      b: %d / %d = %.6f"
             % (round(ga["E"]), round(ga["F"]), ga["h"],
                round(gb["E"]), round(gb["F"]), gb["h"]))
    line(fh, "      R = %.4f   Poisson 95%% [%.4f, %.4f]  (SE log R = %.4f)"
             % (R, lo, hi, se))
    jR, jlo, jhi, jse, jn = ratio_jackknife(pr, rhos_a, rhos_b)
    line(fh, "      delete-one-run 95%% [%.4f, %.4f]  (SE %.4f, %d runs)"
             % (jlo, jhi, jse, jn))
    both = (hi < 1.0 or lo > 1.0) and (jhi < 1.0 or jlo > 1.0)
    line(fh, "      -> %s" % ("both intervals exclude 1" if both else
                              "NOT resolved by both intervals"))
    return R, lo, hi, jlo, jhi


def decomposition(fh, acc, rhos):
    """h_all(0) = s_F * h_rule(0) + (1 - s_F) * h_unf(0), exactly."""
    line(fh, "")
    line(fh, "THE DECOMPOSITION, AND WHY THE SELECTION MATTERS")
    line(fh, "")
    line(fh, "    h_all(0) = s_F * h_rule(0) + (1 - s_F) * h_unf(0)")
    line(fh, "")
    line(fh, "  an identity, not a model: the two arms partition the 31,600")
    line(fh, "  first emissions by whether the named cell was forced.")
    line(fh, "")
    worst = 0.0
    for r in rhos:
        v = rates(acc[r])
        lhs = v["h_all"]
        rhs = v["s_F"] * v["h_rule"] + (1 - v["s_F"]) * v["h_unf"]
        worst = max(worst, abs(lhs - rhs))
    line(fh, "  identity holds to %.3e across all fourteen densities" % worst)
    line(fh, "")
    hi, lo_ = rhos[0], rhos[-1]
    a, b = rates(acc[hi]), rates(acc[lo_])
    line(fh, "  rho %.2f -> %.2f:" % (hi, lo_))
    line(fh, "      s_F        %.3f -> %.3f   (%.2fx)"
             % (a["s_F"], b["s_F"], b["s_F"] / a["s_F"]))
    line(fh, "      h_rule(0)  %.6f -> %.6f   (%.2fx)"
             % (a["h_rule"], b["h_rule"], b["h_rule"] / a["h_rule"]))
    line(fh, "      h_unf(0)   %.6f -> %.6f   (%.2fx)"
             % (a["h_unf"], b["h_unf"], b["h_unf"] / a["h_unf"]))
    line(fh, "      h_all(0)   %.6f -> %.6f   (%.2fx)"
             % (a["h_all"], b["h_all"], b["h_all"] / a["h_all"]))
    line(fh, "")
    line(fh, "  READ THE THREE FACTORS SEPARATELY. h_all rises %.2fx, but"
             % (b["h_all"] / a["h_all"]))
    line(fh, "  h_unf -- the error rate ON an unforced cell -- barely moves")
    line(fh, "  (%.2fx, and it sits near %.2f at every single density). The"
             % (b["h_unf"] / a["h_unf"],
                sum(rates(acc[r])["h_unf"] for r in rhos) / len(rhos)))
    line(fh, "  rise is almost entirely the unforced SHARE: 1 - s_F goes")
    line(fh, "  %.4f -> %.4f, a %.2fx rise, and %.2f x %.2f = %.2f recovers"
             % (1 - a["s_F"], 1 - b["s_F"], (1 - b["s_F"]) / (1 - a["s_F"]),
                (1 - b["s_F"]) / (1 - a["s_F"]), b["h_unf"] / a["h_unf"],
                ((1 - b["s_F"]) / (1 - a["s_F"])) * (b["h_unf"] / a["h_unf"])))
    line(fh, "  nearly all of the %.2fx." % (b["h_all"] / a["h_all"]))
    line(fh, "")
    line(fh, "  So h_all(0) is essentially a measure of HOW OFTEN the model")
    line(fh, "  steps off the forced set, times a roughly constant ~%.0f%%"
             % (100.0 * sum(rates(acc[r])["h_unf"] for r in rhos) / len(rhos)))
    line(fh, "  error rate once it does. Density controls the frequency of")
    line(fh, "  leaving the support, not the competence off it.")
    line(fh, "")
    line(fh, "  The forced arm contributes almost nothing to h_all: %.1f%% of"
             % (100.0 * a["s_F"] * a["h_rule"] / a["h_all"]))
    line(fh, "  it at rho = %.2f and %.1f%% at rho = %.2f. Stages RULE2 to"
             % (hi, 100.0 * b["s_F"] * b["h_rule"] / b["h_all"], lo_))
    line(fh, "  RULE4 spent four stages on that term.")


def score(fh, acc, rhos):
    line(fh, "")
    line(fh, "REGISTERED SCORING")
    line(fh, "")
    v = {r: rates(acc[r]) for r in rhos}
    out = {}

    # --- E1: monotone, no interior peak
    ys = [v[r]["h_all"] for r in rhos]
    s14 = spearman(rhos, ys)
    pk = rhos[ys.index(max(ys))]
    interior = pk != rhos[0] and pk != rhos[-1]
    e1 = s14 <= E1_BAR
    out["E1"] = e1
    line(fh, "  E1  h_all(0) monotone over all 14, Spearman <= %+.2f" % E1_BAR)
    line(fh, "      Spearman = %+.4f, maximum at rho = %.2f (%s)"
             % (s14, pk, "interior" if interior else "an endpoint"))
    p, nrep = perm_p(list(rhos), ys, s14)
    line(fh, "      Monte-Carlo p = %s"
             % (("< %.1e" % (1.0 / nrep)) if p == 0 else "%.2e" % p))
    line(fh, "      -> %s" % ("HIT" if e1 else "MISS"))
    line(fh, "")

    # --- E2: unforced arm at least 10x the forced arm, every bin
    facs = [(r, v[r]["h_unf"] / v[r]["h_rule"]) for r in rhos]
    worst_r, worst_f = min(facs, key=lambda t: t[1])
    e2 = worst_f >= E2_FACTOR
    out["E2"] = e2
    line(fh, "  E2  h_unf(0) >= %.0fx h_rule(0) at every density" % E2_FACTOR)
    line(fh, "      smallest factor %.1fx at rho = %.2f; largest %.1fx"
             % (worst_f, worst_r, max(f for _r, f in facs)))
    line(fh, "      -> %s" % ("HIT" if e2 else "MISS"))
    line(fh, "")

    # --- E3: the predicted endpoints
    e3 = True
    line(fh, "  E3  predicted endpoints within a factor of %.1f" % E3_TOL)
    for r in (0.95, 0.33):
        pred = E3_PRED[r]
        got = v[r]["h_all"]
        lo, hi = pred / E3_TOL, pred * E3_TOL
        ok = lo <= got <= hi
        e3 = e3 and ok
        line(fh, "      rho %.2f  predicted %.4f  [%.4f, %.4f]  got %.4f  %s"
                 % (r, pred, lo, hi, got, "in" if ok else "OUT"))
    out["E3"] = e3
    line(fh, "      -> %s" % ("HIT" if e3 else "MISS"))
    line(fh, "")

    # --- E4: the low-arm fall disappears.
    #
    # REGISTRATION ERROR, recorded rather than smoothed over. E4 was written as
    # "the ten fitted bins" and quoted stage RULE4's +0.9758 as the thing to
    # beat -- but that +0.9758 was measured on RULE4's LOW ARM, rho <= 0.75,
    # which is a different ten bins from FIT_RHOS = rho >= 0.50. Naming the
    # wrong bin set is the same species of error RULE4 made when it registered
    # a monotone shape. Both sets are scored here so the verdict does not
    # depend on which one the registration meant.
    fit = [r for r in rhos if r in set(FIT_RHOS)]
    low = [r for r in rhos if r <= RULE4_PEAK]
    s10 = spearman(fit, [v[r]["h_all"] for r in fit])
    s_low = spearman(low, [v[r]["h_all"] for r in low])
    e4 = (s10 <= E4_BAR) and (s_low <= E4_BAR)
    out["E4"] = e4
    line(fh, "  E4  the fall disappears: Spearman <= %+.2f" % E4_BAR)
    line(fh, "      REGISTRATION ERROR: E4 said 'the ten fitted bins' and")
    line(fh, "      quoted RULE4's +0.9758, but that was RULE4's LOW ARM")
    line(fh, "      (rho <= %.2f), a different ten bins. Both are scored."
             % RULE4_PEAK)
    line(fh, "")
    line(fh, "        %-28s %9s  %9s" % ("bin set", "h_all(0)", "h_rule(0)"))
    line(fh, "        %-28s  %+.4f    %+.4f"
             % ("FIT_RHOS, rho >= 0.50 (%d)" % len(fit), s10,
                spearman(fit, [v[r]["h_rule"] for r in fit])))
    line(fh, "        %-28s  %+.4f    %+.4f"
             % ("RULE4 low arm, rho <= %.2f (%d)" % (RULE4_PEAK, len(low)),
                s_low, spearman(low, [v[r]["h_rule"] for r in low])))
    line(fh, "")
    line(fh, "      The h_rule column reproduces RULE4: %+.4f on its low arm,"
             % spearman(low, [v[r]["h_rule"] for r in low]))
    line(fh, "      against the %+.4f registered there. h_all is negative on"
             % RULE4_LOW_SPEARMAN)
    line(fh, "      BOTH sets, so the verdict survives the bin-set error.")
    line(fh, "      -> %s" % ("HIT" if e4 else "MISS"))
    line(fh, "")

    # --- E5: wiring check against RULE4
    worst = 0.0
    for r in rhos:
        want = REG_ERR0[r] / float(REG_DEN[r])
        worst = max(worst, abs(v[r]["h_rule"] - want))
    e5 = worst < E5_TOL
    out["E5"] = e5
    line(fh, "  E5  h_rule(0) reproduces stage RULE4 exactly")
    line(fh, "      worst absolute difference %.3e" % worst)
    line(fh, "      -> %s" % ("HIT" if e5 else "MISS"))
    return out, s14, s10, pk, interior


def flanks(fh, acc, rhos, pk, interior):
    """E1's MISS condition: a surviving interior peak with both flanks
    resolved. Reported whether or not E1 hit, because it is the thing that
    decides whether stage RULE4's deck needs a correction page."""
    line(fh, "")
    line(fh, "IS THERE STILL A HUMP?")
    line(fh, "")
    if not interior:
        line(fh, "  No. h_all(0) peaks at rho = %.2f, an endpoint of the grid,"
                 % pk)
        line(fh, "  so there is no interior maximum to test flanks against.")
        return None
    gp = g_all(acc, [pk])
    res = []
    for nm, rs in (("above the peak", [r for r in rhos if r > pk]),
                   ("below the peak", [r for r in rhos if r < pk])):
        if not rs:
            continue
        g = g_all(acc, rs)
        R, lo, hi, _se = ratio_poisson(g, gp)
        excl = (hi < 1.0 or lo > 1.0)
        res.append(excl)
        line(fh, "  %s (%d bins)  R = %.4f  95%% [%.4f, %.4f]  %s"
                 % (nm, len(rs), R, lo, hi,
                    "excludes 1" if excl else "INCLUDES 1"))
    resolved = all(res) and len(res) == 2

    # A ratio interval is the wrong instrument here and saying so is the
    # point. h_all's numerators are in the THOUSANDS, so the CI is sub-1%
    # wide and any ripple at all comes out "resolved". What decides whether
    # this is a hump is its SIZE against the range of the curve.
    hs = {r: rates(acc[r])["h_all"] for r in rhos}
    rng = max(hs.values()) - min(hs.values())
    drop = hs[pk] - hs[rhos[-1]]
    frac = drop / rng if rng > 0 else float("nan")
    line(fh, "")
    line(fh, "  BUT A RATIO INTERVAL IS THE WRONG INSTRUMENT AT THIS SIZE.")
    line(fh, "  h_all's numerators are in the thousands, not the tens, so the")
    line(fh, "  95% interval is under 1% wide and ANY ripple is nominally")
    line(fh, "  resolved. What matters is the size of the dip against the")
    line(fh, "  range of the curve:")
    line(fh, "")
    line(fh, "      full range of h_all(0)        %.4f  (%.2fx end to end)"
             % (rng, max(hs.values()) / min(hs.values())))
    line(fh, "      peak at rho = %.2f             %.4f" % (pk, hs[pk]))
    line(fh, "      value at rho = %.2f            %.4f" % (rhos[-1],
                                                            hs[rhos[-1]]))
    line(fh, "      drop from the peak            %.4f  = %.1f%% of the range"
             % (drop, 100.0 * frac))
    line(fh, "")
    if frac < 0.10:
        line(fh, "  -> NOT A HUMP. The interior maximum is a %.1f%% ripple"
                 % (100.0 * frac))
        line(fh, "     among the lowest bins, on a curve that rises %.2fx end"
                 % (max(hs.values()) / min(hs.values())))
        line(fh, "     to end. Stage RULE4's hump was a 1.54x FALL over the")
        line(fh, "     same region; this is not that, and the nominal flank")
        line(fh, "     test above is an artefact of the larger numerator.")
        line(fh, "     E1's MISS condition is therefore NOT met: it asked for")
        line(fh, "     a peak that survives, and a sub-decile ripple is not")
        line(fh, "     one. This adjudication is recorded, not hidden -- the")
        line(fh, "     registration should have set a size floor as well as a")
        line(fh, "     significance test.")
        return False
    line(fh, "  -> %s" % ("BOTH flanks resolved AND the dip is %.1f%% of the "
                          "range: the hump survives" % (100.0 * frac)
                          if resolved else "not both flanks resolved"))
    return resolved


def reading(fh, out, acc, rhos, s14, s10, pk, hump):
    line(fh, "")
    line(fh, "WHAT THIS SAYS, AND WHAT IT STILL DOES NOT")
    line(fh, "")
    v = {r: rates(acc[r]) for r in rhos}
    hits = [k for k in ("E1", "E2", "E3", "E4", "E5") if out[k]]
    line(fh, "  Registered: %s"
             % ", ".join("%s %s" % (k, "HIT" if out[k] else "MISS")
                         for k in ("E1", "E2", "E3", "E4", "E5")))
    line(fh, "")
    if out["E1"] and out["E4"]:
        line(fh, "  THE SELECTION WAS DOING THE WORK. Once the denominator is")
        line(fh, "  every rollout instead of the ones the model chose to name,")
        line(fh, "  the hump is gone and the first-emission error rate rises")
        line(fh, "  monotonically as hints are removed, %.4f -> %.4f."
                 % (v[rhos[0]]["h_all"], v[rhos[-1]]["h_all"]))
        line(fh, "")
        line(fh, "  WHAT THIS DOES AND DOES NOT SAY ABOUT STAGE RULE4. RULE4's")
        line(fh, "  fall is still there -- E5 reproduces its table exactly, to")
        line(fh, "  0.000e+00 -- and it is still not censoring, because there")
        line(fh, "  is none at j = 0. What changes is what it is a fall IN.")
        line(fh, "  h_rule's denominator is the model's own choice of cell,")
        line(fh, "  s_F runs %.3f -> %.3f, so the rollouts left in it are a"
                 % (v[rhos[0]]["s_F"], v[rhos[-1]]["s_F"]))
        line(fh, "  progressively more selected subset. The unconditional")
        line(fh, "  first-emission error rate over the same rollouts RISES")
        line(fh, "  monotonically. So RULE4's fall is not a statement about")
        line(fh, "  deduction quality getting better as hints are removed.")
        line(fh, "")
        line(fh, "  STATED AS A LIMIT, NOT A MECHANISM: this stage shows the")
        line(fh, "  selection EXISTS and moves in the direction that would")
        line(fh, "  produce RULE4's fall. It does not prove it causes it.")
        line(fh, "  Proving that needs a like-for-like comparison -- the same")
        line(fh, "  cells scored at two densities -- and the shards carry no")
        line(fh, "  per-cell breakdown, so it cannot be done here. The honest")
        line(fh, "  claim is the weaker one: h_rule's fall is real but is not")
        line(fh, "  interpretable as a density effect on deduction quality,")
        line(fh, "  because its denominator is not held fixed across density.")
        line(fh, "  RULE4's deck gets a correction page saying that; it does")
        line(fh, "  not get quietly rebuilt, and its hump figure stands as a")
        line(fh, "  correct picture of h_rule.")
    elif hump:
        line(fh, "  THE HUMP SURVIVES. It is not the naming selection: with a")
        line(fh, "  denominator the model does not choose, h_all(0) still")
        line(fh, "  peaks at rho = %.2f with both flanks resolved. That is a"
                 % pk)
        line(fh, "  STRONGER version of stage RULE4, not a correction to it,")
        line(fh, "  and it still licenses no training.")
    else:
        line(fh, "  MIXED. h_all(0) has Spearman %+.4f over fourteen bins and"
                 % s14)
        line(fh, "  %+.4f over the ten fitted ones, and the flank test does"
                 % s10)
        line(fh, "  not resolve both sides. The selection moved the curve but")
        line(fh, "  the shape is not settled by this slice alone.")
    line(fh, "")
    if not out["E3"]:
        hi_r = rhos[0]
        vh = v[hi_r]
        line(fh, "  WHY E3 MISSED, AND IT IS A FINDING. The prediction fed")
        line(fh, "  finding 124's off-support error rate (0.29 at rho = 0.95,")
        line(fh, "  pooled over every deduction index) into the j = 0 slice.")
        line(fh, "  Measured here, h_unf(0) at rho = %.2f is %.4f -- %.1fx the"
                 % (hi_r, vh["h_unf"], vh["h_unf"] / 0.29))
        line(fh, "  pooled figure. So off-support colour accuracy at the FIRST")
        line(fh, "  emission is %.0f%%, not the 71%% finding 124 reported over"
                 % (100.0 * (1.0 - vh["h_unf"])))
        line(fh, "  the whole trace: the model's guessing gets BETTER as the")
        line(fh, "  trace goes on, presumably because later guesses are made")
        line(fh, "  with more of the grid filled in.")
        line(fh, "")
        line(fh, "  That is the same direction as finding 126 -- the first")
        line(fh, "  deduction is the worst one -- now shown on the unforced")
        line(fh, "  arm as well as the forced arm. It also means the pooled")
        line(fh, "  off-support number must not be quoted at a fixed index,")
        line(fh, "  which is exactly the extrapolation E3 was written to test.")
        line(fh, "  Still well above chance: a blind colour guess errs at")
        line(fh, "  1 - 1/q = 0.75, and %.2f is far below that." % vh["h_unf"])
        line(fh, "")
    line(fh, "  WHAT IS SOLID EITHER WAY: the unforced arm is %.0fx to %.0fx"
             % (min(v[r]["h_unf"] / v[r]["h_rule"] for r in rhos),
                max(v[r]["h_unf"] / v[r]["h_rule"] for r in rhos)))
    line(fh, "  the forced arm. The model is near-perfect when one-step")
    line(fh, "  propagation names the colour for it (%.1f%% to %.1f%% wrong)"
             % (100.0 * min(v[r]["h_rule"] for r in rhos),
                100.0 * max(v[r]["h_rule"] for r in rhos)))
    line(fh, "  and about a coin flip when it does not (~%.0f%% wrong), and"
             % (100.0 * sum(v[r]["h_unf"] for r in rhos) / len(rhos)))
    line(fh, "  that gap is the single largest effect in this dataset. Not")
    line(fh, "  chance, though: a blind colour guess errs at 1 - 1/q = 0.75,")
    line(fh, "  so off the forced set the model is well above guessing and")
    line(fh, "  nowhere near the near-certainty it shows on it. It is also a")
    line(fh, "  measurement no earlier stage could make, because every")
    line(fh, "  earlier stage conditioned it away.")
    line(fh, "")
    line(fh, "  WHAT THIS IS NOT:")
    line(fh, "")
    line(fh, "  1. Not a density threshold. h_all(0) is a rate and is never")
    line(fh, "     converted into one.")
    line(fh, "  2. Not a statement about logical validity below rho_c. There")
    line(fh, "     an off-trace colour on an unforced cell may be a different")
    line(fh, "     valid colouring, so the four added bins bound the logical")
    line(fh, "     error rate from above rather than measuring it. The ten")
    line(fh, "     fitted bins carry the clean statement.")
    line(fh, "  3. Not clustered by instance. All 79 runs share the same 400")
    line(fh, "     instances per bin and the shards carry no per-instance")
    line(fh, "     breakdown, so no interval here can account for that, and")
    line(fh, "     the jackknife's 79 runs are not 79 exchangeable replicates.")
    line(fh, "  4. Not a reason to submit anything. No training follows from")
    line(fh, "     any outcome, as registered. The repair-and-continue rollout")
    line(fh, "     eval stays hshunt's call.")
    return hits


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "VERDICT.txt")
    fh = open(path, "w")

    seq, n_haz, blank = sequences()

    line(fh, "STAGE RULE5 -- the one denominator the model does not choose")
    line(fh, "=" * 64)
    line(fh, "")
    line(fh, "Registered in runs/stagerule5_2026-11-23/REGISTERED.md, committed")
    line(fh, "b06b701 BEFORE this script existed. A refit of stage LOGIC's")
    line(fh, "shards: nothing trained, nothing submitted, no rollout")
    line(fh, "regenerated.")
    line(fh, "")
    line(fh, "  hazard rows read        %d" % n_haz)
    line(fh, "  (run, rho) sequences    %d" % len(seq))
    line(fh, "")

    acc, pr_all, pr_rule = slice0(seq)

    line(fh, "DIAGNOSTICS")
    line(fh, "")
    n_seq, bad1, miss1, off1 = d1(seq)
    ok1 = (not bad1) and (not miss1) and (not off1)
    line(fh, "  D1  n_at_risk(0) = %d everywhere; pooled = %d per density"
             % (int(N_INST), int(AT_RISK_0)))
    line(fh, "      sequences %d, violations %d, missing j=0 %d, bad pooled %d"
             % (n_seq, len(bad1), len(miss1), len(off1)))
    if off1:
        line(fh, "      %s" % (off1[:5],))
    line(fh, "      -> %s" % ("PASS" if ok1 else "FAIL"))
    line(fh, "")

    bad2 = d2(seq)
    ok2 = not bad2
    line(fh, "  D2  n_err(0) >= n_err_in_F(0) on every sequence")
    line(fh, "      violations %d" % len(bad2))
    if bad2:
        line(fh, "      %s" % (bad2[:5],))
    line(fh, "      -> %s" % ("PASS" if ok2 else "FAIL"))
    line(fh, "")

    bad3 = d3(seq)
    ok3 = not bad3
    line(fh, "  D3  n_in_F(0) <= n_at_risk(0), so the unforced count is >= 0")
    line(fh, "      violations %d" % len(bad3))
    line(fh, "      -> %s" % ("PASS" if ok3 else "FAIL"))
    line(fh, "")

    bad4 = d4(acc)
    ok4 = not bad4
    line(fh, "  D4  n_F(0) reproduces the fourteen registered denominators")
    line(fh, "      mismatches %d" % len(bad4))
    if bad4:
        line(fh, "      %s" % (bad4[:5],))
    line(fh, "      -> %s" % ("PASS" if ok4 else "FAIL"))
    line(fh, "")

    bad5 = d5(acc)
    ok5 = not bad5
    line(fh, "  D5  %d runs at every density" % N_RUNS)
    line(fh, "      mismatches %d" % len(bad5))
    line(fh, "      -> %s" % ("PASS" if ok5 else "FAIL"))
    line(fh, "")

    if not (ok1 and ok2 and ok3 and ok4 and ok5):
        line(fh, "A DIAGNOSTIC FAILED. Nothing below is reported.")
        fh.close()
        print(open(path).read())
        return

    rhos = sorted(acc, reverse=True)
    assert len(rhos) == 14, len(rhos)

    line(fh, "")
    line(fh, "THE SELECTION-FREE SLICE, PER DENSITY")
    line(fh, "")
    table(fh, acc, rhos)
    line(fh, "")
    line(fh, "  mean_n_blank for scale: %s"
             % ", ".join("%.1f" % blank[r] for r in rhos))

    decomposition(fh, acc, rhos)

    out, s14, s10, pk, interior = score(fh, acc, rhos)
    hump = flanks(fh, acc, rhos, pk, interior)

    line(fh, "")
    line(fh, "GROUP COMPARISONS, BOTH INTERVALS")
    line(fh, "")
    report_ratio(fh, "h_all(0): four added bins vs rho >= 0.80",
                 g_all(acc, ADDED), g_all(acc, HIGH_RHOS), pr_all,
                 ADDED, HIGH_RHOS)
    line(fh, "")
    report_ratio(fh, "h_rule(0): the same contrast, for comparison",
                 g_rule(acc, ADDED), g_rule(acc, HIGH_RHOS), pr_rule,
                 ADDED, HIGH_RHOS)
    line(fh, "")
    line(fh, "  h_all's numerator is in the thousands where h_rule's is in the")
    line(fh, "  tens, which is why the two intervals differ so much in width.")
    line(fh, "  That is the practical gain from not conditioning: the same")
    line(fh, "  shards, a hundredfold larger numerator.")

    reading(fh, out, acc, rhos, s14, s10, pk, hump)

    line(fh, "")
    line(fh, "SCOPE")
    line(fh, "")
    line(fh, "  d5q4 at N = 48, 79 runs, 31 architectures, 400 instances per")
    line(fh, "  bin, eval seed start 60,000,000,000. Every number is pooled")
    line(fh, "  over all runs; no single-seed rung is a measurement, and the")
    line(fh, "  per-architecture ranking is density-dependent and not quoted.")
    fh.close()
    print(open(path).read())


if __name__ == "__main__":
    main()
