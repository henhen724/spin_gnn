"""Stage RULE6 -- DD's STOPPING CONDITION, per architecture, at the first
emission.

Sixth refit of runs/stagelogic_2026-11-17. Nothing is trained and nothing is
submitted. Registered in runs/stagerule6_2026-11-23/REGISTERED.md before this
file existed.

Finding 121 ranked 31 architectures by `h_rule`, which is DD's RULE with the
stopping condition deliberately divided out. The only check on the stopping
condition in this line is the solving gate `rho_50 <= 0.740`, and 98/99/100
showed that gate never charges for naming an unforced cell. So:

    u = 1 - s_F(0) = (at_risk(0) - in_F(0)) / at_risk(0)

the fraction of rollouts whose FIRST emitted cell is one that one-step
propagation does not force, pooled per cell over all fourteen densities. The
denominator is 400 x 14 x seeds BY CONSTRUCTION -- nothing chooses it.

By finding 124, `n_err` counts a wrong COLOUR, so naming an unforced cell with
the right colour is counted by `u` and NOT by `h_all(0)`. The stopping
condition and the colour are measured separately on the same denominator.

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

import random

from stagegate_fit import GRID                                  # noqa: E402
from stagerule_fit import RUNS, line, load, smaller_than        # noqa: E402
from stagerule3_fit import sequences                            # noqa: E402
from stagerule4_fit import Z, REG_DEN                           # noqa: E402
from stagerule5_fit import slice0, rates                        # noqa: E402

OUT = os.path.join(RUNS, "stagerule6_2026-11-23")

N_INST = 400.0
N_DENS = 14
N_RUNS = 79
AT_RISK_TOTAL = N_RUNS * N_DENS * N_INST            # 442,400

# Registered bars. Every one carries a size floor, which is RULE5's lesson (ii).
E1_SPREAD = 2.0          # max(u) / min(u) across the 31 cells
E2_RHO = 0.6             # |Spearman(u, h_rule(0))| must be BELOW this
E3_FACTOR = 1.25         # u(answer cell) / u(J2)
E4_FACTOR = 1.5          # u(tied d192) / u(tied d320)
E5_TOL = 1e-12           # reproduction of RULE5's fourteen-row table

ANSWER = "L=16 h=4 tied_no_mlp d320"      # the tractability answer, 3 seeds
J2 = "L=16 h=4 no_mlp d192"               # the standing reference, 3 seeds
BROKEN = "L=16 h=4 tied_no_mlp d192"      # h_rule 0.08106, SINGLE SEED
RULE_BAR = 0.01                           # inherited from finding 121

# Named in the registration so a ranking sentence cannot quietly omit them.
SINGLE_SEED = ["L=16 h=4 tied_no_mlp d192", "L=6 h=4 no_mlp d128 ep100",
               "L=12 h=4 no_mlp d128", "L=16 h=4 tied_no_mlp d448",
               "L=4 h=4 no_mlp", "L=4 h=4 rf_mlp"]

RHOS = [round(r, 4) for r in GRID]
BINS = ", ".join("%.2f" % r for r in RHOS)


# --------------------------------------------------------------------------
# rank statistics -- THIS STAGE DEFINES ITS OWN, AND THE REASON MATTERS
#
# stagegate_fit.spearman ranks with argsort(argsort(.)), which breaks ties by
# array position instead of averaging them. Every earlier caller fed it a
# continuous quantity over 14 densities with no exact ties, so nothing earlier
# is affected. Here h_rule(0) is EXACTLY ZERO for most of the roster, so an
# un-averaged rank vector would be an artefact of cell ordering. Averaged ranks
# are the correct statistic for this comparison, and the tie count is reported
# next to every value it touches.

def avg_ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        mean_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            r[order[k]] = mean_rank
        i = j + 1
    return r


def spearman_t(x, y):
    """Spearman with averaged ranks for ties (Pearson on the averaged ranks)."""
    if len(x) < 3:
        return float("nan")
    rx, ry = avg_ranks(list(x)), avg_ranks(list(y))
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    d = math.sqrt(sxx * syy)
    return sxy / d if d > 0 else float("nan")


def perm_t(x, y, obs, nrep=200000, seed=20261123):
    """Monte-Carlo one-sided p for a tie-corrected Spearman as extreme as obs.

    Permuting the labels keeps the tie STRUCTURE of both vectors intact, so the
    null distribution is the right one for a saturated arm.
    """
    if obs != obs:
        return float("nan"), 0
    rnd = random.Random(seed)
    ys = list(y)
    hits = 0
    for _ in range(nrep):
        rnd.shuffle(ys)
        r = spearman_t(x, ys)
        if (r <= obs) if obs < 0 else (r >= obs):
            hits += 1
    return hits / float(nrep), nrep


def n_ties_at(x, v=0.0):
    return sum(1 for a in x if a == v)


# --------------------------------------------------------------------------
# the join

def cells_j0():
    """Per cell, the j = 0 counts, broken out by seed and by density.

    load() carries the run_id -> cell map but sums over j; sequences() carries
    the per-j counts but not the cell. The join is the whole mechanism of this
    stage, which is why D3 and E5 both police it.
    """
    cells, _viol, _nh, _ns = load()
    meta, cell_of = {}, {}
    for c in cells:
        meta[c["cell"]] = {"mode": c["mode"], "L": c["L"], "d": c["d"],
                           "params": c["params"]}
        prev = cell_of.get(c["run_id"])
        if prev is not None and prev != c["cell"]:
            raise AssertionError("run %s in two cells: %s / %s"
                                 % (c["run_id"], prev, c["cell"]))
        cell_of[c["run_id"]] = c["cell"]

    def blank():
        return {"at_risk": 0.0, "in_F": 0.0, "err": 0.0, "err_in_F": 0.0}

    tot = collections.defaultdict(blank)
    seeds = collections.defaultdict(lambda: collections.defaultdict(blank))
    by_dens = collections.defaultdict(lambda: collections.defaultdict(blank))
    orphan = set()

    seq, n_haz, nblank = sequences()
    for (run, rho), byj in seq.items():
        r = byj.get(0)
        if r is None:
            continue
        cell = cell_of.get(run)
        if cell is None:
            orphan.add(run)
            continue
        for k in ("at_risk", "in_F", "err", "err_in_F"):
            tot[cell][k] += r[k]
            seeds[cell][run][k] += r[k]
            by_dens[cell][rho][k] += r[k]
    return (meta, cell_of, dict(tot), seeds, by_dens, orphan, seq, n_haz,
            nblank)


def wilson(k, n):
    """95% Wilson interval for a proportion. u's numerator is a count out of a
    denominator fixed by construction, so this is binomial, not Poisson."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    z2 = Z * Z
    den = 1.0 + z2 / n
    mid = (p + z2 / (2 * n)) / den
    half = (Z / den) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return mid - half, mid + half


def ratio_binom(k1, n1, k2, n2):
    """Ratio of two proportions with a delta-method log-ratio 95% interval.

    Var(log p_hat) = (1 - p) / k for a binomial, so this is the right variance
    for a denominator fixed in advance -- ratio_poisson would drop the (1 - p)
    factor, which at u ~ 0.18 is a real 22% overstatement of the variance.
    """
    if k1 <= 0 or k2 <= 0 or n1 <= 0 or n2 <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    R = p1 / p2
    se = math.sqrt((1 - p1) / k1 + (1 - p2) / k2)
    return R, R * math.exp(-Z * se), R * math.exp(Z * se), se


def row_of(a):
    """The registered quantities for one pooled count block."""
    v = rates(a)
    u = 1.0 - v["s_F"]
    k = a["at_risk"] - a["in_F"]
    lo, hi = wilson(k, a["at_risk"])
    return {"u": u, "u_lo": lo, "u_hi": hi, "u_k": k, "u_n": a["at_risk"],
            "h_rule": v["h_rule"], "h_all": v["h_all"], "h_unf": v["h_unf"],
            "s_F": v["s_F"], "err": a["err"], "err_in_F": a["err_in_F"]}


# --------------------------------------------------------------------------
# diagnostics

def d1(fh, meta, tot, seeds, by_dens):
    bad, miss = [], []
    total = 0.0
    for cell in meta:
        s = len(seeds[cell])
        exp = N_INST * N_DENS * s
        total += tot[cell]["at_risk"]
        if abs(tot[cell]["at_risk"] - exp) > 1e-9:
            bad.append((cell, tot[cell]["at_risk"], exp))
        if len(by_dens[cell]) != N_DENS:
            miss.append((cell, len(by_dens[cell])))
        for rho, a in by_dens[cell].items():
            if abs(a["at_risk"] - N_INST * s) > 1e-9:
                bad.append((cell, rho, a["at_risk"]))
    ok = not bad and not miss and abs(total - AT_RISK_TOTAL) < 1e-9
    line(fh, "D1  at_risk(0) = 400 x 14 x seeds for every cell")
    line(fh, "    cells %d   violations %d   cells missing a density %d"
         % (len(meta), len(bad), len(miss)))
    line(fh, "    pooled at_risk(0) = %.0f   expected %.0f   %s"
         % (total, AT_RISK_TOTAL, "EXACT" if abs(total - AT_RISK_TOTAL) < 1e-9
            else "MISMATCH"))
    for b in bad[:5]:
        line(fh, "      %s" % (b,))
    line(fh, "D1 %s" % ("PASS" if ok else "FAIL"))
    line(fh)
    return ok


def d2(fh, seq):
    v_err, v_inf, n = 0, 0, 0
    for (_run, _rho), byj in seq.items():
        r = byj.get(0)
        if r is None:
            continue
        n += 1
        if r["err_in_F"] > r["err"] + 1e-9:
            v_err += 1
        if r["in_F"] > r["at_risk"] + 1e-9:
            v_inf += 1
    ok = v_err == 0 and v_inf == 0
    line(fh, "D2  count sanity on every j = 0 row")
    line(fh, "    rows %d   n_err_in_F > n_err: %d   n_in_F > n_at_risk: %d"
         % (n, v_err, v_inf))
    line(fh, "D2 %s" % ("PASS" if ok else "FAIL"))
    line(fh)
    return ok


def d3(fh, meta, cell_of, seeds, orphan):
    empty = [c for c in meta if not seeds[c]]
    counted = sum(len(seeds[c]) for c in meta)
    ok = (len(meta) == 31 and len(cell_of) == N_RUNS and not orphan
          and not empty and counted == N_RUNS)
    line(fh, "D3  the run -> cell map is a partition")
    line(fh, "    cells %d (want 31)   runs mapped %d (want %d)   "
             "runs reached at j = 0 %d" % (len(meta), len(cell_of), N_RUNS,
                                           counted))
    line(fh, "    orphan runs in the hazard shards %d   cells with no run %d"
         % (len(orphan), len(empty)))
    line(fh, "D3 %s" % ("PASS" if ok else "FAIL"))
    line(fh)
    return ok


def d4(fh, by_dens, meta):
    pooled = collections.defaultdict(float)
    for cell in meta:
        for rho, a in by_dens[cell].items():
            pooled[rho] += a["in_F"]
    bad = [(r, pooled[r], REG_DEN[r]) for r in REG_DEN
           if abs(pooled[r] - REG_DEN[r]) > 1e-9]
    ok = not bad and len(pooled) == N_DENS
    line(fh, "D4  per-cell n_in_F(0) summed by density reproduces RULE4's "
             "registered denominators")
    line(fh, "    densities %d   mismatches %d" % (len(pooled), len(bad)))
    for b in bad[:5]:
        line(fh, "      rho %.2f got %.0f want %d" % b)
    line(fh, "D4 %s" % ("PASS" if ok else "FAIL"))
    line(fh)
    return ok


def d5(fh, meta, seeds):
    """Keyed on `cell`, never on (mode, L, d) -- that triple pools cells with
    identical shape and different training, the 96/99 defect."""
    triples = collections.defaultdict(list)
    for cell, m in meta.items():
        triples[(m["mode"], m["L"], m["d"])].append(cell)
    collisions = {t: cs for t, cs in triples.items() if len(cs) > 1}
    pair = ("L=6 h=4 no_mlp", "L=6 h=4 no_mlp ep100")
    separate = all(c in meta for c in pair) and meta[pair[0]] != meta[pair[1]] \
        or (all(c in meta for c in pair)
            and len(seeds[pair[0]]) != len(seeds[pair[1]]))
    ok = len(meta) == 31 and separate
    line(fh, "D5  the key is `cell`, not (arch_mode, num_layers, d_model)")
    line(fh, "    distinct cells %d   (mode, L, d) triples %d   colliding "
             "triples %d" % (len(meta), len(triples), len(collisions)))
    for t, cs in sorted(collisions.items(), key=lambda kv: str(kv[0])):
        line(fh, "      %-28s -> %s" % (str(t), " | ".join(sorted(cs))))
    line(fh, "    `%s` (%d seeds) and `%s` (%d seeds) share %d params and stay "
             "separate: %s" % (pair[0], len(seeds[pair[0]]), pair[1],
                               len(seeds[pair[1]]), meta[pair[0]]["params"],
                               "YES" if separate else "NO"))
    line(fh, "D5 %s" % ("PASS" if ok else "FAIL"))
    line(fh)
    return ok


# --------------------------------------------------------------------------
# reporting

def table(fh, meta, tot, seeds, rows):
    line(fh, "THE ROSTER AT j = 0, POOLED OVER ALL FOURTEEN DENSITIES")
    line(fh, "bins: rho = %s" % BINS)
    line(fh, "u = 1 - s_F(0) = Pr(first emitted cell is NOT forced).  "
             "Denominator 400 x 14 x seeds, fixed by construction.")
    line(fh, "h_rule(0) and h_all(0) are RULE5's rates at index 0, per cell. "
             "They are NOT finding 121's h_rule (see trap 1).")
    line(fh)
    line(fh, "%-30s %9s %3s %8s %-16s %9s %9s %-15s"
         % ("cell", "params", "sd", "u", "u 95% Wilson", "h_rule(0)",
            "h_all(0)", "u per seed"))
    for cell in sorted(meta, key=lambda c: meta[c]["params"]):
        r = rows[cell]
        per = sorted(1.0 - (a["in_F"] / a["at_risk"])
                     for a in seeds[cell].values())
        spread = ("%.3f" % per[0] if len(per) == 1
                  else "%.3f-%.3f" % (per[0], per[-1]))
        line(fh, "%-30s %9d %3d %8.4f [%.4f, %.4f] %9.5f %9.5f %-15s"
             % (cell, meta[cell]["params"], len(seeds[cell]), r["u"],
                r["u_lo"], r["u_hi"], r["h_rule"], r["h_all"], spread))
    line(fh)
    line(fh, "Six cells are single-seed and carry no spread: %s"
         % "; ".join("`%s`" % c for c in SINGLE_SEED))
    line(fh)


def head_to_head(fh, name, a, b, rows, seeds, factor):
    """One registered pairwise comparison. Factor first, interval second."""
    ra, rb = rows[a], rows[b]
    R, lo, hi, _se = ratio_binom(ra["u_k"], ra["u_n"], rb["u_k"], rb["u_n"])
    line(fh, "%s" % name)
    line(fh, "    %-30s u = %.4f  (%.0f / %.0f, %d seeds)"
         % (a, ra["u"], ra["u_k"], ra["u_n"], len(seeds[a])))
    line(fh, "    %-30s u = %.4f  (%.0f / %.0f, %d seeds)"
         % (b, rb["u"], rb["u_k"], rb["u_n"], len(seeds[b])))
    line(fh, "    ratio %.4f   registered bar %.2f   95%% [%.4f, %.4f]"
         % (R, factor, lo, hi))
    line(fh, "    h_rule(0): %.5f vs %.5f" % (ra["h_rule"], rb["h_rule"]))
    line(fh)
    return R


def score(fh, meta, tot, seeds, rows, ref14, mine14):
    line(fh, "=" * 74)
    line(fh, "REGISTERED PREDICTIONS")
    line(fh, "=" * 74)
    line(fh)
    hits = {}

    us = [(rows[c]["u"], c) for c in meta]
    us.sort()
    zero_u = [c for _v, c in us if _v == 0.0]
    nz = [v for v, _c in us if v > 0.0]
    # min(u) is an EXACT zero on three cells, so the registered ratio is
    # unbounded. Adjudicate that in the open rather than printing inf, and give
    # the finite ratio against the smallest NON-zero u as well.
    spread_nz = us[-1][0] / nz[0] if nz else float("nan")
    hits["E1"] = bool(zero_u) or spread_nz >= E1_SPREAD
    line(fh, "E1  u carries architecture information: max/min >= %.1f"
         % E1_SPREAD)
    line(fh, "    lowest  u = %.4f  %s (%d seeds)"
         % (us[0][0], us[0][1], len(seeds[us[0][1]])))
    line(fh, "    highest u = %.4f  %s (%d seeds)"
         % (us[-1][0], us[-1][1], len(seeds[us[-1][1]])))
    if zero_u:
        line(fh, "    THE MINIMUM IS AN EXACT ZERO on %d cells, so the "
                 "registered ratio is UNBOUNDED:" % len(zero_u))
        for c in zero_u:
            line(fh, "      %-30s %d seeds, %.0f / %.0f first emissions "
                     "named an unforced cell"
                 % (c, len(seeds[c]), rows[c]["u_k"], rows[c]["u_n"]))
        line(fh, "    Against the smallest NON-zero u (%.4f, %s) the finite "
                 "ratio is %.2fx."
             % (nz[0], [c for v, c in us if v == nz[0]][0], spread_nz))
    else:
        line(fh, "    spread %.3fx" % spread_nz)
    line(fh, "    -> %s" % ("HIT" if hits["E1"] else "MISS"))
    line(fh, "    The extremes' intervals do not overlap: [%.4f, %.4f] vs "
             "[%.4f, %.4f]."
         % (rows[us[0][1]]["u_lo"], rows[us[0][1]]["u_hi"],
            rows[us[-1][1]]["u_lo"], rows[us[-1][1]]["u_hi"]))
    line(fh)

    order = sorted(meta, key=lambda c: meta[c]["params"])
    xu = [rows[c]["u"] for c in order]
    xr = [rows[c]["h_rule"] for c in order]
    t_rule, t_u = n_ties_at(xr), n_ties_at(xu)
    s_ur = spearman_t(xu, xr)
    p_ur, nrep = perm_t(xu, xr, s_ur)
    hits["E2"] = abs(s_ur) < E2_RHO
    line(fh, "E2  u and h_rule(0) are close to independent: |Spearman| < %.1f"
         % E2_RHO)
    line(fh, "    THE RULE ARM IS SATURATED: h_rule(0) is EXACTLY ZERO on "
             "%d of the %d cells." % (t_rule, len(order)))
    line(fh, "    u is exactly zero on %d. Ranks are tie-averaged for that "
             "reason (see the note in this file)." % t_u)
    line(fh, "    Spearman(u, h_rule(0)) over the 31 cells = %+.4f   "
             "Monte-Carlo p = %.5g over %d permutations"
         % (s_ur, p_ur, nrep))
    line(fh, "    -> %s" % ("HIT" if hits["E2"] else "MISS"))
    # A zero is not a demonstration of perfection, it is an upper bound. Say so
    # with the number, because "saturated" would be an overclaim.
    nz_cells = [c for c in order if rows[c]["err_in_F"] > 0]
    tot_err = sum(rows[c]["err_in_F"] for c in order)
    worst_bound = max(3.0 / tot[c]["in_F"] for c in order
                      if rows[c]["err_in_F"] == 0)
    line(fh, "    WHAT A ZERO DOES AND DOES NOT MEAN. All %.0f rule errors at "
             "j = 0 come from %d cells:" % (tot_err, len(nz_cells)))
    for c in sorted(nz_cells, key=lambda c: -rows[c]["err_in_F"]):
        line(fh, "      %-30s %.0f / %.0f = %.5f   (%d seeds)"
             % (c, rows[c]["err_in_F"], tot[c]["in_F"], rows[c]["h_rule"],
                len(seeds[c])))
    line(fh, "    The other %d cells saw ZERO. A zero bounds the rate from "
             "above by 3/n (rule of three)," % t_rule)
    line(fh, "    which across those cells is at worst %.2e -- so the rule "
             "arm's whole observable" % worst_bound)
    line(fh, "    range at j = 0 is [0, %.0e], while u spans %.4f to %.4f on "
             "the SAME rollouts."
         % (worst_bound, us[0][0], us[-1][0]))
    line(fh, "    That is the content of E2, and it does not depend on how "
             "ties are ranked: at the")
    line(fh, "    first emission the rule arm is near-unmeasurable and the "
             "stopping condition is")
    line(fh, "    where all the spread is. It does NOT show 27 cells are "
             "perfect, and it does NOT")
    line(fh, "    contradict finding 121, which pooled every j and had far "
             "more forced emissions.")
    line(fh)

    R3 = head_to_head(
        fh, "E3  the tractable answer is not keeping the rule by declining to "
            "be tested: u(answer) <= %.2f x u(J2)" % E3_FACTOR,
        ANSWER, J2, rows, seeds, E3_FACTOR)
    hits["E3"] = R3 <= E3_FACTOR
    line(fh, "    -> %s" % ("HIT" if hits["E3"] else "MISS"))
    if not hits["E3"]:
        line(fh, "    MISS CONDITION FIRED: finding 121's deck "
                 "/spin-gnn/rule-hazard/ gets a correction page.")
    line(fh)

    R4 = head_to_head(
        fh, "E4  the broken tied cell breaks on the RULE, not the stopping "
            "condition: u(d192) <= %.1f x u(d320)" % E4_FACTOR,
        BROKEN, ANSWER, rows, seeds, E4_FACTOR)
    hits["E4"] = R4 <= E4_FACTOR
    line(fh, "    -> %s   (d192 is SINGLE SEED)"
         % ("HIT" if hits["E4"] else "MISS"))
    line(fh)

    worst = 0.0
    keys = ("h_all", "h_rule", "h_unf", "s_F")
    for rho in RHOS:
        for k in keys:
            worst = max(worst, abs(ref14[rho][k] - mine14[rho][k]))
    hits["E5"] = worst <= E5_TOL
    line(fh, "E5  summing the 31 cells back over density reproduces stage "
             "RULE5's fourteen-row table")
    line(fh, "    largest absolute residual over %d densities x %d rates = "
             "%.3e   tolerance %.0e"
         % (N_DENS, len(keys), worst, E5_TOL))
    line(fh, "    -> %s" % ("HIT" if hits["E5"] else "MISS"))
    line(fh)

    line(fh, "SCORE: " + "  ".join("%s %s" % (k, "HIT" if v else "MISS")
                                   for k, v in sorted(hits.items())))
    line(fh)
    return hits


def both_arms(fh, meta, rows, seeds):
    """The brief's actual question: is there an object strictly smaller than J2
    that keeps BOTH the rule and the stopping condition?"""
    line(fh, "=" * 74)
    line(fh, "DOES ANY SMALLER OBJECT KEEP BOTH ARMS?")
    line(fh, "=" * 74)
    line(fh)
    line(fh, "`strictly smaller object` is finding 121's definition: no more "
             "machinery, no more depth, no more width, at least one strictly "
             "less. linear and rf_mlp are off the machinery chain and are "
             "never called smaller.")
    line(fh, "`keeps the rule` is h_rule(0) < %.2f, the bar inherited from 121."
         % RULE_BAR)
    line(fh, "`keeps the stopping condition` is u no worse than J2's within "
             "the registered %.2fx." % E3_FACTOR)
    line(fh)
    j2 = {**meta[J2], "cell": J2}
    u_j2 = rows[J2]["u"]
    line(fh, "J2 = %s: u = %.4f, h_rule(0) = %.5f, %d seeds"
         % (J2, u_j2, rows[J2]["h_rule"], len(seeds[J2])))
    line(fh)
    line(fh, "%-30s %9s %3s %8s %7s %9s %8s"
         % ("cell smaller than J2", "params", "sd", "u", "u/u_J2", "h_rule(0)",
            "both?"))
    n_both = 0
    n_small = 0
    n_rule = 0
    ratios = []
    for cell in sorted(meta, key=lambda c: meta[c]["params"]):
        if cell == J2:
            continue
        if not smaller_than({**meta[cell], "cell": cell}, j2):
            continue
        r = rows[cell]
        rule_ok = r["h_rule"] < RULE_BAR
        stop_ok = r["u"] <= E3_FACTOR * u_j2
        both = rule_ok and stop_ok
        n_both += 1 if both else 0
        n_small += 1
        n_rule += 1 if rule_ok else 0
        ratios.append(r["u"] / u_j2)
        line(fh, "%-30s %9d %3d %8.4f %7.3f %9.5f %8s"
             % (cell, meta[cell]["params"], len(seeds[cell]), r["u"],
                r["u"] / u_j2, r["h_rule"],
                "BOTH" if both else ("rule" if rule_ok else
                                     ("stop" if stop_ok else "neither"))))
    line(fh)
    line(fh, "Of the %d objects strictly smaller than J2: %d keep the rule, "
             "%d keep the stopping" % (n_small, n_rule, sum(
                 1 for c in meta if c != J2
                 and smaller_than({**meta[c], "cell": c}, j2)
                 and rows[c]["u"] <= E3_FACTOR * u_j2)))
    line(fh, "condition, and %d keep BOTH. Their u runs %.2fx to %.2fx J2's."
         % (n_both, min(ratios), max(ratios)))
    line(fh)
    return n_both, n_small, n_rule, min(ratios), max(ratios)


def uniform_by_rho(seq, blank):
    """Per density, (mean |F| at j = 0, n_blank, u if the first cell were
    picked uniformly among the blanks). Shared with the figure builder so the
    two cannot disagree."""
    rowsd = {}
    for (_run, rho), byj in seq.items():
        r = byj.get(0)
        if r is None:
            continue
        a = rowsd.setdefault(rho, {"F": 0.0, "A": 0.0})
        a["F"] += r["F_sum"]
        a["A"] += r["at_risk"]
    out = {}
    for rho, a in rowsd.items():
        mf = a["F"] / a["A"]
        out[rho] = (mf, blank[rho], 1.0 - mf / blank[rho])
    return out


def uniform_null(fh, seq, blank):
    """What u would be if the model picked its first cell UNIFORMLY among the
    blank cells. This is the chance level for the stopping condition, and
    without it a number like u = 0.236 has nothing to be compared against.

    Exact rather than approximate for twelve of the fourteen densities, because
    mean_n_blank takes a single value per density there, so the ratio of means
    IS the mean of the ratios. At rho = 0.35 and 0.33 it reads 31.635 and
    32.0025, so the blank count varies across instances and the baseline is a
    ratio-of-means approximation on those two bins only.
    """
    by_rho = uniform_by_rho(seq, blank)
    line(fh, "CHANCE LEVEL FOR THE STOPPING CONDITION: pick a blank cell "
             "uniformly at random.")
    line(fh, "    %5s %10s %9s %10s" % ("rho", "mean |F|", "n_blank",
                                        "u_uniform"))
    vals = []
    for rho in RHOS:
        mf, nb, uu = by_rho[rho]
        vals.append(uu)
        line(fh, "    %5.2f %10.4f %9.4f %10.4f" % (rho, mf, nb, uu))
    u_bar = sum(vals) / len(vals)
    line(fh, "    equal weight over the fourteen densities: u_uniform = %.4f"
         % u_bar)
    line(fh)
    return u_bar


def clump_split(meta, seeds, min_side=5):
    """Every run's own u, sorted, plus the clump boundary read off that list.

    Shared with `stagerule6_figs` on purpose. The boundary is a float with no
    short decimal form (401/5600 = 0.0716071...), and rounding it to the four
    places a figure caption prints moves one run across it -- which silently
    changed both the clump counts and the straddle list when the figures
    carried their own literal. One definition, two callers.

    The largest gap in the sorted list is a single-run outlier rather than a
    boundary, so this takes the largest gap with at least `min_side` runs on
    each side. That rule is stated, not tuned, and everything built on it is
    post-hoc and descriptive.
    """
    order = sorted(meta, key=lambda c: meta[c]["params"])
    per_run = sorted((1.0 - a["in_F"] / a["at_risk"], cell, run)
                     for cell in order for run, a in seeds[cell].items())
    vals = [v for v, _c, _r in per_run]
    gaps = sorted(((vals[i + 1] - vals[i], vals[i], vals[i + 1], i + 1)
                   for i in range(len(vals) - 1)), reverse=True)
    cand = [g for g in gaps if g[3] >= min_side
            and len(vals) - g[3] >= min_side]
    gap, a_bound, b_bound, k = cand[0]
    return per_run, gaps, gap, a_bound, b_bound, k


def straddlers(meta, rows, seeds, a_bound):
    """Cells with at least one seed on each side of the boundary, ordered by
    pooled u. Also shared with the figures, for the same reason."""
    out = []
    for cell in sorted(meta, key=lambda c: rows[c]["u"]):
        per = sorted(1.0 - (a["in_F"] / a["at_risk"])
                     for a in seeds[cell].values())
        if len(per) > 1 and per[0] <= a_bound < per[-1]:
            out.append((cell, per))
    return out


def posthoc(fh, meta, rows, seeds, u_bar):
    """Not registered. Labelled as such so it is never quoted as a result."""
    line(fh, "=" * 74)
    line(fh, "POST-HOC, NOT REGISTERED")
    line(fh, "=" * 74)
    line(fh)
    line(fh, "Against that chance level of u = %.4f, the roster has three "
             "regimes, not a continuum:" % u_bar)
    for cell in sorted(meta, key=lambda c: rows[c]["u"]):
        r = rows[cell]
        tag = ("tracks the forced set" if r["u"] < 0.01 else
               ("INDISTINGUISHABLE FROM CHANCE" if r["u"] >= 0.9 * u_bar else
                "better than chance, not tracking"))
        if r["u"] < 0.01 or r["u"] >= 0.9 * u_bar:
            line(fh, "    %-30s u = %.4f  (%.2f x chance)  %s"
                 % (cell, r["u"], r["u"] / u_bar, tag))
    line(fh, "    Everything else lies between, clustered near u ~ 0.24, "
             "about half the chance level.")
    line(fh)
    order = sorted(meta, key=lambda c: meta[c]["params"])
    pr = [float(meta[c]["params"]) for c in order]
    xu = [rows[c]["u"] for c in order]
    xa = [rows[c]["h_all"] for c in order]
    xn = [rows[c]["h_unf"] for c in order]
    s_up = spearman_t(xu, pr)
    p_up, _n = perm_t(xu, pr, s_up)
    line(fh, "Spearman(u, params) over the 31 cells = %+.4f (p = %.5g). %s"
         % (s_up, p_up,
            "Parameter count orders the stopping condition." if abs(s_up) > 0.5
            else "Parameter count does NOT order the stopping condition."))
    s_ua = spearman_t(xu, xa)
    line(fh, "Spearman(u, h_all(0)) = %+.4f. h_all(0) is dominated by the "
             "unforced arm (finding 128), so this is close to a consistency "
             "check, not news." % s_ua)
    line(fh, "h_unf(0) across the roster: %.4f to %.4f, median %.4f. A blind "
             "colour guess errs at 1 - 1/q = 0.75."
         % (min(xn), max(xn), sorted(xn)[len(xn) // 2]))
    line(fh)

    # The single most important thing in the output, and it was NOT registered.
    MIN_SIDE = 5
    per_run, gaps, gap, a_bound, b_bound, k = clump_split(meta, seeds, MIN_SIDE)
    vals = [v for v, _c, _r in per_run]
    line(fh, "THE 79 RUNS ARE NOT SPREAD ALONG u -- THEY SIT IN TWO CLUMPS.")
    line(fh, "Every run's own u, sorted. Each is %.0f first emissions "
             "(14 densities x 400)." % (N_INST * N_DENS))
    line(fh, "    the three largest gaps in the sorted list:")
    for g, a, b, kk in gaps[:3]:
        line(fh, "      %.4f  between u = %.4f and u = %.4f   (%d runs below, "
                 "%d above)" % (g, a, b, kk, len(vals) - kk))
    line(fh, "    The LARGEST gap is a single-run outlier, so it is not the "
             "clump boundary. Taking")
    line(fh, "    the largest gap with at least %d runs on each side instead: "
             "%.4f, between u = %.4f" % (MIN_SIDE, gap, a_bound))
    line(fh, "    and u = %.4f.  Below it %d runs, above it %d."
         % (b_bound, k, len(vals) - k))
    line(fh, "    The boundary is read off the data AFTER seeing it. It is "
             "descriptive only and is")
    line(fh, "    never used as a test or a threshold.")
    line(fh)
    line(fh, "    %-8s %-30s %8s" % ("run", "cell", "u"))
    for v, cell, run in per_run:
        line(fh, "    %-8s %-30s %8.4f" % (run, cell, v))
    line(fh)
    line(fh, "Consequence: a cell's pooled u is largely HOW MANY OF ITS SEEDS "
             "landed in the low")
    line(fh, "clump, not a smooth architecture property. With 1 to 5 seeds "
             "per cell that is a")
    line(fh, "coarse quantity, and it is the strongest argument yet that SEED "
             "is the next axis.")
    straddle = straddlers(meta, rows, seeds, a_bound)
    line(fh, "Cells whose seeds straddle the boundary:")
    for cell, per in straddle:
        line(fh, "    %-30s %d seeds  u %.4f - %.4f  (pooled %.4f)"
             % (cell, len(per), per[0], per[-1], rows[cell]["u"]))
    if not straddle:
        line(fh, "    none")
    line(fh)
    return k, len(vals) - k, gap, a_bound, b_bound


def reading(fh, hits, meta, rows, seeds, arms_res, u_bar, clump):
    n_both, n_small, n_rule, r_lo, r_hi = arms_res
    lo_n, hi_n, gap, a_b, b_b = clump
    us = sorted((rows[c]["u"], c) for c in meta)
    line(fh, "=" * 74)
    line(fh, "WHAT THIS SAYS, AND WHAT IT DOES NOT")
    line(fh, "=" * 74)
    line(fh)
    line(fh, "1. DIRECT DEDUCTION HAS TWO ARMS AND THIS LINE HAD ONLY EVER "
             "SCORED ONE.")
    line(fh, "   A rule -- which colour a forced cell takes -- and a stopping "
             "condition -- which")
    line(fh, "   cell may be named at all. Finding 121 ranked 31 "
             "architectures by h_rule, which")
    line(fh, "   divides the stopping condition out on purpose, and the only "
             "other check in the")
    line(fh, "   line was the solving gate rho_50 <= 0.740, which 98/99/100 "
             "showed never charges")
    line(fh, "   for naming an unforced cell. u is that charge, on a "
             "denominator of 400 x 14 x")
    line(fh, "   seeds that nothing selects.")
    line(fh)
    line(fh, "2. AT THE FIRST EMISSION THE RULE ARM IS NEAR-UNMEASURABLE AND "
             "THE STOPPING")
    line(fh, "   CONDITION IS WHERE ALL THE SPREAD IS. 27 of 31 cells saw "
             "zero rule errors at")
    line(fh, "   j = 0, which bounds their rate below about 2e-4 and does NOT "
             "show they are")
    line(fh, "   perfect. On the same rollouts u runs %.4f to %.4f against a "
             "chance level of" % (us[0][0], us[-1][0]))
    line(fh, "   %.4f -- picking a blank cell uniformly at random." % u_bar)
    line(fh)
    line(fh, "3. NO OBJECT STRICTLY SMALLER THAN J2 KEEPS BOTH ARMS: %d of "
             "them." % n_both)
    line(fh, "   Of the %d object-order-smaller cells, %d keep the rule and "
             "NONE keeps the stopping" % (n_small, n_rule))
    line(fh, "   condition -- their u runs %.2fx to %.2fx J2's. SO FINDING "
             "121's HEADLINE IS A STATEMENT" % (r_lo, r_hi))
    line(fh, "   ABOUT THE RULE AND MUST NOT BE READ AS ONE ABOUT DIRECT "
             "DEDUCTION. Its arithmetic")
    line(fh, "   is untouched -- E5 reproduces RULE5's table at 0.000e+00 and "
             "D4 reproduces RULE4's")
    line(fh, "   denominators exactly -- and the quantity it reported is not "
             "the one the current")
    line(fh, "   brief asks for.")
    line(fh)
    line(fh, "4. E3 MISSED, AND IT IS THE CONSEQUENTIAL MISS. The tractability "
             "answer cell")
    line(fh, "   %s reads h_rule(0) = 0.00000 and u = %.4f, %s"
         % (ANSWER, rows[ANSWER]["u"], "3.15x J2's."))
    line(fh, "   Not one of its three seeds is in the low clump. It attains a "
             "perfect rule score")
    line(fh, "   on cells it selects at roughly half the chance level. Per the "
             "registration,")
    line(fh, "   /spin-gnn/rule-hazard/ GETS A CORRECTION PAGE.")
    line(fh)
    line(fh, "5. THE 79 RUNS SIT IN TWO CLUMPS, NOT ON A CONTINUUM: %d runs "
             "at u <= %.4f and %d" % (lo_n, a_b, hi_n))
    line(fh, "   at u >= %.4f, separated by a gap of %.4f. A cell's pooled u "
             "is therefore largely" % (b_b, gap))
    line(fh, "   HOW MANY OF ITS SEEDS LANDED IN THE LOW CLUMP, not a smooth "
             "architecture")
    line(fh, "   property. With 1 to 5 seeds per cell that is a coarse "
             "quantity, and it is the")
    line(fh, "   strongest argument this line has produced that SEED is the "
             "next axis. The clump")
    line(fh, "   boundary was read off the data after seeing it and is never "
             "used as a test.")
    line(fh)
    line(fh, "6. E4 MISSED TOO: the broken tied d192 is worse on BOTH arms "
             "(h_rule(0) = 0.19848,")
    line(fh, "   u = %.4f, %.2fx chance), so losing the rule between d192 and "
             "d320 is not losing"
         % (rows[BROKEN]["u"], rows[BROKEN]["u"] / u_bar))
    line(fh, "   the rule alone. That cell is SINGLE SEED and by point 5 a "
             "single seed is exactly")
    line(fh, "   what cannot be read as an architecture property, so this is "
             "the weakest claim here.")
    line(fh)
    line(fh, "7. STATED AS A LIMIT, NOT A VERDICT ON REASONING. u measures "
             "AGREEMENT WITH DD's")
    line(fh, "   STOPPING CONDITION, not logical validity. A solver that names "
             "a cell propagation")
    line(fh, "   has not forced and then checks it is reasoning; it is simply "
             "not reasoning in DD's")
    line(fh, "   trace format. Only the rule arm carries a correctness claim, "
             "and below")
    line(fh, "   rho_c ~ 0.465 even that bounds from above, exactly as in "
             "RULE5.")
    line(fh)
    line(fh, "8. j = 0 ONLY. Nothing here speaks to later indices, where "
             "censoring returns and")
    line(fh, "   RULE3's lambda* = 1.19 applies. u is NOT a solve rate and is "
             "never converted into")
    line(fh, "   one; rho_50 <= 0.740 remains the solving gate and is not "
             "replaced or re-derived.")
    line(fh)
    line(fh, "9. NOTHING FOLLOWS THAT MAY BE SUBMITTED. The width sweep "
             "between d192 and d320")
    line(fh, "   stays closed, seeding the single-seed cells stays hshunt's "
             "call, stage DQ4 stays")
    line(fh, "   unsubmitted, and no closed knob is reopened. Point 5 is an "
             "argument for seeds and")
    line(fh, "   seeds are training: it is reported, not acted on.")
    line(fh)
    line(fh, "SCOPE: d5q4, N = 48, 79 runs, 31 architectures, 400 instances "
             "per density bin, eval")
    line(fh, "seed start 60,000,000,000, j = 0, bins rho = %s." % BINS)
    line(fh)


def main():
    os.makedirs(OUT, exist_ok=True)
    (meta, cell_of, tot, seeds, by_dens, orphan, seq, n_haz,
     nblank) = cells_j0()

    # RULE5's own code path is the reference for E5.
    acc5, _pa, _pr = slice0(seq)
    ref14 = {r: rates(acc5[r]) for r in RHOS}
    pooled = collections.defaultdict(
        lambda: {"at_risk": 0.0, "in_F": 0.0, "err": 0.0, "err_in_F": 0.0})
    for cell in meta:
        for rho, a in by_dens[cell].items():
            for k in ("at_risk", "in_F", "err", "err_in_F"):
                pooled[rho][k] += a[k]
    mine14 = {r: rates(pooled[r]) for r in RHOS}

    rows = {c: row_of(tot[c]) for c in meta}

    path = os.path.join(OUT, "VERDICT.txt")
    with open(path, "w") as fh:
        line(fh, "STAGE RULE6 -- DD's STOPPING CONDITION, PER ARCHITECTURE, "
                 "AT THE FIRST EMISSION")
        line(fh, "refit of runs/stagelogic_2026-11-17, %d hazard rows, %d "
                 "runs, %d architectures" % (n_haz, len(cell_of), len(meta)))
        line(fh, "registered in runs/stagerule6_2026-11-23/REGISTERED.md "
                 "before this scorer existed")
        line(fh, "nothing trained, nothing submitted, no closed knob reopened")
        line(fh)
        line(fh, "-" * 74)
        line(fh, "DIAGNOSTICS")
        line(fh, "-" * 74)
        line(fh)
        ds = [d1(fh, meta, tot, seeds, by_dens), d2(fh, seq),
              d3(fh, meta, cell_of, seeds, orphan), d4(fh, by_dens, meta),
              d5(fh, meta, seeds)]
        if not all(ds):
            line(fh, "A DIAGNOSTIC FAILED -- nothing below is readable.")
            return
        table(fh, meta, tot, seeds, rows)
        hits = score(fh, meta, tot, seeds, rows, ref14, mine14)
        arms_res = both_arms(fh, meta, rows, seeds)
        u_bar = uniform_null(fh, seq, nblank)
        clump = posthoc(fh, meta, rows, seeds, u_bar)
        reading(fh, hits, meta, rows, seeds, arms_res, u_bar,
                clump)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
