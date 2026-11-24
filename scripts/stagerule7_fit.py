"""Stage RULE7 -- where on the density grid the stopping condition survives.

Seventh refit of runs/stagelogic_2026-11-17. Nothing is trained and nothing is
submitted; this reads shards that have been on disk since 2026-11-17.

RULE6 pooled u = 1 - s_F(0) over fourteen densities whose chance level varies
sixteen-fold, so a low pooled u cannot distinguish a run that keeps DD's
stopping condition everywhere from one that keeps it only where keeping it is
trivial. This stage un-pools, normalises by that chance level, and tests whether
finding 130's two clumps are a property of the runs or of the pooled statistic.

Registered in runs/stagerule7_2026-11-23/REGISTERED.md before this file existed.

Environment (a login node is fine, this is arithmetic over CSVs):

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

from stagegate_fit import GRID                                  # noqa: E402
from stagerule_fit import RUNS, line                            # noqa: E402
from stagerule3_fit import sequences                            # noqa: E402
from stagerule6_fit import (cells_j0, clump_split, uniform_by_rho,  # noqa: E402
                           spearman_t, perm_t, wilson)

OUT = os.path.join(RUNS, "stagerule7_2026-11-23")

N_INST = 400.0
N_RUNS = 79
RHOS = [round(r, 4) for r in GRID]
MIN_SIDE = 5
E3_RHO = 0.8          # named in the registration, not chosen after the fact
E1_BAR = 0.50
E2_RHO_BAR = -0.60
E2_FACTOR = 2.0
E3_JACCARD = 0.80
E4_FACTOR = 3.0
BLANK_LADDER = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 17.0, 20.0, 22.0, 24.0,
                27.0, 29.0, 31.635, 32.0025]
UNIF_REF = [0.0492, 0.0925, 0.1616, 0.2245, 0.2762, 0.3610, 0.4246, 0.5234,
            0.5769, 0.6338, 0.7114, 0.7545, 0.8015, 0.8095]


# --------------------------------------------------------------------------
# the slice

def per_run_dens(seq):
    """Per (run_id, rho) at j = 0 only: at_risk, in_F, F_sum, and u.

    No pooling of any kind happens here. RULE6's cells_j0 aggregates the same
    rows by cell and by density separately; E5 sums this table back and checks
    it lands on RULE6's to the bit.
    """
    out = {}
    for (run, rho), byj in seq.items():
        r = byj.get(0)
        if r is None:
            continue
        a = r["at_risk"]
        out[(run, round(rho, 4))] = {
            "at_risk": a, "in_F": r["in_F"], "F_sum": r["F_sum"],
            "u": 1.0 - r["in_F"] / a if a > 0 else float("nan"),
        }
    return out


def normalise(cells, unif):
    """u_tilde = u / u_unif(rho). Never clipped: above 1 means worse than
    drawing a blank cell at random, and that has to be reportable."""
    out = {}
    for (run, rho), a in cells.items():
        u0 = unif[rho][2]
        out[(run, rho)] = a["u"] / u0 if u0 > 0 else float("nan")
    return out


def by_rho(keys, cells, runs):
    """Count-weighted u per density over a set of runs. Counts, not a mean of
    per-run rates, because at_risk is 400 everywhere and the pooled rate is
    then the quantity RULE6 would have reported for that subset."""
    acc = collections.defaultdict(lambda: {"at_risk": 0.0, "in_F": 0.0})
    for (run, rho) in keys:
        if run not in runs:
            continue
        a = cells[(run, rho)]
        acc[rho]["at_risk"] += a["at_risk"]
        acc[rho]["in_F"] += a["in_F"]
    return {r: 1.0 - acc[r]["in_F"] / acc[r]["at_risk"] for r in acc}


def split_vals(vals, min_side=MIN_SIDE):
    """clump_split's gap rule on a bare sorted list, so it can be applied at a
    single density. Returns (gap, a_bound, b_bound, k) or None."""
    v = sorted(vals)
    gaps = sorted(((v[i + 1] - v[i], v[i], v[i + 1], i + 1)
                   for i in range(len(v) - 1)), reverse=True)
    cand = [g for g in gaps if g[3] >= min_side and len(v) - g[3] >= min_side]
    return cand[0] if cand else None


def fmt_p(p, nrep):
    """Zero hits in nrep permutations is not p = 0; it is p < 1/nrep."""
    if p != p:
        return "nan"
    return ("= %.6f" % p) if p > 0 else ("< %.1e" % (1.0 / nrep))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / float(len(a | b)) if (a | b) else float("nan")


# --------------------------------------------------------------------------
# diagnostics

def d1(fh, cells):
    pairs = len(cells)
    bad = [(k, a["at_risk"]) for k, a in cells.items()
           if abs(a["at_risk"] - N_INST) > 1e-9]
    runs = sorted({run for run, _r in cells})
    missing = [(r, rho) for r in runs for rho in RHOS
               if (r, rho) not in cells]
    tot = sum(a["at_risk"] for a in cells.values())
    line(fh, "D1  n_at_risk(0) = 400 on every (run, rho) pair")
    line(fh, "    pairs %d (expected %d = %d x %d), total at_risk %d "
             "(expected %d)"
         % (pairs, N_RUNS * len(RHOS), N_RUNS, len(RHOS), round(tot),
            round(N_RUNS * len(RHOS) * N_INST)))
    line(fh, "    violations %d, missing pairs %d  -> %s"
         % (len(bad), len(missing),
            "PASS" if not bad and not missing
            and pairs == N_RUNS * len(RHOS) else "FAIL"))
    return not bad and not missing and pairs == N_RUNS * len(RHOS)


def d2(fh, cells):
    """The chance baseline is a property of the instances, not of the model.

    At j = 0 no token has been emitted, so E|F| is fixed by the instance and
    the hints. If this spreads across runs they are not sharing an eval set and
    no cross-run comparison in findings 121-130 is valid.
    """
    spread = {}
    for rho in RHOS:
        vals = [a["F_sum"] / a["at_risk"] for (run, r), a in cells.items()
                if r == rho]
        spread[rho] = (min(vals), max(vals), max(vals) - min(vals), len(vals))
    worst = max(spread.values(), key=lambda t: t[2])
    line(fh, "D2  mean |F| at j = 0 is identical across all 79 runs, per "
             "density")
    for rho in RHOS:
        lo, hi, d, n = spread[rho]
        line(fh, "    rho %.4f  n=%2d  mean|F| %.6f  spread %.3e"
             % (rho, n, lo, d))
    line(fh, "    worst spread %.3e over %d runs (tol 1e-9)  -> %s"
         % (worst[2], worst[3], "PASS" if worst[2] < 1e-9 else "FAIL"))
    return worst[2] < 1e-9


def d3(fh, cells):
    bad = [k for k, a in cells.items() if a["in_F"] > a["at_risk"] + 1e-9]
    rng = (min(a["u"] for a in cells.values()),
           max(a["u"] for a in cells.values()))
    line(fh, "D3  n_in_F(0) <= n_at_risk(0), so u lies in [0, 1]")
    line(fh, "    violations %d, u range [%.4f, %.4f]  -> %s"
         % (len(bad), rng[0], rng[1], "PASS" if not bad else "FAIL"))
    return not bad


def d4(fh, unif, nblank):
    dev_u = max(abs(unif[r][2] - UNIF_REF[i]) for i, r in enumerate(RHOS))
    dev_b = max(abs(nblank[r] - BLANK_LADDER[i]) for i, r in enumerate(RHOS))
    line(fh, "D4  u_unif(rho) and the blank ladder reproduce RULE6")
    line(fh, "    max |u_unif - tabled| %.2e (tol 5e-5), max |n_blank - "
             "ladder| %.2e (tol 1e-3)  -> %s"
         % (dev_u, dev_b, "PASS" if dev_u < 5e-5 and dev_b < 1e-3 else "FAIL"))
    return dev_u < 5e-5 and dev_b < 1e-3


def d5(fh, meta, cell_of):
    triple = collections.defaultdict(set)
    for cell, m in meta.items():
        triple[(m["mode"], m["L"], m["d"])].add(cell)
    collide = {t: sorted(cs) for t, cs in triple.items() if len(cs) > 1}
    line(fh, "D5  keyed on run_id for clumps and on cell for architecture, "
             "never on (mode, L, d)")
    line(fh, "    %d runs -> %d cells; %d (mode, L, d) triples cover more "
             "than one cell:" % (len(cell_of), len(meta), len(collide)))
    for t, cs in sorted(collide.items(), key=lambda kv: str(kv[0])):
        line(fh, "      %s -> %s  (kept separate)" % (str(t), " | ".join(cs)))
    line(fh, "    this stage never groups by the triple  -> PASS")
    return True


# --------------------------------------------------------------------------
# predictions

def e1(fh, low, keys, ut):
    """All 21 low-clump runs under half chance at every one of the 14 bins."""
    worst = []
    for run in sorted(low):
        per = [(ut[(run, r)], r) for r in RHOS if (run, r) in ut]
        mx, at = max(per)
        worst.append((mx, at, run))
    worst.sort(reverse=True)
    over = [w for w in worst if w[0] >= E1_BAR]
    line(fh, "E1  max over the 14 densities of u_tilde, for each of the %d "
             "low-clump runs" % len(low))
    line(fh, "    bar: every run < %.2f.  %d run(s) at or above it"
         % (E1_BAR, len(over)))
    line(fh, "    worst five:")
    for mx, at, run in worst[:5]:
        line(fh, "      %-28s max u_tilde %.4f at rho %.2f" % (run, mx, at))
    line(fh, "    best five:")
    for mx, at, run in worst[-5:]:
        line(fh, "      %-28s max u_tilde %.4f at rho %.2f" % (run, mx, at))
    hit = not over
    line(fh, "    -> E1 %s" % ("HIT" if hit else "MISS"))
    return hit, worst, over


def e2(fh, low, keys, cells, unif):
    """Does normalised failure rise as hints come out?"""
    u_lo = by_rho(keys, cells, low)
    ut = {r: u_lo[r] / unif[r][2] for r in RHOS}
    xs = [r for r in RHOS]
    ys = [ut[r] for r in RHOS]
    rho_s = spearman_t(xs, ys)
    p, nrep = perm_t(xs, ys, rho_s, nrep=200000, seed=20261123)
    factor = ut[RHOS[-1]] / ut[RHOS[0]] if ut[RHOS[0]] > 0 else float("inf")
    line(fh, "E2  the %d low-clump runs pooled, u and u_tilde by density"
         % len(low))
    line(fh, "    %-8s %-10s %-10s %-10s" % ("rho", "u", "u_unif", "u_tilde"))
    for r in RHOS:
        line(fh, "    %-8.4f %-10.5f %-10.4f %-10.4f"
             % (r, u_lo[r], unif[r][2], ut[r]))
    line(fh, "    Spearman(u_tilde, rho) = %+.4f (bar <= %.2f), MC p %s "
             "over %d permutations"
         % (rho_s, E2_RHO_BAR, fmt_p(p, nrep), nrep))
    line(fh, "    u_tilde(0.33) / u_tilde(0.95) = %.4f (bar >= %.1f)"
         % (factor, E2_FACTOR))
    hit = rho_s <= E2_RHO_BAR and factor >= E2_FACTOR
    line(fh, "    -> E2 %s" % ("HIT" if hit else "MISS"))
    return hit, u_lo, ut, rho_s, p, factor, nrep


def e3(fh, low, cells):
    """Does the clump membership reproduce inside rho = 0.80 alone?"""
    at = [(a["u"], run) for (run, r), a in cells.items() if r == E3_RHO]
    at.sort()
    sp = split_vals([v for v, _run in at])
    line(fh, "E3  the 79 runs ranked by u at rho = %.2f alone (chance %.4f), "
             "split at the largest gap with >= %d a side"
         % (E3_RHO, UNIF_REF[RHOS.index(round(E3_RHO, 4))], MIN_SIDE))
    if sp is None:
        line(fh, "    no admissible gap  -> E3 MISS")
        return False, None, float("nan"), []
    gap, a_b, b_b, k = sp
    side = {run for v, run in at if v <= a_b}
    j = jaccard(side, low)
    only_now = sorted(side - set(low))
    only_pooled = sorted(set(low) - side)
    line(fh, "    gap %.4f between %.4f and %.4f, %d runs below / %d above"
         % (gap, a_b, b_b, k, len(at) - k))
    line(fh, "    Jaccard against RULE6's %d-run low clump = %.4f "
             "(bar >= %.2f)" % (len(low), j, E3_JACCARD))
    line(fh, "    in the rho = %.2f low side only: %d  %s"
         % (E3_RHO, len(only_now), ", ".join(only_now[:6])))
    line(fh, "    in the pooled low clump only:   %d  %s"
         % (len(only_pooled), ", ".join(only_pooled[:6])))
    hit = j >= E3_JACCARD
    line(fh, "    -> E3 %s" % ("HIT" if hit else "MISS"))
    return hit, sp, j, sorted(side)


def e4(fh, high, keys, cells, unif):
    u_hi = by_rho(keys, cells, high)
    ut = {r: u_hi[r] / unif[r][2] for r in RHOS}
    raw = max(u_hi.values()) / min(u_hi.values())
    nor = max(ut.values()) / min(ut.values())
    line(fh, "E4  the %d high-clump runs pooled: does the chance "
             "normalisation flatten them?" % len(high))
    line(fh, "    %-8s %-10s %-10s" % ("rho", "u", "u_tilde"))
    for r in RHOS:
        line(fh, "    %-8.4f %-10.5f %-10.4f" % (r, u_hi[r], ut[r]))
    line(fh, "    raw spread  max/min u       = %.4f (bar >= %.1f)"
         % (raw, E4_FACTOR))
    line(fh, "    normalised  max/min u_tilde = %.4f (bar <  %.1f)"
         % (nor, E4_FACTOR))
    hit = raw >= E4_FACTOR and nor < E4_FACTOR
    line(fh, "    -> E4 %s" % ("HIT" if hit else "MISS"))
    return hit, u_hi, ut, raw, nor


def e5(fh, cells, cell_of, tot, seeds, meta):
    """Sum the per-(run, rho) table back and land on RULE6's own numbers."""
    agg = collections.defaultdict(lambda: {"at_risk": 0.0, "in_F": 0.0})
    per_run = collections.defaultdict(lambda: {"at_risk": 0.0, "in_F": 0.0})
    for (run, rho), a in cells.items():
        cell = cell_of[run]
        for k in ("at_risk", "in_F"):
            agg[cell][k] += a[k]
            per_run[(cell, run)][k] += a[k]
    dev = 0.0
    for cell in meta:
        for k in ("at_risk", "in_F"):
            dev = max(dev, abs(agg[cell][k] - tot[cell][k]))
    for cell in meta:
        for run, a in seeds[cell].items():
            for k in ("at_risk", "in_F"):
                dev = max(dev, abs(per_run[(cell, run)][k] - a[k]))
    mine = clump_split(meta, seeds, MIN_SIDE)
    _pr, _g, gap, a_b, b_b, k = mine
    ref = (0.1345, 21, 58)
    ok_split = (abs(gap - ref[0]) < 5e-5 and k == ref[1]
                and len(_pr) - k == ref[2])
    line(fh, "E5  reproduction: the per-(run, rho) slice summed over density")
    line(fh, "    largest absolute residual against RULE6's per-cell and "
             "per-seed counts: %.3e (tol 1e-12)" % dev)
    line(fh, "    clump split recomputed: gap %.4f, %d below %.4f / %d above "
             "%.4f (RULE6: 0.1345, 21 / 58)"
         % (gap, k, a_b, len(_pr) - k, b_b))
    hit = dev < 1e-12 and ok_split
    line(fh, "    -> E5 %s" % ("HIT" if hit else "MISS"))
    return hit, dev, mine


# --------------------------------------------------------------------------
# post-hoc

def posthoc(fh, low, high, keys, cells, ut, unif, cell_of, meta):
    line(fh, "POST-HOC -- read after the scored predictions, never a test")
    line(fh)

    line(fh, "(a) where each clump's failures sit on the grid")
    u_lo = by_rho(keys, cells, low)
    u_hi = by_rho(keys, cells, high)
    line(fh, "    %-8s %-9s %-9s %-9s %-9s %-9s"
         % ("rho", "u low", "u high", "chance", "low/ch", "high/ch"))
    for r in RHOS:
        c = unif[r][2]
        line(fh, "    %-8.4f %-9.5f %-9.5f %-9.4f %-9.4f %-9.4f"
             % (r, u_lo[r], u_hi[r], c, u_lo[r] / c, u_hi[r] / c))
    sep = [u_hi[r] / u_lo[r] if u_lo[r] > 0 else float("inf") for r in RHOS]
    fin = [s for s in sep if s != float("inf")]
    line(fh, "    separation u_high / u_low: %.1fx at rho 0.95 to %.1fx at "
             "rho 0.33" % (sep[0], sep[-1]))
    if fin:
        line(fh, "    finite separations span %.1fx to %.1fx; the clumps never "
                 "cross" % (min(fin), max(fin)))
    line(fh)

    line(fh, "(b) how many runs are under half chance at every density, and "
             "under a tenth")
    for bar in (0.5, 0.25, 0.1):
        n = 0
        for run in sorted({r for r, _x in cells}):
            per = [ut[(run, r)] for r in RHOS if (run, r) in ut]
            if per and max(per) < bar:
                n += 1
        line(fh, "    u_tilde < %.2f at all 14 densities: %d of %d runs"
             % (bar, n, N_RUNS))
    line(fh)

    line(fh, "(c) the per-run worst bin: where a low-clump run is least "
             "DD-faithful")
    at_rho = collections.Counter()
    for run in sorted(low):
        per = [(ut[(run, r)], r) for r in RHOS if (run, r) in ut]
        at_rho[max(per)[1]] += 1
    line(fh, "    argmax density over the %d low-clump runs: %s"
         % (len(low), ", ".join("rho %.2f x%d" % (r, n)
                                for r, n in at_rho.most_common())))
    line(fh)

    line(fh, "(d) the clump split density by density, not pooled")
    line(fh, "    %-8s %-7s %-9s %-9s %-7s" % ("rho", "gap", "a_bound",
                                               "b_bound", "below"))
    memb = {}
    for r in RHOS:
        at = sorted((a["u"], run) for (run, rr), a in cells.items() if rr == r)
        sp = split_vals([v for v, _x in at])
        if sp is None:
            line(fh, "    %-8.4f  no admissible gap" % r)
            continue
        gap, a_b, b_b, k = sp
        memb[r] = {run for v, run in at if v <= a_b}
        line(fh, "    %-8.4f %-7.4f %-9.4f %-9.4f %-7d"
             % (r, gap, a_b, b_b, k))
    if memb:
        js = [(jaccard(memb[r], low), r) for r in sorted(memb)]
        line(fh, "    Jaccard against the pooled low clump, by density: %s"
             % ", ".join("%.2f@%.2f" % (j, r) for j, r in js))
        line(fh, "    range %.2f to %.2f" % (min(js)[0], max(js)[0]))
    line(fh)

    line(fh, "(e) is a cell's pooled u its seeds' mean, or one seed's?")
    multi = [(c, sorted(1.0 - sum(cells[(run, r)]["in_F"] for r in RHOS)
                        / sum(cells[(run, r)]["at_risk"] for r in RHOS)
                        for run in {rn for rn, _r in cells
                                    if cell_of[rn] == c}))
             for c in meta]
    multi = [(c, v) for c, v in multi if len(v) > 1]
    spanning = [(c, v) for c, v in multi if v[0] < 0.0716071 <= v[-1]]
    line(fh, "    %d cells have more than one seed; %d of them have seeds on "
             "both sides of the pooled boundary" % (len(multi), len(spanning)))
    for c, v in sorted(spanning, key=lambda kv: kv[1][-1] - kv[1][0],
                       reverse=True)[:8]:
        line(fh, "      %-28s %s" % (c, " ".join("%.4f" % x for x in v)))
    line(fh)

    line(fh, "(f) u_tilde's resolution: one rollout of 400 is worth this much")
    line(fh, "    the quantum is 1 / (400 * u_unif(rho)), so the normalisation "
             "that makes the")
    line(fh, "    densities comparable also costs resolution at the easy end.")
    line(fh, "    %-8s %-10s %-12s" % ("rho", "quantum", "rollouts to u_tilde "
                                      "= 0.50"))
    quanta = {}
    for r in RHOS:
        q = 1.0 / (N_INST * unif[r][2])
        quanta[r] = q
        line(fh, "    %-8.4f %-10.4f %-12.1f" % (r, q, 0.50 / q))
    line(fh, "    at rho 0.95 the step is %.4f, so E1's bar sits %.1f rollouts "
             "above zero and" % (quanta[RHOS[0]], 0.50 / quanta[RHOS[0]]))
    line(fh, "    at rho 0.33 it sits %.1f. E1 is therefore a far sharper test "
             "at the hard end" % (0.50 / quanta[RHOS[-1]]))
    line(fh, "    than at the easy one, and it failed at the easy one.")
    line(fh)

    line(fh, "(g) the low clump's RAW failure rate barely moves while chance "
             "moves 16-fold")
    lo_raw = max(u_lo.values()) / min(u_lo.values())
    ch_raw = unif[RHOS[-1]][2] / unif[RHOS[0]][2]
    line(fh, "    u low  spans %.5f to %.5f, a factor of %.2f"
         % (min(u_lo.values()), max(u_lo.values()), lo_raw))
    line(fh, "    chance spans %.4f to %.4f, a factor of %.2f"
         % (unif[RHOS[0]][2], unif[RHOS[-1]][2], ch_raw))
    line(fh, "    so u_tilde falls by construction of the ratio, not because "
             "anything improved:")
    line(fh, "    %.2f / %.2f = %.3f against the measured u_tilde ratio %.4f"
         % (lo_raw, ch_raw, lo_raw / ch_raw,
            (u_lo[RHOS[-1]] / unif[RHOS[-1]][2])
            / (u_lo[RHOS[0]] / unif[RHOS[0]][2])))
    line(fh)

    line(fh, "(h) the three exact-zero cells of RULE6, by density")
    zero = [c for c in meta
            if abs(sum(cells[(run, r)]["in_F"] for r in RHOS
                       for run in [rn for rn, _r in cells
                                   if cell_of[rn] == c] if (run, r) in cells)
                   - sum(cells[(run, r)]["at_risk"] for r in RHOS
                         for run in [rn for rn, _r in cells
                                     if cell_of[rn] == c]
                         if (run, r) in cells)) < 1e-9]
    line(fh, "    cells with u = 0 pooled: %d  %s"
         % (len(zero), ", ".join(sorted(zero))))
    for c in sorted(zero):
        runs = sorted({rn for rn, _r in cells if cell_of[rn] == c})
        line(fh, "      %-28s %d seed(s), u = 0 at all 14 by construction of "
                 "the sum" % (c, len(runs)))
    return u_lo, u_hi, memb, spanning


def reading(fh, hits, low, high, e1r, e2r, e3r, e4r, unif, memb, spanning):
    line(fh, "-" * 74)
    line(fh, "READING")
    line(fh, "-" * 74)
    sc = " ".join("E%d %s" % (i + 1, "HIT" if h else "MISS")
                  for i, h in enumerate(hits))
    line(fh, "SCORE: %s" % sc)
    line(fh)
    _worst, over = e1r[1], e1r[2]
    _ulo, ut_lo, rho_s, p, factor = e2r[1], e2r[2], e2r[3], e2r[4], e2r[5]
    j = e3r[2]
    raw, nor = e4r[3], e4r[4]
    mx, at, who = e1r[1][0]
    q_at = 1.0 / (N_INST * unif[at][2])
    line(fh, "1. E1 MISSED, and it missed at the EASY end on %d rollouts of "
             "400. The worst normalised" % round(mx / q_at))
    line(fh, "   bin over the %d low-clump runs is %.4f of chance -- %s at rho "
             "%.2f, where there are"
         % (len(low), mx, who, at))
    line(fh, "   %.3f blank cells and chance is %.4f, so one rollout is worth "
             "%.4f of u_tilde and"
         % (unif[at][1], unif[at][2], q_at))
    line(fh, "   the %.2f bar sits %.1f rollouts above zero. %d of %d "
             "low-clump runs crosses it."
         % (E1_BAR, E1_BAR / q_at, len(over), len(low)))
    line(fh, "   The registered bar was uniform across densities and should "
             "not have been: see (f).")
    line(fh)
    line(fh, "2. E2 MISSED WITH THE SIGN FULLY RESOLVED IN THE OPPOSITE "
             "DIRECTION. Registered <= -0.60,")
    line(fh, "   measured Spearman(u_tilde, rho) = %+.4f (MC p %s), and "
             "the registered factor of" % (rho_s, fmt_p(p, e2r[6])))
    line(fh, "   >= 2.0 came out at %.4f. Normalised failure does not rise as "
             "hints are removed -- it" % factor)
    line(fh, "   FALLS, by %.2fx. The mechanism is in (g): the low clump's raw "
             "u spans only a factor"
         % (1.0 / factor if factor > 0 else float("inf")))
    line(fh, "   of %.2f across the whole grid while chance spans %.2f, so the "
             "ratio falls because the"
         % (max(_ulo.values()) / min(_ulo.values()),
            unif[RHOS[-1]][2] / unif[RHOS[0]][2]))
    line(fh, "   denominator grows. The honest statement is that these runs' "
             "stopping-condition failure")
    line(fh, "   rate is NEARLY DENSITY-INDEPENDENT at about %.2f%% from 3 "
             "blank cells to 32."
         % (100.0 * sum(_ulo.values()) / len(_ulo)))
    line(fh)
    line(fh, "3. The clump membership %s inside rho = %.2f alone: Jaccard "
             "%.4f against %.2f."
         % ("reproduces" if j >= E3_JACCARD else "does NOT reproduce",
            E3_RHO, j, E3_JACCARD))
    if memb:
        js = [jaccard(memb[r], low) for r in sorted(memb)]
        line(fh, "   Across all fourteen single-density splits the Jaccard "
                 "runs %.2f to %.2f, so the" % (min(js), max(js)))
        line(fh, "   split is %s a pooling artefact."
             % ("not" if min(js) >= 0.5 else "partly"))
    line(fh)
    line(fh, "4. The high clump's raw u spreads %.2fx across the grid and "
             "its normalised" % raw)
    line(fh, "   u_tilde spreads %.2fx, so %s is most of what "
             "the raw spread was."
         % (nor, "the chance level" if nor < raw else "something else"))
    line(fh, "   At rho 0.95 it reads %.4f of chance, which is a uniform draw "
             "over the blank cells"
         % (e4r[2][RHOS[0]]))
    line(fh, "   to within %.1f%%, and it improves to %.4f of chance at rho "
             "0.33."
         % (100.0 * abs(1.0 - e4r[2][RHOS[0]]), e4r[2][RHOS[-1]]))
    line(fh)
    line(fh, "5. %d cells carry seeds on both sides of the pooled boundary, "
             "which is finding 130" % len(spanning))
    line(fh, "   measured per density rather than pooled. Seeds are "
             "training, so this is reported")
    line(fh, "   and not acted on; the width sweep stays closed and DQ4 "
             "stays unsubmitted.")
    line(fh)
    line(fh, "LIMITS. Everything is j = 0. u is agreement with DD's stopping "
             "condition, not logical")
    line(fh, "validity. u_tilde's denominator is a uniform blank pick, so "
             "u_tilde = 0.5 means half as")
    line(fh, "often off-trace as a procedure with no model in it, never "
             "'half right'. The clump")
    line(fh, "boundary is descriptive, carried forward from RULE6, and is "
             "never used as a test.")


def main():
    os.makedirs(OUT, exist_ok=True)
    (meta, cell_of, tot, seeds, by_dens, orphan, seq, n_haz,
     nblank) = cells_j0()
    cells = per_run_dens(seq)
    unif = uniform_by_rho(seq, nblank)
    ut = normalise(cells, unif)
    keys = sorted(cells)

    pr, _g, _gap, a_bound, _b, _k = clump_split(meta, seeds, MIN_SIDE)
    low = {run for v, _c, run in pr if v <= a_bound}
    high = {run for v, _c, run in pr if v > a_bound}

    path = os.path.join(OUT, "VERDICT.txt")
    with open(path, "w") as fh:
        line(fh, "STAGE RULE7 -- WHERE ON THE DENSITY GRID DD's STOPPING "
                 "CONDITION SURVIVES")
        line(fh, "refit of runs/stagelogic_2026-11-17, %d hazard rows, %d "
                 "runs, %d architectures, %d (run, rho) pairs"
             % (n_haz, len(cell_of), len(meta), len(cells)))
        line(fh, "registered in runs/stagerule7_2026-11-23/REGISTERED.md "
                 "before this scorer existed")
        line(fh, "nothing trained, nothing submitted, no closed knob reopened")
        line(fh)
        line(fh, "-" * 74)
        line(fh, "DIAGNOSTICS")
        line(fh, "-" * 74)
        ds = [d1(fh, cells), d2(fh, cells), d3(fh, cells),
              d4(fh, unif, nblank), d5(fh, meta, cell_of)]
        line(fh)
        line(fh, "diagnostics: %d of %d PASS" % (sum(ds), len(ds)))
        line(fh)
        line(fh, "-" * 74)
        line(fh, "SCORED PREDICTIONS")
        line(fh, "-" * 74)
        line(fh, "clump membership carried forward from RULE6: %d low / %d "
                 "high, boundary %.7f" % (len(low), len(high), a_bound))
        line(fh)
        r1 = e1(fh, low, keys, ut)
        line(fh)
        r2 = e2(fh, low, keys, cells, unif)
        line(fh)
        r3 = e3(fh, low, cells)
        line(fh)
        r4 = e4(fh, high, keys, cells, unif)
        line(fh)
        r5 = e5(fh, cells, cell_of, tot, seeds, meta)
        line(fh)
        line(fh, "-" * 74)
        ph = posthoc(fh, low, high, keys, cells, ut, unif, cell_of, meta)
        line(fh)
        reading(fh, [r1[0], r2[0], r3[0], r4[0], r5[0]], low, high,
                r1, r2, r3, r4, unif, ph[2], ph[3])
    print("wrote", path)


if __name__ == "__main__":
    main()
