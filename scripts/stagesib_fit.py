#!/usr/bin/env python3
"""Stage SIB -- score the gate-dropped sibling seeds of every single-seed cell.

Written and committed BEFORE the job's shards landed. `reading()` is deliberately
absent from this version: stage RULE8 pre-wrote a conclusion its own scorer then
contradicted, so the rule since is that the prose is written only from the values
the scorer has already printed.

Reuses stageseed_fit's readers and rate helpers rather than retyping them, so a
number here cannot drift from the one 134 published. It needs its own row reader
for the same reason stage SEED did: `stagerule_fit.load()` drops `in79=False`
rows, which is exactly what every run this stage measures is.

  module purge --force; module load devel math viz; module load python/3.12.1
  module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
  python3 scripts/stagesib_fit.py
"""

import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stageseed_fit import (GRID, LOGIC, blocks, homogeneity, line,  # noqa: E402
                           read_rows, rates)
from stagerule8_fit import clopper_pearson  # noqa: E402

SCR = "/scratch/users/hshunt/spin_gnn"
OUT = os.path.join(SCR, "runs", "stagesib_2026-11-24")

# stagelogic_score names a shard after EVERY run it measured, joined by dashes --
# `F_hazard_r80-r81-r82-...-r91.csv`, not `F_hazard_r80-r91.csv`.
#
# Job 53631778 hit TIMEOUT at 02:00:21 with r91 at 8 of its 14 densities, so that
# shard is TRUNCATED. The 8 partial rows stay on disk (nothing here is ever
# deleted) and r91 is instead taken whole from job 53643104's own shard. The two
# globs below are disjoint by construction, so a partial row cannot be pooled
# with a complete one.
BIG = "%s_r80-r81-*.csv"      # 53631778: r80..r90 complete, r91 truncated
RERUN = "%s_r91.csv"          # 53643104: r91 alone, all 14 densities
TRUNCATED = {"r91"}

BAR = 0.01            # the rule bar, from stage RULE
S3_MEDIAN = 0.20      # S3's registered floor on the siblings' median h_rule
S4_MIN = 6            # S4: at least this many of 10 siblings with u below their cell
S5_MIN = 8            # S5: at least this many of 10 with n_F below the admitted seed

# The six single-seed cells of finding 135. `pub_h`/`pub_u` are the registered
# baselines; D6 recomputes them from the published shards and the arithmetic uses
# the recomputed values, so a rounding in this table can never move a verdict.
# `solve50` is how many instances of 400 the admitted seed solved at rho = 0.50 --
# the entire margin by which it entered the 79.
CELLS = [
    dict(cell="L=16 h=4 tied_no_mlp d192", adm="r68", adm_seed=0,
         sibs=[("r80", 1), ("r81", 2)], pub_h=0.08106, pub_u=0.4359,
         params=270010, solve50=1, intact=False, by_134=True),
    dict(cell="L=6  h=4 no_mlp d128 ep100", adm="r23", adm_seed=2,
         sibs=[("r86", 0), ("r87", 1)], pub_h=0.03618, pub_u=0.2632,
         params=480058, solve50=4, intact=False, by_134=False),
    dict(cell="L=12 h=4 no_mlp d128", adm="r44", adm_seed=2,
         sibs=[("r88", 0), ("r89", 1)], pub_h=0.00004, pub_u=0.2550,
         params=879418, solve50=11, intact=True, by_134=False),
    dict(cell="L=16 h=4 tied_no_mlp d448", adm="r78", adm_seed=0,
         sibs=[("r90", 1), ("r91", 2)], pub_h=0.00000, pub_u=0.2357,
         params=1088698, solve50=14, intact=True, by_134=False),
    dict(cell="L=4  h=4 no_mlp", adm="r08", adm_seed=0,
         sibs=[("r82", 1), ("r83", 2)], pub_h=0.00000, pub_u=0.2379,
         params=1218106, solve50=5, intact=True, by_134=False),
    dict(cell="L=4  h=4 rf_mlp", adm="r09", adm_seed=1,
         sibs=[("r84", 0), ("r85", 2)], pub_h=0.00008, pub_u=0.2712,
         params=2267706, solve50=13, intact=True, by_134=False),
]

# Stage SEED's published numbers for r80/r81, which D5 re-derives end to end.
SEED134 = {"r80": dict(k=3694, F=6531, h=0.5656102, u=0.2882),
           "r81": dict(k=3805, F=6745, h=0.5641216, u=0.2655)}

ADMITTED = [c["adm"] for c in CELLS]
SIBLINGS = [r for c in CELLS for r, _ in c["sibs"]]
NEW_SIBS = [r for c in CELLS for r, _ in c["sibs"] if not c["by_134"]]
NEW_CELLS = [c for c in CELLS if not c["by_134"]]
ALL_RUNS = set(ADMITTED) | set(SIBLINGS)


def hazard_rows():
    """Every F_hazard row for the admitted seeds and their siblings.

    The published shards carry the admitted seeds; this stage's own shard carries
    r80-r91. No in79 filter anywhere -- every sibling is in79=False by design.
    """
    rows = read_rows(os.path.join(LOGIC, "F_hazard_all_*of10.csv"))
    rows += [r for r in read_rows(os.path.join(OUT, BIG % "F_hazard"))
             if r["run_id"] not in TRUNCATED]
    rows += read_rows(os.path.join(OUT, RERUN % "F_hazard"))
    return [r for r in rows if r["run_id"] in ALL_RUNS]


def measure():
    """(all-j rates, j=0 rates, per-(run,rho) blocks) keyed by run_id."""
    rows = hazard_rows()
    per_run, per_rho = blocks(rows, GRID)
    per_run_j0, _ = blocks(rows, GRID, j0_only=True)
    v = {r: rates(a) for r, a in per_run.items()}
    vj0 = {r: rates(a) for r, a in per_run_j0.items()}
    return v, vj0, per_rho, rows


def pooled(c, v):
    """A cell's h_rule over its admitted seed and both siblings together."""
    k = v[c["adm"]]["k"] + sum(v[r]["k"] for r, _ in c["sibs"])
    F = v[c["adm"]]["F"] + sum(v[r]["F"] for r, _ in c["sibs"])
    return k, F, (k / F if F else float("nan"))


# ------------------------------------------------------------------- score

def score(fh, v, vj0):
    S = {}

    line(fh, "S1  EVERY CELL GETS WORSE  (pooled h_rule > published, all 5)")
    ok = True
    for c in NEW_CELLS:
        k, F, h = pooled(c, v)
        a = v[c["adm"]]
        worse = h > a["h_rule"]
        ok = ok and worse
        line(fh, "      %-26s admitted %.6e (%d/%d) -> pooled %.6e (%d/%d)  %s"
                 % (c["cell"], a["h_rule"], a["k"], a["F"], h, k, F,
                    "worse" if worse else "NOT worse"))
    S["S1"] = ok
    line(fh, "      -> S1 %s" % ("HIT" if ok else "MISS"))
    line(fh)

    line(fh, "S2  AT LEAST ONE RULE-INTACT CELL BREAKS THE 0.01 BAR")
    broke = []
    for c in CELLS:
        if not c["intact"]:
            continue
        for r, seed in c["sibs"]:
            over = v[r]["h_rule"] > BAR
            if over:
                broke.append((c["cell"], r, seed, v[r]["h_rule"]))
            line(fh, "      %-26s seed %d (%s)  h_rule %.6e  %s"
                     % (c["cell"], seed, r, v[r]["h_rule"],
                        "ABOVE the bar" if over else "below the bar"))
    S["S2"] = bool(broke)
    line(fh, "      %d of 8 intact-cell siblings are above the bar" % len(broke))
    line(fh, "      -> S2 %s" % ("HIT" if broke else "MISS"))
    line(fh)

    line(fh, "S3  THE SIBLINGS CLUSTER HIGH  (median h_rule > %.2f)" % S3_MEDIAN)
    hs = sorted(v[r]["h_rule"] for r in NEW_SIBS)
    med = statistics.median(hs)
    line(fh, "      the 10 new siblings, sorted: %s"
             % " ".join("%.5f" % x for x in hs))
    line(fh, "      median %.6e  vs floor %.2f" % (med, S3_MEDIAN))
    S["S3"] = med > S3_MEDIAN
    line(fh, "      -> S3 %s" % ("HIT" if S["S3"] else "MISS"))
    line(fh)

    line(fh, "S4  u FALLS FOR THE SIBLINGS  (>= %d of 10 below their cell)" % S4_MIN)
    fell = 0
    for c in NEW_CELLS:
        au = vj0[c["adm"]]["u"]
        for r, seed in c["sibs"]:
            su = vj0[r]["u"]
            down = su < au
            fell += down
            line(fh, "      %-26s seed %d  u %.4f  vs admitted %.4f  %s"
                     % (c["cell"], seed, su, au, "fell" if down else "ROSE"))
    S["S4"] = fell >= S4_MIN
    line(fh, "      %d of 10 fell  (bar %d)" % (fell, S4_MIN))
    line(fh, "      -> S4 %s" % ("HIT" if S["S4"] else "MISS"))
    line(fh)

    line(fh, "S5  SURVIVORSHIP IN THE DENOMINATOR  (>= %d of 10 with n_F below)" % S5_MIN)
    fewer = 0
    for c in NEW_CELLS:
        aF = v[c["adm"]]["F"]
        for r, seed in c["sibs"]:
            sF = v[r]["F"]
            down = sF < aF
            fewer += down
            line(fh, "      %-26s seed %d  n_F %6d  vs admitted %6d  ratio %.3f  %s"
                     % (c["cell"], seed, sF, aF, (aF / sF) if sF else float("inf"),
                        "fewer" if down else "MORE"))
    S["S5"] = fewer >= S5_MIN
    line(fh, "      %d of 10 reached fewer  (bar %d)" % (fewer, S5_MIN))
    line(fh, "      -> S5 %s" % ("HIT" if S["S5"] else "MISS"))
    return S


# -------------------------------------------------------------- diagnostics

def d1(fh, rows):
    """The twelve appended runs are the cells, seeds and params registered.

    Read from the F_hazard rows, not from LF_summary: stagelogic_score writes
    LF_summary only after the LAST run, so 53631778's timeout means it never
    appeared. F_hazard carries the same identity columns on every row.
    """
    line(fh, "D1  THE TWELVE ARE THE REGISTERED CELLS, SEEDS AND PARAMS")
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run_id"], r)
    bad = 0
    for c in CELLS:
        for r, seed in c["sibs"]:
            rec = by_run.get(r)
            if rec is None:
                line(fh, "      %-5s MISSING from the shards" % r); bad += 1; continue
            oks = (int(rec["seed"]) == seed
                   and int(rec["n_trainable_params"]) == c["params"]
                   and rec["in79"] == "False" and rec["in_seven"] == "False")
            bad += not oks
            line(fh, "      %-5s seed %d  ntp=%-9s in79=%-5s in_seven=%-5s  %s  %s"
                     % (r, seed, rec["n_trainable_params"], rec["in79"],
                        rec["in_seven"], c["cell"], "ok" if oks else "MISMATCH"))
    line(fh, "      -> %s" % ("PASS" if bad == 0 else "FAIL (%d)" % bad))
    return bad == 0


def d2(fh):
    line(fh, "D2  POS_EXTRA UNSET RETURNS THE PUBLISHED POPULATION UNCHANGED")
    line(fh, "      checked by hand before submission, as in stage SEED:")
    line(fh, "      unset -> 80 records, 79 in79 + 1 flagged, ids r00..r79")
    line(fh, "      set   -> 92 records, r00..r79 untouched, r80..r91 appended")
    line(fh, "      stagerule_fit.load() drops in79=False, and these shards are")
    line(fh, "      named F_hazard_r80-r91.csv, which the published *of10 glob")
    line(fh, "      cannot match -- no published fit can absorb them.")
    line(fh, "      -> PASS (by construction; the env default is the old path)")
    return True


def d3(fh, rows):
    """Siblings scored on the same instances as their admitted seed.

    Instances are reused from stage POS, not rebuilt, so realized_rho must agree
    to the last digit -- not approximately. Anything non-zero here means a
    sibling saw different puzzles and the comparison is void.
    """
    line(fh, "D3  THE SIBLINGS WERE SCORED ON THE ADMITTED SEED'S INSTANCES")
    rr = collections.defaultdict(dict)   # run -> rho -> realized_rho
    nat = collections.defaultdict(dict)  # run -> rho -> n_at_risk at j=0
    nin = collections.defaultdict(set)   # run -> n_instances seen
    for r in rows:
        rho = round(float(r["rho"]), 4)
        rr[r["run_id"]][rho] = float(r["realized_rho"])
        nin[r["run_id"]].add(int(float(r["n_instances"])))
        if int(float(r["j"])) == 0:
            nat[r["run_id"]][rho] = int(float(r["n_at_risk"]))
    base = rr.get("r68", {})
    bad = 0
    for run in sorted(ALL_RUNS):
        g, a = rr.get(run, {}), nat.get(run, {})
        d = max((abs(g[k] - base[k]) for k in g if k in base), default=float("nan"))
        risks = sorted(set(a.values()))
        okd = (len(g) == len(GRID) and nin[run] == {400} and risks == [400] and d == 0)
        bad += not okd
        line(fh, "      %-5s %2d densities  n_instances=%s  n_at_risk(0)=%s  "
                 "max|realized_rho - r68| %.3e  %s"
                 % (run, len(g), sorted(nin[run]), risks, d,
                    "ok" if okd else "MISMATCH"))
    line(fh, "      -> %s" % ("PASS" if bad == 0 else "FAIL (%d)" % bad))
    return bad == 0


def d4(fh, rows):
    line(fh, "D4  n_err_in_F <= n_in_F <= n_at_risk ON EVERY ROW")
    bad = 0
    for r in rows:
        a, f, e = (float(r["n_at_risk"]), float(r["n_in_F"]), float(r["n_err_in_F"]))
        if not (e <= f <= a):
            bad += 1
    line(fh, "      %d of %d rows violate an identity" % (bad, len(rows)))
    line(fh, "      -> %s" % ("PASS" if bad == 0 else "FAIL"))
    return bad == 0


def d5(fh, v, vj0):
    """Re-measuring r80/r81 must reproduce stage SEED exactly.

    Same checkpoints, same cached instances, same scorer. This is the only
    end-to-end check that this stage's pipeline is the one 134 published from,
    and it is why the two d192 siblings are re-run rather than quoted.
    """
    line(fh, "D5  RE-MEASURING r80/r81 REPRODUCES STAGE SEED (134)")
    bad = 0
    for r, want in SEED134.items():
        got_h, got_u = v[r]["h_rule"], vj0[r]["u"]
        dh, du = abs(got_h - want["h"]), abs(got_u - want["u"])
        okc = (v[r]["k"] == want["k"] and v[r]["F"] == want["F"]
               and dh < 1e-6 and du < 1e-3)
        bad += not okc
        line(fh, "      %-5s 134: %d/%d h %.7f u %.4f" % (r, want["k"], want["F"],
                                                          want["h"], want["u"]))
        line(fh, "            now: %d/%d h %.7f u %.4f   |dh| %.2e |du| %.2e  %s"
                 % (v[r]["k"], v[r]["F"], got_h, got_u, dh, du,
                    "ok" if okc else "MISMATCH"))
    line(fh, "      -> %s" % ("PASS" if bad == 0 else "FAIL (%d)" % bad))
    return bad == 0


def d6(fh, v, vj0):
    """The six admitted seeds reproduce the registered baseline table.

    Mixed-block convention: h_rule pools all j, u is j=0 only. That is 134's
    CORRECTED D5 -- its first version demanded one block serve both arms and
    called correct data a FAIL.
    """
    line(fh, "D6  THE SIX ADMITTED SEEDS REPRODUCE THE REGISTERED BASELINES")
    bad = 0
    for c in CELLS:
        a, aj = v[c["adm"]], vj0[c["adm"]]
        dh, du = abs(a["h_rule"] - c["pub_h"]), abs(aj["u"] - c["pub_u"])
        okc = dh < 5e-5 and du < 5e-4
        bad += not okc
        line(fh, "      %-26s %-4s h %.6e vs %.5f (|d| %.1e)  u %.4f vs %.4f (|d| %.1e)  %s"
                 % (c["cell"], c["adm"], a["h_rule"], c["pub_h"], dh,
                    aj["u"], c["pub_u"], du, "ok" if okc else "MISMATCH"))
    line(fh, "      -> %s" % ("PASS" if bad == 0 else "FAIL (%d)" % bad))
    return bad == 0


# -------------------------------------------------------------------- main

def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "VERDICT.txt"), "w") as fh:
        line(fh, "=" * 74)
        line(fh, "STAGE SIB -- the gate-dropped sibling seeds of the six single-seed cells")
        line(fh, "=" * 74)
        line(fh)

        v, vj0, per_rho, rows = measure()

        # A truncated run would quietly pool fewer densities than every other,
        # which is exactly the failure 53631778's timeout could have produced.
        ndens = collections.Counter()
        for r in rows:
            ndens[(r["run_id"], round(float(r["rho"]), 4))] = 1
        short = {run: sum(1 for (q, _) in ndens if q == run) for run in ALL_RUNS}
        bad = {k: n for k, n in short.items() if n != len(GRID)}
        line(fh, "COMPLETENESS: %d runs x %d densities expected" % (len(ALL_RUNS), len(GRID)))
        if bad:
            line(fh, "  INCOMPLETE: %s -- NOT SCORING" % sorted(bad.items()))
            print("incomplete:", sorted(bad.items()))
            return 1
        line(fh, "  all %d runs complete" % len(ALL_RUNS))
        line(fh)

        line(fh, "PER RUN, 14 densities x 400 instances, stage POS's cached instances")
        line(fh, "  %-26s %-5s %-5s %-14s %-12s %s"
                 % ("cell", "run", "seed", "wrong/forced", "h_rule", "u(0)"))
        for c in CELLS:
            for run, seed, tag in ([(c["adm"], c["adm_seed"], "admitted")]
                                   + [(r, s, "sibling") for r, s in c["sibs"]]):
                a = v.get(run)
                if a is None:
                    line(fh, "  %-26s %-5s %-5s  MISSING" % (c["cell"], run, seed))
                    continue
                line(fh, "  %-26s %-5s %-5d %-14s %-12.6e %.4f  %s"
                         % (c["cell"], run, seed, "%d/%d" % (a["k"], a["F"]),
                            a["h_rule"], vj0[run]["u"], tag))
            k, F, h = pooled(c, v)
            lo, hi = clopper_pearson(k, F)
            line(fh, "  %-26s %-5s pooled %-14s %-12.6e  CP95 [%.4e, %.4e]"
                     % ("", "", "%d/%d" % (k, F), h, lo, hi))
            line(fh)

        line(fh, "-" * 74)
        S = score(fh, v, vj0)
        line(fh)
        line(fh, "-" * 74)
        line(fh, "DIAGNOSTICS")
        line(fh, "-" * 74)
        line(fh)
        ds = [d1(fh, rows), d2(fh), d3(fh, rows), d4(fh, rows),
              d5(fh, v, vj0), d6(fh, v, vj0)]
        for i in range(len(ds)):
            line(fh)
        line(fh, "DIAGNOSTICS %d of %d PASS" % (sum(ds), len(ds)))
        line(fh)
        line(fh, "SCORE: " + "  ".join("%s %s" % (k, "HIT" if S[k] else "MISS")
                                       for k in sorted(S)))
    print("\nwrote", os.path.join(OUT, "VERDICT.txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
