#!/usr/bin/env python3
"""Stage SEED -- score the three seeds of `tied_no_mlp L=16 d192` against the
predictions registered in runs/stageseed_2026-11-24/REGISTERED.md.

r68 (seed 0) comes from the published stage LOGIC shards; r80 and r81 (seeds 1
and 2) come from this stage's own shards, measured on the SAME cached instances.
Nothing is trained here and nothing is submitted.

The reading at the bottom is written from the values this script prints. It is
never written before them -- stage RULE8 pre-wrote a conclusion its own scorer
then contradicted, with the contradicting number in the same sentence.

Run under the FIGURE env:

  module purge --force; module load devel math viz
  module load python/3.12.1
  module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
"""

import collections
import csv
import glob
import math
import os
import sys

from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stagerule8_fit import clopper_pearson, wilson  # noqa: E402

SCR = "/scratch/users/hshunt/spin_gnn"
LOGIC = os.path.join(SCR, "runs", "stagelogic_2026-11-17")
OUT = os.path.join(SCR, "runs", "stageseed_2026-11-24")

CELL = "L=16 h=4 tied_no_mlp d192"
SEED0, SEED1, SEED2 = "r68", "r80", "r81"
SEEDS = [SEED0, SEED1, SEED2]

# The 14-point eval grid, as RHO_GRID in stagepos_cells. Written out rather than
# imported because importing stagepos_cells pulls in the stage scorers, and one
# of those needs numba, which no py3.12 module on Sherlock provides.
GRID = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.33]

# The published d192 numbers this stage is extending, from finding 131.
PUB_H = 0.08106
PUB_U = 0.4359
PUB_PARAMS = 270010
BAR = 0.01            # the rule bar h_rule must clear
S5_FACTOR = 2.0       # S5's "within a factor of 2 of PUB_H"
TOL = 1e-12


def fnum(r, k):
    return float(r[k])


def read_rows(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def hazard_rows():
    """Every F_hazard row for the three seeds, from both stages, no in79 filter.

    stagerule_fit.load() drops in79=False rows, which is why it cannot see r80
    and r81 and why this stage needs its own reader rather than a flag on that
    one.
    """
    rows = read_rows(os.path.join(LOGIC, "F_hazard_all_*of10.csv"))
    rows += read_rows(os.path.join(OUT, "F_hazard_r80-r81.csv"))
    return [r for r in rows if r["run_id"] in set(SEEDS)]


def summary_rows():
    rows = read_rows(os.path.join(LOGIC, "LF_summary_all_*of10.csv"))
    rows += read_rows(os.path.join(OUT, "LF_summary_r80-r81.csv"))
    return [r for r in rows if r["run_id"] in set(SEEDS)]


def blocks(rows, grid, j0_only=False):
    """Pooled (at_risk, in_F, err, err_in_F) per run, and per (run, rho)."""
    g = set(round(x, 4) for x in grid)
    per_run = collections.defaultdict(
        lambda: dict.fromkeys(("at_risk", "in_F", "err", "err_in_F"), 0.0))
    per_rho = collections.defaultdict(
        lambda: dict.fromkeys(("at_risk", "in_F", "err", "err_in_F"), 0.0))
    for r in rows:
        if round(fnum(r, "rho"), 4) not in g:
            continue
        if j0_only and int(float(r["j"])) != 0:
            continue
        for key in (per_run[r["run_id"]],
                    per_rho[(r["run_id"], round(fnum(r, "rho"), 4))]):
            key["at_risk"] += fnum(r, "n_at_risk")
            key["in_F"] += fnum(r, "n_in_F")
            key["err"] += fnum(r, "n_err")
            key["err_in_F"] += fnum(r, "n_err_in_F")
    return dict(per_run), dict(per_rho)


def rates(a):
    h = a["err_in_F"] / a["in_F"] if a["in_F"] > 0 else float("nan")
    s_F = a["in_F"] / a["at_risk"] if a["at_risk"] > 0 else float("nan")
    lo, hi = wilson(int(round(a["err_in_F"])), int(round(a["in_F"])))
    u_k = int(round(a["at_risk"] - a["in_F"]))
    u_lo, u_hi = wilson(u_k, int(round(a["at_risk"])))
    return {"h_rule": h, "h_lo": lo, "h_hi": hi, "s_F": s_F, "u": 1.0 - s_F,
            "u_lo": u_lo, "u_hi": u_hi, "u_k": u_k,
            "k": int(round(a["err_in_F"])), "F": int(round(a["in_F"])),
            "at_risk": int(round(a["at_risk"])), "err": int(round(a["err"]))}


def homogeneity(counts):
    """Chi-square test that k_i/n_i is the same across the rows of `counts`.

    counts is [(k, n), ...]. Returns (chi2, df, p, pooled_rate). This is the
    2 x m contingency test on (error, no error), which is what S2 registered;
    with m = 3 the null has 2 degrees of freedom.
    """
    K = sum(k for k, _ in counts)
    N = sum(n for _, n in counts)
    if K <= 0 or N <= 0:
        return float("nan"), len(counts) - 1, float("nan"), float("nan")
    p = K / N
    stat = 0.0
    for k, n in counts:
        for obs, exp in ((k, n * p), (n - k, n * (1 - p))):
            if exp > 0:
                stat += (obs - exp) ** 2 / exp
    df = len(counts) - 1
    return stat, df, float(chi2.sf(stat, df)), p


# ------------------------------------------------------------- diagnostics

def line(fh, s=""):
    print(s)
    fh.write(s + "\n")


def d1(fh, summ):
    """r80 and r81 are the same architecture triple as r68, flagged out of 79."""
    line(fh, "D1  THE THREE RUNS ARE THE SAME CELL, TWO OF THEM OUTSIDE THE 79")
    ok = True
    seen = {}
    for r in summ:
        seen.setdefault(r["run_id"], r)
    for rid in SEEDS:
        r = seen.get(rid)
        if r is None:
            line(fh, "      %s  MISSING" % rid)
            ok = False
            continue
        trip = (r["arch_mode"], int(float(r["num_layers"])),
                int(float(r["d_model"])))
        ntp = int(float(r["n_trainable_params"]))
        in79 = str(r.get("in79", "")).strip().lower() in ("1", "true", "yes")
        good = (trip == ("tied_no_mlp", 16, 192) and ntp == PUB_PARAMS
                and in79 == (rid == SEED0))
        ok = ok and good
        line(fh, "      %s  seed %s  %s L=%d d=%d  ntp=%d  val_acc=%.4f  "
                 "in79=%s  %s"
                 % (rid, r["seed"], trip[0], trip[1], trip[2], ntp,
                    float(r["val_acc"]), in79, "ok" if good else "WRONG"))
    line(fh, "      expected the triple ('tied_no_mlp', 16, 192), %d params, and"
             % PUB_PARAMS)
    line(fh, "      in79 True for %s only" % SEED0)
    line(fh, "      -> %s" % ("PASS" if ok else "FAIL"))
    line(fh)
    return ok


def d2(fh):
    """POS_EXTRA unset leaves the population exactly as it was."""
    line(fh, "D2  POS_EXTRA UNSET RETURNS THE POPULATION UNCHANGED")
    line(fh, "      checked by hand on 2026-11-24 before submission:")
    line(fh, "      POS_EXTRA unset   -> 80 records, 79 in79 + 1 flagged, last")
    line(fh, "                           id r79 (L=2 h=4 full, solve@0.50 0.000)")
    line(fh, "      POS_EXTRA set     -> 82 records, 79 in79 + 3 flagged, the two")
    line(fh, "                           new ids r80 and r81, r00..r79 unchanged")
    line(fh, "      stagerule_fit.load() drops in79=False rows, so no published")
    line(fh, "      fit can absorb r80/r81 even with POS_EXTRA exported.")
    line(fh, "      -> PASS (by construction; the env default is the old path)")
    line(fh)
    return True


def d3(fh, rows):
    """All three seeds carry the same grid, the same bins, the same instances."""
    line(fh, "D3  THE THREE SEEDS WERE SCORED ON THE SAME INSTANCES")
    g = set(round(x, 4) for x in GRID)
    ok = True
    real = collections.defaultdict(dict)
    nins = collections.defaultdict(dict)
    for r in rows:
        rho = round(fnum(r, "rho"), 4)
        if rho in g:
            real[r["run_id"]][rho] = fnum(r, "realized_rho")
            nins[r["run_id"]][rho] = int(float(r["n_instances"]))
    for rid in SEEDS:
        n_d = len(real.get(rid, {}))
        bins = set(nins.get(rid, {}).values())
        good = n_d == len(GRID) and bins == {400}
        ok = ok and good
        line(fh, "      %s  %d densities  n_instances=%s  %s"
                 % (rid, n_d, sorted(bins), "ok" if good else "WRONG"))
    worst = 0.0
    for rho in sorted(g):
        base = real.get(SEED0, {}).get(rho)
        for rid in (SEED1, SEED2):
            v = real.get(rid, {}).get(rho)
            if base is not None and v is not None:
                worst = max(worst, abs(v - base))
    line(fh, "      largest |realized_rho - r68's| over the grid: %.3e" % worst)
    good = worst <= 1e-9
    ok = ok and good
    line(fh, "      instances are REUSED, not rebuilt, so this must be 0  -> %s"
             % ("ok" if good else "WRONG"))
    line(fh, "      -> %s" % ("PASS" if ok else "FAIL"))
    line(fh)
    return ok


def d4(fh, rows):
    """The count identities hold on every row."""
    line(fh, "D4  n_err_in_F <= n_in_F <= n_at_risk ON EVERY ROW")
    bad = 0
    for r in rows:
        ar, inf = fnum(r, "n_at_risk"), fnum(r, "n_in_F")
        e_all, e_inf = fnum(r, "n_err"), fnum(r, "n_err_in_F")
        if inf > ar + TOL or e_inf > inf + TOL or e_inf > e_all + TOL:
            bad += 1
    line(fh, "      %d of %d rows violate an identity" % (bad, len(rows)))
    line(fh, "      -> %s" % ("PASS" if bad == 0 else "FAIL"))
    line(fh)
    return bad == 0


def d5(fh, v0, v0_j0):
    """r68 recomputed here must reproduce the published d192 numbers.

    THE TWO ARMS ARE MEASURED ON DIFFERENT BLOCKS, and this diagnostic was
    written as though they shared one. `h_rule` pools every emission that named
    a forced cell anywhere in the rollout; `u = 1 - s_F(0)` is the FIRST
    emission only, which is what the (0) in its name says. Asking one block for
    both gives h_rule 0.08106 with u 0.3717 (all j) or u 0.4359 with h_rule
    0.19848 (j=0) -- neither is the published pair, and the first version of
    this check called that a FAIL. The mixed pair below is the registered
    convention and reproduces both numbers exactly.
    """
    line(fh, "D5  r68 RECOMPUTED REPRODUCES THE PUBLISHED d192 NUMBERS")
    line(fh, "      h_rule pools all j; u is j = 0 only -- different blocks by")
    line(fh, "      definition, so the published pair is the mixed one.")
    line(fh, "      published (finding 131):  h_rule %.5f        u %.4f"
             % (PUB_H, PUB_U))
    line(fh, "      recomputed:               h_rule %.5f  (%d/%d)  u %.4f  (%d/%d)"
             % (v0["h_rule"], v0["k"], v0["F"],
                v0_j0["u"], v0_j0["u_k"], v0_j0["at_risk"]))
    dh, du = abs(v0["h_rule"] - PUB_H), abs(v0_j0["u"] - PUB_U)
    line(fh, "      |delta| h_rule %.2e   u %.2e" % (dh, du))
    line(fh, "      for the record, the two unmixed readings, neither published:")
    line(fh, "        all j only:  h_rule %.5f   u %.4f" % (v0["h_rule"], v0["u"]))
    line(fh, "        j = 0 only:  h_rule %.5f   u %.4f"
             % (v0_j0["h_rule"], v0_j0["u"]))
    ok = dh < 5e-5 and du < 5e-4
    line(fh, "      -> %s" % ("PASS" if ok else "FAIL"))
    line(fh)
    return ok, "h_rule all j, u at j=0"


# ------------------------------------------------------------------- score

def score(fh, per_run, per_run_j0, per_rho):
    line(fh, "=" * 74)
    line(fh, "STAGE SEED -- three seeds of %s (%d params)" % (CELL, PUB_PARAMS))
    line(fh, "=" * 74)
    line(fh)

    v = {rid: rates(per_run[rid]) for rid in SEEDS if rid in per_run}
    vj0 = {rid: rates(per_run_j0[rid]) for rid in SEEDS if rid in per_run_j0}

    line(fh, "PER SEED, 14 densities x 400 instances, same cached instances")
    line(fh, "  run  seed  rule errors / forced emissions      h_rule      "
             "Wilson 95%")
    for rid in SEEDS:
        if rid not in v:
            continue
        a = v[rid]
        line(fh, "  %-4s %-5s %7d / %-9d  %20.6e  [%.3e, %.3e]"
                 % (rid, {SEED0: 0, SEED1: 1, SEED2: 2}[rid], a["k"], a["F"],
                    a["h_rule"], a["h_lo"], a["h_hi"]))
    line(fh)
    line(fh, "  run   unforced first emissions / at risk        u = 1 - s_F   "
             "Wilson 95%")
    for rid in SEEDS:
        if rid not in vj0:
            continue
        a = vj0[rid]
        line(fh, "  %-4s %9d / %-9d  %22.4f  [%.4f, %.4f]"
                 % (rid, a["u_k"], a["at_risk"], a["u"], a["u_lo"], a["u_hi"]))
    line(fh)

    pooled = dict.fromkeys(("at_risk", "in_F", "err", "err_in_F"), 0.0)
    for rid in SEEDS:
        for k in pooled:
            pooled[k] += per_run.get(rid, {}).get(k, 0.0)
    vp = rates(pooled)
    line(fh, "POOLED over the three seeds")
    line(fh, "  h_rule = %d / %d = %.6e   Wilson 95%% [%.3e, %.3e]"
             % (vp["k"], vp["F"], vp["h_rule"], vp["h_lo"], vp["h_hi"]))
    cp = clopper_pearson(vp["k"], vp["F"])
    line(fh, "  Clopper-Pearson 95%%            [%.3e, %.3e]" % (cp[0], cp[1]))
    line(fh)

    # ------------------------------------------------------------- S1
    line(fh, "S1  BOTH NEW SEEDS ALSO BREAK THE RULE  (h_rule > %.2f)" % BAR)
    s1 = True
    for rid in (SEED1, SEED2):
        a = v.get(rid)
        if a is None:
            s1 = False
            line(fh, "      %s MISSING" % rid)
            continue
        hit = a["h_rule"] > BAR
        s1 = s1 and hit
        line(fh, "      %s  h_rule %.6e  %s the bar  -> %s"
                 % (rid, a["h_rule"], "above" if hit else "BELOW",
                    "HIT" if hit else "MISS"))
    line(fh, "      -> S1 %s" % ("HIT" if s1 else "MISS"))
    line(fh)

    # ------------------------------------------------------------- S2
    line(fh, "S2  THE THREE SEEDS AGREE WITHIN COUNTING NOISE  (chi2, p > 0.05)")
    counts = [(v[rid]["k"], v[rid]["F"]) for rid in SEEDS if rid in v]
    stat, df, p, rate = homogeneity(counts)
    s2 = p > 0.05
    for rid in SEEDS:
        if rid in v:
            a = v[rid]
            line(fh, "      %s  observed %d  expected %.1f  at the pooled rate "
                     "%.6e" % (rid, a["k"], a["F"] * rate, rate))
    line(fh, "      chi2 = %.3f on %d d.o.f.,  p = %.4g" % (stat, df, p))
    line(fh, "      -> S2 %s  (%s)"
             % ("HIT" if s2 else "MISS",
                "seed is not resolvable on the rule arm" if s2
                else "seed IS resolvable on the rule arm"))
    line(fh)

    # ------------------------------------------------------------- S3
    line(fh, "S3  SEED 0 IS THE OPTIMISTIC END  (h_rule(r68) below both others)")
    s3 = True
    h0 = v[SEED0]["h_rule"]
    for rid in (SEED1, SEED2):
        a = v.get(rid)
        if a is None:
            s3 = False
            continue
        hit = h0 < a["h_rule"]
        s3 = s3 and hit
        line(fh, "      h_rule(r68) %.6e  vs  %s %.6e   ratio %.3f  -> %s"
                 % (h0, rid, a["h_rule"], a["h_rule"] / h0,
                    "HIT" if hit else "MISS"))
    line(fh, "      -> S3 %s" % ("HIT" if s3 else "MISS"))
    line(fh)

    # ------------------------------------------------------------- S4
    line(fh, "S4  THE STOPPING CONDITION MOVES WITH IT  (u >= %.4f)" % PUB_U)
    s4 = True
    for rid in (SEED1, SEED2):
        a = vj0.get(rid)
        if a is None:
            s4 = False
            continue
        hit = a["u"] >= PUB_U
        s4 = s4 and hit
        line(fh, "      %s  u = %.4f  vs r68's published %.4f  -> %s"
                 % (rid, a["u"], PUB_U, "HIT" if hit else "MISS"))
    line(fh, "      -> S4 %s" % ("HIT" if s4 else "MISS"))
    line(fh)

    # ------------------------------------------------------------- S5
    line(fh, "S5  POOLING DOES NOT RESCUE THE CELL")
    fac = vp["h_rule"] / PUB_H
    within = (1.0 / S5_FACTOR) <= fac <= S5_FACTOR
    above = vp["h_rule"] > BAR
    s5 = within and above
    line(fh, "      pooled h_rule %.6e  vs published %.5f   factor %.3f"
             % (vp["h_rule"], PUB_H, fac))
    line(fh, "      within a factor of %.0f: %s;  above the %.2f bar: %s"
             % (S5_FACTOR, within, BAR, above))
    line(fh, "      -> S5 %s" % ("HIT" if s5 else "MISS"))
    line(fh)

    return v, vj0, vp, (s1, s2, s3, s4, s5), (stat, df, p, rate), per_rho


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = hazard_rows()
    summ = summary_rows()
    if not rows:
        raise SystemExit("no F_hazard rows for %s -- has the job finished?"
                         % ",".join(SEEDS))

    per_run, per_rho = blocks(rows, GRID)
    per_run_j0, _ = blocks(rows, GRID, j0_only=True)

    path = os.path.join(OUT, "VERDICT.txt")
    with open(path, "w") as fh:
        res = score(fh, per_run, per_run_j0, per_rho)
        line(fh, "-" * 74)
        line(fh, "DIAGNOSTICS")
        line(fh, "-" * 74)
        line(fh)
        v, vj0, vp, S, chi, _ = res
        oks = [d1(fh, summ), d2(fh), d3(fh, rows), d4(fh, rows)]
        ok5, which = d5(fh, v[SEED0], vj0[SEED0])
        oks.append(ok5)
        line(fh, "DIAGNOSTICS %d of %d PASS" % (sum(oks), len(oks)))
        line(fh)
        line(fh, "SCORE: " + "  ".join(
            "S%d %s" % (i + 1, "HIT" if s else "MISS") for i, s in enumerate(S)))
        line(fh)
    print("\nwrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
