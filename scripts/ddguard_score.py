"""Does the ONE-SHARED-BLOCK model fire when DD fires and withhold when DD withholds?

REGISTERED BEFORE THE FIRST EVAL RUNS. This docstring IS the registration --
committed by scripts/slurm/ddguard_register_2026-11-13.sbatch, which the eval job
scripts/slurm/ddguard_eval_2026-11-13.sbatch depends on with --dependency=afterok.
No number exists when this is written.

THIS IS NOT A STAGE AND IT MOVES NO KNOB

No model is trained here. Six checkpoints that already exist are re-read, and no
published number can change. The brief as of 2026-11-13 is the SMALLEST OR MOST
ANALYTICALLY TRACTABLE architecture that still LEARNS DIRECT DEDUCTION, and
`rho_50` is a behavioural proxy for the second half of that sentence. Findings
27/28 built three causal tests that ask it directly, and not one of them has ever
been run on a tied model. Every registered knob is closed (width, cosine decay,
more data, best-val selection, role count, schedule length, training budget) and
nothing here reopens one.

It is also the cheap half of STATUS next-step 1 candidate (b), CHANGE THE
MEASUREMENT. `rho_50` is a 3-seed median of a 14-point grid, and finding 67 put
the old noise floor at sigma_d 0.01412 against an effect of 0.0011 -- 2x to 22x
the effect, which is why stages S through W all read "within noise". The two
statistics below are PAIRED PER CASE on byte-identical case lists, so the arms
differ in the network and in nothing else. That is a different instrument, not a
different arm, and it costs one short job instead of GPU-weeks.

THE TWO ARMS, AND WHY THEY ARE COMPARABLE

  S1  tied_no_mlp T=16 d320    613,818 trainable   rho_50 0.757   acc 0.4830
  J2  no_mlp      L=16 d192  2,504,890 trainable   rho_50 0.580   acc 0.4940

Same depth, same head count, same attention-only block, differing in whether the
16 blocks share one set of weights. S1 is the most analytically tractable object
this project has -- one weight matrix set to read instead of sixteen -- and it
misses the free line by 0.017 of rho_50, which stage S4 put on the accuracy curve
at NEED of token accuracy. J2 is the standing minimum and the reference line the
gap is measured against.

Every key that generates an INSTANCE is identical in the two config.json files --
n_nodes 48, degree 5, n_colors 4, rho 0.3, rho_max 0.95, hint_mode removal,
shuffle_adj True, shuffle_events False, fall True, scratchpad False -- and both
evals are run at the same --seed-start and --n-instances. build_instances is a
pure function of those, and case selection is a pure function of the instances
plus np.random.default_rng(0), so the case list is the same object in all six
cells. Guard 2 below asserts that rather than trusting it. The arms therefore
differ in d_model and arch_mode and in nothing else, which is also the one
respect in which they are NOT matched: 613,818 against 2,504,890 is a 4.1x
budget difference, and stage S exists because that confound could not be removed
without changing the thing under test.

THE TWO MEASUREMENTS

Finding 27, THE RULE (fire). Node v is forced in round 0 because its known
neighbours cover q-1 colours, leaving c free, and colour a is held by exactly one
hint w. Recolour w from a to c. v stays forced, a is now the free colour, and
DD's answer flips c -> a. One token changes.

    rule = #{clean_correct and corrupt_flipped} / #{clean_correct}

conditioned on clean-correct because a model that was wrong before the edit
cannot be scored on following it, and finding 28 conditioned the same way.

Finding 28, THE GUARD (withhold). Same case, different edit: recolour w to b, a
colour ANOTHER neighbour already holds. Then a is held by nobody, the forcing set
drops to q-2, v has two candidates and DD DOES NOT EMIT v THIS ROUND. The read
moves one position left, from the logits that predict v's colour to the logits
that predict the node identity.

    guard = #{clean_emits_v and corrupt_emits_v} / #{clean_emits_v}

LOW is good: a low guard means the model checked determinacy. Every case carries
a placebo edit on a non-neighbour hint that leaves v forced, so

    placebo = #{clean_emits_v and placebo_emits_v} / #{clean_emits_v}

is the upper baseline the headline is read against. Without it a low `guard`
could just be edit perturbation.

WHY THE GUARD IS THE DISCRIMINATING ONE, AND IT IS NOT A GUESS

Finding 28 is the only test in this project that has ever separated two models
with the same rule score. Arms A (solve_rate 1.0000) and B (0.8540) BOTH flip at
1.000 on finding 27, so the elimination rule is fully present in both and the
0.146 between them is invisible to the rule test. On the guard test they are
0.000 against 0.176, McNemar exact p = 1.14e-13. The behavioural gap lived
entirely in the guard. S1 against J2 is a comparable situation -- two models that
both clearly do DD, separated by a behavioural margin nothing architectural has
explained -- so the guard is where finding 28 says to look first.

THE HYPOTHESES, AS THE EXPRESSIONS THAT WILL BE COMPUTED

Finding 39's lesson, and the reason each is written as code rather than a
sentence plus a remembered number: a pre-registration that says "the gap" and
quotes a baseline computed a different way fails as written even when the
science replicates.

  H1  RULE PRESENT IN THE TIED MODEL.
        statistic   min(rule[S1][s] for s in seeds)
        bar         >= RULE_BAR
        predicted   CONFIRMED, near 1.000.
      Finding 27 found the rule fully present in BOTH leak-free arms including
      the weaker one, so a model at S1's accuracy should have it. H1 is here
      because the guard number is uninterpretable without it: if S1 fails the
      rule test, a bad guard score says the model never learned DD at all, which
      is a different and much duller finding than a missing guard.
      FALSIFIER: min rule < RULE_BAR. Then H2 is not scored, because a guard
      deficit in a model without the rule localises nothing.

  H2  THE TIED MODEL WITHHOLDS LESS -- THE HEADLINE.
        statistic   per seed s, paired McNemar on corrupt_emits_v over the cases
                    where BOTH arms have clean_emits_v, counting
                      b = #{S1 emits, J2 does not}
                      c = #{J2 emits, S1 does not}
                    exact two-sided binomtest(min(b,c)*2 ... ) as below, then
                    Holm over the three seed pairs
        bar         CONFIRMED if >= 2 of 3 seed pairs are significant at
                    ALPHA after Holm AND b > c in every significant pair
        predicted   CONFIRMED.
      THE EFFECT SIZE IS DELIBERATELY NOT BARRED. Finding 28's 0.176 was measured
      against a 0.146 solve_rate difference; here the behavioural difference is
      0.177 of rho_50, a different quantity on a different scale, and there is no
      honest way to convert one into the other. Barring a magnitude I cannot
      derive would be inventing a number. The median paired difference is
      reported as the effect size and read as a description.
      FALSIFIER, and it is a real one: no seed pair significant, or the
      significant ones have c > b. Then the guard is NOT what tying costs and
      finding 28's decomposition does not transfer to this pair -- which closes a
      door and is worth the job on its own. Finding 28 makes this entirely
      possible: its LEAKED arm, which flips at 0.008 and never learned the rule,
      still withheld at 0.008. Determinacy looked easy to learn there, so both
      arms scoring ~0 here is the outcome to expect if the guard is cheap.

  H3  THE MEASUREMENT IS SPECIFIC (a guard on the instrument, not a result).
        statistic   placebo[X][s], every arm, every seed
        bar         >= PLACEBO_FLOOR
        predicted   CONFIRMED.
      Finding 28's placebo ran 0.90-0.96 at N=24. PLACEBO_FLOOR is set below that
      because N=48 is a longer sequence and a one-token edit anywhere has more
      room to perturb. Any cell that misses it has its guard number printed
      UNINTERPRETABLE rather than scored, and if that is S1 the headline is void.

  H4  DESCRIPTIVE ONLY, NOT A TEST. guard[S1][s] against that seed's own rho_50
      (S1_RHO50_SEEDS). THREE POINTS IS NOT A CORRELATION and no bar is set. It
      is printed because "3 seeds with the spread quoted" is a standing
      non-negotiable and because a guard deficit that tracked rho_50 across seeds
      would be the first thing to follow up; a claim is not made either way.

CAVEATS, STATED BEFORE THE NUMBERS EXIST

  - ROUND 0 ONLY. Both edits need every known neighbour of v to be a hint. The
    chained version of the rule test (spin_gnn/sudoku_chain.py, finding 27's
    second half) is a THIRD question about the PROVENANCE of the eliminated fact
    and is deliberately not run here; it is the registered follow-up.
  - The arms differ 4.1x in trainable parameters. Any gap found is tying AND
    budget together, exactly as stage S's own rungs were.
  - `corrupt_emits_v` is an argmax over node tokens at one position. It is a
    decision-level statistic and says nothing about where in the network the
    guard lives; localisation would be the next job, not this one.
  - A pass on both H1 and H2 does not close the 0.011 accuracy gap. It would name
    what the gap IS. Nothing here trains anything.

    python3 scripts/ddguard_score.py --register        # configs only, no eval
    python3 scripts/ddguard_score.py --run-dir DIR     # score the eval
"""

import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# IMPORTED, NEVER RETYPED. check_claims in the determinism builder refused a
# render on 2026-11-13 because a published rounding had been typed into a second
# module; 0.0011 and 0.4841 are roundings of values these modules DERIVE.
from scripts.stagev_score import (FREE, S1_ACC, S1_RHO50, S1_RHO50_SEEDS,
                                  S1_TRAINABLE, J2_ACC, RHO_GAP, NEED)
from scripts.stagej_score import PREDICTED_TRAINABLE

# J2's rho_50 is not a literal anywhere: stage V DERIVES the gap as S1 minus J2
# on the same pair, so J2's value is recovered from it rather than retyped.
J2_RHO50 = round(float(S1_RHO50) - RHO_GAP, 3)          # 0.580

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
_SPIN = os.path.join(_SCR, "spin_gnn")
SCAN_S = os.path.join(_SPIN, "sudoku_n48_tiedwidth_S_2026-11-10")
SCAN_J = os.path.join(_SPIN, "sudoku_n48_minarch_J_2026-11-05")

# arm -> seed -> checkpoint directory. The cell names are stage S's and stage J's
# own, unchanged, so the provenance of each checkpoint is readable from the path.
CKPT = {
    "S1": {
        0: os.path.join(SCAN_S, "t00_d0320_ff01024_h04_tied_no_mlp_L16_s0",
                        "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied"),
        1: os.path.join(SCAN_S, "t01_d0320_ff01024_h04_tied_no_mlp_L16_s1",
                        "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_s01"),
        2: os.path.join(SCAN_S, "t02_d0320_ff01024_h04_tied_no_mlp_L16_s2",
                        "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_s02"),
    },
    "J2": {
        0: os.path.join(SCAN_J, "t01_d0192_ff01024_h04_no_mlp_L16_s0",
                        "d0192_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff"),
        1: os.path.join(SCAN_J, "t03_d0192_ff01024_h04_no_mlp_L16_s1",
                        "d0192_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_s01"),
        2: os.path.join(SCAN_J, "t05_d0192_ff01024_h04_no_mlp_L16_s2",
                        "d0192_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_s02"),
    },
}
ARMS = ("S1", "J2")
SEEDS = (0, 1, 2)

# What each arm's config.json must say. arch_mode and d_model are the knob; the
# trainable count is the identity check, and both counts come from the stages
# that measured them.
EXPECT = {
    "S1": {"arch_mode": "tied_no_mlp", "d_model": 320, "num_layers": 16,
           "n_trainable_params": S1_TRAINABLE},
    "J2": {"arch_mode": "no_mlp", "d_model": 192, "num_layers": 16,
           "n_trainable_params": PREDICTED_TRAINABLE["J2"]},
}
# Keys that generate an INSTANCE. Identical across arms is what makes the case
# lists identical and the comparison paired; guard 1 asserts it from the files.
INSTANCE_KEYS = ("n_nodes", "degree", "n_colors", "rho", "rho_max", "hint_mode",
                 "shuffle_adj", "shuffle_events", "fall", "scratchpad")

RULE_BAR = 0.60          # finding 27's own bar: arm A cleared it at 1.000, arm C
                         # missed at 0.388 against a chance of 1/q = 0.250.
PLACEBO_FLOOR = 0.85     # finding 28 measured 0.90-0.96 at N=24; see H3.
ALPHA = 0.05
# The case key, per row of skip_behavior.csv. Row ORDER plus these four fields is
# the case identity; if two arms disagree anywhere the pools diverged and nothing
# below is paired.
CASE_KEY = ("node", "binding_nbr", "n_known", "round0_rank")
# The same identity check for the rule test, whose case finder is a different
# function over the same instance pool and so has its own, shorter, case list.
RULE_KEY = ("node", "true_color", "flipped_color")


def rows_of(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def truthy(s):
    return str(s).strip().lower() in ("1", "true", "yes")


def ratio(rows, cond, hit):
    """#{cond and hit} / #{cond}, with the denominator, so n is always visible."""
    den = [r for r in rows if truthy(r[cond])]
    if not den:
        return float("nan"), 0
    return sum(truthy(r[hit]) for r in den) / len(den), len(den)


def binom_exact(k, n):
    """Two-sided exact McNemar p: 2 * P(X <= k) for X ~ Binomial(n, 1/2), capped at
    1, where n is the discordant count and k = min(b, c). Written out rather than
    imported from scipy because the binomial is symmetric at p0 = 1/2, so the
    two-sided p is exactly twice the lower tail and there is nothing to get wrong;
    one fewer import is one fewer way for a job to die at line 1."""
    if n == 0:
        return 1.0
    from math import comb
    lower = sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * lower)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, pvals[i] * (len(pvals) - rank))
        running = max(running, adj)
        out[i] = running
    return out


def guard_configs():
    """Guard 1: every checkpoint is the cell it is filed under, and the two arms
    agree on every instance-generating key."""
    bad, cfgs = [], {}
    for arm in ARMS:
        for s in SEEDS:
            path = os.path.join(CKPT[arm][s], "config.json")
            if not os.path.exists(path):
                bad.append("%s s%d: no config.json at %s" % (arm, s, path))
                continue
            with open(path) as fh:
                cfg = json.load(fh)
            cfgs[(arm, s)] = cfg
            for k, want in EXPECT[arm].items():
                if cfg.get(k) != want:
                    bad.append("%s s%d: %s is %r, expected %r"
                               % (arm, s, k, cfg.get(k), want))
            if cfg.get("seed") != s:
                bad.append("%s s%d: config seed is %r" % (arm, s, cfg.get("seed")))
    ref = cfgs.get(("S1", 0))
    if ref:
        for key, cfg in cfgs.items():
            for k in INSTANCE_KEYS:
                if cfg.get(k) != ref.get(k):
                    bad.append("%s s%d: instance key %s is %r, S1 s0 has %r"
                               % (key[0], key[1], k, cfg.get(k), ref.get(k)))
    return bad, cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true",
                    help="guard 1 only: the six checkpoints exist and are what "
                         "they claim. Run before any eval, so a wrong checkpoint "
                         "costs no GPU.")
    ap.add_argument("--paths", action="store_true",
                    help="print 'arm seed checkpoint' lines and exit. The eval "
                         "sbatch loops over THIS, so the six paths are written "
                         "once and the launcher cannot drift from the scorer.")
    ap.add_argument("--run-dir", type=str, default=os.path.join(
        _SPIN, "runs", "2026-11-13_ddguard"))
    args = ap.parse_args()

    if args.paths:
        for arm in ARMS:
            for s in SEEDS:
                print("%s %d %s" % (arm, s, CKPT[arm][s]))
        return

    print("THE FIRE-AND-GUARD PROBE: does one shared block run DD's rule AND its guard?")
    print("S1 tied_no_mlp T=16 d320 %d trainable, rho_50 %s, acc %.4f"
          % (S1_TRAINABLE, S1_RHO50, S1_ACC))
    print("J2 no_mlp      L=16 d192 %d trainable, rho_50 %.3f, acc %.4f"
          % (PREDICTED_TRAINABLE["J2"], J2_RHO50, J2_ACC))
    print("free line %.3f, so S1 misses the gate by %.3f of rho_50 = %.7f of accuracy"
          % (FREE, float(S1_RHO50) - FREE, NEED))
    print("bars: rule >= %.2f, placebo >= %.2f, alpha %.2f Holm over 3 seed pairs"
          % (RULE_BAR, PLACEBO_FLOOR, ALPHA))

    print("\n=== GUARD 1: the six checkpoints are the cells they claim ===")
    bad, _ = guard_configs()
    for line in bad:
        print("  FAIL %s" % line)
    if bad:
        sys.exit("guard 1 failed: %d problems" % len(bad))
    print("  PASS: 6 checkpoints, arch_mode/d_model/num_layers/trainable/seed all")
    print("  as filed, and every instance-generating key identical across arms.")

    if args.register:
        print("\nREGISTERED. No eval has run; nothing above touched a GPU.")
        return

    missing = []
    for arm in ARMS:
        for s in SEEDS:
            for sub, fn in (("rule", "behavior.csv"), ("guard", "skip_behavior.csv")):
                p = os.path.join(args.run_dir, sub, "%s_s%d" % (arm, s), fn)
                if not os.path.exists(p):
                    missing.append(p)
    if missing:
        sys.exit("eval incomplete, %d files missing, first: %s"
                 % (len(missing), missing[0]))

    guard = {arm: {} for arm in ARMS}
    grows, rrows = {}, {}
    print("\n=== GUARD 2: the case lists are the same object in all six cells ===")
    for sub, fn, key, store in (("guard", "skip_behavior.csv", CASE_KEY, grows),
                                ("rule", "behavior.csv", RULE_KEY, rrows)):
        ref_keys = None
        for arm in ARMS:
            for s in SEEDS:
                rs = rows_of(os.path.join(args.run_dir, sub, "%s_s%d" % (arm, s), fn))
                store[(arm, s)] = rs
                keys = [tuple(r[k] for k in key) for r in rs]
                if ref_keys is None:
                    ref_keys = keys
                elif keys != ref_keys:
                    n = sum(1 for a, b in zip(keys, ref_keys) if a != b)
                    sys.exit("guard 2 FAILED on the %s test: %s s%d differs from "
                             "S1 s0 in %d of %d case keys (len %d vs %d). NOTHING "
                             "BELOW IS PAIRED."
                             % (sub, arm, s, n, min(len(keys), len(ref_keys)),
                                len(keys), len(ref_keys)))
        print("  PASS %-5s %4d cases, identical %s row for row in all 6 cells"
              % (sub, len(ref_keys), "/".join(key)))
        if sub == "guard":
            n_guard_cases = len(ref_keys)
    print("  so every comparison below is paired: the arms differ in the network")
    print("  and in nothing else.")

    print("\n=== H1: does the tied model APPLY the elimination rule? (fire) ===")
    print("  rule = #{clean_correct and corrupt_flipped} / #{clean_correct}")
    rule = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for s in SEEDS:
            rs = rrows[(arm, s)]
            v, n = ratio(rs, "clean_correct", "corrupt_flipped")
            rule[arm][s] = v
            print("  %s s%d  rule %.4f   (n=%d clean-correct of %d cases)"
                  % (arm, s, v, n, len(rs)))
    h1_stat = min(rule["S1"][s] for s in SEEDS)
    h1 = h1_stat >= RULE_BAR
    print("  statistic min(rule[S1]) = %.4f against bar %.2f -> %s"
          % (h1_stat, RULE_BAR, "CONFIRMED" if h1 else "FALSIFIED"))
    if not h1:
        print("  H2 IS NOT SCORED: a guard deficit in a model that lacks the rule")
        print("  localises nothing. That is H1's registered falsifier.")

    print("\n=== H3: is the measurement specific? (placebo, the instrument guard) ===")
    print("  placebo = #{clean_emits_v and placebo_emits_v} / #{clean_emits_v}")
    placebo, interpretable = {arm: {} for arm in ARMS}, {}
    for arm in ARMS:
        for s in SEEDS:
            rs = grows[(arm, s)]
            pv, n = ratio(rs, "clean_emits_v", "placebo_emits_v")
            gv, _ = ratio(rs, "clean_emits_v", "corrupt_emits_v")
            placebo[arm][s] = pv
            guard[arm][s] = gv
            interpretable[(arm, s)] = pv >= PLACEBO_FLOOR
            print("  %s s%d  clean_emits_v %.4f (n=%d)  placebo %.4f  %s"
                  % (arm, s, n / len(rs) if rs else float("nan"), n, pv,
                     "" if pv >= PLACEBO_FLOOR else "<- BELOW FLOOR"))
    h3 = all(interpretable.values())
    print("  every cell at or above %.2f -> %s"
          % (PLACEBO_FLOOR, "CONFIRMED" if h3 else "FALSIFIED"))

    print("\n=== H2: does the tied model WITHHOLD less? (guard, THE HEADLINE) ===")
    print("  guard = #{clean_emits_v and corrupt_emits_v} / #{clean_emits_v}; LOW is good")
    for arm in ARMS:
        vals = [guard[arm][s] for s in SEEDS]
        tag = "" if all(interpretable[(arm, s)] for s in SEEDS) else "  UNINTERPRETABLE"
        print("  %s  guard %.4f / %.4f / %.4f   median %.4f  spread %.4f%s"
              % (arm, vals[0], vals[1], vals[2], statistics.median(vals),
                 max(vals) - min(vals), tag))
    med_diff = (statistics.median([guard["S1"][s] for s in SEEDS])
                - statistics.median([guard["J2"][s] for s in SEEDS]))
    print("  median S1 minus median J2 = %+.4f   (positive = the tied model "
          "withholds LESS)" % med_diff)

    if h1 and h3:
        print("\n  paired McNemar per seed, over cases where BOTH arms emit v cleanly")
        raw, detail = [], []
        for s in SEEDS:
            a, b_rows = grows[("S1", s)], grows[("J2", s)]
            both = [(x, y) for x, y in zip(a, b_rows)
                    if truthy(x["clean_emits_v"]) and truthy(y["clean_emits_v"])]
            b = sum(1 for x, y in both
                    if truthy(x["corrupt_emits_v"]) and not truthy(y["corrupt_emits_v"]))
            c = sum(1 for x, y in both
                    if truthy(y["corrupt_emits_v"]) and not truthy(x["corrupt_emits_v"]))
            p = binom_exact(min(b, c), b + c)
            raw.append(p)
            detail.append((s, len(both), b, c, p))
        adj = holm(raw)
        sig = 0
        for (s, n, b, c, p), pa in zip(detail, adj):
            ok = pa < ALPHA and b > c
            sig += 1 if ok else 0
            print("    seed %d: n=%d paired, b(S1 only)=%d c(J2 only)=%d, "
                  "p=%.3g Holm %.3g  %s"
                  % (s, n, b, c, p, pa,
                     "SIGNIFICANT, S1 worse" if ok
                     else ("significant, J2 worse" if pa < ALPHA else "ns")))
        h2 = sig >= 2
        print("  %d of 3 seed pairs significant with b > c, bar is 2 -> %s"
              % (sig, "CONFIRMED" if h2 else "FALSIFIED"))
        if not h2:
            print("  FALSIFIED means the guard is NOT what tying costs and finding")
            print("  28's decomposition does not transfer to this pair. That closes")
            print("  a door; it is a result, not a failed run.")
    else:
        h2 = None
        print("  NOT SCORED (H1 %s, H3 %s)" % (h1, h3))

    print("\n=== H4: descriptive only, three points is not a correlation ===")
    # S1_RHO50_SEEDS is stage S's tuple in seed order, so index s IS seed s.
    for s in SEEDS:
        print("  S1 s%d  rho_50 %.3f   guard %.4f" % (s, S1_RHO50_SEEDS[s], guard["S1"][s]))
    print("  No bar, no claim. Printed because the spread is a standing non-negotiable.")

    print("\n=== VERDICT ===")
    for name, val in (("H1 rule present", h1), ("H2 guard discriminates", h2),
                      ("H3 placebo specific", h3)):
        print("  %-24s %s" % (name, {True: "CONFIRMED", False: "FALSIFIED",
                                     None: "NOT SCORED"}[val]))
    print("  Nothing was trained. No published number changes. The gate stays")
    print("  rho_50 <= %.3f and S1 still misses it by %.3f."
          % (FREE, float(S1_RHO50) - FREE))
    out = os.path.join(args.run_dir, "ddguard_summary.json")
    with open(out, "w") as fh:
        json.dump({"rule": rule, "guard": guard, "placebo": placebo,
                   "median_guard_diff": med_diff, "n_cases": n_guard_cases,
                   "H1": h1, "H2": h2, "H3": h3}, fh, indent=2, sort_keys=True)
    print("  wrote %s" % out)


if __name__ == "__main__":
    main()
