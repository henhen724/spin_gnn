"""FINDING 91: is the guard a mechanism, or is it what a thin margin does?

THE QUESTION 86 LEFT OPEN AND 90 DID NOT CLOSE. Stage PAIR's lambda=1 arm
withholds node v under a one-token edit at 0.1087 / 0.2072 / 0.1632 of its
clean-emitting cases against stage Y's 0.0015 / 0.0000 / 0.0017, and 82 showed
the margin really did install the predicate (matched-pair AUROC 0.871 / 0.862 /
0.871 against a placebo at chance). But emission is an ARGMAX, and three
separate results have been pointing at that fact rather than at a mechanism:

  * 84 reran the identical recipe and the raw guard moved 0.1087 -> 0.1399
    while the AUROCs reproduced to 0.0010;
  * 86 found that noise is SPORADIC -- two of three seeds reran bit-identically
    at step 125 -- which is what a tie-break looks like and not what a noisy
    measurement looks like;
  * 90 forced a feedback loop open at a correctly calibrated 10% and moved the
    guard +0.0535 / -0.0426 / -0.0102, i.e. in both directions, mean +0.0002.

So the live question is not whether the predicate exists. It is whether the
predicate can ever REACH the emission anywhere except where the top-two logits
are already nearly tied.

THE QUANTITY THAT SETTLES IT IS ALREADY ON DISK. `run_cfresp` records, per case,
`clean_v_margin` = z_clean[v] - max_{u != v} z_clean[u] at the deciding
position. When v is the clean argmax this is exactly top-1 minus top-2, and it
is the amount a perturbation must move the relative logits to turn v off. The
guardspec rows hold it for 3 seeds x 3 arms x 1,000 cases, so this finding
loads 9,000 json lines and no model: NO GPU, NO TRAINING, NO NEW FORWARD PASS.

HYPOTHESIS, stated before the numbers. Two factors, not one. The MARGIN decides
which cases are ELIGIBLE to flip and the predicate decides WHICH WAY the
eligible ones go. If that is right, the guard is real but confined: it is
expressible only in the thin-margin stratum, and stage PAIR's lambda=1 training
bought its guard largely by thinning margins rather than by building a
mechanism that can withhold a confident emission.

THE PREDICTIONS, registered here and committed before the script is run. On
`warm L1`, among clean-emitting cases, with quartiles taken within each arm's
own margin distribution (Q1 thinnest, Q4 thickest):

  91.1  median clean margin is LOWER on `warm L1` than on `Y`, 3/3.
  91.2  AUROC of (-margin) for predicting the corrupt turn-off >= 0.85, 3/3.
  91.3  the same AUROC for the PLACEBO turn-off >= 0.80, 3/3. Eligibility is a
        property of the margin and not of which edit was made; if instead the
        corrupt AUROC is high and the placebo AUROC sits at chance, the margin
        is NOT the gate and the guard reaches confident emissions too.
  91.4  raw guard rate in Q4 <= 0.02, 3/3, against raw rates of 0.1087 /
        0.2072 / 0.1632 over all four quartiles.
  91.5  the placebo-corrected EXCESS (corrupt-off rate minus placebo-off rate)
        is within +-0.02 of zero in Q4 on 3/3, and >= 0.05 in Q1 on >= 2/3.
  91.6  among clean-on cases that went off, the `both` class (off under corrupt
        AND placebo -- the fragile ones) has the lowest median margin of the
        three overlap classes, 3/3.

WHAT WOULD FALSIFY THE PICTURE: 91.2 below 0.70, or a Q4 guard above 0.08. Either
says flips happen at thick margins too, the churn is not a tie-break, and the
guard is a mechanism that can withhold a confident emission.

WHAT THIS IS NOT. A stratification is not an intervention. Margin quartiles are
chosen by the model's own confidence, so they may also differ in case
difficulty -- the script therefore prints median `n_known` and `round0_rank`
per quartile, because a magnitude checked without its control is not a finding
(the lesson of 76, 79, 80, 83, 88 and 90, in that order). A difficulty gradient
across quartiles is a confound and is to be reported as one, not explained away.

NOTHING REGISTERED CHANGES AND NOTHING IS APPLIED. P4 stays the raw conditional
or the placebo-corrected excess exactly as hshunt left it; this script reads
neither bar, re-stratifies nothing into a verdict, and prints no verdict. The
raw guard rate and denominator are recomputed from the rows and asserted
against each seed's own `guardspec_summary.json`, so if the artifact is not the
one 83-86 published the job fails instead of reporting.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/spin_gnn/scripts")

import stagepair_score as SP
from spin_gnn.sudoku_determinacy import auroc

RUNS = "/scratch/users/hshunt/spin_gnn/runs"
OUT_DIR = os.path.join(RUNS, "stagepair_marginstrat_2026-11-15")
SEEDS = (0, 1, 2)
ARMS = ("Y", "warm L0", "warm L1")
N_BOOT = 2000
BOOT_SEED = 91_000_000


def guardspec_dir(seed):
    return os.path.join(RUNS, "stagepair_guardspec_2026-11-15"
                        + ("" if seed == 0 else f"_s{seed}"))


def load_rows(seed, arm):
    p = os.path.join(guardspec_dir(seed), "rows_%s.jsonl" % arm.replace(" ", "_"))
    with open(p) as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) == SP.PAIR_EVAL_N_CASES, (p, len(rows))
    return rows


def boot_excess(off_c, off_p, rng):
    """Paired bootstrap over cases of (corrupt-off rate - placebo-off rate).

    Paired because the two rates are measured on the SAME cases, which is the
    whole reason the placebo is a usable control here.
    """
    n = len(off_c)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(N_BOOT, n))
    d = off_c[idx].mean(axis=1) - off_p[idx].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(BOOT_SEED)
print("FINDING 91  margin stratification of the guard", flush=True)
print("EVAL-FREE: 9,000 rows already on disk, no model, no GPU, no forward pass")
print(f"bars imported from stagepair_score (GUARD_BAR {SP.GUARD_BAR}, never "
      f"retyped); neither bar is read here\n", flush=True)

per_seed = {}
for seed in SEEDS:
    with open(os.path.join(guardspec_dir(seed), "guardspec_summary.json")) as fh:
        published = json.load(fh)["per_arm"]
    per_arm = {}

    for arm in ARMS:
        rows = load_rows(seed, arm)
        on = [r for r in rows if r["clean_emits_v"]]
        m = np.array([r["clean_v_margin"] for r in on], dtype=np.float64)
        off_c = np.array([not r["corrupt_emits_v"] for r in on], dtype=np.float64)
        off_p = np.array([not r["placebo_emits_v"] for r in on], dtype=np.float64)
        assert (m > 0).all(), "clean_emits_v but a non-positive margin"

        # THE ARTIFACT CHECK. Recomputed from the rows, asserted against the
        # summary this seed's own job wrote. Nothing is retyped from a paper.
        pub = published[arm]["corrupt"]
        assert len(on) == pub["denom_clean_on"], (seed, arm, len(on))
        assert abs(off_c.mean() - pub["rate_off_given_on"]) < 1e-9, (seed, arm)

        # quartiles WITHIN this arm's own margin distribution: the question is
        # where a flip sits in this model's confidence, not on a shared scale.
        edges = np.quantile(m, [0.25, 0.50, 0.75])
        q = np.searchsorted(edges, m, side="right")  # 0..3 = Q1..Q4

        quart = []
        for k in range(4):
            s = q == k
            lo, hi = boot_excess(off_c[s], off_p[s], rng)
            quart.append({
                "quartile": k + 1,
                "n": int(s.sum()),
                "median_margin": float(np.median(m[s])),
                "guard_rate": float(off_c[s].mean()),
                "placebo_rate": float(off_p[s].mean()),
                "excess": float(off_c[s].mean() - off_p[s].mean()),
                "excess_ci": [lo, hi],
                "turns_off_corrupt": int(off_c[s].sum()),
                "turns_off_placebo": int(off_p[s].sum()),
                # the confound control: if the strata differ in difficulty the
                # stratification is not clean and the reader must see it.
                "median_n_known": float(np.median([r["n_known"] for r, t
                                                   in zip(on, s) if t])),
                "median_round0_rank": float(np.median([r["round0_rank"] for r, t
                                                       in zip(on, s) if t])),
            })

        # overlap classes by margin (91.6)
        cls = {"corrupt_only": [], "placebo_only": [], "both": [], "neither": []}
        for r, mi in zip(on, m):
            c, p = (not r["corrupt_emits_v"]), (not r["placebo_emits_v"])
            cls[("both" if p else "corrupt_only") if c
                else ("placebo_only" if p else "neither")].append(mi)

        drop_c = np.abs([r["corrupt_v_logit_drop"] for r in on])
        drop_p = np.abs([r["placebo_v_logit_drop"] for r in on])
        thin = float(np.mean(m < np.median(drop_c)))

        per_arm[arm] = {
            "denom_clean_on": len(on),
            "guard_rate": float(off_c.mean()),
            "placebo_rate": float(off_p.mean()),
            "median_margin": float(np.median(m)),
            "auroc_margin_predicts_corrupt_off": auroc(-m, off_c.astype(bool)),
            "auroc_margin_predicts_placebo_off": auroc(-m, off_p.astype(bool)),
            "median_abs_drop_corrupt": float(np.median(drop_c)),
            "median_abs_drop_placebo": float(np.median(drop_p)),
            "frac_margin_below_median_drop": thin,
            "quartiles": quart,
            "overlap_median_margin": {k: (float(np.median(v)) if v else float("nan"))
                                      for k, v in cls.items()},
            "overlap_n": {k: len(v) for k, v in cls.items()},
        }

        print(f"--- seed {seed}  {arm} " + "-" * 40)
        print(f"  clean-emitting cases (THE DENOMINATOR, conditional on "
              f"clean_emits_v): {len(on)}")
        print(f"  raw guard {off_c.mean():.4f}   placebo {off_p.mean():.4f}   "
              f"median margin {np.median(m):.3f} logits")
        print(f"  AUROC of -margin for turn-off:  corrupt "
              f"{per_arm[arm]['auroc_margin_predicts_corrupt_off']:.4f}   "
              f"placebo {per_arm[arm]['auroc_margin_predicts_placebo_off']:.4f}")
        print(f"  median |logit drop| corrupt {np.median(drop_c):.4f}; share of "
              f"clean-on cases with margin below it {thin:.4f}")
        print(f"  {'Q':<3}{'n':>5}{'medMrg':>9}{'guard':>8}{'placebo':>9}"
              f"{'excess':>9}{'95% CI':>20}{'nKnown':>8}{'rank':>6}")
        for qq in quart:
            print(f"  {qq['quartile']:<3}{qq['n']:>5}{qq['median_margin']:>9.3f}"
                  f"{qq['guard_rate']:>8.4f}{qq['placebo_rate']:>9.4f}"
                  f"{qq['excess']:>9.4f}"
                  f"   [{qq['excess_ci'][0]:+.4f}, {qq['excess_ci'][1]:+.4f}]"
                  f"{qq['median_n_known']:>8.1f}{qq['median_round0_rank']:>6.1f}")
        o, on_ = per_arm[arm]["overlap_median_margin"], per_arm[arm]["overlap_n"]
        print(f"  median margin by overlap class: corrupt_only {o['corrupt_only']:.3f} "
              f"(n={on_['corrupt_only']}), placebo_only {o['placebo_only']:.3f} "
              f"(n={on_['placebo_only']}), both {o['both']:.3f} (n={on_['both']}), "
              f"neither {o['neither']:.3f} (n={on_['neither']})", flush=True)

    per_seed[seed] = per_arm

with open(os.path.join(OUT_DIR, "marginstrat_summary.json"), "w") as fh:
    json.dump({"per_seed": {str(k): v for k, v in per_seed.items()},
               "source": {s: guardspec_dir(s) for s in SEEDS},
               "n_boot": N_BOOT, "boot_seed": BOOT_SEED}, fh, indent=2)
print(f"\nwrote {OUT_DIR}/marginstrat_summary.json")

print("\n" + "=" * 78)
print("THE REGISTERED PREDICTIONS, scored mechanically. No verdict is printed.")
L1 = "warm L1"
p1 = [(per_seed[s][L1]["median_margin"], per_seed[s]["Y"]["median_margin"])
      for s in SEEDS]
print(f"91.1 median margin L1 < Y: "
      + "  ".join(f"s{s} {a:.3f} vs {b:.3f} {'HIT' if a < b else 'MISS'}"
                  for s, (a, b) in zip(SEEDS, p1)))
for tag, key, bar in (("91.2 corrupt", "auroc_margin_predicts_corrupt_off", 0.85),
                      ("91.3 placebo", "auroc_margin_predicts_placebo_off", 0.80)):
    v = [per_seed[s][L1][key] for s in SEEDS]
    print(f"{tag} AUROC >= {bar}: "
          + "  ".join(f"s{s} {x:.4f} {'HIT' if x >= bar else 'MISS'}"
                      for s, x in zip(SEEDS, v)))
q4 = [per_seed[s][L1]["quartiles"][3] for s in SEEDS]
q1 = [per_seed[s][L1]["quartiles"][0] for s in SEEDS]
print("91.4 Q4 guard <= 0.02: "
      + "  ".join(f"s{s} {x['guard_rate']:.4f} of {x['n']} "
                  f"{'HIT' if x['guard_rate'] <= 0.02 else 'MISS'}"
                  for s, x in zip(SEEDS, q4)))
print("91.5 Q4 |excess| <= 0.02: "
      + "  ".join(f"s{s} {x['excess']:+.4f} "
                  f"{'HIT' if abs(x['excess']) <= 0.02 else 'MISS'}"
                  for s, x in zip(SEEDS, q4)))
print("     Q1 excess >= 0.05 on >= 2 of 3: "
      + "  ".join(f"s{s} {x['excess']:+.4f} "
                  f"{'HIT' if x['excess'] >= 0.05 else 'MISS'}"
                  for s, x in zip(SEEDS, q1))
      + f"   [{sum(x['excess'] >= 0.05 for x in q1)} of 3]")
p6 = []
for s in SEEDS:
    o = per_seed[s][L1]["overlap_median_margin"]
    off = {k: o[k] for k in ("corrupt_only", "placebo_only", "both")
           if o[k] == o[k]}          # drop a class the arm has no cases in
    p6.append(f"s{s} both {o['both']:.3f} vs corrupt_only {o['corrupt_only']:.3f} "
              f"placebo_only {o['placebo_only']:.3f} "
              f"{'HIT' if off.get('both') == min(off.values()) else 'MISS'}")
print("91.6 `both` has the lowest median margin:\n     " + "\n     ".join(p6))
print("\nFALSIFIERS: 91.2 below 0.70, or a Q4 guard above 0.08, would say the")
print("churn is not a tie-break. A difficulty gradient down the nKnown or rank")
print("columns is a CONFOUND in the stratification and is reported as one.")
