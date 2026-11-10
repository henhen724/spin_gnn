"""STAGE Q -- does the ATTENTION-ONLY cell actually do direct deduction?

THIS DOCSTRING IS THE REGISTRATION. It is committed before the first arm runs.

WHY THIS STAGE EXISTS, AND WHY IT IS NOW THE PROGRAMME'S MAIN GAP. The brief
changed on 2026-11-10: the goal is the simplest / most analytically tractable
architecture THAT STILL LEARNS DIRECT DEDUCTION. That makes "still learns DD"
the GATE, not a nice-to-have -- and it has never been measured on an
attention-only cell. Findings 27/28 (350/350 round-0 counterfactual flips,
0/350 emissions with the forcing broken, 0.980 chaining) were all run on
FULL-architecture leak-free checkpoints at N=24. Every `no_mlp` claim in this
repo is a claim about matching a SOLVE CURVE. "Attention alone does the
deduction" has been written down but never tested.

THE COUNTERFACTUAL (unchanged, spin_gnn.sudoku_circuits.find_flippable_deductions).
Node u is forced when its known neighbours cover q-1 colours leaving c free.
Recolour the one neighbour uniquely holding colour a to c; now a is the free
colour, so DD's answer for u flips c -> a. Exactly ONE token changes, and it is
a hint: u's token, the graph table and the reasoning prefix are untouched. A
model that computes the forcing rule must follow the flip; a model that reached
the same solve curve some other way has no reason to.

THE ARMS. All four are leak-free (shuffle_adj=True) and FALL, at N=48, d=5,
q=4, same --seed-start, so the graphs, colourings and hints are the same
instances across arms.
  Q_noff_s0/s1/s2  no_mlp L=16 d192, the standing cell J2 (rho_50 0.580, 3/3)
  Q_full_ctrl      full   L=12 d256, the best solver at any size (rho_50 0.385)

--------------------------------------------------------------------------
REGISTERED HYPOTHESES, falsifiers fixed before any number exists.

Q3  PRECONDITION, READ FIRST. clean_correct is read BEFORE the flip rate. A
    model that gets the clean case wrong has no answer to flip, so a low flip
    rate on top of a low clean_correct says nothing about the forcing rule --
    it says the case set is too hard for the cell. Below 0.50 clean_correct the
    audit is INCONCLUSIVE for that arm and the move is a higher-rho case set,
    not a null result. This is first for the same reason validation accuracy is
    read before solve rate everywhere else in this programme: it cannot censor.

Q0  THE GATE / THE HEADLINE. The attention-only cell responds causally to the
    single forced fact: corrupt_flipped >= 0.60, far above the 1/q = 0.25
    chance rate, on at least 2 of 3 seeds.
    REASON TO EXPECT IT: finding 36 already showed that deleting the MLP WRITE
    at layers 0-10 of the trained FULL solver retains 0.97-1.01 of the
    counterfactual effect -- i.e. in the architecture where 27/28 was measured,
    the MLP is not what carries the forcing computation. An architecture with
    no MLP at all should therefore keep the mechanism.
    FALSIFIER: a flip rate at or near 0.25. That would mean the attention-only
    cell reaches J2's solve curve by something that does NOT respond to the one
    fact DD says decides the answer, and the claim "attention alone does direct
    deduction" must be WITHDRAWN -- leaving the 6-layer `full L=6 d256` hub
    (0.592, 4,899,898) as the tractable candidate instead, since it is the
    shallowest cell whose family HAS been audited. That is the most important
    possible outcome of this stage and it is the one that changes the plan.

Q1  THE LEVEL IS NOT COMPARABLE TO FINDINGS 27/28. 350/350 was N=24 at
    shared-pool 1.0000, a model that solves its task perfectly. The N=48 cells
    do not: J2's rho_50 is 0.580, so at the case set's rho it is working near
    its limit. A flip rate BELOW 350/350 is therefore EXPECTED at N=48 and is
    not evidence against Q0. What is licensed is arm-to-arm comparison WITHIN
    N=48; what is not licensed is N=48 against the N=24 number -- the same rule
    finding 55 imposed when it forbade pooling N=48 and N=96 rho_50. Registered
    now so that a lower number cannot be reinterpreted as a failure after the
    fact, nor as a success.

Q2  THE CONTROL IS SHAPE-CONFOUNDED, AND THIS IS SAID BEFORE THE NUMBERS. No
    `full L=16 d192` exists at N=48 -- the only cells on disk at J2's shape are
    its own three seeds. So the control differs from the arm in BOTH arch_mode
    AND shape (L=12 d256 vs L=16 d192). Stage M measured a large SUB-ADDITIVE
    arch x shape interaction at N=96 (interaction +0.389 mean_solve = 2.45
    bars), so an arm-minus-control difference here cannot be attributed to the
    MLP alone. CONSEQUENCE, BINDING: Q0 is scored against the ABSOLUTE chance
    rate 0.25 and against clean_correct, NEVER against the control. The control
    arm is context -- it says whether the case set is answerable at all at N=48
    by the family that 27/28 validated -- and it is reported as context.

Q4  THE SEED STORY. Finding 56: three seeds cannot find a stuck seed but DO
    resolve a real step. All three J2 seeds are audited and the spread is quoted
    raw. If the flip rate varies across seeds by more than the arm-to-control
    gap, the mechanism claim is a SEED-level claim, not an architecture-level
    one, and must be written that way.

WHAT THIS STAGE DOES NOT DO. Behaviour only: no patching, no head ablation, no
sublayer ablation. Those localize a mechanism; this stage asks whether there is
one to localize, which is the gate. Localization is a follow-up and is cheap to
add once the gate is known, and it needs a GPU to be worth running.

WHAT DID NOT CHANGE: the free line 0.740, s* = 0.1484, the 14-point grid, the
plateau criterion 1.64, the LOO bar 0.0423, three seeds with the spread quoted.
This stage adds a gate; it loosens nothing.
"""

import csv
import json
import os
import sys

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
OUT = os.path.join(_SCR, "spin_gnn", "sudoku_n48_ddaudit_Q_2026-11-10")

CHANCE = 0.25          # 1/q, q = 4
GATE = 0.60            # Q0's threshold, fixed at registration
CLEAN_FLOOR = 0.50     # Q3's precondition

# (dirname, label, arch_mode, num_layers, d_model, seed, is_control)
ARMS = [
    ("Q_noff_s0", "J2  no_mlp L=16 d192 s0", "no_mlp", 16, 192, 0, False),
    ("Q_noff_s1", "J2  no_mlp L=16 d192 s1", "no_mlp", 16, 192, 1, False),
    ("Q_noff_s2", "J2  no_mlp L=16 d192 s2", "no_mlp", 16, 192, 2, False),
    ("Q_full_ctrl", "CTRL full L=12 d256 s0", "full", 12, 256, 0, True),
]

# Frozen so the arm cannot quietly become a different cell. rho_50 at N=48.
REGISTERED_RHO50 = {"J2  no_mlp L=16 d192 s0": "0.580 (3-seed median)",
                    "CTRL full L=12 d256 s0": "0.385 (3-seed median)"}


def _read(arm_dir):
    p = os.path.join(OUT, arm_dir, "behavior.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    n = len(rows)
    tf = lambda r, k: r[k].strip().lower() in ("true", "1")
    clean = sum(tf(r, "clean_correct") for r in rows) / n
    flip = sum(tf(r, "corrupt_flipped") for r in rows) / n
    both = sum(tf(r, "clean_correct") and tf(r, "corrupt_flipped") for r in rows) / n
    ld_c = sum(float(r["clean_logit_diff"]) for r in rows) / n
    ld_k = sum(float(r["corrupt_logit_diff"]) for r in rows) / n
    return {"n": n, "clean": clean, "flip": flip, "both": both,
            "ld_clean": ld_c, "ld_corrupt": ld_k}


def main():
    print(__doc__.split("--------")[0].rstrip())
    print("=" * 74)
    print("arms read from %s" % OUT)
    print()

    res = {}
    for d, label, arch, L, dm, seed, is_ctrl in ARMS:
        r = _read(d)
        res[label] = r
        print("=== %s%s ===" % (label, "   [CONTROL -- context only, see Q2]" if is_ctrl else ""))
        if r is None:
            print("  PENDING -- no behavior.csv on disk yet")
            print()
            continue
        print("  %d counterfactuals   registered rho_50 %s"
              % (r["n"], REGISTERED_RHO50.get(label, "n/a")))
        print("  clean_correct    %.3f      <- Q3 reads this FIRST" % r["clean"])
        print("  corrupt_flipped  %.3f      (chance %.3f, Q0 gate %.2f)"
              % (r["flip"], CHANCE, GATE))
        print("  both (uses rule) %.3f" % r["both"])
        print("  mean logit_diff  clean %+.3f   corrupt %+.3f   (gap %.3f)"
              % (r["ld_clean"], r["ld_corrupt"], r["ld_corrupt"] - r["ld_clean"]))
        print()

    arm_labels = [a[1] for a in ARMS if not a[6]]
    ctrl_label = [a[1] for a in ARMS if a[6]][0]
    arms = [res[l] for l in arm_labels if res[l] is not None]

    print("Q3 -- PRECONDITION: clean_correct >= %.2f before the flip rate means anything"
          % CLEAN_FLOOR)
    if not arms:
        print("  no attention-only arm on disk. Nothing below is readable.")
        return 0
    bad = [l for l in arm_labels if res[l] is not None and res[l]["clean"] < CLEAN_FLOOR]
    for l in arm_labels:
        if res[l] is not None:
            print("    %-26s clean_correct %.3f  %s"
                  % (l, res[l]["clean"], "BELOW FLOOR" if res[l]["clean"] < CLEAN_FLOOR else "ok"))
    if bad:
        print("  Q3 FAILS on %d of %d attention-only seed(s)." % (len(bad), len(arm_labels)))
        print("  For those seeds the audit is INCONCLUSIVE, not negative: the cell has no")
        print("  answer to flip on this case set. The registered move is a higher-rho case")
        print("  set, NOT a null result, and Q0 below is not scored on them.")
    else:
        print("  Q3 HELD on every attention-only seed. The flip rate is interpretable.")
    print()

    scorable = [l for l in arm_labels
                if res[l] is not None and res[l]["clean"] >= CLEAN_FLOOR]
    print("Q0 -- THE GATE: corrupt_flipped >= %.2f, far above chance %.2f, on >= 2 of 3 seeds"
          % (GATE, CHANCE))
    print("  scored against the ABSOLUTE chance rate, never against the control (Q2)")
    if len(scorable) < 2:
        print("  NOT SCORABLE: only %d of 3 seeds passed Q3." % len(scorable))
        print("  Report that, not a verdict on the architecture.")
    else:
        passing = [l for l in scorable if res[l]["flip"] >= GATE]
        for l in scorable:
            f = res[l]["flip"]
            print("    %-26s flip %.3f  = %.2fx chance   %s"
                  % (l, f, f / CHANCE, "PASS" if f >= GATE else "fail"))
        if len(passing) >= 2:
            print("  Q0 HELD on %d of %d scorable seeds. The attention-only cell DOES respond"
                  % (len(passing), len(scorable)))
            print("  causally to the one fact DD says decides the answer. 'Attention alone does")
            print("  direct deduction' is licensed AT N=48, and the tied cell of stage P is")
            print("  asking a question about a mechanism that is there.")
        else:
            print("  Q0 FALSIFIED on %d of %d scorable seeds." % (len(scorable) - len(passing),
                                                                 len(scorable)))
            print("  WITHDRAW 'attention alone does direct deduction'. What is measured is that")
            print("  it MATCHES THE SOLVE CURVE. Per the registration the tractable candidate")
            print("  becomes the 6-layer `full L=6 d256` hub (0.592, 4,899,898), whose family is")
            print("  the one findings 27/28 actually audited. This outcome changes the plan and")
            print("  is reported as the headline, not as a footnote.")
    print()

    print("Q1 -- GUARD: the LEVEL is not comparable to findings 27/28's 350/350")
    print("  that was N=24 at shared-pool 1.0000; these cells are N=48 and do not solve")
    print("  their task (J2 rho_50 0.580). A flip rate below 350/350 is EXPECTED here and")
    print("  is neither a failure nor a success. Arm-to-arm within N=48 only.")
    print()

    print("Q4 -- THE SEED STORY (finding 56: 3 seeds resolve a step, not a stuck seed)")
    if len(arms) >= 2:
        fl = [res[l]["flip"] for l in arm_labels if res[l] is not None]
        spread = max(fl) - min(fl)
        print("  per-seed flip: %s" % ", ".join("%.3f" % f for f in fl))
        print("  seed spread %.3f" % spread)
        c = res[ctrl_label]
        if c is not None:
            gap = abs(sum(fl) / len(fl) - c["flip"])
            print("  |mean arm - control| %.3f   (control flip %.3f)" % (gap, c["flip"]))
            if spread > gap:
                print("  SEED SPREAD EXCEEDS THE ARM-CONTROL GAP. Any mechanism claim here is a")
                print("  SEED-level claim, not an architecture-level one, and is written that way.")
            else:
                print("  spread is smaller than the gap: an architecture-level statement is")
                print("  admissible, subject to Q2's shape confound.")
        else:
            print("  control PENDING -- the gap is not yet computable.")
    else:
        print("  fewer than 2 attention-only arms on disk; no spread.")
    print()

    print("Q2 -- GUARD: the control is shape-confounded (no full L=16 d192 exists at N=48)")
    print("  control is full L=12 d256, differing in arch_mode AND shape. Stage M measured a")
    print("  2.45-bar sub-additive arch x shape interaction at N=96, so an arm-minus-control")
    print("  difference here is NOT attributable to the MLP alone. Context, not a verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
