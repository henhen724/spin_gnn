"""Stage R -- does the attention-only cell CHAIN, or only apply the rule once?

THIS DOCSTRING IS THE REGISTRATION. It is committed before the first arm runs.

WHAT STAGE Q SETTLED AND WHAT IT DID NOT. Q measured 350 round-0
counterfactuals on three seeds of J2 (`no_mlp L=16 d192`, rho_50 0.580) and the
`full L=12 d256` control: clean_correct 1.000 everywhere, corrupt_flipped
1.000/0.997/1.000 against a 0.250 chance rate, mean logit gap -17.228 -> +16.508.
Q0 held 3/3 and "attention alone does direct deduction" stopped being an
inference from a solve curve. But `find_flippable_deductions` selects nodes
forced in ROUND 0 whose every known neighbour is a HINT, so what Q established
is that the elimination RULE is applied perfectly over GIVEN facts. That is one
third of findings 27/28. The 0.980 CHAINING number in 27/28 -- also measured on
a full-architecture N=24 checkpoint, never on a `no_mlp` one -- is this stage.

Chaining is the half that matters for the new objective. DD proceeds in
synchronous rounds; a model that applies the rule once is a one-step classifier,
and only a model that eliminates against facts IT DERIVED is running the
algorithm. Stage P asks whether the 16 layers are one operator iterated 16
times; that question is only worth asking if the cell integrates rounds at all.

THE ARMS. The same four checkpoints as stage Q, the same `--seed-start`, hence
the same instances. `spin_gnn/sudoku_chain.py` differs from Q in exactly one
respect: the edited token is a previously EMITTED DEDUCTION in the reasoning
section rather than a hint. One token changes; DD's answer for v flips from its
true colour c to the freed colour a.

REGISTERED BEFORE ANY NUMBER EXISTS, AND SCORED IN THIS ORDER.

R3, the precondition, read FIRST. A cell that gets the clean chained case wrong
has no answer to flip. Below CLEAN_FLOOR = 0.50 clean_correct the audit is
INCONCLUSIVE, not negative. Chained cases are strictly harder than round-0 ones
-- v is forced only after earlier rounds land -- so this is the realistic place
for the stage to come back unscorable, and the move would be a higher-rho case
set, not a null result.

R1, the ambiguity, read SECOND and BEFORE any verdict on a low flip rate.
`sudoku_chain`'s own docstring states it: editing an emitted deduction makes the
prefix inconsistent with the graph and hints, so a model that NOTICES and keeps
its own value scores as "does not chain" while chaining perfectly in free
generation. Registered rule: if flip < GATE and reverts_to_original > flip, the
verdict is DETECTION-AMBIGUOUS, never "does not chain", and the follow-up that
resolves it is free generation, not more cases. A HIGH flip rate carries no such
ambiguity -- it can only mean the edited derived fact was used.

R0, THE GATE. corrupt_flipped >= GATE = 0.60, far above CHANCE = 0.25, on at
least 2 of the 3 J2 seeds. HELD -> the attention-only cell eliminates against a
fact it derived, so "attention alone does direct deduction" covers the
ITERATION and not just the rule, and stage P's tied cell is asking about a
mechanism that spans rounds. FALSIFIED -> the licensed claim shrinks to "applies
the one-step elimination rule", the 16 layers are doing something other than 16
rounds of DD, and that is a hard constraint on any tied reading rather than a
disappointment.

R2, THE CEILING RULE, written in BECAUSE STAGE Q WAS CAUGHT BY IT. Q4 asked
whether the seed spread exceeded the arm-control gap; every Q arm saturated at
1.000, so the rule fired on a 0.003 spread against a 0.001 gap and carried no
information at all. Registered here in advance: per-seed claims and arm-vs-
control claims are made ONLY IF at least one arm falls below CEILING = 0.95.
Otherwise this stage reports "all arms at ceiling, no resolution" and declines
to rank seeds. Inherited from Q2 and still binding: no `full L=16 d192` exists
at N=48, so the control differs in BOTH arch_mode and shape, and R0 is scored
against the absolute chance rate, never against the control.

R4, the ONE licensed cross-stage comparison. Q and R run on the same instance
stream and differ in exactly one respect, the PROVENANCE of the edited fact, so
WITHIN ONE ARM the chained flip rate minus stage Q's round-0 flip rate is a
measurement of what provenance costs. Registered: |delta| <= PROVENANCE_BAR =
0.10 means derived facts are used as readily as given ones. This is the only
cross-stage subtraction the programme licenses -- Q1 and finding 55 forbid
comparing across N and across case sets, and that prohibition stands.

R5, the round profile, a registration rather than a test. `by_round` gives the
flip rate against the round v was deduced in. If the cell chains by composing
layers, flip should not decay with round up to the layer budget, and L=16
exceeds any round these traces reach. A monotone decay is the falsifier and
would be the most useful outcome for a theory: it would tie rounds to layers
quantitatively instead of by analogy.

Behaviour only. No patching, no heads, no sublayer: those localize a mechanism,
and `_capture_clean` is still wrong for tied cells (open problem in STATUS.md).
CPU, because stage O holds GPUs with stage P on them too, and putting the gate
behind the runs it interprets would be backwards.
"""

import csv
import json
import os

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
OUT = os.path.join(_SCR, "spin_gnn", "sudoku_n48_chain_R_2026-11-10")
QOUT = os.path.join(_SCR, "spin_gnn", "sudoku_n48_ddaudit_Q_2026-11-10")

CHANCE = 0.25           # 1/q, q = 4
GATE = 0.60             # R0's threshold, fixed at registration
CLEAN_FLOOR = 0.50      # R3's precondition
CEILING = 0.95          # R2: above this on every arm, no seed-level claims
PROVENANCE_BAR = 0.10   # R4: round-0 vs chained, within one arm

#   dir tag        label                         stage Q's tag
ARMS = [
    ("R_noff_s0", "J2  no_mlp L=16 d192 s0", "Q_noff_s0"),
    ("R_noff_s1", "J2  no_mlp L=16 d192 s1", "Q_noff_s1"),
    ("R_noff_s2", "J2  no_mlp L=16 d192 s2", "Q_noff_s2"),
    ("R_full_ctrl", "CTRL full L=12 d256 s0", "Q_full_ctrl"),
]
SEED_ARMS = [a for a in ARMS if "CTRL" not in a[1]]
REGISTERED_RHO50 = {"J2  no_mlp L=16 d192 s0": "0.580 (3-seed median)",
                    "CTRL full L=12 d256 s0": "0.385 (3-seed median)"}


def read_chain(tag):
    p = os.path.join(OUT, tag, "chain_summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def read_stageq_flip(qtag):
    """Stage Q's round-0 flip rate for the same checkpoint, for R4."""
    p = os.path.join(QOUT, qtag, "behavior.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return sum(r["corrupt_flipped"] == "True" for r in rows) / len(rows)


def main():
    print("=" * 74)
    print("STAGE R -- does the attention-only cell CHAIN?  (registration above)")
    print("=" * 74)

    got = {}
    for tag, label, qtag in ARMS:
        s = read_chain(tag)
        if s is None:
            print("  %-26s PENDING" % label)
            continue
        got[label] = s
        print("  %-26s n=%d  clean %.3f  flip %.3f  revert %.3f  both %.3f"
              % (label, s["n_cases"], s["clean_correct"], s["corrupt_flipped"],
                 s["reverts_to_original"], s["both"]))
    if not got:
        print("\nnothing to score yet.")
        return
    print()

    # ---------------------------------------------------------------- R3
    print("-" * 74)
    print("R3 -- THE PRECONDITION, READ FIRST: clean_correct >= %.2f" % CLEAN_FLOOR)
    print("  chained cases are HARDER than round-0 ones; below the floor this")
    print("  stage is INCONCLUSIVE, not negative.")
    scorable = []
    for tag, label, qtag in ARMS:
        if label not in got:
            continue
        c = got[label]["clean_correct"]
        ok = c >= CLEAN_FLOOR
        print("    %-26s clean_correct %.3f   %s" % (label, c, "OK" if ok else "BELOW FLOOR"))
        if ok:
            scorable.append(label)
    if not any(l in scorable for l in (a[1] for a in SEED_ARMS)):
        print("\n  NO J2 SEED CLEARS THE PRECONDITION -> STAGE R IS INCONCLUSIVE.")
        print("  Not a negative result. The move is a higher-rho case set.")
        return
    print("  R3 HELD on %d of %d arms." % (len(scorable), len(got)))
    print()

    # ---------------------------------------------------------------- R1
    print("-" * 74)
    print("R1 -- THE AMBIGUITY, READ BEFORE ANY VERDICT ON A LOW FLIP RATE")
    print("  editing an emitted deduction makes the prefix inconsistent, so a")
    print("  model that DETECTS the edit and keeps its own answer scores as")
    print("  'does not chain'. reverts_to_original is what tells those apart.")
    ambiguous = []
    for tag, label, qtag in ARMS:
        if label not in scorable:
            continue
        s = got[label]
        flip, rev = s["corrupt_flipped"], s["reverts_to_original"]
        if flip < GATE and rev > flip:
            ambiguous.append(label)
            print("    %-26s flip %.3f < %.2f AND revert %.3f > flip"
                  "  -> DETECTION-AMBIGUOUS" % (label, flip, GATE, rev))
        elif flip < GATE:
            print("    %-26s flip %.3f < %.2f, revert %.3f <= flip"
                  "  -> a genuine failure to chain" % (label, flip, GATE, rev))
        else:
            print("    %-26s flip %.3f >= %.2f  -> no ambiguity to resolve"
                  % (label, flip, GATE))
    if ambiguous:
        print("\n  %d arm(s) DETECTION-AMBIGUOUS. Per the registration this is NOT" % len(ambiguous))
        print("  'does not chain', and the follow-up is free generation, not more cases.")
    print()

    # ---------------------------------------------------------------- R0
    print("-" * 74)
    print("R0 -- THE GATE: corrupt_flipped >= %.2f, far above chance %.2f, on >= 2 of 3 seeds"
          % (GATE, CHANCE))
    print("  scored against the ABSOLUTE chance rate, never against the control (R2)")
    passes = 0
    for tag, label, qtag in SEED_ARMS:
        if label not in scorable:
            print("    %-26s NOT SCORABLE (R3)" % label)
            continue
        flip = got[label]["corrupt_flipped"]
        ok = flip >= GATE
        passes += ok
        print("    %-26s flip %.3f  = %.2fx chance   %s"
              % (label, flip, flip / CHANCE, "PASS" if ok else "FAIL"))
    n_scorable = sum(1 for _, l, _ in SEED_ARMS if l in scorable)
    if passes >= 2:
        print("\n  R0 HELD on %d of %d scorable seeds. The attention-only cell ELIMINATES" % (passes, n_scorable))
        print("  AGAINST A FACT IT DERIVED. 'Attention alone does direct deduction'")
        print("  covers the ITERATION, not only the one-step rule, and stage P's tied")
        print("  cell is asking about a mechanism that spans rounds.")
    else:
        print("\n  R0 FALSIFIED (%d of %d). The licensed claim SHRINKS to 'applies the" % (passes, n_scorable))
        print("  one-step elimination rule over given facts'. Whatever the 16 layers")
        print("  do, it is not 16 rounds of DD -- a hard constraint on any tied")
        print("  reading, and read R1 before calling this a failure to chain.")
    print()

    # ---------------------------------------------------------------- R2
    print("-" * 74)
    print("R2 -- THE CEILING RULE (registered because stage Q's Q4 was caught by it)")
    flips = {l: got[l]["corrupt_flipped"] for _, l, _ in ARMS if l in got}
    below = [l for l, v in flips.items() if v < CEILING]
    if not below:
        print("  every arm >= %.2f. NO RESOLUTION for seed-level or arm-vs-control" % CEILING)
        print("  claims, and per the registration this stage DECLINES to rank seeds.")
        print("  (Q4 fired on a 0.003 spread against a 0.001 gap; that is the error")
        print("   this rule exists to prevent repeating.)")
    else:
        seed_vals = [flips[l] for _, l, _ in SEED_ARMS if l in flips]
        ctrl = [flips[l] for _, l, _ in ARMS if l in flips and "CTRL" in l]
        spread = max(seed_vals) - min(seed_vals) if len(seed_vals) > 1 else 0.0
        print("  %d arm(s) below the ceiling -> seed-level comparison HAS resolution." % len(below))
        print("  per-seed flip: %s / spread %.3f"
              % (", ".join("%.3f" % v for v in seed_vals), spread))
        if ctrl:
            gap = abs(sum(seed_vals) / len(seed_vals) - ctrl[0])
            print("  |mean arm - control| %.3f   (control flip %.3f)" % (gap, ctrl[0]))
            print("  NB R2/Q2: the control differs in BOTH arch_mode and shape, so this")
            print("  gap is context and never the test.")
    print()

    # ---------------------------------------------------------------- R4
    print("-" * 74)
    print("R4 -- PROVENANCE: chained flip minus stage Q's ROUND-0 flip, WITHIN one arm")
    print("  the only cross-stage subtraction this programme licenses: same")
    print("  instances, same seed-start, one difference (hint vs emitted deduction)")
    for tag, label, qtag in ARMS:
        if label not in got:
            continue
        q = read_stageq_flip(qtag)
        if q is None:
            print("    %-26s stage Q arm missing -- delta UNAVAILABLE" % label)
            continue
        d = got[label]["corrupt_flipped"] - q
        verdict = ("derived facts used as readily as given ones"
                   if abs(d) <= PROVENANCE_BAR else "PROVENANCE COSTS IT")
        print("    %-26s round-0 %.3f -> chained %.3f   delta %+.3f   %s"
              % (label, q, got[label]["corrupt_flipped"], d, verdict))
    print()

    # ---------------------------------------------------------------- R5
    print("-" * 74)
    print("R5 -- THE ROUND PROFILE (a registration): flip vs the round v was deduced in")
    print("  composing layers predicts NO decay up to the layer budget; L=16")
    print("  exceeds any round these traces reach. Monotone decay is the")
    print("  falsifier and would tie rounds to layers quantitatively.")
    for tag, label, qtag in ARMS:
        if label not in got:
            continue
        br = got[label].get("by_round", {})
        prof = "  ".join("r%s n=%d flip %.3f" % (k, br[k]["n"], br[k]["flipped"])
                         for k in sorted(br, key=int))
        print("    %-26s %s" % (label, prof or "(no by_round)"))
    print()

    print("-" * 74)
    print("rho_50 of the audited cells, for the record (NOT measured here):")
    for k, v in REGISTERED_RHO50.items():
        print("    %-26s %s" % (k, v))
    print("=" * 74)


if __name__ == "__main__":
    main()
