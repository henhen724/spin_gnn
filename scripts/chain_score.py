#!/usr/bin/env python3
"""PRE-REGISTERED SCORER for the chained test -- finding 69 moved off round 0.

This docstring IS the registration. It is committed before the measurement job
is submitted, and the verdict block below is the only thing allowed to decide
what the deck says.

THE QUESTION. Finding 69 established, at the same six checkpoints, that one edit
to a HINT moves the tied model's elimination (p_freed 0.5253 / 0.5228 / 0.5354
under the broken edit, p_cand exactly 1.0000) and leaves its node schedule
untouched. Every fact it eliminated against was GIVEN. Direct deduction is
iterative; this asks whether the same machinery runs over a fact the model
DERIVED, by moving all four edits into the reasoning section and requiring the
binding neighbour to have been emitted as a deduction earlier in the prefix.

WHAT IS REGISTERED, BEFORE ANY NUMBER EXISTS.

  C1  CHAINING, the headline. The rule edit (deduced w: a -> c) flips the
      emitted colour at v's colour position to a in at least RULE_BAR = 0.60 of
      cases, in all six cells, against chance 1/q = 0.250. Round 0 scored
      1.0000 in all six. C1 passing means the elimination rule runs over the
      model's own output, which is what "it chains" has to mean causally.

  C2  THE DISSOCIATION, tied arm only, as in finding 69. p_freed under the
      broken edit exceeds p_freed under the placebo: paired sign test
      significant after Holm across the three seeds, AND median difference at
      least FREED_BAR = 0.05, in at least 2 of 3 seeds. Direction fixed here.

  C3  THE FALSIFIER, and it is a fork, not a null. If C1 misses, `rule_reverts`
      decides between the two readings, because editing a derived fact makes the
      prefix globally inconsistent in a way a hint edit never is:
        rule_reverts > REVERT_BAR = 0.60  ->  the model DETECTS the edited
            prefix and keeps its own answer. Published conclusion: the chained
            test is confounded by detection, finding 69 stays round-0 only, and
            no claim is made about chaining in either direction.
        rule_reverts <= REVERT_BAR        ->  it neither follows the edit nor
            reverts. Published conclusion: provenance matters, the rule the
            tied model learned applies to given facts and not to derived ones.
      Both of those are the result. Neither licenses a new arm or a third edit.

  C4  SPECIFICITY. The placebo -- another deduced, non-neighbour node recoloured
      to a locally proper colour -- moves p_freed by at most FREED_BAR = 0.05.
      Otherwise C2 has no null and nothing below is read.

  C5  THE PROVENANCE GAP, the cell-level comparison that is the point of the
      run. For each cell, chained rule-flip against finding 69's round-0
      rule-flip at the SAME checkpoint. A provenance gap is called only if the
      chained rate is lower by more than GAP_BAR = 0.10 in at least 2 of 3 seeds
      of an arm. THIS IS NOT PAIRED: the chained case list is a different list
      by construction (round k >= 1, binding neighbour deduced), so this is a
      comparison of two rates at one checkpoint and never a per-case difference.

GUARDS, each of which fails the job rather than printing a number.
  1. Every checkpoint is the cell it is filed under and the arms agree on every
     instance-generating key (guard_configs, shared with findings 68 and 69).
  2. All six case lists are identical row for row on CHAIN_KEY. Six cells that
     disagree on one case are not six measurements of one thing.
  3. Every case is round >= 1 with a binding neighbour that is itself a
     deduction -- the construction's defining property, re-checked from the CSV
     rather than trusted from the module that wrote it.

NOTHING IS TRAINED. rho_50, the gate, the free line and every stage result are
untouched by this file and by the job that feeds it.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ddguard_score import (ALPHA, ARMS, CKPT, RULE_BAR, SEEDS, binom_exact,
                                   guard_configs, holm, ratio, rows_of, truthy)
from scripts.dissoc_score import FREED_BAR, _num, fmean, sign_test

RUN = "/scratch/users/hshunt/spin_gnn/runs/2026-11-13_chain"
REVERT_BAR = 0.60
GAP_BAR = 0.10
# The case key for this construction. `round` and `nbr_round` are in it because
# they are what makes a case chained; two cells that agreed on the node but not
# on the round would not be measuring the same deduction.
CHAIN_KEY = ("node", "round", "round_rank", "binding_nbr", "nbr_round", "n_known")

# Finding 69's published round-0 rule-flip, per cell, for C5. Read from its own
# summary files rather than retyped, because a retyped constant is a defect
# waiting for the next deck to inherit it.
DISSOC_RUN = "/scratch/users/hshunt/spin_gnn/runs/2026-11-13_dissoc"


def cell_dir(arm, seed):
    return os.path.join(RUN, "%s_s%d" % (arm, seed))


def round0_rule_flip(arm, seed):
    """Finding 69's rule-flip at this checkpoint, over ITS rule-valid cases."""
    rows = rows_of(os.path.join(DISSOC_RUN, "%s_s%d" % (arm, seed),
                                "dissoc_behavior.csv"))
    return ratio(rows, "rule_valid", "rule_flips")


def main():
    print("=" * 78)
    print("CHAINING: ONE EDIT TO A FACT THE MODEL DERIVED")
    print("=" * 78)

    bad, _ = guard_configs()
    if bad:
        raise SystemExit("guard 1 FAILED: " + "; ".join(bad))
    print("guard 1 OK: six checkpoints verified, arms agree on every instance key")

    cells, keys = {}, None
    for arm in ARMS:
        for s in SEEDS:
            rows = rows_of(os.path.join(cell_dir(arm, s), "chain4_behavior.csv"))
            k = [tuple(r[f] for f in CHAIN_KEY) for r in rows]
            if keys is None:
                keys = k
            elif k != keys:
                raise SystemExit("guard 2 FAILED: %s s%d case list differs" % (arm, s))
            cells[(arm, s)] = rows
    n = len(keys)
    print("guard 2 OK: six case lists identical row for row, n = %d" % n)

    for (arm, s), rows in cells.items():
        for r in rows:
            if int(r["round"]) < 1:
                raise SystemExit("guard 3 FAILED: %s s%d has a round-0 case" % (arm, s))
            if int(r["nbr_round"]) >= int(r["round"]):
                raise SystemExit("guard 3 FAILED: %s s%d binding neighbour is not earlier"
                                 % (arm, s))
    rmin = min(int(r["round"]) for r in cells[(ARMS[0], SEEDS[0])])
    rmax = max(int(r["round"]) for r in cells[(ARMS[0], SEEDS[0])])
    print("guard 3 OK: every case is round %d-%d with an earlier DEDUCED binding "
          "neighbour" % (rmin, rmax))

    out = {"n_cases": n, "round_min": rmin, "round_max": rmax, "cells": {}}

    # ---- C1, chaining
    print("\nC1  rule edit on a DERIVED fact flips the answer   bar %.2f, chance %.3f"
          % (RULE_BAR, 0.25))
    c1 = {}
    for arm in ARMS:
        line = "    %-3s" % arm
        for s in SEEDS:
            v, _ = fmean(cells[(arm, s)], "rule_flips")
            c1[(arm, s)] = v
            line += "  %.4f" % v
        print(line)
    c1_pass = all(v >= RULE_BAR for v in c1.values())
    print("    C1 %s" % ("CONFIRMED" if c1_pass else "MISSED"))

    print("\n    mean p_freed by variant, and whether v is still named")
    print("    cell     clean   placebo    rule   broken  | emits uncond   b|c")
    for arm in ARMS:
        for s in SEEDS:
            rows = cells[(arm, s)]
            vals = [fmean(rows, "%s_p_freed" % v)[0]
                    for v in ("clean", "placebo", "rule", "broken")]
            unc, _ = fmean(rows, "broken_emits_v")
            cond, _ = ratio(rows, "clean_emits_v", "broken_emits_v")
            print("    %s s%d  %6.4f   %6.4f  %6.4f   %6.4f  |     %6.4f  %6.4f"
                  % (arm, s, vals[0], vals[1], vals[2], vals[3], unc, cond))
            out["cells"]["%s_s%d" % (arm, s)] = {
                "rule_flips": c1[(arm, s)],
                "rule_reverts": fmean(rows, "rule_reverts")[0],
                "emits": unc, "emits_cond": cond,
                "clean_n": len([r for r in rows if truthy(r["clean_emits_v"])]),
                "dissoc": fmean(rows, "dissoc")[0],
            }
            for v in ("clean", "placebo", "rule", "broken"):
                for key in ("p_freed", "p_true", "p_cand", "emits_v"):
                    out["cells"]["%s_s%d" % (arm, s)]["%s_%s" % (v, key)] = \
                        fmean(rows, "%s_%s" % (v, key))[0]

    # ---- C4, specificity. Read before C2, because C2 has no null without it.
    print("\nC4  specificity: |placebo - clean| <= %.2f" % FREED_BAR)
    c4_pass = True
    for arm in ARMS:
        for s in SEEDS:
            rows = cells[(arm, s)]
            d = (sum(_num(r["placebo_p_freed"]) - _num(r["clean_p_freed"])
                     for r in rows) / len(rows))
            ok = abs(d) <= FREED_BAR
            c4_pass = c4_pass and ok
            out["cells"]["%s_s%d" % (arm, s)]["placebo_minus_clean"] = d
            print("    %s s%d  %+.4f  %s" % (arm, s, d, "ok" if ok else "FAILS"))
    print("    C4 %s" % ("CONFIRMED" if c4_pass else "FAILED"))

    # ---- C2, the dissociation, tied arm
    print("\nC2  p_freed(broken) > p_freed(placebo), paired sign test, S1 only")
    print("    bar: Holm p < %.2f AND median diff >= %.2f, in >= 2 of 3 seeds" %
          (ALPHA, FREED_BAR))
    stats = {arm: [sign_test(cells[(arm, s)], "broken_p_freed", "placebo_p_freed")
                   for s in SEEDS] for arm in ARMS}
    adj = holm([st[3] for st in stats["S1"]])
    c2_hits = 0
    for i, s in enumerate(SEEDS):
        b, c_, nd, p, med = stats["S1"][i]
        ok = adj[i] < ALPHA and med >= FREED_BAR
        c2_hits += ok
        out["cells"]["S1_s%d" % s].update(
            {"sign_b": b, "sign_c": c_, "median": med, "p_raw": p, "p_holm": adj[i]})
        print("    s%d  b=%-4d c=%-4d n=%-4d median %+.4f  raw p %.3g  Holm %.3g  %s"
              % (s, b, c_, nd, med, p, adj[i], "PASS" if ok else "miss"))
    c2_pass = c2_hits >= 2
    print("    C2 %s (%d of 3 seeds)" % ("CONFIRMED" if c2_pass else "MISSED", c2_hits))

    print("\n    J2, same test, OFF-POLICY where J2 declines to name v, for contrast")
    for i, s in enumerate(SEEDS):
        b, c_, nd, p, med = stats["J2"][i]
        out["cells"]["J2_s%d" % s].update(
            {"sign_b": b, "sign_c": c_, "median": med, "p_raw": p})
        print("    s%d  b=%-4d c=%-4d median %+.4f  raw p %.3g" % (s, b, c_, med, p))

    # ---- C3, the detection fork
    print("\nC3  rule_reverts: did it keep its own answer instead?   bar %.2f" % REVERT_BAR)
    rev = {}
    for arm in ARMS:
        line = "    %-3s" % arm
        for s in SEEDS:
            v = out["cells"]["%s_s%d" % (arm, s)]["rule_reverts"]
            rev[(arm, s)] = v
            line += "  %.4f" % v
        print(line)
    detects = all(v > REVERT_BAR for v in rev.values())

    # ---- C5, the provenance gap against finding 69
    print("\nC5  chained vs round-0 rule flip, SAME checkpoint, DIFFERENT case list")
    print("    a gap is called at > %.2f lower in >= 2 of 3 seeds of an arm" % GAP_BAR)
    gap_arms = []
    for arm in ARMS:
        hits = 0
        for s in SEEDS:
            r0, r0n = round0_rule_flip(arm, s)
            gap = r0 - c1[(arm, s)]
            hits += gap > GAP_BAR
            out["cells"]["%s_s%d" % (arm, s)].update({"round0_rule_flips": r0,
                                                      "round0_n": r0n, "gap": gap})
            print("    %s s%d  chained %.4f (n=%d)   round-0 %.4f (n=%d)   gap %+.4f"
                  % (arm, s, c1[(arm, s)], n, r0, r0n, gap))
        if hits >= 2:
            gap_arms.append(arm)
    print("    C5: %s" % ("provenance gap in " + ", ".join(gap_arms) if gap_arms
                          else "no provenance gap in either arm"))

    out.update({"C1": bool(c1_pass), "C2": bool(c2_pass), "C4": bool(c4_pass),
                "C3_detects": bool(detects), "C5_gap_arms": gap_arms,
                "revert_bar": REVERT_BAR, "gap_bar": GAP_BAR, "rule_bar": RULE_BAR,
                "freed_bar": FREED_BAR})

    # ---- the verdict, in the order the registration fixed
    print("\n" + "=" * 78)
    if not c4_pass:
        print("VERDICT: C4 FAILED. The placebo moved p_freed, so there is no null and")
        print("no dissociation is claimed. The construction is reported as broken.")
        verdict = "no_null"
    elif c1_pass and not gap_arms:
        print("VERDICT: IT CHAINS. The rule runs over a fact the model DERIVED at the")
        print("same rate as over one it was GIVEN, so finding 69 is not round-0 only,")
        print("and 'it integrates rounds' stops being an inference from accuracy.")
        verdict = "chains"
    elif c1_pass:
        print("VERDICT: IT CHAINS, BUT LESS WELL. The rule follows a derived fact above")
        print("the bar, and still measurably below the round-0 rate in %s."
              % ", ".join(gap_arms))
        print("Provenance costs something; it does not decide the outcome.")
        verdict = "chains_weaker"
    elif detects:
        print("VERDICT: CONFOUNDED BY DETECTION. The rule edit was not followed, and the")
        print("model kept its own answer at a rate above the registered bar, which is")
        print("the signature of noticing an inconsistent prefix. No claim either way")
        print("about chaining; finding 69 stays round-0 only, as its scope page says.")
        verdict = "detection"
    else:
        print("VERDICT: PROVENANCE MATTERS. The rule the tied model learned applies to")
        print("GIVEN facts and not to DERIVED ones: the edit is neither followed nor")
        print("reverted. Finding 69's rule half is bounded to round 0.")
        verdict = "provenance"
    print("=" * 78)
    out["verdict"] = verdict

    path = os.path.join(RUN, "chain_summary.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
