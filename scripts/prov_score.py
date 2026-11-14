#!/usr/bin/env python3
"""PRE-REGISTERED SCORER for the provenance-paired guard test (finding 71).

This docstring IS the registration. It is committed before the measurement job
is submitted, and the verdict block below is the only thing allowed to decide
what the deck says.

THE QUESTION, AND WHY IT NEEDS REGISTERING RATHER THAN REPORTING. Finding 68
measured the determinacy GUARD -- does the model stop naming v once v stops
being determined -- and got S1 1.0000 at all three seeds (never withholds)
against J2 0.0000 / 0.0000 / 0.8444. Finding 70's read-back then noticed, POST
HOC, that J2's guard rate on the chained case list is 0.9903 / 0.8619 / 0.9830.
Nothing asked that question, so it is a lead and not a result. It is also
confounded three ways at once: the chained cases differ from the round-0 ones in
the PROVENANCE of the edited fact (derived vs given), in the DEPTH of v (rounds
1-15 vs round 0), and in CLASH -- the clash theorem forces an edit to a derived
fact to conflict with a visible neighbour, while finding 68 required its hint
edit to keep the revealed colouring proper.

THE DESIGN HOLDS TWO OF THE THREE FIXED. Each case is one node v with two
single-holder covered colours: one held by a HINT (w_g), one by a fact the model
DEDUCED earlier (w_d). Breaking either leaves v with exactly two candidates, so
depth, prefix, readout and the kind of damage done to v are identical, and the
pair differs only in where the broken fact came from. The given edit's new
colour is then chosen to clash with the same number of visible neighbours as the
derived edit, which is the best that can be done about the third confound; G6
repeats the headline on the exactly-matched subset.

WHAT IS REGISTERED, BEFORE ANY NUMBER EXISTS.

  G1  THE HEADLINE, UNTIED ARM (J2) ONLY, because it is the only arm with a
      guard to lose. Among cases where the clean prefix names v, the model keeps
      naming v after the DERIVED edit more often than after the GIVEN edit:
      exact McNemar on the discordant pairs, Holm-adjusted across the three
      seeds, p < ALPHA = 0.05, AND the rate difference at least
      PROV_BAR = 0.10, in at least 2 of 3 seeds. DIRECTION IS FIXED HERE:
      derived > given. The reverse, at the same bars, is reported as REVERSED
      and is a different finding, not this one.
      FALSIFIER: fewer than 2 of 3 seeds. Then the post-hoc gap is NOT about
      the provenance of the edited fact, finding 68's reading stands as written,
      and G5 says where to look instead.

  G2  THE READ CONTROL, and a GATE on G1. If the model keeps naming v after the
      derived edit, the dull explanation is that the derived token never reached
      it. So the rule edit at the SAME token must work: recolouring w to v's
      forced colour c flips the emitted colour at v's colour position to the
      freed colour, in at least RULE_BAR = 0.60 of cases, for BOTH sides, in all
      six cells. Any cell that misses has its guard numbers printed and marked
      uninterpretable, and G1 is not called.

  G3  SPECIFICITY, and a GATE on G1. The placebo -- a node far from v, recoloured
      and clash-matched to the derived edit -- must leave the node readout alone:
      the model still names v in at least PLACEBO_FLOOR = 0.85 of cases where the
      clean prefix did. Without it, "still names v" has no upper baseline and a
      low rate means nothing.

  G4  THE TIED ARM (S1) AS A CONTROL, descriptive, not a gate. S1 has no guard at
      round 0 (1.0000, three seeds), so it has no provenance effect to show:
      |guard_derived - guard_given| <= PROV_BAR expected at every S1 seed. A
      violation does not falsify G1, but it means the effect is something edits
      do in general rather than something the guard does, and the deck says so.

  G5  DEPTH, descriptive, not a test, and the alternative account G1's falsifier
      hands off to. The GIVEN-side guard rate as a function of v's round, per
      cell. If G1 is null and this rises with depth, the post-hoc gap is about
      where v sits, not about which fact is broken -- a different follow-up, and
      one this run's CSV can already size.

  G6  THE CLASH CONTROL. G1 recomputed on the subset where the two edits clash
      with exactly the same number of visible neighbours. If that subset has
      fewer than MATCH_MIN = 100 cases in a cell, G1 stands on the full sample
      and the deck states that the clash confound is bounded, not removed.

AMENDMENT, 2026-11-13, FROM THE CENSUS ALONE AND BEFORE ANY MODEL OUTPUT WAS
READ. Job 52267370's census (`--cases-only`, no forward pass, so this is a
property of the instance generator and nothing else) returned 6828 paired cases
from 1200 instances, and every one of the 400 scored sits at round >= 1:
r1=112, r2=76, r3=54, r4=37, r5=29, r6=23, r7=17, r8=19, r9=12, r10=7, r11=6,
r12=2, r13=2, r14=3, r15=1. Round 0 is empty for a structural reason, not a thin
tail. A round-0 v is forced by HINTS alone, so its hint neighbours already cover
q-1 colours; a neighbour the model deduced earlier in the same round must
therefore repeat one of those colours, which gives that colour two holders and
disqualifies it as a single holder. A round-0 case with both a hint single
holder and a deduced single holder cannot exist -- the same shape of argument as
the clash theorem, and it means this probe can hold depth fixed but cannot reach
the shallowest depth finding 68 measured.

So G5's one-line summary is differenced against the shallowest NON-EMPTY bin
instead of r0. The r0 column stays in the table, because its emptiness is the
fact. Every bar, gate and hypothesis above is unchanged from the 9b59866
registration, and G5 was descriptive rather than a test in that registration.

GUARDS, each of which fails the job rather than printing a number.
  1. Every checkpoint is the cell it is filed under and the arms agree on every
     instance-generating key (guard_configs, shared with findings 68-70).
  2. All six case lists are identical row for row on PROV_KEY.
  3. The given side really is a hint and the derived side really was emitted
     before v -- re-derived from the trace by the instrument into `given_round`
     (-1 for a hint) and `derived_round`, and re-checked here from the CSV.
  4. Every derived edit clashes at least once (the clash theorem, re-derived
     from the CSV), and the placebo is clash-matched in at least MATCH_FLOOR =
     0.90 of cases. Between 0.90 and 0.95 the run is scored with a warning
     carried into the verdict.

NOTHING IS TRAINED. rho_50, the gate, the free line and every stage result are
untouched by this file and by the job that feeds it.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ddguard_score import (ALPHA, ARMS, PLACEBO_FLOOR, RULE_BAR, SEEDS,
                                   binom_exact, guard_configs, holm, ratio,
                                   rows_of, truthy)
from scripts.dissoc_score import fmean

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
RUN = os.path.join(_SCR, "spin_gnn", "runs", "2026-11-13_prov")
PROV_BAR = 0.10
MATCH_MIN = 100
MATCH_FLOOR = 0.90
# The case key for this construction. Both holders are in it because a pair that
# agreed on v but not on which two facts it breaks is not the same case.
PROV_KEY = ("node", "round", "round_rank", "w_given", "w_derived", "n_known")


def cell_dir(arm, seed):
    return os.path.join(RUN, "%s_s%d" % (arm, seed))


def mcnemar(rows, hi, lo, cond):
    """b, c, n, p for the paired binary pair (hi, lo) over rows where cond holds.

    b counts cases where `hi` fired and `lo` did not. The denominator is printed
    everywhere it is used, because this project has twice shipped a conditional
    rate labelled as an unconditional one.
    """
    den = [r for r in rows if truthy(r[cond])]
    b = sum(1 for r in den if truthy(r[hi]) and not truthy(r[lo]))
    c = sum(1 for r in den if truthy(r[lo]) and not truthy(r[hi]))
    return b, c, len(den), binom_exact(min(b, c), b + c)


def guard_pair(rows):
    """(given rate, derived rate, n) conditional on the clean prefix naming v --
    finding 68's denominator, via finding 68's own `ratio`."""
    g, n = ratio(rows, "clean_emits_v", "names_v_given")
    d, _ = ratio(rows, "clean_emits_v", "names_v_derived")
    return g, d, n


def main():
    print("=" * 78)
    print("IS THE GUARD SPECIFIC TO GIVEN FACTS?  (provenance-paired, fixed v)")
    print("=" * 78)

    bad, _ = guard_configs()
    if bad:
        raise SystemExit("guard 1 FAILED: " + "; ".join(bad))
    print("guard 1 OK: six checkpoints verified, arms agree on every instance key")

    cells, keys = {}, None
    for arm in ARMS:
        for s in SEEDS:
            rows = rows_of(os.path.join(cell_dir(arm, s), "prov_behavior.csv"))
            k = [tuple(r[f] for f in PROV_KEY) for r in rows]
            if keys is None:
                keys = k
            elif k != keys:
                raise SystemExit("guard 2 FAILED: %s s%d case list differs" % (arm, s))
            cells[(arm, s)] = rows
    n = len(keys)
    print("guard 2 OK: six case lists identical row for row, n = %d" % n)

    base = cells[(ARMS[0], SEEDS[0])]
    for r in base:
        if int(r["given_round"]) != -1:
            raise SystemExit("guard 3 FAILED: the given side was emitted in round %s, "
                             "so it is not a hint" % r["given_round"])
        if not 0 <= int(r["derived_round"]) <= int(r["round"]):
            raise SystemExit("guard 3 FAILED: derived side round %s against v's round %s"
                             % (r["derived_round"], r["round"]))
    rmin = min(int(r["round"]) for r in base)
    rmax = max(int(r["round"]) for r in base)
    print("guard 3 OK: given side is a hint, derived side was emitted earlier; "
          "v spans rounds %d-%d" % (rmin, rmax))

    lo = min(int(r["broken_clashes_derived"]) for r in base)
    if lo < 1:
        raise SystemExit("guard 4 FAILED: a derived edit with 0 clashes exists, so "
                         "the clash theorem's premise is wrong")
    pmatch = sum(1 for r in base if truthy(r["placebo_matched"])) / len(base)
    if pmatch < MATCH_FLOOR:
        raise SystemExit("guard 4 FAILED: placebo clash-matched in only %.4f of cases"
                         % pmatch)
    cmatch = sum(1 for r in base if truthy(r["clash_matched"])) / len(base)
    print("guard 4 OK: every derived edit clashes >= 1 (the theorem); placebo "
          "matched %.4f, the two edits matched %.4f" % (pmatch, cmatch))
    weak_placebo = pmatch < 0.95

    out = {"n_cases": n, "round_min": rmin, "round_max": rmax,
           "placebo_matched": pmatch, "clash_matched": cmatch,
           "prov_bar": PROV_BAR, "rule_bar": RULE_BAR,
           "placebo_floor": PLACEBO_FLOOR, "match_min": MATCH_MIN,
           "cells": {}}

    # ---- G2, the read control. First, because G1 is not called without it.
    print("\nG2  does each edit reach the model? rule flip, both sides   bar %.2f, "
          "chance %.3f" % (RULE_BAR, 0.25))
    print("    cell     given    derived")
    g2_pass = True
    for arm in ARMS:
        for s in SEEDS:
            rows = cells[(arm, s)]
            fg, _ = fmean(rows, "rule_flips_given")
            fd, _ = fmean(rows, "rule_flips_derived")
            ok = fg >= RULE_BAR and fd >= RULE_BAR
            g2_pass = g2_pass and ok
            out["cells"]["%s_s%d" % (arm, s)] = {
                "rule_flips_given": fg, "rule_flips_derived": fd,
                "rule_reverts_given": fmean(rows, "rule_reverts_given")[0],
                "rule_reverts_derived": fmean(rows, "rule_reverts_derived")[0]}
            print("    %s s%d  %6.4f   %6.4f  %s" % (arm, s, fg, fd,
                                                     "ok" if ok else "MISSES"))
    print("    G2 %s" % ("CONFIRMED" if g2_pass else "MISSED -- G1 uninterpretable"))

    # ---- G3, specificity
    print("\nG3  placebo still names v, among cases the clean prefix named   "
          "floor %.2f" % PLACEBO_FLOOR)
    g3_pass = True
    for arm in ARMS:
        for s in SEEDS:
            rows = cells[(arm, s)]
            pl, den = ratio(rows, "clean_emits_v", "placebo_emits_v")
            unc, _ = fmean(rows, "placebo_emits_v")
            ok = pl >= PLACEBO_FLOOR
            g3_pass = g3_pass and ok
            out["cells"]["%s_s%d" % (arm, s)].update(
                {"placebo_cond": pl, "placebo_uncond": unc, "clean_n": den})
            print("    %s s%d  conditional %.4f (n=%d)   unconditional %.4f  %s"
                  % (arm, s, pl, den, unc, "ok" if ok else "FAILS"))
    print("    G3 %s" % ("CONFIRMED" if g3_pass else "FAILED -- no upper baseline"))

    # ---- the guard rates themselves, both denominators, labelled in words
    print("\nTHE TWO GUARD RATES.  'still names v' is finding 68's guard quantity:")
    print("HIGH means the model kept naming v after v stopped being determined.")
    print("    cell     given    derived    difference   | unconditional g / d")
    for arm in ARMS:
        for s in SEEDS:
            rows = cells[(arm, s)]
            g, d, den = guard_pair(rows)
            ug, _ = fmean(rows, "names_v_given")
            ud, _ = fmean(rows, "names_v_derived")
            out["cells"]["%s_s%d" % (arm, s)].update(
                {"guard_given": g, "guard_derived": d, "guard_diff": d - g,
                 "guard_given_uncond": ug, "guard_derived_uncond": ud,
                 "guard_n": den})
            print("    %s s%d  %6.4f   %6.4f    %+7.4f    |  %6.4f / %6.4f"
                  % (arm, s, g, d, d - g, ug, ud))
    print("    the first three columns are conditional on the clean prefix naming v;")
    print("    the last two divide by every case, matched or not.")

    # ---- G1, the headline, J2 only
    print("\nG1  J2: does breaking a DERIVED fact leave v named more often than")
    print("    breaking a GIVEN one?  bar: Holm p < %.2f AND difference >= %.2f,"
          % (ALPHA, PROV_BAR))
    print("    in >= 2 of 3 seeds, direction derived > given")
    stats = [mcnemar(cells[("J2", s)], "names_v_derived", "names_v_given",
                     "clean_emits_v") for s in SEEDS]
    adj = holm([st[3] for st in stats])
    hits = rev_hits = 0
    for i, s in enumerate(SEEDS):
        b, c_, den, p = stats[i]
        diff = out["cells"]["J2_s%d" % s]["guard_diff"]
        ok = adj[i] < ALPHA and diff >= PROV_BAR
        rok = adj[i] < ALPHA and -diff >= PROV_BAR
        hits += ok
        rev_hits += rok
        out["cells"]["J2_s%d" % s].update({"mcnemar_b": b, "mcnemar_c": c_,
                                           "p_raw": p, "p_holm": adj[i]})
        print("    s%d  b=%-4d c=%-4d n=%-4d diff %+.4f  raw p %.3g  Holm %.3g  %s"
              % (s, b, c_, den, diff, p, adj[i],
                 "PASS" if ok else ("REVERSED" if rok else "miss")))
    g1_pass = hits >= 2
    g1_rev = rev_hits >= 2
    print("    G1 %s (%d of 3 seeds forward, %d reversed)"
          % ("CONFIRMED" if g1_pass else ("REVERSED" if g1_rev else "MISSED"),
             hits, rev_hits))

    # ---- G4, the tied arm
    print("\nG4  S1 control: no guard, so no provenance effect expected   "
          "|difference| <= %.2f" % PROV_BAR)
    g4_quiet = True
    for s in SEEDS:
        diff = out["cells"]["S1_s%d" % s]["guard_diff"]
        ok = abs(diff) <= PROV_BAR
        g4_quiet = g4_quiet and ok
        print("    s%d  %+.4f  %s" % (s, diff, "ok" if ok else "MOVES"))
    print("    G4: %s" % ("S1 is flat, as a model without a guard should be"
                          if g4_quiet else
                          "S1 MOVES TOO -- the effect is not specific to the guard"))

    # ---- G5, depth, descriptive
    print("\nG5  given-side guard rate by v's round (descriptive, no bar)")
    bins = [(0, 0), (1, 1), (2, 2), (3, 10 ** 6)]
    labels = ["r0", "r1", "r2", "r3+"]
    print("    cell     " + "".join("%-14s" % ("%s (n)" % t) for t in labels))
    for arm in ARMS:
        for s in SEEDS:
            rows = cells[(arm, s)]
            line, prof = "    %s s%d  " % (arm, s), {}
            for (lo_r, hi_r), lab in zip(bins, labels):
                sub = [r for r in rows if lo_r <= int(r["round"]) <= hi_r]
                if sub:
                    v, den = ratio(sub, "clean_emits_v", "names_v_given")
                    prof[lab] = {"rate": v, "n": den}
                    line += "%-14s" % ("%.4f (%d)" % (v, den))
                else:
                    prof[lab] = {"rate": float("nan"), "n": 0}
                    line += "%-14s" % "-"
            out["cells"]["%s_s%d" % (arm, s)]["depth"] = prof
            print(line)
    # Against the shallowest NON-EMPTY bin: r0 cannot exist (see AMENDMENT), so
    # differencing against it would print nan three times and say nothing.
    shallow = next((t for t in labels
                    if out["cells"]["J2_s%d" % SEEDS[0]]["depth"][t]["n"]), labels[0])
    deep_rise = []
    for s in SEEDS:
        prof = out["cells"]["J2_s%d" % s]["depth"]
        a, b_ = prof[shallow], prof["r3+"]
        deep_rise.append(b_["rate"] - a["rate"] if a["n"] and b_["n"] else float("nan"))
    out["depth_shallow_bin"] = shallow
    print("    J2 r3+ minus %s: " % shallow + "  ".join("%+.4f" % x for x in deep_rise))

    # ---- G6, the clash-matched subset
    print("\nG6  the same J2 comparison on the exactly clash-matched cases only")
    g6 = []
    for s in SEEDS:
        sub = [r for r in cells[("J2", s)] if truthy(r["clash_matched"])]
        if len(sub) < MATCH_MIN:
            print("    s%d  only %d matched cases (< %d) -- not read"
                  % (s, len(sub), MATCH_MIN))
            g6.append(None)
            continue
        g, d, den = guard_pair(sub)
        b, c_, _, p = mcnemar(sub, "names_v_derived", "names_v_given", "clean_emits_v")
        g6.append({"given": g, "derived": d, "diff": d - g, "n": den,
                   "b": b, "c": c_, "p_raw": p, "n_matched": len(sub)})
        print("    s%d  n=%d matched, %d named by clean:  given %.4f  derived %.4f"
              "  diff %+.4f  raw p %.3g" % (s, len(sub), den, g, d, d - g, p))
    out["G6"] = g6
    if any(x is None for x in g6):
        g6_agrees = None          # too few matched cases somewhere; not readable
    else:
        g6_agrees = all((x["diff"] >= PROV_BAR) == g1_pass for x in g6)

    out.update({"G1": bool(g1_pass), "G1_reversed": bool(g1_rev),
                "G2": bool(g2_pass), "G3": bool(g3_pass), "G4_quiet": bool(g4_quiet),
                "G5_deep_rise": deep_rise, "G6_agrees": g6_agrees,
                "weak_placebo": bool(weak_placebo)})

    # ---- the verdict, in the order the registration fixed
    print("\n" + "=" * 78)
    if not g2_pass:
        print("VERDICT: AN EDIT DID NOT REACH THE MODEL. The rule test missed on at")
        print("least one side, so 'still names v' cannot be read as a guard decision")
        print("and G1 is not called. The construction is reported as bounded.")
        verdict = "unread_edit"
    elif not g3_pass:
        print("VERDICT: NO UPPER BASELINE. The clash-matched placebo moved the node")
        print("readout on its own, so naming v is not specific to determinacy here")
        print("and nothing below it is read.")
        verdict = "no_null"
    elif g1_pass:
        print("VERDICT: THE GUARD IS SPECIFIC TO GIVEN FACTS. At the SAME v, the")
        print("untied model withholds when the broken fact was a hint and emits")
        print("anyway when it was one of its own deductions, with both edits shown")
        print("to be read by the rule test. Finding 68's 'J2 has the guard' narrows")
        print("to hint-grounded eliminations, and the post-hoc lead is registered.")
        verdict = "provenance_specific"
    elif g1_rev:
        print("VERDICT: REVERSED, AND REPORTED AS SUCH. The untied model withholds")
        print("MORE when the broken fact was derived. The registered direction was")
        print("the other one; this is a different finding and is not relabelled as")
        print("a confirmation of the post-hoc lead.")
        verdict = "reversed"
    else:
        print("VERDICT: NOT PROVENANCE. At fixed v the two guard rates do not")
        print("separate, so the post-hoc chained-vs-round-0 gap is not explained by")
        print("where the broken fact came from. Finding 68 stands as written, and")
        print("G5's depth profile is where the gap has to come from instead.")
        verdict = "not_provenance"
    if weak_placebo:
        print("CAVEAT: the placebo clash-matched in only %.4f of cases (< 0.95), so"
              % pmatch)
        print("the null is matched on relevance more tightly than on corruption.")
    if g6 and any(x is None for x in g6):
        print("CAVEAT: the clash-matched subset was too small to re-read G1 in at")
        print("least one seed, so the clash confound is bounded, not removed.")
    print("=" * 78)
    out["verdict"] = verdict

    os.makedirs(RUN, exist_ok=True)
    path = os.path.join(RUN, "prov_summary.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
