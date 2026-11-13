"""PRE-REGISTRATION: one edit, two positions. Does the tied model's elimination
see the edit its scheduler ignores?

This docstring is the registration. It is committed BEFORE the measurement job
runs, and the hypotheses, bars and falsifier below are fixed at that commit. The
measurement is spin_gnn/sudoku_dissoc.py over the six checkpoints finding 68 used,
unchanged; NOTHING IS TRAINED.

WHY. Finding 68 reads the rule with one edit (hint w: a -> c) at the colour
position and the guard with a different edit (w: a -> b) at the node position. The
tied model S1 follows the first 350/350 and ignores the second 350/350. That
pattern has two readings and finding 68 cannot separate them:

  (i)  the edit reaches the model, changes what is deducible, and the model's node
       schedule does not consult it           -> a real rule/guard dissociation;
  (ii) the w -> b edit never reaches the model at all -> "insensitive to this
       edit", and the guard claim is an artifact of which edit was used where.

The separating measurement is the fourth cell of the 2x2: the SAME w -> b edit,
read at the COLOUR position. Under the broken hints v has exactly two candidates,
{c, a}, where a is the colour w was freed from. If elimination ran, a's
probability must RISE. If it does not rise, reading (ii) is the correct one.

THE QUANTITY. p_freed = the model's softmax probability on colour a at v's colour
position, with v's node token teacher-forced (it is physically present in all four
sequences, since every edit is to a hint colour earlier in the prefix). Four
one-token variants of one sequence, three of them at the SAME token index:

  clean    nothing changed    DD: p_freed = 0  (a is held by w)
  placebo  far hint, proper   DD: p_freed = 0  (irrelevant to v)
  rule     w: a -> c          DD: p_freed = 1  (a is the unique survivor)
  broken   w: a -> b          DD: 0 < p_freed  (a is live again, shares with c)

HYPOTHESES, BARS AND DIRECTIONS, FIXED AT THIS COMMIT

  H1  RULE REPLICATION. On cases where the rule edit is defined, the emitted
      colour becomes a: rule_flips >= RULE_BAR (0.60, finding 27's own bar) in
      ALL SIX cells, against a chance rate of 1/q = 0.250. This is finding 27's
      statistic on finding 68's case list, which no run has measured: the fire
      test used its own, shorter list. H1 failing means the quad construction is
      wrong and nothing below is read.

  H2  THE SAME-EDIT DISSOCIATION (headline, S1 only). Per S1 seed, over all cases:
      the paired difference p_freed(broken) - p_freed(placebo) has b > c with a
      two-sided exact sign-test p < ALPHA after Holm across the three seeds, AND a
      median difference >= FREED_BAR. CONFIRMED when at least 2 of 3 seeds meet
      both. Direction is pre-registered: broken ABOVE placebo. With the node
      readout already at 1.0000 in every S1 seed, H2 says one edit moves the
      elimination and leaves the schedule untouched.

  H3  THE FALSIFIER. If H2 misses in S1 -- median below FREED_BAR, or the sign
      test not significant -- the registered conclusion is that the
      determinacy-destroying edit does not reach the tied model's colour
      computation either, and finding 68's headline NARROWS from "applies DD's
      rule and does not run DD's scheduler" to "is insensitive to this edit at
      both positions". That narrowing is published as the result. No new arm, no
      re-description of the question, and no third edit is tried on these
      checkpoints afterwards.

  H4  SPECIFICITY. |mean p_freed(placebo) - mean p_freed(clean)| <= FREED_BAR in
      all six cells. A one-token edit with nothing to do with v must not move v's
      colour readout. If it does, H2's comparison has no null and nothing is read.

  H5  EXACT REPRODUCTION OF FINDING 68. broken_emits_v per cell equals finding
      68's corrupt_emits_v per cell: the same argmax, at the same position, on the
      same sequence, from the same checkpoint. Expected EXACTLY equal; the bar is
      agreement on at least 349 of 350 cases per cell, and any disagreeing case is
      printed. A real gap means one of the two runs built a different sequence,
      which would invalidate the pairing rather than be a finding.

SCOPE, REGISTERED IN ADVANCE SO IT CANNOT BE DROPPED LATER

  - J2's colour readout under the broken edit is OFF-POLICY. J2 stops naming v
    (0.0000 / 0.0000 / 0.8444), so "what colour would J2 give v" is a conditional
    it never faces. It is reported as a comparison of machinery, not of behaviour,
    and H2 is registered for S1 only.
  - Round 0 only, as in finding 68.
  - The arms still differ 4.1x in trainable parameters; nothing here separates
    tying from budget.
  - Nothing here trains a model, moves rho_50, or closes the 0.011 accuracy gap.

GUARDS, all fail closed before any hypothesis is read
  1  every checkpoint is the cell it is filed under, and the arms agree on every
     instance-generating key (reused from ddguard_score, not reimplemented);
  2  the six case lists are identical row for row on CASE_KEY;
  3  the case list is byte-identical to finding 68's, row for row on CASE_KEY
     against the ddguard run's own skip_behavior.csv -- this is what makes every
     number here paired with a published one.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ddguard_score import (ALPHA, ARMS, CASE_KEY, CKPT, RULE_BAR, SEEDS,
                                   binom_exact, guard_configs, holm, ratio, rows_of,
                                   truthy)

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
RUN = os.path.join(_SCR, "spin_gnn", "runs", "2026-11-13_dissoc")
DDGUARD = os.path.join(_SCR, "spin_gnn", "runs", "2026-11-13_ddguard", "guard")

# One bar for both H2 and H4, deliberately: the effect H2 needs and the drift H4
# forbids are the same quantity on the same scale, so a single threshold cannot be
# set to make one pass while the other also passes on noise.
FREED_BAR = 0.05
# Finding 68's guard readout, for H5. Imported would be better; the scorer wrote
# these to ddguard_summary.json, which H5 reads directly rather than retyping.
DDGUARD_SUMMARY = os.path.join(_SCR, "spin_gnn", "runs", "2026-11-13_ddguard",
                               "ddguard_summary.json")


def cell_dir(arm, seed):
    return os.path.join(RUN, "%s_s%d" % (arm, seed))


def key_of(row):
    return tuple(str(row[k]) for k in CASE_KEY)


def _num(x):
    """Every CSV cell is a string, and the boolean columns arrive as 'True'.
    float('True') raises, which is what killed job 52224240 after the whole
    measurement had already succeeded."""
    try:
        return float(x)
    except ValueError:
        return float(truthy(x))


def fmean(rows, key, cond=None):
    g = [r for r in rows if cond is None or truthy(r[cond])]
    if not g:
        return float("nan"), 0
    return sum(_num(r[key]) for r in g) / len(g), len(g)


def sign_test(rows, hi, lo):
    """b, c, n and the two-sided exact p for the paired difference hi - lo."""
    d = [float(r[hi]) - float(r[lo]) for r in rows]
    b = sum(1 for x in d if x > 0)
    c = sum(1 for x in d if x < 0)
    med = sorted(d)[len(d) // 2] if d else float("nan")
    return b, c, b + c, binom_exact(min(b, c), b + c), med


def main():
    print("=" * 78)
    print("ONE EDIT, TWO POSITIONS -- scoring the same-edit dissociation")
    print("=" * 78)

    bad, _ = guard_configs()
    if bad:
        print("GUARD 1 FAILED -- checkpoints are not the cells they are filed under:")
        for b in bad:
            print("   " + b)
        raise SystemExit(1)
    print("guard 1 OK: six checkpoints verified, arms agree on every instance key")

    cells = {}
    for arm in ARMS:
        for s in SEEDS:
            path = os.path.join(cell_dir(arm, s), "dissoc_behavior.csv")
            if not os.path.exists(path):
                raise SystemExit("missing cell: %s" % path)
            cells[(arm, s)] = rows_of(path)

    ref = [key_of(r) for r in cells[("S1", 0)]]
    for k, rows in cells.items():
        got = [key_of(r) for r in rows]
        if got != ref:
            raise SystemExit("GUARD 2 FAILED: %s s%d case list differs from S1 s0 "
                             "(%d vs %d rows)" % (k[0], k[1], len(got), len(ref)))
    print("guard 2 OK: six case lists identical row for row, n = %d" % len(ref))

    for arm in ARMS:
        for s in SEEDS:
            p = os.path.join(DDGUARD, "%s_s%d" % (arm, s), "skip_behavior.csv")
            if not os.path.exists(p):
                raise SystemExit("GUARD 3: finding 68's cell is missing: %s" % p)
            old = [key_of(r) for r in rows_of(p)]
            if old != ref:
                raise SystemExit("GUARD 3 FAILED: %s s%d does not match finding 68's "
                                 "case list (%d vs %d rows)" % (arm, s, len(old), len(ref)))
    print("guard 3 OK: the case list IS finding 68's, so every number here is paired")

    out = {"n_cases": len(ref), "rule": {}, "freed": {}, "emits": {}, "sign": {},
           "placebo_drift": {}, "dissoc": {}, "reproduce": {}}

    # ---- H1 -----------------------------------------------------------------
    print("\nH1  rule_flips on rule-valid cases   bar %.2f, chance 0.250" % RULE_BAR)
    h1 = True
    for arm in ARMS:
        row = []
        for s in SEEDS:
            r, n = ratio(cells[(arm, s)], "rule_valid", "rule_flips")
            out["rule"].setdefault(arm, {})[s] = r
            out.setdefault("n_rule_valid", {})["%s_s%d" % (arm, s)] = n
            h1 = h1 and (r >= RULE_BAR)
            row.append("%.4f (n=%d)" % (r, n))
        print("    %-3s %s" % (arm, "  ".join(row)))
    print("    H1 %s" % ("CONFIRMED" if h1 else "FALSIFIED -- construction is wrong"))
    if not h1:
        print("    nothing below is read; the quad does not implement the edits.")
        with open(os.path.join(RUN, "dissoc_summary.json"), "w") as f:
            json.dump({"H1": False}, f, indent=2)
        raise SystemExit(1)

    # ---- the 4x2 table ------------------------------------------------------
    # Both emits columns are printed because they have DIFFERENT DENOMINATORS and
    # mixing them silently is this project's recurring deck bug: "uncond" is over
    # all cases, "b|c" only over the cases the model named v on cleanly.
    print("\n    mean p_freed by variant, and whether v is still named")
    print("    cell     clean   placebo    rule   broken  | emits uncond   b|c")
    for arm in ARMS:
        for s in SEEDS:
            rows = cells[(arm, s)]
            vals = {v: fmean(rows, "%s_p_freed" % v)[0]
                    for v in ("clean", "placebo", "rule", "broken")}
            ev = fmean(rows, "broken_emits_v")[0]
            evc, _ = ratio(rows, "clean_emits_v", "broken_emits_v")
            out["freed"].setdefault(arm, {})[s] = vals
            out["emits"].setdefault(arm, {})[s] = ev
            out.setdefault("emits_cond", {}).setdefault(arm, {})[s] = evc
            print("    %s s%d  %7.4f %8.4f %7.4f %8.4f  | %10.4f %7.4f"
                  % (arm, s, vals["clean"], vals["placebo"], vals["rule"],
                     vals["broken"], ev, evc))

    # ---- H4, read BEFORE H2 because it is H2's null -------------------------
    print("\nH4  specificity: |placebo - clean| <= %.2f" % FREED_BAR)
    h4 = True
    for arm in ARMS:
        for s in SEEDS:
            v = out["freed"][arm][s]
            drift = v["placebo"] - v["clean"]
            out["placebo_drift"].setdefault(arm, {})[s] = drift
            ok = abs(drift) <= FREED_BAR
            h4 = h4 and ok
            print("    %s s%d  %+.4f  %s" % (arm, s, drift, "ok" if ok else "OVER"))
    print("    H4 %s" % ("CONFIRMED" if h4 else "FALSIFIED -- H2 has no null"))

    # ---- H2 -----------------------------------------------------------------
    print("\nH2  p_freed(broken) > p_freed(placebo), paired sign test, S1 only")
    print("    bar: Holm p < %.2f AND median diff >= %.2f, in >= 2 of 3 seeds"
          % (ALPHA, FREED_BAR))
    stats = [sign_test(cells[("S1", s)], "broken_p_freed", "placebo_p_freed")
             for s in SEEDS]
    adj = holm([st[3] for st in stats])
    h2_seeds = 0
    for s, (b, c, n, p, med), pa in zip(SEEDS, stats, adj):
        ok = (pa < ALPHA) and (b > c) and (med >= FREED_BAR)
        h2_seeds += int(ok)
        out["sign"]["S1_s%d" % s] = {"b": b, "c": c, "n": n, "p": p, "holm": pa,
                                     "median": med, "passes": ok}
        print("    s%d  b=%-4d c=%-4d n=%-4d median %+.4f  raw p %.3g  Holm %.3g  %s"
              % (s, b, c, n, med, p, pa, "PASS" if ok else "miss"))
    h2 = h2_seeds >= 2
    print("    H2 %s (%d of 3 seeds)"
          % ("CONFIRMED" if h2 else "FALSIFIED -> H3 is the result", h2_seeds))

    # J2 for contrast only; registered as off-policy, never as H2's comparator.
    print("\n    J2, same test, OFF-POLICY (J2 stops naming v), for contrast only")
    for s in SEEDS:
        b, c, n, p, med = sign_test(cells[("J2", s)], "broken_p_freed", "placebo_p_freed")
        out["sign"]["J2_s%d" % s] = {"b": b, "c": c, "n": n, "p": p, "median": med}
        print("    s%d  b=%-4d c=%-4d median %+.4f  raw p %.3g" % (s, b, c, med, p))

    print("\n    per-case dissociation rate (names v AND frees a)")
    for arm in ARMS:
        vals = []
        for s in SEEDS:
            r, _ = fmean(cells[(arm, s)], "dissoc")
            out["dissoc"].setdefault(arm, {})[s] = r
            vals.append("%.4f" % r)
        print("    %-3s %s" % (arm, "  ".join(vals)))

    # ---- H5 -----------------------------------------------------------------
    print("\nH5  broken_emits_v reproduces finding 68's corrupt_emits_v")
    h5 = True
    with open(DDGUARD_SUMMARY) as f:
        ddg = json.load(f)
    for arm in ARMS:
        for s in SEEDS:
            new = cells[(arm, s)]
            old = rows_of(os.path.join(DDGUARD, "%s_s%d" % (arm, s), "skip_behavior.csv"))
            agree = sum(1 for a, b in zip(new, old)
                        if truthy(a["broken_emits_v"]) == truthy(b["corrupt_emits_v"]))
            published = ddg["guard"][arm][str(s)]
            # CONDITIONAL on the model naming v cleanly -- which is the statistic
            # ddguard_score registered and published. Comparing it against this
            # run's UNCONDITIONAL mean would be the denominator mistake the guard
            # deck's read-back caught on 2026-11-13; the rates would differ by a
            # third and look like a reproduction failure that is not one.
            here, here_n = ratio(new, "clean_emits_v", "broken_emits_v")
            uncond = fmean(new, "broken_emits_v")[0]
            ok = agree >= len(ref) - 1
            h5 = h5 and ok
            out["reproduce"]["%s_s%d" % (arm, s)] = {
                "agree": agree, "n": len(ref), "published": published,
                "here": here, "clean_n": here_n, "unconditional": uncond}
            print("    %s s%d  agree %d/%d   broken|clean %.4f  finding 68 %.4f"
                  "   (uncond %.4f over %d clean)  %s"
                  % (arm, s, agree, len(ref), here, published, uncond, here_n,
                     "ok" if ok else "DISAGREES"))
            if not ok:
                for i, (a, b) in enumerate(zip(new, old)):
                    if truthy(a["broken_emits_v"]) != truthy(b["corrupt_emits_v"]):
                        print("        case %d node %s: here %s, finding 68 %s"
                              % (i, a["node"], a["broken_emits_v"], b["corrupt_emits_v"]))
    print("    H5 %s" % ("CONFIRMED" if h5 else "FALSIFIED -- the pairing is broken"))

    out["H1"], out["H2"], out["H4"], out["H5"] = h1, h2, h4, h5
    out["H3_is_the_result"] = not h2
    out["FREED_BAR"] = FREED_BAR
    with open(os.path.join(RUN, "dissoc_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 78)
    if not (h4 and h5):
        print("VERDICT: a guard hypothesis failed. H2 is NOT read; fix the measurement.")
    elif h2:
        print("VERDICT: SAME-EDIT DISSOCIATION. One token moves the tied model's")
        print("elimination and leaves its node schedule at 1.0000. Finding 68's")
        print("decomposition stands at full strength: rule yes, scheduler no.")
    else:
        print("VERDICT: H3. The determinacy-destroying edit does not reach the tied")
        print("model's colour computation either. Finding 68 narrows to 'insensitive")
        print("to this edit at both positions'; the guard claim is NOT supported by a")
        print("same-edit dissociation and the deck must say so.")
    print("=" * 78)
    print("wrote %s/dissoc_summary.json" % RUN)


if __name__ == "__main__":
    main()
