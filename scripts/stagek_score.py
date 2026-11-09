"""Score stage K (job 51668331) against what its sbatch header registered.

Registered and committed BEFORE the first cell landed, as for G/H/I/J/F. The
numbers in here are transcribed from
scripts/slurm/sudoku2_n48_minarch_K_2026-11-09.sbatch; nothing is chosen after
seeing a result.

THE QUESTION. Stage J moved the standing minimum to no_mlp L=16 d192,
2,504,890 trainable at rho_50 0.580, and STATUS reads the frontier as "a floor
near 2M parameters, everything above it flat". That floor was read off the
L=12 width ladder alone:

    no_mlp L=12 d128   CENSORED        879,418
    no_mlp L=12 d160   0.773         1,345,018
    no_mlp L=12 d192   0.754         1,908,922
    no_mlp L=12 d224   0.579         2,571,130   FREE
    no_mlp L=12 d256   0.605         3,331,642   FREE
    no_mlp L=16 d192   0.580         2,504,890   FREE  <- standing minimum

The load-bearing comparison is the last row against the third: at a FIXED
width of 192, going L=12 -> L=16 moved rho_50 0.754 -> 0.580 and carried the
rung across the 0.740 line. Every rung below 2M in that table is an L=12 rung.
Stage K asks whether the same depth step carries d160 over too.

    KA   no_mlp L=16 d160   1,759,738 predicted   1.42x below the minimum
    KB   no_mlp L=16 d128   1,145,658 predicted   2.19x below the minimum

Rungs are lettered (KA, KB) and verdicts numbered (K0-K4) so that no verdict
can be misread as the name of a rung.

FOUR GUARDS, the same ones that matter everywhere in this programme:

  1. K1 GATES THE ARCHITECTURE READING OF KB. A censored rung that never left
     finding 44's 1.64 plateau is a TRAINABILITY statement, not a statement
     about architecture -- exactly what H1 was at d128 and L=12. The verdict
     prints that distinction rather than letting "d128 fails" stand alone.
  2. s* = 0.1484 is FROZEN and never recomputed. Any rho_50 move under it is
     "no separable cost", never an improvement.
  3. N=48 ONLY. Stage F's N=96 rungs and the s*_96 = 0.119 yardstick are a
     different problem size and are never pooled with these: the N=48 rho grid
     ends at 0.33, the N=96 one at 0.32. main() asserts no N=96 dir is in the
     scan list before reading anything.
  4. The standing-minimum rule is unchanged from H3 = I3 = J3: 3 of 3 crossing
     AND median <= 0.740. Two of three is never enough (finding 51).
"""
import argparse
import os
from collections import defaultdict

from scripts.minarch_analysis import (SCAN_DIRS, SCAN_E, SCAN_H, SCAN_I,
                                      SCAN_J, cell_key, cell_metrics, load_all)
from scripts.stagec_score import (S_MEAN, S_RHO50, escapes, rho_txt, rung,
                                  summarize, verdict)
from scripts.stagej_score import LOO_BAR, SUFFICIENCY_DIRS

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_K = os.path.join(_SCR, "spin_gnn", "sudoku_n48_minarch_K_2026-11-09")

HUB_RHO50 = 0.592          # full L=6 d256, what s* is anchored to
HUB_TRAINABLE = 4899898
FREE = 0.740               # hub + s*, the standing-minimum rule's ceiling
PLATEAU = 1.64             # finding 44, an N=48 number

KA = (16, 4, "no_mlp", 160, 1024, 0)
KB = (16, 4, "no_mlp", 128, 1024, 0)
# The L=12 width ladder plus the standing minimum, as one-knob neighbours.
H1 = (12, 4, "no_mlp", 128, 1024, 0)
I1 = (12, 4, "no_mlp", 160, 1024, 0)
H2 = (12, 4, "no_mlp", 192, 1024, 0)
J1 = (12, 4, "no_mlp", 224, 1024, 0)
J2 = (16, 4, "no_mlp", 192, 1024, 0)
E2 = (12, 4, "no_mlp")

PREDICTED_TRAINABLE = {"KA": 1759738, "KB": 1145658}

# K0. The one measured depth step, and the transport of it onto d160.
I1_RHO50 = 0.773                          # no_mlp L=12 d160, KA's one-knob neighbour
DEPTH_STEP_AT_D192 = 0.754 - 0.580        # 0.174, H2 -> J2 at a fixed width
PRED_KA_POINT = 0.599                     # 0.773 - 0.174, I1 carried over
PRED_KA_LO, PRED_KA_HI = 0.55, 0.70
PRED_KA_FREE = True                       # the robust binary call
# K1. KB is registered to censor, for the reason H1 censored.
PRED_KB_CENSORS = True
PRED_KB_ESCAPES_IF_READ = 2               # what it would take to be architecture

STANDING = ("J2", "attention-only L=16 d192", 2504890, 0.580)

# Frozen off disk 2026-11-09, before stage K had a cell, and re-read by main():
#   (n, n_cross, rho_50 text, mean_solve, spread, trainable, escapes)
PARTNER = {
    "H1": (3, 1, "CENSORED", 0.0411, 0.2729, 879418, 1),
    "I1": (3, 3, "0.773", 0.3000, 0.0164, 1345018, 3),
    "H2": (3, 3, "0.754", 0.3220, 0.1450, 1908922, 3),
    "J1": (3, 3, "0.579", 0.5521, 0.2970, 2571130, 3),
    "J2": (3, 3, "0.580", 0.5527, 0.2586, 2504890, 3),
    "E2": (3, 3, "0.605", 0.5214, 0.0480, 3331642, 3),
}
PARTNER_KEY = {"H1": H1, "I1": I1, "H2": H2, "J1": J1, "J2": J2, "E2": E2}


def report(tag, cells, key, asks, predicted_trainable=None):
    ms = rung(cells, key)
    print(f"\n=== {tag} -- {asks} ===")
    if not ms:
        print("  nothing on disk yet.")
        return None
    s = summarize(ms)
    n_esc, nll = escapes(ms)
    print(f"  {s['n']}/3 seeds on disk, {s['n_cross']}/{s['n']} crossed 0.5")
    print(f"  rho_50 median {rho_txt(s):>9}   (standing minimum {STANDING[3]:.3f}, "
          f"I1 {PARTNER['I1'][2]}, H2 {PARTNER['H2'][2]}, free at or below {FREE})")
    print(f"  mean_solve    {s['mean_solve']:>9.4f}   [spread {s['spread_mean']:.4f}]")
    if predicted_trainable is not None:
        print(f"  trainable {s['trainable']:,}   (header arithmetic "
              f"{predicted_trainable:,}"
              + ("" if s["trainable"] == predicted_trainable
                 else "  <-- DISAGREES, quote the disk") + ")")
    else:
        print(f"  trainable {s['trainable']:,}")
    print(f"  plateau escapes {n_esc}/{s['n']}   final train NLL "
          + "/".join(f"{v:.4f}" for v in nll))
    return dict(s=s, n_esc=n_esc, nll=nll, ms=ms)


def score_k0(ka):
    """The headline: does the d192 depth gain transport to d160?"""
    print("\nK0 -- THE HEADLINE: does the depth gain measured at d192 TRANSPORT")
    print(f"  to d160? At a fixed width of 192, L=12 -> L=16 moved rho_50")
    print(f"  {PARTNER['H2'][2]} -> {PARTNER['J2'][2]}, a gain of {DEPTH_STEP_AT_D192:.3f}. Carried onto I1's")
    print(f"  {PARTNER['I1'][2]} that predicts {PRED_KA_POINT:.3f}.")
    print(f"  registered: KA median in [{PRED_KA_LO:.2f}, {PRED_KA_HI:.2f}], and the robust")
    print(f"  binary call -- the one the standing-minimum rule uses -- KA IS FREE")
    print(f"  (median <= {FREE:.3f}) on 3 of 3 crossing.")
    if ka is None:
        print("  nothing on disk; no verdict.")
        return None
    s, n_esc = ka["s"], ka["n_esc"]
    print(f"  observed: {rho_txt(s)}, {s['n_cross']} of {s['n']} crossing, "
          f"{n_esc} of {s['n']} escapes")
    if not s["rho50_readable"]:
        verdict(f"""
K0 FALSIFIED HIGH, BY CENSORING. KA does not produce a median at all, so the
depth gain measured at d192 does NOT transport down to d160: (L, d) is not
separable on this grid and the ~2M floor stands as written. Whether that is an
ARCHITECTURE statement depends on K1's gate -- {n_esc} of {s['n']} seeds got below the
{PLATEAU} plateau. {'They trained and still cannot solve, so this is about architecture.' if n_esc >= 2 else 'They did not train, so this is a TRAINABILITY statement and says nothing about d160 as an architecture.'}
mean_solve, which never censors: {s['mean_solve']:.4f} against I1's {PARTNER['I1'][3]:.4f} at L=12.""")
        return "censored"
    free = s["n"] == 3 and s["n_cross"] == 3 and s["rho50"] <= FREE
    inside = PRED_KA_LO <= s["rho50"] <= PRED_KA_HI
    drho = s["rho50"] - I1_RHO50             # the depth step AT d160
    print(f"  vs I1 (same width, L=12): {drho:+.3f}  "
          f"[the d192 step was {-DEPTH_STEP_AT_D192:+.3f}]")
    if not free:
        why = (f"only {s['n_cross']} of {s['n']} seeds cross" if s["n_cross"] != 3
               else f"median {s['rho50']:.3f} is {(s['rho50'] - FREE) / S_RHO50:+.2f} s* past {FREE:.3f}")
        verdict(f"""
K0 FALSIFIED HIGH. KA measured {s['rho50']:.3f} but is NOT free: {why}. The depth step
that carried d192 over the line does not carry d160 over it, so the gain is not
a constant that can be added to any width -- (L, d) is NOT separable on this
grid. The ~2M floor in STATUS stands, and the standing minimum does not move.
Read this against I1 above: depth moved d160 by {drho:+.3f} where it moved d192 by
{-DEPTH_STEP_AT_D192:+.3f}, so the step SHRINKS as width falls.""")
        return "high"
    if s["rho50"] < PRED_KA_LO:
        verdict(f"""
K0 FALSIFIED LOW, AND IT EARNS NO VICTORY LAP. KA is free at {s['rho50']:.3f}, below the
registered floor of {PRED_KA_LO:.2f}. The rung qualifies and K2 will move the minimum, but
the TRANSPORT MODEL that predicted it is wrong in our favour: depth bought
{abs(drho):.3f} at d160 against {DEPTH_STEP_AT_D192:.3f} at d192, so the gain GROWS as width falls and
(L, d) is not separable. Stage J's convexity reading was fit on the assumption
these axes add; that reading must be rewritten before any of it is quoted
again, and the next width point should be chosen on the new shape, not the old.""")
        return "low"
    if inside:
        verdict(f"""
K0 HELD, AND THE FLOOR IN STATUS WAS AN ARTIFACT OF THE L=12 LADDER. KA is free at
{s['rho50']:.3f}, inside the registered [{PRED_KA_LO:.2f}, {PRED_KA_HI:.2f}], within {abs(s['rho50'] - PRED_KA_POINT):.3f} of the {PRED_KA_POINT:.3f} that
transporting the d192 depth step onto d160 predicted. Depth and width therefore
trade off ADDITIVELY over this stretch, to the accuracy this grid can resolve,
and "there is a floor near 2M parameters" was a statement about L=12 and not
about the task. The frontier extends below 2M.""")
        return "held"
    verdict(f"""
K0 HELD ON THE BINARY CALL, MISSED ON THE INTERVAL. KA is free at {s['rho50']:.3f} -- the
standing-minimum rule is satisfied and the floor claim is falsified -- but the
median is outside the registered [{PRED_KA_LO:.2f}, {PRED_KA_HI:.2f}]. The direction is right and the
level is not, so quote the binary result and NOT the transport arithmetic: the
{DEPTH_STEP_AT_D192:.3f} step measured at d192 became {abs(drho):.3f} at d160.""")
    return "held_binary"


def score_k1(kb):
    """The breaking point, which is the result if it fires."""
    print("\nK1 -- THE BREAKING POINT: d128 censored at L=12 with a train_nll")
    print(f"  that never left the {PLATEAU} plateau ({PARTNER['H1'][6]} of 3 escapes), a")
    print("  TRAINABILITY failure rather than a capacity one.")
    print(f"  registered: KB CENSORS AGAIN (<= 2 of 3 crossing), because depth")
    print("  does not pay for a per-layer width bottleneck.")
    if kb is None:
        print("  nothing on disk; no verdict.")
        return None
    s, n_esc = kb["s"], kb["n_esc"]
    print(f"  observed: {rho_txt(s)}, {s['n_cross']} of {s['n']} crossing, "
          f"{n_esc} of {s['n']} escapes")
    if not s["rho50_readable"] or s["n_cross"] < 3:
        arch = n_esc >= PRED_KB_ESCAPES_IF_READ
        verdict(f"""
K1 HELD: THE BREAKING POINT IS AT d128 AND DEPTH DOES NOT MOVE IT. KB crosses on
{s['n_cross']} of {s['n']} seeds, as H1 did at L=12 on {PARTNER['H1'][1]} of 3. Four extra layers did not buy
the width back, so the wall at d128 is a property of the WIDTH, not of the
depth budget -- and by the brief's own instruction the breaking point is the
result.
THE GATE: {n_esc} of {s['n']} seeds got below the {PLATEAU} plateau. {'That is enough, so these cells LEARNED and still cannot solve: the wall is an ARCHITECTURE statement.' if arch else 'That is NOT enough (needed ' + str(PRED_KB_ESCAPES_IF_READ) + '), so this is a TRAINABILITY statement -- d128 says nothing about architecture here, exactly as H1 did not at L=12. Do not quote it as a capacity limit.'}
mean_solve, which never censors: {s['mean_solve']:.4f} against H1's {PARTNER['H1'][3]:.4f}.""")
        return "censored" if arch else "untrained"
    verdict(f"""
K1 FALSIFIED, AND THIS IS THE LARGEST RESULT IN THE PROGRAMME SINCE STAGE E. KB
crosses on 3 of 3 at {s['rho50']:.3f}, where the same width at L=12 censored. The d128
wall was an artifact of the DEPTH BUDGET, not of width, and the floor moves to
~{s['trainable']:,}. Hold it to the same rules as everything else before believing it:
{s['n_cross']} of {s['n']} crossing and {n_esc} of {s['n']} escapes here, and the registration asked for this
to be RE-RUN on fresh seeds before it is quoted anywhere outside this file.""")
    return "crossed"


def score_k2(ka, ka_call, kb, kb_call):
    """The standing-minimum rule, unchanged from H3 = I3 = J3."""
    print("\nK2 -- THE STANDING-MINIMUM RULE, unchanged from H3, I3 and J3:")
    print(f"  a rung replaces the standing minimum ONLY IF 3 of 3 seeds cross")
    print(f"  AND the median rho_50 is at or below {FREE:.3f}. Two of three is not")
    print(f"  enough (finding 51).")
    tag0, desc0, tr0, rho0 = STANDING
    print(f"  standing: {tag0}, {desc0}, {tr0:,} trainable at {rho0:.3f}")
    fired = []
    for tag, r, call, desc in (("KA", ka, ka_call, "attention-only L=16 d160"),
                               ("KB", kb, kb_call, "attention-only L=16 d128")):
        if r is None:
            print(f"  {tag}: nothing on disk.")
            continue
        s = r["s"]
        if call == "untrained":
            print(f"  {tag} DOES NOT FIRE: the rung did not train, so its rho_50 "
                  f"is not a measurement of its architecture.")
        elif not s["rho50_readable"]:
            print(f"  {tag} DOES NOT FIRE: CENSORED.")
        elif not (s["n"] == 3 and s["n_cross"] == 3):
            print(f"  {tag} DOES NOT FIRE: only {s['n_cross']} of {s['n']} seeds "
                  f"cross. Both conditions or neither.")
        elif s["rho50"] > FREE:
            print(f"  {tag} DOES NOT FIRE: median {s['rho50']:.3f} did not clear "
                  f"{FREE:.3f} ({(s['rho50'] - FREE) / S_RHO50:+.2f} s* past it).")
        else:
            fired.append((tag, desc, s))
            print(f"  {tag} QUALIFIES: 3 of 3 cross at median {s['rho50']:.3f} "
                  f"<= {FREE:.3f}.")
    if not fired:
        print(f"  K2 DOES NOT FIRE. The standing minimum stays {tag0}, {tr0:,} "
              f"trainable at {rho0:.3f}.")
        return None
    tag, desc, s = min(fired, key=lambda f: f[2]["trainable"])
    if len(fired) == 2:
        print(f"  BOTH qualify. The registration fixed the tie-break IN ADVANCE")
        print(f"  (parameter count, as in J3), so the winner is {tag}, the smaller.")
    print(f"  K2 FIRES. The standing minimum moves from {tag0} ({tr0:,} at "
          f"{rho0:.3f}) to {desc}, {s['trainable']:,} at {s['rho50']:.3f}:")
    print(f"    size DOWN {tr0 / s['trainable']:.2f}x, and "
          f"{HUB_TRAINABLE / s['trainable']:.2f}x below the hub "
          f"({HUB_TRAINABLE:,} at {HUB_RHO50:.3f}).")
    drho = s["rho50"] - rho0
    print(f"    rho_50 moves {drho:+.3f} = {abs(drho) / S_RHO50:.2f} s*, which is "
          f"{'BELOW' if abs(drho) <= S_RHO50 else 'ABOVE'} the separability bar "
          f"s* = {S_RHO50}.")
    if abs(drho) <= S_RHO50:
        print(f"    THE CLAIM IS THEREFORE \"{tr0 / s['trainable']:.2f}x smaller at no "
              f"separable cost in rho_50\", never \"better\".")
    else:
        print(f"    THE COST IS SEPARABLE, so the claim is "
              f"\"{tr0 / s['trainable']:.2f}x smaller and {abs(drho):.3f} WORSE in "
              f"rho_50\" -- both halves, always together.")
    return (tag, desc, s, drho)


def score_k3(moved):
    """Separability, stated once, in the form the write-up must use."""
    print("\nK3 -- SEPARABILITY, so that 'smaller' is never read as 'better'.")
    print(f"  s* = {S_RHO50} (rho_50) / {S_MEAN} (mean_solve), FROZEN, not recomputed.")
    tag0, _desc0, tr0, rho0 = STANDING
    if moved is None:
        print(f"  Nothing moved, so there is nothing to separate. The minimum")
        print(f"  stays {tag0} at {rho0:.3f}.")
        return
    tag, _desc, s, drho = moved
    print(f"  BEFORE {tag0}: rho_50 {rho0:.3f} at {tr0:,} trainable")
    print(f"  AFTER  {tag}: rho_50 {s['rho50']:.3f} at {s['trainable']:,} trainable")
    print(f"  delta  {drho:+.3f} rho_50 = {abs(drho) / S_RHO50:.2f} s*, "
          f"{tr0 / s['trainable']:.2f}x smaller")
    print("  Same pooled eval on both sides: 14 rho bins x 400 instances,")
    print("  --shuffle-adj --fall, N=48 d5 q4, 240000 x 50ep. Not pooled with")
    print("  anything at N=96.")


def score_k4():
    print("\nK4 -- OUT-OF-SAMPLE SUFFICIENCY, free, and reported ahead of")
    print(f"  everything if it fails: BOTH new rungs land on the depth ladder's")
    print(f"  accuracy-to-solve curve within the LOO bar {LOO_BAR}. Today the fit")
    print(f"  is 22/22 well-determined held-out rungs over 47 cells.")
    print("  A rung falling OFF the curve is where a width-specific failure mode")
    print("  would show up, and it would be the headline over K0.")
    print("  Run (the nine sufficiency dirs, plus stage K):")
    print("    python3 -m scripts.acc_vs_solve --scan-dir \\\n      " +
          " \\\n      ".join(SUFFICIENCY_DIRS + [SCAN_K]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+",
                    default=SCAN_DIRS + [SCAN_E, SCAN_H, SCAN_I, SCAN_J, SCAN_K],
                    help="SCAN_E/H/I/J/K are out of SCAN_DIRS on purpose; they "
                         "are added here only. SCAN_G is a different epoch "
                         "budget and is used for the sufficiency refit only.")
    args = ap.parse_args()

    # Guard 3: N=48 only. A single N=96 dir here would pool two rho grids.
    for d in args.scan_dir:
        assert "n96" not in os.path.basename(d).lower(), (
            "stage K is N=48 only; the N=96 grid ends at 0.32 and is never "
            "pooled with this one", d)

    by = defaultdict(list)
    for r in load_all(args.scan_dir):
        # n_nodes comes off sweep.csv as a STRING. Comparing it to an int
        # silently drops every record and reports "nothing on disk", which is
        # exactly what this guard is supposed to prevent -- so coerce it.
        if int(r.get("n_nodes", 48)) != 48:
            continue
        by[cell_key(r)].append(r)
    cells = {k: dict(cell_metrics(v, args.boot), path=v[0]["path"], seed=k[-1])
             for k, v in by.items()}

    print("STAGE K: does depth buy back width BELOW the standing minimum?")
    print(f"  {len(cells)} cells read; TWO rungs, the L=12 width ladder one depth up")
    print(f"  frozen bar s* = {S_RHO50} rho_50 / {S_MEAN} mean_solve, NOT recomputed")
    print(f"  registered: KA in [{PRED_KA_LO:.2f}, {PRED_KA_HI:.2f}] and FREE; "
          f"KB CENSORS")

    print("\nFROZEN-LITERAL CONTROL (the L=12 ladder and J2, re-read from disk):")
    for tag in ("H1", "I1", "H2", "J1", "J2", "E2"):
        ms = rung(cells, PARTNER_KEY[tag])
        if not ms:
            print(f"  {tag}: NOT ON DISK -- cannot verify the literals")
            continue
        s, (ne, _nll) = summarize(ms), escapes(ms)
        got = (s["n"], s["n_cross"], rho_txt(s), round(s["mean_solve"], 4),
               round(s["spread_mean"], 4), s["trainable"], ne)
        want = PARTNER[tag]
        print(f"  {tag} {'matches' if got == want else 'DISAGREES WITH'} "
              f"the frozen literals")
        if got != want:
            print(f"    frozen {want}")
            print(f"    disk   {got}")
            print("    the registered grid changed; fix that before reading "
                  "anything below.")

    ka = report("KA", cells, KA, "no_mlp L=16 d160, the rung that tests the floor",
                PREDICTED_TRAINABLE["KA"])
    kb = report("KB", cells, KB, "no_mlp L=16 d128, the rung that should break",
                PREDICTED_TRAINABLE["KB"])

    print("\n" + "=" * 72)
    ka_call = score_k0(ka)
    kb_call = score_k1(kb)
    moved = score_k2(ka, ka_call, kb, kb_call)
    score_k3(moved)
    print("=" * 72)
    score_k4()


if __name__ == "__main__":
    main()
