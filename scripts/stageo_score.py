"""Score stage O: is the free/not-free transition set by SIZE or by SHAPE?

REGISTERED BEFORE THE FIRST CELL LANDS. This docstring IS the registration --
committed by scripts/slurm/stageo_register_2026-11-10.sbatch, which the array
scripts/slurm/sudoku2_n48_sizeshape_O_2026-11-10.sbatch depends on with
--dependency=afterok. No cell can start ahead of it.

THE OBSERVATION THIS STAGE TESTS. Every attention-only (`no_mlp`) N=48 rung the
programme has measured, sorted by trainable count rather than by shape:

    rung                 trainable   rho_50   cross   escapes   call
    H1  L=12 d128          879,418   CENSORED  1/3      1/3      censors
    KB  L=16 d128        1,145,658   0.761*    6/6      5/6      past 0.740
    I1  L=12 d160        1,345,018   0.773     3/3      3/3      past 0.740
    KA  L=16 d160        1,759,738   0.757     3/3      3/3      past 0.740
    H2  L=12 d192        1,908,922   0.754     3/3      3/3      past 0.740
    ------------------- the transition is in here --------------------
    J2  L=16 d192        2,504,890   0.580     3/3      3/3      FREE (standing min)
    J1  L=12 d224        2,571,130   0.579     3/3      3/3      free
    E2  L=12 (no_mlp)    3,331,642   0.605     3/3      3/3      free
                                     * six-seed median, finding 56

Two DIFFERENT shapes reach ~2.5M -- 16 layers of 192 and 12 layers of 224 --
and return the same rho_50 to 0.001. Two different shapes sit at ~1.8-1.9M --
16 of 160 and 12 of 192 -- and return the same rho_50 to 0.003. Inside this
family rho_50 reads like a function of TRAINABLE COUNT and not of how the
parameters are arranged, with a sharp step somewhere between 1.91M and 2.50M.
That has never been tested directly: no two cells of MATCHED size and different
shape exist in the gap, so the pattern could equally be a coincidence of a
ladder that only ever moved one knob at a time.

WHAT STAGE O ADDS: TWO RUNGS OF MATCHED SIZE AND DIFFERENT SHAPE, INSIDE THE GAP

    OA  no_mlp L=16 d176   2,115,930 trainable   the 16-layer ladder's next width
    OB  no_mlp L=20 d160   2,174,458 trainable   the deepest cell in the programme

Matched to 2.8% in trainable count, 4 layers and 16 channels apart in shape, and
both between H2's 1,908,922 (past the line) and J2's 2,504,890 (free). Every
other flag is stage K's and stage N's, copied rather than retyped: nhead 4, ff
1024, warmup 0, 240k x 50ep, `--shuffle-adj --fall`, the 14-point grid ending at
0.33, 400 instances per bin, 3 seeds (0/1/2) per rung.

THE BARS ARE N=48 BARS, REUSED AND NOT REFROZEN: s* = 0.1484 (rho_50), 0.2918
(mean_solve), free line 0.740, plateau 1.64, LOO bar 0.0423. Stage M's N=96 bars
(0.117 / 0.1586 / 0.0079) are a different size and appear nowhere here; finding
55 is the standing reason the two sizes are never pooled or differenced.

O0  THE HEADLINE, AND IT IS A DISCRIMINATING COMPARISON. OA and OB land on the
    SAME side of the free line, and their medians agree within one s*. That is
    what "rho_50 is a function of trainable count" predicts of a 2.8% size match.
    THE COMPETING HYPOTHESIS PREDICTS THE OPPOSITE: the only shape step this
    programme has ever measured as moving anything is depth at fixed width
    (H2 -> J2, -0.174), so "depth is what buys deduction" says OB -- four layers
    deeper than any cell here -- comes in free while OA, a width step below a
    rung that is already past the line, does not. One run separates them.

O1  THE LEVEL, AS A POINT PREDICTION. Both OA and OB are PAST the free line
    0.740. Interpolating in log-trainable between the two bracketing pairs puts
    the step near 2.2M, and both cells sit just under it. REGISTERED UPSIDE,
    stated before the fact: if either is free under the standing replacement
    rule -- 3 of 3 crossing AND median rho_50 <= 0.740 -- it REPLACES the
    standing minimum (OA 1.18x smaller than 2,504,890, OB 1.15x). If both are
    free, the smaller trainable count wins the tie, as the rule says.

O2  THE CROSSING. All 6 cells return a MEASURED rho_50 on a grid whose floor is
    0.33. Both rungs are more than twice the size of the only rung that has ever
    censored (H1, 879,418), so a censor here is this stage's result: it would
    mean the transition is not monotone in size and the ordering in the table
    above is not a ladder at all.

O3  THE PLATEAU. At least 5 of 6 cells escape train NLL 1.64. Every rung at or
    above 1.75M has escaped 3 of 3; only the 1.15M KB has ever had a stuck seed
    (5 of 6, finding 56). Fewer than 5 would say these cells are
    optimization-limited rather than capacity-limited, which reinterprets the
    level and outranks O0 -- a stuck seed is a fact about the recipe, not about
    the architecture.

O4  ACCURACY FIRST, AS EVERY STAGE. Validation token accuracy is read before
    rho_50 because it never censors. All 6 cells sit on scripts/acc_vs_solve.py's
    N=48 fitted curve within the LOO bar 0.0423. A cell falling OFF it is a
    seed-specific or shape-specific failure mode and is the headline over O0.
    That curve is an N=48 object and is applied only because this stage is N=48.

GUARDS, each one a mistake this programme has already made once
  1. N=96 bars are never used here, and no N=48 number is differenced against an
     N=96 one (finding 55).
  2. rho_50 CENSORS. A censored seed is a result, not a missing number, and the
     bound-aware median is what gets quoted.
  3. SCAN_O stays OUT of SCAN_DIRS, as E/G/H/I/J/K/N all do, so a new rung
     cannot enter a pooled yardstick by accident.
  4. The four bracketing rungs are re-read from disk and checked against the
     literals in the table above. A drifted bracket invalidates the whole
     comparison, and silently.
  5. OA and OB are separate rungs and are never pooled with each other: matched
     SIZE is the design, matched CELL is not. Only seeds of the same cell pool.
  6. Every checkpoint is the final epoch, here as everywhere in this programme.
"""

import argparse
import os
import sys
from statistics import median

from scripts.minarch_analysis import SCAN_DIRS, SCAN_H, SCAN_I, SCAN_J
from scripts.stagec_score import (PLATEAU, S_MEAN, S_RHO50, escapes, rho_txt,
                                  rung, summarize)
from scripts.stagej_score import LOO_BAR, SUFFICIENCY_DIRS
from scripts.stagek_score import FREE, SCAN_K, STANDING
from scripts.stagen_score import SCAN_N, acc_of, load_pooled

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_O = os.path.join(_SCR, "spin_gnn", "sudoku_n48_sizeshape_O_2026-11-10")

# The two new rungs. Keys are (num_layers, nhead, arch_mode, d_model, ff, warmup).
OA = (16, 4, "no_mlp", 176, 1024, 0)
OB = (20, 4, "no_mlp", 160, 1024, 0)

# (630*d + 58) + (4*d^2 + 8*d)*L, the N=48 formula verified on KA and KB.
PREDICTED_TRAINABLE = {"OA": 2115930, "OB": 2174458}

SEEDS = (0, 1, 2)

# The bracket, frozen from the table above, re-read from disk by main().
#   rung key -> (label, trainable, rho_50 text, escapes_of_3)
BRACKET = {
    (16, 4, "no_mlp", 160, 1024, 0): ("KA  L=16 d160", 1759738, "0.757", 3),
    (12, 4, "no_mlp", 192, 1024, 0): ("H2  L=12 d192", 1908922, "0.754", 3),
    (16, 4, "no_mlp", 192, 1024, 0): ("J2  L=16 d192", 2504890, "0.580", 3),
    (12, 4, "no_mlp", 224, 1024, 0): ("J1  L=12 d224", 2571130, "0.579", 3),
}
BRACKET_DIRS = [SCAN_H, SCAN_I, SCAN_J, SCAN_K]

GAP_LO, GAP_HI = 1908922, 2504890         # past the line / free, the two ends
STANDING_TRAINABLE = STANDING[2]          # 2,504,890
STANDING_RHO50 = STANDING[3]              # 0.580


def control(cells):
    """Guard 4: the bracket is what this stage's claim is measured against."""
    print("\n=== CONTROL -- the bracketing rungs, re-read from disk ============")
    ok = True
    for key, (label, trainable, txt, esc) in BRACKET.items():
        ms = rung(cells, key)
        if not ms:
            print(f"  {label}: NOT ON DISK in the dirs scanned -- cannot check")
            ok = False
            continue
        s = summarize(ms)
        n_esc, _ = escapes(ms)
        got = rho_txt(s)
        same = (got == txt and s["trainable"] == trainable and n_esc == esc)
        print(f"  {label}  rho_50 {got:>9}  trainable {s['trainable']:>10,}  "
              f"escapes {n_esc}/{s['n']}   "
              + ("matches the frozen literals" if same else "<-- DISAGREES"))
        ok = ok and same
    if not ok:
        print("  STOP: the bracket drifted or is incomplete. Every verdict below")
        print("  is measured against it, so fix the bracket before reading on.")
    return ok


def report(tag, cells, key, asks):
    ms = rung(cells, key)
    print(f"\n=== {tag} -- {asks} ===")
    if not ms:
        print("  nothing on disk yet.")
        return None
    s = summarize(ms)
    n_esc, nll = escapes(ms)
    pred = PREDICTED_TRAINABLE[tag]
    print(f"  {s['n']}/3 seeds on disk, {s['n_cross']}/{s['n']} crossed 0.5")
    print(f"  rho_50 median {rho_txt(s):>9}   (free at or below {FREE}, standing "
          f"minimum {STANDING_RHO50:.3f}, H2 0.754, J2 0.580)")
    print(f"  mean_solve    {s['mean_solve']:>9.4f}   [spread {s['spread_mean']:.4f}]")
    print(f"  trainable {s['trainable']:,}   (registration predicted {pred:,}"
          + ("" if s["trainable"] == pred else "  <-- DISAGREES, quote the disk")
          + ")")
    print(f"  plateau escapes {n_esc}/{s['n']}   final train NLL "
          + "/".join(f"{v:.4f}" for v in nll))
    for m in sorted(ms, key=lambda r: r["seed"]):
        flag = "ok" if m["flag"] == "ok" else m["flag"].upper()
        print(f"    seed {m['seed']}  rho_50 {m['rho50']:.3f} {flag:>9}  "
              f"mean_solve {m['mean_solve']:.4f}")
    if s["trainable"] != pred:
        print("  STOP: this is not the cell the registration describes. The")
        print("  matched-size design is the whole comparison; a different count")
        print("  means O0 is not being measured.")
    if not (GAP_LO < s["trainable"] < GAP_HI):
        print(f"  NOTE: {s['trainable']:,} is NOT inside the measured gap "
              f"({GAP_LO:,}, {GAP_HI:,}); O0 assumed it was.")
    return s


def score_accuracy_first(cells):
    print("\n=== O4a -- VALIDATION TOKEN ACCURACY, READ FIRST =================")
    print("  The standing non-negotiable: accuracy never censors, so it is read")
    print("  before rho_50 and rho_50 only confirms it.")
    any_row = False
    for tag, key in (("OA", OA), ("OB", OB)):
        ms = rung(cells, key)
        if not ms:
            continue
        any_row = True
        rows = sorted(acc_of(ms))
        for seed, a in rows:
            print(f"    {tag}  seed {seed}  val token acc {a:.4f}")
        print(f"    {tag}  median {median([a for _, a in rows]):.4f}")
    if not any_row:
        print("  nothing on disk yet")
        return
    a = rung(cells, OA)
    b = rung(cells, OB)
    if a and b:
        ma = median([v for _, v in acc_of(a)])
        mb = median([v for _, v in acc_of(b)])
        print(f"  OA - OB in accuracy: {ma - mb:+.4f}")
        print("  A difference here with no difference in rho_50, or the reverse,")
        print("  is what O4 is for -- see the curve command at the end.")


def score_o0(sa, sb):
    print("\n=== O0 -- SIZE OR SHAPE? THE DISCRIMINATING COMPARISON ===========")
    print("  registered: at matched trainable count (2.8% apart) OA and OB land")
    print(f"  on the SAME side of {FREE:.3f} and agree within one s* = {S_RHO50}.")
    print("  The competing hypothesis -- depth is what buys deduction -- predicts")
    print("  OB free and OA not.")
    if not (sa and sb):
        print("  PENDING: both rungs are needed; nothing is called on one.")
        return
    if sa["n"] < 3 or sb["n"] < 3:
        print(f"  INCOMPLETE: OA {sa['n']}/3, OB {sb['n']}/3 seeds. Reported, not")
        print("  called -- a shape comparison on unequal seed counts is not one.")
        return
    if not (sa["rho50_readable"] and sb["rho50_readable"]):
        print("  CENSORED on at least one rung -- see O2. A side-of-the-line call")
        print("  against a bound is a bound, not a measurement.")
        return
    fa, fb = sa["rho50"] <= FREE, sb["rho50"] <= FREE
    d = sa["rho50"] - sb["rho50"]
    print(f"  OA {sa['rho50']:.3f} ({'free' if fa else 'past'})   "
          f"OB {sb['rho50']:.3f} ({'free' if fb else 'past'})   "
          f"OA - OB {d:+.3f} = {d / S_RHO50:+.2f} s*")
    if fa == fb and abs(d) <= S_RHO50:
        print("  O0 HELD. Same side of the line and inside one s*: within this")
        print("  family rho_50 behaves as a function of trainable count, and the")
        print("  16-vs-20 layer, 176-vs-160 channel rearrangement of the same")
        print("  budget does not move it. The depth hypothesis is falsified in")
        print("  the one place it made a different prediction.")
    elif fa != fb:
        print("  O0 FALSIFIED: the two rungs straddle the free line at matched")
        print("  size, so SHAPE decides and trainable count does not. Which one")
        print("  is free says which shape:")
        print("    OB free, OA not  -> depth, as the H2 -> J2 step suggested.")
        print("    OA free, OB not  -> width, and depth beyond 16 HURTS, which")
        print("                        nothing in this programme predicted.")
    else:
        print(f"  O0 PARTLY FALSIFIED: same side of the line, but {abs(d):.3f} "
              f"= {abs(d) / S_RHO50:.2f} s* apart.")
        print("  Shape moves the level without moving the verdict. The honest")
        print("  reading is that size sets the side and shape sets the margin.")


def score_o1(sa, sb):
    print("\n=== O1 -- THE LEVEL, AND THE STANDING-MINIMUM CALL ===============")
    print(f"  registered PREDICTION: both rungs PAST {FREE:.3f}; standing minimum")
    print(f"  UNCHANGED at {STANDING_TRAINABLE:,} trainable / rho_50 "
          f"{STANDING_RHO50:.3f}.")
    print("  registered UPSIDE: either rung free under the standing rule (3 of 3")
    print("  crossing AND median <= 0.740) REPLACES it; smaller count wins a tie.")
    winners = []
    for tag, s in (("OA", sa), ("OB", sb)):
        if not s:
            print(f"  {tag}: PENDING")
            continue
        if s["n"] < 3:
            print(f"  {tag}: INCOMPLETE ({s['n']}/3) -- the rule is written for"
                  " three and is not applied to fewer.")
            continue
        if not s["rho50_readable"]:
            print(f"  {tag}: CENSORED -- does not fire, and the reason is a bound.")
            continue
        gap = (s["rho50"] - FREE) / S_RHO50
        if s["n_cross"] == 3 and s["rho50"] <= FREE:
            print(f"  {tag}: FIRES. {s['rho50']:.3f} is {gap:+.2f} s* relative to "
                  f"{FREE:.3f} with 3 of 3 crossing.")
            winners.append((s["trainable"], tag, s))
        else:
            why = ("not all three crossed" if s["n_cross"] < 3
                   else f"median is {gap:+.2f} s* past {FREE:.3f}")
            print(f"  {tag}: does not fire ({why}).")
    if not winners:
        print(f"  O1 HELD: no rung takes the minimum. Before/after on the same")
        print(f"  pooled s* eval: {STANDING_RHO50:.3f} at {STANDING_TRAINABLE:,}"
              f" -> {STANDING_RHO50:.3f} at {STANDING_TRAINABLE:,}, UNCHANGED.")
        print("  The transition stays bracketed by H2 (1,908,922, past) and J2")
        print("  (2,504,890, free), now with two measured points inside it.")
        return
    winners.sort()
    trainable, tag, s = winners[0]
    print(f"  *** O1's PREDICTION FALSIFIED, IN THE REGISTERED UPSIDE ***")
    print(f"  {tag} REPLACES THE STANDING MINIMUM:")
    print(f"    {STANDING_TRAINABLE:,} -> {trainable:,} trainable, "
          f"{STANDING_TRAINABLE / trainable:.2f}x smaller, rho_50 "
          f"{STANDING_RHO50:.3f} -> {s['rho50']:.3f}.")
    if len(winners) > 1:
        print(f"  (both rungs fired; the tie went to the smaller count, as the")
        print("   rule says. The other is a free rung too and is reported.)")
    print("  Before quoting it: this is an N=48 claim and finding 55 says the")
    print("  N=48 minimum does not transfer to N=96. Say N=48 in the sentence.")


def score_o2(sa, sb):
    print("\n=== O2 -- THE CROSSING ===========================================")
    print("  registered: all 6 cells return a measured rho_50 (grid floor 0.33).")
    print("  Both rungs are >2x the only rung that has censored (H1, 879,418).")
    n = nc = 0
    for s in (sa, sb):
        if s:
            n += s["n"]
            nc += s["n_cross"]
    if not n:
        print("  PENDING: nothing on disk yet.")
        return
    print(f"  {nc}/{n} cells crossed 0.5")
    if n < 6:
        print(f"  INCOMPLETE: {n} of 6 cells. Not called.")
    elif nc == 6:
        print("  O2 HELD. Size is monotone through the gap, so the table in the")
        print("  registration is a ladder and O0's comparison is on it.")
    else:
        print(f"  O2 FALSIFIED: {6 - nc} of 6 censored at 2.1M, above rungs that")
        print("  cross at 1.15M. The transition is NOT monotone in size, the")
        print("  ordering by trainable count is not a ladder, and that outranks")
        print("  O0 -- it removes the frame the comparison was posed in.")


def score_o3(cells):
    print("\n=== O3 -- THE PLATEAU ============================================")
    print(f"  registered: at least 5 of 6 cells escape train NLL {PLATEAU}.")
    print("  Every rung at or above 1,759,738 has escaped 3 of 3; only the 1.15M")
    print("  KB has ever had a stuck seed (5 of 6, finding 56).")
    tot = esc_tot = 0
    for tag, key in (("OA", OA), ("OB", OB)):
        ms = rung(cells, key)
        if not ms:
            print(f"  {tag}: PENDING")
            continue
        n_esc, nll = escapes(ms)
        tot += len(ms)
        esc_tot += n_esc
        print(f"  {tag}: {n_esc}/{len(ms)} escaped   "
              + "/".join(f"{v:.4f}" for v in nll))
    if tot < 6:
        print(f"  INCOMPLETE: {tot} of 6 cells. Not called.")
    elif esc_tot >= 5:
        print(f"  O3 HELD ({esc_tot}/6). These levels are about capacity, so O0")
        print("  is a statement about architecture rather than about the recipe.")
    else:
        print(f"  O3 FALSIFIED: {esc_tot} of 6 escaped. At 2.1M these cells are")
        print("  OPTIMIZATION-limited, which reinterprets the level and outranks")
        print("  O0: a better schedule, not a different shape, is the next thing")
        print("  to try, and that is a change of direction.")


def score_o4():
    print("\n=== O4b -- OUT-OF-SAMPLE SUFFICIENCY, reported ahead of everything")
    print(f"  if it fails: all 6 cells land on the N=48 accuracy-to-solve curve")
    print(f"  within the LOO bar {LOO_BAR}. A cell OFF it is the headline over O0.")
    print("  Run (the nine sufficiency dirs, plus K, N and O):")
    print("    python3 -m scripts.acc_vs_solve --scan-dir \\\n      " +
          " \\\n      ".join(SUFFICIENCY_DIRS + [SCAN_K, SCAN_N, SCAN_O]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    # SCAN_DIRS is scanned too, as stage K's scorer does: the bracket rungs must
    # be FOUND, and a bracket that is merely absent reads like one that matches.
    ap.add_argument("--scan-dir", nargs="+",
                    default=[SCAN_O] + BRACKET_DIRS + SCAN_DIRS)
    args = ap.parse_args()

    # Guard 3: a new rung must not enter a pooled yardstick.
    assert SCAN_O not in SCAN_DIRS, "SCAN_O leaked into the pooled yardstick dirs"
    assert SCAN_K not in SCAN_DIRS, "SCAN_K leaked into the pooled yardstick dirs"
    assert SCAN_N not in SCAN_DIRS, "SCAN_N leaked into the pooled yardstick dirs"

    print("STAGE O -- size or shape? matched-size rungs inside the measured gap")
    print("=" * 70)
    print(f"  scanning: {' '.join(args.scan_dir)}")
    print(f"  bars (N=48, reused not refrozen): s* {S_RHO50} rho_50 / {S_MEAN} "
          f"mean_solve, free {FREE}, plateau {PLATEAU}, LOO {LOO_BAR}")
    print("  stage M's N=96 bars are a different size and appear nowhere below.")
    print(f"  the gap under test: ({GAP_LO:,}, {GAP_HI:,}) trainable")

    cells = load_pooled(args.scan_dir, args.boot)
    bracket_ok = control(cells)

    score_accuracy_first(cells)
    sa = report("OA", cells, OA, "no_mlp L=16 d176, the width route into the gap")
    sb = report("OB", cells, OB, "no_mlp L=20 d160, the depth route into the gap")
    score_o0(sa, sb)
    score_o1(sa, sb)
    score_o2(sa, sb)
    score_o3(cells)
    score_o4()

    print("\n" + "=" * 70)
    print("Read in this order: O4a (accuracy, never censors) -> O2 (did they")
    print("cross) -> O0 (size or shape) -> O1 (the minimum) -> O3 (plateau).")
    print("O2 or O3 falsified outranks O0: a censor makes the level a bound, and")
    print("an optimization-limited rung changes what to try next rather than what")
    print("to believe about shape.")

    # Guard 4, as an exit code rather than as a line of output: at registration
    # this makes the array's afterok dependency fail, so no cell starts against a
    # bracket that is not on disk; at scoring time it marks the run unreadable.
    if not bracket_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
