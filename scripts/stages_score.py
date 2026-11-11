"""Score stage S: buy the capacity back at FIXED sharing -- is the tied plateau a
capacity floor or an obstruction that tying itself creates?

REGISTERED BEFORE THE FIRST CELL LANDS. This docstring IS the registration --
committed by scripts/slurm/stages_register_2026-11-10.sbatch, which the array
scripts/slurm/sudoku2_n48_tiedwidth_S_2026-11-10.sbatch depends on with
--dependency=afterok. No cell can start ahead of it.

WHAT STAGE P RETURNED, AND WHY THIS IS THE FOLLOW-UP IT NAMED

    P1  tied_no_mlp T=16 d192   270,010   rho_50 CENSORED (1/3 crossing)
        plateau escapes 0/3     train NLL 1.6646 / 1.6721 / 1.6681
        val token acc median 0.4567  vs the untied twin J2's 0.4940

  P2 was FALSIFIED (0 of 3 escaped train NLL 1.64), and P2 falsified outranks
  P0 by its own registration: an optimization-limited cell's rho_50 is a fact
  about the recipe, so stage P returned NO architecture result about tying.
  P1's confound stands exactly as it was named in advance -- 270,010 is below
  H1's 879,418 censoring floor, so "tying costs the deduction" and "270k is too
  few" both predict what was seen.

WHY WIDTH AND NOT THE SCHEDULE, THOUGH P2 REGISTERED "warmup or LR"

  P2's registered next move was a schedule change. The trajectory does not
  support the mechanism a warmup ramp fixes. Seed 0 runs 2.3154 -> 1.8909 ->
  1.8252 -> 1.8013 -> ... -> 1.6662 with no spike anywhere, a monotone
  decelerating descent, best val NLL at epoch 44-47 of 50, and a tail slope of
  -0.00041 nll/epoch. That is a curve ASYMPTOTING to a floor near 1.66, not one
  destabilized at init; all three seeds have the same shape and land within
  0.0075 of each other. Warmup and a lower LR both address early instability,
  and there is none to address.

  The knob that DOES discriminate is the one P1 registered, and it tests P2's
  diagnosis as well as P1's: if the floor near 1.66 is a CAPACITY floor, train
  NLL falls with width and a wide enough tied cell escapes 1.64. If it is an
  obstruction that sharing creates, the floor sits in the same place at every
  width, because no amount of width changes that one block must serve 16 roles.
  One knob, both questions. The schedule arm stays available and is registered
  as the follow-up under S5 if S1 says the floor does not move.

WHAT CHANGES: EXACTLY ONE KNOB, d_model

    P1  tied_no_mlp T=16 d192     270,010   below  H1's floor   (stage P, on disk)
    S1  tied_no_mlp T=16 d320     613,818   below  H1's floor   this stage
    S2  tied_no_mlp T=16 d448   1,088,698   ABOVE  H1's floor   this stage

  Everything else is P1's: arch_mode tied_no_mlp, T=16 applications of ONE
  shared block, nhead 4, ff 1024 (the feedforward writes identically zero),
  warmup 0, lr 1e-3 constant, 240k x 50ep, `--shuffle-adj --fall`, the 14-point
  grid to 0.33 at 400 per bin, seeds 0/1/2. Stage P's three d192 seeds are the
  third point of the width trend and are re-read from disk, not retyped.

  A tied cell pays the per-block term ONCE however many times it is applied:
  (630d + 58) + (4d^2 + 8d). Both widths divide by nhead 4 (80 and 112).

S0  THE HEADLINE. At d448 -- ABOVE the only attention-only rung that has ever
    censored -- the tied cell is FREE: 3 of 3 seeds cross and median rho_50
    <= 0.740. If it holds, one operator applied 16 times does direct deduction
    at N=48, "one layer = one deduction round" is a property of the
    ARCHITECTURE rather than an interpretation of a trained network, and the
    object a theory has to analyze is a single block F with h_{t+1} = F(h_t).
    THE COMPETING HYPOTHESIS PREDICTS THE OPPOSITE: if the 16 blocks are doing
    16 DIFFERENT things, forcing them to be one map costs the deduction at
    every width, and d448 censors or lands past the line with the plateau
    escaped. That second case is a clean negative and IS the result -- see S5.

S1  CAPACITY FLOOR OR SHARING OBSTRUCTION? READ BEFORE ANY LEVEL, as P2 was.
    Registered: final train NLL falls MONOTONICALLY across d192 -> d320 -> d448
    and at least 2 of 3 d448 seeds escape 1.64.
      falls and escapes  -> the stage P floor was CAPACITY. Stage P's censor was
                            about size, P1 resolves to "270k is too few", and
                            S0 is readable as an architecture claim.
      flat (all three widths inside 0.01 of each other) -> the floor is a
                            property of the TIED OBJECT. No schedule and no
                            width moves it, P2's "try warmup or LR" is not what
                            the evidence supports, and the finding is that one
                            shared block cannot be fit to 16 roles by this
                            recipe at this budget. S0 is then NOT read as
                            "tying cannot do DD" -- it is read as "tying cannot
                            be TRAINED to do DD here", which is a weaker and
                            more honest claim.
      falls but does not escape -> partial. Report the slope and the gap to
                            1.64, extrapolate nothing, and say which width
                            would be needed if the trend continued.

S2  ACCURACY FIRST, AS EVERY STAGE. Validation token accuracy is read before
    rho_50 because it never censors. Registered: accuracy rises monotonically
    with width from P1's 0.4567 toward the untied twin J2's 0.4940. An accuracy
    that moves while rho_50 stays censored is the more sensitive instrument and
    is reported as such; a rho_50 that moves while accuracy does not would mean
    the two have come apart, which is S4's business.

S3  THE DISCRIMINATING CELL, AND WHAT THE OTHER ONE IS FOR. Only d448 is above
    H1's 879,418, so only a censor THERE is about tying rather than about size.
    d320 at 613,818 is still BELOW the floor and CANNOT discriminate on its
    own: it is registered as a TREND point for S1 and nothing is called on it
    alone. This is the same mistake stage O made with one arm cancelled -- a
    two-arm comparison called on one arm -- and it is pre-empted here by saying
    in advance which arm carries which question.

S4  OUT-OF-SAMPLE SUFFICIENCY. All 6 cells land on scripts/acc_vs_solve.py's
    N=48 fitted accuracy-to-solve curve within the LOO bar 0.0423. A cell OFF
    it is the headline over S0: it would mean tying changes the
    accuracy-to-solve RELATION and not merely the level, which is a different
    and more interesting claim than any level S0 can return. That curve is an
    N=48 object and is never applied to N=96 (finding 55).

S5  WHAT RUNS NEXT, REGISTERED NOW RATHER THAN CHOSEN AFTERWARDS.
      S0 holds              -> the T-sweep registered as stage P's P4, on the
                               d448 checkpoint: rho_50 and accuracy at T in
                               {8, 12, 16, 20, 24} on FIXED weights. No untied
                               cell can offer that dial. It needs the
                               sudoku_circuits._capture_clean fix first, which
                               iterates len(model.encoder.layers) -- 1 for a
                               tied cell, 16 for its untied twin.
      S1 flat (no movement) -> the schedule arm P2 asked for, as a 2-cell
                               screen at d448: warmup 5 and lr 3e-4 cosine, one
                               knob each off this stage. If the floor does not
                               move there either, tying is recorded as a
                               NEGATIVE result for the analyzability route and
                               the route shifts to the shallow untied hub
                               (`full L=6 d256`, 6 composed layers, rho_50
                               0.592) -- which is already free and is the most
                               tractable object currently known to do DD.
      S0 falsified, plateau escaped -> a clean negative: capacity was bought
                               back, the cell trained, and it still does not
                               deduce. That is a real constraint on any theory
                               of this model and it is reported as the result,
                               not as a failure to be fixed.

GUARDS, each one a mistake this programme has already made once
  1. N=48 only (finding 55: the N=48 minimum does not transfer to N=96). No
     N=96 bar appears here and no N=48 number is differenced against an N=96 one.
  2. rho_50 CENSORS. A censored seed is a result, not a missing number, and the
     bound-aware median is what gets quoted.
  3. SCAN_S stays OUT of SCAN_DIRS, as E/G/H/I/J/K/N/O/P all do, so a new rung
     cannot enter a pooled yardstick by accident.
  4. The stage P baseline AND the untied twin J2 are re-read from disk and
     checked against their frozen literals. The width trend is a difference
     against P1; a drifted baseline invalidates it silently, so this exits
     non-zero instead.
  5. The tied and untied counts must DIFFER at every width. If they ever
     coincide the sharing did not happen and the comparison is void.
  6. Nothing is called on one arm (S3). Every checkpoint is the final epoch,
     here as everywhere in this programme.
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
from scripts.stageo_score import SCAN_O
from scripts.stagep_score import CENSOR_FLOOR, SCAN_P

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_S = os.path.join(_SCR, "spin_gnn", "sudoku_n48_tiedwidth_S_2026-11-10")

# Rung keys are (num_layers, nhead, arch_mode, d_model, ff, warmup); num_layers
# is how many times the SHARED block is applied and costs no parameters.
P1 = (16, 4, "tied_no_mlp", 192, 1024, 0)       # stage P, on disk
S1 = (16, 4, "tied_no_mlp", 320, 1024, 0)       # this stage, below H1's floor
S2 = (16, 4, "tied_no_mlp", 448, 1024, 0)       # this stage, ABOVE H1's floor


def tied_count(d):
    """A tied cell pays the per-block term once, however many times applied."""
    return (630 * d + 58) + (4 * d * d + 8 * d)


def untied_count(d, L):
    return (630 * d + 58) + (4 * d * d + 8 * d) * L


# The width trend, widest last. (key, label, predicted trainable)
WIDTHS = [
    (P1, "P1  tied T=16 d192", tied_count(192)),
    (S1, "S1  tied T=16 d320", tied_count(320)),
    (S2, "S2  tied T=16 d448", tied_count(448)),
]

# Frozen literals, re-read from disk by control(). P1 is stage P's own result
# (scorer 51815318); J2 and H2 are finding 53's.
#   rung key -> (label, trainable, rho_50 text, escapes_of_3)
BASELINES = {
    P1: ("P1  tied_no_mlp L=16 d192 (stage P)", 270010, "CENSORED", 0),
    (16, 4, "no_mlp", 192, 1024, 0): ("J2  no_mlp L=16 d192 (untied twin)",
                                      2504890, "0.580", 3),
}
BASELINE_DIRS = [SCAN_P, SCAN_H, SCAN_I, SCAN_J, SCAN_K]

P1_ACC = 0.4567             # stage P median val token accuracy
P1_NLL = (1.6646, 1.6721, 1.6681)
FLAT_BAND = 0.01            # S1: "all three widths inside 0.01" = no movement
HUB = ("full L=6 d256", 4899898, 0.592)     # the fallback route under S5


def control(cells):
    """Guard 4: the trend is a difference against P1, and S0 against J2."""
    print("\n=== CONTROL -- stage P's baseline and the untied twin, from disk ==")
    ok = True
    for key, (label, trainable, txt, esc) in BASELINES.items():
        ms = rung(cells, key)
        if not ms:
            print(f"  {label}: NOT ON DISK in the dirs scanned -- cannot check")
            ok = False
            continue
        s = summarize(ms)
        n_esc, _ = escapes(ms)
        got = rho_txt(s)
        same = (got == txt and s["trainable"] == trainable and n_esc == esc)
        print(f"  {label}\n      rho_50 {got:>9}  trainable {s['trainable']:>10,}  "
              f"escapes {n_esc}/{s['n']}   "
              + ("matches the frozen literals" if same else "<-- DISAGREES"))
        ok = ok and same
    if not ok:
        print("  STOP: a baseline drifted or is incomplete. The width trend is a")
        print("  difference against stage P's d192 cell; without it there is no")
        print("  trend, and S0 has no untied twin to be free or not free against.")
    return ok


def score_accuracy_first(cells):
    print("\n=== S2 -- VALIDATION TOKEN ACCURACY, READ FIRST ==================")
    print("  The standing non-negotiable: accuracy never censors, so it is read")
    print("  before rho_50 and rho_50 only confirms it. Registered: accuracy")
    print(f"  rises monotonically with width from P1's {P1_ACC:.4f} toward the")
    print("  untied twin J2's 0.4940.")
    meds = []
    for key, label, _ in WIDTHS:
        ms = rung(cells, key)
        if not ms:
            print(f"  {label}: nothing on disk yet")
            meds.append(None)
            continue
        rows = sorted(acc_of(ms))
        for seed, a in rows:
            print(f"    {label}  seed {seed}  val token acc {a:.4f}")
        m = median([a for _, a in rows])
        print(f"    {label}  median {m:.4f}")
        meds.append(m)
    got = [m for m in meds if m is not None]
    if len(got) == len(WIDTHS):
        mono = all(b >= a for a, b in zip(got, got[1:]))
        print(f"  accuracy by width: " + " -> ".join(f"{m:.4f}" for m in got))
        print("  MONOTONE in width" if mono else
              "  NOT monotone in width -- width is not simply buying accuracy")
    return meds


def report(cells):
    """The two new rungs, each reported on its own before anything is called."""
    out = {}
    for key, label, pred in WIDTHS[1:]:
        d = key[3]
        print(f"\n=== {label} -- one shared block, {pred:,} trainable ===")
        ms = rung(cells, key)
        if not ms:
            print("  nothing on disk yet.")
            out[key] = None
            continue
        s = summarize(ms)
        n_esc, nll = escapes(ms)
        print(f"  {s['n']}/3 seeds on disk, {s['n_cross']}/{s['n']} crossed 0.5")
        print(f"  rho_50 median {rho_txt(s):>9}   (free at or below {FREE}, untied "
              f"twin J2 0.580, stage P d192 CENSORED)")
        print(f"  mean_solve    {s['mean_solve']:>9.4f}   "
              f"[spread {s['spread_mean']:.4f}]")
        print(f"  trainable {s['trainable']:,}   (registration predicted {pred:,}"
              + ("" if s["trainable"] == pred else "  <-- DISAGREES, quote the disk")
              + ")")
        over = "ABOVE" if s["trainable"] > CENSOR_FLOOR else "below"
        print(f"  {over} H1's censoring floor {CENSOR_FLOOR:,} -- "
              + ("a censor here is about TYING, not about size" if over == "ABOVE"
                 else "a censor here cannot separate tying from size (S3)"))
        print(f"  plateau escapes {n_esc}/{s['n']}   final train NLL "
              + "/".join(f"{v:.4f}" for v in nll))
        for m in sorted(ms, key=lambda r: r["seed"]):
            flag = "ok" if m["flag"] == "ok" else m["flag"].upper()
            print(f"    seed {m['seed']}  rho_50 {m['rho50']:.3f} {flag:>9}  "
                  f"mean_solve {m['mean_solve']:.4f}")
        if s["trainable"] != pred:
            print("  STOP: this is not the cell the registration describes.")
        if s["trainable"] == untied_count(d, 16):          # guard 5
            print("  STOP: the count equals the UNTIED cell's at this width. The")
            print("  sharing did not happen and the comparison is void.")
        out[key] = s
    return out


def score_s1(cells):
    """The plateau, read before any level -- stage P's P2 precedent."""
    print("\n=== S1 -- CAPACITY FLOOR OR SHARING OBSTRUCTION? READ FIRST ======")
    print(f"  registered: final train NLL falls MONOTONICALLY across d192 -> d320")
    print(f"  -> d448, and at least 2 of 3 d448 seeds escape {PLATEAU}. Flat means")
    print("  the floor belongs to the tied OBJECT and no width or schedule moves")
    print("  it. This outranks S0, exactly as P2 outranked P0.")
    nlls, escs = [], []
    for key, label, _ in WIDTHS:
        ms = rung(cells, key)
        if not ms:
            print(f"  {label}: PENDING")
            nlls.append(None)
            escs.append(None)
            continue
        n_esc, nll = escapes(ms)
        med = median(nll)
        print(f"  {label}  escaped {n_esc}/{len(ms)}  median train NLL {med:.4f}   "
              + "/".join(f"{v:.4f}" for v in nll))
        nlls.append(med)
        escs.append(n_esc)
    if any(v is None for v in nlls):
        print("  INCOMPLETE: the trend needs all three widths. Not called.")
        return None
    mono = all(b <= a for a, b in zip(nlls, nlls[1:]))
    span = max(nlls) - min(nlls)
    print(f"  train NLL by width: " + " -> ".join(f"{v:.4f}" for v in nlls)
          + f"   (span {span:.4f}, flat band {FLAT_BAND})")
    wide_esc = escs[-1]
    if mono and wide_esc >= 2:
        print(f"  S1 READS: CAPACITY FLOOR. The floor moves with width and d448")
        print(f"  escapes {PLATEAU} at {wide_esc}/3. Stage P's censor was about SIZE,")
        print("  P1 resolves to '270k is too few', and S0 below is readable as an")
        print("  architecture claim about tying.")
        if wide_esc == 2:
            print("  CAVEAT (finding 56): three seeds cannot find a stuck seed.")
    elif span <= FLAT_BAND:
        print("  S1 READS: SHARING OBSTRUCTION. The floor sits in the same place at")
        print(f"  every width (span {span:.4f} <= {FLAT_BAND}). Width does not move it,")
        print("  and P2's 'try warmup or LR' is not what the evidence supports --")
        print("  one shared block cannot be fit to 16 roles by this recipe at this")
        print("  budget. S0 must then be read as 'tying cannot be TRAINED to do DD")
        print("  here', NOT as 'tying cannot do DD'. The registered next move is")
        print("  S5's 2-cell schedule screen at d448.")
    else:
        need = nlls[-1] - PLATEAU
        print(f"  S1 READS: PARTIAL. {'monotone' if mono else 'NOT monotone'}, and")
        print(f"  d448 is {need:+.4f} from {PLATEAU} with {wide_esc}/3 escaping.")
        print("  Report the slope and the gap; extrapolate no width from it.")
    return {"nlls": nlls, "escapes": escs, "monotone": mono, "span": span}


def score_s0(s_wide, s1):
    print("\n=== S0 -- DOES A TIED CELL DO DD ONCE CAPACITY IS BOUGHT BACK? ===")
    print(f"  registered: at d448 ({tied_count(448):,} trainable, ABOVE H1's")
    print(f"  {CENSOR_FLOOR:,}), 3 of 3 seeds cross and median rho_50 <= {FREE}.")
    print("  The competing hypothesis -- the 16 blocks do 16 DIFFERENT things --")
    print("  predicts censored or past the line at EVERY width.")
    if not s_wide:
        print("  PENDING: nothing on disk yet.")
        return
    if s_wide["n"] < 3:
        print(f"  INCOMPLETE: {s_wide['n']}/3 seeds. Reported, not called.")
        return
    optlim = s1 is not None and s1["span"] <= FLAT_BAND
    if not s_wide["rho50_readable"]:
        print("  CENSORED at d448, which is ABOVE the censoring floor -- so this")
        print("  one is NOT explained by size.")
        if optlim:
            print("  BUT S1 says the floor never moved, so the cell did not train.")
            print("  The honest reading is 'tying cannot be TRAINED to do DD by this")
            print("  recipe', not 'tying cannot do DD'. S5's schedule screen is next.")
        else:
            print("  S0 FALSIFIED, and with the plateau escaped this is the CLEAN")
            print("  NEGATIVE registered under S5: capacity was bought back, the cell")
            print("  trained, and it still does not deduce. The 16 applications are a")
            print("  PIPELINE of distinct stages, not an iteration of one rule. That")
            print("  is a real constraint on any theory of this model and it is the")
            print(f"  result. The analyzability route shifts to the hub {HUB[0]} at")
            print(f"  {HUB[1]:,} / rho_50 {HUB[2]} -- {6} composed layers, already free.")
        return
    d = s_wide["rho50"] - STANDING[3]
    free = s_wide["rho50"] <= FREE
    print(f"  tied d448 {s_wide['rho50']:.3f} ({'free' if free else 'past'})   untied "
          f"J2 {STANDING[3]:.3f} (free)   tied - untied {d:+.3f} = {d / S_RHO50:+.2f} s*")
    if free and s_wide["n_cross"] == 3:
        print("  S0 HELD. One operator applied 16 times does direct deduction at")
        print("  N=48. 'One layer = one deduction round' is a property of the")
        print("  ARCHITECTURE, not an interpretation of a trained network, and the")
        print("  object to analyze is ONE block F with h_{t+1} = F(h_t) -- at")
        print(f"  {tied_count(448):,} trainable against J2's {STANDING[2]:,} "
              f"({STANDING[2] / tied_count(448):.2f}x) at the same depth.")
        print("  Under the 11-10 brief that is the most tractable cell in the")
        print("  programme: one operator to read instead of 16, and T becomes an")
        print("  EVAL-TIME dial on fixed weights. Next is the T-sweep (S5).")
        if not optlim and s1 is not None and s1["escapes"][-1] < 2:
            print("  CAVEAT: it is free while fewer than 2 of 3 escaped the plateau.")
            print("  Report both; a free cell that never escaped is a curiosity about")
            print("  the plateau criterion, not a licence to ignore it.")
    else:
        why = ("not all three crossed" if s_wide["n_cross"] < 3
               else f"median is {(s_wide['rho50'] - FREE) / S_RHO50:+.2f} s* past {FREE}")
        print(f"  S0 FALSIFIED ({why}), at a width ABOVE the censoring floor.")
        print("  Read S1 for whether the cell trained at all before calling this a")
        print("  statement about tying.")


def score_s3(seen):
    print("\n=== S3 -- WHICH ARM CARRIES WHICH QUESTION ========================")
    print(f"  Only d448 ({tied_count(448):,}) is above H1's {CENSOR_FLOOR:,}, so only")
    print("  a censor THERE is about tying rather than size. d320 at")
    print(f"  {tied_count(320):,} is still BELOW the floor: it is a TREND point for")
    print("  S1 and nothing is called on it alone.")
    have = [lbl for (k, lbl, _) in WIDTHS[1:] if seen.get(k)]
    if len(have) < 2:
        print(f"  {len(have)} of 2 new arms on disk. Stage O was called on one arm")
        print("  after the other was cancelled; this stage says in advance that it")
        print("  will not be. Nothing discriminating is called until d448 lands.")


def score_s4():
    print("\n=== S4 -- OUT-OF-SAMPLE SUFFICIENCY, reported ahead of everything =")
    print("  if it fails: all 6 cells land on the N=48 accuracy-to-solve curve")
    print(f"  within the LOO bar {LOO_BAR}. A cell OFF it is the headline over S0 --")
    print("  it would mean tying changes the accuracy-to-solve RELATION, not just")
    print("  the level. Run (the nine sufficiency dirs, plus K, N, O, P and S):")
    print("    python3 -m scripts.acc_vs_solve --scan-dir \\\n      " +
          " \\\n      ".join(SUFFICIENCY_DIRS
                             + [SCAN_K, SCAN_N, SCAN_O, SCAN_P, SCAN_S]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+",
                    default=[SCAN_S] + BASELINE_DIRS + SCAN_DIRS)
    args = ap.parse_args()

    # Guard 3: a new rung must not enter a pooled yardstick.
    for name, d in (("SCAN_S", SCAN_S), ("SCAN_P", SCAN_P), ("SCAN_O", SCAN_O),
                    ("SCAN_K", SCAN_K), ("SCAN_N", SCAN_N)):
        assert d not in SCAN_DIRS, f"{name} leaked into the pooled yardstick dirs"
    # Guard 5, as an assertion on the registration's own arithmetic.
    assert tied_count(192) == 270010, tied_count(192)
    assert tied_count(320) == 613818, tied_count(320)
    assert tied_count(448) == 1088698, tied_count(448)
    assert untied_count(192, 16) == 2504890, untied_count(192, 16)
    for d in (192, 320, 448):
        assert tied_count(d) != untied_count(d, 16)
        assert d % 4 == 0, ("d_model must divide by nhead 4", d)
    assert tied_count(320) < CENSOR_FLOOR < tied_count(448), (
        "the registration's whole point is that d448 is above the censoring "
        "floor and d320 is below it")

    print("STAGE S -- tied attention-only: buy the capacity back, sharing fixed")
    print("=" * 70)
    print(f"  scanning: {' '.join(args.scan_dir)}")
    print(f"  bars (N=48, reused not refrozen): s* {S_RHO50} rho_50 / {S_MEAN} "
          f"mean_solve, free {FREE}, plateau {PLATEAU}, LOO {LOO_BAR}")
    print("  stage M's N=96 bars are a different size and appear nowhere below.")
    print(f"  the width trend at FIXED sharing: {tied_count(192):,} -> "
          f"{tied_count(320):,} -> {tied_count(448):,} trainable, T=16 throughout")
    print(f"  H1's censoring floor {CENSOR_FLOOR:,} sits BETWEEN d320 and d448, "
          "which is why d448 is the discriminating cell")

    cells = load_pooled(args.scan_dir, args.boot)
    base_ok = control(cells)

    score_accuracy_first(cells)
    seen = report(cells)
    s1 = score_s1(cells)                 # before any level, as registered
    score_s0(seen.get(S2), s1)
    score_s3(seen)
    score_s4()

    print("\n" + "=" * 70)
    print("Read in this order: S2 (accuracy, never censors) -> S1 (does the floor")
    print("move with width) -> S0 (does the tied cell deduce) -> S3 (which arm may")
    print("be called) -> S4 (the curve).")
    print("S1 outranks S0 by registration: a cell that never left the plateau says")
    print("nothing about tying, only about training it. S5's follow-ups are in the")
    print("docstring and are scored by the next stage, not by this file.")

    if not base_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
