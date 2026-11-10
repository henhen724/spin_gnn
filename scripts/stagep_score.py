"""Score stage P: is the depth a COMPOSITION of 16 maps, or ONE map iterated 16x?

REGISTERED BEFORE THE FIRST CELL LANDS. This docstring IS the registration --
committed by scripts/slurm/stagep_register_2026-11-10.sbatch, which the array
scripts/slurm/sudoku2_n48_tied_P_2026-11-10.sbatch depends on with
--dependency=afterok. No cell can start ahead of it.

THE BRIEF CHANGED ON 2026-11-10, AND THIS IS THE FIRST STAGE UNDER IT. The goal
is a conceptually simple model that might be theoretically ANALYZABLE, and
explicitly NOT a model with the fewest parameters. What that changes here, and
what it deliberately does not:

  CHANGED. Trainable count stops being the figure of merit and stops being the
  tie-break between free cells. The programme had drifted into ranking by it --
  which is why the standing cell is a 16-layer attention-only model at 2.5M
  rather than the 6-layer `full L=6 d256` hub at 4.9M that scores the same
  (0.580 vs 0.592), even though 6 composed layers is a far more tractable object
  than 16. That ordering was ranking the candidates backwards.

  NOT CHANGED. The free line 0.740, s* = 0.1484, the 14-point grid, the plateau
  criterion 1.64, the LOO bar 0.0423, 3 seeds with the spread quoted, and the
  rule that a replacement needs every seed to cross. Those are how the
  measurement is made, not what is being optimized, and nothing about the new
  objective licenses loosening them mid-flight.

WHAT THE OBJECT BECOMES, which is the reason this cut leads

    untied (J2, the standing cell)       tied (P1, this stage)
      for i in 1..16:                      for i in 1..16:
          h <- Block_i(h)                      h <- Block(h)      <- SHARED
      a composition of 16 distinct         a fixed-point iteration
      maps; "layer 9" is its own             h_{t+1} = F(h_t);
      object with its own weights           "layer 9" is the 9th application
                                            of ONE operator

Direct deduction proceeds in SYNCHRONOUS ROUNDS: every forced node is emitted,
then the next round reads the result. A tied stack is the same shape of
computation -- one update rule applied until nothing further is forced. If the
tied cell learns DD, then "one layer = one deduction round" stops being an
analogy about a trained network and becomes a structural property of the
architecture, and the object a theory has to analyze is a single operator F
rather than sixteen unrelated ones. The 9.28x parameter drop is a side effect.

WHAT CHANGES: EXACTLY ONE KNOB

    J2  no_mlp      L=16 d192   2,504,890 trainable   rho_50 0.580  3/3   FREE
    P1  tied_no_mlp T=16 d192     270,010 trainable   ?                   this

Same depth, same width, nhead 4, ff 1024, warmup 0, 240k x 50ep, `--shuffle-adj
--fall`, the 14-point grid to 0.33 at 400 per bin, seeds 0/1/2. The only
difference is that the 16 blocks are one block.

VERIFIED BEFORE SUBMISSION (scripts/_tied_dry_check.py, 7/7): the untied path
still counts 2,504,890 exactly so nothing regressed; the tied cell counts
270,010 = (630d+58) + (4d^2+8d) exactly; encoder.layers has ONE entry while
n_iters is 16; the tied forward differs from the same weights applied once by
2.81 max-abs logit, so the iteration does not silently collapse; the feedforward
writes identically zero; gradient reaches the shared block and the frozen MLP
gets none; a saved tied checkpoint reloads with num_layers=16 and reproduces its
logits bit for bit.

P0  THE HEADLINE. At J2's exact shape, tying keeps the cell FREE: 3 of 3 seeds
    cross and the median rho_50 is <= 0.740, within one s* of J2's 0.580. That is
    what "the 16 layers are running one update rule" predicts. THE COMPETING
    HYPOTHESIS PREDICTS THE OPPOSITE: if the 16 blocks are doing 16 DIFFERENT
    things -- a pipeline, not an iteration -- then forcing them to be one map
    costs the deduction, and the cell lands past the line or censors.

P1  CAPACITY OR SHARING, AND THE CONFOUND IS NAMED IN ADVANCE. 270,010 is BELOW
    the only attention-only rung that has ever censored (H1 `L=12 d128`,
    879,418, 1/3 crossing). So:
      free       -> the untied "parameter floor" near 1.1M was an artifact of
                    counting UNTIED parameters. Capacity was never the binding
                    constraint; 270k suffices when the same block is reused, and
                    the floor is about what distinct weights buy, not about
                    capacity.
      censored   -> THIS STAGE CANNOT SEPARATE "tying costs the deduction" from
                    "270k is simply too few", and will not pretend to. The
                    REGISTERED follow-up, fixed now rather than chosen after the
                    numbers land, is a tied cell at d320 (613,818 trainable) or
                    d448 (1,088,698) -- sharing held fixed, capacity bought back.
                    d448 exceeds H1's floor, so a censor THERE would be about
                    tying and not about size.

P2  THE PLATEAU, AND IT IS THE PREDICTED FAILURE MODE. At least 2 of 3 cells
    escape train NLL 1.64. A tied block must serve 16 roles at once, which is a
    harder optimization than 16 blocks serving one role each, so a stuck seed
    here is more likely than anywhere in the untied ladder. READ THIS BEFORE ANY
    LEVEL: fewer than 2 escaping means the cell is optimization-limited, the
    rho_50 is a fact about the recipe rather than about the architecture, and
    the next move is a schedule (warmup, LR) and not a different architecture.
    Finding 56's lesson applies in full -- three seeds cannot find a stuck seed,
    so an escape count of 2/3 is reported with that caveat attached.

P3  ACCURACY FIRST, AS EVERY STAGE. Validation token accuracy is read before
    rho_50 because it never censors. All 3 cells sit on scripts/acc_vs_solve.py's
    N=48 fitted curve within the LOO bar 0.0423. A tied cell falling OFF that
    curve is the headline over P0: it would mean tying changes the
    accuracy-to-solve RELATION and not merely the level, which is a different
    and more interesting claim than anything P0 can return.

P4  THE ANALYZABILITY PAYOFF, REGISTERED AS A FOLLOW-UP AND NOT AS A PREDICTION.
    No rho_50 number changes the fact that a tied cell IS one operator applied
    T times. If P0 holds, T becomes a dial at EVAL time on fixed weights, which
    no untied cell can offer, and the next measurement is the T-sweep: rho_50 and
    accuracy at T in {8, 12, 16, 20, 24} on the SAME trained checkpoint. Its
    reading is registered here too, so it is not chosen afterwards:
      rises with T past 16  -> the block is an iterable deduction operator and
                               depth at train time was a budget, not a shape.
      flat past 16          -> it has learned a fixed point and stops early,
                               which is what DD does when nothing more is forced.
      falls past 16         -> the 16 applications are a pipeline after all, and
                               the fixed-point reading is wrong.
    None of these is scored by THIS file; it is the next stage's registration.

GUARDS, each one a mistake this programme has already made once
  1. N=48 only (finding 55: the N=48 minimum does not transfer to N=96). No N=96
     bar appears here and no N=48 number is differenced against an N=96 one.
  2. rho_50 CENSORS. A censored seed is a result, not a missing number, and the
     bound-aware median is what gets quoted.
  3. SCAN_P stays OUT of SCAN_DIRS, as E/G/H/I/J/K/N/O all do, so a new rung
     cannot enter a pooled yardstick by accident.
  4. The untied twin J2 is re-read from disk and checked against its frozen
     literals. The whole stage is a difference against it; a drifted control
     invalidates the comparison silently, so this exits non-zero instead.
  5. The tied and untied counts must DIFFER. If they ever coincide the sharing
     did not happen and the comparison is void.
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
from scripts.stageo_score import SCAN_O

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_P = os.path.join(_SCR, "spin_gnn", "sudoku_n48_tied_P_2026-11-10")

# The new rung. Key is (num_layers, nhead, arch_mode, d_model, ff, warmup);
# num_layers is how many times the SHARED block is applied.
P1 = (16, 4, "tied_no_mlp", 192, 1024, 0)

# A tied cell pays the per-block term ONCE, however many times it is applied.
PREDICTED_TRAINABLE = 270010            # (630*192+58) + (4*192^2 + 8*192)
UNTIED_AT_SAME_SHAPE = 2504890          # ... + (4*192^2 + 8*192)*16, i.e. J2

# The untied twin, frozen from finding 53 and re-read from disk by control().
#   rung key -> (label, trainable, rho_50 text, escapes_of_3)
TWIN = {
    (16, 4, "no_mlp", 192, 1024, 0): ("J2  no_mlp L=16 d192", 2504890, "0.580", 3),
    (12, 4, "no_mlp", 192, 1024, 0): ("H2  no_mlp L=12 d192", 1908922, "0.754", 3),
}
TWIN_DIRS = [SCAN_H, SCAN_I, SCAN_J, SCAN_K]

CENSOR_FLOOR = 879418       # H1 no_mlp L=12 d128, the only rung that has censored
FOLLOWUP = {"d320": 613818, "d448": 1088698}    # P1's registered follow-up widths

STANDING_TRAINABLE = STANDING[2]        # 2,504,890
STANDING_RHO50 = STANDING[3]            # 0.580


def control(cells):
    """Guard 4: the twin is what this stage's claim is a difference against."""
    print("\n=== CONTROL -- the untied twin, re-read from disk =================")
    ok = True
    for key, (label, trainable, txt, esc) in TWIN.items():
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
        print("  STOP: the twin drifted or is incomplete. P0 is a difference")
        print("  against J2 at the SAME shape; without it there is no comparison.")
    return ok


def report(cells):
    ms = rung(cells, P1)
    print("\n=== P1 cell -- tied_no_mlp T=16 d192, J2's shape with shared layers ===")
    if not ms:
        print("  nothing on disk yet.")
        return None
    s = summarize(ms)
    n_esc, nll = escapes(ms)
    print(f"  {s['n']}/3 seeds on disk, {s['n_cross']}/{s['n']} crossed 0.5")
    print(f"  rho_50 median {rho_txt(s):>9}   (free at or below {FREE}, untied twin "
          f"J2 {STANDING_RHO50:.3f}, H2 0.754)")
    print(f"  mean_solve    {s['mean_solve']:>9.4f}   [spread {s['spread_mean']:.4f}]")
    print(f"  trainable {s['trainable']:,}   (registration predicted "
          f"{PREDICTED_TRAINABLE:,}"
          + ("" if s["trainable"] == PREDICTED_TRAINABLE else
             "  <-- DISAGREES, quote the disk")
          + ")")
    print(f"  plateau escapes {n_esc}/{s['n']}   final train NLL "
          + "/".join(f"{v:.4f}" for v in nll))
    for m in sorted(ms, key=lambda r: r["seed"]):
        flag = "ok" if m["flag"] == "ok" else m["flag"].upper()
        print(f"    seed {m['seed']}  rho_50 {m['rho50']:.3f} {flag:>9}  "
              f"mean_solve {m['mean_solve']:.4f}")
    if s["trainable"] != PREDICTED_TRAINABLE:
        print("  STOP: this is not the cell the registration describes.")
    # Guard 5.
    if s["trainable"] == UNTIED_AT_SAME_SHAPE:
        print("  STOP: the count equals the UNTIED twin's. The sharing did not")
        print("  happen and P0 is void.")
    return s


def score_accuracy_first(cells):
    print("\n=== P3a -- VALIDATION TOKEN ACCURACY, READ FIRST =================")
    print("  The standing non-negotiable: accuracy never censors, so it is read")
    print("  before rho_50 and rho_50 only confirms it.")
    ms = rung(cells, P1)
    if not ms:
        print("  nothing on disk yet")
        return
    rows = sorted(acc_of(ms))
    for seed, a in rows:
        print(f"    P1  seed {seed}  val token acc {a:.4f}")
    print(f"    P1  median {median([a for _, a in rows]):.4f}")
    twin = rung(cells, (16, 4, "no_mlp", 192, 1024, 0))
    if twin:
        mt = median([v for _, v in acc_of(twin)])
        mp = median([a for _, a in rows])
        print(f"    J2 (untied twin) median {mt:.4f}   tied - untied {mp - mt:+.4f}")
        print("  An accuracy difference with no rho_50 difference, or the reverse,")
        print("  is what P3 is for -- see the curve command at the end.")


def score_p0(s):
    print("\n=== P0 -- IS THE DEPTH AN ITERATION OR A PIPELINE? ===============")
    print("  registered: at J2's exact shape, tying keeps the cell FREE -- 3 of 3")
    print(f"  crossing and median rho_50 <= {FREE:.3f}, within one s* = {S_RHO50}")
    print(f"  of J2's {STANDING_RHO50:.3f}. The competing hypothesis -- the 16")
    print("  blocks do 16 DIFFERENT things -- predicts past the line or censored.")
    if not s:
        print("  PENDING: nothing on disk yet.")
        return
    if s["n"] < 3:
        print(f"  INCOMPLETE: {s['n']}/3 seeds. Reported, not called.")
        return
    if not s["rho50_readable"]:
        print("  CENSORED -- see P1. A side-of-the-line call against a bound is a")
        print("  bound, not a measurement, and P1 says what it does and does not")
        print("  separate.")
        return
    d = s["rho50"] - STANDING_RHO50
    free = s["rho50"] <= FREE
    print(f"  tied {s['rho50']:.3f} ({'free' if free else 'past'})   untied "
          f"{STANDING_RHO50:.3f} (free)   tied - untied {d:+.3f} = "
          f"{d / S_RHO50:+.2f} s*")
    if free and s["n_cross"] == 3 and abs(d) <= S_RHO50:
        print("  P0 HELD. One operator applied 16 times reaches the same place as")
        print("  16 distinct operators, at the same depth and width. 'One layer =")
        print("  one deduction round' is now a property of the ARCHITECTURE and")
        print("  not an interpretation of a trained network, and the object to")
        print("  analyze is a single block F with h_{t+1} = F(h_t).")
        print("  Next: the T-sweep registered as P4 -- T is now an eval-time dial.")
    elif free and s["n_cross"] == 3:
        print(f"  P0 PARTLY HELD: free, but {abs(d):.3f} = {abs(d)/S_RHO50:.2f} s*")
        print("  from the twin. Sharing is affordable but not free; the honest")
        print("  reading is that the iteration works and costs a measurable margin.")
    else:
        why = ("not all three crossed" if s["n_cross"] < 3
               else f"median is {(s['rho50']-FREE)/S_RHO50:+.2f} s* past {FREE:.3f}")
        print(f"  P0 FALSIFIED ({why}). Forcing the 16 blocks to be ONE map costs")
        print("  the deduction, so the depth is a PIPELINE of distinct stages and")
        print("  not an iteration of one rule. That is a real constraint on any")
        print("  theory of this model, and it is the result -- read P1 next for")
        print("  what it does NOT establish.")


def score_p1(s):
    print("\n=== P1 -- CAPACITY OR SHARING? THE CONFOUND, NAMED IN ADVANCE ====")
    print(f"  {PREDICTED_TRAINABLE:,} is BELOW the only attention-only rung that")
    print(f"  has ever censored (H1 L=12 d128, {CENSOR_FLOOR:,}, 1/3 crossing).")
    if not s or s["n"] < 3:
        print("  PENDING or incomplete.")
        return
    if s["rho50_readable"] and s["rho50"] <= FREE and s["n_cross"] == 3:
        print("  P1 READS: the untied floor near 1.1M was an artifact of counting")
        print("  UNTIED parameters. Capacity was never the binding constraint --")
        print(f"  {PREDICTED_TRAINABLE:,} suffices when one block is reused, "
              f"{CENSOR_FLOOR / PREDICTED_TRAINABLE:.2f}x BELOW the censoring floor.")
        print("  What distinct weights buy is not capacity, and the 'smallest")
        print("  architecture' ladder was measuring the wrong quantity.")
    elif not s["rho50_readable"] or s["rho50"] > FREE:
        print("  P1 READS: NOT SEPARATED, and this stage does not pretend to.")
        print("  'Tying costs the deduction' and '270k is too few' both predict")
        print("  this. The registered follow-up holds sharing fixed and buys the")
        print("  capacity back:")
        for name, n in sorted(FOLLOWUP.items(), key=lambda kv: kv[1]):
            over = "ABOVE" if n > CENSOR_FLOOR else "below"
            print(f"    tied T=16 {name}  {n:,} trainable  ({over} H1's "
                  f"{CENSOR_FLOOR:,})")
        print("  d448 is above the floor, so a censor THERE is about tying and not")
        print("  about size. That is the discriminating cell, and it was chosen")
        print("  before these numbers landed.")


def score_p2(cells):
    print("\n=== P2 -- THE PLATEAU, THE PREDICTED FAILURE MODE ================")
    print(f"  registered: at least 2 of 3 cells escape train NLL {PLATEAU}. A tied")
    print("  block serves 16 roles at once, a harder optimization than 16 blocks")
    print("  serving one role each, so a stuck seed is likelier here than anywhere")
    print("  in the untied ladder. READ THIS BEFORE ANY LEVEL.")
    ms = rung(cells, P1)
    if not ms:
        print("  PENDING")
        return
    n_esc, nll = escapes(ms)
    print(f"  {n_esc}/{len(ms)} escaped   " + "/".join(f"{v:.4f}" for v in nll))
    if len(ms) < 3:
        print(f"  INCOMPLETE: {len(ms)} of 3 cells. Not called.")
    elif n_esc >= 2:
        print(f"  P2 HELD ({n_esc}/3). The level is about the architecture rather")
        print("  than the recipe, so P0 is readable as an architecture claim.")
        if n_esc == 2:
            print("  CAVEAT (finding 56): three seeds cannot find a stuck seed. 2/3")
            print("  is consistent with a recipe problem that more seeds would show.")
    else:
        print(f"  P2 FALSIFIED: {n_esc} of 3 escaped. This cell is OPTIMIZATION-")
        print("  limited, so its rho_50 is a fact about the schedule and not about")
        print("  tying. The next move is warmup or LR, NOT another architecture --")
        print("  and that outranks P0, which is not read as an architecture result.")


def score_p3():
    print("\n=== P3b -- OUT-OF-SAMPLE SUFFICIENCY, reported ahead of everything")
    print("  if it fails: all 3 cells land on the N=48 accuracy-to-solve curve")
    print(f"  within the LOO bar {LOO_BAR}. A cell OFF it is the headline over P0 --")
    print("  it would mean tying changes the accuracy-to-solve RELATION, not just")
    print("  the level. Run (the nine sufficiency dirs, plus K, N, O and P):")
    print("    python3 -m scripts.acc_vs_solve --scan-dir \\\n      " +
          " \\\n      ".join(SUFFICIENCY_DIRS + [SCAN_K, SCAN_N, SCAN_O, SCAN_P]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    # SCAN_DIRS is scanned too, as stage K's and O's scorers do: the twin must be
    # FOUND, and a twin that is merely absent reads like one that matches.
    ap.add_argument("--scan-dir", nargs="+",
                    default=[SCAN_P] + TWIN_DIRS + SCAN_DIRS)
    args = ap.parse_args()

    # Guard 3: a new rung must not enter a pooled yardstick.
    for name, d in (("SCAN_P", SCAN_P), ("SCAN_O", SCAN_O), ("SCAN_K", SCAN_K),
                    ("SCAN_N", SCAN_N)):
        assert d not in SCAN_DIRS, f"{name} leaked into the pooled yardstick dirs"
    # Guard 5, as an assertion on the registration's own arithmetic.
    assert PREDICTED_TRAINABLE != UNTIED_AT_SAME_SHAPE
    assert PREDICTED_TRAINABLE == (630 * 192 + 58) + (4 * 192 * 192 + 8 * 192)
    assert UNTIED_AT_SAME_SHAPE == (630 * 192 + 58) + (4 * 192 * 192 + 8 * 192) * 16

    print("STAGE P -- tied attention-only: one operator iterated, not 16 composed")
    print("=" * 70)
    print(f"  scanning: {' '.join(args.scan_dir)}")
    print(f"  bars (N=48, reused not refrozen): s* {S_RHO50} rho_50 / {S_MEAN} "
          f"mean_solve, free {FREE}, plateau {PLATEAU}, LOO {LOO_BAR}")
    print("  the brief of 11-10: ANALYZABILITY, not parameter count. The bars above")
    print("  are measurement and are unchanged; what changed is that trainable")
    print("  count is no longer the figure of merit or the tie-break.")
    print(f"  tied {PREDICTED_TRAINABLE:,} vs untied {UNTIED_AT_SAME_SHAPE:,} at the "
          f"SAME shape ({UNTIED_AT_SAME_SHAPE / PREDICTED_TRAINABLE:.2f}x)")

    cells = load_pooled(args.scan_dir, args.boot)
    twin_ok = control(cells)

    score_accuracy_first(cells)
    s = report(cells)
    score_p2(cells)          # before any level, as registered
    score_p0(s)
    score_p1(s)
    score_p3()

    print("\n" + "=" * 70)
    print("Read in this order: P3a (accuracy, never censors) -> P2 (plateau, the")
    print("predicted failure mode) -> P0 (iteration or pipeline) -> P1 (what a")
    print("censor does and does not separate) -> P3b (the curve).")
    print("P2 falsified outranks P0: an optimization-limited cell changes what to")
    print("try next rather than what to believe about tying. P4's T-sweep is a")
    print("follow-up registration and is scored by the next stage, not by this file.")

    if not twin_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
