"""Score stages D and E: can an ablated model be handed capacity back and cross?

    python3 -m scripts.capacity_back [--boot 2000]

THE QUESTION. scripts/rung_collapse.py found that seven complete rungs sit at
rho_50 0.740-0.765 across a 2.8x parameter range and six different
interventions, and that the architecture label inside that band predicts
nothing the seed does not (p = 0.569, with a positive control at p = 0.045 on
a separation 26x larger, so the honest claim is no DETECTABLE separation).

Two readings survive that table and it cannot choose between them:

  (A) BARRIER   0.76 is a floor any INCOMPLETE architecture hits however much
                capacity it is given back. The full model passes through it at
                L=3-4 and keeps descending only because it is complete.
  (B) CAPACITY  the ablations merely happened to cost about the same here.

The reason the grid cannot decide is structural and was invisible row by row:
EVERY ABLATED RUNG IN IT WAS RUN AT AT MOST THE HUB'S DEPTH, so no ablated
model has ever been handed more of a resource than the model it is compared
against. Stage D (51151957) gives one more WIDTH; stage E (51171824) gives one
more DEPTH. They are two arms of a single question, which is why they are
scored by one script.

THIS FILE IS THE REGISTRATION, TRANSCRIBED. Every threshold below is copied
from the sbatch headers, which were committed before the cells existed
(57a8210 for stage E, written while stage D task 1 was 2h54m into its run).
Nothing here is chosen after seeing a number.

  D1  full d192          registered mean_solve window [0.10, 0.45]
  D2  no_mlp d384        D2's own prediction: rho_50 <= 0.740
                         stage E's E3 is on record backing D2's FALSIFIER
  E1  no_mlp L=8         (A): rho_50 in [0.72, 0.80]   (B): <= 0.740
  E2  no_mlp L=12        (A): rho_50 in [0.72, 0.80]   (B): <= 0.619

  FALSIFIER OF (A), from the stage E header, "stated so it cannot be moved
  later": EITHER rung with a median rho_50 below 0.740 on at least 2 of 3
  crossing seeds. One such rung ends the barrier reading outright; there is no
  partial credit and no "but the other rung held".

  THE TIE-BREAK, also pre-registered, because (A)'s [0.72, 0.80] and (B)'s
  <= 0.740 OVERLAP on [0.72, 0.740]: a median of 0.740 exactly, or anywhere
  inside [0.72, 0.74], is scored as (A) and written as "did not clear the
  hub", because (B) is the claim that needs the evidence and a boundary value
  is not evidence. Written down in advance because stage C's P3 landed on its
  threshold to three decimals and had to be rescued by hand afterwards.
  So the operative falsification threshold is a median strictly BELOW 0.72,
  and this script says which branch it took every time.

  TRAINABILITY IS CHECKED FIRST AND OUTRANKS EVERYTHING. no_mlp escapes
  finding 44's 1.64 plateau on 2 of 3 seeds at L=6 and 0 of 3 at L=4, so a
  DEEPER attention-only stack failing to escape is the EXPECTED failure mode.
  A rung with 0 escapes is written as "did not train in 375k steps" and is
  NEVER evidence for (A) -- that is the trap stage C's P2 fell into. The
  budget is deliberately not raised to compensate.

Scored against the frozen bar s* = 0.1484 / 0.2918, not recomputed. SCAN_E is
added by hand because it is deliberately absent from SCAN_DIRS: stage E's
rungs carry entirely default hyperparameters and would otherwise pool into the
bar that every verdict in stages A-D was already measured against.
"""
import argparse
from collections import defaultdict

from scripts.minarch_analysis import (SCAN_C, SCAN_D, SCAN_DEFAULT, SCAN_E, SCAN_L6,
                                      cell_key, cell_metrics, load_all)
from scripts.stagec_score import (HUB, S_MEAN, S_RHO50, banner, escapes, rho_txt, rung,
                                  summarize, verdict)

# (A)'s window, and the overlap region the tie-break assigns to (A).
BARRIER_LO, BARRIER_HI = 0.72, 0.80
FREE = 0.740            # hub 0.592 + s* 0.1484, the free/not-free threshold
E2_CAPACITY = 0.619     # full L=8's 0.471 + s*, what (B) predicts for L=12

FULL_L8, FULL_L12 = (8, 4, "full"), (12, 4, "full")
NOMLP_L6 = (6, 4, "no_mlp")
D1 = (6, 4, "full", 192, 1024, 0)
D2 = (6, 4, "no_mlp", 384, 1024, 0)
E1, E2 = (8, 4, "no_mlp"), (12, 4, "no_mlp")


def barrier_call(s, n_esc):
    """The pre-registered decision for one rung, trainability first.

    Returns (verdict text, status). Only "falsified" and "held" are evidence;
    the three inconclusive statuses exist so that a rung which never got the
    chance to break (A) cannot be quietly counted as a rung that held it.
    """
    if n_esc == 0:
        return ("DID NOT TRAIN IN 375k STEPS -- 0 of %d seeds escaped the 1.64 plateau. "
                "NOT SCOREABLE, and explicitly NOT evidence for (A): an untrained cell "
                "carries no information about where a trained one would land. The "
                "follow-up is a longer-budget rerun as its own cell, not a reread of "
                "this one." % s["n"], "untrained")
    if s["n"] < 3:
        return ("INCOMPLETE -- %d of 3 seeds on disk. The registered rule is "
                "median-vs-median at 3 seeds; nothing is scored yet." % s["n"], "incomplete")
    if s["n_cross"] < 2:
        return ("CENSORED-DOMINANT -- only %d of 3 seeds crossed 0.5, so the falsifier's "
                "'at least 2 of 3 crossing seeds' clause cannot fire. Consistent with "
                "(A) but it is not evidence for it; this rung is simply worse than the "
                "band." % s["n_cross"], "censored")
    if not s["rho50_readable"]:
        return ("CENSORED median -- worse than the band, and the falsifier cannot fire.",
                "censored")
    r = s["rho50"]
    if r < BARRIER_LO:
        return ("(A) FALSIFIED BY THIS RUNG ALONE -- median rho_50 %.3f is below the "
                "tie-break floor %.2f on %d of 3 crossing seeds. Capacity given back DOES "
                "cross the band, so 0.76 is not a barrier and the pile-up is reading (B)."
                % (r, BARRIER_LO, s["n_cross"]), "falsified")
    if r <= FREE:
        return ("TIE-BREAK REGION [%.2f, %.3f] -- median rho_50 %.3f, margin %+.3f on the "
                "hub. SCORED AS (A) by the rule written before the cell existed: 'did not "
                "clear the hub'. (B) is the claim that needs evidence and a boundary value "
                "is not evidence. Report the margin, never the sign alone."
                % (BARRIER_LO, FREE, r, r - FREE), "held")
    if r <= BARRIER_HI:
        return ("(A) HOLDS ON THIS RUNG -- median rho_50 %.3f sits inside the registered "
                "[%.2f, %.2f] window, on the 0.76 rung, with more of a resource than the "
                "hub and nothing to show for it." % (r, BARRIER_LO, BARRIER_HI), "held")
    return ("OUTSIDE (A)'s WINDOW ON THE WRONG SIDE -- median rho_50 %.3f is above %.2f, "
            "so the extra capacity made it WORSE than the band. That is its own outcome "
            "and is not a vote for either reading." % (r, BARRIER_HI), "worse")


def paired(cells, key, what):
    """One line for a reference rung that was measured at the same budget."""
    ms = rung(cells, key)
    if not ms:
        return f"  {what:<34} not on disk"
    s = summarize(ms)
    return (f"  {what:<34} rho_50 {rho_txt(s):>8}   mean_solve {s['mean_solve']:.4f}   "
            f"{s['trainable']:>9,} trainable")


def score(cells, tag, asks, key, hub, extras, predictions):
    ms = rung(cells, key)
    if not ms:
        print(f"\n=== {tag} -- {asks} ===")
        print("  nothing on disk yet.")
        return "pending"
    s = summarize(ms)
    banner(tag, asks, s, hub, ms)
    n_esc, _ = escapes(ms)
    for line in predictions:
        print(f"  registered: {line}")
    for line in extras:
        print(line)
    text, status = barrier_call(s, n_esc)
    verdict(text)
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    # SCAN_E by hand: it is out of SCAN_DIRS on purpose, see the docstring.
    dirs = [SCAN_DEFAULT, SCAN_L6, SCAN_C, SCAN_D, SCAN_E]
    by = defaultdict(list)
    for r in load_all(dirs):
        by[cell_key(r)].append(r)
    cells = {k: dict(cell_metrics(v, args.boot), path=v[0]["path"]) for k, v in by.items()}

    hub = summarize(rung(cells, HUB))
    print("STAGES D AND E: giving an ablated model capacity back, on two axes")
    print(f"  {len(cells)} cells read from {len(dirs)} directories "
          f"(SCAN_E added by hand; it is out of SCAN_DIRS on purpose)")
    print(f"  frozen bar s* = {S_RHO50} rho_50 / {S_MEAN} mean_solve, NOT recomputed")
    print(f"  hub (L=6 d256 ff1024 h4 full): rho_50 {hub['rho50']:.3f}  "
          f"mean_solve {hub['mean_solve']:.4f}  {hub['trainable']:,} trainable")
    print(f"  (A) BARRIER window [{BARRIER_LO:.2f}, {BARRIER_HI:.2f}];  "
          f"falsified only by a median below {BARRIER_LO:.2f} on >=2 of 3 crossing seeds")
    print("\nPAIRED REFERENCE RUNGS, all at this exact budget and grid:")
    for key, what in ((FULL_L8, "full L=8 (E1's depth partner)"),
                      (FULL_L12, "full L=12 (E2's depth partner)"),
                      (NOMLP_L6, "no_mlp L=6 d256 (the single knob)")):
        print(paired(cells, key, what))

    status = {}

    status["D1"] = (score(
        cells, "D1", "bisect the width cliff, full d192", D1, hub,
        extras=[],
        predictions=[
            "mean_solve inside [0.10, 0.45]",
            "and 'strictly between its neighbours' on the width ladder "
            "(d256 0.5389 -> d192 -> d128 0.0273)",
        ]))

    status["D2"] = (score(
        cells, "D2", "THE WIDTH ARM, no_mlp d384", D2, hub,
        extras=["  note: E3 (stage E header, committed before this cell existed) is on "
                "record\n        backing THIS rung's FALSIFIER over its own prediction."],
        predictions=[
            f"D2's own prediction: rho_50 <= {FREE} (1.5x width buys the MLP back)",
            f"(A)'s prediction: lands on the 0.76 rung beside d256 no_mlp",
        ]))

    status["E1"] = (score(
        cells, "E1", "THE DEPTH ARM, no_mlp L=8 (interpolation)", E1, hub,
        extras=[],
        predictions=[
            f"(A): rho_50 in [{BARRIER_LO}, {BARRIER_HI}], mean_solve in [0.26, 0.38]",
            f"(B): rho_50 <= {FREE}, i.e. joins or beats the hub",
        ]))

    status["E2"] = (score(
        cells, "E2", "THE DEPTH ARM, no_mlp L=12 (extrapolation, the one that can kill (A))",
        E2, hub,
        extras=[],
        predictions=[
            f"(A): rho_50 in [{BARRIER_LO}, {BARRIER_HI}] again",
            f"(B): rho_50 <= {E2_CAPACITY}, full L=8's rung",
        ]))

    print("\n" + "=" * 72)
    # The three arms that can actually break (A); D1 only bisects the width cliff.
    arms = {k: status[k] for k in ("D2", "E1", "E2")}
    print("  arm status: " + ", ".join(f"{k}={v}" for k, v in sorted(arms.items())) +
          f",  D1(bisector)={status['D1']}")
    open_arms = [k for k, v in arms.items() if v not in ("falsified", "held", "worse")]
    if "falsified" in arms.values():
        who = [k for k, v in arms.items() if v == "falsified"]
        print(f"VERDICT: (A) BARRIER IS FALSIFIED, by {', '.join(who)}. An ablated rung handed")
        print("  more capacity than the hub crossed below the tie-break floor. The pile-up")
        print("  at 0.76 is reading (B) CAPACITY: those ablations cost about the same at")
        print("  that budget, and there is no floor. One rung is enough -- the registration")
        print("  says there is no partial credit and no 'but the other arm held'. Do not")
        print("  write 'the MLP is load-bearing in kind' anywhere.")
    elif open_arms:
        print(f"VERDICT: NOT YET. {3 - len(open_arms)} of 3 arms reached a terminal state; "
              f"{', '.join(open_arms)} still open.")
        print("  (A) cannot be confirmed by rungs that have not landed, and a rung that")
        print("  did not train, or whose median is censored, is NOT a rung that held.")
    else:
        held = [k for k, v in arms.items() if v == "held"]
        print(f"VERDICT: (A) BARRIER HOLDS ON EVERY ROW THAT COULD HAVE BROKEN IT "
              f"({len(held)} of 3 landed in the window; the rest came back WORSE, which is")
        print("  not a vote for either reading).")
        print("  Width (D2) and depth (E1, E2) both failed to cross, which is the")
        print("  two-independent-axes outcome the stage E header named. The strongest")
        print("  reading the grid supports: the minimal architecture that still does")
        print("  direct deduction is the COMPLETE one, and the smallest complete one")
        print("  measured is the hub. STILL NOT 'the architectures are equivalent' --")
        print("  see the resolution-limit caveat in scripts/rung_collapse.py.")
    print("=" * 72)

    print("\nE4 -- OUT-OF-SAMPLE SUFFICIENCY TEST, free and reported AHEAD of E1/E2 if it")
    print("  fails: run `python3 -m scripts.acc_vs_solve` and check that these rungs land")
    print("  on the depth ladder's accuracy-to-solve curve within the LOO bar 0.0423.")
    print("  Falling off would be the grid's first evidence of an architectural change")
    print("  doing something SOLVE-SPECIFIC rather than moving one quality axis.")


if __name__ == "__main__":
    main()
