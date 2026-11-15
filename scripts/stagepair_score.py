#!/usr/bin/env python3
"""Stage PAIR's registration: does supervising the CONTRAST produce a
representation that tracks determinacy under intervention?

Committed before any cell of this arm is trained. Nothing here reads a result;
the bars, the prediction, the trivial-solution control and the consequence of
each outcome are fixed in advance, and score() only applies them to whatever
the run writes.

The name is PAIR rather than the next letter because the arm is defined by what
it supervises -- matched clean/corrupted pairs -- and the letters have run out
at Z.

THE QUESTION. Stage Z (finding 78) measured the same quantity two ways and got
two answers: stage Y's determinacy channel ranks determined above undetermined
nodes at AUROC 0.802 across the population with the revealed-neighbour count
held fixed, and at 0.49 -- two of three seeds BELOW chance -- on matched pairs
that differ by the single token which flips the predicate, with the placebo
control correctly at 0.47. The representation is a partial predicate that is
not causally wired: it carries real information about determinacy across
instances and none about determinacy under the intervention that changes it.

Stage Z closes the two explanations that were about PLUMBING (the coupling was
too weak; a monotone gate can only suppress) by showing there is nothing to
transmit: the channel's median response to the corruption is 0.0103 against the
placebo's 0.0102, on a quantity whose spread across nodes is 3.5. What it does
NOT close is the OBJECTIVE. Stages X and Y both supervise a static per-node
ranking on the CLEAN input, and finding 78 shows that objective is satisfiable
at AUROC 0.95 by something with no counterfactual sensitivity at all. Nothing
in either loss ever asked g to differ between a forced v and an un-forced v.

    HYPOTHESIS. The counterfactual blindness is a property of the objective,
    not of the architecture. Supervise the contrast directly and the
    representation tracks the predicate under intervention.

THE ARM. Stage Y's recipe exactly -- tied_no_mlp T=16 d320 ff1024 h4 with the
determinacy channel, 614,138 trainable, 240,000 instances, batch 32, lr 1e-3
constant, 3 seeds, --shuffle-adj -- plus one term:

    L_pair = mean over cases [ MARGIN - ( g_clean[p, v] - g_broken[p, v] ) ]_+

where v is a round-0 node whose forcing one hint recolour destroys, p is the
position whose next-token prediction is v's node identity, g_broken is the
channel on the prompt with that one token rewritten, and the mean runs over the
instances in the batch that carry such a case. MARGIN = 1.0 and LAMBDA_PAIR =
1.0, both FIXED A PRIORI and deliberately never swept, as lambda was for X and
Y. The corrupted forward feeds only this term: after the edit the deduction is
no longer forced, so the planted trace is no longer a correct continuation and
an NLL on it would train the model to assert a deduction its prompt does not
support.

PRICED FIRST, by 52415691 (scripts/stagepair_price.py), because STATUS's next
step said to:

    yield          0.8034 of instances carry a pair (16,068 of 20,000)
    scan cost      0.04 ms/instance, 0.2 min at 240,000 -- free
    step cost      76.55 ms -> 151.43 ms, ratio 1.978x, peak 4.55 -> 8.99 GiB
    projection     22.0 min/epoch, 18.3 h for 50 epochs

so the arm costs what STATUS guessed it would (~2x) and the corruption itself
costs nothing: build_triple is a copy with one index written, no solver call.

THE TRIVIAL SOLUTION, AND THE CONTROL THAT CATCHES IT. This is the load-bearing
part of the registration, not an afterthought. The corrupted prompt differs
from the clean one by EXACTLY ONE TOKEN. So L_pair can be driven to zero by a
detector for the EDIT rather than for determinacy: "a hint colour token at a
position belonging to a neighbour of v differs from what the rest of the prompt
implies" is a local, linearly available feature, and a model that learns it
scores a perfect margin while computing nothing about forcing. An arm that
reports a high matched-pair AUROC and nothing else cannot tell the two apart.

The placebo is already built for every case and is exactly the control. A
placebo recolours a hint that is NOT a neighbour of v and leaves v forced, so:

    a determinacy representation      g_clean - g_placebo ~= 0
    an edit detector                  g_clean - g_placebo large, like corrupt

P2 below is therefore not a nice-to-have. P2 FAILING VOIDS P1: a high corrupted
AUROC with a high placebo AUROC is the trivial solution, and is reported as
`edit_detector`, not as a success.

The placebo is NOT trained on. It appears only at evaluation, so it stays an
out-of-sample control rather than another objective the model can satisfy.

WHAT IS MEASURED, on held-out cases from a stream no training instance touched.

  P1  MATCHED-PAIR AUROC, CORRUPTED. The quantity stage Z put at 0.49: over
      cases, P(g_clean[p,v] > g_broken[p,v]), ties at one half. This is the
      registered success metric, NOT the pooled determinacy AUROC, which
      finding 78 showed is satisfiable without the predicate.

  P2  MATCHED-PAIR AUROC, PLACEBO. The trivial-solution control above.

  P3  POOLED DETERMINACY AUROC. What X and Y both bought and what this arm must
      not lose: the population-level ranking, same cells and same stream as
      stage Z's cross-check.

  P4  BEHAVIOUR. The conditional guard rate on sudoku_skip's 350 cases, whose
      denominator is READ and never assumed (stage Y's were 236/228/217, never
      350), and rho_50 on the 14-point grid. P1 is about the representation;
      P4 is about whether anything downstream uses it. They can dissociate and
      the verdict distinguishes them.

  P5  THE TAX. Validation token accuracy FIRST, before any of the above, and
      the plateau level, so "costs accuracy" stays distinguishable from
      "has not converged" -- finding 77b, where stage Y's seed 2 was still
      stuck at 50 epochs and could not be interpreted at all.

CONVERGENCE, REGISTERED UP FRONT because finding 77b says an arm with this
channel must. A seed whose final train NLL exceeds FLOOR + STUCK_TOL is STUCK
and its P1 is NOT INTERPRETED, exactly as stage Y reported "Y4 failed, Y1-Y3
not interpreted" rather than scoring a seed that never left the plateau. The
budget stays at stage Y's 50 epochs so the comparison holds one thing fixed;
what changes is that the stuck case is declared in advance instead of
discovered.
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# the setting, copied from the arm this one extends
# ---------------------------------------------------------------------------
N_SEEDS = 3
MIN_SEEDS = 2                        # of 3, as every stage in this project
EPOCHS = 50                          # stage Y's budget, deliberately unchanged
MARGIN = 1.0                         # fixed a priori, never swept
LAMBDA_PAIR = 1.0                    # fixed a priori, never swept
N_TRAINABLE = 614138 + 0             # the term adds no parameter

# Streams. Training uses 0..240299. The evaluation cases must come from
# neither that nor stage Z's two streams, or "held out" is not true.
PAIR_EVAL_SEED_START = 920_000_000
PAIR_EVAL_N_CASES = 1000             # vs stage Z's 350; the term makes the
                                     # matched-pair number the headline, so it
                                     # gets a tighter error bar
GUARD_SEED_START = 600_000_000       # stage X, Y and the control all used it
AUROC_SEED_START = 70_000_000_000    # determinacy_both_arms.csv

# ---------------------------------------------------------------------------
# what stage Z measured, which is what this arm is held against
# ---------------------------------------------------------------------------
Z_MATCHED_CORRUPT = (0.4457, 0.5714, 0.4543)   # the number this arm must move
Z_MATCHED_PLACEBO = (0.4800, 0.4686, 0.4714)   # the control, already at chance
Z_WITHIN_STRATUM = (0.8059, 0.8021, 0.8019)
Y_AUROC_PUBLISHED = (0.9493, 0.9478, 0.9480)
Y_GUARD_DENOMS = (236, 228, 217)               # NEVER 350
FLOOR = 1.600                                  # the plateau every arm lands on
STUCK_TOL = 0.030                              # above FLOOR+this: not converged
FREE = 0.740                                   # the rho_50 gate
Y_RHO50 = (0.772, 0.758, None)                 # None: censored above the grid

# ---------------------------------------------------------------------------
# the bars
# ---------------------------------------------------------------------------
# P1. Chance is 0.50 and stage Z measured 0.49. 0.65 is a move of kind rather
# than of degree: it cannot be reached by noise on 1000 cases (the binomial
# standard error at chance is 0.016, so 0.65 is >9 sigma) and it is well short
# of the 1.0 a perfect edit detector would score, so clearing it is evidence
# the contrast was learned at all, not evidence of how.
PAIR_BAR = 0.65
# P2. The placebo must STAY where stage Z found it. This band is the same one
# stage Z used for its own placebo check.
PLACEBO_BAND = (0.40, 0.60)
# P3. What must not be lost.
POOLED_BAR = 0.90
# P4. A guard rate that would be a change of kind rather than the 1-in-681
# stage Y produced. Deliberately low: the question is whether the behaviour
# moves at all, not whether it is solved.
GUARD_BAR = 0.10

# ---------------------------------------------------------------------------
# the prediction, in writing, before the run
# ---------------------------------------------------------------------------
# P1 passes and P2 holds. Reason: the margin is a direct, dense supervision of
# the exact quantity stage Z found at chance, on 80% of instances every epoch,
# and the channel demonstrably has the capacity to rank determinacy (0.95
# pooled, 0.80 within stratum) -- what it lacked was ever being asked.
PRED_PAIR = 0.85
PRED_PAIR_TOL = 0.10
# P2 is where the real uncertainty is, and the prediction is deliberately not
# confident: the edit detector is a SIMPLER function than the predicate, so a
# gradient-following model has every reason to find it first. Predicting the
# placebo stays in band is predicting that it does not, and this is the
# prediction most likely to miss.
PRED_PLACEBO = 0.55
PRED_PLACEBO_TOL = 0.08
# P4 fails. Reason: stage Z showed the representation never moved, so the
# guard's null had a sufficient cause upstream; this arm removes that cause but
# nothing in it wires the now-responsive g to a DECISION, and finding 74 showed
# the untied guard re-ranks to another forced node rather than stopping. A
# responsive representation with an unresponsive behaviour is the predicted
# outcome, and it is the informative one.
PRED_GUARD_FIRES = False
PRED_RHO50_WORSE_THAN_Y = True       # the term is a second tax on the budget


def matched_pair_auroc(d):
    """P(g_clean > g_broken) with ties at one half, from a list of deltas.

    The same estimator stage Z used, so the two numbers are comparable without
    a correction. Kept here rather than imported so the registration is
    readable on its own.
    """
    d = list(d)
    if not d:
        return None
    wins = sum(1 for x in d if x > 0)
    ties = sum(1 for x in d if x == 0)
    return (wins + 0.5 * ties) / len(d)


def converged(per_seed):
    """Which seeds may be interpreted at all. Finding 77b, registered here."""
    return [p["final_train_nll"] <= FLOOR + STUCK_TOL for p in per_seed]


def check_p1(per_seed, ok):
    """P1 on the seeds that converged, MIN_SEEDS of them required."""
    scored = [p["pair_auroc_corrupt"] for p, c in zip(per_seed, ok) if c]
    if len(scored) < MIN_SEEDS:
        return {"pass": False, "reason": "fewer than MIN_SEEDS converged",
                "n_converged": len(scored)}
    n = sum(1 for a in scored if a >= PAIR_BAR)
    return {"pass": n >= MIN_SEEDS, "n_over_bar": n, "scored": scored,
            "bar": PAIR_BAR}


def check_p2(per_seed, ok):
    """P2, the trivial-solution control. In band on EVERY converged seed."""
    scored = [p["pair_auroc_placebo"] for p, c in zip(per_seed, ok) if c]
    in_band = [PLACEBO_BAND[0] < a < PLACEBO_BAND[1] for a in scored]
    return {"pass": bool(scored) and all(in_band), "scored": scored,
            "band": list(PLACEBO_BAND)}


def check_p3(per_seed, ok):
    scored = [p["pooled_auroc"] for p, c in zip(per_seed, ok) if c]
    return {"pass": bool(scored) and all(a >= POOLED_BAR for a in scored),
            "scored": scored, "bar": POOLED_BAR}


def check_p4(per_seed, ok):
    """The guard rate is CONDITIONAL and its denominator is read, never
    assumed. Three scripts in this project have made that mistake."""
    rows = [(p["guard_rate"], p["guard_denom"])
            for p, c in zip(per_seed, ok) if c]
    for _rate, denom in rows:
        if denom == 350:
            raise AssertionError(
                "guard_denom == 350 means the UNCONDITIONAL denominator was "
                "written; stage Y's were 236/228/217. Read it, do not assume "
                "it.")
    return {"pass": bool(rows) and sum(1 for r, _d in rows if r >= GUARD_BAR)
            >= MIN_SEEDS,
            "rates": [r for r, _d in rows], "denoms": [d for _r, d in rows],
            "bar": GUARD_BAR}


def score_the_prediction(per_seed, ok):
    """Scored by the stated route, not just the verdict -- finding 77c, where a
    prediction got the right answer for the wrong reason and was pinned as a
    miss."""
    corr = [p["pair_auroc_corrupt"] for p, c in zip(per_seed, ok) if c]
    plac = [p["pair_auroc_placebo"] for p, c in zip(per_seed, ok) if c]
    guard = [p["guard_rate"] for p, c in zip(per_seed, ok) if c]
    if not corr:
        return {"scored": False, "reason": "no seed converged"}
    mean_c = sum(corr) / len(corr)
    mean_p = sum(plac) / len(plac)
    return {
        "scored": True,
        "pair_predicted": PRED_PAIR, "pair_measured": mean_c,
        "pair_hit": abs(mean_c - PRED_PAIR) <= PRED_PAIR_TOL,
        "placebo_predicted": PRED_PLACEBO, "placebo_measured": mean_p,
        "placebo_hit": abs(mean_p - PRED_PLACEBO) <= PRED_PLACEBO_TOL,
        "guard_predicted_fires": PRED_GUARD_FIRES,
        "guard_fires": bool(guard) and max(guard) >= GUARD_BAR,
    }


def verdict(p1, p2, p3, p4, n_converged):
    """Every enumerated outcome has a name fixed before the run."""
    if n_converged < MIN_SEEDS:
        return "not_converged"
    if p1["pass"] and not p2["pass"]:
        # The one the placebo exists to catch, and the reason P2 is not
        # optional: this looks like a success on P1 alone.
        return "edit_detector"
    if p1["pass"] and p2["pass"] and not p3["pass"]:
        return "counterfactual_at_the_cost_of_the_population_ranking"
    if p1["pass"] and p2["pass"] and p4["pass"]:
        return "representation_and_behaviour"
    if p1["pass"] and p2["pass"]:
        # The predicted outcome. A representation that now tracks the predicate
        # under intervention, with a behaviour that still does not use it --
        # which would move the open question from "is it represented" to "what
        # reads it", and is the first arm in X/Y/Z able to say that.
        return "responsive_representation_unused"
    if not p1["pass"]:
        # The term is dense, direct supervision of the exact quantity. If it
        # cannot move it, the blindness is not the objective's doing and the
        # architecture is implicated -- a bigger result than a pass.
        return "contrast_not_learnable"
    return "unclassified"


def score(out_dir):
    with open(os.path.join(out_dir, "pair_summary.json")) as f:
        S = json.load(f)
    per = S["per_seed"]
    assert len(per) == N_SEEDS, len(per)
    ok = converged(per)
    p1, p2 = check_p1(per, ok), check_p2(per, ok)
    p3, p4 = check_p3(per, ok), check_p4(per, ok)
    v = verdict(p1, p2, p3, p4, sum(ok))
    return {"verdict": v, "converged": ok, "P1": p1, "P2": p2, "P3": p3,
            "P4": p4, "prediction": score_the_prediction(per, ok),
            "held_against": {"stage_Z_matched_corrupt": list(Z_MATCHED_CORRUPT),
                             "stage_Z_matched_placebo": list(Z_MATCHED_PLACEBO),
                             "stage_Y_pooled": list(Y_AUROC_PUBLISHED)}}


def _self_test():
    """The bars applied to synthetic rows, so a typo in a comparison is caught
    before a 18-hour cell lands rather than after."""
    def seed(c, p, pooled=0.95, guard=0.0, denom=230, nll=FLOOR):
        return {"pair_auroc_corrupt": c, "pair_auroc_placebo": p,
                "pooled_auroc": pooled, "guard_rate": guard,
                "guard_denom": denom, "final_train_nll": nll}

    # the predicted outcome
    per = [seed(0.86, 0.52), seed(0.84, 0.49), seed(0.88, 0.55)]
    ok = converged(per)
    assert all(ok)
    r = verdict(check_p1(per, ok), check_p2(per, ok), check_p3(per, ok),
                check_p4(per, ok), sum(ok))
    assert r == "responsive_representation_unused", r

    # the trivial solution: P1 clears and the placebo moves with it
    per = [seed(0.95, 0.93), seed(0.96, 0.94), seed(0.94, 0.92)]
    ok = converged(per)
    r = verdict(check_p1(per, ok), check_p2(per, ok), check_p3(per, ok),
                check_p4(per, ok), sum(ok))
    assert r == "edit_detector", r

    # the term does not move the contrast at all -- stage Z's numbers again
    per = [seed(c, p) for c, p in zip(Z_MATCHED_CORRUPT, Z_MATCHED_PLACEBO)]
    ok = converged(per)
    r = verdict(check_p1(per, ok), check_p2(per, ok), check_p3(per, ok),
                check_p4(per, ok), sum(ok))
    assert r == "contrast_not_learnable", r

    # two seeds stuck: nothing is interpreted, whatever P1 says
    per = [seed(0.90, 0.50), seed(0.90, 0.50, nll=FLOOR + 0.2),
           seed(0.90, 0.50, nll=FLOOR + 0.2)]
    ok = converged(per)
    assert sum(ok) == 1
    r = verdict(check_p1(per, ok), check_p2(per, ok), check_p3(per, ok),
                check_p4(per, ok), sum(ok))
    assert r == "not_converged", r

    # the unconditional denominator must raise, not be scored
    per = [seed(0.86, 0.52, denom=350)] * 3
    try:
        check_p4(per, converged(per))
    except AssertionError as e:
        assert "350" in str(e)
    else:
        raise AssertionError("guard_denom == 350 was accepted")

    # the estimator, including the tie convention
    assert matched_pair_auroc([1.0, 1.0, -1.0, -1.0]) == 0.5
    assert matched_pair_auroc([0.0, 0.0]) == 0.5
    assert matched_pair_auroc([1.0, 0.0]) == 0.75
    assert matched_pair_auroc([]) is None

    # a prediction that is right about P1 and wrong about the placebo is a
    # partial hit and must say so, per finding 77c
    per = [seed(0.86, 0.93), seed(0.84, 0.94), seed(0.88, 0.92)]
    sp = score_the_prediction(per, converged(per))
    assert sp["pair_hit"] and not sp["placebo_hit"], sp
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    if not a.out_dir:
        ap.error("out_dir or --self-test")
    print(json.dumps(score(a.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
