#!/usr/bin/env python3
"""Stage Z's registration: is the installed determinacy signal the PREDICATE or
a COVARIATE of it?

Committed before the probe runs. Nothing here reads a stage-Z result; the bars,
the prediction and the consequence of each outcome are fixed in advance, and
`score()` only applies them to whatever the probe writes.

THE QUESTION. Stages X and Y both installed a determinacy readout at
AUROC ~= 0.95 (X on the existing readout, 0.9462/0.9467/0.9469; Y on its own
channel, 0.9493/0.9478/0.9480) and NEITHER produced a guard: break node v's
determinacy and the model names v anyway, at the control's rate. Two
explanations for that null result have been refuted -- the readout answering
two questions at once (Y separates them) and a monotone block gate only being
able to suppress (Y's channel re-ranks per node). A third survives, and it has
never been tested:

    THE SIGNAL THAT WAS INSTALLED MAY NOT BE THE PREDICATE AT ALL.

DD's determinacy predicate at node v is

    forced(v)  <=>  v uncoloured  AND  |{colours among v's revealed
                                         neighbours}| == q - 1

(sudoku_determinacy.determinacy_targets:139-143 computes exactly that: the
target is `n_known_colors == n_colors - 1`, masked where v is already
coloured.) The DISTINCT-COLOUR COUNT is the predicate's ingredient. But a node
with more revealed neighbours is more likely to show q-1 distinct colours, so
the plain COUNT of revealed neighbours is a covariate of the predicate -- and
the two come apart exactly where the guard probe operates.

WHY THAT IS TESTABLE HERE, AND WHY NOW. sudoku_skip's corruption RECOLOURS one
revealed neighbour w of v (sudoku_skip.py:119-127): w stays revealed, so

    n_known_nbrs(v)    is INVARIANT under the corruption, by construction
    n_known_colors(v)  falls from q-1 to q-2, which is what destroys forcing

A readout that tracks the count is therefore BLIND to the one edit the guard
probe tests, no matter how it is coupled to the emission. That predicts both
null results at once, and it predicts them for a reason that no amount of
re-coupling (a third auxiliary term, a different lambda, a wider channel) can
fix. If it is right, the "install the representation" programme is closed and a
successor has to supervise something a count cannot satisfy.

TWO MEASUREMENTS, BOTH ON EXISTING CHECKPOINTS, NO TRAINING.

  Z1  COUNTERFACTUAL RESPONSE OF THE CHANNEL. On stage Y's 350 guard cases
      (the same stream, seed_start 600000000), read the channel's own score
      g[p, v] on the clean, corrupted and placebo inputs. The corruption
      destroys v's determinacy; the placebo recolours a hint that is not v's
      neighbour and leaves v forced. If g represents the predicate, g[p, v]
      must fall under the corruption and not under the placebo.

      Stage Y only: load_checkpoint rebuilds the channel from config.json, and
      `return_determinacy=True` raises on a model built without one
      (sat_transformer2.py:263-267). Stage X has no separate score -- its
      "representation" IS the node logit height, whose counterfactual response
      was already measured and is at chance against the placebo.

  Z2  THE COVARIATE CEILING ON THE AUROC. On the same 600-instance set the
      published AUROCs were measured on (seed_start 70000000000), compute the
      AUROC of the MODEL-FREE count n_known_nbrs over determinacy_targets' own
      scored cells. No model is loaded. If a count alone reaches ~0.95, the
      model's 0.95 is not evidence that it computes the predicate.

THE PREDICTION, REGISTERED. Z1 FAILS and Z2 CLEARS ITS BAR: the channel does
not respond to the corruption (win rate at chance, normalised drop under 0.05,
no case where its response is large enough to have mattered), and the count
alone scores 0.92 +- 0.04. Point prediction for the count's AUROC: 0.920.

WHAT EACH OUTCOME MEANS -- fixed now, so no result gets read twice.

  Z1 fail, Z2 pass   The installed signal is a covariate, not the predicate.
                     X and Y coupled a count to the emission and a count
                     cannot express the guard. The programme is closed, and a
                     successor must supervise the distinct-colour count (or
                     forcing itself) with a loss a count cannot satisfy --
                     which means supervising the CORRUPTED instance, where the
                     two disagree.
  Z1 pass            The representation IS counterfactually responsive and the
                     failure is purely in the coupling. Then Y's channel is
                     the thing to re-examine (one rank-1 vector s, one lambda,
                     never swept) and this explanation is dead.
  Z1 fail, Z2 fail   The signal is neither the predicate nor a count. A third
                     explanation is needed and none is registered here.

Bars are two-sided on purpose: Z1 PASSING is the result that refutes the
hypothesis this file was written to state.
"""

import json
import os
import sys

# --------------------------------------------------------------------------
# the case sets, quoted from the jobs that defined them so a later run cannot
# drift onto a different stream and still compare
# --------------------------------------------------------------------------
GUARD_SEED_START = 600_000_000      # sudoku_skip default; stage X, Y and the control all used it
GUARD_N_INSTANCES = 400
GUARD_N_CASES = 350
AUROC_SEED_START = 70_000_000_000   # determinacy_both_arms.csv, all six rows
AUROC_N_INSTANCES = 600

# Stage Y's published numbers, read back by the probe as a self-consistency
# check rather than retyped into a claim: the probe rebuilds the same case list
# and must reproduce the same unconditional emit rate.
Y_EMITS_V = (0.6742857142857143, 0.6742857142857143, 0.6742857142857143)
Y_AUROC_PUBLISHED = (0.9492910644901765, 0.947801015501777, 0.9480)
X_AUROC_PUBLISHED = (0.9462, 0.9467, 0.9469)
CONTROL_AUROC_PUBLISHED = (0.7635, 0.7610, 0.7774)

# --------------------------------------------------------------------------
# bars
# --------------------------------------------------------------------------
# Z1. Chance on the win rate is 0.50 by construction: the corruption and the
# placebo are both one-token hint recolours, so a model indifferent to which
# token was edited scores 0.50.
WIN_BAR = 0.60
# The signed drop g_clean - g_broken, in units of g's own spread ACROSS NODES
# at the same position -- the spread the AUROC is computed over. Scale-free,
# because g's absolute scale is not fixed by anything.
NORM_BAR = 0.25
# The fraction of cases where the channel's own counterfactual response, in the
# units that reach the emission (d log sigma(g)), is at least v's clean margin
# over its best rival -- i.e. large enough to have changed the argmax on its
# own. This is the behavioural version of the question and the only one that
# could have produced a guard.
#
# CONDITIONAL on v being the clean argmax, and the probe writes flip_denom so
# the denominator is read and not assumed. Where v is not the clean argmax the
# margin is negative and any positive response clears it trivially, which would
# report a guard that is really just a case the model got wrong to begin with.
# The published guard rates are conditional on clean_emits_v for the same
# reason; stage Y's denominators were 236/228/217, never 350.
FLIP_BAR = 0.10
MIN_SEEDS = 2                        # of 3, as every stage in this project

# Z2. If a model-free count clears this on the same scored cells, the
# published 0.95 is not evidence of a predicate computation.
COV_BAR = 0.90

# -------------------------------------------------------------------------
# Z3, REGISTERED 2026-11-15 AFTER Z1 AND Z2 RETURNED THE UNREGISTERED OUTCOME.
# -------------------------------------------------------------------------
# Z1 and Z2 together landed on "neither predicate nor count": the channel does
# not respond to the corruption at all (win rate 0.4914/0.5171/0.4571, flip
# fraction 0.0000 over 236/228/217) and yet the plain count tops out at 0.8857
# against the model's 0.9493. The model therefore knows something past the
# count. Z3 asks how much, on a scale whose two ends are fixed by construction:
#
#   WITHIN-STRATUM AUROC = P(a forced cell outranks an unforced cell WITH THE
#   SAME n_known_nbrs), pooled over strata by pair count.
#
#     n_known_nbrs    -> exactly 0.5  (constant within its own stratum, all ties)
#     n_known_colors  -> exactly 1.0  (the target is a threshold of it)
#
# Both ends are asserted by the probe, so the scale is self-calibrating and the
# model's value is directly interpretable: 0.5 means the published 0.95 is the
# count and nothing else, 1.0 means it resolves the predicate given the count.
#
# This is the question Z2 was too coarse to settle. A pooled AUROC mixes "ranks
# a 5-neighbour node over a 2-neighbour node" with "ranks a forced node over an
# unforced one at the same degree of revelation", and only the second is the
# predicate. Stratifying removes the first by construction.
WITHIN_BAR = 0.90        # at or above: it resolves the predicate given the count
WITHIN_FLOOR = 0.70      # below: the extra information is weak and diffuse
# Registered prediction: between the two, nearer the top. The model beats the
# count by 0.064 of pooled AUROC, which is not nothing, but Z1 shows whatever it
# has is not wired to the token that carries it.
PRED_WITHIN = 0.80
PRED_WITHIN_TOL = 0.08


def check_z3(within, tautologies):
    """`tautologies` is {covariate: within-stratum AUROC}; the two known ends
    must come back exact or the stratification is not what it claims."""
    nbrs = tautologies.get("n_known_nbrs")
    cols = tautologies.get("n_known_colors")
    return {
        "within_stratum_auroc": within,
        "scale_calibrated": bool(nbrs is not None and cols is not None
                                 and abs(nbrs - 0.5) < 1e-9 and abs(cols - 1.0) < 1e-9),
        "resolves_predicate_given_count": bool(within >= WITHIN_BAR),
        "extra_information_is_weak": bool(within < WITHIN_FLOOR),
        "prediction_within_tolerance": bool(abs(within - PRED_WITHIN) <= PRED_WITHIN_TOL),
    }

# The registered point prediction, scored like every other prediction here.
PRED_COV_AUROC = 0.920
PRED_COV_TOL = 0.040
PRED_WIN_RANGE = (0.45, 0.60)
PRED_NORM_MAX = 0.05
PRED_FLIP = 0.00


def check_z1(per_seed):
    """per_seed: list of dicts with win_rate, norm_drop, flip_frac."""
    win = sum(s["win_rate"] >= WIN_BAR for s in per_seed)
    norm = sum(s["norm_drop"] >= NORM_BAR for s in per_seed)
    return {
        "seeds_clearing_win_bar": win,
        "seeds_clearing_norm_bar": norm,
        "responsive": bool(win >= MIN_SEEDS and norm >= MIN_SEEDS),
    }


def check_z2(cov_auroc):
    return {
        "covariate_auroc": cov_auroc,
        "auroc_is_substantially_a_count": bool(cov_auroc >= COV_BAR),
    }


def score_the_prediction(per_seed, cov_auroc):
    """Did the registered prediction hold? Scored on its own terms, including
    the route and not only the verdict -- finding 77c's lesson."""
    lo, hi = PRED_WIN_RANGE
    return {
        "predicted_cov_auroc": PRED_COV_AUROC,
        "measured_cov_auroc": cov_auroc,
        "cov_auroc_error": cov_auroc - PRED_COV_AUROC,
        "cov_auroc_within_tolerance": abs(cov_auroc - PRED_COV_AUROC) <= PRED_COV_TOL,
        "win_rates_in_predicted_range": [bool(lo <= s["win_rate"] < hi) for s in per_seed],
        "norm_drops_under_predicted_max": [bool(abs(s["norm_drop"]) < PRED_NORM_MAX)
                                           for s in per_seed],
        "flip_fracs_at_predicted_zero": [bool(s["flip_frac"] == PRED_FLIP) for s in per_seed],
    }


def verdict(z1, z2):
    """The three registered outcomes, in the order the docstring fixes them."""
    if z1["responsive"]:
        return "representation_is_responsive__failure_is_the_coupling"
    if z2["auroc_is_substantially_a_count"]:
        return "representation_is_a_covariate__programme_closed"
    return "neither_predicate_nor_count__unregistered"


def verdict_with_z3(z1, z2, z3):
    """Z3 splits the unregistered outcome Z1+Z2 landed on. Only reached when
    the stratified measurement is present."""
    if z1["responsive"]:
        return "representation_is_responsive__failure_is_the_coupling"
    if z3["resolves_predicate_given_count"]:
        # it computes the predicate and the emission ignores it anyway
        return "computes_the_predicate_and_does_not_use_it"
    if z3["extra_information_is_weak"]:
        return "signal_is_the_count_plus_diffuse_residue"
    return "partial_predicate_not_causally_wired"


def score(out_dir):
    """Apply the bars to the probe's own output. Reads, never assumes."""
    with open(os.path.join(out_dir, "cfresp_summary.json")) as fh:
        s = json.load(fh)
    per_seed = s["per_seed"]
    cov = s["covariate_auroc"]["n_known_nbrs"]
    z1, z2 = check_z1(per_seed), check_z2(cov)
    out = {
        "z1": z1,
        "z2": z2,
        "prediction": score_the_prediction(per_seed, cov),
        "verdict": verdict(z1, z2),
    }
    strat = s.get("stratified")
    if strat:
        # pooled across seeds the same way every other stage pools: per seed,
        # then reported together, never averaged into one number that hides a
        # seed. The verdict uses the median.
        withins = [c["model"] for c in strat["per_seed"]]
        srt = sorted(withins)
        med = srt[len(srt) // 2] if len(srt) % 2 else 0.5 * (srt[len(srt) // 2 - 1]
                                                             + srt[len(srt) // 2])
        z3 = check_z3(float(med), strat["tautologies"])
        z3["per_seed"] = withins
        out["z3"] = z3
        out["verdict"] = verdict_with_z3(z1, z2, z3)
    return out


def _self_test():
    """Both registered outcomes, plus the boundary, on synthetic numbers."""
    blind = [{"win_rate": 0.50, "norm_drop": 0.01, "flip_frac": 0.0}] * 3
    live = [{"win_rate": 0.82, "norm_drop": 0.61, "flip_frac": 0.44}] * 3
    z1b, z1l = check_z1(blind), check_z1(live)
    assert not z1b["responsive"] and z1l["responsive"]
    assert verdict(z1b, check_z2(0.93)) == "representation_is_a_covariate__programme_closed"
    assert verdict(z1b, check_z2(0.71)) == "neither_predicate_nor_count__unregistered"
    assert verdict(z1l, check_z2(0.93)).startswith("representation_is_responsive")
    # one seed over the bar is not two
    mixed = [{"win_rate": 0.90, "norm_drop": 0.90, "flip_frac": 0.5},
             {"win_rate": 0.50, "norm_drop": 0.00, "flip_frac": 0.0},
             {"win_rate": 0.50, "norm_drop": 0.00, "flip_frac": 0.0}]
    assert not check_z1(mixed)["responsive"]
    p = score_the_prediction(blind, 0.918)
    assert p["cov_auroc_within_tolerance"]
    assert all(p["win_rates_in_predicted_range"])
    assert all(p["flip_fracs_at_predicted_zero"])
    assert not score_the_prediction(blind, 0.74)["cov_auroc_within_tolerance"]
    # the bars must be two-sided: a responsive channel has to be reportable
    assert check_z1(live)["seeds_clearing_win_bar"] == 3

    # Z3. The scale's two ends are asserted, so a mis-stratified run cannot
    # report a plausible-looking middle value and pass.
    taut = {"n_known_nbrs": 0.5, "n_known_colors": 1.0}
    assert check_z3(0.95, taut)["scale_calibrated"]
    assert not check_z3(0.95, {"n_known_nbrs": 0.61, "n_known_colors": 1.0})["scale_calibrated"]
    assert check_z3(0.95, taut)["resolves_predicate_given_count"]
    assert check_z3(0.62, taut)["extra_information_is_weak"]
    assert not check_z3(0.80, taut)["resolves_predicate_given_count"]
    assert not check_z3(0.80, taut)["extra_information_is_weak"]
    assert check_z3(0.80, taut)["prediction_within_tolerance"]
    assert not check_z3(0.55, taut)["prediction_within_tolerance"]
    # all three Z3 verdicts reachable, and Z1 still overrides
    assert verdict_with_z3(z1b, check_z2(0.74), check_z3(0.95, taut)) \
        == "computes_the_predicate_and_does_not_use_it"
    assert verdict_with_z3(z1b, check_z2(0.74), check_z3(0.62, taut)) \
        == "signal_is_the_count_plus_diffuse_residue"
    assert verdict_with_z3(z1b, check_z2(0.74), check_z3(0.80, taut)) \
        == "partial_predicate_not_causally_wired"
    assert verdict_with_z3(z1l, check_z2(0.74), check_z3(0.95, taut)) \
        .startswith("representation_is_responsive")
    print("ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    elif len(sys.argv) > 1:
        print(json.dumps(score(sys.argv[1]), indent=2))
    else:
        print(__doc__)
