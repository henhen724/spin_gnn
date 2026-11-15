"""Stage Y: make the determinacy signal RE-RANK the candidates instead of
gating the emission on or off. Registered before the cells are submitted;
this docstring is the registration.

WHY THERE IS A STAGE Y
----------------------
Stage X added a multi-label BCE on the existing readout's node-token logits
against DD's deducibility predicate, lambda = 1.0, zero new parameters, and
it worked as a *representation* intervention and failed as a *behavioural*
one:

All three seeds, control -> stage X, never a single seed quoted alone:

  determinacy AUROC   0.7610/0.7635/0.7774 (finding 73, untrained)
                   -> 0.9398/0.9416/0.9425 (stage X, measured)
  val token accuracy  0.4830/0.4830/0.4833 -> 0.4780/0.4767/0.4778
  aggregate solve     0.313/0.310/0.323    -> 0.147/0.133/0.153
  solved/unsolved     0.993/0.990/0.993    -> 0.653/0.713/0.690

The registered scorer returned `representation_without_use`: X3 and X4
passed, X1 and X2 failed. rho_50 went 0.754/0.757/0.765 -> 0.855/0.864/0.866,
the wrong way by 0.103 of mean shift.

The decisive one is X2, the counterfactual guard probe of finding 68, run
unchanged on the stage-X cells. The guard rate conditional on clean_emits_v
is 1.0000/1.0000/0.9568 against the control's 1.0000/1.0000/1.0000: when
node v's determinacy is destroyed, the stage-X model names v anyway, at the
control's rate, in every seed. So determinacy at AUROC 0.9467 inside the
deciding logits bought NO counterfactual responsiveness at all.

That rules out the obvious reading of the behavioural damage. There is no
over-eager guard, because there is no conditional guard: nothing in the
emission responds to whether the particular node in question is determined.
What the BCE bought is a change in the ABSOLUTE height of the node block
against the terminal and round-marker tokens -- a density-independent bias
toward stopping -- which is exactly the degree of freedom cross-entropy
never constrained and exactly the one a per-node BCE on absolute logit
height can move. The 14-point grid shows it directly: the SOLVED/UNSOLVED
call is flat at ~1.00 for the control at every density, and for stage X it
traces a U, ~0.93 at rho 0.94, bottoming at 0.42-0.45 around rho 0.69-0.75,
back to ~0.94 at rho 0.33. Right where stopping is right, right where
"stuck" is right, near chance in the middle where the call requires
carrying the deduction on.

A global bias is not a guard, and this is the lesson stage Y is built on:
a per-node loss on absolute logit height does not produce per-node
conditional behaviour. It produces a shift of the whole block.

What this does NOT rest on is the easy end, k = 0..4 nodes left to colour,
where stopping is most of the task. There the term helps in two seeds and
hurts in the third:

  k=0/1/2, seed 0   0.540/0.320/0.000 -> 0.160/0.110/0.000
  k=0/1/2, seed 1   0.330/0.070/0.140 -> 0.950/0.980/1.000
  k=0/1/2, seed 2   0.440/0.930/0.070 -> 0.890/0.940/0.830

The control is erratic there on its own (k=2 is 0.000/0.140/0.070 against
k=3's 0.970/0.960/0.960), it is 100 instances a point, and reading the
first draft of this file off seed 1 alone produced a confident claim that
the term had fixed a known pathology. It had not, in seed 0. The easy end
is reported and not relied on.

Finding 74 says that is the wrong shape. Re-reading finding 68's
skip_behavior.csv for what the untied model emits INSTEAD of the node whose
determinacy was destroyed: it emits a node token 96.6% of the time, that
node is still forced 90.3% of the time, and it names the broken node again
in 0 of 350 cases. The real guard does not stop. It moves the argmax to a
different, still-forced node. Stage X could only ever have built the other
kind, because a BCE on the absolute height of z_{p,v} says "v is/is not
deducible" one node at a time and never says "u before v".

THE ONE THING STAGE Y CHANGES
-----------------------------
The determinacy signal gets its own output channel, and the emission is a
sum of the two channels, so determinacy enters as a term that can reorder
node tokens relative to each other:

    z_{p,v}                      the existing readout, unchanged
    g_{p,v} = <h[p] * s, E_v>    the determinacy channel, s in R^d learned
    emission over node tokens:   z'_{p,v} = z_{p,v} + log sigmoid(g_{p,v})
    emission over everything else: unchanged

`E` is the input embedding (`token_embed`, line 65) and `W` the readout
(`readout`, line 113); they are SEPARATE matrices in this model, never
tied, so g is a genuinely different function of h than z is and the sum can
change which node is on top. `s` is one learned vector, d_model = 320 new
parameters: 613,818 -> 614,138, +0.05%, and the arm ratio against J2's
2,504,890 goes 4.08x -> 4.08x. The parameter budget is not what this stage
spends.

Two properties are the point, and both are structural rather than hoped for:

  1. It can re-rank. The monotone alternative cannot. A gate applied to the
     node block as a whole, z -> z + log sigmoid(a(z - tau)) for two learned
     scalars, also breaks the softmax's additive-constant gauge and costs
     almost nothing, and I was going to build it until finding 74. It is
     strictly increasing in z, so it preserves the order WITHIN the node
     block exactly and can only suppress the block against the non-node
     tokens. That is a stop-shaped guard: stage X's failure mode with a
     cheaper implementation.

  2. Cross-entropy and the BCE stop competing for the same numbers. In
     stage X both losses wrote to z_{p,v}: CE wants one node's logit far
     above every other, the BCE wants the absolute height of all ~35% of
     determined uncoloured nodes high at once. Those are different
     objectives on one set of scalars and -0.0055 of token accuracy is what
     the conflict cost. Here the BCE writes to g and CE writes to z, and
     they meet only in the sum.

Everything else is held: the same tied cell (tied_no_mlp, T=16, d320,
ff1024, h4), the same 240k instances, the same 50 epochs, the same three
seeds 0/1/2, the same lambda = 1.0 NOT swept, the same deducibility target
at the same pre_node positions over still-uncoloured nodes only (positive
rate 0.3544, scored fraction 0.2466), the same 14-point rho grid at 400
instances per point.

BARS, IN THE ORDER THEY ARE SCORED
----------------------------------
Y4 first, because it decides whether the rest means anything, and note that
stage X PASSED a bar of this shape at 0.4776 -- a floor the intervention
cannot fail is not a control, and this one is set against the measured
control instead of a historical floor.

  Y4  SANITY. val token accuracy >= 0.4800 in all 3 seeds, against the
      control's 0.4830/0.4830/0.4833. Stage X scored 0.4780/0.4767/0.4778
      and would FAIL this. If Y4 fails the separate channel did not buy back
      the interference and Y1-Y3 are not interpreted.

  Y1  THE POINT. rho_50 completely separated BELOW the control's
      0.754/0.757/0.765, i.e. max(Y) < 0.754. Whether it also clears the
      0.740 free line is reported SEPARATELY and is not what Y1 asks; the
      gate is a project goal, not this stage's hypothesis.

  Y2  THE MECHANISM. guard rate conditional on clean_emits_v below 0.90 in
      at least 2 of 3 seeds, against the control's 1.0000/1.0000/1.0000.

  Y3  THE SHAPE, and this bar is new in stage Y. Among the cases where the
      guard fires, the fraction that emit a node token >= 0.90 and the
      fraction whose emitted node is still forced >= 0.60, in at least 2 of
      3 seeds -- the untied model's 0.966 and 0.903 are the thing being
      reproduced. Y2 without Y3 is stage X's stop-shaped guard again and
      counts as a refutation of the design, not a partial success.

WHAT STAGE Y CANNOT SHOW
------------------------
It cannot show the signal has to arrive this way. g is read from the same
h[p] the readout already sees, so a negative result rules out this coupling
and not the existence of some other one. It cannot separate the two
properties above: if Y succeeds, re-ranking capacity and the end of the
CE/BCE conflict both changed at once, and telling them apart needs a third
arm with g trained by BCE but DETACHED from the emission, which is the
natural stage Z and is not submitted here. It cannot speak to the untied
model's guard being partly provenance-specific (finding 71) or to its being
a window at t ~ 3-6 (finding 72); stage Y installs a guard at every
position and does not try to make it a window.

STATUS: REGISTERED 2026-11-14, after 52285051 and 52285735 both completed
and stagex_score returned `representation_without_use`. The draft was
written before X2 was readable and said explicitly that an X2 pass would
make the diagnosis wrong and change the design. X2 failed at
1.0000/1.0000/0.9568, which is the branch the design was built for, and
nothing in it moved.

SUBMITTED 2026-11-14 at commit 062c9a4, after this file was already
committed at a05847a: array 52336800 (3 cells, sganguli, 20:00:00) with
52336801 chained --dependency=afterok to run Y2, Y3 and print the verdict
this module returns. The channel was smoke-tested first
(scripts/stagey_smoke.py, five assertions, all pass) and the training job
re-runs that script before it trains, because three of the five -- an
off-by-default flag that does not take, an emission that drops
log sigma(g), a BCE that still reaches the readout -- would otherwise
produce a plausible run of the WRONG stage.

CANCELLED 2026-11-14 at 2h29 and RESUBMITTED at commit ca7e5ab as array
52348732 with 52348737 chained --dependency=afterok, into a NEW dated
directory (..._Y_2026-11-14b) so the abandoned cells are kept and cannot be
confused with these. 52336800 initialized the
channel's s to ONES, on reasoning that was written into the code comment as
if it were a measurement: that g = <h, E_v> would be "a small common
offset". It is not. h is LayerNorm'd and E_v ~ N(0, I), so at init g had
std 21 and log sigma(g) std 10.3, reaching -59, against node logits whose
own spread is 0.61 -- a random PER-NODE penalty seventeen times the signal
it perturbs, scrambling the ranking before step 1. Seeds 0 and 2 trained
through it (epoch 12: val acc 0.4600/0.4582, AUROC 0.940); seed 1 never
left the initial plateau (train_nll 3.67 -> 3.96 at epoch 0, flat at ~3.95
through epoch 12, val acc 0.032, AUROC 0.59, det_bce pinned at ln 2). Y4
requires ALL THREE seeds >= 0.4800, so score() would have returned
"Y4 failed, Y1-Y3 not interpreted" whatever cells 0 and 2 did; 21 more
GPU-h for an uninterpretable verdict, on a model whose channel began as
noise injection rather than as the design registered above.

THE ONLY CHANGE FROM 52336800 is this initialization, s = 0 instead of
s = 1: g is then identically 0 and log sigma(g) = -log 2 for every node at
every position, so the order within the node block starts exactly the
control's, which is what the original comment claimed and did not deliver.
It is not a saddle -- dL/ds_i = sum (dL/dg) h_i E_vi is O(1) -- and the
count is still +320. THE DESIGN, THE FOUR BARS AND THIS REGISTRATION ARE
UNTOUCHED: no bar was moved after seeing a number, because no bar was
readable from a cancelled array. The smoke test gained a SIXTH assertion,
that the channel's contribution is constant across the node axis at init
and moves after one BCE step, since nothing in the first five looks at the
SCALE of the perturbation -- which is how a design error passed a smoke
test that asserts the forward map is algebraically right.

================================================================================
MID-RUN PREDICTION, written 2026-11-14 19:30 at epoch 32 of 50 of 52348732,
BEFORE any cell finished and before rho_50 or the guard rate existed for any
of them. THE BARS ABOVE ARE NOT TOUCHED. Recorded so Y4 and Y1 are a test of
this rather than a story told afterwards. This is the second use of the
instrument; the first, stage X's `f95e2b7`, landed 0.007 from the measured
value.
================================================================================

All three cells are healthy -- no crash, no plateau, the init bug is gone --
and three things are already readable off the training logs.

  THE CHANNEL LEARNS DETERMINACY EXACTLY AS WELL AS THE READOUT DID, AND THIS
  IS THE SURPRISE. At the same epoch 32, stage Y's det_bce is 0.2773 / 0.2858
  / 0.2937 against stage X's 0.2787 / 0.2869 / 0.2882, and stage Y's val
  AUROC is 0.9444 / 0.9424 / 0.9410 against stage X's ~0.940. Separating the
  BCE's parameters from the readout's -- finding 75's diagnosis and half the
  reason this stage exists -- did not change WHAT the determinacy term
  learns. Whatever it changes, it is not the quality of the representation.

  AND IT COSTS THREE TIMES WHAT STAGE X COST. Against the stage-S control at
  the SAME epoch (0.4832 / 0.4829 / 0.4822), stage Y's val token accuracy
  runs -0.0225 / -0.0011 / -0.0267, mean -0.0168, against stage X's -0.0060.
  The gap this whole line of work exists to close is +0.0011 (stage V's NEED,
  derived not typed), so the cost is about FIFTEEN TIMES the target, the
  wrong way.

  THE DAMAGE IS A BASIN LOTTERY, NOT A UNIFORM TAX, AND THAT IS NEW. Seeds 0
  and 2 plateau by epoch 6 at 0.4570 / 0.4508 and gain +0.004 over the next
  26 epochs; seed 1 sits at the control's level, 0.4818 against 0.4829. The
  seed spread is 0.0263 against the control's 0.0010 and stage X's 0.0022 --
  more than TWENTY TIMES either. The control's seed 2 escaped a plateau of
  this depth between epochs 10 and 32 (0.4583 -> 0.4822); stage Y's two have
  not moved, so this is read as a basin they are stuck in and not a delay.
  Finding 67 said pinning the kernels does not remove the seed lottery; this
  says the channel AMPLIFIES it.

THE PREDICTION, per seed because the seeds are bimodal and a mean would hide
it, pushed through the same S4 accuracy-to-solve slope stage V used and stage
X's prediction used (ACC_GAP of accuracy per RHO_GAP of rho_50, imported and
never retyped):

    Y4 IS EXPECTED TO FAIL on seeds 0 and 2 and pass on seed 1, so score()
    will return "Y4 failed, Y1-Y3 not interpreted". Stage X's epoch-32 to
    epoch-49 climb was +0.0010, so the final accuracies are expected near
    0.462 / 0.483 / 0.457 against a bar of 0.4800.

    Y1 IS EXPECTED TO FAIL, AND TO FAIL OFF THE END OF THE GRID. The per-seed
    deficits predict rho_50 near 1.12 / 0.78 / 1.20 against the control's
    0.754 / 0.757 / 0.765. Two of those are ABOVE GRID_TOP = 0.95, i.e. the
    prediction is that seeds 0 and 2 never reach a 0.5 solve rate anywhere on
    the 14-point grid and rho_50 comes back CENSORED with flag '>top'.

That last line has an operational consequence, which is why it is written
down before the fact: `stagey_guard_2026-11-14.sbatch` asserted flag == 'ok'
and would have died on the assertion before writing stagey_verdict.json. It
now reports the flag and maps '>top' to +inf, which is a Y1 failure by every
route and cannot flatter the hypothesis -- a model that never crosses 0.5 is
worse than any value the grid can return. score() is untouched.

WHY THE RUN IS NOT CANCELLED, and the reasoning is stage X's. Y2 and Y3 are
the mechanism questions and they get SHARPER, not weaker, when the accuracy
moves the wrong way: the probe still measures the guard rate and its shape on
every cell and prints both. Y4's veto means the registered scorer declines to
INTERPRET them, which is the registration working as written -- but the
numbers will exist, and "the channel reached stage X's AUROC, cost 3x stage
X's accuracy, and did or did not produce a conditional guard" is the sentence
stage Z has to start from either way. Cancelling would delete it.

AND NO LAMBDA SWEEP, NO WARMUP ON THE CHANNEL, NO BEST-VAL SELECTION IS
STARTED ON THE STRENGTH OF THIS. Two seeds landing in a bad basin is exactly
the observation that makes a warmup schedule or a smaller lambda tempting;
both are new knobs, the registration fixed lambda a priori, and best-val
selection is a knob this project already closed. If the basin finding
survives the finished cells it is a NEW stage with its own registration.
"""
import os
import sys

CONTROL_ACC = (0.4830, 0.4830, 0.4833)
CONTROL_RHO50 = (0.754, 0.757, 0.765)
CONTROL_GUARD = (1.0000, 1.0000, 1.0000)
UNTIED_SHAPE = {"node_token": 0.966, "still_forced": 0.903}

ACC_BAR = 0.4800
RHO_BAR = min(CONTROL_RHO50)
GUARD_BAR = 0.90
SHAPE_NODE_BAR = 0.90
SHAPE_FORCED_BAR = 0.60
MIN_SEEDS = 2
LAMBDA = 1.0
FREE_LINE = 0.740
NEW_PARAMS = 320


# ---- The mid-run prediction of 2026-11-14 19:30, read off both arms' logs at
# epoch 32 of 50, before any cell finished. NOT A BAR. Nothing above this line
# was touched when these were added; see the MID-RUN PREDICTION section of the
# docstring. score() never reads any of it.
Y_ACC_AT_32 = (0.4607, 0.4818, 0.4555)        # stage Y, epoch 32, seeds 0/1/2
S_ACC_AT_32 = (0.4832, 0.4829, 0.4822)        # stage S, the SAME epoch
X_ACC_AT_32 = (0.4770, 0.4768, 0.4748)        # stage X, the same epoch again
Y_BCE_AT_32 = (0.2773, 0.2858, 0.2937)        # the channel's BCE
X_BCE_AT_32 = (0.2787, 0.2869, 0.2882)        # stage X's, on the readout
Y_AUROC_AT_32 = (0.9444, 0.9424, 0.9410)
GRID_TOP = 0.95        # the easiest density on the 14-point grid
LATE_CLIMB = 0.0010    # stage X's measured epoch 32 -> 49 gain in accuracy


def predict_rho50(y_acc=Y_ACC_AT_32, s_acc=S_ACC_AT_32):
    """What the epoch-32 accuracy deficit implies for rho_50, PER SEED,
    pushed through the same accuracy-to-solve slope stage V used to set its
    own bar and stage X's prediction used before this one. Imported from
    scripts.stagev_score, never retyped, and imported lazily so a job that
    wants only this file's constants does not pay for stage V's import chain.

    Per seed rather than on the mean because stage Y's seeds are bimodal: one
    at the control and two in a lower basin. A mean of those predicts a
    number no seed is near, and would hide the only new thing in the run.

    A negative deficit predicts a HIGHER rho_50, which is worse. A prediction
    above GRID_TOP means the grid cannot measure it -- the model is expected
    never to reach a 0.5 solve rate at any density, so rho_50 comes back
    censored with flag '>top' rather than as a number.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    import scripts.stagev_score as sv
    slope = sv.RHO_GAP / sv.ACC_GAP                 # rho_50 per unit accuracy
    deficits = [y - s for y, s in zip(y_acc, s_acc)]
    predicted = [c - d * slope for c, d in zip(CONTROL_RHO50, deficits)]
    return {
        "acc_deficit_per_seed": deficits,
        "acc_deficit_mean": sum(deficits) / len(deficits),
        "slope_rho50_per_acc": slope,
        "need_to_close_the_gap": sv.NEED,
        "deficit_in_units_of_need": (sum(deficits) / len(deficits)) / sv.NEED,
        "predicted_rho50_per_seed": predicted,
        "predicted_final_acc": [a + LATE_CLIMB for a in y_acc],
        "seeds_expected_to_pass_y4": sum(a + LATE_CLIMB >= ACC_BAR
                                         for a in y_acc),
        "seeds_expected_censored_above_grid": sum(p > GRID_TOP
                                                  for p in predicted),
    }


def check_y4(acc):
    return {"bar": "Y4 sanity", "pass": all(a >= ACC_BAR for a in acc),
            "values": list(acc), "threshold": ACC_BAR,
            "control": list(CONTROL_ACC)}


def check_y1(rho):
    return {"bar": "Y1 rho_50", "pass": max(rho) < RHO_BAR, "values": list(rho),
            "threshold": RHO_BAR, "control": list(CONTROL_RHO50),
            "clears_free_line_separately": max(rho) < FREE_LINE}


def check_y2(guard):
    n = sum(g < GUARD_BAR for g in guard)
    return {"bar": "Y2 guard", "pass": n >= MIN_SEEDS, "values": list(guard),
            "threshold": GUARD_BAR, "seeds_passing": n,
            "control": list(CONTROL_GUARD)}


def check_y3(shapes):
    """shapes: one dict per seed with node_token and still_forced, or None
    where the guard never fired. A seed that never fires cannot pass."""
    n = sum(bool(s) and s["node_token"] >= SHAPE_NODE_BAR
            and s["still_forced"] >= SHAPE_FORCED_BAR for s in shapes)
    return {"bar": "Y3 shape", "pass": n >= MIN_SEEDS, "values": list(shapes),
            "thresholds": {"node_token": SHAPE_NODE_BAR,
                           "still_forced": SHAPE_FORCED_BAR},
            "seeds_passing": n, "untied_reference": UNTIED_SHAPE}


def score(acc, rho, guard, shapes):
    out = {"Y4": check_y4(acc), "Y1": check_y1(rho), "Y2": check_y2(guard),
           "Y3": check_y3(shapes)}
    out["verdict"] = (
        "Y4 failed, Y1-Y3 not interpreted" if not out["Y4"]["pass"]
        else "design refuted: guard is stop-shaped again"
        if out["Y2"]["pass"] and not out["Y3"]["pass"]
        else "supported" if all(out[k]["pass"] for k in ("Y1", "Y2", "Y3"))
        else "refuted")
    return out


def _self_test():
    good = score((0.4841, 0.4852, 0.4833), (0.735, 0.741, 0.748),
                 (0.42, 0.51, 0.88),
                 [{"node_token": 0.95, "still_forced": 0.72}] * 3)
    assert good["verdict"] == "supported", good
    assert good["Y1"]["clears_free_line_separately"] is False, good["Y1"]

    # stage X's own numbers, scored against stage Y's bars: Y4 fails.
    x = score((0.4780, 0.4767, 0.4778), (0.86, 0.85, 0.87),
              (1.0, 1.0, 1.0), [None] * 3)
    assert x["verdict"] == "Y4 failed, Y1-Y3 not interpreted", x

    # a stop-shaped guard that moves rho_50: refuted on purpose by Y3.
    stop = score((0.4841, 0.4852, 0.4833), (0.735, 0.741, 0.748),
                 (0.3, 0.3, 0.3),
                 [{"node_token": 0.05, "still_forced": 0.01}] * 3)
    assert stop["verdict"] == "design refuted: guard is stop-shaped again", stop

    # A censored rho_50 is +inf by the time it reaches score(): the probe maps
    # '>top' there because a model that never crosses 0.5 is worse than any
    # value the grid can return. Asserted so the mapping cannot be quietly
    # inverted later into something that flatters the arm.
    cens = score((0.4841, 0.4852, 0.4833), (float("inf"), 0.78, float("inf")),
                 (0.42, 0.51, 0.88),
                 [{"node_token": 0.95, "still_forced": 0.72}] * 3)
    assert cens["Y1"]["pass"] is False, cens["Y1"]
    assert cens["Y1"]["clears_free_line_separately"] is False, cens["Y1"]
    assert cens["verdict"] == "refuted", cens

    # The mid-run prediction. Pinned so that if stage V's slope is ever edited
    # the prediction recorded on 2026-11-14 stops matching and somebody has to
    # say which of the two numbers changed. Four claims -- a deficit many times
    # the gap, two seeds below the Y4 bar, two rho_50 predictions off the top
    # of the grid, and a channel whose BCE matches stage X's readout -- and
    # none of them is a bar.
    p = predict_rho50()
    assert -0.0180 < p["acc_deficit_mean"] < -0.0150, p
    assert p["deficit_in_units_of_need"] < -13.0, p
    assert p["seeds_expected_to_pass_y4"] == 1, p
    assert p["seeds_expected_censored_above_grid"] == 2, p
    assert all(y > CONTROL_RHO50[i]
               for i, y in enumerate(p["predicted_rho50_per_seed"])), p
    assert max(abs(y - x) for y, x in zip(Y_BCE_AT_32, X_BCE_AT_32)) < 0.006, (
        Y_BCE_AT_32, X_BCE_AT_32)
    print("ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
