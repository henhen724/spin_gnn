"""STAGE X REGISTRATION. This docstring is the registration: it is committed
before the first cell lands, and the bars below are not moved afterwards.

================================================================================
THE QUESTION
================================================================================

Stage S measured that one shared block learns direct deduction 0.011 of token
accuracy less well than an untied stack: S1 (`tied_no_mlp` T=16 d320 ff1024
h4, 613,818 trainable) reaches rho_50 0.757 (0.754 / 0.757 / 0.765) against
the free line of 0.740, missing the gate by 0.017. Stages T, U, V and W then
closed every hyperparameter knob that could have explained it -- iteration
count, role count, schedule length, training budget -- and every residual sat
inside the 0.0423 LOO bar.

Findings 68-72 named the gap instead of closing it:

  68  S1 applies DD's elimination rule flawlessly (rule-flip 1.0000) and does
      not run DD's SCHEDULER. Destroy a node's determinacy with one token and
      S1 still names it: guard rate 1.0000 / 1.0000 / 1.0000, spread 0.0000,
      against the untied arm's 0.0000 / 0.0000 / 0.8444.
  69  The determinacy is nevertheless COMPUTED. Under the same broken edit
      S1's p_freed lands at 0.5253 / 0.5228 / 0.5354 -- correctly uncertain --
      while its node readout does not move to four decimals. Present and
      unused.
  72  In the untied arm the guard occupies a window of 2-3 applications near
      t = 3-6; it is a localised piece of computation, not a property of the
      whole stack.

Stage X asks the one training question those findings leave open, and it is
not a knob any earlier stage touched:

    IF THE READOUT IS TRAINED TO SAY WHICH NODES ARE DEDUCIBLE, DOES THE
    GUARD APPEAR, AND DOES ANY OF THE 0.017 CLOSE?

================================================================================
WHY THE MAIN LOSS CANNOT ALREADY HAVE TAUGHT THIS -- the gauge argument
================================================================================

Cross-entropy sees the readout's logit vector only through a softmax, which
is invariant under adding a constant to every logit. So the main loss
constrains the logits only up to that constant. It can say "node v outranks
the others here"; it cannot say "node v is not emittable at all", because
emittability is a statement about absolute height, and absolute height is
exactly the direction cross-entropy integrates out.

A guard is a statement about absolute height. Nothing in stages S-W ever
trained it. That is the hypothesis for why the determinacy is computed
(finding 69) and unused (finding 68): not because the model cannot represent
it, but because no gradient ever asked the readout to commit to it.

================================================================================
THE INTERVENTION -- one knob, zero parameters
================================================================================

A stage-X cell is S1 exactly: same architecture, same 613,818 trainable
parameters, same state dict, same data, same schedule, same seeds. One term
is added to the loss:

    L = CE_next_token  +  lambda * BCE_determinacy

where BCE_determinacy is a multi-label binary cross-entropy on the readout's
NODE logits, evaluated at every position where the model is about to emit a
node, over the nodes still uncoloured there, against DD's own deducibility
predicate (spin_gnn.sudoku_determinacy; pinned against
sudoku_activations.dd_state_labels by tests/test_sudoku_determinacy.py).

Already-coloured nodes are masked out rather than labelled negative: they are
trivially undeducible and at N=48 they are most of every row. What is scored
is exactly DD's scheduler question.

THE TARGET STATISTICS DEPEND ON THE DENSITY, AND THE TRAINING MIX IS THE ONE
THAT MATTERS. An earlier 200-instance measurement at a single high hint
density gave scored_fraction 0.156 and positive_rate 0.654; that is NOT what
these cells train on. Measured on the actual training distribution (the
sweep's rho range at N=48 d=5 q=4 fall shuffle_adj, 2000 instances):

    scored_fraction 0.2486     positive_rate 0.3543
    157,470 positive / 286,992 negative / 1,343,058 masked

Both readings are right for their density -- at high rho few nodes are open
and most of those are determined, at low rho many are open and few are -- and
the training mix is the better-conditioned of the two for a BCE: a quarter of
all cells carry signal and the classes sit nearer 1:2 than 2:1.

LAMBDA IS 1.0 AND IS NOT SWEPT. Both terms are means of per-item losses; at
N=48 the determinacy term starts at 0.733 against a main term of 3.95, so it
enters at roughly 0.19x the main loss. One value, chosen a priori, no tuning
run. A sweep over lambda would be a second knob and is not part of this stage.

PINNED KERNELS. Cells run with --deterministic, which stage S did not have.
Finding 67 measured that pinning removes run-to-run noise without shifting the
distribution (the pinned pair landed at 2.4249, inside the unpinned spread
2.3656-2.5950), and with 3 seeds the variance that matters is the seed-to-seed
basin lottery, which pinning does not touch. So this is reproducibility, not a
second knob -- but it IS a difference from the stage-S control and a reader is
entitled to discount it.

================================================================================
HYPOTHESES AND BARS -- fixed here, before any cell runs
================================================================================

The stage-S control is S1's three measured cells: rho_50 0.754 / 0.757 /
0.765, final val token accuracy 0.4830 (spread 0.0010), guard rate 1.0000 /
1.0000 / 1.0000 (conditional on clean_emits_v, finding 68's denominator).

X4  NO-HARM GATE, read first. Every stage-X seed's final val token accuracy
    must be at least S1_VAL_ACC - 0.05. If this fails the determinacy term
    broke training and X1-X3 are uninterpretable; the stage reports that and
    stops. Checked FIRST, before any other number is looked at.

X1  PRIMARY. Does rho_50 move? CONFIRMED iff every stage-X seed's rho_50 is
    strictly below every stage-S seed's rho_50 -- complete separation of two
    groups of 3, which is the one-sided rank test at exactly p = 1/20 = 0.05.
    Additionally and separately reported: whether all 3 seeds clear the free
    line at rho_50 <= 0.740, which is the standing gate and the only thing
    that would make a tied cell a standing result.

X2  MECHANISM. Does the guard appear? Re-run finding 68's guard probe,
    unchanged, on the stage-X checkpoints over its byte-identical 350 round-0
    cases. CONFIRMED iff the conditional guard rate is below 0.90 in at least
    2 of the 3 seeds. S1 measured 1.0000 in all three with spread 0.0000, so
    0.90 is a drop of ten times the control's entire spread.

X3  THE TERM DID WHAT IT SAYS. CONFIRMED iff every stage-X seed's determinacy
    AUROC exceeds every stage-S seed's. BOTH groups are measured by
    scripts/determinacy_eval.py on ONE fixed held-out instance set -- not from
    history.json, whose per-epoch column is computed on each cell's own
    validation draw (seed + n_train, so the three seeds see three different
    sets) and does not exist at all for a stage-S cell trained before the term.
    An AUROC above chance is NOT the bar: the main loss alone already lifts the
    next node's logit and the next node is always deducible, so the stage-S
    cells are expected to score above 0.5 and they are the control.

OUTCOMES ARE NOT RANKED BY PREFERENCE. X1 confirmed with X2 confirmed says the
missing guard was part of the gap. X3 confirmed with X2 refuted is the
sharpest possible NEGATIVE result and the one finding 69 predicts: the
determinacy can be made linearly explicit in the very logits that choose the
emission, and the emission still ignores it -- which would say the schedule
must be structurally coupled to the determinacy (a logit mask), not merely
trained alongside it. X3 refuted means the term did not take and nothing else
is interpretable.

================================================================================
MID-RUN PREDICTION, written 2026-11-14 04:30 at epoch 30 of 50, BEFORE any
cell finished and before rho_50 or the guard rate existed for any of them.
THE BARS ABOVE ARE NOT TOUCHED. This is a prediction, recorded so X1 is a
test of it rather than a story told afterwards.
================================================================================

At epoch 30 all three cells are healthy and none has crashed, but two things
are already clear from the training logs:

  THE TERM TOOK, AND IT TOOK IMMEDIATELY. val determinacy AUROC is 0.9303 /
  0.9312 / 0.9351 at epoch 5 and 0.9403 / 0.9411 / 0.9402 at epoch 30 -- it
  saturates almost at once and then barely moves, against finding 73's
  untrained control of 0.7610 / 0.7635 / 0.7774. X3 is expected to CONFIRM
  decisively. That the information needed only a few epochs to become
  linearly explicit is itself finding 69's thesis at the logit level: it was
  nearly there already.

  AND IT COSTS MAIN-TASK ACCURACY, PERSISTENTLY. Against the stage-S control
  at the SAME epoch, val token accuracy runs -0.0059 / -0.0055 / -0.0065 at
  epoch 30 and -0.0055 / -0.0074 / +0.0125 at epoch 10; the deficit is flat
  rather than closing, and the one positive is seed 2's control still climbing
  out of the plateau (its train NLL was 1.6434 at epoch 20 against stage X's
  1.6482, and it passed afterwards). Train NLL -- the MAIN term alone, the
  determinacy term is excluded from that column by construction -- runs
  +0.0327 / +0.0311 / +0.0477.

THE PREDICTION. The accuracy deficit is about -0.0060, and the gap this stage
exists to close is +0.0011 (stage V's NEED, derived not typed). So the term is
costing roughly FIVE TIMES the target, in the wrong direction. Pushed through
the same S4 accuracy-to-solve slope stage V used -- ACC_GAP 0.0110 of accuracy
per RHO_GAP 0.177 of rho_50, about 16.1 rho_50 per unit accuracy -- a -0.0060
accuracy deficit predicts rho_50 near 0.853 against S1's 0.757.

    X1 IS THEREFORE EXPECTED TO BE REFUTED, AND REFUTED BADLY RATHER THAN
    NARROWLY: not a near miss at 0.75, but a move the WRONG WAY to roughly
    0.85. predict_rho50() below computes it from the imported constants.

X4 still passes comfortably -- 0.4775 against a floor of 0.4330 -- which is
worth noting as a flaw in X4 rather than a reassurance: the no-harm gate was
set wide enough to catch "the term broke training" and is far too wide to
catch "the term costs five times the effect we are chasing". A future stage
should gate on the control's accuracy at the same epoch, not on a fixed 0.05.

WHY THE RUN IS NOT CANCELLED. X2 is the mechanism question and it does not
depend on X1 landing well -- it gets SHARPER, not weaker, if the accuracy
moves the wrong way. With the determinacy at AUROC 0.94 in the very logits
whose argmax picks the next node, "does the model now withhold a node whose
determinacy was destroyed" is the cleanest form of that question this project
has been able to ask. A guard rate still at 1.0000 under those conditions
would be a strong negative, and a strong negative here is the registered
most-informative outcome.

AND NO LAMBDA SWEEP IS STARTED ON THE STRENGTH OF THIS. The registration fixed
lambda = 1.0 a priori and said a sweep would be a second knob; seeing a
mid-run number is not a licence to go back on that. If a smaller lambda is
worth running it is a NEW stage with its own registration, and the argument
for it has to survive X2 first -- if the guard does not appear at AUROC 0.94,
a gentler version of the same term is not the follow-up, structural coupling
is.

================================================================================
WHAT THIS STAGE CANNOT SHOW
================================================================================

- It does not reopen any closed knob. Width, cosine decay, data volume,
  best-val selection, role count, schedule length and training budget stay
  closed; none of them is varied here.
- A stage-X cell is still 4.08x smaller than J2, so a comparison against the
  untied arm remains tying AND budget, exactly as in findings 68-72.
- The determinacy target is built from the ground-truth trace, so stage X
  trains on supervision direct deduction itself provides. It says nothing
  about a model that must discover the predicate unaided.
"""
import json
import os
import sys

# ---- The stage-S control, from stage S's own cells. Never retyped from a deck.
S1_RHO50 = (0.754, 0.757, 0.765)
S1_VAL_ACC = 0.4830
S1_GUARD = (1.0000, 1.0000, 1.0000)

# ---- Bars. Fixed by this file, at the commit that registers it.
FREE_LINE = 0.740          # the standing gate: rho_50 at or below it
HARM_DROP = 0.05           # X4: val accuracy may fall by at most this
GUARD_BAR = 0.90           # X2: a guard rate below this counts as a guard
GUARD_MIN_SEEDS = 2        # X2: in at least this many of 3 seeds
LAMBDA = 1.0               # registered, not swept
BASIN = 1.561              # train NLL line between stage U's two populations
PLATEAU = 1.64             # finding 44, an N=48 number. Neither is a stage-X
                           # bar; both are carried so the launchers read the
                           # project's lines from one place instead of retyping.


# ---- The mid-run prediction of 2026-11-14 04:30, read off both arms'
# history.json at epoch 30 of 50, before any cell finished. NOT A BAR. Nothing
# above this line was touched when these were added; see the MID-RUN
# PREDICTION section of the docstring.
X_ACC_AT_30 = (0.4775, 0.4784, 0.4755)        # stage X, epoch 30, seeds 0/1/2
S_ACC_AT_30 = (0.4834, 0.4839, 0.4820)        # stage S, the SAME epoch
X_AUROC_AT_30 = (0.9403, 0.9411, 0.9402)      # per-cell val draw, from history
S_AUROC_UNTRAINED = (0.7610, 0.7635, 0.7774)  # finding 73, one fixed set


def predict_rho50(x_acc=X_ACC_AT_30, s_acc=S_ACC_AT_30):
    """What the epoch-30 accuracy deficit implies for rho_50, pushed through
    the SAME accuracy-to-solve slope stage V used to set its own bar: ACC_GAP
    of token accuracy goes with RHO_GAP of rho_50, the J2/S1 pair. Imported
    from scripts.stagev_score, never retyped -- and imported lazily, so a job
    that only wants this file's constants does not pay for stage V's import
    chain. A negative deficit predicts a HIGHER rho_50, which is worse.
    """
    # Run as `python3 scripts/stagex_score.py` the repo root is NOT on the
    # path, only scripts/ is; the sibling import has to be made to work from
    # either entry point or the prediction is unreachable half the time.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    import scripts.stagev_score as sv
    slope = sv.RHO_GAP / sv.ACC_GAP                 # rho_50 per unit accuracy
    mean = lambda t: sum(t) / len(t)                # noqa: E731
    deficit = mean(x_acc) - mean(s_acc)
    return {
        "acc_deficit": deficit,
        "slope_rho50_per_acc": slope,
        "need_to_close_the_gap": sv.NEED,
        "deficit_in_units_of_need": deficit / sv.NEED,
        "s1_rho50_mean": mean(S1_RHO50),
        "predicted_rho50": mean(S1_RHO50) - deficit * slope,
    }


def _num(v):
    """Every CSV cell is a string and booleans arrive as 'True' (finding 69's
    durable lesson). None and '' become nan rather than raising."""
    if v is None or v == "":
        return float("nan")
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, str) and v.strip() in ("True", "False"):
        return float(v.strip() == "True")
    return float(v)


def check_harm(x_val_acc):
    """X4, the gate that is read first."""
    floor = S1_VAL_ACC - HARM_DROP
    worst = min(_num(a) for a in x_val_acc)
    return {"bar": floor, "worst_seed": worst, "pass": bool(worst >= floor)}


def check_rho(x_rho50):
    """X1. Complete separation of 3 against 3 is p = 1/20 one-sided."""
    x = [_num(r) for r in x_rho50]
    separated = max(x) < min(S1_RHO50)
    return {
        "stage_x": sorted(x), "stage_s": sorted(S1_RHO50),
        "separated": bool(separated),
        "mean_shift": sum(x) / len(x) - sum(S1_RHO50) / len(S1_RHO50),
        "clears_free_line": bool(max(x) <= FREE_LINE),
        "pass": bool(separated),
    }


def check_guard(x_guard):
    """X2. Lower is better: the rate is how often the model still names a node
    whose determinacy was destroyed, conditional on it naming that node on the
    clean prompt."""
    g = [_num(v) for v in x_guard]
    n_below = sum(1 for v in g if v < GUARD_BAR)
    return {
        "stage_x": g, "stage_s": list(S1_GUARD), "bar": GUARD_BAR,
        "seeds_below_bar": n_below, "need": GUARD_MIN_SEEDS,
        "pass": bool(n_below >= GUARD_MIN_SEEDS),
    }


def check_auroc(x_auroc, s_auroc):
    """X3. Both groups measured with the same function on the same instances."""
    x = [_num(v) for v in x_auroc]
    s = [_num(v) for v in s_auroc]
    return {
        "stage_x": sorted(x), "stage_s": sorted(s),
        "separated": bool(min(x) > max(s)),
        "pass": bool(min(x) > max(s)),
    }


def verdict(harm, rho, guard, au):
    """One word, decided by the bars above and nothing else."""
    if not harm["pass"]:
        return "broke_training"
    if not au["pass"]:
        return "term_did_not_take"
    if rho["pass"] and guard["pass"]:
        return "guard_learned_and_gap_moved"
    if guard["pass"]:
        return "guard_learned_gap_unmoved"
    if rho["pass"]:
        return "gap_moved_without_guard"
    return "representation_without_use"


def score(x_val_acc, x_rho50, x_guard, x_auroc, s_auroc):
    harm = check_harm(x_val_acc)
    rho = check_rho(x_rho50)
    guard = check_guard(x_guard)
    au = check_auroc(x_auroc, s_auroc)
    return {"X4_no_harm": harm, "X1_rho50": rho, "X2_guard": guard,
            "X3_auroc": au, "verdict": verdict(harm, rho, guard, au),
            "lambda": LAMBDA}


def _self_test():
    """Run by the registration job, so the bar logic is executed and committed
    before any cell exists rather than written and left unexercised."""
    broke = score([0.40, 0.42, 0.41], [0.80] * 3, [1.0] * 3, [0.9] * 3, [0.6] * 3)
    assert broke["verdict"] == "broke_training", broke["verdict"]
    assert not broke["X4_no_harm"]["pass"]

    flat = score([0.483] * 3, [0.80] * 3, [1.0] * 3, [0.55] * 3, [0.60] * 3)
    assert flat["verdict"] == "term_did_not_take", flat["verdict"]

    null = score([0.483] * 3, [0.757] * 3, [1.0] * 3, [0.90] * 3, [0.60] * 3)
    assert null["verdict"] == "representation_without_use", null["verdict"]

    guard_only = score([0.483] * 3, [0.760] * 3, [0.10, 0.20, 0.95], [0.9] * 3, [0.6] * 3)
    assert guard_only["verdict"] == "guard_learned_gap_unmoved", guard_only["verdict"]
    assert guard_only["X2_guard"]["seeds_below_bar"] == 2

    win = score([0.490] * 3, [0.700, 0.710, 0.720], [0.05, 0.10, 0.15], [0.9] * 3, [0.6] * 3)
    assert win["verdict"] == "guard_learned_and_gap_moved", win["verdict"]
    assert win["X1_rho50"]["separated"] and win["X1_rho50"]["clears_free_line"]

    # Separation is strict: one stage-X seed tying the best stage-S seed fails.
    tie = score([0.483] * 3, [0.700, 0.710, 0.754], [0.05] * 3, [0.9] * 3, [0.6] * 3)
    assert not tie["X1_rho50"]["separated"]

    # X1 can pass without clearing the standing gate, and says so separately.
    short = score([0.483] * 3, [0.745, 0.748, 0.750], [1.0] * 3, [0.9] * 3, [0.6] * 3)
    assert short["X1_rho50"]["separated"] and not short["X1_rho50"]["clears_free_line"]
    assert short["verdict"] == "gap_moved_without_guard", short["verdict"]

    assert _num("True") == 1.0 and _num("") != _num("")

    # The mid-run prediction. Pinned so that if stage V's slope is ever edited
    # the prediction recorded on 2026-11-14 stops matching and somebody has to
    # say which of the two numbers changed. A deficit costing several times
    # the gap, and a predicted rho_50 the WRONG side of the control, are the
    # two claims; neither is a bar and score() never reads them.
    p = predict_rho50()
    assert -0.0070 < p["acc_deficit"] < -0.0050, p
    assert p["deficit_in_units_of_need"] < -4.0, p
    assert p["predicted_rho50"] > max(S1_RHO50), p
    assert min(X_AUROC_AT_30) > max(S_AUROC_UNTRAINED), (X_AUROC_AT_30,)

    print("stagex_score self-test: OK")
    print(f"registered bars: free_line={FREE_LINE} harm_drop={HARM_DROP} "
          f"guard_bar={GUARD_BAR} need={GUARD_MIN_SEEDS}/3 lambda={LAMBDA}")
    print(f"control: S1 rho_50={S1_RHO50} val_acc={S1_VAL_ACC} guard={S1_GUARD}")
    print("mid-run prediction (2026-11-14, epoch 30, NOT a bar): "
          + json.dumps({k: round(v, 4) for k, v in predict_rho50().items()},
                       sort_keys=True))


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
        raise SystemExit(0)
    path = sys.argv[1]
    with open(path) as f:
        summary = json.load(f)
    out = score(summary["x_val_acc"], summary["x_rho50"], summary["x_guard"],
                summary["x_auroc"], summary["s_auroc"])
    print(json.dumps(out, indent=2))
    with open(os.path.join(os.path.dirname(path) or ".", "stagex_scored.json"), "w") as f:
        json.dump(out, f, indent=2)
