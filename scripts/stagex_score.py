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
trivially undeducible and at N=48 they are about five sixths of every row.
What is scored is exactly DD's scheduler question. Measured on 200 instances
at N=48 d=5 q=4 fall shuffle_adj: 15.6% of cells are scored, and 65.4% of
those are positive -- 7.5 open nodes at the median position, 4.9 deducible.

LAMBDA IS 1.0 AND IS NOT SWEPT. Both terms are means of per-item losses; at
initialization the determinacy term is ~0.65 against a main term of ~3.4, so
it enters at roughly 0.2x the main loss. One value, chosen a priori, no tuning
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

X3  THE TERM DID WHAT IT SAYS. CONFIRMED iff every stage-X seed's final val
    determinacy AUROC exceeds every stage-S seed's, the stage-S cells being
    evaluated with the same function on the same val instances. An AUROC above
    chance is NOT the bar: the main loss alone already lifts the next node's
    logit and the next node is always deducible, so the stage-S cells are
    expected to score above 0.5 and they are the control.

OUTCOMES ARE NOT RANKED BY PREFERENCE. X1 confirmed with X2 confirmed says the
missing guard was part of the gap. X3 confirmed with X2 refuted is the
sharpest possible NEGATIVE result and the one finding 69 predicts: the
determinacy can be made linearly explicit in the very logits that choose the
emission, and the emission still ignores it -- which would say the schedule
must be structurally coupled to the determinacy (a logit mask), not merely
trained alongside it. X3 refuted means the term did not take and nothing else
is interpretable.

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
    print("stagex_score self-test: OK")
    print(f"registered bars: free_line={FREE_LINE} harm_drop={HARM_DROP} "
          f"guard_bar={GUARD_BAR} need={GUARD_MIN_SEEDS}/3 lambda={LAMBDA}")
    print(f"control: S1 rho_50={S1_RHO50} val_acc={S1_VAL_ACC} guard={S1_GUARD}")


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
