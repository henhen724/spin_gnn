"""Can a body have the predicate AND the guard without the churn?

THE QUESTION, AND WHY IT NEEDS A NEW ARM. Findings 82-86 leave stage PAIR in a
sharp place. The satisfied margin buys a real predicate on every body (matched
pair AUROC 0.8710 / 0.8620 / 0.8710, placebo 0.4360 / 0.4450 / 0.4420, and the
predicate is in place by step 75 / 25 / 25 of 3,000), but the guard it buys comes
with churn -- one-token edits flip whether v is emitted in 142 / 175 / 177 of
1,000 cases against stage Y's 4 -- and the two rank together at Spearman +0.879 /
+0.806 / +0.881. Every trajectory snapshot is either quiet and guardless or loud
and guarded. Nothing in 84-86 can say whether that is a law of the cell or a
consequence of WHERE the predicate is allowed to act, because in stage Y and
stage PAIR alike the gate is read off the FINAL hidden state and written to the
emission. g can change what the network says; it cannot change what the network
computes. Finding 78 called it a predicate that is not causally wired; 68-74 put
the same thing as the shared block computing the ambiguity and then scheduling as
if certain. A finer grid cannot test it. A new arm can.

THE ARM, AND IT IS ONE KNOB. `determinacy_feedback=True` recomputes the gate
after every application of the shared block and folds it back into the residual
stream, so iteration t+1 reads iteration t's determinacy estimate:

    g_t   = <h_t * s, E_v>                        exactly stage Y's channel
    a_t   = log sigma(g_t) + log 2                zero when g_t = 0
    h_t  <- h_t + c * (a_t @ E_nodes) / n_nodes   t = 0 .. T-2

Not after the last application, so the readout and the channel still see a hidden
state reached the way stage Y's was and the P1-P4 read-out measures the same
object. The recipe, the data stream, the body warmed from, the 1,000 eval cases,
every bar: unchanged from `stagepair_pairprobe.py`. The only difference between
`loop L1` here and stage PAIR's published `warm L1` is the flag.

THE PRICE IS ONE SCALAR. c is the single new parameter -- 614,139 against
614,138, +0.00016% -- and it is initialized at ZERO, so the arm starts at exactly
the stage Y body rather than near it (`scripts/_loop_dry_check.py` asserts the
logits are bit-identical at c = 0 with a non-zero `det_scale`; a dropped `log 2`
would show up there and nowhere else). c = 0 is not a saddle, dL/dc is O(1), so
|c| at the end is itself a measurement: it says whether the objective WANTED the
loop. Re-using `token_embed`'s node rows in both directions is what keeps the
count at +1, and `_fb_rel` reports mean|delta|/mean|h| because c alone says
nothing without the scale of a (findings 76 and 79: a magnitude checked at a
degenerate point is not checked).

FOUR ROWS, AND TWO OF THEM COST NOTHING. Measured on the same case list at the
same eval stream in the same job, which is the only way the columns are
comparable at all:

    Y         that seed's stage Y body, reloaded          eval only
    pair L1   stage PAIR's published warm L1, reloaded    eval only
    loop L0   3,000 steps, loop ON, lambda=0              trained here
    loop L1   3,000 steps, loop ON, lambda=1              trained here

`pair L1` is loaded from `runs/stagepair_pairprobe_2026-11-15*/warm_L1/` and its
published row is READ from that directory's `pair_summary.json`, never retyped --
so the control is the same weights findings 82-83 scored, re-measured beside the
new arm rather than compared across jobs. `loop L0` is the control the pairprobe's
own design argument demands: without it, any movement in `loop L1` could be the
loop's mere presence rather than the margin's use of it.

VALIDATION TOKEN ACCURACY IS PRINTED FIRST AND IT NEVER CENSORS. A feedback loop
through sixteen iterations can wreck a solved body, and if it does, nothing
downstream is readable as a statement about guards. It is measured on a held-out
stream (`VAL_SEED_START`, 400 instances) that no arm trained on, for all four
rows, and reported before any AUROC.

THE REGISTERED PREDICTIONS, WRITTEN AND COMMITTED BEFORE THE FIRST RUN EXISTS.
The reading behind them is that the churn is the cost of a guard built out of
emission-level re-ranking: with no way to act on its own determinacy estimate
during the computation, the only lever the margin has on the output is the node
block's internal order at the last step, and leaning on that lever moves v's
emission on edits that have nothing to do with v. Given a lever inside the
iteration, the body should be able to suppress v specifically.

  * THE LOOP OPENS. |c| ends in [0.05, 2.0] on all three seeds. If it ends
    below 0.05 the objective did not want the feedback, the arm is stage PAIR
    with a dead parameter, and the right report is that -- not a guard result.
  * THE PREDICATE SURVIVES. `loop L1` clears P1 (corrupt >= 0.65) on 3/3 and
    stays inside P2's (0.40, 0.60) placebo band on 3/3. The loop is a change to
    the architecture, not to the margin, so the thing the margin already buys
    should still arrive.
  * THE LOAD-BEARING ONE: the raw guard rate on `loop L1` is at least its own
    `pair L1` row on at least 2 of 3 seeds. This is the claim that the loop is
    what the guard was missing, and the reason to spend the GPU hour.
  * AND NOT BY BEING LOUDER: total flips on `loop L1` are at most that seed's
    `pair L1` flips on at least 2 of 3. A guard bought with MORE churn is the
    thing stage PAIR already has; the prediction is a guard at equal or lower
    churn. The two together are the finding, and either alone is not.
  * THE LOOP ALONE BUYS NO GUARD. `loop L0` stays below P4's 0.10 bar on 3/3.
    The margin supplies the target; the loop only supplies the reach.
  * IT DOES NOT BREAK THE SOLVER. Validation token accuracy on both loop arms is
    within 0.02 of this job's own Y row, measured here and not retyped.

WHAT FALSIFIES IT. A `loop L1` guard below its `pair L1` row on 2 of 3, which
would say the reach was never the constraint and the churn is a property of the
cell. Separately and more bluntly: train nll above 2.5 or validation token
accuracy below 0.30 on `loop L1` would say the feedback destroys the body, in
which case the honest report is that the arm is broken and every AUROC in the
table is unreadable -- not a negative result about guards.

NOTHING REGISTERED CHANGES AND NOTHING IS APPLIED. The bars (P1-P4, the placebo
band, the pooled bar), the 1,000 eval cases, the 1.630 convergence rule, the form
of `L_pair`, lambda, the margin and the 3,000-step training length are all
exactly stage PAIR's and all still hshunt's to change. This arm adds an
architecture flag that is OFF for every existing checkpoint and measures it with
the registered instruments. NO VERDICT IS PRINTED.
"""

import json
import os
import sys
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/spin_gnn/scripts")

import stagepair_score as SP
from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sat_dataset2 import (SATTokenDataset, iterate_token_batches,
                                   make_token_batch_pair)
from spin_gnn.sat_transformer2 import SATReasoningTransformer, sat2_nll_loss
from spin_gnn.sudoku_cfresp import (run_cfresp, run_stratified,
                                    summarize_cfresp)
from spin_gnn.sudoku_determinacy import (determinacy_loss_from_channel,
                                         determinacy_targets)
from spin_gnn.sudoku_pair import build_pair_index
from spin_gnn.sudoku_skip import find_unforceable_deductions
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab, evaluate

_Y = "/scratch/users/hshunt/spin_gnn/sudoku_n48_determinacy_Y_2026-11-14b"
_CELL = "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_chan"
Y_DIRS = {
    0: f"{_Y}/t00_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s0/{_CELL}",
    1: f"{_Y}/t01_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s1/{_CELL}_s01",
    2: f"{_Y}/t02_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s2/{_CELL}_s02",
}

# The recipe, copied from `stagepair_pairprobe.py` so the arm is that arm plus a
# flag. Changing any of these would make the comparison two knobs.
N_INSTANCES = 30_000
BATCH = 32
STEPS = 3_000
LOG_EVERY = 500
MAX_LEN = 512
LR = 1e-3
MARGIN = SP.MARGIN
SEED = int(os.environ.get("LOOP_SEED", "0"))
assert SEED in Y_DIRS, "no stage Y checkpoint for seed %d" % SEED
CKPT_DIR = Y_DIRS[SEED]
SUF = "" if SEED == 0 else f"_s{SEED}"
PAIR_DIR = f"/scratch/users/hshunt/spin_gnn/runs/stagepair_pairprobe_2026-11-15{SUF}"
OUT_DIR = f"/scratch/users/hshunt/spin_gnn/runs/stageloop_probe_2026-11-15{SUF}"
DATA_SEED = SEED + 777

# Registered eval streams, imported and never retyped.
EVAL_SEED_START = SP.PAIR_EVAL_SEED_START
EVAL_N_CASES = SP.PAIR_EVAL_N_CASES
AUROC_SEED_START = SP.AUROC_SEED_START
EVAL_N_INSTANCES = 1_400
AUROC_N_INSTANCES = 600
# Held-out, and distinct from every stream this project uses: the fine-tuning
# data (777-779), the pair index (DATA_SEED + 500,000,000), the guard spec
# (600,000,000), the pair eval (920,000,000) and the stratified AUROC
# (70,000,000,000). Validation token accuracy is the check that never censors,
# so it gets its own instances rather than a slice of a training stream.
VAL_SEED_START = 300_000_000
VAL_N_INSTANCES = 400

# THE REGISTERED PREDICTIONS, read back against the table at the end. Bars are
# imported from `stagepair_score`; these are claims ABOUT this arm, not bars.
PREDICTION = {
    "loop_scale_abs": (0.05, 2.0),
    "p1_on_loop_l1": SP.PAIR_BAR,
    "guard_beats_pair_on_n_seeds": 2,
    "flips_at_most_pair_on_n_seeds": 2,
    "l0_guard_below": SP.GUARD_BAR,
    "val_acc_tol_vs_Y": 0.02,
    "break_nll_above": 2.5,
    "break_val_acc_below": 0.30,
}

# ---------------------------------------------------------------------------
# THE FORCED LOOP (`LOOP_FIX=0.10,0.30`), AND THE SECOND REGISTERED HYPOTHESIS,
# WRITTEN 2026-11-15 AFTER FINDING 87 AND BEFORE ANY FORCED RUN EXISTS.
#
# Finding 87 came back a null of a particular kind: `loop L1` ends at |c| =
# 0.0007 / 0.0006 / 0.0002 on the three seeds, with the feedback at 0.5% / 0.7% /
# 0.4% of the residual stream, and every other column lands on stage PAIR's
# published row inside the noise 84-86 measured (corrupt 0.8720 / 0.8710 / 0.8680
# against 0.8710 / 0.8620 / 0.8710; guard 0.1072 / 0.2211 / 0.1416 against 0.1087
# / 0.2072 / 0.1632). The loop was available, differentiable and free, and three
# independent trainings all left it shut.
#
# That is a measurement about the OBJECTIVE, not about the architecture, and the
# two are worth separating. Adam moves a scalar by ~lr per step regardless of
# gradient size, so 3,000 steps could have carried c to 3.0; |c| staying at 1e-3
# means the sign of dL/dc kept flipping. The delta is exactly LINEAR in c, so the
# useful range is set by the body: at ||s|| = 7.05 the gate is large, and one
# Adam step of 1e-3 is already ~1% of h. The arm explored that range and declined
# it. What it cannot say is what a body that MUST read its own determinacy
# estimate would do -- and that is the architectural question.
#
# `LOOP_FIX` sets c to a measured constant and FREEZES it. The constant is not
# guessed: with c = 1 the delta's size is measured on one batch of this very
# body, and because the delta is linear in c, `c = target / r1` puts the feedback
# at exactly `target` of the residual stream at the warm start. So `LOOP_FIX=0.10`
# means "ten per cent of h, on this body, by measurement". A frozen scalar is not
# trainable, so the count returns to the registered 614,138 exactly.
#
# THE PREDICTIONS, registered here and read back at the end:
#   * THE BODY SURVIVES THE MILD SETTING. `fix0.10 L1` ends within 0.02 of this
#     job's own Y validation token accuracy; `fix0.30 L1` within 0.05. A bigger
#     drop at 0.30 means that setting is too strong to read as anything but
#     damage, and it is reported as damage.
#   * THE PREDICATE STILL ARRIVES. `fix0.10 L1` clears P1 (>= 0.65) and stays in
#     P2's (0.40, 0.60) band on 3/3. The margin is unchanged; only its reach is.
#   * THE LOAD-BEARING ONE, same claim as 87's and now with the loop actually
#     open: `fix0.10 L1`'s raw guard is at least its own `pair L1` row on >= 2 of
#     3 seeds, at flips no higher than that row's.
#   * THE LOOP IS A NOISE CHANNEL UNTIL SHOWN OTHERWISE -- and this is the
#     reading I expect to win, because it explains 87. A changed hint changes g
#     at many nodes at once, so an open loop propagates that change into the
#     computation: `fix0.10 L0`, which has NO margin at all, should still show
#     more flips than stage Y's 4 / 1 / 7 on 3/3. If it does, the optimizer's
#     refusal in 87 was not indifference -- it was avoidance, and the honest
#     reading of the whole stage-LOOP arm is that the feedback is a noise channel
#     rather than a control channel.
#
# WHAT WOULD OVERTURN THAT: `fix0.10 L1` showing a HIGHER guard at LOWER churn
# than `pair L1` on >= 2 of 3. That is the result this whole line has been
# looking for, and it is exactly what 87's null did not deliver.
#
# Nothing registered changes here either, and no verdict is printed.
FIX_TARGETS = tuple(float(t) for t in
                    os.environ.get("LOOP_FIX", "").split(",") if t.strip())
assert all(0 < t < 1 for t in FIX_TARGETS), FIX_TARGETS
if FIX_TARGETS:
    OUT_DIR += "_fixed"
    PREDICTION = {
        **PREDICTION,
        "fix_targets": list(FIX_TARGETS),
        "val_acc_tol_vs_Y": 0.02,        # at the mild setting
        "val_acc_tol_strong": 0.05,      # at 0.30, where damage is expected
        "l0_flips_above_Y": True,        # the noise-channel reading
    }

# A SMOKE RUN IS NOT A MEASUREMENT. `LOOP_SMOKE=1` shrinks every size so the
# path runs end to end in ~2 minutes and a crash costs that instead of a GPU
# hour. Every number it prints is wrong on purpose; it writes to its own
# directory and exists only to find the KeyError before the real run does.
SMOKE = os.environ.get("LOOP_SMOKE") == "1"
if SMOKE:
    print("=" * 72)
    print("LOOP_SMOKE=1 -- SHAPES ONLY. Every size below is wrong on purpose "
          "and no number\nprinted by this run is a measurement of anything.")
    print("=" * 72, flush=True)
    N_INSTANCES, STEPS, LOG_EVERY = 400, 20, 10
    EVAL_N_CASES, EVAL_N_INSTANCES = 40, 80
    # not 40: `run_stratified` needs both classes in a stratum and `evaluate`
    # drops the last partial batch, so a 40-instance smoke can divide by zero in
    # code paths the real run never touches
    AUROC_N_INSTANCES, VAL_N_INSTANCES = 120, 96
    OUT_DIR += "_smoke"

dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"device={dev}  SEED={SEED}  steps={STEPS:,}/arm", flush=True)
print(f"eval seed_start={EVAL_SEED_START:,}  cases={EVAL_N_CASES:,}", flush=True)
print(f"stage Y body:   {CKPT_DIR}", flush=True)
print(f"stage PAIR arm: {PAIR_DIR}/warm_L1", flush=True)
print(f"out: {OUT_DIR}", flush=True)

with open(os.path.join(CKPT_DIR, "config.json")) as fh:
    config = json.load(fh)
assert config.get("determinacy_channel"), "stage Y checkpoint has no channel"
assert config["max_len"] == MAX_LEN, config["max_len"]

# The published stage PAIR row, READ from the artifact rather than retyped, so
# the control row in this table and the one in findings 82-83 are the same
# measurement of the same weights.
with open(os.path.join(PAIR_DIR, "pair_summary.json")) as fh:
    PAIR_PUBLISHED = json.load(fh)["per_arm"]["warm L1"]
print(f"published pair L1: corrupt {PAIR_PUBLISHED['pair_auroc_corrupt']:.4f}  "
      f"guard {PAIR_PUBLISHED['guard_rate']:.4f} of "
      f"{PAIR_PUBLISHED['guard_denom']}  nll "
      f"{PAIR_PUBLISHED['final_train_nll']:.4f}", flush=True)

sudoku_cfg = SudokuReasoningConfig(n_nodes=48, degree=5, n_colors=4,
                                   shuffle_adj=True, fall=True, seed=DATA_SEED)
vocab = build_vocab(sudoku_cfg)
y_state = torch.load(os.path.join(CKPT_DIR, "model.pt"), map_location="cpu",
                     weights_only=True)
pair_state = torch.load(os.path.join(PAIR_DIR, "warm_L1", "model.pt"),
                        map_location="cpu", weights_only=True)


def build(feedback, state):
    """The registered cell, with or without the loop, warm-started from `state`.

    The parameter count is asserted against the imported `SP.N_TRAINABLE` plus
    the one scalar the loop costs -- never a retyped 614,139 -- and `max_len` is
    pinned because the positional table is trainable and so is a term in it.
    """
    m = SATReasoningTransformer(
        vocab.size, d_model=320, num_layers=16, nhead=4, max_len=MAX_LEN,
        dim_feedforward=1024, norm_first=True, arch_mode="tied_no_mlp",
        n_blocks=1, determinacy_channel=True,
        node_offset=vocab.node_offset, n_nodes=vocab.n_nodes,
        determinacy_feedback=feedback)
    assert m.pos_embed.shape[0] == MAX_LEN
    missing, unexpected = m.load_state_dict(state, strict=not feedback)
    if feedback:
        # the loop arms warm-start from bodies that predate the parameter
        assert list(missing) in ([], ["loop_scale"]), list(missing)
        assert not list(unexpected), list(unexpected)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == SP.N_TRAINABLE + (1 if feedback else 0), (
        "not the registered cell: %s trainable" % f"{n:,}")
    m = m.to(dev)
    s = float(m.det_scale.norm())
    assert s > 0.05, ("det_scale is %.3e -- dead channel, probe is vacuous" % s)
    if feedback:
        assert float(m.loop_scale.abs()) == 0.0, "the loop must start closed"
    return m


# ---------------------------------------------------------------------------
# data: the training stream, the held-out validation stream
# ---------------------------------------------------------------------------
print(f"\nbuilding the {N_INSTANCES:,} fine-tuning instances (data seed "
      f"{DATA_SEED})", flush=True)
inst = build_instances(sudoku_cfg, N_INSTANCES, sudoku_cfg.seed)
tokens = [encode_instance(adj, trace, vocab) for adj, _, _, trace in inst]
aux = [determinacy_targets(adj, trace, vocab, tok)
       for (adj, _c, _h, trace), tok in zip(inst, tokens)]
pair = build_pair_index(inst, tokens, vocab, sudoku_cfg.seed + 500_000_000)
ds = SATTokenDataset(tokens, max_len=MAX_LEN, aux=aux, pair=pair)
idx = list(range(len(ds)))
print(f"dataset={len(ds):,}", flush=True)

print(f"building the {VAL_N_INSTANCES} held-out validation instances (seed "
      f"{VAL_SEED_START:,})", flush=True)
val_inst = build_instances(sudoku_cfg, VAL_N_INSTANCES, VAL_SEED_START)
val_tokens = [encode_instance(adj, trace, vocab)
              for adj, _, _, trace in val_inst]
val_ds = SATTokenDataset(val_tokens, max_len=MAX_LEN)
val_idx = list(range(len(val_ds)))
print(f"validation set={len(val_ds):,}", flush=True)


# ---------------------------------------------------------------------------
# the two trained arms
# ---------------------------------------------------------------------------
def calibrate_c(model, target):
    """Set c so the feedback is `target` of the residual stream, and FREEZE it.

    The delta is exactly linear in c, so one forward at c = 1 on one batch of
    this body measures the whole scaling: c = target / r1. Guessing the constant
    from another body is what findings 76 and 79 are about -- at ||s|| = 7.05 a c
    of 0.1 is a different perturbation from a c of 0.1 at ||s|| = 2.
    """
    batch = next(iter(iterate_token_batches(ds, idx, BATCH, shuffle=False,
                                            collate=make_token_batch_pair)))
    x, _y, mask = (t.to(dev) for t in batch[:3])
    with torch.no_grad():
        model.loop_scale.fill_(1.0)
        model(x, valid_mask=mask)
        r1 = float(model._fb_rel)
        c = target / r1
        model.loop_scale.fill_(c)
    model.loop_scale.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == SP.N_TRAINABLE, "a frozen c must leave the registered count: %s" % n
    print(f"  calibrated: mean|delta|/mean|h| = {r1:.4f} at c=1, so c = "
          f"{c:.6f} for a target of {target:.2f}  (c is FROZEN)", flush=True)
    return c


def train_arm(pair_w, label, fix_target=None):
    """3,000 steps from stage Y with the loop on. Same loop as the pairprobe."""
    print(f"\n=== {label}: lambda={pair_w:g}, feedback ON"
          + (f", c FIXED at {fix_target:.2f} of h " if fix_target else " ")
          + "=" * 20, flush=True)
    torch.manual_seed(SEED)
    model = build(True, y_state)
    if fix_target is not None:
        calibrate_c(model, fix_target)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=LR)
    rng = np.random.default_rng(SEED)
    print(f"{'step':>5} {'main':>7} {'hinge':>7} {'gap':>8} {'sat':>5} "
          f"{'|s|':>8} {'c':>8} {'fb/h':>7}", flush=True)
    step, main_v, hinge_v, gap_v, sat_v = 0, float("nan"), float("nan"), \
        float("nan"), float("nan")
    tail, hist = [], []
    while step < STEPS:
        for batch in iterate_token_batches(ds, idx, BATCH, rng=rng,
                                           collate=make_token_batch_pair):
            if step >= STEPS:
                break
            x, y, mask = (t.to(dev) for t in batch[:3])
            pr = batch[5].to(dev)
            logits, g = model(x, valid_mask=mask, return_determinacy=True)
            main = sat2_nll_loss(logits, y, mask)
            loss = main
            det = determinacy_loss_from_channel(g, batch[3].to(dev),
                                                batch[4].to(dev))
            if det is not None:
                loss = loss + det
            if pair_w > 0:
                sel = (pr[:, 0] >= 0).nonzero(as_tuple=True)[0]
                if sel.numel():
                    xb = x.clone()
                    xb[sel, pr[sel, 0]] = pr[sel, 1]
                    _zb, gb = model(xb, valid_mask=mask,
                                    return_determinacy=True)
                    gap = (g[sel, pr[sel, 2], pr[sel, 3]]
                           - gb[sel, pr[sel, 2], pr[sel, 3]])
                    hinge = torch.clamp(MARGIN - gap, min=0)
                    loss = loss + pair_w * hinge.mean()
                    hinge_v, gap_v = float(hinge.mean()), float(gap.mean())
                    sat_v = float((gap >= MARGIN).float().mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            main_v = float(main)
            if step >= STEPS - 250:
                tail.append(main_v)
            if step % LOG_EVERY == 0:
                c = float(model.loop_scale)
                fb = float(model._fb_rel)
                print(f"{step:5d} {main_v:7.4f} {hinge_v:7.4f} {gap_v:+8.4f} "
                      f"{sat_v:5.2f} {float(model.det_scale.norm()):8.4f} "
                      f"{c:+8.4f} {fb:7.4f}", flush=True)
                hist.append({"step": step, "train_nll": main_v,
                             "hinge": hinge_v, "gap": gap_v, "sat": sat_v,
                             "det_scale_norm": float(model.det_scale.norm()),
                             "loop_scale": c, "fb_rel": fb})
            step += 1
    mean_tail = float(np.mean(tail))
    c = float(model.loop_scale)
    print(f"{'end':>5} {main_v:7.4f} {hinge_v:7.4f} {gap_v:+8.4f} {sat_v:5.2f} "
          f"{float(model.det_scale.norm()):8.4f} {c:+8.4f} "
          f"{float(model._fb_rel):7.4f}   (mean of last 250: {mean_tail:.4f})",
          flush=True)
    d = os.path.join(OUT_DIR, label.replace(" ", "_"))
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, "model.pt"))
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump({**config, "determinacy_feedback": True}, fh, indent=2)
    print(f"  checkpoint kept: {d}", flush=True)
    return model, {"final_train_nll": main_v,
                   "mean_train_nll_last250": mean_tail,
                   "final_hinge": hinge_v, "final_sat": sat_v,
                   "lambda": pair_w, "loop_scale": c,
                   "fb_rel": float(model._fb_rel),
                   "det_scale_norm": float(model.det_scale.norm()),
                   "fix_target": fix_target, "history": hist}


# The learned-c arms are finding 87 and are NOT re-run when a forced arm is
# asked for: `LOOP_FIX` writes to its own directory and leaves 87's artifact
# alone. The mild target carries the lambda=0 control, because the noise-channel
# reading is a claim about an open loop with no margin at all.
if FIX_TARGETS:
    T0 = FIX_TARGETS[0]
    TO_TRAIN = ([(0.0, f"fix{T0:.2f} L0", T0)]
                + [(1.0, f"fix{t:.2f} L1", t) for t in FIX_TARGETS])
else:
    TO_TRAIN = [(0.0, "loop L0", None), (1.0, "loop L1", None)]
L0_LABEL = TO_TRAIN[0][1]
L1_LABEL = TO_TRAIN[1][1]

models, train = {}, {}
for w, label, fx in TO_TRAIN:
    m, tr = train_arm(w, label, fx)
    m.eval()
    models[label], train[label] = m, tr

# the two reloaded rows, eval only
models["Y"] = build(False, y_state)
models["Y"].eval()
with open(os.path.join(CKPT_DIR, "history.json")) as fh:
    y_nll = float(json.load(fh)[-1]["train_nll"])
train["Y"] = {"final_train_nll": y_nll, "mean_train_nll_last250": y_nll,
              "final_hinge": float("nan"), "final_sat": float("nan"),
              "lambda": None, "loop_scale": float("nan"),
              "fb_rel": float("nan"), "fix_target": None,
              "det_scale_norm": float(models["Y"].det_scale.norm())}
models["pair L1"] = build(False, pair_state)
models["pair L1"].eval()
train["pair L1"] = {
    "final_train_nll": PAIR_PUBLISHED["final_train_nll"],
    "mean_train_nll_last250": PAIR_PUBLISHED["mean_train_nll_last250"],
    "final_hinge": PAIR_PUBLISHED["final_hinge"],
    "final_sat": PAIR_PUBLISHED["final_sat"], "lambda": 1.0,
    "loop_scale": float("nan"), "fb_rel": float("nan"),
    "fix_target": None, "det_scale_norm": float(models["pair L1"].det_scale.norm())}

ARMS = ("Y", "pair L1") + tuple(label for _w, label, _f in TO_TRAIN)

# ---------------------------------------------------------------------------
# VALIDATION TOKEN ACCURACY FIRST. It never censors.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("validation token accuracy, held-out stream, BEFORE any AUROC", flush=True)
val = {}
for label in ARMS:
    v = evaluate(models[label], val_ds, val_idx, dev)
    val[label] = v
    print(f"  {label:<9} accuracy {v['accuracy']:.4f}   nll {v['nll']:.4f}",
          flush=True)
    models[label].eval()

# ---------------------------------------------------------------------------
# one case list, every row
# ---------------------------------------------------------------------------
print(f"\nbuilding the eval case list at seed_start {EVAL_SEED_START:,}",
      flush=True)
crng = np.random.default_rng(0)
cases = []
for adj, colors, hints, trace in build_instances(sudoku_cfg, EVAL_N_INSTANCES,
                                                 EVAL_SEED_START):
    for case in find_unforceable_deductions(adj, colors, hints, trace, vocab,
                                            crng):
        cases.append((adj, colors, hints, trace, case))
print(f"{len(cases):,} breakable round-0 deductions from "
      f"{EVAL_N_INSTANCES:,} instances", flush=True)
assert len(cases) >= EVAL_N_CASES, (
    "only %d cases; raise EVAL_N_INSTANCES rather than lowering the registered "
    "PAIR_EVAL_N_CASES" % len(cases))
cases = cases[:EVAL_N_CASES]

det_instances = list(build_instances(sudoku_cfg, AUROC_N_INSTANCES,
                                     AUROC_SEED_START))

per_arm = {}
for label in ARMS:
    print(f"\n--- probing {label} " + "-" * 40, flush=True)
    rows = run_cfresp(models[label], vocab, dev, cases)
    s = summarize_cfresp(rows)
    st = run_stratified(models[label], vocab, dev, det_instances)

    # THE DENOMINATOR, counted and not assumed -- cases where the CLEAN prompt
    # emits v at all. `check_p4` raises on 350 for exactly this reason.
    clean = [r for r in rows if r["clean_emits_v"]]
    denom = len(clean)
    guard = (float(np.mean([not r["corrupt_emits_v"] for r in clean]))
             if clean else float("nan"))
    # the control P4 never had: the placebo leaves v forced, so a turn-off there
    # is a withdrawal the predicate does not license
    placebo_guard = (float(np.mean([not r["placebo_emits_v"] for r in clean]))
                     if clean else float("nan"))
    b = sum(1 for r in clean
            if not r["corrupt_emits_v"] and r["placebo_emits_v"])
    c_ = sum(1 for r in clean
             if r["corrupt_emits_v"] and not r["placebo_emits_v"])
    excess = (b - c_) / denom if denom else float("nan")
    se = np.sqrt(b + c_) / denom if denom else float("nan")
    # same definition as `make_predicate_web.mcnemar`: EITHER edit changing
    # whether v is emitted, which is the 142 / 175 / 177 column
    flips = sum(1 for r in rows
                if r["clean_emits_v"] != r["corrupt_emits_v"]
                or r["clean_emits_v"] != r["placebo_emits_v"])

    p = dict(train[label])
    p.pop("history", None)
    p.update({
        "arm": label,
        "val_accuracy": val[label]["accuracy"], "val_nll": val[label]["nll"],
        "pair_auroc_corrupt": SP.matched_pair_auroc(
            [r["dg_corrupt"] for r in rows]),
        "pair_auroc_placebo": SP.matched_pair_auroc(
            [r["dg_placebo"] for r in rows]),
        "pooled_auroc": st["model_pooled"],
        "within_stratum_auroc": st["model"],
        "guard_rate": guard, "guard_denom": denom,
        "placebo_guard_rate": placebo_guard,
        "excess": excess, "excess_lo": excess - 1.96 * se,
        "excess_hi": excess + 1.96 * se,
        "corrupt_only": b, "placebo_only": c_,
        "p": stats.binomtest(b, b + c_, 0.5).pvalue if b + c_ else float("nan"),
        "total_flips": flips, "n_cases": len(rows),
        "median_abs_dg_corrupt": s["median_abs_dg_corrupt"],
        "median_abs_dg_placebo": s["median_abs_dg_placebo"],
        "median_g_spread": s["median_g_spread"],
        "clean_emits_v": s["clean_emits_v"],
        "corrupt_emits_v": s["corrupt_emits_v"],
        "placebo_emits_v": s["placebo_emits_v"],
    })
    per_arm[label] = p
    print(f"  pair AUROC corrupt {p['pair_auroc_corrupt']:.4f}   placebo "
          f"{p['pair_auroc_placebo']:.4f}")
    print(f"  pooled {p['pooled_auroc']:.4f}  within-stratum "
          f"{p['within_stratum_auroc']:.4f}")
    print(f"  guard {guard:.4f} of {denom} cases where the clean prompt emits v "
          f"(NOT of {len(rows)});  flips {flips}", flush=True)

with open(os.path.join(OUT_DIR, "loop_summary.json"), "w") as fh:
    json.dump({"per_arm": per_arm,
               "history": {k: v.get("history", []) for k, v in train.items()},
               "prediction": PREDICTION,
               "pair_published": PAIR_PUBLISHED,
               "train": {"steps": STEPS, "data_seed": DATA_SEED, "lr": LR,
                         "margin": MARGIN, "seed": SEED,
                         "n_instances": N_INSTANCES, "batch": BATCH},
               "eval": {"seed_start": EVAL_SEED_START, "n_cases": EVAL_N_CASES,
                        "n_instances": EVAL_N_INSTANCES,
                        "auroc_seed_start": AUROC_SEED_START,
                        "auroc_n_instances": AUROC_N_INSTANCES,
                        "val_seed_start": VAL_SEED_START,
                        "val_n_instances": VAL_N_INSTANCES}},
              fh, indent=2)
print(f"\nwrote {OUT_DIR}/loop_summary.json")

# ---------------------------------------------------------------------------
# the table, and the hypothesis read against it
# ---------------------------------------------------------------------------
print("\n" + "=" * 94)
print(f"{'arm':<11} {'val acc':>8} {'corrupt':>8} {'placebo':>8} {'pooled':>7} "
      f"{'guard':>7} {'denom':>6} {'flips':>6} {'c':>8} {'nll':>8}")
for label in ARMS:
    p = per_arm[label]
    print(f"{label:<11} {p['val_accuracy']:8.4f} {p['pair_auroc_corrupt']:8.4f} "
          f"{p['pair_auroc_placebo']:8.4f} {p['pooled_auroc']:7.4f} "
          f"{p['guard_rate']:7.4f} {p['guard_denom']:6d} "
          f"{p['total_flips']:6d} {p['loop_scale']:+8.4f} "
          f"{p['final_train_nll']:8.4f}")

print(f"\nbars, imported and read rather than applied: P1 corrupt >= "
      f"{SP.PAIR_BAR}, P2 placebo in {list(SP.PLACEBO_BAND)},")
print(f"P3 pooled >= {SP.POOLED_BAR}, P4 guard >= {SP.GUARD_BAR}. One seed per "
      f"job, so scoring needs the assembly step.")

l1, l0, pr_ = per_arm[L1_LABEL], per_arm[L0_LABEL], per_arm["pair L1"]
lo, hi = PREDICTION["loop_scale_abs"]
print("\nthe registered predictions, read against the table:")
if FIX_TARGETS:
    print(f"  1. c is FIXED, not learned: {l1['loop_scale']:+.6f} on {L1_LABEL} "
          f"for a target of {l1['fix_target']:.2f} of the residual stream, "
          f"measured feedback {l1['fb_rel']:.4f}. Finding 87's LEARNED c ended "
          f"at 0.0007 / 0.0006 / 0.0002, which is what this arm exists to "
          f"bypass.")
else:
    print(f"  1. the loop opens: |c| = {abs(l1['loop_scale']):.4f} on "
          f"{L1_LABEL} (predicted {lo}-{hi}), "
          f"{abs(l0['loop_scale']):.4f} on {L0_LABEL}; feedback magnitude "
          f"mean|delta|/mean|h| = {l1['fb_rel']:.4f}")
print(f"  2. the predicate survives: corrupt {l1['pair_auroc_corrupt']:.4f} "
      f"(P1 {SP.PAIR_BAR}), placebo {l1['pair_auroc_placebo']:.4f} "
      f"(P2 {list(SP.PLACEBO_BAND)})")
print(f"  3. THE LOAD-BEARING ONE: {L1_LABEL} guard {l1['guard_rate']:.4f} of "
      f"{l1['guard_denom']} vs pair L1 {pr_['guard_rate']:.4f} of "
      f"{pr_['guard_denom']} -- "
      f"{'HIGHER' if l1['guard_rate'] >= pr_['guard_rate'] else 'LOWER'} "
      f"on this seed")
print(f"  4. and not by being louder: flips {l1['total_flips']} vs pair L1's "
      f"{pr_['total_flips']} -- "
      f"{'at most' if l1['total_flips'] <= pr_['total_flips'] else 'MORE THAN'}"
      f" the control")
print(f"  5. the loop alone buys no guard: {L0_LABEL} guard "
      f"{l0['guard_rate']:.4f} of {l0['guard_denom']} "
      f"({'below' if l0['guard_rate'] < SP.GUARD_BAR else 'AT OR ABOVE'} "
      f"{SP.GUARD_BAR})")
dv0 = abs(l0["val_accuracy"] - per_arm["Y"]["val_accuracy"])
dv1 = abs(l1["val_accuracy"] - per_arm["Y"]["val_accuracy"])
print(f"  6. it does not break the solver: val accuracy {l0['val_accuracy']:.4f}"
      f" / {l1['val_accuracy']:.4f} against Y's "
      f"{per_arm['Y']['val_accuracy']:.4f} (|d| {dv0:.4f} / {dv1:.4f}, "
      f"predicted <= {PREDICTION['val_acc_tol_vs_Y']})")

# the two read-backs that exist only for the forced arm. Both are claims about
# an OPEN loop, which finding 87 never had, so neither has a learned-c analogue.
if FIX_TARGETS:
    tol_s = PREDICTION["val_acc_tol_strong"]
    for _w, lab, fx in TO_TRAIN[2:]:
        dv = abs(per_arm[lab]["val_accuracy"] - per_arm["Y"]["val_accuracy"])
        print(f"  6b. the STRONG setting is allowed to hurt, up to {tol_s}: "
              f"{lab} val accuracy {per_arm[lab]['val_accuracy']:.4f}, |d| "
              f"{dv:.4f} vs Y -- "
              f"{'within' if dv <= tol_s else 'BEYOND'} it"
              f"{'' if dv <= tol_s else '; read that row as damage, not as a guard'}")
    # the noise-channel reading, and the one I expect to win. Stage Y's flips on
    # the three seeds of finding 87 were 4 / 1 / 7; this job measures its own Y
    # row on its own cases, so BOTH are printed and the in-job one is the test.
    yf = per_arm["Y"]["total_flips"]
    print(f"  7. THE NOISE-CHANNEL READING: {L0_LABEL} has no margin at all, and "
          f"flips\n      {l0['total_flips']} of {EVAL_N_CASES} against "
          f"this job's own Y row at {yf} (finding 87's stage Y: 4 / 1 / 7) -- "
          f"{'MORE' if l0['total_flips'] > yf else 'not more'} than Y. "
          f"{'An open loop moves the computation even with nothing asking it to, so 87 was avoidance rather than indifference.' if l0['total_flips'] > yf else 'The open loop alone does not churn, so the feedback is not a noise channel on this seed.'}")

broken = [lab for _w, lab, _f in TO_TRAIN
          if per_arm[lab]["final_train_nll"] > PREDICTION["break_nll_above"]
          or per_arm[lab]["val_accuracy"] < PREDICTION["break_val_acc_below"]]
print(f"  the break falsifier (train nll > {PREDICTION['break_nll_above']} or "
      f"val acc < {PREDICTION['break_val_acc_below']}): "
      f"{broken if broken else 'neither arm'}")
if broken:
    print("  -> the feedback DESTROYED the body on the arms named above, and "
          "every AUROC in\n     that row is unreadable as a statement about "
          "guards.")

print("\nThe churn column (flips) and the guard column are the finding only "
      "TOGETHER: a guard\nbought with more churn is what stage PAIR already "
      "has. Nothing registered changed --\nthe bars, the cases, the 1.630 rule, "
      "the form of L_pair and the 3,000 steps are all\nstage PAIR's and all "
      "still hshunt's to change. NO VERDICT IS PRINTED.")
