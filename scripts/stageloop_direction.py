"""Is the determinacy injection a COMMON-MODE shift? Finding 88's open hypothesis.

THE QUESTION, AND WHY IT IS WORTH ONE CPU JOB. Finding 88 forced the stage-LOOP
feedback open at a measured 10% and 30% of the residual stream and the body was
destroyed before a single gradient step: train nll 9.18 / 6.45 / 6.73 nats at
step 0 against the body's 1.60 and log 58 = 4.060. The `/spin-gnn/loop/` deck
states one hypothesis for WHY and states it explicitly as unmeasured:

    delta_p = (c / n_nodes) * sum_v [log sigma(g_{p,v}) + log 2] * E_v

is an average of NODE-TOKEN embedding rows added at EVERY one of the ~391
positions, and the only thing that distinguishes one position from another is
the gate `a_{p,v}`. If the gate looks similar at every position, delta is nearly
the SAME VECTOR everywhere -- a large common-mode translation of the whole
residual stream rather than a position-specific correction. Finding 88
calibrated the MAGNITUDE of the injection by measurement and never looked at its
DIRECTION. This script looks, on the same three bodies, with no training and no
GPU.

IT MEASURES GEOMETRY, NOT CAUSATION. Even a perfectly common-mode injection
being what WRECKS the body is an inference; a forward pass can only say whether
the geometric claim is true. If it is false the deck's hypothesis is withdrawn,
not reworded.

WHAT IS MEASURED. One batch of 32 instances from each seed's own training-stream
prefix (DATA_SEED = SEED + 777), through that seed's stage Y body -- the body
88's forced arms warm-started from, loaded from `Y_DIRS` below. A forward hook on
the shared block captures, at every one of the 16 applications, the hidden state
going in and the hidden state coming out; `delta_t` is then recomputed from the
captured hidden state by the model's OWN
`SATReasoningTransformer.determinacy_feedback_delta`, never reimplemented here.
For a set of vectors {v_p} over the VALID (non-pad) positions of one sequence:

    cm(v)  = || mean_p v_p ||  /  mean_p || v_p ||          in [0, 1]
    cos(v) = mean over p != q of  <v_p, v_q> / (|v_p| |v_q|)

cm is 1 when every position gets the identical vector and ~1/sqrt(L) = 0.051 at
L = 391 for independent random directions; cos is 1 and 0 at those two ends. Both
are scale-free, so neither depends on c and neither can be rescued or ruined by
the calibration flaws 88 found.

THE CONTROLS, AND THE REASON THERE ARE TWO. A number without the arm that should
NOT show the effect is not a finding (findings 80, 83, 88). So the same two
statistics are computed for (a) the shared block's OWN per-application update
`dh_t = blk(h_t) - h_t`, which is what a position-specific correction produced by
this very cell looks like, and (b) the residual stream `h_t` itself, because if
h is already largely common-mode then a common-mode delta is unremarkable.

THREE MODES, one forward each, same hook:
    unit    loop_scale = 1            reproduces `calibrate_c`'s measurement
    open    loop_scale = c_solved     the trajectory 88's `fix0.10` arms saw
    closed  loop_scale = 0            the unperturbed body: direction before
                                      the perturbation compounds
`c_solved` is READ from each seed's `loop_summary.json` (`fix0.10 L1`), never
retyped, and `unit` mode's ratio at the last application is compared against the
`0.10 / c_solved` that calibration implies.

THE REGISTERED PREDICTIONS, WRITTEN AND COMMITTED BEFORE THIS SCRIPT IS RUN:
  * THE INJECTION IS COMMON-MODE. cos(delta) > 0.80 at every application on 3/3
    bodies, in closed mode. This is the deck's hypothesis, stated so it can fail.
  * THE BODY'S OWN UPDATE IS NOT. cos(dh) is at least 0.30 BELOW cos(delta) at
    every application on 3/3. Without this the first prediction says nothing: it
    is the control that makes "common-mode" a property of the injection rather
    than of this cell's activations.
  * THE RESIDUAL STREAM IS IN BETWEEN. cos(h) lands below cos(delta) on 3/3.
  * r1 REPRODUCES, LOOSELY. `unit` mode's ratio at t = 14 is within 20% relative
    of `0.10 / c_solved`. LOOSE ON PURPOSE: this batch is the first 32 of a
    64-instance build and 88's was the first 32 of a 30,000-instance build, and
    nothing guarantees `build_instances` is prefix-identical across counts. A
    miss here is a statement about the batch, not about the geometry; the
    direction statistics do not depend on it.
  * AND THE RATIO IS NOT CONSTANT IN t. max_t r1_t / min_t r1_t > 1.5 in unit
    mode -- the other face of 88's calibration flaw, that one forward pass at
    c = 1 cannot set the size of an injection the loop applies 15 times.

WHAT FALSIFIES THE HYPOTHESIS: cos(delta) within 0.10 of cos(dh), which would
say direction is not what singles the injection out and would leave 88's
destruction unexplained by this account.

NOTHING REGISTERED CHANGES, NOTHING IS TRAINED, NO CHECKPOINT IS WRITTEN AND NO
VERDICT IS PRINTED. No bar in `stagepair_score` is touched; this script reads
weights and prints a table.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/spin_gnn/scripts")

import stagepair_score as SP
from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sat_dataset2 import (SATTokenDataset, iterate_token_batches,
                                   make_token_batch)
from spin_gnn.sat_transformer2 import SATReasoningTransformer
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab

# Copied verbatim from `stageloop_probe.py` -- the same three bodies 87 and 88
# warm-started from. Every path is asserted to exist before anything is loaded.
_Y = "/scratch/users/hshunt/spin_gnn/sudoku_n48_determinacy_Y_2026-11-14b"
_CELL = "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_chan"
Y_DIRS = {
    0: f"{_Y}/t00_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s0/{_CELL}",
    1: f"{_Y}/t01_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s1/{_CELL}_s01",
    2: f"{_Y}/t02_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s2/{_CELL}_s02",
}
FIXED_DIRS = {
    s: "/scratch/users/hshunt/spin_gnn/runs/stageloop_probe_2026-11-15%s_fixed"
       % ("" if s == 0 else f"_s{s}")
    for s in (0, 1, 2)
}

MAX_LEN = 512
BATCH = 32
N_BUILD = 64            # the batch is the first BATCH of this, unshuffled
T_ITERS = 16
FIX_TARGET = 0.10       # the mild setting, the one that cannot be called "too strong"
OUT_DIR = "/scratch/users/hshunt/spin_gnn/runs/stageloop_direction_2026-11-15"

PREDICTION = {
    "cos_delta_above": 0.80,
    "cos_delta_minus_cos_dh_above": 0.30,
    "cos_h_below_cos_delta": True,
    "r1_rel_tol": 0.20,
    "r1_spread_over_t_above": 1.5,
    "falsifier_cos_gap_below": 0.10,
}

SMOKE = os.environ.get("DIR_SMOKE") == "1"
if SMOKE:
    print("=" * 72)
    print("DIR_SMOKE=1 -- SHAPES ONLY. Sizes are wrong on purpose and no number\n"
          "below is a measurement of anything.")
    print("=" * 72, flush=True)
    N_BUILD, BATCH = 8, 4
    OUT_DIR += "_smoke"

os.makedirs(OUT_DIR, exist_ok=True)
dev = "cpu"             # one forward pass per mode: a GPU would be idle waiting
torch.set_grad_enabled(False)
print(f"device={dev}  batch={BATCH}  out={OUT_DIR}", flush=True)


def stats(v, mask):
    """cm and mean pairwise cosine of (B, L, d) `v` over the valid positions.

    Returned per batch element, so the spread across sequences is visible rather
    than averaged away. The cosine uses
    ||sum_p u_p||^2 = L + sum_{p != q} <u_p, u_q> on unit vectors, which is the
    L x L Gram matrix's off-diagonal mean without ever forming the Gram matrix.
    """
    out_cm, out_cos = [], []
    for b in range(v.shape[0]):
        w = v[b][mask[b]].double()
        n = w.shape[0]
        norms = w.norm(dim=-1)
        assert n > 1, "a sequence with one valid position cannot have a cosine"
        if float(norms.min()) <= 0:
            out_cm.append(float("nan"))
            out_cos.append(float("nan"))
            continue
        out_cm.append(float(w.mean(0).norm() / norms.mean()))
        u = w / norms[:, None]
        out_cos.append(float((u.sum(0).pow(2).sum() - n) / (n * (n - 1))))
    return float(np.mean(out_cm)), float(np.mean(out_cos))


def build_body(seed):
    """That seed's stage Y body with the loop parameter added and set to zero."""
    d = Y_DIRS[seed]
    assert os.path.isdir(d), d
    with open(os.path.join(d, "config.json")) as fh:
        config = json.load(fh)
    assert config.get("determinacy_channel"), "no channel in this checkpoint"
    assert config["max_len"] == MAX_LEN, config["max_len"]
    state = torch.load(os.path.join(d, "model.pt"), map_location="cpu",
                       weights_only=True)
    m = SATReasoningTransformer(
        VOCAB.size, d_model=320, num_layers=T_ITERS, nhead=4, max_len=MAX_LEN,
        dim_feedforward=1024, norm_first=True, arch_mode="tied_no_mlp",
        n_blocks=1, determinacy_channel=True,
        node_offset=VOCAB.node_offset, n_nodes=VOCAB.n_nodes,
        determinacy_feedback=True)
    assert m.pos_embed.shape[0] == MAX_LEN
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert list(missing) == ["loop_scale"], list(missing)
    assert not list(unexpected), list(unexpected)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == SP.N_TRAINABLE + 1, "not the registered cell: %s" % f"{n:,}"
    assert float(m.det_scale.norm()) > 0.05, "dead channel, probe is vacuous"
    assert float(m.loop_scale.abs()) == 0.0
    # the hook below counts on ONE shared block applied T_ITERS times: if the
    # cell ever grew a second block the hook's call order would stop meaning t
    assert len(m.encoder.layers) == 1, len(m.encoder.layers)
    assert m.n_iters == T_ITERS, m.n_iters
    return m.eval()


def solved_c(seed):
    """88's calibrated constant for the 0.10 target, READ from the artifact."""
    p = os.path.join(FIXED_DIRS[seed], "loop_summary.json")
    assert os.path.isfile(p), p
    with open(p) as fh:
        row = json.load(fh)["per_arm"]["fix%.2f L1" % FIX_TARGET]
    c = float(row["loop_scale"])
    assert c > 0, c
    return c


def run_mode(model, x, mask, c):
    """One forward at `loop_scale = c`, with the per-application geometry.

    The hook fires once per application of the shared block, in order, and sees
    the hidden state BEFORE the injection (which happens outside the module), so
    `delta_t` recomputed from it is exactly the vector the loop adds at that t.
    It is recomputed at loop_scale = 1 because cm and cos do not depend on c and
    the ratio at any c is c * r1_t.
    """
    caught = []
    h = model.encoder.layers[0]
    handle = h.register_forward_hook(
        lambda mod, args, out: caught.append((args[0].detach(), out.detach())))
    model.loop_scale.fill_(c)
    try:
        model(x, valid_mask=mask)
    finally:
        handle.remove()
        model.loop_scale.fill_(0.0)
    assert len(caught) == T_ITERS, len(caught)

    rows = []
    for t, (h_in, h_out) in enumerate(caught):
        if t >= T_ITERS - 1:
            break                      # the loop never fires after the last one
        model.loop_scale.fill_(1.0)
        delta = model.determinacy_feedback_delta(h_out)
        model.loop_scale.fill_(0.0)
        cm_d, cos_d = stats(delta, mask)
        cm_u, cos_u = stats(h_out - h_in, mask)
        cm_h, cos_h = stats(h_out, mask)
        r1 = float(delta[mask].abs().mean() / h_out[mask].abs().mean())
        rows.append({"t": t, "r1": r1,
                     "cm_delta": cm_d, "cos_delta": cos_d,
                     "cm_dh": cm_u, "cos_dh": cos_u,
                     "cm_h": cm_h, "cos_h": cos_h})
    return rows


VOCAB = None
summary = {"prediction": PREDICTION, "per_seed": {}}
for seed in (0, 1, 2):
    data_seed = seed + 777
    cfg = SudokuReasoningConfig(n_nodes=48, degree=5, n_colors=4,
                                shuffle_adj=True, fall=True, seed=data_seed)
    VOCAB = build_vocab(cfg)
    print(f"\n{'=' * 72}\nseed {seed}   data seed {data_seed}   "
          f"{Y_DIRS[seed].split('/')[-2]}", flush=True)
    inst = build_instances(cfg, N_BUILD, data_seed)
    tokens = [encode_instance(adj, trace, VOCAB) for adj, _, _, trace in inst]
    ds = SATTokenDataset(tokens, max_len=MAX_LEN)
    batch = next(iter(iterate_token_batches(ds, list(range(len(ds))), BATCH,
                                            shuffle=False,
                                            collate=make_token_batch)))
    x, _y, mask = (t.to(dev) for t in batch[:3])
    lens = mask.sum(1)
    print(f"  batch {tuple(x.shape)}  valid positions "
          f"{int(lens.min())}-{int(lens.max())}  "
          f"random-direction baseline cm = {1 / float(lens.double().mean().sqrt()):.4f}",
          flush=True)

    model = build_body(seed)
    c = solved_c(seed)
    per_mode = {}
    for name, cc in (("unit", 1.0), ("open", c), ("closed", 0.0)):
        rows = run_mode(model, x, mask, cc)
        per_mode[name] = rows
        print(f"\n  mode {name}  (loop_scale = {cc:.4f})", flush=True)
        print(f"  {'t':>3} {'r1':>8} {'cm(d)':>7} {'cos(d)':>7} "
              f"{'cm(dh)':>7} {'cos(dh)':>7} {'cm(h)':>7} {'cos(h)':>7}",
              flush=True)
        for r in rows:
            print(f"  {r['t']:3d} {r['r1']:8.4f} {r['cm_delta']:7.4f} "
                  f"{r['cos_delta']:7.4f} {r['cm_dh']:7.4f} {r['cos_dh']:7.4f} "
                  f"{r['cm_h']:7.4f} {r['cos_h']:7.4f}", flush=True)

    r1_last = per_mode["unit"][-1]["r1"]
    r1s = [r["r1"] for r in per_mode["unit"]]
    implied = FIX_TARGET / c
    summary["per_seed"][str(seed)] = {
        "c_solved": c, "r1_implied_by_calibration": implied,
        "r1_unit_last": r1_last,
        "r1_rel_diff": abs(r1_last - implied) / implied,
        "r1_spread_over_t": max(r1s) / min(r1s),
        "val_positions": [int(lens.min()), int(lens.max())],
        "modes": per_mode,
    }
    print(f"\n  calibration: 88 solved c = {c:.6f} for a {FIX_TARGET:.2f} "
          f"target, implying r1 = {implied:.4f}", flush=True)
    print(f"  this batch, unit mode, t=14: r1 = {r1_last:.4f}  "
          f"(relative difference {abs(r1_last - implied) / implied:.3f})",
          flush=True)
    print(f"  r1 over t: {min(r1s):.4f} to {max(r1s):.4f}, "
          f"ratio {max(r1s) / min(r1s):.2f}", flush=True)

# ---------------------------------------------------------------------------
# the registered predictions, read back against the table. PRINTED, NOT ENFORCED
# ---------------------------------------------------------------------------
print(f"\n{'=' * 72}\nREGISTERED PREDICTIONS, read back\n{'=' * 72}", flush=True)
for key, mode in (("closed", "closed"), ("open", "open")):
    cosd = [min(r["cos_delta"] for r in summary["per_seed"][str(s)]["modes"][mode])
            for s in (0, 1, 2)]
    cosu = [max(r["cos_dh"] for r in summary["per_seed"][str(s)]["modes"][mode])
            for s in (0, 1, 2)]
    cosh = [max(r["cos_h"] for r in summary["per_seed"][str(s)]["modes"][mode])
            for s in (0, 1, 2)]
    gap = [d - u for d, u in zip(cosd, cosu)]
    print(f"\n  {mode} mode, worst application per seed:")
    print(f"    min cos(delta)  {cosd[0]:.4f} {cosd[1]:.4f} {cosd[2]:.4f}   "
          f"predicted > {PREDICTION['cos_delta_above']}: "
          f"{sum(v > PREDICTION['cos_delta_above'] for v in cosd)}/3")
    print(f"    max cos(dh)     {cosu[0]:.4f} {cosu[1]:.4f} {cosu[2]:.4f}")
    print(f"    gap             {gap[0]:.4f} {gap[1]:.4f} {gap[2]:.4f}   "
          f"predicted > {PREDICTION['cos_delta_minus_cos_dh_above']}: "
          f"{sum(v > PREDICTION['cos_delta_minus_cos_dh_above'] for v in gap)}/3"
          f"   falsifier < {PREDICTION['falsifier_cos_gap_below']}: "
          f"{sum(v < PREDICTION['falsifier_cos_gap_below'] for v in gap)}/3")
    print(f"    max cos(h)      {cosh[0]:.4f} {cosh[1]:.4f} {cosh[2]:.4f}   "
          f"predicted below cos(delta): "
          f"{sum(h < d for h, d in zip(cosh, cosd))}/3")

rel = [summary["per_seed"][str(s)]["r1_rel_diff"] for s in (0, 1, 2)]
spr = [summary["per_seed"][str(s)]["r1_spread_over_t"] for s in (0, 1, 2)]
print(f"\n  r1 vs calibration  {rel[0]:.3f} {rel[1]:.3f} {rel[2]:.3f}   "
      f"predicted < {PREDICTION['r1_rel_tol']}: "
      f"{sum(v < PREDICTION['r1_rel_tol'] for v in rel)}/3 (loose on purpose)")
print(f"  r1 spread over t   {spr[0]:.2f} {spr[1]:.2f} {spr[2]:.2f}   "
      f"predicted > {PREDICTION['r1_spread_over_t_above']}: "
      f"{sum(v > PREDICTION['r1_spread_over_t_above'] for v in spr)}/3")

with open(os.path.join(OUT_DIR, "direction_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"\nwrote {OUT_DIR}/direction_summary.json", flush=True)
print("NO VERDICT. This is geometry on three bodies; whether the geometry is "
      "what\ndestroyed them in finding 88 is an inference a forward pass "
      "cannot make.", flush=True)
