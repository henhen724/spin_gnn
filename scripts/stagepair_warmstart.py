"""Is stage PAIR's contrast learnable AT ALL, by a body that already solves?

NOT A RESULT AND NOT A STAGE. No bar is read, no checkpoint is kept, nothing
here goes in a deck, and nothing it finds changes the registered term. It is the
probe STATUS named: "a single arm at lambda=1 starting from a trained stage-Y
checkpoint -- that asks whether the contrast is learnable at all, separately
from whether it can be learned from scratch."

WHY IT IS WORTH A GPU EVEN THOUGH 52441848 IS ALREADY RUNNING. That job asks why
the from-scratch run fails and its answer, whatever it is, leaves this question
untouched. Here the body ALREADY solves (stage Y, 50 epochs, val accuracy
0.4830) and the channel is ALREADY live (det_scale is nonzero, so dg/dh =
det_scale * E_v is not identically zero and the margin can actually reach the
body). Every excuse available to a fresh init is gone. So:

  - if the hinge falls from 1.0 here, the target is reachable and the
    from-scratch failure is an optimization problem, which is what warming the
    weight in or lowering lambda would address;

  - if the hinge does NOT fall from 1.0 even here, the contrast is not
    representable by this cell at all, every one of those three design changes
    is pointless, and the finding is much larger than a schedule.

THE SECOND ARM IS NOT OPTIONAL. Restarting Adam at lr 1e-3 on a converged model
can wreck it on its own, and without lambda=0 from the SAME checkpoint on the
SAME fresh data any damage would get misattributed to the margin. That is the
error this project has made before -- reading a number without its control.

FRESH DATA, deliberately: seed SEED + 777, so these instances and these pairs
are ones the stage-Y run never saw in its 50 epochs. "Learnable" has to mean
more than "memorizable", and the main nll on unseen instances doubles as the
check that the main task is still intact.

WHAT IS NOT CONTROLLED. lr is 1e-3 because that is what stage Y trained at;
restarting the optimizer state at a converged point is still a perturbation and
the lambda=0 arm is how its size is read, not a claim it is zero.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/users/hshunt/spin_gnn")

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sat_dataset2 import (SATTokenDataset, iterate_token_batches,
                                   make_token_batch_pair)
from spin_gnn.sat_transformer2 import SATReasoningTransformer, sat2_nll_loss
from spin_gnn.sudoku_determinacy import (determinacy_loss_from_channel,
                                         determinacy_targets)
from spin_gnn.sudoku_pair import build_pair_index
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab

CKPT = ("/scratch/users/hshunt/spin_gnn/sudoku_n48_determinacy_Y_2026-11-14b/"
        "t00_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s0/"
        "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_chan/"
        "model.pt")

N_INSTANCES = 30_000
BATCH = 32
STEPS = 3_000
LOG_EVERY = 125
SEED = 0
DATA_SEED = SEED + 777
MARGIN = 1.0
MAX_LEN = 512
LR = 1e-3
ARMS = [(0.0, "lambda=0 control"), (1.0, "lambda=1 margin")]

dev = "cuda" if torch.cuda.is_available() else "cpu"
assert os.path.exists(CKPT), CKPT
print(f"device={dev}  steps={STEPS:,}  data_seed={DATA_SEED}")
print(f"ckpt {CKPT}")

sudoku_cfg = SudokuReasoningConfig(n_nodes=48, degree=5, n_colors=4,
                                   shuffle_adj=True, fall=True, seed=DATA_SEED)
vocab = build_vocab(sudoku_cfg)
inst = build_instances(sudoku_cfg, N_INSTANCES, sudoku_cfg.seed)
tokens = [encode_instance(adj, trace, vocab) for adj, _, _, trace in inst]
aux = [determinacy_targets(adj, trace, vocab, tok)
       for (adj, _c, _h, trace), tok in zip(inst, tokens)]
pair = build_pair_index(inst, tokens, vocab, sudoku_cfg.seed + 500_000_000)
n_cases = sum(1 for p in pair if p[0] >= 0)
print(f"pair yield {n_cases / len(pair):.4f} ({n_cases:,}/{len(pair):,})")

ds = SATTokenDataset(tokens, max_len=MAX_LEN, aux=aux, pair=pair)
idx = list(range(len(ds)))
print(f"max_len={MAX_LEN}  dataset={len(ds):,} of {len(tokens):,}", flush=True)

state = torch.load(CKPT, map_location="cpu", weights_only=True)


def warm():
    m = SATReasoningTransformer(
        vocab.size, d_model=320, num_layers=16, nhead=4, max_len=MAX_LEN,
        dim_feedforward=1024, norm_first=True, arch_mode="tied_no_mlp",
        n_blocks=1, determinacy_channel=True,
        node_offset=vocab.node_offset, n_nodes=vocab.n_nodes)
    # strict=True on purpose: a stage-Y state dict and a stage-PAIR one are the
    # same keys and the same shapes, so anything that does not load exactly is a
    # different cell and the probe would be measuring the wrong model.
    m.load_state_dict(state, strict=True)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == 614_138, ("not the registered cell: %s trainable" % f"{n:,}")
    m = m.to(dev)
    # The whole premise of the probe: the channel is LIVE, so the margin's
    # gradient can reach the body. At det_scale = 0 it provably cannot, which is
    # what made smoke check 5 vacuous.
    s = float(m.det_scale.norm())
    assert s > 0.05, ("det_scale is %.3e -- this checkpoint's channel is dead "
                      "and the probe would be vacuous" % s)
    print(f"loaded: |det_scale| = {s:.4f}", flush=True)
    return m


def body_grad_norm(body):
    tot = 0.0
    for p in body:
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return tot ** 0.5


def isolated(model, body, x, y, mask, pr):
    model.zero_grad(set_to_none=True)
    pb = float("nan")
    sel = (pr[:, 0] >= 0).nonzero(as_tuple=True)[0]
    if sel.numel():
        _z, g = model(x, valid_mask=mask, return_determinacy=True)
        xb = x.clone()
        xb[sel, pr[sel, 0]] = pr[sel, 1]
        _zb, gb = model(xb, valid_mask=mask, return_determinacy=True)
        gap = g[sel, pr[sel, 2], pr[sel, 3]] - gb[sel, pr[sel, 2], pr[sel, 3]]
        torch.clamp(MARGIN - gap, min=0).mean().backward()
        pb = body_grad_norm(body)
    model.zero_grad(set_to_none=True)
    z, _g = model(x, valid_mask=mask, return_determinacy=True)
    sat2_nll_loss(z, y, mask).backward()
    mb = body_grad_norm(body)
    model.zero_grad(set_to_none=True)
    return pb, mb


def run(pair_w, label):
    print(f"\n=== {label}: pair_weight={pair_w:g}, from stage Y " + "=" * 20,
          flush=True)
    torch.manual_seed(SEED)
    model = warm()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=LR)
    rng = np.random.default_rng(SEED)
    body = [p for n, p in model.named_parameters()
            if p.requires_grad and n != "det_scale"]
    print(f"{'step':>5} {'main':>7} {'det_bce':>8} {'hinge':>7} {'gap':>8} "
          f"{'sat':>5} {'|s|':>8} {'dPair':>10} {'dMain':>10} {'ratio':>7}",
          flush=True)
    step, trace = 0, []
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
            hinge_v = gap_v = sat_v = float("nan")
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
                    # The fraction of cases already clearing the margin. This is
                    # the number the question turns on: a hinge mean that falls
                    # because a few cases went far positive is not the same
                    # thing as a contrast the model has actually learned.
                    sat_v = float((gap >= MARGIN).float().mean())
            do_log = step % LOG_EVERY == 0
            opt.zero_grad()
            loss.backward()
            opt.step()
            if do_log:
                pb, mb = isolated(model, body, x, y, mask, pr)
                r = pb / mb if mb else float("inf")
                trace.append((step, float(main), hinge_v, gap_v))
                print(f"{step:5d} {float(main):7.4f} "
                      f"{float(det) if det is not None else float('nan'):8.4f} "
                      f"{hinge_v:7.4f} {gap_v:+8.4f} {sat_v:5.2f} "
                      f"{float(model.det_scale.norm()):8.4f} "
                      f"{pb:10.3e} {mb:10.3e} {r:7.3f}", flush=True)
            step += 1
    return trace


out = {}
for w, name in ARMS:
    out[name] = run(w, name)

print("\n" + "=" * 72)
print("stage Y's own final train nll was 1.5610 (basin) at val accuracy 0.4830.")
for _w, name in ARMS:
    t = out[name]
    print(f"  {name:<18} main {t[0][1]:.4f} -> {t[-1][1]:.4f}   "
          f"hinge {t[0][2]:.4f} -> {t[-1][2]:.4f}   "
          f"gap {t[0][3]:+.4f} -> {t[-1][3]:+.4f}")
print("No verdict is printed. The question is whether the hinge left 1.0 at all,")
print("and what the lambda=0 arm did to `main` over the same 3,000 steps.")
