"""How well must the body already solve before the margin is satisfiable?

NOT A RESULT AND NOT A STAGE. No bar is read, no checkpoint is kept, nothing
goes in a deck. It does not register a schedule; it measures the number a
schedule would need.

WHAT THE TWO JOBS BEFORE IT ESTABLISHED.

52441848, three arms from one init on one data order, 7,500 steps each:

    control       lambda=0     main 4.31 -> 2.4740     learns
    registered    lambda=1     main 4.31 -> 3.9509     uniform (log 58 = 4.060)
    forward-only  lambda=1e-6  main 4.31 -> 2.2635     learns

So the second forward pass and its doubled dropout are ACQUITTED: paying for
both while exerting no force costs nothing (2.2635, in fact slightly better than
the control). The force is the cause. The registered arm is flat at 3.93-3.98
from step 1,000 onward, which dates the collapse to the first 1,000 steps.

52441962, from the stage-Y seed-0 checkpoint on 30,000 unseen instances:

    lambda=0   main 1.6117 -> 1.5865    (restarting Adam at 1e-3 is benign)
    lambda=1   main 1.6117 -> 1.6688    hinge 1.1006 -> 0.4336, gap -0.085 -> +0.575

So the contrast IS learnable by this exact cell with zero new parameters, at a
cost of about 0.06 of train nll, and the main task survives.

AND THE HYPOTHESIS THOSE TWO KILLED. The prediction was that the margin's body
gradient crosses the main loss's during training and that this crossing is the
mechanism. It crosses, but the crossing is not the mechanism: the lambda=0
control runs the whole way at ratio about 2 and learns fine, and the warm-start
lambda=1 arm runs at ratio 3-16 and learns fine WHILE satisfying the margin. A
ratio above 1 is normal. What distinguishes the failing arm is the ABSOLUTE
magnitude -- dPair/body reaching 2.5e+02 to 3.0e+03 against the control's 1.4-2.4
at the same steps, with dMain/body dragged up to 2.6-190 against the control's
0.8 -- on a body the margin has pushed somewhere the main loss never leaves.

THE QUESTION THIS LEAVES, and the only one it asks. The target is reachable from
main = 1.61 and unreachable from main = 4.31. Everything between is unmeasured,
and that interval is exactly what any "warm the weight in" schedule is a guess
about. So: train ONE lambda=0 trunk, and at several points along it fork a copy
and give the fork 1,500 steps at lambda=1. Whether the fork's hinge falls or its
main collapses, at each of three values of main, is the number.

The fork carries the OPTIMIZER STATE as well as the weights, because a schedule
that ramps lambda inside one run would, and a fork with fresh Adam moments would
be answering a question nobody is going to ask.
"""

import copy
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

N_INSTANCES = 60_000
BATCH = 32
TRUNK_STEPS = 15_000
FORK_AT = (2_500, 7_500, 15_000)
FORK_STEPS = 1_500
LOG_EVERY = 250
SEED = 0
MARGIN = 1.0
MAX_LEN = 512
LR = 1e-3

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}  trunk={TRUNK_STEPS:,}  forks at {FORK_AT} x {FORK_STEPS:,}")

sudoku_cfg = SudokuReasoningConfig(n_nodes=48, degree=5, n_colors=4,
                                   shuffle_adj=True, fall=True, seed=SEED)
vocab = build_vocab(sudoku_cfg)
inst = build_instances(sudoku_cfg, N_INSTANCES, sudoku_cfg.seed)
tokens = [encode_instance(adj, trace, vocab) for adj, _, _, trace in inst]
aux = [determinacy_targets(adj, trace, vocab, tok)
       for (adj, _c, _h, trace), tok in zip(inst, tokens)]
pair = build_pair_index(inst, tokens, vocab, sudoku_cfg.seed + 500_000_000)
ds = SATTokenDataset(tokens, max_len=MAX_LEN, aux=aux, pair=pair)
idx = list(range(len(ds)))
print(f"max_len={MAX_LEN}  dataset={len(ds):,}", flush=True)


def fresh():
    torch.manual_seed(SEED)
    m = SATReasoningTransformer(
        vocab.size, d_model=320, num_layers=16, nhead=4, max_len=MAX_LEN,
        dim_feedforward=1024, norm_first=True, arch_mode="tied_no_mlp",
        n_blocks=1, determinacy_channel=True,
        node_offset=vocab.node_offset, n_nodes=vocab.n_nodes).to(dev)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == 614_138, ("not the registered cell: %s trainable" % f"{n:,}")
    return m


def body_of(model):
    return [p for n, p in model.named_parameters()
            if p.requires_grad and n != "det_scale"]


def grad_norm(ps):
    t = 0.0
    for p in ps:
        if p.grad is not None:
            t += float(p.grad.detach().pow(2).sum())
    return t ** 0.5


def step_once(model, opt, batch, pair_w):
    """One optimizer step. Returns the scalars worth printing."""
    x, y, mask = (t.to(dev) for t in batch[:3])
    pr = batch[5].to(dev)
    logits, g = model(x, valid_mask=mask, return_determinacy=True)
    main = sat2_nll_loss(logits, y, mask)
    loss = main
    det = determinacy_loss_from_channel(g, batch[3].to(dev), batch[4].to(dev))
    if det is not None:
        loss = loss + det
    hinge_v = gap_v = sat_v = float("nan")
    if pair_w > 0:
        sel = (pr[:, 0] >= 0).nonzero(as_tuple=True)[0]
        if sel.numel():
            xb = x.clone()
            xb[sel, pr[sel, 0]] = pr[sel, 1]
            _zb, gb = model(xb, valid_mask=mask, return_determinacy=True)
            gap = (g[sel, pr[sel, 2], pr[sel, 3]]
                   - gb[sel, pr[sel, 2], pr[sel, 3]])
            hinge = torch.clamp(MARGIN - gap, min=0)
            loss = loss + pair_w * hinge.mean()
            hinge_v, gap_v = float(hinge.mean()), float(gap.mean())
            sat_v = float((gap >= MARGIN).float().mean())
    opt.zero_grad()
    loss.backward()
    gb_now = grad_norm(body_of(model))
    opt.step()
    return (float(main), float(det) if det is not None else float("nan"),
            hinge_v, gap_v, sat_v, gb_now)


def fork(model, opt, start_step, main_at_fork):
    """1,500 steps at lambda=1 from a copy of the trunk, weights AND moments."""
    m = copy.deepcopy(model)
    o = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR)
    o.load_state_dict(copy.deepcopy(opt.state_dict()))
    print(f"\n--- fork at trunk step {start_step:,}, main there = "
          f"{main_at_fork:.4f}, now lambda=1 " + "-" * 12, flush=True)
    print(f"{'k':>5} {'main':>7} {'det_bce':>8} {'hinge':>7} {'gap':>8} "
          f"{'sat':>5} {'|s|':>8} {'|dbody|':>10}", flush=True)
    rng = np.random.default_rng(SEED + 31 + start_step)
    k = 0
    last = None
    while k < FORK_STEPS:
        for batch in iterate_token_batches(ds, idx, BATCH, rng=rng,
                                           collate=make_token_batch_pair):
            if k >= FORK_STEPS:
                break
            r = step_once(m, o, batch, 1.0)
            last = r
            if k % LOG_EVERY == 0:
                print(f"{k:5d} {r[0]:7.4f} {r[1]:8.4f} {r[2]:7.4f} {r[3]:+8.4f} "
                      f"{r[4]:5.2f} {float(m.det_scale.norm()):8.4f} "
                      f"{r[5]:10.3e}", flush=True)
            k += 1
    print(f"{'end':>5} {last[0]:7.4f} {last[1]:8.4f} {last[2]:7.4f} "
          f"{last[3]:+8.4f} {last[4]:5.2f}", flush=True)
    return {"at": start_step, "main_before": main_at_fork,
            "main_after": last[0], "hinge_after": last[2],
            "gap_after": last[3]}


model = fresh()
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
rng = np.random.default_rng(SEED)
print(f"\n=== trunk: lambda=0 " + "=" * 42, flush=True)
print(f"{'step':>6} {'main':>7} {'det_bce':>8} {'|s|':>8} {'|dbody|':>10}",
      flush=True)

results, step, main_now = [], 0, float("nan")
pending = list(FORK_AT)
while step < TRUNK_STEPS:
    for batch in iterate_token_batches(ds, idx, BATCH, rng=rng,
                                       collate=make_token_batch_pair):
        if step >= TRUNK_STEPS:
            break
        r = step_once(model, opt, batch, 0.0)
        main_now = r[0]
        if step % 2_500 == 0:
            print(f"{step:6d} {r[0]:7.4f} {r[1]:8.4f} "
                  f"{float(model.det_scale.norm()):8.4f} {r[5]:10.3e}",
                  flush=True)
        step += 1
        if pending and step == pending[0]:
            # Forked AFTER the step, so `step` is the number of lambda=0 steps
            # the fork has had. The trunk is untouched by the fork and carries
            # on to the next point.
            results.append(fork(model, opt, pending.pop(0), main_now))
            print(f"\n=== trunk resumes at {step:,} " + "=" * 34, flush=True)

print("\n" + "=" * 72)
print("fork results -- lambda switched from 0 to 1 after N steps of lambda=0:")
print(f"{'N':>7} {'main before':>12} {'main after':>11} {'hinge after':>12} "
      f"{'gap after':>10}")
for r in results:
    print(f"{r['at']:7,d} {r['main_before']:12.4f} {r['main_after']:11.4f} "
          f"{r['hinge_after']:12.4f} {r['gap_after']:+10.4f}")
print("\nReference points, neither of them from this job: from a fresh init")
print("(main 4.31) the margin pins main at 3.95 and the hinge at 1.00 forever;")
print("from stage Y (main 1.61) the hinge reaches 0.43 and main holds at 1.67.")
print("No verdict is printed.")
