"""Why stage PAIR's array (52423780) sat at uniform output.

NOT A RESULT AND NOT A STAGE. No bar is read and no verdict is made. Nothing
this prints goes in a deck.

THE OBSERVATION. 52423780_[0-2], all three seeds, at epoch 5:

    train_nll=3.9540  val_accuracy=0.0312  det_bce=0.7085  val_det_auroc=0.5854
    pair_hinge=1.0001  gap=+0.0001  active=0.999

against the stage-Y control (52348732_0) -- same data (byte-identical
determinacy-target counts), same cell, same 614,138 parameters -- at epoch 0:

    train_nll=2.6375  val_accuracy=0.4328  det_bce=0.5636  val_det_auroc=0.7752

log(58) = 4.060 and the majority-class baseline is 0.0306, so the pair arm is at
uniform output after 45,000 steps while the control was solving after 7,500.
det_bce 0.709 is log(2): the channel is at chance too.

ALREADY RULED OUT, so this does not re-test it: the optimizer step (trainer
676-678); det_scale frozen (9.67e-02 after the preflight's 20 steps); exploded
embeddings (node block 124.6 = sqrt(48*320), its N(0,1) init); exploded g
(|g|max 1.5-2.3); the data; the CLI (differs only in the two pair flags).

ALSO RULED OUT, by smoke assertion 8: the margin out-pulling the main loss on
the shared body AT A FRESH INIT with a live channel. At |s| = 0.429 that ratio
is 0.393. Which leaves the question this script exists to answer -- what the
ratio does DURING training, where nobody has measured it.

THE HYPOTHESIS. The hinge is unsatisfiable early. Clean and broken prompts
differ by one token in about 400, so a randomly initialized body maps them to
nearly the same representation, the gap stays at zero, and max(0, m - gap) is
therefore linear-and-active essentially always: a CONSTANT-magnitude force that
never decays (active 99.9% across 45,000 steps) whose useful direction is close
to null. Predictions: (1) the isolated body-gradient ratio RISES THROUGH 1
during training even though it starts at 0.393; (2) an arm that pays for the
second forward pass but exerts no force learns normally, which would acquit the
extra forward and its doubled dropout.

THE THREE ARMS. lambda = 0 is the control. lambda = 1 is the term as
registered. lambda = 1e-6 still runs the second forward, still consumes its
dropout, and still clones the batch -- so it differs from the control in
everything except the force, and from the registered arm in nothing except the
force. That is the arm that separates "the margin's push" from "the cost of
looking twice".

WHAT IS MEASURED EVERY LOG_EVERY STEPS, on the CURRENT parameters rather than at
init: the two terms' gradients on the shared body SEPARATELY, by backward()ing
each alone. That ratio is the number smoke assertion 8 can only give at step 0.

BUDGET. 7,500 steps is the control's real epoch 0, so `main` is directly
comparable to its 2.6375. 60,000 instances rather than the 20,000 of 52430713,
so 7,500 steps is about four passes and not twelve. 52430713's 900 steps was
12% of an epoch -- too few to date an onset, and its auto-verdict line used a
threshold chosen for a budget that could not support it. That line is gone;
this script prints the tables and the final numbers and draws no conclusion.
"""

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
STEPS = 7_500          # the control's real epoch 0, so `main` is comparable
LOG_EVERY = 250
SEED = 0
MARGIN = 1.0
ARMS = [(0.0, "control"), (1.0, "registered"), (1e-6, "forward-only")]

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}  instances={N_INSTANCES:,}  batch={BATCH}  steps={STEPS:,}")

# fall=True as well as shuffle_adj: --fall changes the GRAPH ensemble, so a
# diagnostic without it would not be reproducing the array's instances.
sudoku_cfg = SudokuReasoningConfig(n_nodes=48, degree=5, n_colors=4,
                                   shuffle_adj=True, fall=True, seed=SEED)
vocab = build_vocab(sudoku_cfg)
inst = build_instances(sudoku_cfg, N_INSTANCES, sudoku_cfg.seed)
tokens = [encode_instance(adj, trace, vocab) for adj, _, _, trace in inst]
aux = [determinacy_targets(adj, trace, vocab, tok)
       for (adj, _c, _h, trace), tok in zip(inst, tokens)]
pair = build_pair_index(inst, tokens, vocab, sudoku_cfg.seed + 500_000_000)
n_cases = sum(1 for p in pair if p[0] >= 0)
print(f"pair yield {n_cases / len(pair):.4f} ({n_cases:,}/{len(pair):,})")

# 512, the array's own max_len, and not a value sized to this subset. The
# positional table is trainable, so max_len is a term in the parameter count --
# at any other value the body is a different size and `main` is no longer
# comparable to the control's 2.6375. max_len DROPS over-long sequences rather
# than truncating, but the array drops the same ones, so that is shared
# behaviour and not a defect of the diagnostic.
ml = 512
ds = SATTokenDataset(tokens, max_len=ml, aux=aux, pair=pair)
idx = list(range(len(ds)))
print(f"max_len={ml}  dataset={len(ds):,} of {len(tokens):,} "
      f"({len(tokens) - len(ds):,} dropped as over-long)", flush=True)

NODE_LO = vocab.node_offset
NODE_HI = vocab.node_offset + vocab.n_nodes


def fresh():
    torch.manual_seed(SEED)
    m = SATReasoningTransformer(
        vocab.size, d_model=320, num_layers=16, nhead=4, max_len=ml,
        dim_feedforward=1024, norm_first=True, arch_mode="tied_no_mlp",
        n_blocks=1, determinacy_channel=True,
        node_offset=vocab.node_offset, n_nodes=vocab.n_nodes).to(dev)
    # The literal, because the whole comparison is to a 614,138-parameter cell
    # and a silent change of cell would make every number here meaningless.
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == 614_138, ("not the registered cell: %s trainable" % f"{n:,}")
    return m


def body_grad_norm(model, body):
    tot = 0.0
    for p in body:
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return tot ** 0.5


def isolated_ratio(model, body, x, y, mask, pr):
    """||dL_pair/dbody|| and ||dL_main/dbody|| on the CURRENT parameters.

    Each term is backward()ed alone, which is the only way to attribute the
    body's gradient: in the training step the three are summed before backward
    and what the optimizer sees is a total nobody can decompose afterwards.
    Returns (pair_body, main_body) with pair_body nan when the batch carries no
    case to score.
    """
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
        pb = body_grad_norm(model, body)
    model.zero_grad(set_to_none=True)
    z, _g = model(x, valid_mask=mask, return_determinacy=True)
    sat2_nll_loss(z, y, mask).backward()
    mb = body_grad_norm(model, body)
    model.zero_grad(set_to_none=True)
    return pb, mb


def run(pair_w, label):
    print(f"\n=== {label}: pair_weight={pair_w:g} " + "=" * 34, flush=True)
    model = fresh()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=1e-3)
    # Same seed in every arm, so all three see the same data in the same order
    # and a divergence is the term's and not the shuffle's.
    rng = np.random.default_rng(SEED)
    body = [p for n, p in model.named_parameters()
            if p.requires_grad and n != "det_scale"]
    print(f"{'step':>5} {'main':>7} {'det_bce':>8} {'hinge':>7} {'gap':>8} "
          f"{'|s|':>8} {'g_absmax':>9} {'dPair/body':>11} {'dMain/body':>11} "
          f"{'ratio':>7}", flush=True)
    step = 0
    last_main = float("nan")
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
            apos, atgt = batch[3].to(dev), batch[4].to(dev)
            det = determinacy_loss_from_channel(g, apos, atgt)
            if det is not None:
                loss = loss + det
            hinge_v = gap_v = float("nan")
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
            do_log = step % LOG_EVERY == 0
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_main = float(main)
            if do_log:
                # After the step, so the ratio describes the parameters the next
                # step will actually use. Costs two extra backwards every
                # LOG_EVERY steps and is not in the optimizer's path.
                pb, mb = isolated_ratio(model, body, x, y, mask, pr)
                r = pb / mb if mb else float("inf")
                print(f"{step:5d} {float(main):7.4f} "
                      f"{float(det) if det is not None else float('nan'):8.4f} "
                      f"{hinge_v:7.4f} {gap_v:+8.4f} "
                      f"{float(model.det_scale.norm()):8.3e} "
                      f"{float(g.abs().max()):9.3f} "
                      f"{pb:11.3e} {mb:11.3e} {r:7.3f}", flush=True)
            step += 1
    return last_main


final = {}
for w, name in ARMS:
    final[name] = run(w, name)

print("\n" + "=" * 70)
print(f"main nll after {STEPS:,} steps (the control's real epoch 0 was 2.6375):")
for _w, name in ARMS:
    print(f"  {name:<14} {final[name]:.4f}")
print("No verdict is printed. Read the ratio column and the three finals.")
