"""Why stage PAIR's array (52423780) sat at uniform output.

NOT A RESULT AND NOT A STAGE. No bar is read and no verdict is made. This
script exists to localise one bug and will be deleted or left as a diagnostic;
nothing it prints goes in a deck.

THE OBSERVATION. 52423780_[0-2], all three seeds, at epoch 5:

    train_nll=3.9540  val_accuracy=0.0312  det_bce=0.7085  val_det_auroc=0.5854
    pair_hinge=1.0001  gap=+0.0001  active=0.999

against the stage-Y control (52348732_0), which is the SAME data (identical
determinacy-target counts), the SAME cell and the SAME 614,138 parameters, at
epoch 0:

    train_nll=2.6375  val_accuracy=0.4328  det_bce=0.5636  val_det_auroc=0.7752

log(58) = 4.060 and the majority-class baseline is 0.0306, so the pair arm is at
uniform output after 45,000 steps while the control was already solving after
7,500. det_bce 0.709 is log(2) = 0.693, i.e. the channel is at chance too. Both
heads are pinned.

WHAT IS ALREADY RULED OUT, so this script does not re-test it:
  - the optimizer step (zero_grad -> backward -> step, trainer lines 676-678);
  - det_scale being frozen: the preflight's 20 steps moved it to 9.67e-02;
  - an exploded embedding: the preflight's token_embed node block is 124.36,
    which is sqrt(48*320) = 124.0, i.e. still at its N(0,1) init;
  - the data: the determinacy-target counts are byte-identical to the control's.

THE HYPOTHESIS THIS TESTS. gap == 0.0000 held to four decimals for 45,000 steps
is the strange number: at s=1 the smoke test measured |g_clean - g_broken| =
0.392-2.610 at the same (p, v), so the contrast is reachable. A gap pinned at
exactly zero while |g| is free to grow is what a SHARED SHIFT looks like -- the
hinge pushing g the same way on both prompts, which reduces nothing and leaves
g's magnitude as the only thing that moves. Because the node block's emission is

    z_node = z_node + log sigmoid(g)

a g driven large and negative subtracts an unbounded amount from exactly the
node logits, and the trace is about half node tokens, so the main task dies
while the margin stays pinned at 1.0. det_bce near log 2 argues against large
|g| and is the reason this is a hypothesis and not a conclusion.

WHAT IT MEASURES. Two arms from one init and one data order, pair_weight 0 and
1, instrumented every LOG_EVERY steps with the things the trainer does not
print: g's magnitude and sign, the two heads' gradient norms against the body's,
and whether the gap is zero because the forwards agree or because both are being
dragged. If the control learns here and the pair arm does not, the term is the
cause and the instrumentation says which way it fails.
"""

import sys
import torch

sys.path.insert(0, "/home/users/hshunt/spin_gnn")

from spin_gnn.config import (SudokuReasoningConfig, SudokuReasoningModelConfig,
                             SudokuReasoningTrainConfig)
from spin_gnn.sat_dataset2 import (SATTokenDataset, iterate_token_batches,
                                   make_token_batch_pair)
from spin_gnn.sat_transformer2 import SATReasoningTransformer, sat2_nll_loss
from spin_gnn.sudoku_determinacy import (determinacy_loss_from_channel,
                                         determinacy_targets)
from spin_gnn.sudoku_pair import build_pair_index
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab
import numpy as np

N_INSTANCES = 20_000
BATCH = 32
STEPS = 900          # ~29k instances seen, vs the control's 7,500-step epoch 0
LOG_EVERY = 150
SEED = 0
MARGIN = 1.0

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}  instances={N_INSTANCES}  batch={BATCH}  steps={STEPS}")

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

ml = max(s.shape[0] for s in tokens) + 98
ds = SATTokenDataset(tokens, max_len=ml, aux=aux, pair=pair)
idx = list(range(len(ds)))
print(f"max_len={ml}  dataset={len(ds):,}")

NODE_LO = vocab.node_offset
NODE_HI = vocab.node_offset + vocab.n_nodes


def fresh():
    torch.manual_seed(SEED)
    return SATReasoningTransformer(
        vocab.size, d_model=320, num_layers=16, nhead=4, max_len=ml,
        dim_feedforward=1024, norm_first=True, arch_mode="tied_no_mlp",
        n_blocks=1, determinacy_channel=True,
        node_offset=vocab.node_offset, n_nodes=vocab.n_nodes).to(dev)


def gnorm(params):
    tot = 0.0
    for p in params:
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return tot ** 0.5


def run(pair_w, label):
    print(f"\n=== {label}: pair_weight={pair_w:g} "
          f"{'=' * (46 - len(label))}")
    model = fresh()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=1e-3)
    # Same seed for the batch rng in both arms, so the two see the same data in
    # the same order and any divergence is the term's and not the shuffle's.
    rng = np.random.default_rng(SEED)
    body = [p for n, p in model.named_parameters()
            if n != "det_scale" and not n.startswith("token_embed")]
    emb = [model.token_embed.weight]
    print(f"{'step':>5} {'main':>7} {'det_bce':>8} {'hinge':>7} {'gap':>8} "
          f"{'|s|':>8} {'g_mean':>9} {'g_absmax':>9} {'|E_node|':>9} "
          f"{'gr_s':>8} {'gr_E':>8} {'gr_body':>8}")
    step = 0
    while step < STEPS:
        for batch in iterate_token_batches(ds, idx, BATCH, rng=rng,
                                           collate=make_token_batch_pair):
            if step >= STEPS:
                break
            x, y, mask = (t.to(dev) for t in batch[:3])
            logits, g = model(x, valid_mask=mask, return_determinacy=True)
            main = sat2_nll_loss(logits, y, mask)
            loss = main
            apos, atgt = batch[3].to(dev), batch[4].to(dev)
            det = determinacy_loss_from_channel(g, apos, atgt)
            if det is not None:
                loss = loss + det
            hinge_v = gap_v = float("nan")
            if pair_w > 0:
                pr = batch[5].to(dev)
                sel = (pr[:, 0] >= 0).nonzero(as_tuple=True)[0]
                if sel.numel():
                    epos, etok = pr[sel, 0], pr[sel, 1]
                    ppos, vnode = pr[sel, 2], pr[sel, 3]
                    xb = x.clone()
                    xb[sel, epos] = etok
                    _zb, gb = model(xb, valid_mask=mask, return_determinacy=True)
                    gap = g[sel, ppos, vnode] - gb[sel, ppos, vnode]
                    hinge = torch.clamp(MARGIN - gap, min=0)
                    loss = loss + pair_w * hinge.mean()
                    hinge_v = float(hinge.mean())
                    gap_v = float(gap.mean())
            opt.zero_grad()
            loss.backward()
            if step % LOG_EVERY == 0:
                gs = gnorm([model.det_scale])
                ge = gnorm(emb)
                gb_ = gnorm(body)
                print(f"{step:5d} {float(main):7.4f} "
                      f"{float(det) if det is not None else float('nan'):8.4f} "
                      f"{hinge_v:7.4f} {gap_v:+8.4f} "
                      f"{float(model.det_scale.norm()):8.3e} "
                      f"{float(g.mean()):+9.4f} {float(g.abs().max()):9.3f} "
                      f"{float(model.token_embed.weight[NODE_LO:NODE_HI].norm()):9.3f} "
                      f"{gs:8.2e} {ge:8.2e} {gb_:8.2e}")
            opt.step()
            step += 1
    return float(main)


ctrl = run(0.0, "control (stage Y)")
pairm = run(1.0, "pair arm")
print(f"\nmain nll after {STEPS} steps: control {ctrl:.4f}  pair {pairm:.4f}")
print("CONTROL LEARNS, PAIR DOES NOT" if ctrl < 2.5 <= pairm else
      "both arms behave the same -- the term is NOT the cause")
