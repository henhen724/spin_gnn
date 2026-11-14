"""Does stage Y's determinacy channel do what stagey_score.py registered it
to do, and does it leave every earlier stage alone? Five assertions, seconds
on CPU, nothing trained.

Run before the array, because three of these five can only fail silently:

  1  COUNT.      613,818 trainable without the channel, 614,138 with it, on
                 the S1 cell (tied_no_mlp, T=16, d=320, ff=1024, h=4,
                 max_len=512). The registration says "+320 parameters, +0.05%"
                 and that claim is checkable before any GPU time is spent.
                 max_len is in the count -- the positional table is trainable
                 -- so this asserts at 512 and nowhere else.

  2  OFF IS OFF. With determinacy_channel=False the logits are bit-identical
                 to a model built without the argument at the same seed, and
                 no det_scale appears in the state dict. Stages S through X
                 must remain reproducible from this file.

  3  EMISSION.   z'_node - z_node == log sigma(g) exactly, and z'_other ==
                 z_other exactly. This is the whole forward-side claim.

  4  SEPARATION. d(BCE)/d(readout.weight) is EXACTLY zero. Stage X's verdict
                 was representation_without_use, diagnosed as the BCE and the
                 cross-entropy competing for the same scalars; the fix is only
                 real if the BCE no longer touches the readout at all.

  5  RE-RANK.    The emission can reorder the node block -- a node with the
                 larger z can lose to one with the smaller. Finding 74 says
                 the untied guard re-ranks rather than stops, and a monotone
                 gate z -> z + log sigma(a(z - tau)) provably cannot do this.
                 Asserted on the model's own forward with a fitted s, not on
                 the formula in the abstract.

  6  INERT AT INIT, LIVE AFTER ONE STEP. The channel's contribution must be
                 CONSTANT across the node axis at initialization -- the same
                 number for every node at every position -- so the first step
                 starts from the control's ranking, and it must move off that
                 after one BCE step.

                 THIS CHECK IS WHY STAGE Y IS BEING RESUBMITTED. The first
                 array (52336800) initialized s to ones, on the untested
                 reasoning that g = <h, E_v> would be "a small common offset".
                 It is not: h is LayerNorm'd and E_v ~ N(0, I), so log
                 sigma(g) had std 10.3 and reached -59 against node logits of
                 spread 0.61 -- a random per-node penalty 17x the signal.
                 Seeds 0 and 2 trained through it, seed 1 never left the
                 initial plateau, and Y4 needs all three. Nothing in the five
                 checks above looks at the SCALE of the perturbation, which is
                 how a design error passed a smoke test that asserts the
                 forward map is algebraically right.

    python3 scripts/stagey_smoke.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spin_gnn.sat_transformer2 import SATReasoningTransformer
from spin_gnn.sudoku_determinacy import determinacy_loss_from_channel

# The S1 cell. max_len=512 is not cosmetic: the positional table is trainable,
# so 511 would give 613,498 and the assertion below would be about a different
# model.
CELL = dict(vocab_size=58, d_model=320, num_layers=16, nhead=4, max_len=512,
            dim_feedforward=1024, norm_first=True, arch_mode="tied_no_mlp",
            n_blocks=1)
NODE_OFFSET, N_NODES = 2, 48
BASE, WITH = 613_818, 614_138


def build(channel, seed=0):
    torch.manual_seed(seed)
    kw = dict(CELL)
    if channel:
        kw.update(determinacy_channel=True, node_offset=NODE_OFFSET,
                  n_nodes=N_NODES)
    return SATReasoningTransformer(**kw)


def check(name, ok, detail=""):
    print("%-12s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    return ok


def main():
    torch.use_deterministic_algorithms(False)
    ok = []

    off, on = build(False), build(True)
    n_off, n_on = off.trainable_parameter_count(), on.trainable_parameter_count()
    ok.append(check("1 count", (n_off, n_on) == (BASE, WITH),
                    "%d -> %d, +%d (+%.3f%%)"
                    % (n_off, n_on, n_on - n_off, 100 * (n_on - n_off) / n_off)))

    x = torch.randint(0, CELL["vocab_size"], (3, 40))
    mask = torch.ones(3, 40, dtype=torch.bool)
    mask[2, 35:] = False
    off.eval(), on.eval()
    with torch.no_grad():
        # A model built with no determinacy argument at all, i.e. the call
        # every script in the repo makes today.
        torch.manual_seed(0)
        legacy = SATReasoningTransformer(**CELL).eval()
        same = torch.equal(legacy(x, valid_mask=mask), off(x, valid_mask=mask))
    ok.append(check("2 off is off",
                    same and "det_scale" not in off.state_dict()
                    and "det_scale" in on.state_dict(),
                    "bit-identical logits, det_scale absent"))

    lo, hi = NODE_OFFSET, NODE_OFFSET + N_NODES
    with torch.no_grad():
        z_on, g = on(x, valid_mask=mask, return_determinacy=True)
        # The same hidden state through the same readout, with the channel's
        # contribution removed by hand.
        h = on.token_embed(x) + on.pos_embed[:x.shape[1]]
        cm = torch.triu(torch.ones(40, 40, dtype=torch.bool), diagonal=1)
        for t in range(on.n_iters):
            h = on.encoder.layers[t % on.n_blocks](
                h, src_mask=cm, src_key_padding_mask=~mask)
        if on.encoder.norm is not None:
            h = on.encoder.norm(h)
        z = on.readout(h)
    # Compared as a SUM, not as z_on - z == log sigma(g): the subtraction
    # undoes a float32 rounding and misses bit-exactness by ~2e-6 even when
    # the forward is exactly right. The non-node columns below are compared
    # directly, where exactness is the real claim.
    node_ok = torch.equal(z_on[..., lo:hi], z[..., lo:hi] + F.logsigmoid(g))
    rest_ok = (torch.equal(z_on[..., :lo], z[..., :lo])
               and torch.equal(z_on[..., hi:], z[..., hi:]))
    ok.append(check("3 emission", node_ok and rest_ok,
                    "node block shifted by log sigma(g), %d other columns "
                    "untouched" % (CELL["vocab_size"] - N_NODES)))

    pos = torch.randint(0, 40, (3, 6))
    tgt = torch.randint(-1, 2, (3, 6, N_NODES))
    on.zero_grad()
    _, g = on(x, valid_mask=mask, return_determinacy=True)
    determinacy_loss_from_channel(g, pos, tgt).backward()
    rg = on.readout.weight.grad
    ds = on.det_scale.grad
    ok.append(check("4 separation",
                    (rg is None or float(rg.abs().max()) == 0.0)
                    and ds is not None and float(ds.abs().max()) > 0,
                    "d(BCE)/d(readout) = %s, d(BCE)/d(det_scale) max |.| = %.3e"
                    % ("None" if rg is None else "%.3e" % float(rg.abs().max()),
                       float(ds.abs().max()))))

    # 5. Re-ranking is the property a monotone gate lacks, so it is asserted
    # on the real forward: take the top two node logits at one position and
    # fit s by gradient descent to swap them, with z held fixed by construction
    # (only det_scale is in the optimizer, so the readout cannot move).
    rr = build(True, seed=1).eval()
    with torch.no_grad():
        z0 = rr(x[:1, :20], valid_mask=None)[0, -1, lo:hi]
    a, b = torch.topk(z0, 2).indices.tolist()
    opt = torch.optim.Adam([rr.det_scale], lr=0.05)
    swapped = False
    for _ in range(300):
        zz = rr(x[:1, :20], valid_mask=None)[0, -1, lo:hi]
        if float(zz[b] - zz[a]) > 0:
            swapped = True
            break
        opt.zero_grad()
        (zz[a] - zz[b]).backward()
        opt.step()
    with torch.no_grad():
        z1 = rr(x[:1, :20], valid_mask=None)[0, -1, lo:hi]
    ok.append(check("5 re-rank", swapped,
                    "node %d beat node %d: z gap %+.3f -> %+.3f with only the "
                    "320-vector moving" % (b, a, float(z0[b] - z0[a]),
                                           float(z1[b] - z1[a]))))

    # 6. The scale check the first submission did not have.
    fresh = build(True, seed=2).eval()
    with torch.no_grad():
        zf, gf = fresh(x, valid_mask=mask, return_determinacy=True)
        contrib = F.logsigmoid(gf)
        # Spread ACROSS NODES at each position: this is what reorders the
        # block, and it must be exactly zero before training.
        across = float(contrib.std(dim=-1).max())
        node_spread = float(zf[..., lo:hi].std())
    fresh.train()
    _, gf = fresh(x, valid_mask=mask, return_determinacy=True)
    determinacy_loss_from_channel(gf, pos, tgt).backward()
    with torch.no_grad():
        fresh.det_scale -= 0.1 * fresh.det_scale.grad
    with torch.no_grad():
        _, g2 = fresh(x, valid_mask=mask, return_determinacy=True)
        moved = float(F.logsigmoid(g2).std(dim=-1).max())
    ok.append(check("6 init scale", across == 0.0 and moved > 0,
                    "across nodes at init %.2e (logits' own spread %.2f), "
                    "after one step %.3e" % (across, node_spread, moved)))

    print("\n%s" % ("all six pass" if all(ok) else "SOMETHING FAILED"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
