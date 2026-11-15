"""Does stage PAIR's counterfactual margin do what stagepair_score.py
registered it to do, and -- the reason this file exists -- is it the right SIZE
at initialization? Seven assertions, a couple of minutes on CPU, nothing
trained.

WHY SCALE AT INIT IS THE SUBJECT HERE. Finding 76 is the lesson: stage Y's
smoke test made five assertions, every one of them true, that the channel's
forward map was algebraically what the design said -- and the first array still
had to be cancelled at 2h29, because s was initialized to ones and nothing in
those five checks looked at the SCALE of the resulting perturbation. g had std
21 against node logits whose spread is 0.61. An algebraically correct term of
the wrong magnitude passes every structural check and destroys the run.

So the three checks this stage turns on are 3, 4 and 5 below, and they are about
magnitude and reach rather than algebra.

  1  INDEX.      The four integers are the right four integers, checked against
                 stage Z's own build_triple on real instances rather than
                 against the position formula that produced them: writing
                 pair[1] at pair[0] into x must reproduce build_triple's broken
                 prompt exactly, the difference must be exactly one token, and
                 the token after pair[2] must be v's node token.

  2  ID SPACE.   g's last axis is the node block ALREADY selected, so pair[3] is
                 a node index in [0, 48) and NOT a vocabulary id. A vocabulary
                 id is < 48 for the low nodes and would score the wrong node at
                 identical cost, silently. Asserted against the channel's own
                 arithmetic through the embedding row node_offset + v.

  3  MARGIN AT INIT IS EXACTLY THE MARGIN. det_scale = 0 makes g identically 0
                 on BOTH prompts, so g_clean - g_broken is identically 0 and
                 L_pair == pair_margin exactly, for every row, with the hinge
                 active. This is the one number a reader of the history can use
                 to confirm epoch 0 started where the design says it starts: the
                 logged train_pair_hinge must be 1.0000 and train_pair_active
                 1.000 at the first epoch's first steps.

  4  REACH AT INIT. Because g = (h * s) @ E_v^T, dg/dh = s * E_v = 0 at init, so
                 the margin's gradient reaches ONLY the 320-dim det_scale and
                 NOTHING in the body, the embedding or the readout. That is a
                 property of this term at this initialization and it is worth
                 asserting rather than reasoning about: it means epoch 0 cannot
                 corrupt the control's trajectory through the margin, and it
                 means the term is alive only through 320 numbers until they
                 move off zero.

  5  RELATIVE SCALE. The two terms that write to those 320 numbers at init are
                 the determinacy BCE and this margin. Their gradient norms on
                 det_scale are measured and printed, because a margin that
                 swamps the BCE by orders of magnitude would be stage Y's
                 initialization mistake again in a different place -- the BCE is
                 what makes g mean determinacy at all, and the margin only
                 contrasts it.

  6  THE CONTRAST IS VISIBLE. With s temporarily set to ones, g_clean[p,v] must
                 differ from g_broken[p,v]: the body's hidden state at the
                 predict position already distinguishes the two prompts in the
                 direction E_v. If it did not, dL_pair/ds would be identically
                 zero for every input and the term would be unlearnable rather
                 than merely hard. s is restored to zeros and that is asserted.

  7  ONE STEP MOVES IT. One descent step on L_pair alone drops the hinge below
                 the margin. Stage Y's s = 0 is not a saddle for the BCE and it
                 must not be one for this either.

    python3 scripts/stagepair_smoke.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spin_gnn.config import SudokuReasoningConfig  # noqa: E402
from spin_gnn.sat_dataset2 import SATTokenDataset, make_token_batch_pair  # noqa: E402
from spin_gnn.sat_transformer2 import SATReasoningTransformer, sat2_nll_loss  # noqa: E402
from spin_gnn.sudoku_determinacy import (  # noqa: E402
    determinacy_loss_from_channel, determinacy_targets,
)
from spin_gnn.sudoku_pair import PAIR_NONE, build_pair_index  # noqa: E402
from spin_gnn.sudoku_skip import build_triple, find_unforceable_deductions  # noqa: E402
from spin_gnn.sudoku_tokens import encode_instance  # noqa: E402
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab  # noqa: E402

# The S1/stage-Y cell, unchanged. max_len=512 is in the parameter count because
# the positional table is trainable, and it also has to clear the longest N=48
# trace (408 tokens at the deepest density).
CELL = dict(vocab_size=58, d_model=320, num_layers=16, nhead=4, max_len=512,
            dim_feedforward=1024, norm_first=True, arch_mode="tied_no_mlp",
            n_blocks=1)
NODE_OFFSET, N_NODES = 2, 48
TRAINABLE = 614_138
# Imported, never retyped: the registration is the source of the margin and the
# weight, exactly as stage Y's launcher imported its bars from stagey_score.
import scripts.stagepair_score as sp  # noqa: E402

N_INSTANCES = 64
N_ROWS = 4          # rows in the gradient batch; 16 applications on CPU is slow
SMOKE_SEED = 910_000_000


def check(name, ok, detail=""):
    print("%-18s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    return ok


def build_model(seed=0):
    torch.manual_seed(seed)
    return SATReasoningTransformer(determinacy_channel=True, node_offset=NODE_OFFSET,
                                   n_nodes=N_NODES, **CELL)


def main():
    torch.use_deterministic_algorithms(False)
    torch.manual_seed(0)
    ok = []

    cfg = SudokuReasoningConfig(n_nodes=48, degree=5, n_colors=4, seed=SMOKE_SEED,
                                shuffle_adj=True, fall=True)
    vocab = build_vocab(cfg)
    inst = build_instances(cfg, N_INSTANCES, cfg.seed)
    toks = [encode_instance(adj, trace, vocab) for adj, _c, _h, trace in inst]
    pairs = build_pair_index(inst, toks, vocab, cfg.seed + 500_000_000)
    hits = [i for i, p in enumerate(pairs) if p != PAIR_NONE]
    assert len(hits) >= N_ROWS, ("too few cases to smoke-test on", len(hits))

    # ---- 1. the four integers against stage Z's own construction -------------
    bad = []
    for i in hits[:16]:
        epos, etok, ppos, vnode = pairs[i]
        adj, colors, hints, trace = inst[i]
        # Re-derive the case the same way build_pair_index did, then build the
        # triple stage Z builds from it and compare the two edits.
        case = find_unforceable_deductions(adj, colors, hints, trace, vocab,
                                           np.random.default_rng(0),
                                           max_per_instance=1)
        if not case:
            bad.append("instance %d lost its case on re-scan" % i)
            continue
        clean, broken, _placebo, z_ppos = build_triple(adj, trace, vocab, case[0])
        mine = clean.copy()
        mine[epos] = etok
        if not np.array_equal(mine, broken):
            bad.append("instance %d: the 4-integer edit != build_triple's broken" % i)
        if int((clean != mine).sum()) != 1:
            bad.append("instance %d: the edit is not one token" % i)
        if z_ppos != ppos:
            bad.append("instance %d: predict pos %d != build_triple's %d"
                       % (i, ppos, z_ppos))
        if int(vocab.node_index(clean[ppos + 1])) != vnode:
            bad.append("instance %d: the token after the predict pos is not v" % i)
    ok.append(check("1 index", not bad,
                    "%d cases agree with build_triple on all four integers%s"
                    % (len(hits[:16]), "" if not bad else "; " + "; ".join(bad[:3]))))

    # ---- 2. the node axis is node indices, not token ids ---------------------
    m = build_model()
    with torch.no_grad():
        m.det_scale.fill_(0.7)
        h = torch.randn(1, 1, CELL["d_model"])
        g = m.determinacy_logits(h)
        v = int(pairs[hits[0]][3])
        want = float(((h[0, 0] * m.det_scale) *
                      m.token_embed.weight[NODE_OFFSET + v]).sum())
        got = float(g[0, 0, v])
        # And the wrong id space would give a different number, which is the
        # whole point: the two agree only because v is a node INDEX.
        v_tok = int(vocab.node_token(v))
        m.det_scale.zero_()
    ok.append(check("2 id space", abs(got - want) < 1e-4,
                    "g[.., %d] = <h*s, E_{offset+%d}> to %.1e; node %d is token "
                    "%d, and indexing g at %d would score node %d"
                    % (v, v, abs(got - want), v, v_tok, v_tok, v_tok)))

    # ---- the batch every scale check below runs on --------------------------
    aux = [determinacy_targets(adj, trace, vocab, t)
           for (adj, _c, _h, trace), t in zip(inst, toks)]
    ds = SATTokenDataset(toks, max_len=None, aux=aux, pair=pairs)
    # Indices into the DATASET, which may have dropped short sequences; re-read
    # the pair list off the dataset rather than reusing `hits`, since that is
    # exactly the lockstep the collate depends on.
    ds_hits = [i for i in range(len(ds)) if ds.pair[i] != PAIR_NONE][:N_ROWS]
    x, y, mask, apos, atgt, pair = make_token_batch_pair(ds, ds_hits)
    rows = torch.arange(len(ds_hits))
    epos, etok, ppos, vnode = pair[:, 0], pair[:, 1], pair[:, 2], pair[:, 3]
    assert int((pair[:, 0] < 0).sum()) == 0, "a padded row got into the batch"
    assert int(epos.max()) < x.shape[1] and int(ppos.max()) < x.shape[1], (
        "a position indexes past x", int(epos.max()), int(ppos.max()), x.shape)

    def pair_terms(model):
        """(hinge mean, gap tensor) on this batch, with grad."""
        _z, gc = model(x, valid_mask=mask, return_determinacy=True)
        xb = x.clone()
        xb[rows, epos] = etok
        _zb, gb = model(xb, valid_mask=mask, return_determinacy=True)
        gap = gc[rows, ppos, vnode] - gb[rows, ppos, vnode]
        return torch.clamp(sp.MARGIN - gap, min=0).mean(), gap

    # eval(), not train(): nn.TransformerEncoderLayer's dropout defaults to 0.1
    # and is on in training mode, which would make checks 5 to 7 comparisons
    # between two different dropout masks. Dropout is the only train/eval
    # difference in this model -- there is no batch norm -- so every gradient
    # measured below is the training path's with the noise removed.
    m.eval()
    hinge, gap = pair_terms(m)
    ok.append(check("3 margin at init",
                    float(hinge) == sp.MARGIN and float(gap.abs().max()) == 0.0,
                    "L_pair = %.6f == MARGIN %.3f exactly, max|gap| %.1e over "
                    "%d rows, hinge active on all of them"
                    % (float(hinge), sp.MARGIN, float(gap.abs().max()), len(ds_hits))))

    # ---- 4. at init the margin's gradient reaches det_scale and nothing else --
    m.zero_grad(set_to_none=True)
    hinge.backward()
    reached = {n: float(p.grad.abs().max()) for n, p in m.named_parameters()
               if p.requires_grad and p.grad is not None and p.grad.abs().max() > 0}
    g_pair_s = float(m.det_scale.grad.norm())
    ok.append(check("4 reach at init",
                    list(reached) == ["det_scale"] and g_pair_s > 0,
                    "nonzero grad on %s; ||dL_pair/ds|| = %.3e (dg/dh = s*E_v = 0 "
                    "at init, so the body is untouched)"
                    % (sorted(reached) or "nothing", g_pair_s)))

    # ---- 5. how big that is next to the two terms it shares the run with -----
    m.zero_grad(set_to_none=True)
    z, gc = m(x, valid_mask=mask, return_determinacy=True)
    determinacy_loss_from_channel(gc, apos, atgt).backward()
    g_bce_s = float(m.det_scale.grad.norm())
    m.zero_grad(set_to_none=True)
    z, _gc = m(x, valid_mask=mask, return_determinacy=True)
    sat2_nll_loss(z, y, mask).backward()
    g_main = float(torch.cat([p.grad.flatten() for p in m.parameters()
                             if p.requires_grad and p.grad is not None]).norm())
    ratio = g_pair_s / g_bce_s if g_bce_s else float("inf")
    # A band, not a point: the only failure this can see is the one stage Y hit,
    # a term whose gradient dwarfs the term that gives g its meaning. Printed
    # either way, because the number itself is what a reader wants.
    ok.append(check("5 relative scale", np.isfinite(ratio) and 1e-3 < ratio < 1e3,
                    "||dL_pair/ds|| %.3e vs ||dL_bce/ds|| %.3e, ratio %.3f; "
                    "||dL_main/dtheta|| %.3e over the whole model"
                    % (g_pair_s, g_bce_s, ratio, g_main)))

    # ---- 6. the body already distinguishes the two prompts ------------------
    with torch.no_grad():
        m.det_scale.fill_(1.0)
        _z1, gc1 = m(x, valid_mask=mask, return_determinacy=True)
        xb = x.clone()
        xb[rows, epos] = etok
        _z2, gb1 = m(xb, valid_mask=mask, return_determinacy=True)
        d_at_v = (gc1[rows, ppos, vnode] - gb1[rows, ppos, vnode]).abs()
        spread = float(gc1[rows, ppos].std())
        m.det_scale.zero_()
        restored = bool((m.det_scale == 0).all())
    ok.append(check("6 contrast visible",
                    float(d_at_v.min()) > 0 and restored,
                    "at s=1, |g_clean - g_broken| at (p,v) is %.3f-%.3f against "
                    "g's spread %.3f across the node block; s restored to zero"
                    % (float(d_at_v.min()), float(d_at_v.max()), spread)))

    # ---- 7. one step moves it -----------------------------------------------
    m.zero_grad(set_to_none=True)
    hinge0, _ = pair_terms(m)
    hinge0.backward()
    with torch.no_grad():
        m.det_scale -= 0.1 * m.det_scale.grad
    with torch.no_grad():
        hinge1, gap1 = pair_terms(m)
    ok.append(check("7 one step moves it", float(hinge1) < float(hinge0),
                    "hinge %.6f -> %.6f, mean gap %+.4f after one 0.1 step on "
                    "the 320-vector" % (float(hinge0), float(hinge1),
                                        float(gap1.mean()))))

    # ---- 8. the same scale, measured where the margin can actually reach ----
    # WHY THIS EXISTS, and it is the lesson of 52423780. Check 5 measures the
    # ratio at initialization, where det_scale = 0 -- and dg/dh = det_scale * E_v
    # is then identically zero, so the margin CANNOT touch the body no matter how
    # large it is. Check 4 asserts exactly that. So check 5's reassuring 0.729 was
    # true and vacuous: it bounded a quantity that is zero by construction at the
    # only point it looked, and the thing it was supposed to rule out happens
    # later, once the channel is live and the hinge's gradient flows through BOTH
    # forward passes into the shared block. That is finding 76's mistake again one
    # level up -- checking a magnitude at the one point where it is degenerate.
    #
    # det_scale is set to the magnitude a real run actually reaches: 52430713
    # measured |s| = 0.4267 after 750 steps, so 0.024 per component over 320 of
    # them gives 0.024*sqrt(320) = 0.429. Not 1.0, which is check 6's probe value
    # and far outside anything training visits.
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        m.det_scale.fill_(0.024)
    body = [p for n, p in m.named_parameters()
            if p.requires_grad and n != "det_scale"]
    hinge_live, gap_live = pair_terms(m)
    hinge_live.backward()
    pair_body = float(torch.cat([p.grad.flatten() for p in body
                                 if p.grad is not None]).norm())
    m.zero_grad(set_to_none=True)
    z_live, _g = m(x, valid_mask=mask, return_determinacy=True)
    sat2_nll_loss(z_live, y, mask).backward()
    main_body = float(torch.cat([p.grad.flatten() for p in body
                                 if p.grad is not None]).norm())
    with torch.no_grad():
        m.det_scale.zero_()
    body_ratio = pair_body / main_body if main_body else float("inf")
    # An auxiliary term at lambda = 1 must not out-pull the objective it is
    # auxiliary to, on the parameters they share. Above 1.0 the margin is
    # steering the shared block and the main task is the passenger, which is
    # what 52423780 looked like from the outside: main pinned at uniform while
    # the body's gradient ran 40x the control's.
    ok.append(check("8 scale with a live channel", body_ratio < 1.0,
                    "at |s|=%.3f, ||dL_pair/dbody|| %.3e vs ||dL_main/dbody|| "
                    "%.3e, ratio %.3f (mean gap %+.4f)"
                    % (0.024 * (CELL["d_model"] ** 0.5), pair_body, main_body,
                       body_ratio, float(gap_live.mean()))))

    print("\nregistered: margin %.3f  lambda_pair %.3f  trainable %s (unchanged "
          "from stage Y)" % (sp.MARGIN, sp.LAMBDA_PAIR, format(sp.N_TRAINABLE, ",")))
    assert sp.N_TRAINABLE == TRAINABLE == m.trainable_parameter_count()
    print("%s" % ("all eight pass" if all(ok) else "SOMETHING FAILED"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
