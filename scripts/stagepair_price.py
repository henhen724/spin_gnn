"""Price the pair-margin arm before registering it. MEASURES ONLY.

STATUS.md's next step 1 says to price this arm before registering it, so this
script is the price and nothing else: it trains nothing, writes no checkpoint,
registers no bar and makes no verdict. Its whole output is three numbers a
registration needs and that cannot be guessed:

  A. YIELD. What fraction of training instances carry a matched pair at all.
     The term can only be applied to an instance where one round-0 deduction
     can be un-forced by editing a single hint colour, and
     find_unforceable_deductions is the thing that decides. If the yield is
     low the term is applied to few instances per epoch and the arm is weaker
     than it looks; if it is high the extra forward is paid on nearly every
     step. Stage Z needed 400 instances to harvest 350 cases at
     max_per_instance=4, which says nothing about the fraction of INSTANCES
     with at least one case, which is what a training term sees.

  B. SCAN COST. find_unforceable_deductions is pure Python over each trace's
     round-0 deductions. The training run already pays build_instances; the
     scan is the only NEW setup cost, so the two are timed separately and only
     the second is the arm's.

  C. STEP COST. The margin needs g on the corrupted prompt as well as the
     clean one, and the loss depends on both, so the backward runs through two
     forwards. STATUS estimates "about 2x the ~10 min/epoch" -- this measures
     the real ratio at the real shape, including the fact that the corrupted
     forward also computes a readout nobody reads.

The corruption itself is free, and that is worth stating because it is the
reason this arm is affordable at all: build_triple is clean.copy() with ONE
index written (sudoku_skip.py:185-187, which asserts the one-token property),
with no solver call and no second instance generated. So the per-instance
storage is an edit -- (hint colour position, new colour token, predict
position, node index) -- four integers, not a second token array, and that is
how this script stores it. The last two live in different id spaces and mixing
them is silent rather than loud; see pair_index.

Two design choices that a registration will inherit, recorded here because the
measurement assumes them:

  max_per_instance=1. Stage Z's probe harvested up to 4 cases per instance
  because it wanted cases; a training term wants instances weighted equally,
  and 4 pairs from one lucky instance would weight it 4x.

  The corrupted forward feeds ONLY the margin. Its next-token targets are not
  usable: after the edit the deduction is no longer forced, so the planted
  trace is no longer the correct continuation. Adding an NLL on it would
  train the model to assert a deduction the prompt does not support.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spin_gnn.config import SudokuReasoningConfig  # noqa: E402
from spin_gnn.sat_dataset2 import (  # noqa: E402
    SATTokenDataset, make_token_batch_aux)
from spin_gnn.sat_transformer2 import (  # noqa: E402
    SATReasoningTransformer, sat2_nll_loss)
from spin_gnn.sudoku_determinacy import (  # noqa: E402
    determinacy_loss_from_channel, determinacy_targets)
from spin_gnn.sudoku_skip import (  # noqa: E402
    build_triple, find_unforceable_deductions)
from spin_gnn.sudoku_tokens import encode_instance  # noqa: E402
from spin_gnn.train_sudoku_transformer import (  # noqa: E402
    build_instances, build_vocab)

# The stage-Y cell this arm would extend, read off its own config.json rather
# than retyped from a deck: tied_no_mlp T=16 d320 ff1024 h4 with the channel on,
# 614,138 trainable. STATUS asserts max_len == 512 before asserting the count,
# so the same is asserted here.
Y_CONFIG = dict(n_nodes=48, degree=5, n_colors=4, rho=0.3, rho_max=0.95,
                hint_mode="removal", shuffle_adj=True, shuffle_events=False,
                scratchpad=False, fall=True)
Y_MODEL = dict(d_model=320, num_layers=16, nhead=4, dim_feedforward=1024,
               norm_first=True, arch_mode="tied_no_mlp", n_blocks=1)
Y_MAX_LEN = 512
Y_TRAINABLE = 614138
Y_N_TRAIN = 240000
Y_BATCH = 32
Y_EPOCHS = 50
# 9h15m over 50 epochs, from the stage-Y array's own elapsed time.
Y_MIN_PER_EPOCH = 11.1


def scan(cfg, n_instances, seed, vocab, verify):
    """(instances, cases, build_s, scan_s). cases[i] is None or one case."""
    t0 = time.time()
    instances = build_instances(cfg, n_instances, seed)
    build_s = time.time() - t0

    rng = np.random.default_rng(seed)
    t0 = time.time()
    cases = []
    for adj, colors, hints, trace in instances:
        found = find_unforceable_deductions(adj, colors, hints, trace, vocab,
                                            rng, max_per_instance=1)
        cases.append(found[0] if found else None)
    scan_s = time.time() - t0

    # The edit has to be a one-token change at a hint colour, and build_triple
    # asserts exactly that -- so running it is the check. Done on a sample
    # rather than all of them because it re-encodes the instance, which the
    # training path would not: there the clean tokens already exist.
    checked = 0
    for i, (inst, case) in enumerate(zip(instances, cases)):
        if case is None or checked >= verify:
            continue
        adj, _c, _h, trace = inst
        clean, broken, _placebo, pos = build_triple(adj, trace, vocab, case)
        assert int((clean != broken).sum()) == 1
        assert 0 <= pos < len(clean)
        checked += 1
    assert checked == min(verify, sum(c is not None for c in cases))
    return instances, cases, build_s, scan_s


def pair_index(cases, vocab, idx):
    """The four integers per instance: (edit pos, edit TOKEN, predict pos, node
    INDEX). Everything the margin needs, without a second token array.

    The two id spaces are deliberately different and mixing them is silent.
    The edit is written into the token stream, so it is a vocabulary id
    (vocab.color_token). The node is read out of g, whose last axis is the node
    block ALREADY selected -- n_nodes wide, no node_offset, see
    determinacy_loss_from_channel -- so it is a node index in [0, n_nodes). A
    vocab id here is < n_nodes for the low nodes and would quietly score the
    wrong node rather than raising.
    """
    out = []
    for i in idx:
        c = cases[i]
        node = int(c["node"])
        assert 0 <= node < vocab.n_nodes, (node, vocab.n_nodes)
        out.append((int(c["hint_color_pos"]), int(vocab.color_token(c["new_color"])),
                    -1, node))
    return out


def time_steps(model, batch, edits, device, n_steps, warmup, det_w, pair_w,
               margin, pair):
    """Median ms per optimizer step, with and without the second forward.

    Median rather than mean: the first steps after warmup still catch
    allocator growth, and one 200 ms outlier would move a mean that the
    extrapolation to 50 epochs then multiplies by 375,000.
    """
    x, y, mask, apos, atgt = (t.to(device) for t in batch)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=1e-3)
    rows, epos, etok, ppos, vnode = edits
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for step in range(warmup + n_steps):
        t0 = time.time()
        logits, g = model(x, valid_mask=mask, return_determinacy=True)
        loss = sat2_nll_loss(logits, y, mask)
        det = determinacy_loss_from_channel(g, apos, atgt)
        if det is not None:
            loss = loss + det_w * det
        if pair:
            xb = x.clone()
            xb[rows, epos] = etok
            # valid_mask is the same tensor: the edit changes one token's
            # VALUE, never the length, so no padding moves.
            _zb, gb = model(xb, valid_mask=mask, return_determinacy=True)
            gc_v = g[rows, ppos, vnode]
            gb_v = gb[rows, ppos, vnode]
            loss = loss + pair_w * torch.clamp(margin - (gc_v - gb_v), min=0).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize(device)
        if step >= warmup:
            times.append(1000.0 * (time.time() - t0))
    return (float(np.median(times)), float(np.percentile(times, 90)),
            torch.cuda.max_memory_allocated(device) / 2 ** 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-instances", type=int, default=20000,
                    help="instances scanned for the yield; 20k is 8.3%% of the "
                         "240k training set")
    ap.add_argument("--seed", type=int, default=900000000,
                    help="disjoint from training (0..240299) and from stage "
                         "Z's probe streams (6e8, 7e10)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--verify", type=int, default=200)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("part C needs a GPU; the ratio off one is not the price")
    cfg = SudokuReasoningConfig(seed=args.seed, **Y_CONFIG)
    vocab = build_vocab(cfg)

    print("=== A+B. yield and scan cost ===", flush=True)
    instances, cases, build_s, scan_s = scan(
        cfg, args.n_instances, args.seed, vocab, args.verify)
    have = [i for i, c in enumerate(cases) if c is not None]
    yield_frac = len(have) / len(cases)
    print(f"  instances scanned      {len(cases):,}")
    print(f"  with a matched pair    {len(have):,}  ({yield_frac:.4f})")
    print(f"  build_instances        {build_s:7.1f} s  "
          f"({1000 * build_s / len(cases):.2f} ms/instance, ALREADY PAID)")
    print(f"  the scan               {scan_s:7.1f} s  "
          f"({1000 * scan_s / len(cases):.2f} ms/instance, NEW)")
    print(f"  scan at {Y_N_TRAIN:,}       "
          f"{scan_s * Y_N_TRAIN / len(cases) / 60:7.1f} min, once per run")
    ranks = [cases[i]["round0_rank"] for i in have]
    knowns = [cases[i]["n_known"] for i in have]
    print(f"  round0_rank            median {int(np.median(ranks))}, "
          f"max {max(ranks)}")
    print(f"  n_known neighbours     median {int(np.median(knowns))}, "
          f"min {min(knowns)}, max {max(knowns)}")

    print("\n=== C. step cost at the stage-Y shape ===", flush=True)
    tokens = [encode_instance(adj, trace, vocab)
              for adj, _c, _h, trace in instances]
    aux = [determinacy_targets(adj, trace, vocab, tok)
           for (adj, _c, _h, trace), tok in zip(instances, tokens)]
    dataset = SATTokenDataset(tokens, max_len=Y_MAX_LEN, aux=aux)
    idx = have[:Y_BATCH]
    if len(idx) < Y_BATCH:
        raise SystemExit(f"only {len(idx)} instances with a pair; need {Y_BATCH}")
    batch = make_token_batch_aux(dataset, idx)
    print(f"  batch {Y_BATCH} x {batch[0].shape[1]} tokens, "
          f"{int(batch[2].sum())} valid, "
          f"{int((batch[4] >= 0).sum()):,} scored determinacy cells")

    quad = pair_index(cases, vocab, idx)
    # predict_pos is recomputed against the PADDED batch row rather than the
    # bare sequence, which is the same index here only because make_token_batch
    # left-aligns every row. Asserted rather than assumed.
    for j, i in enumerate(idx):
        adj, _c, _h, trace = instances[i]
        _cl, _br, _pl, pos = build_triple(adj, trace, vocab, cases[i])
        assert int(batch[0][j, quad[j][0]]) != quad[j][1], "edit is a no-op"
        quad[j] = (quad[j][0], quad[j][1], int(pos), quad[j][3])
    edits = tuple(torch.tensor(t, dtype=torch.long, device=device)
                  for t in (list(range(len(idx))), [q[0] for q in quad],
                            [q[1] for q in quad], [q[2] for q in quad],
                            [q[3] for q in quad]))

    model = SATReasoningTransformer(
        vocab.size, max_len=Y_MAX_LEN, determinacy_channel=True,
        node_offset=vocab.node_offset, n_nodes=vocab.n_nodes, **Y_MODEL
    ).to(device)
    n_trainable = model.trainable_parameter_count()
    assert n_trainable == Y_TRAINABLE, (n_trainable, Y_TRAINABLE)
    print(f"  {n_trainable:,} trainable, max_len {Y_MAX_LEN} -- the stage-Y cell")

    base_ms, base_p90, base_gb = time_steps(
        model, batch, edits, device, args.steps, args.warmup,
        det_w=1.0, pair_w=0.0, margin=args.margin, pair=False)
    pair_ms, pair_p90, pair_gb = time_steps(
        model, batch, edits, device, args.steps, args.warmup,
        det_w=1.0, pair_w=1.0, margin=args.margin, pair=True)
    ratio = pair_ms / base_ms
    print(f"  stage-Y step           {base_ms:7.2f} ms  (p90 {base_p90:.2f}, "
          f"{base_gb:.2f} GiB peak)")
    print(f"  with the pair term     {pair_ms:7.2f} ms  (p90 {pair_p90:.2f}, "
          f"{pair_gb:.2f} GiB peak)")
    print(f"  ratio                  {ratio:7.3f} x")

    # The extrapolation is anchored on the MEASURED 11.1 min/epoch rather than
    # on base_ms x steps, because this batch is one fixed batch held in memory
    # and a real epoch also pays data loading and bucketing. The ratio is what
    # transfers; the absolute is stage Y's own.
    steps_per_epoch = Y_N_TRAIN // Y_BATCH
    epoch_min = Y_MIN_PER_EPOCH * ratio
    run_h = epoch_min * Y_EPOCHS / 60
    print(f"\n  {steps_per_epoch:,} steps/epoch, {Y_MIN_PER_EPOCH} min/epoch "
          f"measured for stage Y")
    print(f"  -> {epoch_min:.1f} min/epoch, {run_h:.1f} h for {Y_EPOCHS} epochs, "
          f"+{scan_s * Y_N_TRAIN / len(cases) / 60:.1f} min setup")
    # The timed batch is built from instances that ALL carry a pair, which is
    # the worst case and the right thing to measure. A real batch carries a
    # pair on `yield` of its rows, and the corrupted forward can be run on that
    # sub-batch alone. This line is an ESTIMATE of that saving, not a
    # measurement of it: it assumes the forward's cost is linear in batch size,
    # which at batch 32 on one GPU it roughly is but was not checked.
    adj_ratio = 1.0 + (ratio - 1.0) * yield_frac
    print(f"  corrupting only the {yield_frac:.1%} of rows that carry a pair, "
          f"estimated {adj_ratio:.3f} x -> {Y_MIN_PER_EPOCH * adj_ratio:.1f} "
          f"min/epoch, {Y_MIN_PER_EPOCH * adj_ratio * Y_EPOCHS / 60:.1f} h")

    out = {
        "n_scanned": len(cases), "n_with_pair": len(have),
        "yield": yield_frac, "build_s": build_s, "scan_s": scan_s,
        "scan_min_at_240k": scan_s * Y_N_TRAIN / len(cases) / 60,
        "base_ms": base_ms, "pair_ms": pair_ms, "ratio": ratio,
        "base_p90_ms": base_p90, "pair_p90_ms": pair_p90,
        "base_peak_gib": base_gb, "pair_peak_gib": pair_gb,
        "projected_min_per_epoch": epoch_min, "projected_h_50_epochs": run_h,
        "yield_adjusted_ratio_estimate": adj_ratio,
        "yield_adjusted_h_50_epochs": Y_MIN_PER_EPOCH * adj_ratio * Y_EPOCHS / 60,
        "batch_tokens": int(batch[0].shape[1]), "steps_timed": args.steps,
        "round0_rank_median": float(np.median(ranks)),
        "n_known_median": float(np.median(knowns)),
        "seed": args.seed, "margin": args.margin,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
