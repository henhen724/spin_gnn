"""Verify SudokuReasoningConfig.shuffle_events is semantics-preserving.

The claim the flag rests on (see its docstring in config.py) is that permuting
the events inside a DD round changes only the presentation order: the round
partition, its membership, the reasoning depth T, the graph, the hints and the
target colours are all identical, and only the sequence in which a round's
deductions are written out differs. If that claim were false the retrain would
be measuring two things at once and nothing could be concluded from it.

Checks, on the same seed_start in both arms:
  1. adj, colors, hints identical (the flag must not touch the instance).
  2. the round partition as SETS is identical, so T and membership are too.
  3. the within-round order actually differs on a substantial fraction of the
     multi-event rounds (otherwise the flag is a no-op and the control is
     vacuous).
  4. with the flag off, every round is still in ascending node index -- i.e.
     the default path reproduces the canonical stream every run before
     2026-10-27 trained on.
  5. the encoded token sequences have identical LENGTH (same events, same
     count), so max_len and the parameter count are unaffected.

CPU only, a few hundred small instances: login-node appropriate.

Usage: python -m scripts.verify_shuffle_events [--n-nodes 24] [--n 200]
"""
import argparse
from dataclasses import replace
from math import lgamma

import numpy as np

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_tokens import Vocab, encode_instance
from spin_gnn.train_sudoku_transformer import build_instances

ap = argparse.ArgumentParser()
ap.add_argument("--n-nodes", type=int, default=24)
ap.add_argument("--degree", type=int, default=5)
ap.add_argument("--n-colors", type=int, default=4)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed-start", type=int, default=0)
args = ap.parse_args()

base = SudokuReasoningConfig(n_nodes=args.n_nodes, degree=args.degree, n_colors=args.n_colors,
                             rho=0.3, rho_max=0.95)
off = build_instances(base, args.n, args.seed_start)
on = build_instances(replace(base, shuffle_events=True), args.n, args.seed_start)
assert len(off) == len(on) == args.n

n_rounds = n_multi = n_reordered = n_sorted_on = 0
for (adj_a, col_a, hints_a, tr_a), (adj_b, col_b, hints_b, tr_b) in zip(off, on):
    assert np.array_equal(adj_a, adj_b), "adj differs: the flag touched the graph"
    assert np.array_equal(col_a, col_b), "colors differ: the flag touched the planted coloring"
    assert hints_a == hints_b, "hints differ: the flag touched the hint set"
    assert tr_a["resolved"] == tr_b["resolved"]
    assert len(tr_a["rounds"]) == len(tr_b["rounds"]), "T differs: the flag changed the round count"
    for ra, rb in zip(tr_a["rounds"], tr_b["rounds"]):
        n_rounds += 1
        assert sorted(map(tuple, ra)) == sorted(map(tuple, rb)), \
            "round membership differs: the flag changed WHICH nodes are deduced when"
        assert list(map(tuple, ra)) == sorted(map(tuple, ra)), \
            "the shuffle_events=False arm is not in canonical sorted order"
        if len(ra) > 1:
            n_multi += 1
            if list(map(tuple, ra)) != list(map(tuple, rb)):
                n_reordered += 1
            if list(map(tuple, rb)) == sorted(map(tuple, rb)):
                n_sorted_on += 1

vocab = Vocab(args.n_nodes, args.n_colors, args.degree)
floor_nats, floor_acc, floor_tokens = 0.0, 0.0, 0
for (adj_a, col_a, hints_a, tr_a), (adj_b, col_b, hints_b, tr_b) in zip(off, on):
    sa = encode_instance(adj_a, tr_a, vocab)
    sb = encode_instance(adj_b, tr_b, vocab)
    assert len(sa) == len(sb), "encoded length differs: max_len and n_params would move too"
    # The price of the control, in the currency val_nll is measured in: a round
    # of k simultaneous deductions presented in a uniformly random order costs
    # log(k!) nats that NO model can predict. evaluate() averages the nll over
    # every valid token of the sequence, graph section included, so that is the
    # denominator. A True-arm val_nll higher than the False arm's by about this
    # much means the model lost nothing it could have had.
    floor_nats += sum(lgamma(len(rd) + 1) for rd in tr_a["rounds"])
    # Same statement for token accuracy. The round's frontier is still fully
    # determined, so an ideal model knows WHICH k nodes come next; having seen
    # j of them it must pick uniformly among the k-j left, i.e. it gets H_k of
    # the round's k node tokens right instead of all k. Colour tokens are
    # unaffected -- the colour travels with its node.
    floor_acc += sum(len(rd) - sum(1.0 / m for m in range(1, len(rd) + 1)) for rd in tr_a["rounds"])
    floor_tokens += len(sa) - 1

print(f"instances      {args.n} at N={args.n_nodes} d{args.degree} q{args.n_colors}")
print(f"rounds         {n_rounds} total, {n_multi} with >1 event ({n_multi / n_rounds:.1%})")
print(f"reordered      {n_reordered}/{n_multi} of the multi-event rounds ({n_reordered / max(n_multi, 1):.1%})")
print(f"still sorted   {n_sorted_on}/{n_multi} in the shuffle_events=True arm "
      f"({n_sorted_on / max(n_multi, 1):.1%}; a 2-event round is sorted half the time by chance)")
print(f"nll floor      +{floor_nats / floor_tokens:.4f} nats/token irreducible in the True arm "
      f"({floor_nats / args.n:.2f} nats per instance over {floor_tokens / args.n:.1f} tokens)")
print(f"accuracy floor -{floor_acc / floor_tokens:.4f} token accuracy, same reason "
      f"({floor_acc / args.n:.2f} unpredictable node tokens per instance)")
print("OK: partition, membership, T, graph, hints and encoded length all unchanged")
