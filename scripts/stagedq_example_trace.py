"""One small (d=4, q=3) instance, printed as a human-readable trace AND as
the exact token ids the model sees.

This exists for the deck. The standing rule is that every technical term used
anywhere in a deck is defined in that deck, and that a representative input
and an example trace are explanation slides rather than an appendix -- a
reader who has not seen the code cannot check an axis label against anything
otherwise.

N is deliberately tiny (legality at d=4, q=3 is just 3 | N) so the whole
sequence fits on a slide. The trained cells are N=96 and N=120; nothing here
is a result.

Run:  python3 scripts/stagedq_example_trace.py [--n 12] [--rho 0.5]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_tokens import Vocab, encode_instance
from spin_gnn.train_sudoku_transformer import build_instances


def name(vocab, t):
    """A printable name for a token id, from the grammar alone."""
    specials = ["BEGIN_HINTS", "BEGIN_GRAPH", "BEGIN_REASONING",
                "BEGIN_ROUND", "SOLVED", "UNRESOLVED"]
    if t < len(specials):
        return specials[t]
    if t < vocab.node_offset:
        return "COLOR_%d" % (t - len(specials))
    return "v%d" % vocab.node_index(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--colors", type=int, default=3)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20261121)
    args = ap.parse_args()

    n, d, q = args.n, args.degree, args.colors
    assert n % q == 0 and (d * n) % (q * (q - 1)) == 0, "illegal cell"
    cfg = SudokuReasoningConfig(n_nodes=n, degree=d, n_colors=q,
                                rho=args.rho, rho_max=args.rho,
                                fall=True, shuffle_adj=True)
    adj, colors, hints, trace = build_instances(cfg, 1, args.seed)[0]
    vocab = Vocab(n, q, d)
    toks = encode_instance(adj, trace, vocab)

    print("INSTANCE  N=%d  degree=%d  colours=%d  rho=%.2f" % (n, d, q,
                                                               args.rho))
    print("vocab size %d = N + 6 + q = %d + 6 + %d   node_offset %d"
          % (vocab.size, n, q, vocab.node_offset))
    print("planted colouring:", " ".join("v%d=%d" % (i, c)
                                         for i, c in enumerate(colors)))
    print("given hints (%d of %d nodes): %s"
          % (len(trace["hints"]), n,
             " ".join("v%d=%d" % (k, v)
                      for k, v in sorted(trace["hints"].items()))))
    print("\nADJACENCY (each undirected edge appears in BOTH its endpoints'"
          " rows)")
    for i in range(n):
        print("  v%-3d : %s" % (i, " ".join("v%d" % j for j in adj[i])))

    print("\nDEDUCTION ROUNDS (one round = naked-single elimination applied to"
          " every node at once)")
    for r, events in enumerate(trace["rounds"]):
        print("  round %d: %s" % (r + 1, " ".join("v%d<-%d" % (ev[0], ev[1])
                                                  for ev in events)))
    print("  T = %d rounds;  n_blank = %d nodes were deduced"
          % (len(trace["rounds"]), sum(len(e) for e in trace["rounds"])))

    print("\nTOKEN STREAM (%d tokens; the analytic length is"
          " N(d+3) + T + 4 = %d)" % (len(toks), n * (d + 3)
                                     + len(trace["rounds"]) + 4))
    line = []
    for t in toks:
        line.append(name(vocab, int(t)))
        if name(vocab, int(t)).startswith("BEGIN") or len(line) >= 14:
            print("  " + " ".join(line))
            line = []
    if line:
        print("  " + " ".join(line))
    print("\nraw ids:", " ".join(str(int(t)) for t in toks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
