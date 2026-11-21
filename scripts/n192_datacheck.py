"""Is the N=192 TRAINING DATA sound, or did the model correctly learn that
`reason_color` is unpredictable?

Stage NSPLIT measured, on both N=192 seeds and at every one of the five
densities, a `reason_color` cross-entropy of 1.3863 nats against
ln(4) = 1.386294 and an accuracy of 0.248 against 1/q = 0.25. The same
quantity at N=96 is 0.0037 nats / 0.9991. A model sitting exactly on the
marginal prior has two possible causes and they have opposite consequences:

  (A) the model never learned colour deduction at this size -- a result;
  (B) the deduced colour is not actually forced by the context in the N=192
      training data, in which case the uniform prediction is CORRECT and
      1.3863 is the Bayes floor, not a failure -- a bug.

Under (B) nothing in the ladder may be reported as a model limitation. So
this is checked before anything is written up.

THE TEST. For every deduction (node, c) emitted in round r, DD's naked-single
rule says the colours of `node`'s neighbours that were known STRICTLY BEFORE
round r must cover exactly the q-1 colours other than c. Then c is the unique
legal colour and a perfect predictor scores 0 nats. We check, per deduction:

  F1  the neighbour colours known before round r are exactly {0..q-1} \\ {c}
      -- the deduction is FORCED (so the Bayes NLL is 0, not ln q);
  F2  c equals the planted colour at that node (the trace is correct);
  F3  the context needed for F1 is inside the emitted sequence -- i.e. the
      adjacency row of `node` and every one of those earlier deductions were
      already written. This is what makes the token stream self-contained.

N=96 is regenerated identically as the control: whatever this script reports
for N=192 it must report nothing worse for N=96, where the model reaches
0.9991.

Also reported: the marginal colour distribution (a skewed marginal would put
the prior BELOW ln q and change the reading of 1.3863), and the encoded
sequence length against the run's max_len of 1984, since a truncated training
sequence would silently delete the reasoning tail.

Run under the TORCH module set (build_instances imports torch transitively):
  module purge --force
  module load devel math python/3.12.1 py-pytorch/2.4.1_py312 \\
      py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312
  python3 -u -m scripts.n192_datacheck --n-instances 40
"""
import argparse
import collections
import math

import numpy as np

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_tokens import Vocab, encode_instance
from spin_gnn.train_sudoku_transformer import build_instances

# The ladder cell's own generation settings, copied from
# scripts/slurm/sudoku2_n192_ladder_2026-11-20.sbatch. Only n_nodes differs
# between the two sizes, which is the whole point of the control.
DEGREE, N_COLORS = 5, 4
RHO, RHO_MAX = 0.3, 0.95
MAX_LEN = 1984  # resolve_max_len_margin(...) == 386 margin at N=192


def check_size(n_nodes, n_instances, seed_start):
    cfg = SudokuReasoningConfig(
        n_nodes=n_nodes, degree=DEGREE, n_colors=N_COLORS,
        rho=RHO, rho_max=RHO_MAX, fall=True, shuffle_adj=True,
    )
    vocab = Vocab(n_nodes, N_COLORS, DEGREE)
    inst = build_instances(cfg, n_instances, seed_start)

    n_ded = 0
    bad_forced = []       # F1
    bad_planted = []      # F2
    bad_context = []      # F3
    colour_counts = collections.Counter()
    lengths = []
    all_colours = set(range(N_COLORS))

    for idx, (adj, colors, hints, trace) in enumerate(inst):
        known = {int(k): int(v) for k, v in hints.items()}
        # F3: the graph section emits every node's row before BEGIN_REASONING,
        # so adjacency is always available; what has to be checked is that the
        # neighbours forcing this deduction were themselves already emitted.
        emitted = set(known)
        for r, round_events in enumerate(trace["rounds"]):
            for node, c in round_events:
                node, c = int(node), int(c)
                n_ded += 1
                colour_counts[c] += 1
                nbr_known = {known[int(u)] for u in adj[node] if int(u) in known}
                if nbr_known != all_colours - {c}:
                    bad_forced.append((idx, r, node, c, sorted(nbr_known)))
                if c != int(colors[node]):
                    bad_planted.append((idx, r, node, c, int(colors[node])))
                missing = [int(u) for u in adj[node]
                           if int(u) in known and int(u) not in emitted]
                if missing:
                    bad_context.append((idx, r, node, missing))
            for node, c in round_events:
                known[int(node)] = int(c)
                emitted.add(int(node))
        lengths.append(len(encode_instance(adj, trace, vocab)))

    return {
        "n_nodes": n_nodes, "n_instances": n_instances, "n_ded": n_ded,
        "bad_forced": bad_forced, "bad_planted": bad_planted,
        "bad_context": bad_context, "colour_counts": colour_counts,
        "lengths": lengths,
    }


def report(res):
    n = res["n_ded"]
    print("=" * 66)
    print("N=%d   %d instances   %d deductions"
          % (res["n_nodes"], res["n_instances"], n))
    print("=" * 66)
    for key, label in (("bad_forced", "F1 deduction NOT forced by prior-round neighbours"),
                       ("bad_planted", "F2 deduced colour != planted colour"),
                       ("bad_context", "F3 forcing neighbour not yet emitted")):
        bad = res[key]
        verdict = "PASS" if not bad else "FAIL"
        print("  %-52s %s  %d/%d" % (label, verdict, len(bad), n))
        for row in bad[:3]:
            print("        e.g. %r" % (row,))

    counts = res["colour_counts"]
    tot = sum(counts.values())
    print("\n  marginal over reason_color:")
    ent = 0.0
    for c in range(N_COLORS):
        p = counts[c] / tot
        ent -= p * math.log(p) if p else 0.0
        print("        COLOR_%d  %.4f" % (c, p))
    print("        marginal entropy %.6f nats   (ln q = %.6f)"
          % (ent, math.log(N_COLORS)))
    print("        -> a context-blind predictor scores %.4f nats here" % ent)

    L = np.array(res["lengths"])
    print("\n  encoded length: min %d  mean %.1f  max %d   (max_len %d)"
          % (L.min(), L.mean(), L.max(), MAX_LEN))
    print("        over max_len: %d of %d" % (int((L > MAX_LEN).sum()), len(L)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-instances", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[96, 192])
    args = ap.parse_args()
    for n_nodes in args.sizes:
        report(check_size(n_nodes, args.n_instances, args.seed_start))
        print()


if __name__ == "__main__":
    main()
