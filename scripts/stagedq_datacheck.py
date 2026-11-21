"""STAGE DQ preflight -- is the (d=4, q=3) pipeline sound before it trains?

scripts/n192_datacheck.py established the habit this follows: when a cell is
about to be read as a statement about the MODEL, first rule out that it is a
statement about the DATA. That script is frozen because the published N=192
verdict and deck cite it; this is the same three checks with (degree,
n_colors, rho, rho_max) exposed, so one committed scorer covers both
ensembles.

(d=4, q=3) has never been tokenized or trained here -- every cell to date is
d=5, q=4. Three things change and each could silently break:

  * the vocabulary. Vocab's n_special is 6 + n_colors, so node_offset moves
    10 -> 9 and size moves N+10 -> N+9. Nothing in sudoku_tokens.py hardcodes
    either, but "nothing greps" is not "it round-trips".
  * legality. q | N and q(q-1) | d*N becomes 3 | N and 6 | 4N, both of which
    reduce to 3 | N -- against 12 | N at q=4 d=5. N=96 and N=120 are legal.
  * the chance level. A context-blind predictor on reason_color scores ln q,
    which moves 1.3863 -> 1.0986. Finding 116's diagnostic number is NOT
    portable, and the marginal measured here is what a d4q3 cell gets
    compared against.

F1/F2/F3 are exactly n192_datacheck's. Added here:

  F4  encode -> decode round-trips to the same token ids, which is the check
      that actually exercises the moved node_offset;
  D   the DEPTH AND DENSITY MIX the sampler really produces, because stage
      DQ's whole claim is that this ensemble reaches DD depth at small N.
      Reported against rho_c ~ 0.25, measured in runs/stagedq_ensemble_*/
      as the density where T/log N starts to fan out across sizes (the same
      diagnostic recovers rho_c ~ 0.465 on the d=5 q=4 arm, which is the
      published value, so it is calibrated).

Run under the TORCH module set (build_instances imports torch transitively):
  module purge --force
  module load devel math python/3.12.1 py-pytorch/2.4.1_py312 \\
      py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312
  python3 -u -m scripts.stagedq_datacheck --n-instances 40
"""
import argparse
import collections
import math

import numpy as np

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_tokens import Vocab, encode_instance
from spin_gnn.train_sudoku_transformer import build_instances

RHO_C = {(5, 4): 0.465, (4, 3): 0.25}


def check_cell(n_nodes, degree, n_colors, rho, rho_max, n_instances,
               seed_start, max_len):
    if n_nodes % n_colors or (degree * n_nodes) % (n_colors * (n_colors - 1)):
        raise ValueError(
            f"illegal cell: need q | N and q(q-1) | d*N "
            f"(N={n_nodes}, d={degree}, q={n_colors})")
    cfg = SudokuReasoningConfig(
        n_nodes=n_nodes, degree=degree, n_colors=n_colors,
        rho=rho, rho_max=rho_max, fall=True, shuffle_adj=True,
    )
    vocab = Vocab(n_nodes, n_colors, degree)
    inst = build_instances(cfg, n_instances, seed_start)

    n_ded = 0
    bad_forced, bad_planted, bad_context, bad_roundtrip = [], [], [], []
    colour_counts = collections.Counter()
    lengths, depths, densities = [], [], []
    all_colours = set(range(n_colors))

    for idx, (adj, colors, hints, trace) in enumerate(inst):
        known = {int(k): int(v) for k, v in hints.items()}
        emitted = set(known)
        densities.append(len(known) / n_nodes)
        depths.append(len(trace["rounds"]))
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

        toks = encode_instance(adj, trace, vocab)
        lengths.append(len(toks))
        # F4: every node id in the stream must decode back inside [0, N).
        for t in toks:
            if vocab.is_node_token(t):
                i = vocab.node_index(t)
                if not (0 <= i < n_nodes) or vocab.node_token(i) != t:
                    bad_roundtrip.append((idx, int(t), int(i)))
                    break

    return dict(n_nodes=n_nodes, degree=degree, n_colors=n_colors,
                rho=rho, rho_max=rho_max, n_instances=n_instances,
                n_ded=n_ded, bad_forced=bad_forced, bad_planted=bad_planted,
                bad_context=bad_context, bad_roundtrip=bad_roundtrip,
                colour_counts=colour_counts, lengths=lengths,
                depths=depths, densities=densities, max_len=max_len,
                vocab_size=vocab.size, node_offset=vocab.node_offset)


def report(res):
    q = res["n_colors"]
    n = res["n_ded"]
    print("=" * 70)
    print("N=%d  d=%d  q=%d   rho~U[%.2f, %.2f]   %d instances   %d deductions"
          % (res["n_nodes"], res["degree"], q, res["rho"], res["rho_max"],
             res["n_instances"], n))
    print("    vocab size %d   node_offset %d   (n_special = 6 + q = %d)"
          % (res["vocab_size"], res["node_offset"], 6 + q))
    print("=" * 70)
    ok = True
    for key, label in (
            ("bad_forced", "F1 deduction NOT forced by prior-round neighbours"),
            ("bad_planted", "F2 deduced colour != planted colour"),
            ("bad_context", "F3 forcing neighbour not yet emitted"),
            ("bad_roundtrip", "F4 node token does not round-trip")):
        bad = res[key]
        ok &= not bad
        print("  %-52s %s  %d/%d"
              % (label, "PASS" if not bad else "FAIL", len(bad), n))
        for row in bad[:3]:
            print("        e.g. %r" % (row,))

    counts = res["colour_counts"]
    tot = sum(counts.values())
    print("\n  marginal over reason_color (the finding-116 chance level):")
    ent = 0.0
    for c in range(q):
        p = counts[c] / tot
        ent -= p * math.log(p) if p else 0.0
        print("        COLOR_%d  %.4f" % (c, p))
    print("        marginal entropy %.6f nats   (ln q = %.6f)"
          % (ent, math.log(q)))
    print("        -> a context-blind predictor scores %.4f nats here" % ent)
    print("        -> a cell in THIS ensemble sitting at %.4f on reason_color"
          % ent)
    print("           has finding 116's failure. The number is per-ensemble:")
    print("           ln 3 = 1.0986 at q=3, ln 4 = 1.3863 at q=4, and neither")
    print("           may be quoted against the other.")

    L = np.array(res["lengths"])
    print("\n  encoded length: min %d  mean %.1f  max %d   (max_len %d)"
          % (L.min(), L.mean(), L.max(), res["max_len"]))
    print("        over max_len: %d of %d" % (int((L > res["max_len"]).sum()),
                                              len(L)))

    T = np.array(res["depths"])
    R = np.array(res["densities"])
    rc = RHO_C.get((res["degree"], q), float("nan"))
    slow = float((R < rc).mean())
    print("\n  D  depth/density mix actually sampled:")
    print("        T      min %d  median %.1f  mean %.1f  max %d"
          % (T.min(), float(np.median(T)), T.mean(), T.max()))
    print("        rho    min %.3f  median %.3f  max %.3f"
          % (R.min(), float(np.median(R)), R.max()))
    print("        rho_c ~ %.3f  ->  %.1f%% of instances are SLOW phase"
          % (rc, 100 * slow))
    print("        deepest decile mean T = %.1f"
          % T[np.argsort(T)[-max(1, len(T) // 10):]].mean())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-instances", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=1024)
    # cells: N:d:q:rho:rho_max
    ap.add_argument("--cells", nargs="+",
                    default=["96:4:3:0.14:0.95",
                             "120:4:3:0.14:0.95",
                             "96:5:4:0.30:0.95"])
    args = ap.parse_args()
    all_ok = True
    for spec in args.cells:
        N, d, q, r0, r1 = spec.split(":")
        res = check_cell(int(N), int(d), int(q), float(r0), float(r1),
                         args.n_instances, args.seed_start, args.max_len)
        all_ok &= report(res)
        print()
    print("OVERALL: %s" % ("PASS" if all_ok else "FAIL"))


if __name__ == "__main__":
    main()
