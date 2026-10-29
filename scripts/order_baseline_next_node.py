"""How much of `next_node` at pre_node positions is explained by EMISSION ORDER?

Finding 21 left one thing unexplained: at pre_node positions -- where the node
about to be emitted is not yet on the page -- the dictionaries find plenty of
next_node-family features in EVERY arm (at layer 12: 53 features / best F1
0.816 for the 1.0000 model, 15 / 0.545 for its 0.8540 twin, 60 / 0.741 for the
leaked one). Finding 16 measured that round events are emitted in
ASCENDING NODE INDEX for 100% of rounds, and `shuffle_adj` permutes each node's
neighbour list only (spin_gnn/config.py:184) -- it does not touch event order.
So "the index goes up" is an uncontrolled explanation for every pre_node
next_node feature in this project, in BOTH arms.

This script puts a number on the ceiling, with no model and no SAE: it scores
hand-written order rules against the same `next_node` predicates, on the same
metric (max over v of F1 between "rule predicts v" and "next_node == v"), so
the numbers sit on the same scale as `best_next_node_f1` in sae_summary.csv.

The rules, in increasing order of how much DD work they presuppose:

  const       always predict the single most frequent node index. The floor.
  succ        predict prev + 1, where prev is the last node emitted in this
              round (-1 at the start of a round). Pure counting: it needs
              nothing but the previous token.
  min_unknown smallest index > prev whose colour is not yet known. Uses the
              known set, which is read straight off the hints and the emitted
              deductions -- no frontier, so no deduction. THIS IS THE CONTROL:
              it is what a model can do with bookkeeping and ascending order
              alone, and the leaked model gets its bookkeeping cheaply.
  min_remain  smallest index among this round's not-yet-emitted frontier
              members. Exact by construction (that IS ascending order over the
              frontier), so it must score 1.000 -- a self-check on the harness,
              and the demonstration that the only hard part is the frontier.

Read it as: anything `min_unknown` already achieves is not evidence of
deduction. The gap between `min_unknown` and 1.000 is what a pre_node
next_node feature has to earn.

Light enough for a login node (no GPU, no model): ~1 min for 1000 instances.
"""
import argparse
import json
import sys

import numpy as np

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_activations import dd_state_labels
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab


def max_f1_over_nodes(pred, truth, n_nodes):
    """max over v of F1("pred == v", "truth == v"), the sae_summary.csv metric.

    A rule that is right often but spreads its hits over all 24 node ids can
    still score low here, exactly as a feature would: the SAE metric asks for
    one predicate per feature, not for overall accuracy."""
    best, best_v = 0.0, -1
    for v in range(n_nodes):
        p, t = pred == v, truth == v
        tp = int(np.sum(p & t))
        if tp == 0:
            continue
        f1 = 2 * tp / (int(p.sum()) + int(t.sum()))
        if f1 > best:
            best, best_v = f1, v
    return best, best_v


def main():
    ap = argparse.ArgumentParser()
    # Defaults mirror the checkpoints of findings 19/21: N=24, degree 5, q=4.
    ap.add_argument("--n-nodes", type=int, default=24)
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--n-colors", type=int, default=4)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--rho-max", type=float, default=0.95)
    ap.add_argument("--n-instances", type=int, default=1000)
    # Written so a deck can read these numbers instead of hand-typing them.
    ap.add_argument("--json", default=None, help="also write the summary here")
    # Same stream as sudoku_sae.py (instance i at seed_start + i), so this is a
    # strict prefix of the positions the SAE was scored on.
    ap.add_argument("--seed-start", type=int, default=600_000_000)
    args = ap.parse_args()

    cfg = SudokuReasoningConfig(
        n_nodes=args.n_nodes, degree=args.degree, n_colors=args.n_colors,
        rho=args.rho, rho_max=args.rho_max, hint_mode="removal",
        # shuffle_adj changes adj[v] only; nothing this script measures reads
        # adj, so the order statistics below hold in both arms. Set True so the
        # instances are the leak-free ones by default.
        shuffle_adj=True,
    )
    vocab = build_vocab(cfg)
    print(f"N={cfg.n_nodes} degree={cfg.degree} q={cfg.n_colors} "
          f"instances={args.n_instances} seed_start={args.seed_start}")

    truth, preds = [], {k: [] for k in ("const", "succ", "min_unknown", "min_remain")}
    n_rounds_ascending = n_rounds = 0
    for adj, _colors, _hints, trace in build_instances(cfg, args.n_instances, args.seed_start):
        for rd in trace["rounds"]:
            ids = [int(u) for u, _ in rd]
            n_rounds += 1
            n_rounds_ascending += int(ids == sorted(ids))
        positions, lab = dd_state_labels(adj, trace, vocab)
        is_node = lab["kind_is_node"]
        # prev = the node emitted just before this one IN THIS ROUND; -1 at k=0.
        prev = np.full(len(positions), -1, dtype=np.int64)
        for p, info in enumerate(positions):
            if info["k"] > 0:
                prev[p] = lab["next_node"][p - 2]  # the previous deduction's pre_node row
        idx = np.flatnonzero(is_node)
        truth.append(lab["next_node"][idx])
        preds["succ"].append(np.clip(prev[idx] + 1, 0, cfg.n_nodes - 1))
        for name, mask_key, use_known in (("min_unknown", "known_mask", True),
                                          ("min_remain", "remaining_mask", False)):
            out = np.empty(idx.size, dtype=np.int64)
            for j, p in enumerate(idx):
                m = lab[mask_key][p].astype(bool)
                cand = np.flatnonzero(~m if use_known else m)
                cand = cand[cand > prev[p]]
                out[j] = cand[0] if cand.size else cfg.n_nodes - 1
            preds[name].append(out)

    truth = np.concatenate(truth)
    for k in ("succ", "min_unknown", "min_remain"):
        preds[k] = np.concatenate(preds[k])
    mode = int(np.bincount(truth, minlength=cfg.n_nodes).argmax())
    preds["const"] = np.full(truth.size, mode, dtype=np.int64)

    print(f"rounds with events in ascending node index: "
          f"{n_rounds_ascending}/{n_rounds} = {n_rounds_ascending / n_rounds:.4f}")
    print(f"pre_node positions scored: {truth.size}\n")
    print(f"{'rule':<12}{'accuracy':>10}{'max-over-v F1':>16}{'argmax v':>10}")
    summary = {"n_instances": args.n_instances, "n_rounds": n_rounds,
               "n_rounds_ascending": n_rounds_ascending,
               "ascending_frac": n_rounds_ascending / n_rounds,
               "n_positions": int(truth.size), "rules": {}}
    for k in ("const", "succ", "min_unknown", "min_remain"):
        acc = float(np.mean(preds[k] == truth))
        f1, v = max_f1_over_nodes(preds[k], truth, cfg.n_nodes)
        print(f"{k:<12}{acc:>10.4f}{f1:>16.4f}{v:>10d}")
        summary["rules"][k] = {"accuracy": acc, "max_f1": f1, "argmax_v": int(v)}
    print("\nCompare with sae_summary_by_kind.csv, pre_node, best_next_node_f1 at L12:")
    print("  leak-free 1.0000: 0.816 (53 features); its 0.8540 twin: 0.545 (15); leaked: 0.741 (60).")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
