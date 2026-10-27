"""Which rule does the model follow where candidate algorithms disagree?

Motivation. The next_node probe reaches 0.764 balanced accuracy on the best
checkpoint, which looks like evidence the model knows the deducible set. It is
not, on its own: solve_with_dd_trace emits each round as
`sorted((node, color) for node in nxt)`, so within a round node ids ascend.
A rule as dumb as "emit the smallest id you have not emitted yet" therefore
scores well above chance without representing deducibility at all. Any claim
about next_node needs that baseline subtracted.

Design. Agreement rates on all positions are uninformative, because every
sane rule agrees with DD most of the time -- that is why DD's answers look
learnable. So this scores only DISAGREEMENT positions: those where DD's next
node and a baseline heuristic's next node differ. There, the model's
prediction is a vote. Following DD on disagreements is evidence it computes
deducibility; following the heuristic is evidence it is doing bookkeeping
plus an ordering convention, which would also explain the probe results
(known_mask 0.92, frontier_mask 0.67).

Baselines, all of which require only the set of already-colored nodes -- the
thing probes show the model tracks very well -- and never the graph:

  ascending        smallest uncolored node id
  ascending_after  smallest uncolored id above the last emitted node, wrapping
                   to the smallest uncolored (exploits within-round sorting)
  random_uncolored a floor: how often agreement happens by chance given the
                   same uncolored set

And one that does use the graph, as the DD-flavored alternative:

  most_known_nbrs  uncolored node with the most known neighbours, ties to the
                   smallest id. Close to DD's forcing condition but not equal
                   to it, so it separates "uses neighbour counts" from
                   "applies the forcing rule".

Teacher-forced on the true trace, so position i's context is always correct
and the only thing varying is which rule explains the next token.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .sudoku_activations import reasoning_positions
from .sudoku_tokens import encode_instance
from .train_sudoku_transformer import build_instances, load_checkpoint


def _known_before(trace, round_index, k):
    """Nodes colored before the k-th deduction of round `round_index`."""
    known = dict(trace["hints"])
    for r in range(round_index):
        known.update(dict(trace["rounds"][r]))
    known.update(dict(trace["rounds"][round_index][:k]))
    return known


def baseline_predictions(adj, trace, n_nodes, round_index, k, rng):
    """Each baseline's next-node guess from the state before this deduction."""
    known = _known_before(trace, round_index, k)
    uncolored = [v for v in range(n_nodes) if v not in known]
    if not uncolored:
        return {}
    round_events = trace["rounds"][round_index]
    last_emitted = int(round_events[k - 1][0]) if k > 0 else -1
    above = [v for v in uncolored if v > last_emitted]
    n_known_nbrs = {v: sum(1 for u in adj[v] if int(u) in known) for v in uncolored}
    best = max(n_known_nbrs.values())
    return {
        "ascending": min(uncolored),
        "ascending_after": min(above) if above else min(uncolored),
        "random_uncolored": int(rng.choice(uncolored)),
        "most_known_nbrs": min(v for v in uncolored if n_known_nbrs[v] == best),
    }


def score_instance(model, adj, trace, vocab, device, rng):
    tokens = encode_instance(adj, trace, vocab)
    positions = reasoning_positions(tokens, vocab)
    with torch.no_grad():
        logits = model(torch.from_numpy(tokens).unsqueeze(0).to(device))[0]
    node_lo = vocab.node_offset
    node_hi = vocab.node_offset + vocab.n_nodes

    rows = []
    for info in positions:
        if info["kind"] != "pre_node":
            continue
        r, k = info["round"], info["k"]
        dd_answer = int(trace["rounds"][r][k][0])
        base = baseline_predictions(adj, trace, vocab.n_nodes, r, k, rng)
        if not base:
            continue
        # restrict to node tokens: the grammar only allows a node here, and
        # scoring over the full vocab would mix in grammar errors. argmax over
        # the slice is already the node index.
        pred_node = int(logits[info["pos"], node_lo:node_hi].argmax())
        for name, guess in base.items():
            if guess == dd_answer:
                continue  # agrees with DD; tells us nothing
            rows.append({
                "baseline": name, "round": r, "k": k,
                "dd_answer": dd_answer, "baseline_answer": guess,
                "model_pred": pred_node,
                "model_follows_dd": pred_node == dd_answer,
                "model_follows_baseline": pred_node == guess,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n-instances", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-start", type=int, default=400_000_000,
                    help="disjoint from training, probes (5e8/9e8), series (7e8), transition (8e8), circuits (6e8)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-csv", type=str, required=True)
    args = ap.parse_args()

    model, vocab, config = load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    cfg = SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        rho=config["rho"], rho_max=config["rho_max"],
        scratchpad=config.get("scratchpad", False), hint_mode=config.get("hint_mode", "removal"),
        # Analyzing a _shuf checkpoint without this feeds it the color-sorted
        # adjacency it was trained to ignore, i.e. the wrong distribution.
        shuffle_adj=bool(config.get("shuffle_adj", False)),
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    for adj, _colors, _hints, trace in build_instances(cfg, args.n_instances, args.seed_start):
        rows.extend(score_instance(model, adj, trace, vocab, args.device, rng))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nDISAGREEMENT positions only (where DD and the baseline differ)")
    print(f"{'baseline':>18} {'n':>7} {'follows DD':>12} {'follows base':>14} {'neither':>9}")
    summary = []
    for name in sorted({r["baseline"] for r in rows}):
        sub = [r for r in rows if r["baseline"] == name]
        dd = sum(r["model_follows_dd"] for r in sub) / len(sub)
        bl = sum(r["model_follows_baseline"] for r in sub) / len(sub)
        print(f"{name:>18} {len(sub):7d} {dd:12.3f} {bl:14.3f} {1 - dd - bl:9.3f}")
        summary.append({"baseline": name, "n": len(sub), "follows_dd": dd, "follows_baseline": bl})
    with open(args.out_csv.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
