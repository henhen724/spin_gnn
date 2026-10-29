"""Does it CHAIN? The round-0 counterfactual, moved to deduced neighbours.

sudoku_circuits tests one thing: a node forced in round 0, whose every known
neighbour is a HINT. Recolour the binding hint and the leak-free 1.0000 model
flips to DD's new answer on 350 of 350 cases. That establishes the elimination
rule over GIVEN facts.

DD is iterative, and that is not what round 0 tests. A node v forced in round
k >= 1 is forced partly by neighbours that were themselves DEDUCED in earlier
rounds -- facts the model had to produce, not read. Using those is the
difference between applying an inference rule once and running the algorithm.
Behaviourally the model must already chain somehow: finding 23 has every
rho=0.60 depth bin at 1.0000, and a node that only becomes determined after
five rounds cannot be got right without the preceding four. But "must be doing
something that integrates rounds" is an inference from accuracy, not a causal
measurement, and this project has been burned by exactly that step before.

THE TEST. Find v deduced in round k >= 1 whose forcing set covers q-1 colours,
where the colour a is held by exactly one known neighbour w AND w was itself
DEDUCED in a round < k. Edit w's colour token IN THE REASONING SECTION -- not a
hint -- so that a becomes free. DD's answer for v then flips from c to a. If the
model's prediction at v's position follows, it is eliminating against a fact it
derived, which is chaining.

The round-0 test and this one differ in exactly one respect: which token is
edited, a hint or a previously emitted deduction. That is deliberate, so the
two flip rates are directly comparable and any gap between them is about the
PROVENANCE of the fact rather than about the rule.

HONEST CAVEAT, stated up front. Editing an emitted deduction makes the prefix
inconsistent with the graph and hints: the model is being shown a reasoning
step it would not itself have written. A model that notices the contradiction
and reverts to its own value would score as "does not chain" here while still
chaining perfectly in free generation. So a LOW flip rate is genuinely
ambiguous between "does not chain" and "detects the edit", and the
`reverts_to_original` column is recorded to tell those apart: reverting to c is
the signature of detection, while a third colour is neither. A HIGH flip rate
has no such ambiguity -- it can only mean the edited fact was used.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .sudoku_tokens import encode_instance
from .train_sudoku_transformer import build_instances, load_checkpoint


def reasoning_node_positions(tokens, vocab):
    """{node: index of its node token} over EVERY round, not just round 0.

    sudoku_circuits.reasoning_color_position stops at the first non-node token,
    which is the next BEGIN_ROUND, so it only ever sees round 0. Chaining needs
    later rounds by definition."""
    start = int(np.flatnonzero(tokens == vocab.BEGIN_REASONING)[0]) + 1
    pos, i, rnd = {}, start, -1
    while i < len(tokens):
        t = int(tokens[i])
        if t == vocab.BEGIN_ROUND:
            rnd += 1
            i += 1
            continue
        if not vocab.is_node_token(t) or i + 1 >= len(tokens):
            break
        pos[int(vocab.node_index(t))] = (i, rnd)
        i += 2
    return pos


def find_chained_deductions(adj, colors, hints, trace, vocab, max_per_instance=4):
    """Nodes forced in round k>=1 by a neighbour that was DEDUCED in round <k,
    where recolouring that neighbour flips DD's answer."""
    rounds = trace["rounds"]
    if len(rounds) < 2:
        return []
    # colour known BEFORE round k, and the round each deduction landed in
    known_before, deduced_round = dict(hints), {}
    out = []
    for k, rnd in enumerate(rounds):
        if k >= 1:
            for node, c in rnd:
                node, c = int(node), int(c)
                nbrs = [int(u) for u in adj[node]]
                known = {u: known_before[u] for u in nbrs if u in known_before}
                covered = sorted(set(known.values()))
                if len(covered) != vocab.n_colors - 1 or c in covered:
                    continue
                counts = {col: sum(1 for v in known.values() if v == col) for col in covered}
                singles = [col for col, k2 in counts.items() if k2 == 1]
                if not singles:
                    continue
                # the binding neighbour must be DEDUCED, not hinted -- that is
                # the whole point of this module
                for a in singles:
                    w = next(u for u, col in known.items() if col == a)
                    if w in deduced_round:
                        out.append({
                            "node": node, "round": k, "true_color": c,
                            "flipped_color": a, "recolored_nbr": w,
                            "nbr_round": deduced_round[w],
                            "n_known": len(known),
                        })
                        break
                if len(out) >= max_per_instance:
                    return out
        for node, c in rnd:
            known_before[int(node)] = int(c)
            deduced_round[int(node)] = k
    return out


def build_chain_pair(adj, trace, vocab, case):
    """(clean, corrupt, predict_pos). One REASONING colour token differs."""
    clean = encode_instance(adj, trace, vocab)
    pos_map = reasoning_node_positions(clean, vocab)
    w_pos = pos_map[case["recolored_nbr"]][0] + 1   # the colour follows the node
    v_pos = pos_map[case["node"]][0]                # predicts v's colour
    assert vocab.is_color_token(clean[w_pos]), "expected a colour token after the node"
    assert int(vocab.color_index(clean[w_pos])) == case["flipped_color"]
    corrupt = clean.copy()
    corrupt[w_pos] = vocab.color_token(case["true_color"])
    assert (clean != corrupt).sum() == 1, "counterfactual must be a one-token change"
    return clean, corrupt, v_pos, w_pos


def run_chain_behavior(model, vocab, device, cases):
    rows = []
    for adj, colors, hints, trace, case in cases:
        clean, corrupt, v_pos, w_pos = build_chain_pair(adj, trace, vocab, case)
        with torch.no_grad():
            lc = model(torch.from_numpy(clean).unsqueeze(0).to(device))
            lk = model(torch.from_numpy(corrupt).unsqueeze(0).to(device))
        true_tok = vocab.color_token(case["true_color"])
        flip_tok = vocab.color_token(case["flipped_color"])
        pred_clean = int(lc[0, v_pos].argmax())
        pred_corrupt = int(lk[0, v_pos].argmax())
        rows.append({
            "node": case["node"], "round": case["round"],
            "binding_nbr": case["recolored_nbr"], "nbr_round": case["nbr_round"],
            "n_known": case["n_known"],
            "clean_correct": pred_clean == true_tok,
            "corrupt_flipped": pred_corrupt == flip_tok,
            # the detection signature: it kept its own answer instead
            "reverts_to_original": pred_corrupt == true_tok,
            "clean_logit_diff": float(lc[0, v_pos][flip_tok] - lc[0, v_pos][true_tok]),
            "corrupt_logit_diff": float(lk[0, v_pos][flip_tok] - lk[0, v_pos][true_tok]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n-instances", type=int, default=2000)
    ap.add_argument("--n-cases", type=int, default=350)
    ap.add_argument("--seed-start", type=int, default=600_000_000,
                    help="same stream as sudoku_circuits, so round-0 and chained cases "
                         "come from the same instances and the flip rates compare")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    model, vocab, config = load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    sudoku_cfg = SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        rho=config["rho"], rho_max=config["rho_max"],
        scratchpad=config.get("scratchpad", False), hint_mode=config.get("hint_mode", "removal"),
        shuffle_adj=bool(config.get("shuffle_adj", False)),
        shuffle_events=bool(config.get("shuffle_events", False)),
    )

    cases = []
    for adj, colors, hints, trace in build_instances(sudoku_cfg, args.n_instances, args.seed_start):
        for case in find_chained_deductions(adj, colors, hints, trace, vocab):
            cases.append((adj, colors, hints, trace, case))
    print(f"{len(cases)} chained deductions (round>=1, binding neighbour deduced) "
          f"from {args.n_instances} instances")
    if not cases:
        raise SystemExit("no chained cases found -- check rho/rho_max give multi-round traces")
    cases = cases[:args.n_cases]
    os.makedirs(args.out_dir, exist_ok=True)

    rows = run_chain_behavior(model, vocab, args.device, cases)
    n = len(rows)
    clean_acc = sum(r["clean_correct"] for r in rows) / n
    flip = sum(r["corrupt_flipped"] for r in rows) / n
    revert = sum(r["reverts_to_original"] for r in rows) / n
    both = sum(r["clean_correct"] and r["corrupt_flipped"] for r in rows) / n
    print(f"\nCHAINED BEHAVIOR over {n} counterfactuals on DEDUCED neighbours")
    print(f"  clean_correct         {clean_acc:.3f}")
    print(f"  corrupt_flipped       {flip:.3f}   (chance ~{1 / vocab.n_colors:.3f})")
    print(f"  reverts_to_original   {revert:.3f}   (the 'detected the edit' signature)")
    print(f"  both (chains the rule) {both:.3f}")
    print(f"  mean logit_diff  clean {np.mean([r['clean_logit_diff'] for r in rows]):+.3f}"
          f"  corrupt {np.mean([r['corrupt_logit_diff'] for r in rows]):+.3f}")

    by_round = {}
    for r in rows:
        by_round.setdefault(r["round"], []).append(r)
    print("\n  by the round v was deduced in")
    for k in sorted(by_round):
        g = by_round[k]
        print(f"    round {k}: n={len(g):4d}  clean {sum(x['clean_correct'] for x in g) / len(g):.3f}"
              f"  flipped {sum(x['corrupt_flipped'] for x in g) / len(g):.3f}")

    with open(os.path.join(args.out_dir, "chain_behavior.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, "chain_summary.json"), "w") as f:
        json.dump({"checkpoint": args.checkpoint, "n_cases": n,
                   "clean_correct": clean_acc, "corrupt_flipped": flip,
                   "reverts_to_original": revert, "both": both,
                   "chance": 1 / vocab.n_colors,
                   "by_round": {str(k): {"n": len(g),
                                         "clean": sum(x["clean_correct"] for x in g) / len(g),
                                         "flipped": sum(x["corrupt_flipped"] for x in g) / len(g)}
                                for k, g in by_round.items()},
                   "seed_start": args.seed_start}, f, indent=2)
    print(f"\nwrote {args.out_dir}/chain_behavior.csv ({n} rows)")


if __name__ == "__main__":
    main()
