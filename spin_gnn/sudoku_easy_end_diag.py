"""Why `solve_rate_k0` collapses while the shared pool keeps rising.

The sweep reports solve_rate for k = 0..4 *uncolored* nodes and calls their
mean `mean_o1_solve_rate`, read so far as "the easy end". For the
n_train=80000 model those numbers went backwards hard (k0 0.60 -> 0.07,
mean_o1 0.918 -> 0.776) while shared-pool solve_rate rose 0.9847 -> 0.9960,
which was logged as an unexplained contradiction.

It is not a contradiction. `eval_solve_by_rho` pins a bin's target density at
rho = (n_nodes - k)/n_nodes, and `removal_path` stops at
int(np.floor(rho * N)) hints, so the training ceiling rho_max=0.95 is 22
hints out of 24 -- exactly the k=2 bin. So of the five "easy" bins:

    k=0   24 hints, rho 1.0000, DD depth T=0   OUTSIDE training support
    k=1   23 hints, rho 0.9583, T=1            OUTSIDE training support
    k=2   22 hints, rho 0.9167, T>=1           the training ceiling
    k=3   21 hints, rho 0.8750                 inside
    k=4   20 hints, rho 0.8333                 inside

k=0 is doubly unusual: with every node hinted the true trace has *zero*
rounds, a sequence shape no training instance has (every training instance
needs at least one deduction round). So `solve_rate_k0` is not an easiness
probe at all, it is an out-of-distribution extrapolation probe, and
`mean_o1` mixes two OOD bins in with three in-distribution ones.

This module measures that directly: solve_rate per hint count across and
beyond the training ceiling, plus, for every failure, which of three things
went wrong -- the model invented reasoning rounds the instance does not have,
it called the outcome wrong, or it emitted something unparseable. A flat
in-distribution profile with a cliff at 23-24 hints says the easy end never
collapsed; a model that is also bad at k>=2 would say it did.
"""
import argparse
import csv
import json
import os
from dataclasses import replace

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .sudoku_tokens import check_solution, decode, encode_instance
from .train_sudoku_transformer import build_instances, generate, load_checkpoint


def classify_failure(decoded, n_rounds_true, solved):
    """One label per instance, in the order a failure is diagnosed.

    `extra_rounds` is the OOD signature being tested for: the model keeps
    deducing on an instance that has nothing left to deduce.
    """
    if solved:
        return "solved"
    if not decoded["well_formed"]:
        return "malformed"
    if decoded["outcome"] != "SOLVED":
        return "wrong_outcome"
    if len(decoded["rounds"]) > n_rounds_true:
        return "extra_rounds"
    if len(decoded["rounds"]) < n_rounds_true:
        return "missing_rounds"
    return "wrong_colors"


LABELS = ("malformed", "wrong_outcome", "extra_rounds", "missing_rounds", "wrong_colors")


def diagnose_bin(model, vocab, device, cfg, n_hints, n_instances, seed_start, max_new_tokens):
    rho = n_hints / cfg.n_nodes
    instances = build_instances(replace(cfg, rho=rho, rho_max=rho), n_instances, seed_start)
    counts = dict.fromkeys(LABELS, 0)
    n_solved = n_truncated = 0
    rounds_emitted, rounds_true, realized_hints = [], [], []

    for adj, colors, hints, trace in instances:
        full_tokens = encode_instance(adj, trace, vocab)
        reasoning_start = int(np.flatnonzero(full_tokens == vocab.BEGIN_REASONING)[0]) + 1
        generated = generate(model, full_tokens[:reasoning_start], vocab, max_new_tokens,
                             device=device, temperature=0.0)
        decoded = decode(generated, vocab)

        continuation = generated[reasoning_start:]
        if not ((continuation == vocab.SOLVED) | (continuation == vocab.UNRESOLVED)).any():
            n_truncated += 1

        true_outcome = "SOLVED" if trace["resolved"] else "UNRESOLVED"
        solved = (decoded["outcome"] == true_outcome
                  and check_solution(decoded, colors, vocab.n_nodes))
        n_solved += bool(solved)
        label = classify_failure(decoded, len(trace["rounds"]), solved)
        if label != "solved":
            counts[label] += 1
        rounds_emitted.append(len(decoded["rounds"]))
        rounds_true.append(len(trace["rounds"]))
        realized_hints.append(len(hints))

    n = len(instances)
    n_failed = n - n_solved
    row = {
        "n_hints_target": n_hints,
        "mean_hints_realized": sum(realized_hints) / n,
        "rho_realized": sum(realized_hints) / (n * cfg.n_nodes),
        "n_instances": n,
        "solve_rate": n_solved / n,
        "truncated_rate": n_truncated / n,
        "mean_rounds_true": sum(rounds_true) / n,
        "mean_rounds_emitted": sum(rounds_emitted) / n,
        "n_failed": n_failed,
    }
    for label in LABELS:
        row[f"n_{label}"] = counts[label]
        # Share of the failures, not of the bin: with solve_rate near 1.0 the
        # per-bin fractions are all ~0 and say nothing about the mechanism.
        row[f"frac_fail_{label}"] = counts[label] / n_failed if n_failed else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--hint-counts", type=int, nargs="+", default=[24, 23, 22, 21, 20, 18, 16],
                    help="absolute hint counts; the training ceiling at N=24, rho_max=0.95 is 22")
    ap.add_argument("--n-instances", type=int, default=300)
    # Disjoint from the shared pool (5e10) and from every training seed.
    ap.add_argument("--seed-start", type=int, default=60_000_000_000)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ckpt in args.checkpoint:
        model, vocab, config = load_checkpoint(ckpt, device=device)
        name = (f"{os.path.basename(os.path.dirname(os.path.abspath(ckpt.rstrip('/'))))}"
                f"/{os.path.basename(ckpt.rstrip('/'))}")
        cfg = SudokuReasoningConfig(
            n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
            rho=config["rho"], rho_max=config["rho_max"],
            hint_mode=config.get("hint_mode", "removal"),
            scratchpad=config.get("scratchpad", False),
        )
        # removal_path snapshots at int(np.floor(rho * N)) hints, so the most
        # heavily hinted instance training ever saw is floor(rho_max * N) --
        # 22 of 24 at rho_max=0.95, NOT the 23 that round() would suggest.
        ceiling = int(np.floor(config["rho_max"] * config["n_nodes"]))
        print(f"\n=== {name} (trained rho in [{cfg.rho}, {cfg.rho_max}] "
              f"-> at most {ceiling}/{config['n_nodes']} hints) ===", flush=True)
        for n_hints in args.hint_counts:
            row = diagnose_bin(model, vocab, device, cfg, n_hints, args.n_instances,
                               args.seed_start, config["max_len"])
            row["checkpoint"] = name
            row["train_hint_ceiling"] = ceiling
            row["in_distribution"] = n_hints <= ceiling
            rows.append(row)
            print(f"  {n_hints:2d} hints (rho {row['rho_realized']:.4f}, "
                  f"{'in' if row['in_distribution'] else 'OUT of'} distribution): "
                  f"solve_rate={row['solve_rate']:.3f}  T_true={row['mean_rounds_true']:.2f} "
                  f"T_emitted={row['mean_rounds_emitted']:.2f}  "
                  f"failures={row['n_failed']}"
                  + (f" ({row['frac_fail_extra_rounds']:.0%} extra_rounds, "
                     f"{row['frac_fail_wrong_colors']:.0%} wrong_colors, "
                     f"{row['frac_fail_wrong_outcome']:.0%} wrong_outcome)"
                     if row["n_failed"] else ""), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    field_order = ["checkpoint", "n_hints_target", "in_distribution", "train_hint_ceiling"]
    fields = field_order + [k for k in rows[0] if k not in field_order]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out_csv} ({len(rows)} bins)")

    with open(os.path.splitext(args.out_csv)[0] + "_args.json", "w") as f:
        json.dump(vars(args), f, indent=1)


if __name__ == "__main__":
    main()
