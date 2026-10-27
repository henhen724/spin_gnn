"""Entry point: runs spin_gnn.sudoku_attention_distance over the validation
set of a trained sudoku-dd checkpoint and produces the comparison the user
asked for -- DD's required token-reference distance vs. the model's actual
attention-weight decay with distance.

Usage:
    python -m spin_gnn.sudoku_attention_distance_analysis \
        --checkpoint-dir $SCRATCH/spin_gnn/sudoku_hparam_sweep/d0064_h04_L12_ff01024_lr1e-03_pre \
        --n-instances 300 --seed-start 5000 \
        --out-dir $SCRATCH/spin_gnn/sudoku_attention_distance/<job_id>
"""
import argparse
import json
import os

import numpy as np
import torch

from spin_gnn.sudoku_attention_distance import (
    attention_weight_by_distance, forward_capture_attention, required_distances,
    solve_with_dd_provenance, token_positions,
)
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import SudokuReasoningConfig, build_instances, load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--n-instances", type=int, default=300)
    p.add_argument("--seed-start", type=int, default=5000)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, vocab, config = load_checkpoint(args.checkpoint_dir, device="cpu")
    cfg = SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        rho=config["rho"], rho_max=config["rho_max"],
        # All three were being left at their defaults, so a checkpoint trained
        # with any of them set was analyzed on instances it never saw. shuffle_adj
        # is the one that matters now: without it a _shuf checkpoint is fed the
        # color-sorted adjacency it was trained to ignore.
        hint_mode=config.get("hint_mode") or "removal",
        scratchpad=bool(config.get("scratchpad", False)),
        shuffle_adj=bool(config.get("shuffle_adj", False)),
    )
    instances = build_instances(cfg, n_instances=args.n_instances, seed_start=args.seed_start)

    max_distance = config["max_len"] - 1
    num_layers = config["num_layers"]
    sum_by_layer = np.zeros((num_layers, max_distance + 1))
    count_by_distance = np.zeros(max_distance + 1)

    all_graph_distances, all_evidence_distances, all_max_distances, all_round_depths = [], [], [], []
    n_resolved, n_skipped_long = 0, 0

    for idx, (adj, colors, hints, trace) in enumerate(instances):
        if not trace["resolved"]:
            continue
        n_resolved += 1
        prov_trace = solve_with_dd_provenance(adj, colors, hints, config["n_colors"])
        assert prov_trace["rounds"] == trace["rounds"], "provenance reimplementation diverged from reference trace"

        tokens = encode_instance(adj, trace, vocab)
        if len(tokens) > config["max_len"]:
            n_skipped_long += 1
            continue

        positions = token_positions(adj, trace, vocab)
        recs = required_distances(adj, trace, prov_trace["provenance"], positions)
        deduced_at = prov_trace["deduced_at"]
        for r in recs:
            all_graph_distances.append(r["graph_distance"])
            all_evidence_distances.extend(r["evidence_distances"])
            all_max_distances.append(r["max_distance"])
            all_round_depths.append(deduced_at[r["node"]][1])

        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        _, attn_maps = forward_capture_attention(model, x)
        query_positions = [r["query_pos"] for r in recs]
        s, c = attention_weight_by_distance(attn_maps, query_positions, max_distance)
        sum_by_layer += s
        count_by_distance += c

        if (idx + 1) % 50 == 0:
            print(f"processed {idx + 1}/{len(instances)}", flush=True)

    mean_by_layer = np.divide(sum_by_layer, count_by_distance, out=np.full_like(sum_by_layer, np.nan),
                               where=count_by_distance > 0)

    # Second pass: is attention *targeted* at the specific tokens DD needs,
    # beyond what the generic distance-decay curve above would predict? For
    # each deduction, compare the attention weight actually placed on its
    # graph-row position and each evidence position against
    # mean_by_layer[layer, that same distance] (the baseline for a token at
    # that distance with no regard to content). enrichment >> 1 means the
    # model is doing real content-based lookup; enrichment ~= 1 means it is
    # attending by distance/recency alone, not by what's actually relevant.
    enrichment_sum = np.zeros(num_layers)
    enrichment_count = np.zeros(num_layers)
    for adj, colors, hints, trace in instances:
        if not trace["resolved"]:
            continue
        prov_trace = solve_with_dd_provenance(adj, colors, hints, config["n_colors"])
        tokens = encode_instance(adj, trace, vocab)
        if len(tokens) > config["max_len"]:
            continue
        positions = token_positions(adj, trace, vocab)
        recs = required_distances(adj, trace, prov_trace["provenance"], positions)
        if not recs:
            continue
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        _, attn_maps = forward_capture_attention(model, x)
        for r in recs:
            q = r["query_pos"]
            target_positions = [q - r["graph_distance"]] + [q - d for d in r["evidence_distances"]]
            target_distances = [r["graph_distance"]] + list(r["evidence_distances"])
            for layer_idx in range(num_layers):
                head_mean = attn_maps[layer_idx][:, q, :].mean(dim=0).detach().cpu().numpy()
                for tp, td in zip(target_positions, target_distances):
                    baseline = mean_by_layer[layer_idx, td]
                    if baseline > 0 and not np.isnan(baseline):
                        enrichment_sum[layer_idx] += head_mean[tp] / baseline
                        enrichment_count[layer_idx] += 1
    enrichment_by_layer = np.divide(enrichment_sum, enrichment_count,
                                     out=np.full_like(enrichment_sum, np.nan), where=enrichment_count > 0)

    np.savez(
        os.path.join(args.out_dir, "attention_distance_results.npz"),
        mean_attn_by_layer_distance=mean_by_layer, count_by_distance=count_by_distance,
        graph_distances=np.array(all_graph_distances), evidence_distances=np.array(all_evidence_distances),
        max_distances=np.array(all_max_distances), round_depths=np.array(all_round_depths),
        enrichment_by_layer=enrichment_by_layer,
    )
    summary = {
        "checkpoint_dir": args.checkpoint_dir, "n_instances_requested": args.n_instances,
        "n_resolved": n_resolved, "n_skipped_too_long": n_skipped_long,
        "n_deductions": len(all_max_distances),
        "max_required_distance": int(max(all_max_distances)) if all_max_distances else None,
        "median_required_distance": float(np.median(all_max_distances)) if all_max_distances else None,
        "mean_required_distance": float(np.mean(all_max_distances)) if all_max_distances else None,
        "max_len": config["max_len"], "num_layers": num_layers,
        "enrichment_by_layer": [None if np.isnan(v) else float(v) for v in enrichment_by_layer],
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
