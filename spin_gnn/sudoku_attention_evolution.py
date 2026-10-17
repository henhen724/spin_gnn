"""Tracks how the transformer's attention targeting (sudoku_attention_distance's
enrichment metric) evolves over training, for the best sudoku-dd hparam-sweep
config (d_model=64, nhead=4, num_layers=12, dim_feedforward=1024).

sudoku_attention_distance_analysis.py showed that at the *final* checkpoint,
attention reaches the distances DD needs but isn't targeted (enrichment only
~1-1.7x a distance-matched baseline). This script retrains that config with
checkpoint_epochs snapshots (train_sudoku_model already supports this) and
reruns the enrichment computation at each snapshot, to see whether targeting
is still improving (undertrained) or has plateaued early / never moved much
off ~1.0 (the attention head isn't learning this at all).

Usage:
    python -m spin_gnn.sudoku_attention_evolution --out-dir <dir> [--epochs 300] ...
"""
import argparse
import json
import os

import numpy as np
import torch

from spin_gnn.config import SudokuReasoningConfig, SudokuReasoningModelConfig, SudokuReasoningTrainConfig
from spin_gnn.sat_transformer2 import SATReasoningTransformer
from spin_gnn.sudoku_attention_distance import (
    attention_weight_by_distance, forward_capture_attention, required_distances,
    solve_with_dd_provenance, token_positions,
)
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import train_sudoku_model


def enrichment_at_checkpoint(model, vocab, instances, config_dict, n_colors, max_distance):
    """One pass to get the distance-decay baseline, one pass to get
    enrichment at the specific needed tokens -- same logic as
    sudoku_attention_distance_analysis.main, factored out so it can be
    called once per checkpoint without retracing instances each time."""
    num_layers = config_dict["num_layers"]
    sum_by_layer = np.zeros((num_layers, max_distance + 1))
    count_by_distance = np.zeros(max_distance + 1)
    cache = []  # (tokens, recs) per instance, reused for pass 2

    for adj, colors, hints, trace in instances:
        if not trace["resolved"]:
            continue
        prov_trace = solve_with_dd_provenance(adj, colors, hints, n_colors)
        tokens = encode_instance(adj, trace, vocab)
        if len(tokens) > config_dict["max_len"]:
            continue
        positions = token_positions(adj, trace, vocab)
        recs = required_distances(adj, trace, prov_trace["provenance"], positions)
        if not recs:
            continue
        cache.append((tokens, recs))

        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        _, attn_maps = forward_capture_attention(model, x)
        query_positions = [r["query_pos"] for r in recs]
        s, c = attention_weight_by_distance(attn_maps, query_positions, max_distance)
        sum_by_layer += s
        count_by_distance += c

    mean_by_layer = np.divide(sum_by_layer, count_by_distance, out=np.full_like(sum_by_layer, np.nan),
                               where=count_by_distance > 0)

    enrichment_sum = np.zeros(num_layers)
    enrichment_count = np.zeros(num_layers)
    for tokens, recs in cache:
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
    return enrichment_by_layer, mean_by_layer, count_by_distance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--dim-feedforward", type=int, default=1024)
    p.add_argument("--n-train-instances", type=int, default=5000)
    p.add_argument("--n-val-instances", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--checkpoint-epochs", type=int, nargs="+",
                   default=[0, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300])
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sudoku_cfg = SudokuReasoningConfig()
    model_cfg = SudokuReasoningModelConfig(
        d_model=args.d_model, num_layers=args.num_layers, nhead=args.nhead, dim_feedforward=args.dim_feedforward,
    )
    train_cfg = SudokuReasoningTrainConfig(
        n_train_instances=args.n_train_instances, n_val_instances=args.n_val_instances,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
    )
    checkpoint_epochs = [e for e in args.checkpoint_epochs if e <= args.epochs]

    result = train_sudoku_model(sudoku_cfg, model_cfg, train_cfg, checkpoint_epochs=checkpoint_epochs)
    vocab = result["vocab"]
    val_instances = result["val_instances"]
    model_max_len = result["model_max_len"]
    history_by_epoch = {h["epoch"] + 1: h for h in result["history"]}
    history_by_epoch[0] = {"epoch": -1, "accuracy": None, "nll": None}

    config_dict = {
        "num_layers": args.num_layers, "max_len": model_max_len,
    }
    max_distance = model_max_len - 1

    evolution = []
    for epoch in sorted(result["checkpoints"].keys()):
        state_dict = result["checkpoints"][epoch]
        model = SATReasoningTransformer(
            vocab.size, d_model=args.d_model, num_layers=args.num_layers, nhead=args.nhead,
            max_len=model_max_len, dim_feedforward=args.dim_feedforward, norm_first=model_cfg.norm_first,
        )
        model.load_state_dict(state_dict)
        model.eval()

        enrichment_by_layer, _, _ = enrichment_at_checkpoint(
            model, vocab, val_instances, config_dict, sudoku_cfg.n_colors, max_distance,
        )
        val_acc = history_by_epoch.get(epoch, {}).get("accuracy")
        record = {
            "epoch": epoch, "val_accuracy": val_acc,
            "enrichment_by_layer": [None if np.isnan(v) else float(v) for v in enrichment_by_layer],
            "mean_enrichment": float(np.nanmean(enrichment_by_layer)),
        }
        evolution.append(record)
        print(f"epoch {epoch}: val_accuracy={val_acc} mean_enrichment={record['mean_enrichment']:.4f}", flush=True)

    with open(os.path.join(args.out_dir, "evolution.json"), "w") as f:
        json.dump({
            "config": {"d_model": args.d_model, "nhead": args.nhead, "num_layers": args.num_layers,
                       "dim_feedforward": args.dim_feedforward, "epochs": args.epochs,
                       "n_train_instances": args.n_train_instances},
            "final_solve_metrics": result["solve_metrics"],
            "evolution": evolution,
        }, f, indent=2)
    print(f"wrote {os.path.join(args.out_dir, 'evolution.json')}")


if __name__ == "__main__":
    main()
