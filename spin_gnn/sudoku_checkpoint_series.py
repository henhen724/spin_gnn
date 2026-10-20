"""Train once and save a checkpoint series to disk, for "when during
training does DD appear" analyses.

train_sudoku_model already snapshots state dicts in memory via
checkpoint_epochs, but every existing entry point either discards them
(sudoku_hparam_sweep saves only the final model) or uses them in-process
(sudoku_transition_sweep). The probe and SAE work needs them on disk, each
in the directory layout load_checkpoint expects, so the same analysis can be
re-run over the series without retraining.

Also records solve_rate at each snapshot in a CSV, so the emergence of DD
structure can be lined up against the capability curve: the question is
whether DD state becomes decodable before, with, or after solve_rate moves.

Usage: python -m spin_gnn.sudoku_checkpoint_series --out-dir $SCRATCH/...
"""
import argparse
import csv
import os

import torch

from .config import SudokuReasoningConfig, SudokuReasoningModelConfig, SudokuReasoningTrainConfig
from .sat_transformer2 import SATReasoningTransformer
from .train_sudoku_transformer import build_instances, save_checkpoint, solve_rate, train_sudoku_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=24)
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--n-colors", type=int, default=4)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--rho-max", type=float, default=0.95)
    ap.add_argument("--hint-mode", type=str, default="removal", choices=["removal", "random"])
    ap.add_argument("--scratchpad", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=12)
    ap.add_argument("--dim-feedforward", type=int, default=1024)
    ap.add_argument("--no-norm-first", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-train-instances", type=int, default=5000)
    ap.add_argument("--n-val-instances", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=900)
    # Log-ish spacing: DD structure, if it appears at all, is most likely to
    # appear early, and the late epochs are where overfitting showed up.
    ap.add_argument("--checkpoint-epochs", type=str,
                    default="1,2,5,10,20,35,50,75,100,150,200,300,450,600,750,900")
    ap.add_argument("--n-solve-instances", type=int, default=150)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    checkpoint_epochs = sorted({int(e) for e in args.checkpoint_epochs.split(",") if int(e) <= args.epochs})

    sudoku_cfg = SudokuReasoningConfig(
        n_nodes=args.n_nodes, degree=args.degree, n_colors=args.n_colors, rho=args.rho, rho_max=args.rho_max,
        seed=args.seed, hint_mode=args.hint_mode, scratchpad=args.scratchpad,
    )
    model_cfg = SudokuReasoningModelConfig(
        d_model=args.d_model, num_layers=args.num_layers, nhead=args.nhead,
        dim_feedforward=args.dim_feedforward, norm_first=not args.no_norm_first,
    )
    train_cfg = SudokuReasoningTrainConfig(
        n_train_instances=args.n_train_instances, n_val_instances=args.n_val_instances,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
    )

    result = train_sudoku_model(sudoku_cfg, model_cfg, train_cfg, checkpoint_epochs=checkpoint_epochs)
    vocab = result["vocab"]
    device = next(iter(result["model"].parameters())).device
    os.makedirs(args.out_dir, exist_ok=True)

    # One fixed instance pool for every snapshot, so solve_rate differences
    # across epochs are the model changing and nothing else.
    solve_instances = build_instances(sudoku_cfg, args.n_solve_instances, seed_start=700_000_000)

    rows = []
    for epoch in sorted(result["checkpoints"]):
        model = SATReasoningTransformer(
            vocab.size, d_model=model_cfg.d_model, num_layers=model_cfg.num_layers, nhead=model_cfg.nhead,
            max_len=result["model_max_len"], dim_feedforward=model_cfg.dim_feedforward or None,
            norm_first=model_cfg.norm_first,
        ).to(device)
        model.load_state_dict(result["checkpoints"][epoch])
        model.eval()

        metrics = solve_rate(model, solve_instances, vocab, device, result["model_max_len"])
        ckpt_dir = os.path.join(args.out_dir, f"epoch{epoch:05d}")
        save_checkpoint(
            ckpt_dir, sudoku_cfg, model_cfg, model, vocab, result["model_max_len"],
            result["baseline_accuracy"], metrics, result["history"][:epoch],
        )
        hist = result["history"][epoch - 1] if epoch - 1 < len(result["history"]) else {}
        rows.append({
            "epoch": epoch, "checkpoint_dir": ckpt_dir,
            "val_nll": hist.get("nll"), "val_accuracy": hist.get("accuracy"),
            **{k: metrics.get(k) for k in ("solve_rate", "outcome_accuracy", "well_formed_rate")},
        })
        print(f"epoch {epoch:5d}: solve_rate={metrics['solve_rate']:.3f} "
              f"well_formed={metrics['well_formed_rate']:.3f} -> {ckpt_dir}")

    csv_path = os.path.join(args.out_dir, "series.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} snapshots)")


if __name__ == "__main__":
    main()
