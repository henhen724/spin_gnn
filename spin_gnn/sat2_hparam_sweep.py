"""Hyperparameter + learning-rate sweep for SATReasoningTransformer
(spin_gnn.ksat2's event-based encoding) at a fixed problem size (n_vars,
alpha -> m_clauses). Trains one (d_model, nhead, num_layers, lr) config,
appends a CSV row with the actual target metric -- solve_rate (does free
generation from just hints+statement reach the correct, verified answer),
not just token-level accuracy -- and saves a checkpoint.

Usage: python -m spin_gnn.sat2_hparam_sweep --n-vars 16 --alpha 1.25 \\
    --d-model 64 --nhead 4 --num-layers 2 --lr 1e-3 --out-dir $SCRATCH/...
"""
import argparse
import csv
import os

from .config import SATReasoningConfig, SATReasoningModelConfig, SATReasoningTrainConfig
from .train_sat_transformer2 import save_checkpoint, train_sat2_model


def run_one(args):
    sat_cfg = SATReasoningConfig(n_vars=args.n_vars, k=args.k, alpha=args.alpha, n_hints=args.n_hints, seed=args.seed,
                                  use_clause_status=args.use_clause_status)
    model_cfg = SATReasoningModelConfig(
        d_model=args.d_model, num_layers=args.num_layers, nhead=args.nhead, dim_feedforward=args.dim_feedforward,
        norm_first=args.norm_first,
    )
    train_cfg = SATReasoningTrainConfig(
        n_train_instances=args.n_train_instances, n_val_instances=args.n_val_instances,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
    )
    result = train_sat2_model(sat_cfg, model_cfg, train_cfg)

    m_clauses = result["vocab"].m_clauses
    resolved_ff = result["model"].encoder.layers[0].linear1.out_features
    norm_tag = "pre" if args.norm_first else "post"
    status_tag = "cs1" if args.use_clause_status else "cs0"
    tag = (
        f"m{m_clauses:03d}_d{args.d_model:04d}_h{args.nhead:02d}_L{args.num_layers:02d}"
        f"_ff{resolved_ff:05d}_lr{args.lr:.0e}_{norm_tag}_{status_tag}"
    )
    out_dir = os.path.join(args.out_dir, tag)
    save_checkpoint(
        out_dir, sat_cfg, model_cfg, result["model"], result["vocab"], result["model_max_len"],
        result["baseline_accuracy"], result["solve_metrics"], result["history"],
    )

    best_val_accuracy = max(r["accuracy"] for r in result["history"])
    row = {
        "n_vars": args.n_vars, "k": args.k, "alpha": args.alpha, "m_clauses": m_clauses, "n_hints": args.n_hints,
        "d_model": args.d_model, "nhead": args.nhead, "num_layers": args.num_layers, "dim_feedforward": resolved_ff,
        "norm_first": args.norm_first, "use_clause_status": args.use_clause_status, "lr": args.lr,
        "n_train_instances": args.n_train_instances, "n_val_instances": args.n_val_instances,
        "epochs": args.epochs, "baseline_accuracy": result["baseline_accuracy"], "best_val_accuracy": best_val_accuracy,
        "final_val_accuracy": result["history"][-1]["accuracy"], "final_val_nll": result["history"][-1]["nll"],
        "solve_rate": result["solve_metrics"]["solve_rate"],
        "outcome_accuracy": result["solve_metrics"]["outcome_accuracy"],
        "well_formed_rate": result["solve_metrics"]["well_formed_rate"],
        "checkpoint_dir": out_dir,
    }

    # norm_tag in the filename (not just the "norm_first" column) because
    # older sweep_*.csv files predate that column entirely -- appending a
    # differently-shaped row to them would desync the file's header from
    # its data rows for any later csv.DictReader.
    csv_path = os.path.join(
        args.out_dir, f"sweep_m{m_clauses:03d}_h{args.nhead:02d}_L{args.num_layers:02d}_{norm_tag}_{status_tag}.csv"
    )
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"wrote {csv_path}: solve_rate={row['solve_rate']:.4f} outcome_accuracy={row['outcome_accuracy']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vars", type=int, default=16)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.25)
    ap.add_argument("--n-hints", type=int, default=0)
    ap.add_argument("--use-clause-status", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dim-feedforward", type=int, default=0)
    ap.add_argument("--norm-first", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-train-instances", type=int, default=3000)
    ap.add_argument("--n-val-instances", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out-dir", type=str, required=True)
    parsed = ap.parse_args()
    if parsed.d_model % parsed.nhead != 0:
        raise ValueError(f"d_model={parsed.d_model} must be divisible by nhead={parsed.nhead}")
    run_one(parsed)
