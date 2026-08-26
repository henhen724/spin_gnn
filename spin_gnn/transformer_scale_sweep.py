"""One data point of the transformer-baseline scaling sweep: mirrors
spin_gnn/scale_sweep.py exactly (same grid semantics, same held-out test-pool
construction, same output CSV convention) but trains SpinTransformer instead
of SpinGNN, with d_model chosen so the transformer's total parameter count
matches SpinGNN's at this context_length (parameter counts differ across the
sweep's context_length axis for both models, so this is re-solved per point,
see transformer_baseline.find_matching_d_model).

Run via scripts/slurm/transformer_scale_sweep.sbatch (one array task per grid
point), not directly on the login node.
"""
import copy
import csv
import json
import os

from .config import ExperimentConfig
from .train import build_windows
from .transformer_baseline import (
    count_params,
    find_matching_d_model,
    gnn_param_count,
    train_transformer,
)


def run_one(cfg, n_train_sequences, n_test_realizations, gnn_hidden_dim, gnn_num_layers, num_layers, out_dir):
    k = cfg.disorder.context_length
    target_params = gnn_param_count(k, hidden_dim=gnn_hidden_dim, num_layers=gnn_num_layers)
    d_model = find_matching_d_model(target_params, k, cfg.disorder.n_spins, num_layers=num_layers)

    train_cfg = copy.deepcopy(cfg)
    train_cfg.disorder.seed = 10_000 * k
    train_cfg.train.n_realizations = n_train_sequences

    test_cfg = copy.deepcopy(cfg)
    test_cfg.disorder.seed = 10_000 * k + 5_000
    test_cfg.train.n_realizations = n_test_realizations
    test_windows = build_windows(test_cfg)

    model, history = train_transformer(train_cfg, d_model, num_layers=num_layers, test_windows=test_windows)

    best = min(history, key=lambda r: r["nll"])
    row = {
        "model": "transformer",
        "context_length": k,
        "n_train_sequences": n_train_sequences,
        "n_test_realizations": n_test_realizations,
        "gamma_kind": cfg.disorder.gamma_kind,
        "gamma_param": cfg.disorder.gamma_param,
        "n_spins": cfg.disorder.n_spins,
        "beta": cfg.disorder.beta,
        "d_model": d_model,
        "num_layers": num_layers,
        "target_gnn_params": target_params,
        "transformer_params": count_params(model),
        "trajectory_length": cfg.train.trajectory_length,
        "epochs": cfg.train.epochs,
        "best_epoch": best["epoch"],
        "min_test_nll": best["nll"],
        "test_accuracy_at_min": best["accuracy"],
        "bayes_test_nll_at_min": best["bayes_nll"],
        "nll_gap_at_min": best["nll_gap"],
        "bayes_accuracy_at_min": best["bayes_accuracy"],
    }

    os.makedirs(out_dir, exist_ok=True)
    tag = f"K{k:03d}_N{n_train_sequences:06d}"
    with open(os.path.join(out_dir, f"scale_{tag}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    with open(os.path.join(out_dir, f"history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(row)
    return row


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-spins", type=int, default=32)
    ap.add_argument("--context-length", type=int, required=True)
    ap.add_argument("--n-train-sequences", type=int, required=True)
    ap.add_argument("--n-test-realizations", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma-kind", choices=["identity", "ones", "alternating", "power_law"], default="power_law")
    ap.add_argument("--gamma-param", type=float, default=1.0)
    ap.add_argument("--gnn-hidden-dim", type=int, default=32, help="the SpinGNN config to match parameter count against")
    ap.add_argument("--gnn-num-layers", type=int, default=2)
    ap.add_argument("--num-layers", type=int, default=2, help="transformer depth")
    ap.add_argument("--trajectory-length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", required=True, help="e.g. $SCRATCH/spin_gnn/scale_sweep/transformer_power1")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    cfg.disorder.n_spins = args.n_spins
    cfg.disorder.context_length = args.context_length
    cfg.disorder.beta = args.beta
    cfg.disorder.gamma_kind = args.gamma_kind
    cfg.disorder.gamma_param = args.gamma_param
    cfg.train.trajectory_length = args.trajectory_length
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr

    run_one(
        cfg,
        args.n_train_sequences,
        args.n_test_realizations,
        args.gnn_hidden_dim,
        args.gnn_num_layers,
        args.num_layers,
        args.out_dir,
    )
