"""One data point of a data/context-length scaling-law sweep.

A single invocation trains one SpinGNN at one (context_length, n_train_sequences)
grid point and reports the minimum test loss (NLL) reached over training,
evaluated against a *held-out* pool of disorder realizations -- never seen
during training -- so the result reflects generalization across the disorder
ensemble, not just to a later time-slice of the training trajectories.

The held-out test pool is deterministic in `context_length` alone (fixed
--n-test-realizations, seeded from context_length only), so every
n_train_sequences point at a given context_length is scored against the
*same* test pool -- required for the comparison across n_train_sequences to
be measuring only the effect of training-set size.

Training realizations are seeded contiguously (seed_base, seed_base+1, ...),
so the training set for a smaller n_train_sequences is a literal prefix of
the training set for any larger n_train_sequences at the same context_length
-- reduces variance when comparing across the sweep's data-size axis.

Run via scripts/slurm/scale_sweep.sbatch (one array task per grid point), not
directly on the login node.
"""
import copy
import csv
import json
import os

from .config import ExperimentConfig
from .train import build_windows, train_model


def run_one(cfg: ExperimentConfig, n_train_sequences, n_test_realizations, out_dir):
    k = cfg.disorder.context_length
    train_cfg = copy.deepcopy(cfg)
    train_cfg.disorder.seed = 10_000 * k
    train_cfg.train.n_realizations = n_train_sequences

    test_cfg = copy.deepcopy(cfg)
    test_cfg.disorder.seed = 10_000 * k + 5_000
    test_cfg.train.n_realizations = n_test_realizations
    test_windows = build_windows(test_cfg)

    model, history = train_model(train_cfg, test_windows=test_windows)

    best = min(history, key=lambda r: r["nll"])
    row = {
        "context_length": k,
        "n_train_sequences": n_train_sequences,
        "n_test_realizations": n_test_realizations,
        "gamma_kind": cfg.disorder.gamma_kind,
        "gamma_param": cfg.disorder.gamma_param,
        "n_spins": cfg.disorder.n_spins,
        "beta": cfg.disorder.beta,
        "hidden_dim": cfg.model.hidden_dim,
        "num_layers": cfg.model.num_layers,
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
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--trajectory-length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", required=True, help="e.g. $SCRATCH/spin_gnn/scale_sweep")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    cfg.disorder.n_spins = args.n_spins
    cfg.disorder.context_length = args.context_length
    cfg.disorder.beta = args.beta
    cfg.disorder.gamma_kind = args.gamma_kind
    cfg.disorder.gamma_param = args.gamma_param
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.num_layers = args.num_layers
    cfg.train.trajectory_length = args.trajectory_length
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr

    run_one(cfg, args.n_train_sequences, args.n_test_realizations, args.out_dir)
