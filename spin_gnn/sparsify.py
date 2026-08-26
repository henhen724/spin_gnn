"""Random-sparsification sweep: train an independent SpinGNN at each edge
keep-probability against a *fixed* set of disorder realizations and
trajectories (only cfg.sparsification.keep_prob varies), then compare
val NLL-gap-to-Bayes-optimal and accuracy across keep_prob. This isolates the
effect of throwing away random edges of the fully-connected coupling graph
from the effect of disorder/trajectory sampling noise.

Run via scripts/slurm/sparsify_sweep.sbatch (GPU job, one array task per
keep_prob is also a reasonable way to parallelize this across nodes).
"""
import copy
import csv
import os

from .config import ExperimentConfig
from .train import save_checkpoint, train_model


def run_sparsification_sweep(base_cfg: ExperimentConfig, keep_probs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for p in keep_probs:
        cfg = copy.deepcopy(base_cfg)
        cfg.sparsification.keep_prob = p
        model, history = train_model(cfg)
        final = history[-1]
        row = {"keep_prob": p, **{k: v for k, v in final.items() if k != "epoch"}}
        rows.append(row)
        print(f"keep_prob={p}: {row}")
        save_checkpoint(model, history, cfg, os.path.join(out_dir, f"keep_prob_{p}"))

    csv_path = os.path.join(out_dir, "sparsification_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")
    return rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-spins", type=int, default=200)
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--n-realizations", type=int, default=8)
    ap.add_argument("--trajectory-length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument(
        "--keep-probs",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01],
    )
    ap.add_argument("--out-dir", required=True, help="e.g. $SCRATCH/spin_gnn/sparsify_sweep")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    cfg.disorder.n_spins = args.n_spins
    cfg.disorder.context_length = args.context_length
    cfg.disorder.beta = args.beta
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.num_layers = args.num_layers
    cfg.train.n_realizations = args.n_realizations
    cfg.train.trajectory_length = args.trajectory_length
    cfg.train.epochs = args.epochs

    run_sparsification_sweep(cfg, args.keep_probs, args.out_dir)
