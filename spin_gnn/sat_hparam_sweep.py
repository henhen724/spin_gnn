"""One data point of the SATTransformer hyperparameter sweep: mirrors
spin_gnn/transformer_scale_sweep.py's role (one array-job task per grid
point, one CSV row + a saved checkpoint per point), but sweeps the model's
own capacity/shape at a fixed, much larger instance count instead of
matching parameter counts across a context-length axis -- the question here
is "does scaling this architecture find a config that reaches high
per-dimension next-token accuracy (and, ideally, a nonzero
val_satisfying_completion_rate) on the k-SAT trace", not a fair-comparison
question against another architecture.

Originally swept (d_model, nhead) (scripts/slurm/sat_hparam_sweep.sbatch),
then nhead alone at fixed d_model=256 (sat_nhead_sweep.sbatch); this module
now also supports sweeping num_layers/dim_feedforward at fixed d_model/nhead
(sat_mlp_sweep.sbatch) -- the tag/checkpoint dir and CSV row always include
all four dimensions, so runs across any of these sweeps never collide.

Run via scripts/slurm/sat_hparam_sweep.sbatch (one array task per grid
point), not directly on the login node -- training needs a CUDA-enabled
py-pytorch env.
"""
import csv
import os

from .config import SATConfig, SATModelConfig, SATTrainConfig
from .train_sat_transformer import save_checkpoint, train_sat_model


def run_one(sat_cfg: SATConfig, train_cfg: SATTrainConfig, d_model, nhead, num_layers, out_dir, dim_feedforward=0):
    model_cfg = SATModelConfig(d_model=d_model, num_layers=num_layers, nhead=nhead, dim_feedforward=dim_feedforward)
    model, history, baseline_acc = train_sat_model(sat_cfg, model_cfg, train_cfg)
    resolved_ff = model.encoder.layers[0].linear1.out_features

    # Tag includes every varied dimension, not just d_model/nhead -- the
    # num_layers/dim_feedforward follow-up sweep holds d_model/nhead fixed,
    # so those two alone would collide across every point in that grid.
    tag = f"d{d_model:04d}_h{nhead:02d}_L{num_layers:02d}_ff{resolved_ff:05d}"
    ckpt_dir = os.path.join(out_dir, tag)
    save_checkpoint(model, history, ckpt_dir, sat_cfg, model_cfg, baseline_acc=baseline_acc)

    best = max(history, key=lambda r: r["accuracy"])
    row = {
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": resolved_ff,
        "n_vars": sat_cfg.n_vars,
        "k": sat_cfg.k,
        "alpha": sat_cfg.alpha,
        "n_train_instances": train_cfg.n_train_instances,
        "n_val_instances": train_cfg.n_val_instances,
        "max_trace_tokens": train_cfg.max_trace_tokens,
        "epochs": train_cfg.epochs,
        "baseline_accuracy": baseline_acc,
        "best_epoch": best["epoch"],
        "best_accuracy": best["accuracy"],
        "final_accuracy": history[-1]["accuracy"],
        "final_nll": history[-1]["nll"],
        "final_train_loss": history[-1]["train_loss"],
        "final_val_satisfying_completion_rate": history[-1]["val_satisfying_completion_rate"],
        "checkpoint_dir": ckpt_dir,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"sweep_{tag}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(row)
    return row


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-vars", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=4.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, required=True)
    ap.add_argument("--nhead", type=int, required=True)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dim-feedforward", type=int, default=0, help="0 = model's default (4 * d_model)")
    ap.add_argument("--n-train-instances", type=int, default=10_000)
    ap.add_argument("--n-val-instances", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-trace-tokens", type=int, default=2000, help="0 = no cap")
    ap.add_argument("--out-dir", required=True, help="e.g. $SCRATCH/spin_gnn/sat_hparam_sweep")
    args = ap.parse_args()

    if args.d_model % args.nhead != 0:
        raise ValueError(f"d_model={args.d_model} must be divisible by nhead={args.nhead}")

    sat_cfg = SATConfig(n_vars=args.n_vars, k=args.k, alpha=args.alpha, seed=args.seed)
    train_cfg = SATTrainConfig(
        n_train_instances=args.n_train_instances, n_val_instances=args.n_val_instances,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
        max_trace_tokens=args.max_trace_tokens,
    )

    run_one(
        sat_cfg, train_cfg, args.d_model, args.nhead, args.num_layers, args.out_dir,
        dim_feedforward=args.dim_feedforward,
    )
