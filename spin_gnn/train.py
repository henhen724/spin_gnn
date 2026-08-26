"""Self-supervised next-spin-prediction training for SpinGNN.

The task is literally next-token prediction over a 2-symbol vocabulary
({down, up}) per node: given the K-lag spin history and the (possibly
sparsified) coupling graph, minimize cross-entropy against the realized next
spin at every node. Because the exact Bayes-optimal posterior is known in
closed form (spin_gnn.simulate.local_field), we also report the NLL *gap* to
that optimum -- the natural metric for "how much does random sparsification
of the coupling graph hurt", which is what scripts/slurm/sparsify_sweep.sbatch
sweeps over.

Run via Slurm (scripts/slurm/train.sbatch), not on the login node -- building
the model needs a CUDA-enabled py-pytorch + torch_geometric env.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .dataset import SpinTrajectoryWindows, iterate_batches
from .graph import build_spin_graph
from .model import SpinGNN
from .simulate import generate_realization


def build_windows(cfg: ExperimentConfig):
    """Generate `n_realizations` independent quenched disorder + trajectories
    and wrap each in a SpinTrajectoryWindows sharing that realization's
    (sparsified) graph."""
    windows = []
    for i in range(cfg.train.n_realizations):
        seed = cfg.disorder.seed + i
        realization = generate_realization(cfg.disorder, cfg.train.trajectory_length, seed)
        spin_graph = build_spin_graph(
            realization["J"],
            cfg.disorder.context_length,
            keep_prob=cfg.sparsification.keep_prob,
            compensate=cfg.sparsification.compensate,
            seed=cfg.sparsification.seed + i,
        )
        windows.append(SpinTrajectoryWindows(realization, spin_graph))
    return windows


def bayes_optimal_nll(p_true, y):
    p = p_true.clamp(1e-6, 1 - 1e-6)
    p_of_y = torch.where(y == 1, p, 1 - p)
    return -torch.log(p_of_y).mean().item()


def evaluate(model, dataset, indices, device, batch_size=8):
    model.eval()
    total_nll, total_bayes_nll, total_correct, total_bayes_correct, total_n = 0.0, 0.0, 0, 0, 0
    with torch.no_grad():
        for x, edge_index, edge_attr, y, p_true in iterate_batches(
            dataset, indices, batch_size, shuffle=False
        ):
            x, edge_index, edge_attr, y, p_true = (
                t.to(device) for t in (x, edge_index, edge_attr, y, p_true)
            )
            logits = model(x, edge_index, edge_attr)
            nll = F.cross_entropy(logits, y, reduction="sum").item()
            pred = logits.argmax(dim=-1)
            bayes_pred = (p_true > 0.5).long()
            total_nll += nll
            total_bayes_nll += bayes_optimal_nll(p_true, y) * y.numel()
            total_correct += (pred == y).sum().item()
            total_bayes_correct += (bayes_pred == y).sum().item()
            total_n += y.numel()
    return {
        "nll": total_nll / total_n,
        "bayes_nll": total_bayes_nll / total_n,
        "nll_gap": total_nll / total_n - total_bayes_nll / total_n,
        "accuracy": total_correct / total_n,
        # thresholding the *true* analytic posterior at 0.5 -- the accuracy
        # ceiling a perfect model could reach, capped below 100% by the
        # process's own stochasticity (spins are sampled, not deterministic,
        # given their context).
        "bayes_accuracy": total_bayes_correct / total_n,
    }


def train_model(cfg: ExperimentConfig, test_windows=None):
    """If `test_windows` is given (a list of SpinTrajectoryWindows built from
    disorder realizations disjoint from cfg's training realizations), every
    window of every trajectory in it is used for evaluation each epoch, and
    *all* windows of cfg's own realizations are used for training -- i.e.
    evaluation measures generalization to unseen disorder, not just to a
    later time-slice of the same trajectories. If omitted, falls back to the
    within-trajectory time-split via cfg.train.val_fraction (cheaper, weaker
    generalization claim)."""
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    windows = build_windows(cfg)

    if test_windows is None:
        splits = [w.time_split(cfg.train.val_fraction) for w in windows]
        train_indices = [s[0] for s in splits]
        eval_windows = windows
        eval_indices = [s[1] for s in splits]
    else:
        train_indices = [list(range(len(w))) for w in windows]
        eval_windows = test_windows
        eval_indices = [list(range(len(w))) for w in test_windows]

    model = SpinGNN(
        cfg.disorder.context_length,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    rng = np.random.default_rng(cfg.disorder.seed)

    history = []
    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        realization_order = rng.permutation(len(windows))
        for w_idx in realization_order:
            dataset = windows[w_idx]
            for x, edge_index, edge_attr, y, _ in iterate_batches(
                dataset, train_indices[w_idx], cfg.train.batch_size, rng=rng
            ):
                x, edge_index, edge_attr, y = (
                    t.to(device) for t in (x, edge_index, edge_attr, y)
                )
                opt.zero_grad()
                logits = model(x, edge_index, edge_attr)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * y.numel()
                epoch_n += y.numel()

        val_metrics = [
            evaluate(model, w, eval_indices[i], device) for i, w in enumerate(eval_windows)
        ]
        avg_val = {
            key: float(np.mean([m[key] for m in val_metrics])) for key in val_metrics[0]
        }
        record = {"epoch": epoch, "train_loss": epoch_loss / epoch_n, **avg_val}
        history.append(record)
        print(record)

    return model, history


def save_checkpoint(model, history, cfg: ExperimentConfig, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-spins", type=int, default=200)
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--keep-prob", type=float, default=1.0)
    ap.add_argument("--n-realizations", type=int, default=8)
    ap.add_argument("--trajectory-length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", required=True, help="e.g. $SCRATCH/spin_gnn/checkpoints/run0")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    cfg.disorder.n_spins = args.n_spins
    cfg.disorder.context_length = args.context_length
    cfg.disorder.beta = args.beta
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.num_layers = args.num_layers
    cfg.sparsification.keep_prob = args.keep_prob
    cfg.train.n_realizations = args.n_realizations
    cfg.train.trajectory_length = args.trajectory_length
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr

    trained_model, run_history = train_model(cfg)
    save_checkpoint(trained_model, run_history, cfg, args.out_dir)
