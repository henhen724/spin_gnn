"""Next-token-prediction training for SATTransformer over spin_gnn.ksat
solver traces -- does attention learn to reproduce unit-propagation /
most-constrained-variable-branching dynamics from the raw trace alone?

Generates n_train_instances/n_val_instances disjoint random k-SAT instances
(split by instance -- independent problems, so no time-split is needed the
way spin_gnn.train's held-out realizations need one), solves each with
spin_gnn.ksat.solve_with_trace, and trains next-token prediction over the
resulting SATTraceDataset (spin_gnn.sat_dataset). Alongside the usual
train/val cross-entropy and per-dimension accuracy, this also reports:

- `baseline_accuracy`: accuracy of always predicting each token dimension's
  majority class (see baseline_accuracy() below). Many clause-token
  dimensions are trivially 0 by construction, so raw accuracy is only
  meaningful relative to this, not in isolation.
- `val_satisfying_completion_rate`: teacher-force the model on each SAT val
  instance's trace up through the second-to-last token, and check whether
  its single greedy (argmax) next-token prediction is itself an assignment
  that satisfies every original clause. That's the "did it learn to reach a
  correct final answer" probe -- token-level NTP accuracy only measures
  matching one particular valid trace, which is a much easier bar.

`SATTrainConfig.max_trace_tokens` (0 = no cap) drops instances whose trace
exceeds that length before batching -- necessary at large instance counts,
since trace length is heavy-tailed and self-attention is O(L^2); see
SATTraceDataset's docstring.

Run via Slurm (scripts/slurm/sat_train_transformer.sbatch), not on the
Sherlock login node -- building/training the model needs a CUDA-enabled
py-pytorch env.
"""
import json
import os

import numpy as np
import torch

from .config import SATConfig, SATModelConfig, SATTrainConfig
from .ksat import sample_random_ksat, solve_with_trace
from .sat_dataset import SATTraceDataset, iterate_sat_batches
from .sat_transformer import SATTransformer, sat_accuracy, sat_nll_loss


def build_traces(sat_cfg: SATConfig, n_instances, seed_start):
    """Each returned trace dict is solve_with_trace's output plus a
    "clauses" key, for satisfying_completion_rate's satisfaction check."""
    traces = []
    for i in range(n_instances):
        rng = np.random.default_rng(seed_start + i)
        clauses = sample_random_ksat(sat_cfg, rng)
        trace = solve_with_trace(clauses, sat_cfg.n_vars)
        trace["clauses"] = clauses
        traces.append(trace)
    return traces


def baseline_accuracy(dataset: SATTraceDataset):
    """Accuracy of always predicting each token dimension's majority class
    over the whole dataset -- context for judging a trained model's
    accuracy against, since many clause-token dimensions are trivially 0 by
    construction (a variable absent from a k-clause when k << n_vars), so a
    high raw accuracy number can be mostly free."""
    dim = dataset.n_vars + 1
    counts = np.zeros((dim, 3), dtype=np.int64)  # class -1,0,1 -> column 0,1,2
    for seq in dataset.sequences:
        target_idx = (seq[1:] + 1).astype(np.int64)  # matches __getitem__'s (input, target) split
        for c in range(3):
            counts[:, c] += (target_idx == c).sum(axis=0)
    return float(counts.max(axis=1).sum() / counts.sum())


def evaluate(model, dataset, indices, device, batch_size=8):
    model.eval()
    total_loss, total_n = 0.0, 0
    total_correct, total_dims = 0.0, 0
    with torch.no_grad():
        for x, y, mask in iterate_sat_batches(dataset, indices, batch_size, shuffle=False):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, valid_mask=mask)
            loss = sat_nll_loss(logits, y, mask)
            acc = sat_accuracy(logits, y, mask)
            n_valid = mask.sum().item()
            n_dims = n_valid * (dataset.n_vars + 1)
            total_loss += loss.item() * n_valid
            total_correct += acc * n_dims
            total_dims += n_dims
            total_n += n_valid
    return {"nll": total_loss / total_n, "accuracy": total_correct / total_dims}


def satisfying_completion_rate(model, traces, n_vars, device):
    """Among SAT instances (with >=2 trace tokens, and short enough to fit
    the model's positional embedding), teacher-force on the trace up through
    the second-to-last token and check whether the model's greedy next-token
    prediction satisfies every original clause. See module docstring.

    Filters by model.pos_embed.shape[0] (not just train_cfg.max_trace_tokens)
    because this is called on the *raw* trace list, not through
    SATTraceDataset -- unlike the batched train/eval path, nothing else caps
    sequence length here, and feeding a longer-than-max_len sequence crashes
    inside SATTransformer.forward."""
    model.eval()
    max_len = model.pos_embed.shape[0]
    sat_traces = [
        t for t in traces
        if t["satisfiable"] and 2 <= t["tokens"].shape[0] and t["tokens"].shape[0] - 1 <= max_len
    ]
    if not sat_traces:
        return None
    successes = 0
    with torch.no_grad():
        for trace in sat_traces:
            x = torch.from_numpy(trace["tokens"][:-1]).unsqueeze(0).to(device)
            logits = model(x)
            pred = logits[0, -1].argmax(dim=-1).cpu().numpy() - 1  # (n_vars+1,) in {-1,0,1}
            assignment = pred[:n_vars].astype(np.int8)
            clauses = trace["clauses"]
            satisfied = all(np.any(clauses[c] * assignment == 1) for c in range(clauses.shape[0]))
            successes += int(satisfied)
    return successes / len(sat_traces)


def train_sat_model(sat_cfg: SATConfig, model_cfg: SATModelConfig, train_cfg: SATTrainConfig):
    device = train_cfg.device if torch.cuda.is_available() else "cpu"

    train_traces = build_traces(sat_cfg, train_cfg.n_train_instances, sat_cfg.seed)
    val_traces = build_traces(sat_cfg, train_cfg.n_val_instances, sat_cfg.seed + train_cfg.n_train_instances)

    max_tokens = train_cfg.max_trace_tokens or None
    train_dataset = SATTraceDataset(train_traces, sat_cfg.n_vars, max_tokens=max_tokens)
    val_dataset = SATTraceDataset(val_traces, sat_cfg.n_vars, max_tokens=max_tokens)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            f"max_trace_tokens={train_cfg.max_trace_tokens} left an empty dataset "
            f"(train={len(train_dataset)}, val={len(val_dataset)} instances) -- "
            "increase max_trace_tokens or n_train_instances/n_val_instances."
        )
    max_len = max(seq.shape[0] for seq in train_dataset.sequences + val_dataset.sequences)

    baseline_acc = baseline_accuracy(val_dataset)
    print(f"majority-class baseline accuracy (val): {baseline_acc:.4f}")

    model = SATTransformer(
        sat_cfg.n_vars, d_model=model_cfg.d_model, num_layers=model_cfg.num_layers,
        nhead=model_cfg.nhead, max_len=max_len,
        dim_feedforward=model_cfg.dim_feedforward or None,
        qk_dim=model_cfg.qk_dim or None,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    rng = np.random.default_rng(sat_cfg.seed)

    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))

    history = []
    for epoch in range(train_cfg.epochs):
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        for x, y, mask in iterate_sat_batches(train_dataset, train_indices, train_cfg.batch_size, rng=rng):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            opt.zero_grad()
            logits = model(x, valid_mask=mask)
            loss = sat_nll_loss(logits, y, mask)
            loss.backward()
            opt.step()
            n_valid = mask.sum().item()
            epoch_loss += loss.item() * n_valid
            epoch_n += n_valid

        val_metrics = evaluate(model, val_dataset, val_indices, device, batch_size=train_cfg.batch_size)
        completion_rate = satisfying_completion_rate(model, val_traces, sat_cfg.n_vars, device)
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / epoch_n,
            **val_metrics,
            "val_satisfying_completion_rate": completion_rate,
            "baseline_accuracy": baseline_acc,
        }
        history.append(record)
        print(record)

    return model, history, baseline_acc


def save_checkpoint(model, history, out_dir, sat_cfg: SATConfig, model_cfg: SATModelConfig, baseline_acc=None):
    """Saves model.pt + history.json + config.json. config.json records
    everything needed to reconstruct the exact architecture and reload
    model.pt later (notably max_len: the positional embedding is a
    fixed-shape parameter baked into the state_dict, so a reload must
    reconstruct SATTransformer with the *same* max_len used at training
    time, not just the same n_vars/d_model/num_layers/nhead) -- this is what
    lets a notebook load a trained checkpoint and generate from it without
    retraining."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    # Read the *resolved* dim_feedforward off the model itself, not
    # model_cfg.dim_feedforward (which may be the 0="auto" sentinel) -- a
    # reload must reconstruct the exact same architecture regardless of how
    # the original value was specified. .layers[0] only exists on the
    # built-in nn.TransformerEncoder path; the custom low-rank path stores
    # its blocks directly in a plain ModuleList.
    first_layer = model.encoder[0] if model._custom_attn else model.encoder.layers[0]
    config = {
        "n_vars": sat_cfg.n_vars, "k": sat_cfg.k, "alpha": sat_cfg.alpha,
        "d_model": model_cfg.d_model, "num_layers": model_cfg.num_layers, "nhead": model_cfg.nhead,
        "dim_feedforward": first_layer.linear1.out_features,
        "qk_dim": model.qk_dim,
        "max_len": model.pos_embed.shape[0],
        "baseline_accuracy": baseline_acc,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_checkpoint(out_dir, device="cpu"):
    """Inverse of save_checkpoint: reconstructs SATTransformer from
    config.json and loads model.pt's weights, for exploring a trained model
    (generation, attention inspection, etc.) without retraining."""
    with open(os.path.join(out_dir, "config.json")) as f:
        config = json.load(f)
    model = SATTransformer(
        config["n_vars"], d_model=config["d_model"], num_layers=config["num_layers"],
        nhead=config["nhead"], max_len=config["max_len"],
        # .get(...): older checkpoints (before dim_feedforward/qk_dim were
        # configurable) have no such keys -- None resolves to SATTransformer's
        # own defaults (4 * d_model, full-rank d_model respectively), which is
        # what those models were actually trained with.
        dim_feedforward=config.get("dim_feedforward"),
        qk_dim=config.get("qk_dim"),
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(out_dir, "model.pt"), map_location=device, weights_only=True))
    model.eval()
    return model, config


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-vars", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=4.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=1)
    ap.add_argument("--dim-feedforward", type=int, default=0, help="0 = SATTransformer's default (4 * d_model)")
    ap.add_argument("--qk-dim", type=int, default=0, help="0 = full rank (= d_model)")
    ap.add_argument("--n-train-instances", type=int, default=256)
    ap.add_argument("--n-val-instances", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max-trace-tokens", type=int, default=0, help="0 = no cap")
    ap.add_argument("--out-dir", required=True, help="e.g. $SCRATCH/spin_gnn/sat_checkpoints/run0")
    args = ap.parse_args()

    sat_cfg = SATConfig(n_vars=args.n_vars, k=args.k, alpha=args.alpha, seed=args.seed)
    model_cfg = SATModelConfig(
        d_model=args.d_model, num_layers=args.num_layers, nhead=args.nhead,
        dim_feedforward=args.dim_feedforward, qk_dim=args.qk_dim,
    )
    train_cfg = SATTrainConfig(
        n_train_instances=args.n_train_instances, n_val_instances=args.n_val_instances,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
        max_trace_tokens=args.max_trace_tokens,
    )

    trained_model, run_history, run_baseline_acc = train_sat_model(sat_cfg, model_cfg, train_cfg)
    save_checkpoint(trained_model, run_history, args.out_dir, sat_cfg, model_cfg, baseline_acc=run_baseline_acc)
