"""Next-token-prediction training for SATReasoningTransformer over
spin_gnn.ksat2's event-based reasoning traces.

Mirrors train_sat_transformer.py's structure, but the encoding is different
enough (categorical token ids vs. real-valued vectors) that model, dataset,
and eval are all separate modules (sat_transformer2/sat_dataset2) rather
than extending the originals -- see spin_gnn/ksat2.py's docstring for why.

Reports, alongside train/val cross-entropy and per-token accuracy:
- `baseline_accuracy`: accuracy of always predicting the single dataset-wide
  majority target token (context for judging per-token accuracy against).
- `solve_rate`/`outcome_accuracy`/`well_formed_rate`
  (sat_transformer2.solve_rate): the actual end-to-end capability -- prompt
  with hints+statement only, let the model freely generate its own
  reasoning trace and final answer, then check it against the true
  instance. This is the metric the hyperparameter/clause-count scan
  optimizes for ("totally solves the problem"), not per-token accuracy,
  which (per the copy_vs_delta_breakdown finding on the v1 encoding) can
  look deceptively high while end-to-end success stays near zero.

Run via Slurm (scripts/slurm/sat2_*.sbatch), not the Sherlock login node.
"""
import json
import os

import numpy as np
import torch

from .config import SATReasoningConfig, SATReasoningModelConfig, SATReasoningTrainConfig
from .ksat import sample_random_ksat
from .ksat2 import sample_hints, solve_with_reasoning_trace
from .ksat2_tokens import Vocab, encode_instance
from .sat_dataset2 import SATTokenDataset, iterate_token_batches
from .sat_transformer2 import SATReasoningTransformer, sat2_accuracy, sat2_nll_loss, solve_rate


def build_vocab(sat_cfg: SATReasoningConfig):
    m_clauses = max(1, round(sat_cfg.alpha * sat_cfg.n_vars))
    return Vocab(sat_cfg.n_vars, m_clauses, use_clause_status=sat_cfg.use_clause_status)


def build_instances(sat_cfg: SATReasoningConfig, n_instances, seed_start):
    """Each element: (clauses, n_vars, hints, trace) -- trace is
    ksat2.solve_with_reasoning_trace's return dict."""
    instances = []
    for i in range(n_instances):
        rng = np.random.default_rng(seed_start + i)
        clauses = sample_random_ksat(sat_cfg, rng)
        hints = (
            sample_hints(sat_cfg.n_vars, sat_cfg.n_hints, np.random.default_rng(seed_start + i + 10_000_000))
            if sat_cfg.n_hints else {}
        )
        trace = solve_with_reasoning_trace(clauses, sat_cfg.n_vars, hints=hints)
        instances.append((clauses, sat_cfg.n_vars, hints, trace))
    return instances


def baseline_accuracy(dataset: SATTokenDataset, vocab_size):
    counts = np.zeros(vocab_size, dtype=np.int64)
    for seq in dataset.sequences:
        target = seq[1:]
        counts += np.bincount(target, minlength=vocab_size)
    return float(counts.max() / counts.sum())


def evaluate(model, dataset, indices, device, batch_size=32):
    model.eval()
    total_loss, total_n = 0.0, 0
    total_correct, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y, mask in iterate_token_batches(dataset, indices, batch_size, shuffle=False):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, valid_mask=mask)
            loss = sat2_nll_loss(logits, y, mask)
            acc = sat2_accuracy(logits, y, mask)
            n_valid = mask.sum().item()
            total_loss += loss.item() * n_valid
            total_correct += acc * n_valid
            total_tokens += n_valid
            total_n += n_valid
    return {"nll": total_loss / total_n, "accuracy": total_correct / total_tokens}


def train_sat2_model(sat_cfg: SATReasoningConfig, model_cfg: SATReasoningModelConfig, train_cfg: SATReasoningTrainConfig):
    device = train_cfg.device if torch.cuda.is_available() else "cpu"
    # Previously unseeded: every prior run drew model init (and every
    # dropout mask during training) from whatever state torch's global RNG
    # happened to be in at process start -- the dominant source of the
    # ~3-25x run-to-run solve_rate variance seen across every sweep so far
    # (e.g. 12.3% vs 4.3% at an otherwise-identical config). Seeding it from
    # sat_cfg.seed -- the same value that already controls data
    # sampling/batch order -- makes two runs with the same --seed fully
    # reproducible (data-seed and model-seed share one knob, standard
    # practice), and two runs with different --seed genuinely independent
    # replicates. Not a guarantee of bit-for-bit determinism on GPU (cuDNN
    # algorithm selection/reduction order aren't pinned here), but this
    # removes the dominant noise source either way.
    torch.manual_seed(sat_cfg.seed)
    vocab = build_vocab(sat_cfg)

    train_instances = build_instances(sat_cfg, train_cfg.n_train_instances, sat_cfg.seed)
    val_instances = build_instances(sat_cfg, train_cfg.n_val_instances, sat_cfg.seed + train_cfg.n_train_instances)

    train_tokens = [encode_instance(c, n, h, t, vocab) for c, n, h, t in train_instances]
    val_tokens = [encode_instance(c, n, h, t, vocab) for c, n, h, t in val_instances]

    max_len = train_cfg.max_len or None
    train_dataset = SATTokenDataset(train_tokens, max_len=max_len)
    val_dataset = SATTokenDataset(val_tokens, max_len=max_len)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"empty dataset: train={len(train_dataset)}, val={len(val_dataset)}")
    model_max_len = max(seq.shape[0] for seq in train_dataset.sequences + val_dataset.sequences)

    baseline_acc = baseline_accuracy(val_dataset, vocab.size)
    print(f"vocab_size={vocab.size} majority-class baseline accuracy (val): {baseline_acc:.4f}")

    model = SATReasoningTransformer(
        vocab.size, d_model=model_cfg.d_model, num_layers=model_cfg.num_layers,
        nhead=model_cfg.nhead, max_len=model_max_len,
        dim_feedforward=model_cfg.dim_feedforward or None, norm_first=model_cfg.norm_first,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    rng = np.random.default_rng(sat_cfg.seed)

    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))

    history = []
    for epoch in range(train_cfg.epochs):
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        for x, y, mask in iterate_token_batches(train_dataset, train_indices, train_cfg.batch_size, rng=rng):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, valid_mask=mask)
            loss = sat2_nll_loss(logits, y, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            n_valid = mask.sum().item()
            epoch_loss += loss.item() * n_valid
            epoch_n += n_valid

        val_metrics = evaluate(model, val_dataset, val_indices, device, batch_size=train_cfg.batch_size)
        record = {"epoch": epoch, "train_nll": epoch_loss / epoch_n, **val_metrics}
        history.append(record)
        print(f"epoch {epoch}: train_nll={record['train_nll']:.4f} val_nll={val_metrics['nll']:.4f} "
              f"val_accuracy={val_metrics['accuracy']:.4f}")

    solve_metrics = solve_rate(model, val_instances, vocab, device, max_new_tokens=model_max_len)
    print(f"solve_rate={solve_metrics['solve_rate']:.4f} outcome_accuracy={solve_metrics['outcome_accuracy']:.4f} "
          f"well_formed_rate={solve_metrics['well_formed_rate']:.4f}")

    return {
        "model": model, "vocab": vocab, "history": history, "baseline_accuracy": baseline_acc,
        "solve_metrics": solve_metrics, "model_max_len": model_max_len, "val_instances": val_instances,
    }


def save_checkpoint(out_dir, sat_cfg, model_cfg, model, vocab, model_max_len, baseline_acc, solve_metrics, history):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    config = {
        "n_vars": sat_cfg.n_vars, "k": sat_cfg.k, "alpha": sat_cfg.alpha, "n_hints": sat_cfg.n_hints,
        "m_clauses": vocab.m_clauses, "vocab_size": vocab.size, "use_clause_status": vocab.use_clause_status,
        "d_model": model_cfg.d_model, "num_layers": model_cfg.num_layers, "nhead": model_cfg.nhead,
        "dim_feedforward": model.encoder.layers[0].linear1.out_features, "norm_first": model.norm_first,
        "max_len": model_max_len, "baseline_accuracy": baseline_acc, "solve_metrics": solve_metrics,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


def load_checkpoint(out_dir, device="cpu"):
    with open(os.path.join(out_dir, "config.json")) as f:
        config = json.load(f)
    vocab = Vocab(config["n_vars"], config["m_clauses"], use_clause_status=config.get("use_clause_status", False))
    model = SATReasoningTransformer(
        vocab.size, d_model=config["d_model"], num_layers=config["num_layers"],
        nhead=config["nhead"], max_len=config["max_len"], dim_feedforward=config.get("dim_feedforward"),
        norm_first=config.get("norm_first", False),
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(out_dir, "model.pt"), map_location=device, weights_only=True))
    model.eval()
    return model, vocab, config


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vars", type=int, default=16)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.25)
    ap.add_argument("--n-hints", type=int, default=0)
    ap.add_argument("--use-clause-status", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=1)
    ap.add_argument("--dim-feedforward", type=int, default=0)
    ap.add_argument("--norm-first", action="store_true")
    ap.add_argument("--n-train-instances", type=int, default=2000)
    ap.add_argument("--n-val-instances", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    sat_cfg = SATReasoningConfig(n_vars=args.n_vars, k=args.k, alpha=args.alpha, n_hints=args.n_hints,
                                  seed=args.seed, use_clause_status=args.use_clause_status)
    model_cfg = SATReasoningModelConfig(
        d_model=args.d_model, num_layers=args.num_layers, nhead=args.nhead, dim_feedforward=args.dim_feedforward,
        norm_first=args.norm_first,
    )
    train_cfg = SATReasoningTrainConfig(
        n_train_instances=args.n_train_instances, n_val_instances=args.n_val_instances,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
    )
    result = train_sat2_model(sat_cfg, model_cfg, train_cfg)
    if args.out_dir:
        save_checkpoint(
            args.out_dir, sat_cfg, model_cfg, result["model"], result["vocab"], result["model_max_len"],
            result["baseline_accuracy"], result["solve_metrics"], result["history"],
        )
