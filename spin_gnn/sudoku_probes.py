"""Linear probes for DD state in the residual stream, layer by layer.

Asks one question: is the information DD needs linearly present, and where?
The design hinges on a contrast rather than on any single probe's score:

  known_mask    -- bookkeeping. Which nodes have a color yet. Trackable by
                   copying and position alone; a high score here says little.
  frontier_mask -- the DD predicate. Which nodes are deducible right now,
                   which requires joining the graph table to the known set.
  next_node     -- the specific node about to be emitted.
  n_known_nbrs  -- the per-node count DD thresholds at n_colors-1; the
                   natural intermediate quantity if the model computes DD
                   the way DD does.

Layer 0 is the token+position embedding, before any attention or MLP has
run, and is the control: a probe that already scores well at layer 0 is
reading the tokenization, not a learned computation. The headline is the
layer-0 -> deep-layer *lift* on frontier_mask relative to the lift on
known_mask.

Probes are logistic regressions fit with plain gradient descent on a train
split and scored on a held-out split of DIFFERENT instances, so a probe
cannot win by memorizing positions.

Usage:
    python -m spin_gnn.sudoku_probes --checkpoint <dir> --out-csv <csv>
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .sudoku_activations import collect_dataset
from .train_sudoku_transformer import build_instances, load_checkpoint

# (label key, kind): "multilabel" = N independent per-node bits,
# "categorical" = one n_nodes-way or n_colors-way choice.
PROBE_TARGETS = [
    ("frontier_mask", "multilabel"),
    ("remaining_mask", "multilabel"),
    ("known_mask", "multilabel"),
    ("next_node", "categorical"),
    ("next_color", "categorical"),
]


def _average_ranks(x):
    """Ranks of x, 1-based, ties sharing their mean rank."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i, n = 0, len(x)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(scores, labels):
    """P(score of a random positive > score of a random negative), ties half.

    Threshold-free, which is the point: balanced accuracy thresholds the BCE
    probe's logits at 0, and on a ~15%-positive target that threshold is
    badly placed -- the probe under-predicts positives, so balanced accuracy
    reports a floor rather than what the direction in activation space
    actually separates. AUROC asks only whether positives are ranked above
    negatives. For a binary-valued *rule* (0/1 scores) AUROC collapses to
    balanced accuracy exactly, so hand baselines stay directly comparable.
    """
    pos = labels > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _average_ranks(np.asarray(scores, dtype=np.float64))
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores, labels):
    """Area under precision-recall, the step ("interpolation-free") estimate.

    Complements AUROC on a sparse target: AUROC is dominated by the easy
    negatives, AP by whether the top of the ranking is actually positive.
    Chance is the base rate, so read it against `positive_rate`.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n_pos = int((labels > 0.5).sum())
    if n_pos == 0:
        return float("nan")
    y = labels[np.argsort(-scores, kind="mergesort")] > 0.5
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float(precision[y].sum() / n_pos)


def fit_linear_probe(x_tr, y_tr, x_te, y_te, kind, epochs=300, lr=0.05, weight_decay=1e-4, device="cpu"):
    """Returns a dict of held-out metrics. For multilabel targets the headline
    is balanced accuracy, because frontier_mask is sparse (a round's frontier
    is a handful of the n_nodes nodes) and plain accuracy is dominated by the
    all-zeros prediction. balanced_accuracy is a threshold-dependent FLOOR,
    though (see auroc); quote `auroc`/`average_precision` instead."""
    x_tr = x_tr.to(device).float()
    x_te = x_te.to(device).float()
    mu, sigma = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
    x_tr, x_te = (x_tr - mu) / sigma, (x_te - mu) / sigma

    if kind == "multilabel":
        y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
        y_te_t = torch.as_tensor(y_te, dtype=torch.float32, device=device)
        out_dim = y_tr.shape[1]
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        y_tr_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)
        y_te_t = torch.as_tensor(y_te, dtype=torch.long, device=device)
        out_dim = int(max(y_tr.max(), y_te.max())) + 1
        loss_fn = torch.nn.CrossEntropyLoss()

    probe = torch.nn.Linear(x_tr.shape[1], out_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(probe(x_tr), y_tr_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = probe(x_te)
        if kind == "multilabel":
            pred = (logits > 0).float()
            tp = float((pred * y_te_t).sum())
            fp = float((pred * (1 - y_te_t)).sum())
            fn = float(((1 - pred) * y_te_t).sum())
            tn = float(((1 - pred) * (1 - y_te_t)).sum())
            tpr = tp / max(1.0, tp + fn)
            tnr = tn / max(1.0, tn + fp)
            base = float(y_te_t.mean())
            # Pooled over every (position, node) bit, like the rates above.
            s = logits.flatten().double().cpu().numpy()
            y = y_te_t.flatten().double().cpu().numpy()
            return {
                "accuracy": float((pred == y_te_t).float().mean()),
                "balanced_accuracy": 0.5 * (tpr + tnr),
                "auroc": auroc(s, y),
                "average_precision": average_precision(s, y),
                "f1": 2 * tp / max(1.0, 2 * tp + fp + fn),
                "positive_rate": base,
                # Always-predict-majority on a sparse target.
                "majority_accuracy": max(base, 1 - base),
            }
        pred = logits.argmax(dim=-1)
        counts = np.bincount(y_te, minlength=out_dim)
        return {
            "accuracy": float((pred == y_te_t).float().mean()),
            "balanced_accuracy": float((pred == y_te_t).float().mean()),
            # Undefined for a k-way choice; top-1 accuracy is already the
            # threshold-free quantity there, so there is nothing to correct.
            "auroc": float("nan"),
            "average_precision": float("nan"),
            "f1": float("nan"),
            "positive_rate": float("nan"),
            "majority_accuracy": float(counts.max() / max(1, counts.sum())),
        }


def probe_checkpoint(ckpt_dir, n_train_instances, n_test_instances, max_pos, device, epochs, seed=0):
    model, vocab, config = load_checkpoint(ckpt_dir, device=device)
    sudoku_cfg = SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        rho=config["rho"], rho_max=config["rho_max"],
        hint_mode=config.get("hint_mode", "removal"), scratchpad=config.get("scratchpad", False),
    )
    # Disjoint instance pools: a probe must generalize across instances, not
    # just across positions within instances it has already seen.
    train_inst = build_instances(sudoku_cfg, n_train_instances, seed_start=500_000_000 + seed)
    test_inst = build_instances(sudoku_cfg, n_test_instances, seed_start=900_000_000 + seed)

    acts_tr, lab_tr = collect_dataset(model, train_inst, vocab, device, max_pos, np.random.default_rng(seed))
    acts_te, lab_te = collect_dataset(model, test_inst, vocab, device, max_pos, np.random.default_rng(seed + 1))
    print(f"{os.path.basename(ckpt_dir)}: {acts_tr.shape[1]} train / {acts_te.shape[1]} test positions, "
          f"{acts_tr.shape[0]} layers")

    rows = []
    for layer in range(acts_tr.shape[0]):
        for target, kind in PROBE_TARGETS:
            m = fit_linear_probe(
                acts_tr[layer], lab_tr[target], acts_te[layer], lab_te[target], kind,
                epochs=epochs, device=device,
            )
            rows.append({
                "checkpoint": os.path.basename(ckpt_dir), "seed": seed,
                "layer": layer, "target": target, "kind": kind,
                "n_train_positions": int(acts_tr.shape[1]), "n_test_positions": int(acts_te.shape[1]),
                **{k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()},
            })
        f = next(r for r in rows if r["layer"] == layer and r["target"] == "frontier_mask")
        k = next(r for r in rows if r["layer"] == layer and r["target"] == "known_mask")
        print(f"  layer {layer:2d}: frontier bal_acc={f['balanced_accuracy']:.3f}  "
              f"known bal_acc={k['balanced_accuracy']:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, nargs="+", required=True,
                    help="one or more checkpoint directories (as written by save_checkpoint)")
    ap.add_argument("--n-train-instances", type=int, default=120)
    ap.add_argument("--n-test-instances", type=int, default=60)
    ap.add_argument("--max-positions-per-instance", type=int, default=12)
    ap.add_argument("--probe-epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, nargs="+", default=[0],
                    help="one run per seed; the seed shifts BOTH instance pools and the "
                         "position sampling, so spread across seeds is the measurement "
                         "noise a single-seed probe curve does not show")
    ap.add_argument("--out-csv", type=str, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for seed in args.seed:
        for ckpt in args.checkpoint:
            rows.extend(probe_checkpoint(
                ckpt, args.n_train_instances, args.n_test_instances, args.max_positions_per_instance,
                device, args.probe_epochs, seed,
            ))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
