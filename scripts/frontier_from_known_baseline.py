"""How much of the frontier_mask probe is explained by bookkeeping alone?

The probe sweep reports frontier_mask at balanced accuracy ~0.67 against a
layer-0 control of ~0.57, and that gap has been read as "a weak but nonzero
trace of DD's deducibility predicate". It may be nothing of the kind.

frontier is a SUBSET of not-known: a node that already has a color is never
deducible. So a reader that recovers only known_mask -- which the same sweep
says the model represents at 0.92 -- can score on frontier_mask for free,
without any notion of deducibility. The question this script answers is
whether 0.67 is above or below that free score.

No model is involved. Everything here is computed from DD's own ground-truth
labels, so it measures a property of the TASK, not of any network:

  not_known   pred = 1 - known_mask, the simplest bookkeeping rule
  known_probe a linear probe given ONLY the N known_mask bits as features
              (the fair ceiling on "what bookkeeping alone can imply")
  degree_aug  known_mask plus n_known_nbrs, i.e. bookkeeping plus a count that
              needs the graph but not the forcing rule
  shuffled    known_mask shuffled across positions, the chance control

Scoring is pooled over all (position, node) bits with the same tpr/tnr
definition fit_linear_probe uses, so the numbers are directly comparable to
the balanced_accuracy column in probes.csv / series_probes.csv.
"""
import argparse
import json

import numpy as np
import torch

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_activations import dd_state_labels
from spin_gnn.sudoku_probes import fit_linear_probe
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab


def balanced_accuracy(pred, y):
    """Pooled over every (position, node) bit, matching fit_linear_probe."""
    pred, y = pred.astype(np.float64), y.astype(np.float64)
    tp = float((pred * y).sum())
    fp = float((pred * (1 - y)).sum())
    fn = float(((1 - pred) * y).sum())
    tn = float(((1 - pred) * (1 - y)).sum())
    return 0.5 * (tp / max(1.0, tp + fn) + tn / max(1.0, tn + fp))


def gather(cfg, vocab, n_instances, seed_start):
    frontier, known, nbrs = [], [], []
    for adj, _colors, hints, trace in build_instances(cfg, n_instances, seed_start=seed_start):
        _pos, lab = dd_state_labels(adj, trace, vocab)
        frontier.append(lab["frontier_mask"])
        known.append(lab["known_mask"])
        nbrs.append(lab["n_known_nbrs"])
    return (np.concatenate(frontier).astype(np.int8),
            np.concatenate(known).astype(np.int8),
            np.concatenate(nbrs).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--probe-epochs", type=int, default=300)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    cfg = SudokuReasoningConfig(n_nodes=24, degree=5, n_colors=4)
    vocab = build_vocab(cfg)
    f_tr, k_tr, nb_tr = gather(cfg, vocab, args.n_train, 500_000_000)
    f_te, k_te, nb_te = gather(cfg, vocab, args.n_test, 900_000_000)
    print(f"{f_tr.shape[0]} train / {f_te.shape[0]} test positions, {f_tr.shape[1]} nodes")
    print(f"frontier positive rate: {f_te.mean():.4f}   known positive rate: {k_te.mean():.4f}")

    res = {}
    res["not_known"] = balanced_accuracy(1 - k_te, f_te)

    res["known_probe"] = fit_linear_probe(
        torch.as_tensor(k_tr, dtype=torch.float32), f_tr,
        torch.as_tensor(k_te, dtype=torch.float32), f_te,
        "multilabel", epochs=args.probe_epochs,
    )["balanced_accuracy"]

    res["degree_aug"] = fit_linear_probe(
        torch.as_tensor(np.hstack([k_tr, nb_tr]), dtype=torch.float32), f_tr,
        torch.as_tensor(np.hstack([k_te, nb_te]), dtype=torch.float32), f_te,
        "multilabel", epochs=args.probe_epochs,
    )["balanced_accuracy"]

    rng = np.random.default_rng(0)
    res["shuffled"] = fit_linear_probe(
        torch.as_tensor(k_tr[rng.permutation(len(k_tr))], dtype=torch.float32), f_tr,
        torch.as_tensor(k_te[rng.permutation(len(k_te))], dtype=torch.float32), f_te,
        "multilabel", epochs=args.probe_epochs,
    )["balanced_accuracy"]

    print("\nbalanced accuracy on frontier_mask, from label information only:")
    for name in ("shuffled", "not_known", "known_probe", "degree_aug"):
        print(f"  {name:>12}: {res[name]:.4f}")
    print("\nfor comparison, the measured model probe (48580216, d0128 @900ep):")
    print("  frontier_mask peak 0.679   known_mask peak 0.919   layer-0 control 0.571")

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
