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
  known_probe a linear probe given ONLY the N known_mask bits as features,
              fit by the same procedure as the model probes. NOT a ceiling on
              "what bookkeeping alone can imply", as this script first claimed:
              it is trained with BCE on a ~15%-positive target, so it
              under-predicts positives and its balanced accuracy is a FLOOR.
  degree_aug  known_mask plus n_known_nbrs, i.e. bookkeeping plus a count that
              needs the graph but not the forcing rule
  shuffled    known_mask shuffled across positions, the chance control

Scoring is pooled over all (position, node) bits with the same tpr/tnr
definition fit_linear_probe uses, so the numbers are directly comparable to
the balanced_accuracy column in probes.csv / series_probes.csv.

That comparability is the problem, though: the first run of this script came
back with not_known at 0.734 (ABOVE the model's 0.679) and known_probe at
0.588 (BELOW it), which is not a contradiction but a demonstration that
balanced accuracy at a fixed 0-logit threshold ranks these procedures by how
aggressively they predict positives rather than by what they know. The
threshold-free columns -- auroc, average_precision -- are therefore reported
alongside, and are what should be quoted.
"""
import argparse
import json

import numpy as np
import torch

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_activations import dd_state_labels
from spin_gnn.sudoku_probes import auroc, average_precision, fit_linear_probe
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

    def probe_on(x_tr, x_te):
        m = fit_linear_probe(
            torch.as_tensor(x_tr, dtype=torch.float32), f_tr,
            torch.as_tensor(x_te, dtype=torch.float32), f_te,
            "multilabel", epochs=args.probe_epochs,
        )
        return {k: m[k] for k in ("balanced_accuracy", "auroc", "average_precision")}

    res = {}
    # A 0/1 rule, so its AUROC equals its balanced accuracy by construction
    # (tested in tests/test_sudoku_probes.py); reported anyway so the table
    # has one number per column and the identity is visible rather than
    # asserted in prose.
    res["not_known"] = {
        "balanced_accuracy": balanced_accuracy(1 - k_te, f_te),
        "auroc": auroc((1 - k_te).ravel(), f_te.ravel()),
        "average_precision": average_precision((1 - k_te).ravel(), f_te.ravel()),
    }
    res["known_probe"] = probe_on(k_tr, k_te)
    res["degree_aug"] = probe_on(np.hstack([k_tr, nb_tr]), np.hstack([k_te, nb_te]))

    rng = np.random.default_rng(0)
    res["shuffled"] = probe_on(k_tr[rng.permutation(len(k_tr))], k_te[rng.permutation(len(k_te))])
    res["positive_rate"] = {"balanced_accuracy": 0.5, "auroc": 0.5,
                            "average_precision": float(f_te.mean())}

    print("\nfrontier_mask predictability from label information only:")
    print(f"  {'':>12}  {'bal_acc':>8} {'auroc':>8} {'avg_prec':>9}")
    for name in ("positive_rate", "shuffled", "not_known", "known_probe", "degree_aug"):
        r = res[name]
        print(f"  {name:>12}: {r['balanced_accuracy']:8.4f} {r['auroc']:8.4f} "
              f"{r['average_precision']:9.4f}")
    print("  (`positive_rate` is the chance row: AP's chance level is the base rate,")
    print("   not 0.5, so AP must be read against it.)")
    print("\nfor comparison, the measured model probe (48580216, d0128 @900ep):")
    print("  frontier_mask peak 0.679   known_mask peak 0.919   layer-0 control 0.571")
    print("  -- balanced accuracy, which is why the re-probe with --seed 0 1 2 is")
    print("     needed before any of these can be compared on auroc.")

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
