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

Those three are the FREE scores: bookkeeping, and a count that needs the
graph but not the forcing rule. Two more rows bracket the other end, and they
are a CEILING rather than a free score, because they are handed the quantity
DD actually thresholds. n_known_nbrs is not that quantity: a node is
deducible exactly when its known neighbours cover q-1 DISTINCT colours, and
at degree 5 with q=4 the known neighbours usually repeat, so a node can have
four known neighbours and still not be forced.

  color_aug       a probe given known_mask plus n_known_colors
  rule_on_visible DD's rule computed, not fitted: unknown and n_known_colors
                  >= q-1. It lands near 0.83 rather than 1.0, and the reason
                  is worth more than the number -- see the comment at the
                  call site. frontier_mask is the ROUND's set; the visible
                  state has already moved on within the round.

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

from spin_gnn.ensemble_args import FIELDS, add_ensemble_args, ensemble_config
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
    keys = ("frontier_mask", "known_mask", "n_known_nbrs", "n_known_colors")
    out = {k: [] for k in keys}
    for adj, _colors, hints, trace in build_instances(cfg, n_instances, seed_start=seed_start):
        _pos, lab = dd_state_labels(adj, trace, vocab)
        for k in keys:
            out[k].append(lab[k])
    cast = {"frontier_mask": np.int8, "known_mask": np.int8,
            "n_known_nbrs": np.float32, "n_known_colors": np.float32}
    return {k: np.concatenate(v).astype(cast[k]) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    # The baseline is a property of the TASK, so it has to be recomputed for
    # every ensemble a probe run is read against; it was pinned to N=24 here.
    add_ensemble_args(ap)
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--probe-epochs", type=int, default=300)
    # The model probes are run at three seeds and quoted with their spread, so
    # a point estimate on the other side of the comparison is the weaker half.
    # This shifts both instance pools, which is the only randomness here.
    ap.add_argument("--pool-offset", type=int, default=0,
                    help="shift both instance pools; repeat the run to get a spread")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    cfg = ensemble_config(args)
    vocab = build_vocab(cfg)
    off = args.pool_offset * 10_000_000
    tr = gather(cfg, vocab, args.n_train, 500_000_000 + off)
    te = gather(cfg, vocab, args.n_test, 900_000_000 + off)
    f_tr, k_tr, nb_tr, nc_tr = (tr["frontier_mask"], tr["known_mask"],
                                tr["n_known_nbrs"], tr["n_known_colors"])
    f_te, k_te, nb_te, nc_te = (te["frontier_mask"], te["known_mask"],
                                te["n_known_nbrs"], te["n_known_colors"])
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
    res["color_aug"] = probe_on(np.hstack([k_tr, nc_tr]), np.hstack([k_te, nc_te]))

    # DD's forcing rule, computed rather than fitted: v is deducible when it is
    # unknown and its known neighbours already show q-1 distinct colours (a
    # proper colouring makes q-1 the maximum, so this is a threshold, not an
    # equality test in disguise).
    #
    # This does NOT reproduce frontier_mask, and the gap is the point. It was
    # written with an assert at 0.999 and the assert fired at 0.832: the label
    # is the ROUND's set -- every node round r emits, fixed at the round
    # boundary -- while n_known_colors is the state VISIBLE at this position,
    # which already includes the deductions this round has written out. So a
    # node emitted earlier in the round is in the label and `known` by now, and
    # a node this round's own emissions have just unlocked is deducible now but
    # belongs to round r+1. A perfect DD reasoner scoring the visible state
    # therefore lands near 0.83, not 1.0, and that -- not 1.0 -- is the ceiling
    # a model probe should be read against.
    rule = ((nc_te >= cfg.n_colors - 1) & (k_te == 0)).astype(np.int8)
    res["rule_on_visible"] = {
        "balanced_accuracy": balanced_accuracy(rule, f_te),
        "auroc": auroc(rule.ravel().astype(np.float32), f_te.ravel()),
        "average_precision": average_precision(rule.ravel().astype(np.float32), f_te.ravel()),
    }

    rng = np.random.default_rng(0)
    res["shuffled"] = probe_on(k_tr[rng.permutation(len(k_tr))], k_te[rng.permutation(len(k_te))])
    res["positive_rate"] = {"balanced_accuracy": 0.5, "auroc": 0.5,
                            "average_precision": float(f_te.mean())}

    print("\nfrontier_mask predictability from label information only:")
    print(f"  {'':>13}  {'bal_acc':>8} {'auroc':>8} {'avg_prec':>9}")
    for name in ("positive_rate", "shuffled", "not_known", "known_probe", "degree_aug",
                 "color_aug", "rule_on_visible"):
        r = res[name]
        print(f"  {name:>13}: {r['balanced_accuracy']:8.4f} {r['auroc']:8.4f} "
              f"{r['average_precision']:9.4f}")
    print("  (`positive_rate` is the chance row: AP's chance level is the base rate,")
    print("   not 0.5, so AP must be read against it.)")
    print("  THE FREE SCORE a model probe must beat is the best of not_known /")
    print("  known_probe / degree_aug -- bookkeeping, and a count that needs the graph")
    print("  but not the forcing rule. color_aug and rule_on_visible are the other end:")
    print("  they are handed n_known_colors, which IS DD's computation, so they are the")
    print("  CEILING, not a free score -- and rule_on_visible shows that ceiling is well")
    print("  short of 1.0, because frontier_mask is the ROUND's set and the visible")
    print("  state has already moved on within the round.")

    if args.out_json:
        # Carry the ensemble: this number is a property of the task, so an
        # N=96 fall baseline and the N=24 one are not interchangeable.
        res["ensemble"] = {f: getattr(cfg, f) for f in FIELDS}
        res["run"] = {"n_train": args.n_train, "n_test": args.n_test,
                      "probe_epochs": args.probe_epochs,
                      "pool_offset": args.pool_offset,
                      "n_test_positions": int(f_te.shape[0]),
                      "positive_rate": float(f_te.mean())}
        with open(args.out_json, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
