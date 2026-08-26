"""Standalone Bayes-optimal accuracy ceiling per (gamma_kind, context_length).

bayes_accuracy = how often thresholding the *true* analytic p_true at 0.5
matches the actually-realized next spin, on a sweep's held-out test pool.
It depends only on (gamma_kind, gamma_param, context_length, n_spins, beta,
trajectory_length, n_test_realizations) -- NOT on n_train_sequences or model
type, since scale_sweep.py and transformer_scale_sweep.py both build their
held-out pool the same way (test_cfg.disorder.seed = 10_000*context_length +
5_000), independent of the training-set-size axis. So this needs computing
once per (gamma_kind, context_length), not once per sweep row, and needs no
trained model at all -- just regenerating the same (deterministic, seeded)
test realizations.

Written to backfill sweeps that ran before train.evaluate() started
recording bayes_accuracy directly (any new sweep run already has
bayes_accuracy_at_min in its own output CSV -- this script is only needed to
retrofit that reference value onto earlier results without re-running
training).
"""
import csv
import os

from .config import ExperimentConfig
from .train import build_windows

K_VALUES = [2, 4, 8, 16, 32]
GAMMA_SETTINGS = [("power_law", 1.0), ("ones", 0.0), ("identity", 0.0)]


def bayes_accuracy_for(cfg, n_test_realizations=8):
    k = cfg.disorder.context_length
    cfg.disorder.seed = 10_000 * k + 5_000
    cfg.train.n_realizations = n_test_realizations
    test_windows = build_windows(cfg)

    total_correct, total_n = 0, 0
    for w in test_windows:
        for i in range(len(w)):
            _, y, p_true = w[i]
            bayes_pred = (p_true > 0.5).long()
            total_correct += (bayes_pred == y).sum().item()
            total_n += y.numel()
    return total_correct / total_n


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-spins", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--trajectory-length", type=int, default=512)
    ap.add_argument("--n-test-realizations", type=int, default=8)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    rows = []
    for gamma_kind, gamma_param in GAMMA_SETTINGS:
        for k in K_VALUES:
            cfg = ExperimentConfig()
            cfg.disorder.n_spins = args.n_spins
            cfg.disorder.beta = args.beta
            cfg.disorder.context_length = k
            cfg.disorder.gamma_kind = gamma_kind
            cfg.disorder.gamma_param = gamma_param
            cfg.train.trajectory_length = args.trajectory_length
            acc = bayes_accuracy_for(cfg, args.n_test_realizations)
            row = {
                "gamma_kind": gamma_kind,
                "gamma_param": gamma_param,
                "context_length": k,
                "bayes_accuracy": acc,
            }
            rows.append(row)
            print(row)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out_csv}")
