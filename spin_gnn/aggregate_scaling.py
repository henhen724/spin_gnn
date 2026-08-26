"""Concatenate per-job scale_K*_N*.csv rows from a scale_sweep.sbatch run into
one table, sorted by (context_length, n_train_sequences), for fitting/plotting
how min_test_nll scales with context length and training-set size.

Usage: python -m spin_gnn.aggregate_scaling $SCRATCH/spin_gnn/scale_sweep/power1
"""
import csv
import glob
import os
import sys


def aggregate(sweep_dir):
    paths = sorted(glob.glob(os.path.join(sweep_dir, "scale_K*_N*.csv")))
    if not paths:
        raise FileNotFoundError(f"no scale_*.csv files found under {sweep_dir}")
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    rows.sort(key=lambda r: (int(r["context_length"]), int(r["n_train_sequences"])))

    out_path = os.path.join(sweep_dir, "aggregated.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows, {len(paths)} source files)")
    return rows


if __name__ == "__main__":
    aggregate(sys.argv[1])
