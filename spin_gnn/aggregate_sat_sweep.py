"""Concatenate sat_hparam_sweep.py's per-task sweep_d*_h*.csv rows from a
sweep directory into one table, sorted by (d_model, nhead), for finding the
best-performing config. Mirrors spin_gnn/aggregate_scaling.py.

Usage: python -m spin_gnn.aggregate_sat_sweep $SCRATCH/spin_gnn/sat_hparam_sweep
"""
import csv
import glob
import os
import sys


def aggregate(sweep_dir):
    paths = sorted(glob.glob(os.path.join(sweep_dir, "sweep_d*_h*.csv")))
    if not paths:
        raise FileNotFoundError(f"no sweep_*.csv files found under {sweep_dir}")
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))

    # Older sweeps (before dim_feedforward was a configurable/recorded field)
    # wrote rows without that column at all -- csv.DictWriter requires every
    # row to have exactly the fieldnames' keys, so mixing old- and new-schema
    # rows crashes unless backfilled first. Those older runs always used
    # SATTransformer's implicit default (4 * d_model), so backfill the real
    # value rather than leaving it blank -- keeps the column meaningful
    # (a real number, not NaN) for every row once loaded downstream.
    for r in rows:
        if "dim_feedforward" not in r:
            r["dim_feedforward"] = str(4 * int(r["d_model"]))

    rows.sort(key=lambda r: (int(r["d_model"]), int(r["nhead"])))

    out_path = os.path.join(sweep_dir, "aggregated.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows, {len(paths)} source files)")

    best = max(rows, key=lambda r: float(r["best_accuracy"]))
    print(
        f"best: d_model={best['d_model']} nhead={best['nhead']} "
        f"best_accuracy={float(best['best_accuracy']):.4f} "
        f"(baseline={float(best['baseline_accuracy']):.4f}) "
        f"checkpoint_dir={best['checkpoint_dir']}"
    )
    return rows


if __name__ == "__main__":
    aggregate(sys.argv[1])
