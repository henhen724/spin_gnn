"""Concatenate sat2_hparam_sweep.py's per-task sweep_m*_h*_L*.csv rows from
a sweep directory into one table. Mirrors aggregate_sat_sweep.py.

Usage: python -m spin_gnn.aggregate_sat2_sweep $SCRATCH/spin_gnn/sat2_hparam_sweep
"""
import csv
import glob
import os
import sys


def aggregate(sweep_dir):
    paths = sorted(glob.glob(os.path.join(sweep_dir, "sweep_m*_h*_L*.csv")))
    if not paths:
        raise FileNotFoundError(f"no sweep_*.csv files found under {sweep_dir}")
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))

    # Older sweeps predate the norm_first column entirely (pre-norm didn't
    # exist as an option yet) -- they always used nn.TransformerEncoderLayer's
    # own default, post-norm. Backfill rather than leave heterogeneous row
    # schemas, same story as aggregate_sat_sweep.py's dim_feedforward/qk_dim
    # backfill.
    for r in rows:
        if "norm_first" not in r:
            r["norm_first"] = "False"

    rows.sort(key=lambda r: (int(r["m_clauses"]), int(r["nhead"]), int(r["num_layers"])))

    out_path = os.path.join(sweep_dir, "aggregated.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows, {len(paths)} source files)")

    best = max(rows, key=lambda r: float(r["solve_rate"]))
    print(
        f"best solve_rate: m_clauses={best['m_clauses']} d_model={best['d_model']} nhead={best['nhead']} "
        f"num_layers={best['num_layers']} lr={best['lr']} solve_rate={float(best['solve_rate']):.4f} "
        f"checkpoint_dir={best['checkpoint_dir']}"
    )
    return rows


if __name__ == "__main__":
    aggregate(sys.argv[1])
