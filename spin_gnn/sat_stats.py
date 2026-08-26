"""Statistics of the unit-propagation / MCV-branching solver dynamics
(spin_gnn.ksat) across many random k-SAT instances.

Samples `--n-instances` independent random k-SAT instances from a SATConfig,
runs solve_with_trace on each, and reports per-instance solver-trace
statistics (how much unit propagation happened, how much branching/
backtracking was needed, how long the resulting token trace is) both as a
printed summary and a CSV under `--out-dir` -- the exploratory pass that
train_sat_transformer.py's training data is drawn from the same generator as.

CPU-only and cheap at the toy scale this defaults to; still run it via Slurm
(scripts/slurm/sat_stats.sbatch), not the Sherlock login node, per this
repo's existing convention (see simulate.py's module docstring).
"""
import csv
import os

import numpy as np

from .config import SATConfig
from .ksat import sample_random_ksat, solve_with_trace


def token_covariance(traces, max_tokens_per_trace=None):
    """Empirical covariance matrix ((n_vars+1, n_vars+1)) of the trace token
    vectors, pooling every token from every trace together (assignment and
    clause tokens alike). Reveals correlation structure the raw per-instance
    stats don't: e.g. whether the "is this a clause vs. assignment token"
    dimension (column n_vars) is a strong axis of variation, or whether
    particular variable columns move together across the token population.
    max_tokens_per_trace optionally subsamples each trace's tokens (their
    first N) to bound memory when pooling many long traces."""
    parts = []
    for t in traces:
        tokens = t["tokens"]
        if tokens.shape[0] == 0:
            continue
        parts.append(tokens[:max_tokens_per_trace] if max_tokens_per_trace else tokens)
    all_tokens = np.concatenate(parts, axis=0)
    return np.cov(all_tokens, rowvar=False)


def _event_counts(events):
    counts = {"decision": 0, "unit_prop": 0, "flip": 0, "backtrack": 0}
    depth, max_depth = 0, 0
    for e in events:
        counts[e["kind"]] += 1
        if e["kind"] == "decision":
            depth += 1
            max_depth = max(max_depth, depth)
        elif e["kind"] == "backtrack":
            depth -= 1
    return counts, max_depth


def instance_stats(cfg: SATConfig, seed):
    rng = np.random.default_rng(seed)
    clauses = sample_random_ksat(cfg, rng)
    trace = solve_with_trace(clauses, cfg.n_vars)
    counts, max_depth = _event_counts(trace["events"])
    return {
        "seed": seed,
        "n_vars": cfg.n_vars,
        "m_clauses": clauses.shape[0],
        "k": cfg.k,
        "alpha": cfg.alpha,
        "satisfiable": int(trace["satisfiable"]),
        "num_decisions": counts["decision"],
        "num_unit_props": counts["unit_prop"],
        "num_flips": counts["flip"],
        "num_backtracks": counts["backtrack"],
        "max_decision_depth": max_depth,
        "num_steps": len(trace["step_lengths"]),
        "num_tokens": int(trace["tokens"].shape[0]),
    }


def run(cfg: SATConfig, n_instances, seed_start, out_dir):
    rows = [instance_stats(cfg, seed_start + i) for i in range(n_instances)]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sat_stats_n{cfg.n_vars}_k{cfg.k}_a{cfg.alpha}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} instances)")

    numeric_fields = [
        "satisfiable", "num_decisions", "num_unit_props", "num_flips",
        "num_backtracks", "max_decision_depth", "num_steps", "num_tokens",
    ]
    print(f"\nn_vars={cfg.n_vars} k={cfg.k} alpha={cfg.alpha} m_clauses={rows[0]['m_clauses']} "
          f"n_instances={n_instances}")
    for field in numeric_fields:
        values = np.array([r[field] for r in rows], dtype=float)
        print(f"  {field:20s} mean={values.mean():8.2f}  std={values.std():8.2f}  "
              f"min={values.min():6.0f}  max={values.max():6.0f}")
    return rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-vars", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=4.2)
    ap.add_argument("--n-instances", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--out-dir", required=True, help="e.g. $SCRATCH/spin_gnn/sat_stats")
    args = ap.parse_args()

    cfg = SATConfig(n_vars=args.n_vars, k=args.k, alpha=args.alpha)
    run(cfg, args.n_instances, args.seed_start, args.out_dir)
