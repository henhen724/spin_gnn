"""Ground-truth DD recovery curve under *independent random* hint sampling.

This is the reference curve the transformer work is aiming at. It is a
different measurement from spin_gnn.sudoku_sweep / sudoku_transition_sweep,
which both build instances via `removal_path` (greedy hint removal that only
accepts a removal if DD still recovers the whole coloring). Those are
resolvable by construction at every density, so they have no solvability
transition at all -- rho there sets reasoning depth, not solvability.

Here hints are n_hints = round(rho * N) nodes drawn uniformly (sudoku_trace.
sample_hints), so DD genuinely fails below some rho_c. Two observables per
(N, rho):

  resolved_frac -- fraction of instances DD solves completely. The sharp
                   (in N) order parameter; this is what "phase transition"
                   refers to.
  mean_frac_recovered -- mean fraction of nodes DD colors, resolved or not.
                   A smoother observable that still separates the phases and
                   is far less finite-size noisy at small N.

Usage:
    python -m scripts.dd_random_hint_transition --sizes 24 48 96 \
        --n-instances 200 --out-csv $SCRATCH/....csv
"""
import argparse
import csv
import os
import time

import numpy as np

from spin_gnn.sudoku_dd import planted_regular_graph
from spin_gnn.sudoku_trace import sample_hints, solve_with_dd_trace


def measure(N, d, q, rho_grid, n_instances, seed_start):
    """One row per rho. Each instance gets a fresh planted graph, so the
    curve averages over graph disorder as well as hint placement."""
    rows = []
    for rho in rho_grid:
        n_hints = int(round(rho * N))
        n_resolved = 0
        frac_recovered = []
        n_rounds = []
        n_built = 0
        for i in range(n_instances):
            # planted_regular_graph occasionally fails to find a simple pairing
            # (more often at small N) -- retry with a perturbed seed rather
            # than letting one bad draw bias the bin.
            for attempt in range(20):
                rng = np.random.default_rng([seed_start, N, int(rho * 10_000), i, attempt])
                try:
                    adj, colors = planted_regular_graph(N, d, q, rng)
                    break
                except RuntimeError:
                    continue
            else:
                continue
            n_built += 1
            hints = sample_hints(N, max(0, min(N, n_hints)), colors, rng)
            trace = solve_with_dd_trace(adj, colors, hints, q)
            n_resolved += bool(trace["resolved"])
            n_unresolved = len(trace["unresolved_nodes"])
            frac_recovered.append((N - n_unresolved) / N)
            n_rounds.append(len(trace["rounds"]))
        rows.append({
            "n_nodes": N, "degree": d, "n_colors": q, "rho": round(float(rho), 4),
            "n_hints": n_hints, "n_instances": n_built,
            "resolved_frac": n_resolved / max(1, n_built),
            "mean_frac_recovered": float(np.mean(frac_recovered)) if frac_recovered else float("nan"),
            "mean_n_rounds": float(np.mean(n_rounds)) if n_rounds else float("nan"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[24, 48, 96])
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--n-colors", type=int, default=4)
    ap.add_argument("--rho-min", type=float, default=0.0)
    ap.add_argument("--rho-max", type=float, default=1.0)
    ap.add_argument("--n-rho", type=int, default=21)
    ap.add_argument("--n-instances", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=777)
    ap.add_argument("--out-csv", type=str, required=True)
    args = ap.parse_args()

    rho_grid = np.linspace(args.rho_min, args.rho_max, args.n_rho)
    all_rows = []
    for N in args.sizes:
        t0 = time.time()
        rows = measure(N, args.degree, args.n_colors, rho_grid, args.n_instances, args.seed_start)
        all_rows.extend(rows)
        # Report the steepest-slope rho as a crude rho_c estimate per size.
        r = np.array([row["resolved_frac"] for row in rows])
        rc = rho_grid[int(np.argmax(np.diff(r, prepend=r[0])))]
        print(f"N={N}: {time.time() - t0:.1f}s, steepest-rise rho~{rc:.3f}, "
              f"resolved_frac {r.min():.2f}->{r.max():.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {args.out_csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
