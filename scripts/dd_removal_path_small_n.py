"""Removal-path deduction time T(rho) at transformer-reachable sizes.

The hint-density transition lives in the removal-path ensemble, not in
independent random hints: scripts/dd_random_hint_transition.py showed the
random-hint DD process is subcritical at every density and N-independent
(mean recovery ~12% beyond the hints at best, cascade depth 1.6 -> 3.4 from
N=24 to N=384), so no threshold exists there to learn.

In the removal-path ensemble the existing large-N sweep (sudoku_dd_results,
q=4 d=5) shows rho_min ~ 0.4777 essentially independent of N, with the
deduction time T diverging as rho -> rho_c+ (T ~ 6 -> 21 between rho=0.75
and 0.50 at N=996, 11 -> 46 at N=49992). That divergence is the transition.

This script measures the same curve at N the transformer can actually be
trained on, to find the smallest N whose T(rho) rise is big enough that a
collapse in model solve_rate could be attributed to it. Sequence length is
roughly 11*N tokens, so this sets the cost of the whole program.

Usage:
    python -m scripts.dd_removal_path_small_n --sizes 24 48 96 192 --out-csv <csv>
"""
import argparse
import csv
import os
import time

import numpy as np

from spin_gnn.sudoku_dd import on_grid, planted_regular_graph, removal_path


def measure(N, d, q, rho_grid, n_seeds, seed_start):
    Ts, ms, rho_mins = [], [], []
    for s in range(n_seeds):
        for attempt in range(20):
            rng = np.random.default_rng([q, d, N, seed_start + s, attempt])
            try:
                adj, colors = planted_regular_graph(N, d, q, rng)
                break
            except RuntimeError:
                continue
        else:
            continue
        p = removal_path(adj, colors, q, rng)
        T, m = on_grid(p, rho_grid)
        Ts.append(T)
        ms.append(m)
        rho_mins.append(p["rho_min"])
    T = np.nanmean(np.stack(Ts), axis=0)
    m = np.nanmean(np.stack(ms), axis=0)
    reach = np.mean(~np.isnan(np.stack(Ts)), axis=0)
    return [
        {
            "n_nodes": N, "degree": d, "n_colors": q, "rho": round(float(r), 4),
            "mean_T": float(T[i]) if not np.isnan(T[i]) else "",
            "mean_m": float(m[i]) if not np.isnan(m[i]) else "",
            "frac_paths_reaching_rho": float(reach[i]),
            "mean_rho_min": float(np.mean(rho_mins)),
            "n_seeds": len(Ts),
        }
        for i, r in enumerate(rho_grid)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[24, 48, 96, 192, 384, 996])
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--n-colors", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--out-csv", type=str, required=True)
    args = ap.parse_args()

    rho_grid = np.round(np.arange(1.0, 0.40, -0.01), 4)
    rows = []
    for N in args.sizes:
        t0 = time.time()
        r = measure(N, args.degree, args.n_colors, rho_grid, args.n_seeds, args.seed_start)
        rows.extend(r)
        # Compare at the deepest rho that MOST paths reach. The very last
        # reachable bin averages over the few unusually deep paths only, which
        # overstates the rise.
        solid = [x for x in r if x["mean_T"] != "" and x["frac_paths_reaching_rho"] >= 0.8]
        t_hi = next((x["mean_T"] for x in solid if abs(x["rho"] - 0.90) < 1e-6), float("nan"))
        lo = solid[-1] if solid else {"rho": float("nan"), "mean_T": float("nan")}
        print(f"N={N}: {time.time() - t0:.1f}s  rho_min={r[0]['mean_rho_min']:.4f}  "
              f"T(0.90)={t_hi:.1f} -> T({lo['rho']:.2f})={lo['mean_T']:.1f}  "
              f"rise={lo['mean_T'] / t_hi:.1f}x")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
