"""Sweep greedy hint-removal paths (spin_gnn.sudoku_dd.removal_path) over
(degree d, graph size N, seed) and save each path sampled on a fixed rho
grid, plus full deduction-time snapshots at a few densities.

    python -m spin_gnn.sudoku_sweep --degrees 5 7 --sizes 996 2028 4140 \
        --seeds 20 --out-dir sudoku_dd_results

One .npz per (q, d, N, seed); pure CPU, parallelized over a process pool.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .sudoku_dd import on_grid, planted_regular_graph, removal_path

RHO_GRID = np.round(np.arange(1.0, 0.0, -0.005), 4)
SNAPSHOT_RHOS = (0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25)


def run_one(args):
    q, d, N, seed, out_dir = args
    path = os.path.join(out_dir, f"q{q}_d{d}_N{N}_s{seed}.npz")
    if os.path.exists(path):
        return path
    rng = np.random.default_rng([q, d, N, seed])
    adj, colors = planted_regular_graph(N, d, q, rng)
    p = removal_path(adj, colors, q, rng, snapshot_rhos=SNAPSHOT_RHOS)
    T, m = on_grid(p, RHO_GRID)
    snaps = {f"snap_{r}": p["snapshots"][r].astype(np.int32) for r in p["snapshots"]}
    np.savez_compressed(path, rho_grid=RHO_GRID, T=T, m=m, rho_min=p["rho_min"],
                        final_t=p["final_t"].astype(np.int32), q=q, d=d, N=N, seed=seed, **snaps)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--degrees", type=int, nargs="+", default=[5, 7])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[996, 2028, 4140, 8436, 17196, 35028, 49992])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out-dir", default="sudoku_dd_results")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = [(a.q, d, N, s, a.out_dir) for N in sorted(a.sizes, reverse=True)
            for d in a.degrees for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, path in enumerate(ex.map(run_one, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {path}", flush=True)


if __name__ == "__main__":
    main()
