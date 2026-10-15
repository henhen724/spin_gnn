"""Quick check: did the greedy sweep recover the fast/slow reasoning-depth
transition claimed in the slides (SUDOKU_DD.md)? For q=4, d in {5,7}:
does T/log(N) collapse across N above rho_c and fan out below, and where
does the greedy path's own rho_min sit relative to the claimed rho_c?
"""
import glob
import os

import numpy as np

RESULTS_DIR = os.path.join(os.environ["SCRATCH"], "spin_gnn", "sudoku_dd_results")
RHO_C = {5: 0.46, 7: 0.27}


def load_d(d):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"q4_d{d}_N*.npz")))
    by_n = {}
    for f in files:
        with np.load(f) as data:
            record = {"N": int(data["N"]), "rho_grid": data["rho_grid"],
                      "rho_min": float(data["rho_min"]), "T": data["T"].copy(), "m": data["m"].copy()}
        by_n.setdefault(record["N"], []).append(record)
    return by_n


for d in (5, 7):
    print(f"\n=== d={d} (slides' rho_c ~ {RHO_C[d]}) ===")
    by_n = load_d(d)
    rho_grid = None
    for n in sorted(by_n):
        seeds = by_n[n]
        rho_mins = [float(s["rho_min"]) for s in seeds]
        if rho_grid is None:
            rho_grid = seeds[0]["rho_grid"]
        print(f"  N={n:6d}  n_seeds={len(seeds):2d}  "
              f"rho_min: mean={np.mean(rho_mins):.4f} min={np.min(rho_mins):.4f} max={np.max(rho_mins):.4f}")

    # T/log(N) and <m> at a few representative rho values, averaged over
    # seeds that actually reached that rho (NaN above rho_min, by construction).
    print(f"  --- T/log(N) and <m> at select rho (NaN-aware mean over seeds) ---")
    probe_rhos = [0.9, 0.7, 0.6, 0.5, RHO_C[d], 0.2, 0.15]
    for rho in probe_rhos:
        idx = int(np.argmin(np.abs(rho_grid - rho)))
        line = f"  rho={rho_grid[idx]:.3f}:  "
        for n in sorted(by_n):
            seeds = by_n[n]
            t_vals = np.array([float(s["T"][idx]) for s in seeds])
            m_vals = np.array([float(s["m"][idx]) for s in seeds])
            valid = ~np.isnan(t_vals)
            if valid.sum() == 0:
                line += f"N={n}:(none) "
                continue
            t_over_logn = t_vals[valid] / np.log(n)
            line += f"N={n}:T/logN={np.mean(t_over_logn):.3f}(n={valid.sum()}) "
        print(line)
