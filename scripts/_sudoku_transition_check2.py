"""Finer follow-up: for d=7, scan rho densely near the claimed rho_c=0.27
and fit log(T) vs log(N) at each rho to estimate the growth exponent
gamma(rho) -- gamma ~ 0 is the fast phase (T ~ log N), gamma > 0 growing
is the slow phase (T ~ N^gamma)."""
import glob
import os

import numpy as np

RESULTS_DIR = os.path.join(os.environ["SCRATCH"], "spin_gnn", "sudoku_dd_results")


def load_d(d):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"q4_d{d}_N*.npz")))
    by_n = {}
    for f in files:
        with np.load(f) as data:
            record = {"N": int(data["N"]), "rho_grid": data["rho_grid"],
                      "rho_min": float(data["rho_min"]), "T": data["T"].copy()}
        by_n.setdefault(record["N"], []).append(record)
    return by_n


for d in (5, 7):
    print(f"\n=== d={d}: gamma(rho) = slope of log(T) vs log(N) ===")
    by_n = load_d(d)
    ns = sorted(by_n)
    rho_grid = by_n[ns[0]][0]["rho_grid"]

    probe = np.arange(0.5, 0.15, -0.02)
    for rho in probe:
        idx = int(np.argmin(np.abs(rho_grid - rho)))
        log_ns, log_ts = [], []
        n_at_rho = []
        for n in ns:
            t_vals = np.array([float(s["T"][idx]) for s in by_n[n]])
            valid = ~np.isnan(t_vals) & (t_vals > 0)
            if valid.sum() < 5:  # need a reasonable number of seeds to trust the mean
                continue
            log_ns.append(np.log(n))
            log_ts.append(np.log(np.mean(t_vals[valid])))
            n_at_rho.append(valid.sum())
        if len(log_ns) < 3:
            print(f"  rho={rho_grid[idx]:.3f}: too few N points with enough seeds ({len(log_ns)})")
            continue
        slope, intercept = np.polyfit(log_ns, log_ts, 1)
        print(f"  rho={rho_grid[idx]:.3f}: gamma={slope:.3f}  (N points used: {len(log_ns)}, "
              f"seed counts: {n_at_rho})")
