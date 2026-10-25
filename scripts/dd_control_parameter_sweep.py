"""Can (degree, n_colors) put DD's critical point at a trainable N?

WHAT IS ALREADY KNOWN, and what it leaves open. The removal-path ensemble has
a finite-density critical point: the minimal reachable hint density rho_min is
~0.4777 and essentially N-independent (0.442 at N=24 -> 0.477 by N~96, flat to
N=996), and the deduction depth T diverges as rho -> rho_min+. That divergence
is the transition. Its LOCATION is therefore already asymptotic at N=24, but
its STRENGTH is not: max mean T is 7.2 at N=24, 14.9 at N=96, 28.1 at N=996,
46 at N=49992. A transformer trained at N=24 is looking at a critical point
whose order parameter has a range of about seven integers.

Going up in N is the obvious fix and it is brutally expensive: the token
sequence is ~11*N long, so with quadratic attention the cost per instance goes
up ~8x from N=24 to N=48 and ~59x to N=96. Before paying that, this sweep asks
whether a different corner of (degree, n_colors) gives a stronger divergence at
an N we can already train on. All existing work is at (d=5, q=4).

The mechanism makes a specific prediction. DD colors a node only when its
already-colored neighbours exclude q-1 of the q colors, so propagation
strength should be governed by the ratio d/(q-1) -- how many neighbours there
are per color that has to be excluded. Weaker propagation (smaller ratio)
should mean DD needs more hints and works harder near the floor.

PRE-REGISTERED PREDICTIONS.
 (a) rho_min increases monotonically as d/(q-1) decreases. Concretely
     rho_min(d=3, q=4) [ratio 1.00] > 0.477 > rho_min(d=10, q=4) [3.33].
 (b) At N=96 at least one cell with d/(q-1) <= 1.2 reaches max mean T >= 22,
     i.e. >= 1.5x the 14.9 that (5,4) gives at the same N -- enough that N=96,
     or even N=48, would show a divergence as strong as (5,4) needs N~1000 for.
 (c) Falsifier for the whole idea: rho_min and max T are flat in (d,q) at
     fixed N (max T within +-20% of (5,4) everywhere). That would say the
     critical structure is universal, there is no cheap corner, and the
     program genuinely needs large N.
 (d) Finite-size scaling, the test of whether rho_min is a critical point at
     all rather than merely where greedy removal stops: for (5, 4), T at a
     fixed rho close to the floor should grow as a power of N while T at
     rho=0.90 stays roughly N-independent. Existing data is consistent --
     T(rho=0.5) goes 5.1 -> 21.3 over N=24 -> 996, about N^0.38 -- but T at
     high density was never isolated, so the columns T_at_rho90/70/55 test it
     directly. If T at rho=0.90 grows with N just as fast, nothing is
     critical and the "transition" is just longer chains in bigger graphs.
 (e) Control, random-hint branch: in EVERY cell the number of *unhinted* nodes
     at resolved_frac = 0.5 stays O(1) rather than growing proportionally to
     N, confirming that the random-hint ensemble has no finite-density
     solvability transition for any (d, q) and not just for (5, 4).

Usage:
    python -m scripts.dd_control_parameter_sweep --out-dir $SCRATCH/...
"""
import argparse
import csv
import itertools
import os
import time

import numpy as np

from scripts.dd_random_hint_transition import measure as measure_random
from scripts.dd_removal_path_small_n import measure as measure_removal


def cells(degrees, colors):
    """(d, q) pairs worth measuring.

    DD can never force a node with fewer neighbours than colors to exclude, so
    d >= q-1 is a hard floor; q=2 is bipartite and degenerate.
    """
    return [(d, q) for q, d in itertools.product(colors, degrees) if q >= 3 and d >= q - 1]


def at_rho(rows, target):
    """mean_T at the grid point nearest `target`, or nan if no path reached it.

    Reported at fixed rho on purpose. Whether rho_min is a genuine critical
    point or just where greedy removal gets stuck is decided by how T at FIXED
    density scales with N: growing like a power of N close to the floor while
    staying flat far from it is criticality; growing everywhere is not.
    """
    cands = [r for r in rows if r["mean_T"] != "" and r["frac_paths_reaching_rho"] >= 0.8]
    if not cands:
        return float("nan")
    best = min(cands, key=lambda r: abs(float(r["rho"]) - target))
    if abs(float(best["rho"]) - target) > 0.03:
        return float("nan")
    return float(best["mean_T"])


def crossing(xs, ys, level):
    """First x where ys rises through `level`, linearly interpolated."""
    for i in range(1, len(ys)):
        if ys[i - 1] < level <= ys[i]:
            if ys[i] == ys[i - 1]:
                return xs[i]
            return xs[i - 1] + (level - ys[i - 1]) * (xs[i] - xs[i - 1]) / (ys[i] - ys[i - 1])
    return float("nan")


def random_is_affordable(N, d, q, max_fill=0.4):
    """Whether `planted_regular_graph` will build this cell in reasonable time.

    A planted q-colouring puts N/q nodes in each class, so a node's d
    neighbours must come from the (q-1)N/q nodes of other colours. As d
    approaches that ceiling the rejection sampler retries for a long time:
    measured on run 49431733, d=6 q=3 N=24 (fill 0.375) took 4 s, d=8 q=3 N=24
    (fill 0.500) took 49 s, and d=10 q=3 N=24 (fill 0.625) did not finish in
    15 minutes. The removal branch does not use this constructor and is
    unaffected.

    Skipping is honest here rather than lossy: the cells this excludes are the
    ones whose 50% crossing came back `nan` anyway, because at that density
    almost every node is hinted and the curve never crosses.
    """
    return d <= max_fill * (q - 1) * N / q


def random_is_constructible(N, d, q):
    """Whether `planted_regular_graph` accepts this (N, d, q) at all.

    It balances colour classes exactly and distributes each node's d edges
    evenly over the q-1 other classes, so it requires q | N and q(q-1) | d*N
    (sudoku_dd.py:65). Run 49437796 died on exactly this at (N=24, d=4, q=5) --
    none of the random sizes 24/96/384 is divisible by 5 -- taking the whole
    job down under `set -e` nine cells short of the full grid.

    This is a hard constraint of the constructor, not a cost heuristic like
    `random_is_affordable`, so it is checked separately and reported
    separately: the affected cells are not expensive, they are impossible at
    these sizes, and the removal branch they share a cell with is unaffected.
    """
    return N % q == 0 and (d * N) % (q * (q - 1)) == 0


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_all(out_dir, removal_rows, random_rows, summary):
    """Rewrite all three CSVs. Called after every cell, not once at the end.

    Run 49431733 wrote nothing until the final line and so would have lost
    every completed cell when it hit its wall clock. Rewriting three small
    files per cell costs milliseconds against minutes of measurement.
    """
    write_csv(os.path.join(out_dir, "removal_path_T_vs_rho.csv"), removal_rows)
    write_csv(os.path.join(out_dir, "random_hint_resolved.csv"), random_rows)
    write_csv(os.path.join(out_dir, "summary.csv"), summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", type=int, nargs="+", default=[3, 4, 5, 6, 8, 10])
    ap.add_argument("--colors", type=int, nargs="+", default=[3, 4, 5, 6])
    # A removal path costs ~1 ms at N=24 and ~2.4 ms at N=192 (measured, warm
    # JIT), so the removal branch is nearly free and can span the sizes needed
    # for finite-size scaling. The random branch is ~100x dearer per point --
    # planted_regular_graph retries a lot at small N -- so it gets 3 sizes.
    ap.add_argument("--sizes-removal", type=int, nargs="+",
                    default=[24, 48, 96, 192, 384, 996])
    ap.add_argument("--sizes-random", type=int, nargs="+", default=[24, 96, 384])
    ap.add_argument("--n-seeds-removal", type=int, default=400)
    ap.add_argument("--n-seeds-removal-large", type=int, default=200)
    ap.add_argument("--large-n", type=int, default=192)
    ap.add_argument("--n-instances-random", type=int, default=300)
    ap.add_argument("--seed-start", type=int, default=4242)
    ap.add_argument("--max-fill", type=float, default=0.4,
                    help="skip random-ensemble cells whose planted-graph fill "
                         "d / ((q-1)N/q) exceeds this; see random_is_affordable")
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    grid = cells(args.degrees, args.colors)
    print(f"{len(grid)} (degree, n_colors) cells: {grid}", flush=True)

    # Fine and wide on the removal side: the cost there is one removal path
    # per seed, and the grid only interpolates that path, so extra points are
    # free. It has to start well below 0.477 because strong-propagation cells
    # (d=10, q=3) should push their floor far lower.
    removal_rho = np.round(np.arange(0.14, 1.001, 0.01), 4)
    # Each random-hint grid point costs a fresh set of instances, so points go
    # where the threshold is: coarse below 0.9, 0.01 steps above it, since the
    # crossing at (5,4) N=384 already sits at 0.987.
    random_rho = np.round(np.unique(np.concatenate(
        [np.arange(0.50, 0.90, 0.05), np.arange(0.90, 1.001, 0.01)])), 4)

    removal_rows, random_rows, summary = [], [], []

    for d, q in grid:
        ratio = d / (q - 1)
        for N in args.sizes_removal:
            n_seeds = args.n_seeds_removal_large if N >= args.large_n else args.n_seeds_removal
            t0 = time.time()
            rows = measure_removal(N, d, q, removal_rho, n_seeds, args.seed_start)
            removal_rows.extend(rows)
            Ts = [float(r["mean_T"]) for r in rows if r["mean_T"] != ""]
            rho_min = float(rows[0]["mean_rho_min"])
            # T measured a little above the floor: at the floor itself only a
            # few paths reach, so mean_T there is the noisiest point on the
            # curve and a bad summary of the divergence.
            near = [float(r["mean_T"]) for r in rows
                    if r["mean_T"] != "" and rho_min <= float(r["rho"]) <= rho_min + 0.06]
            summary.append({
                "ensemble": "removal", "n_nodes": N, "degree": d, "n_colors": q,
                "ratio_d_over_qm1": round(ratio, 4), "n_seeds": n_seeds,
                "rho_min": rho_min,
                "max_mean_T": max(Ts) if Ts else float("nan"),
                "mean_T_near_floor": float(np.mean(near)) if near else float("nan"),
                "T_at_rho90": at_rho(rows, 0.90),
                "T_at_rho70": at_rho(rows, 0.70),
                "T_at_rho55": at_rho(rows, 0.55),
                "rho50": "", "width_10_90": "", "unhinted_at_rho50": "",
            })
            print(f"  removal d={d} q={q} (ratio {ratio:.2f}) N={N}: "
                  f"rho_min={rho_min:.4f} max_T={max(Ts) if Ts else float('nan'):.2f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)

        # Write before entering the random branch as well as after the cell:
        # 49437796 lost this cell's six completed removal rows because they
        # were still only in memory when the random branch raised.
        write_all(args.out_dir, removal_rows, random_rows, summary)

        for N in args.sizes_random:
            if not random_is_constructible(N, d, q):
                print(f"  random  d={d} q={q} N={N}: SKIPPED, planted_regular_graph "
                      f"needs q|N and q(q-1)|d*N", flush=True)
                continue
            if not random_is_affordable(N, d, q, args.max_fill):
                print(f"  random  d={d} q={q} N={N}: SKIPPED, planted fill "
                      f"{d / ((q - 1) * N / q):.2f} > {args.max_fill}", flush=True)
                continue
            t0 = time.time()
            rows = measure_random(N, d, q, random_rho, args.n_instances_random, args.seed_start)
            random_rows.extend(rows)
            xs = [float(r["rho"]) for r in rows]
            ys = [float(r["resolved_frac"]) for r in rows]
            lo, mid, hi = (crossing(xs, ys, f) for f in (0.1, 0.5, 0.9))
            summary.append({
                "ensemble": "random", "n_nodes": N, "degree": d, "n_colors": q,
                "ratio_d_over_qm1": round(ratio, 4), "n_seeds": args.n_instances_random,
                "rho_min": "", "max_mean_T": max(float(r["mean_n_rounds"]) for r in rows),
                "mean_T_near_floor": "",
                "T_at_rho90": "", "T_at_rho70": "", "T_at_rho55": "",
                "rho50": mid, "width_10_90": hi - lo,
                # The decisive number for prediction (d): O(1) means no
                # finite-density transition, ~N means there is one.
                "unhinted_at_rho50": N * (1 - mid) if mid == mid else float("nan"),
            })
            print(f"  random  d={d} q={q} N={N}: rho50={mid:.3f} "
                  f"width={hi - lo:.3f} unhinted@50%={N * (1 - mid):.1f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)

        write_all(args.out_dir, removal_rows, random_rows, summary)

    write_all(args.out_dir, removal_rows, random_rows, summary)
    print(f"done: {len(removal_rows)} removal rows, {len(random_rows)} random rows, "
          f"{len(summary)} summary rows", flush=True)


if __name__ == "__main__":
    main()
