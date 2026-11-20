"""CPU-side sizing probe for the large-N fall ensemble: what does a training
instance COST at N > 96, and where does its hint-density grid end?

WHY THIS EXISTS. `sudoku_fastslow_2026-11-12/fastslow.json` measured the
fast/slow exponent out to N=49992 but its `tokens` block is N=48 ONLY, so the
sequence length -- which sets `max_len`, which sets both the positional table
(a term in the parameter count) and the O(max_len^2) attention cost -- has
never been measured above N=96. Finding 44's N=96 sbatch pre-registered
"the next move is N=192"; this probe prices that move before any GPU time is
spent on it.

WHAT IT MEASURES, per N, with fall=True (the ONLY correct ensemble here --
`fall` is an instance key, not a hyperparameter, and the unconstrained stub
split leaves 38.27% of nodes permanently undeducible at q=4 d=5, flooring
rho_min at 0.477, ABOVE rho_c ~ 0.46, i.e. entirely in DD's fast phase):

  rho_min   the sparsest hint set greedy removal reaches (grid floor)
  T         DD rounds at the floor and across a density grid
  tokens    encoded length, mean/max, against the analytic 8N + T + 1
  secs      wall-clock per instance, which is the DATA-GENERATION cost

THE ANALYTIC LENGTH. A non-scratchpad sequence is
  1 + 2*rho*N          BEGIN_HINTS and (NODE COLOR) per hint
  + 1 + N*(1+degree)   BEGIN_GRAPH and one 6-token row per node
  + 1 + 2*n + T        BEGIN_REASONING, (NODE COLOR) per deduced node,
                       one BEGIN_ROUND per round
  + 1                  SOLVED
with n = N*(1-rho) blanks, so the rho-dependence CANCELS:
  2*rho*N + 2*N*(1-rho) = 2N
leaving  tokens = N*(degree+3) + T + 4  =  8N + T + 4  at degree=5.
Depth therefore enters the length budget ONLY through T -- which is why
finding 43 could turn the fall fix on and watch mean length move 393 -> 394
at N=48 while mean depth went 7.9 -> 13.8. Raising N is the expensive axis;
raising depth at fixed N is nearly free.

Run: python3 scripts/largeN_sizing_probe.py [--out PATH]
Needs numpy/numba only (no torch, no GPU).
"""
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/users/hshunt/spin_gnn")

from spin_gnn.sudoku_dd import planted_regular_graph, removal_path, on_grid
from spin_gnn.sudoku_tokens import Vocab, encode_instance
from spin_gnn.sudoku_trace import solve_with_dd_trace

# N must satisfy planted_regular_graph's constraints at q=4 d=5:
# q | N and q(q-1) | d*N  =>  12 | N.
SIZES = (48, 96, 144, 192, 252, 384)
DEGREE = 5
N_COLORS = 4
# The density grid each fall scan has used, extended down past every measured
# rho_min (N=48 ends 0.33, N=96 ends 0.32, and this probe reports where the
# larger sizes actually stop).
GRID = (0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.45, 0.40, 0.35, 0.33, 0.31, 0.30)
# The trainer's auto headroom above the longest training sequence
# (train_sudoku_transformer.resolve_max_len_margin: -1 => 2*n_nodes + 2).
MARGIN = lambda n: 2 * n + 2


def probe_size(n_nodes, n_instances, seed_start, fall=True):
    rows = []
    t0 = time.time()
    for i in range(n_instances):
        rng = np.random.default_rng(seed_start + i)
        adj, colors = planted_regular_graph(
            n_nodes, DEGREE, N_COLORS, rng, fall=fall)
        path = removal_path(adj, colors, N_COLORS, rng, snapshot_rhos=GRID)
        T_grid, _ = on_grid(path, np.asarray(GRID, dtype=float))
        # Encode at the path's own floor -- the deepest, and so longest, point.
        n_hints = int(round(path["rho_min"] * n_nodes))
        rows.append({
            "rho_min": float(path["rho_min"]),
            # "T" is the DD depth along the whole removal path; its last
            # entry is the depth at the path's own floor, the deepest point.
            "T_end": float(np.asarray(path["T"])[-1]),
            "T_grid": [float(x) for x in np.atleast_1d(T_grid)],
            "n_hints_min": n_hints,
        })
    secs = time.time() - t0

    # One fully-encoded instance at the floor, to check the analytic length.
    rng = np.random.default_rng(seed_start + 10_000)
    adj, colors = planted_regular_graph(
        n_nodes, DEGREE, N_COLORS, rng, fall=fall)
    path = removal_path(adj, colors, N_COLORS, rng)
    # Hints at the floor, built exactly as train_sudoku_transformer.
    # build_instances does it: the nodes never removed, i.e. removal time 0.
    t_final = path["final_t"]
    hints = {int(v): int(colors[v]) for v in np.flatnonzero(t_final == 0)}
    trace = solve_with_dd_trace(adj, colors, hints, N_COLORS)
    assert trace["resolved"], "removal_path floor must stay fully resolvable"
    vocab = Vocab(n_nodes, N_COLORS, DEGREE, scratchpad=False)
    tokens = encode_instance(adj, trace, vocab)

    rho_mins = np.array([r["rho_min"] for r in rows])
    T_ends = np.array([r["T_end"] for r in rows])
    T_stack = np.array([r["T_grid"] for r in rows], dtype=float)
    analytic = n_nodes * (DEGREE + 3) + float(np.nanmax(T_ends)) + 4

    return {
        "n_nodes": n_nodes,
        "fall": fall,
        "n_instances": n_instances,
        "secs_total": round(secs, 2),
        "secs_per_instance": round(secs / max(n_instances, 1), 4),
        "rho_min": {"mean": float(rho_mins.mean()), "sd": float(rho_mins.std()),
                    "min": float(rho_mins.min()), "max": float(rho_mins.max())},
        "T_end": {"mean": float(T_ends.mean()), "sd": float(T_ends.std()),
                  "min": float(T_ends.min()), "max": float(T_ends.max())},
        "T_grid_median": [None if np.all(np.isnan(c)) else float(np.nanmedian(c))
                          for c in T_stack.T],
        "vocab_size": int(vocab.size),
        "encoded_len_at_floor": int(tokens.shape[0]),
        "analytic_len_at_Tmax": int(analytic),
        "max_len_with_margin": int(analytic + MARGIN(n_nodes)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-instances", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=90_000_000)
    args = ap.parse_args()

    doc = {"degree": DEGREE, "q": N_COLORS, "grid": list(GRID),
           "sizes": [], "note": "fall=True only; see module docstring"}
    print(f"{'N':>6} {'rho_min':>16} {'T_end':>14} {'enc@floor':>10} "
          f"{'max_len':>9} {'s/inst':>8}")
    for n in SIZES:
        # Fewer instances at the big sizes: removal_path is O(N * T) per path.
        k = args.n_instances if n <= 192 else max(12, args.n_instances // 3)
        row = probe_size(n, k, args.seed_start)
        doc["sizes"].append(row)
        print(f"{n:>6} {row['rho_min']['mean']:>7.3f}+-{row['rho_min']['sd']:<7.3f} "
              f"{row['T_end']['mean']:>6.1f}+-{row['T_end']['sd']:<6.1f} "
              f"{row['encoded_len_at_floor']:>10} "
              f"{row['max_len_with_margin']:>9} {row['secs_per_instance']:>8.3f}")

    print("\nmedian T on the density grid (None = below every path's rho_min)")
    print("     N  " + "".join(f"{g:>7.2f}" for g in GRID))
    for row in doc["sizes"]:
        cells = "".join("      -" if v is None else f"{v:>7.1f}"
                        for v in row["T_grid_median"])
        print(f"{row['n_nodes']:>6}  {cells}")

    # Relative training cost, normalised to N=96, the last size actually
    # trained. Attention is O(max_len^2 * d), the position-wise parts are
    # O(max_len * d^2), so quote BOTH bounds rather than a single guess.
    base = next(r for r in doc["sizes"] if r["n_nodes"] == 96)
    b = base["max_len_with_margin"]
    print(f"\nper-epoch cost relative to N=96 (max_len {b}), same instance count")
    print(f"{'N':>6} {'max_len':>9} {'linear':>8} {'quadratic':>10}")
    for row in doc["sizes"]:
        r = row["max_len_with_margin"] / b
        row["cost_vs_n96_linear"] = round(r, 3)
        row["cost_vs_n96_quadratic"] = round(r * r, 3)
        print(f"{row['n_nodes']:>6} {row['max_len_with_margin']:>9} "
              f"{r:>7.2f}x {r * r:>9.2f}x")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
