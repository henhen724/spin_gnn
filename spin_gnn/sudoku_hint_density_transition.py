"""Does the transformer's solve rate collapse where DD's reasoning depth diverges?

This is the hint-density phase transition measured on the ensemble the models
are actually trained on, evaluated on saved checkpoints (no retraining).

Which transition. There are two candidate ensembles and only one has a
transition:

  random hints -- independent uniform hints. Measured in
    scripts/dd_random_hint_transition.py and found SUBCRITICAL at every
    density for d=5, q=4: mean recovery beyond the hints peaks at +0.12 and
    the cascade depth grows only 1.64 -> 3.36 while N grows 16x. There is no
    threshold here at any N, so there is nothing for a model to learn.

  removal path (the default, and what these checkpoints trained on) -- start
    fully hinted and greedily remove hints while DD still recovers
    everything. Every instance is resolvable by construction, so there is no
    *solvability* transition; rho instead controls the serial reasoning depth
    T, and T DIVERGES as rho approaches the floor rho_min that the greedy
    path cannot cross. From the large-N sweep at d=5, q=4: rho_min = 0.4756 /
    0.4768 / 0.4779 / 0.4778 for N = 996 / 4140 / 17196 / 49992, i.e.
    essentially N-independent, with T(0.50) = 21.2 / 28.5 / 40.6 / 45.8
    against T(0.75) ~ 6-11. That divergence is the transition.

The rise survives at N=24, the size these models train on (rho_min ~ 0.458,
T ~ 1 at rho=0.9 rising ~7x toward the floor), which is why this can be run
on existing checkpoints rather than requiring a scale-up to ~1700 tokens.

What would count as the transition showing up in the model: solve_rate falls
as realized_rho decreases, and the fall tracks dd_mean_T rather than rho
itself. The accompanying control is that DD solves every instance here
(dd_resolved_frac == 1), so any drop is the model failing to execute a longer
deduction chain, never the instance becoming unsolvable.

Two traps this module is built to avoid:

  * Below rho_min a requested rho cannot be delivered and silently returns
    rho_min, so requested-rho bins under the floor are the same point
    measured several times. Everything keys off `realized_rho`.
  * Generation length. A deeper trace is a longer token sequence, so a model
    cut off by max_new_tokens looks like a reasoning failure. The checkpoint's
    own max_len is used and `truncated_rate` is reported per bin so that
    confound is visible instead of silent.

Usage:
    python -m spin_gnn.sudoku_hint_density_transition \
        --checkpoint <dir> [<dir> ...] --out-csv <csv>
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .train_sudoku_transformer import eval_solve_by_rho, load_checkpoint


def transition_curve(ckpt_dir, rho_grid, n_per_bin, seed_start, device):
    model, vocab, config = load_checkpoint(ckpt_dir, device=device)
    model.eval()
    sudoku_cfg = SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        scratchpad=config.get("scratchpad", False),
        hint_mode=config.get("hint_mode", "removal"),
        # Analyzing a _shuf checkpoint without this feeds it the color-sorted
        # adjacency it was trained to ignore.
        shuffle_adj=bool(config.get("shuffle_adj", False)),
    )
    with torch.no_grad():
        records = eval_solve_by_rho(
            model, vocab, device, sudoku_cfg, rho_grid, n_per_bin,
            seed_start=seed_start, max_new_tokens=config["max_len"],
        )
    tag = os.path.basename(os.path.normpath(ckpt_dir))
    for r in records:
        r["checkpoint"] = tag
        r["n_nodes"] = config["n_nodes"]
        r["d_model"] = config["d_model"]
        r["num_layers"] = config["num_layers"]
        r["hint_mode"] = config.get("hint_mode", "removal")
        r["train_solve_rate"] = config.get("solve_metrics", {}).get("solve_rate")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, nargs="+", required=True)
    ap.add_argument("--n-per-bin", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=800_000_000,
                    help="disjoint from training (seed), probes (5e8/9e8) and the series pool (7e8)")
    ap.add_argument("--rho-min", type=float, default=0.44)
    ap.add_argument("--rho-max", type=float, default=1.0)
    ap.add_argument("--rho-step", type=float, default=0.02)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-csv", type=str, required=True)
    args = ap.parse_args()

    rho_grid = np.round(np.arange(args.rho_max, args.rho_min - 1e-9, -args.rho_step), 4)
    rows = []
    for ckpt in args.checkpoint:
        t0 = time.time()
        recs = transition_curve(ckpt, rho_grid, args.n_per_bin, args.seed_start, args.device)
        rows.extend(recs)
        print(f"\n{os.path.basename(os.path.normpath(ckpt))}  ({time.time() - t0:.0f}s)")
        print(f"{'req_rho':>8} {'real_rho':>9} {'dd_T':>6} {'solve':>7} {'wellfm':>7} {'trunc':>7}")
        for r in recs:
            print(f"{r['rho']:8.3f} {r['realized_rho']:9.4f} {r['dd_mean_T']:6.2f} "
                  f"{r['solve_rate']:7.3f} {r['well_formed_rate']:7.3f} "
                  f"{r.get('truncated_rate', float('nan')):7.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out_csv} ({len(rows)} rows)")
    with open(args.out_csv.replace(".csv", "_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
