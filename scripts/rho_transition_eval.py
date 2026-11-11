"""Solve rate across the fast/slow transition, for checkpoints that exist.

`sudoku_hparam_sweep` only probes the easy end (rho = 1 down to (N-4)/N) and
`sudoku_depth_eval` bins by reasoning depth at one or two fixed densities.
Neither traces solve rate along rho itself, which is what a transition curve
is. `train_sudoku_transformer.eval_solve_by_rho` already computes exactly
that -- pinned-density bins plus `realized_rho` and `dd_mean_T` per bin --
but its only caller trains its own model first. This is a thin CLI over it
that LOADS checkpoints instead, so one (model size x training set) grid can
be re-read on any rho grid without retraining, and so the eval config is
taken from each checkpoint rather than re-specified on the command line.

Why this exists now: until 2026-11-02 the question was unanswerable below
rho ~ 0.47. Nodes missing a colour in their neighbourhood can never be
deduced, so they are permanent hints and greedy removal stalls on top of
them; `eval_solve_by_rho` silently returns that floor for every requested
rho beneath it, which is why its docstring warns to plot `realized_rho`
rather than the requested value. With SudokuReasoningConfig.fall the floor
is gone (rho_min 0.469 -> 0.332 at N=48) and the grid can actually reach
the slow phase.

The two columns that make rows comparable across checkpoints:
  realized_rho -- the density actually achieved. Still the honest x-axis:
    even under `fall` a greedy path has its own rho_min, so a requested rho
    below it collapses onto the floor and duplicates a point.
  dd_mean_T    -- mean DD rounds on the same instances, i.e. how much serial
    reasoning that bin actually demands. This is the x-axis to use when
    comparing ACROSS n_nodes, where equal rho means different depth.

Usage:
    python3 -m scripts.rho_transition_eval \
        --checkpoint $SCRATCH/spin_gnn/<run>/L12/<tag> ... \
        --rho-grid 0.95 0.85 0.75 0.65 0.55 0.45 0.40 0.35 0.30 \
        --n-per-bin 300 --out-csv $SCRATCH/spin_gnn/<run>/rho_curve.csv
"""
import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sat_transformer2 import is_tied, model_depth
from spin_gnn.train_sudoku_transformer import eval_solve_by_rho, load_checkpoint


def cfg_from_checkpoint(config):
    """The instance distribution the checkpoint was TRAINED on.

    Every field here is read from config.json rather than the command line:
    an eval that silently differs from training in any instance key is
    scoring the model on a different task. `rho`/`rho_max` are overridden
    per bin by eval_solve_by_rho (it pins both to the bin's density), so
    they are carried only for completeness.
    """
    return SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        rho=config["rho"], rho_max=config["rho_max"],
        hint_mode=config.get("hint_mode", "removal"),
        scratchpad=config.get("scratchpad", False),
        shuffle_adj=config.get("shuffle_adj", False),
        shuffle_events=config.get("shuffle_events", False),
        fall=config.get("fall", False),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--rho-grid", type=float, nargs="+",
                    default=[0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.40, 0.35, 0.30])
    ap.add_argument("--n-per-bin", type=int, default=300,
                    help="instances per rho bin. The Wilson half-width at p~0.9 is "
                         "0.034 at 300 and 0.019 at 1000; pick from the effect you "
                         "need to resolve, remembering finding 41's 0.1627 seed spread.")
    ap.add_argument("--seed-start", type=int, default=60_000_000_000,
                    help="fixed by default so every checkpoint sees the SAME instances "
                         "at each rho and curves can be compared point by point")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--n-iters", type=int, default=0,
                    help="TIED CHECKPOINTS ONLY: apply the one shared block this "
                         "many times instead of the trained count, on FIXED weights. "
                         "A dial no untied cell has -- `encoder.layers` holds the "
                         "whole depth there, so changing it would change the model. "
                         "0 leaves the trained count alone.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  rho grid={args.rho_grid}  n_per_bin={args.n_per_bin}")

    rows = []
    for ckpt in args.checkpoint:
        model, vocab, config = load_checkpoint(ckpt, device=device)
        sudoku_cfg = cfg_from_checkpoint(config)
        name = os.path.basename(os.path.normpath(ckpt))
        trained_iters = model_depth(model)
        if args.n_iters:
            # Refuse on untied rather than silently scoring the trained depth:
            # setting n_iters on a model whose forward() iterates encoder.layers
            # does nothing at all, and a column saying T=24 over a depth-16 pass
            # is worse than an error.
            if not is_tied(model):
                raise SystemExit(
                    "--n-iters is meaningless for %s (arch_mode=%s): an untied "
                    "stack's depth IS its weights." % (name, config.get("arch_mode", "full")))
            model.n_iters = args.n_iters
            name = "%s_T%02d" % (name, args.n_iters)
            print(f"  n_iters {trained_iters} -> {args.n_iters} on fixed weights")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n=== {name} ===")
        print(f"  L={config['num_layers']} d_model={config['d_model']} "
              f"nhead={config.get('nhead')} arch={config.get('arch_mode', 'full')} "
              f"ff={config.get('dim_feedforward')} params={n_params} "
              f"n_train={config.get('n_train_instances')} seed={config.get('seed')} "
              f"fall={sudoku_cfg.fall} shuffle_adj={sudoku_cfg.shuffle_adj}")
        records = eval_solve_by_rho(
            model, vocab, device, sudoku_cfg, args.rho_grid,
            args.n_per_bin, args.seed_start, config["max_len"],
        )
        print(f"  {'rho_req':>8} {'real_rho':>9} {'dd_T':>6} {'solve':>7} {'outcome':>8} {'trunc':>7}")
        for r in records:
            print(f"  {r['rho']:8.3f} {r['realized_rho']:9.4f} {r['dd_mean_T']:6.2f} "
                  f"{r['solve_rate']:7.4f} {r['outcome_accuracy']:8.4f} "
                  f"{r.get('truncated_rate', float('nan')):7.4f}")
            rows.append({
                "checkpoint": name, "path": ckpt,
                "num_layers": config["num_layers"], "d_model": config["d_model"],
                "nhead": config.get("nhead"),
                "dim_feedforward": config.get("dim_feedforward"),
                # The minimal-architecture grid varies these two, and without
                # them in the row a curve cannot be told apart from the full
                # cell at the same (L, d_model, ff) -- the same argument that
                # put fall and shuffle_adj here. New columns, appended by
                # DictWriter from the first row, so every existing reader
                # (DictReader everywhere) is unaffected.
                "arch_mode": config.get("arch_mode", "full"),
                # Applications of the block at EVAL, which for a tied cell is a
                # dial and for an untied one is num_layers by construction. A
                # row is otherwise indistinguishable from the trained-T row of
                # the same checkpoint.
                "n_iters_eval": model_depth(model), "n_iters_trained": trained_iters,
                "n_params": n_params,
                "n_trainable_params": config.get("n_trainable_params", n_params),
                "n_train_instances": config.get("n_train_instances"),
                "epochs": config.get("epochs"), "seed": config.get("seed", 0),
                "n_nodes": config["n_nodes"], "degree": config["degree"],
                "n_colors": config["n_colors"],
                "fall": sudoku_cfg.fall, "shuffle_adj": sudoku_cfg.shuffle_adj,
                "val_nll": config.get("nll"), "n_per_bin": args.n_per_bin,
                **r,
            })
        # A curve whose low-rho tail has collapsed onto one density is not a
        # curve there, so say so rather than letting a flat tail be read as a
        # result (eval_solve_by_rho's own warning, made loud).
        floor = min(r["realized_rho"] for r in records)
        pinned = [r["rho"] for r in records if abs(r["realized_rho"] - floor) < 1e-9]
        if len(pinned) > 1:
            print(f"  NOTE: requested rho {pinned} all realized {floor:.4f} -- that is this "
                  f"ensemble's rho_min, one point measured {len(pinned)} times, not a tail.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
