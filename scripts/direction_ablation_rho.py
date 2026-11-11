"""Does the model USE the DD predicate it represents? Ablate it and re-measure rho_50.

Runs, for one checkpoint, the full pre-registered condition set from
spin_gnn.sudoku_direction_ablate: fit probes at every layer on clean
activations, build a count-matched k-dimensional subspace per condition, then
re-run the SAME 14-point x 400-instance rho eval the N=96 fall scan used
(--seed-start 60000000000, so every condition and the original scan see
IDENTICAL instances) with that subspace projected out of every block's output.

Two things come out of it per condition: a read-back (did the ablation
actually remove what it was aimed at, and did it leave the other targets
alone) and a rho_50.

CHOOSING k. k is the dimensionality removed, identical across conditions so
the comparison is fair. --k auto walks {2, 4, 8, 16, 32} and takes the
SMALLEST k whose next_node ablation drops the clean next_node probe to within
--auto-tol of the majority baseline, i.e. the smallest manipulation strong
enough to be worth interpreting. next_node is the chooser: it is the one
target finding 46 showed tracks rho_50, so it is the condition that has to
bite for the experiment to mean anything. The sweep's full table is printed
either way, and --k <int> pins it.

Usage:
    python3 -m scripts.direction_ablation_rho --checkpoint <dir> --out-json <f> \
        --conditions clean ablate_next_node ablate_frontier_mask ...
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from scripts.n96_fallscan_analysis import rho_50
from scripts.rho_transition_eval import cfg_from_checkpoint
from spin_gnn.sat_transformer2 import is_tied, model_depth
from spin_gnn.sudoku_direction_ablate import (
    ABLATION_TARGETS, SubspaceAblation, build_bases, fit_probes_all_layers, raw_directions,
    readback, spectrum_fraction,
)
from spin_gnn.train_sudoku_transformer import eval_solve_by_rho, load_checkpoint

CONDITIONS = ["clean", "ablate_next_node", "ablate_frontier_mask", "ablate_next_color",
              "ablate_random"]
K_SWEEP = [2, 4, 8, 16, 32]


def choose_k(model, vocab, device, inst, packs, n_layers, max_pos, seed, d_model,
             ablate_from, tol, majority):
    """The pre-registered k rule, with the whole sweep printed."""
    print("\n=== k sweep: how hard does ablating next_node have to hit to bite? ===")
    print(f"  majority baseline for next_node top-1 is {majority:.4f}; "
          f"the rule takes the smallest k reaching <= {majority + tol:.4f}")
    chosen, table = None, []
    for k in K_SWEEP:
        bases = build_bases(packs, "ablate_next_node", n_layers, k, ablate_from,
                            d_model, np.random.default_rng(0))
        rb = readback(model, vocab, device, inst, packs, bases, n_layers, max_pos, seed)
        frac = spectrum_fraction(raw_directions(packs[(n_layers, "next_node")]), k)
        table.append({"k": k, "next_node_top1": rb["next_node"]["top1"],
                      "frontier_auroc": rb["frontier_mask"]["auroc"],
                      "spectrum_fraction_final_layer": frac})
        print(f"  k={k:2d}: next_node top1 {rb['next_node']['top1']:.4f}   "
              f"frontier AUROC {rb['frontier_mask']['auroc']:.4f}   "
              f"top-k holds {frac:.3f} of the probe's spectrum")
        if chosen is None and rb["next_node"]["top1"] <= majority + tol:
            chosen = k
    if chosen is None:
        chosen = K_SWEEP[-1]
        print(f"  NO k IN THE SWEEP BIT. Falling back to k={chosen}, and every rho_50 "
              f"below has to be read as a WEAK manipulation, not a clean deletion.")
    else:
        print(f"  -> k = {chosen}")
    return chosen, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--conditions", type=str, nargs="+", default=CONDITIONS)
    ap.add_argument("--k", type=str, default="auto",
                    help="subspace dimension, identical across conditions; 'auto' applies "
                         "the pre-registered rule in choose_k")
    ap.add_argument("--auto-tol", type=float, default=0.05)
    ap.add_argument("--ablate-from", type=int, default=1,
                    help="first block whose output is ablated (1 = everywhere). The last "
                         "block alone would only touch the logits, since nothing attends "
                         "to it.")
    ap.add_argument("--probe-epochs", type=int, default=600)
    ap.add_argument("--probe-seed", type=int, default=0)
    # 50979059's protocol, NOT sudoku_probes' function defaults (120/60/12).
    # The defaults give ~1436 train positions, and a 256 -> 96 probe fit on
    # that is badly underfit: t04's next_node came back at top-1 0.3986 against
    # the 0.8704 the probe job published. Ablating the singular subspace of an
    # underfit probe aims the knife at a bad estimate of the direction and
    # biases every condition toward "nothing is load-bearing", which is the
    # conclusion this job is most at risk of reaching for the wrong reason.
    ap.add_argument("--n-train-instances", type=int, default=400)
    ap.add_argument("--n-test-instances", type=int, default=160)
    ap.add_argument("--max-positions-per-instance", type=int, default=20)
    ap.add_argument("--rho-grid", type=float, nargs="+",
                    default=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45,
                             0.4, 0.35, 0.32])
    ap.add_argument("--n-per-bin", type=int, default=400)
    ap.add_argument("--seed-start", type=int, default=60_000_000_000,
                    help="fixed so every condition -- and the original fall scan -- sees "
                         "the SAME instances; changing it unpairs the comparison")
    ap.add_argument("--random-seed", type=int, default=0,
                    help="seed for the ablate_random subspaces only")
    ap.add_argument("--out-json", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, config = load_checkpoint(args.checkpoint, device=device)
    sudoku_cfg = cfg_from_checkpoint(config)
    d_model = int(config["d_model"])
    # model_depth, not len(encoder.layers): on a tied checkpoint the latter
    # prints 1 for a depth-16 network, and this line is the first thing a
    # reader checks the run against.
    print(f"checkpoint {os.path.basename(args.checkpoint)}  device {device}  "
          f"d_model {d_model}  depth {model_depth(model)}  tied {is_tied(model)}")

    t0 = time.time()
    packs, clean_metrics, test_inst, n_layers = fit_probes_all_layers(
        model, vocab, sudoku_cfg, device, args.n_train_instances, args.n_test_instances,
        args.max_positions_per_instance, args.probe_epochs, args.probe_seed)
    print(f"  probes fit in {time.time() - t0:.0f}s")

    majority = clean_metrics[(n_layers, "next_node")]["majority_accuracy"]
    if args.k == "auto":
        k, k_table = choose_k(model, vocab, device, test_inst, packs, n_layers,
                              args.max_positions_per_instance, args.probe_seed, d_model,
                              args.ablate_from, args.auto_tol, majority)
    else:
        k, k_table = int(args.k), []

    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "config": {key: config.get(key) for key in
                   ("n_nodes", "degree", "n_colors", "num_layers", "d_model",
                    "dim_feedforward", "nhead", "seed", "max_len", "fall", "shuffle_adj")},
        "args": {key: getattr(args, key) for key in
                 ("k", "auto_tol", "ablate_from", "probe_epochs", "probe_seed",
                  "n_train_instances", "n_test_instances", "max_positions_per_instance",
                  "rho_grid", "n_per_bin", "seed_start", "random_seed")},
        "k_used": k, "n_layers": n_layers, "device": device,
        "k_sweep": k_table,
        "clean_probe_final_layer": {
            t: {key: clean_metrics[(n_layers, t)][key]
                for key in ("auroc", "average_precision", "balanced_accuracy",
                            "majority_accuracy")}
            for t, _kind in ABLATION_TARGETS},
        "conditions": {},
    }

    rows = []
    for cond in args.conditions:
        tc = time.time()
        bases = build_bases(packs, cond, n_layers, k, args.ablate_from, d_model,
                            np.random.default_rng(args.random_seed))
        rb = readback(model, vocab, device, test_inst, packs, bases, n_layers,
                      args.max_positions_per_instance, args.probe_seed)
        print(f"\n=== {cond} (k={k}, layers {args.ablate_from}..{n_layers}) ===")
        print(f"  read-back: frontier AUROC {rb['frontier_mask']['auroc']:.4f}   "
              f"next_node top1 {rb['next_node']['top1']:.4f}   "
              f"next_color top1 {rb['next_color']['top1']:.4f}")
        with SubspaceAblation(model, bases, device=device):
            records = eval_solve_by_rho(model, vocab, device, sudoku_cfg, args.rho_grid,
                                        args.n_per_bin, args.seed_start, config["max_len"])
        curve = [(r["realized_rho"], r["solve_rate"]) for r in records]
        value, flag = rho_50(curve)
        # rho_50 SATURATES: once no bin reaches 0.5 it pins at the top of the
        # grid and cannot tell "a bit worse" from "destroyed", which is exactly
        # the regime a strong ablation lands in. mean_solve_rate keeps moving.
        # well_formed/truncated separate "stopped reasoning" from "stopped
        # writing legal syntax", two very different things for an ablation to do.
        summary = {m: float(np.mean([r[m] for r in records]))
                   for m in ("solve_rate", "well_formed_rate", "truncated_rate")}
        print(f"  rho_50 = {value:.4f} ({flag})   mean solve_rate "
              f"{summary['solve_rate']:.4f}   well_formed {summary['well_formed_rate']:.3f}   "
              f"truncated {summary['truncated_rate']:.3f}   [{time.time() - tc:.0f}s]")
        out["conditions"][cond] = {
            "readback": rb, "rho_50": value, "rho_50_flag": flag,
            "mean_solve_rate": summary["solve_rate"],
            "mean_well_formed_rate": summary["well_formed_rate"],
            "mean_truncated_rate": summary["truncated_rate"],
            "spectrum_fraction_final_layer": (
                None if cond in ("clean", "ablate_random")
                else spectrum_fraction(raw_directions(packs[(n_layers, cond[len("ablate_"):])]), k)),
            "curve": records,
        }
        for r in records:
            rows.append({"checkpoint": os.path.basename(args.checkpoint), "condition": cond,
                         "k": k, "ablate_from": args.ablate_from, **r})

        # Rewritten after every condition, so a job that hits its time limit
        # mid-sweep still leaves the conditions it did finish on disk.
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, default=float)
        if args.out_csv:
            with open(args.out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    print("\n=== by condition. rho_50 LOWER is better, mean solve_rate HIGHER is better. ===")
    print("    Read every row against ablate_random, not against clean: removing ANY k "
          "dimensions hurts,\n    and only the excess over the count-matched random "
          "subspace is attributable to the direction.")
    base = out["conditions"].get("clean", {})
    for cond in args.conditions:
        c = out["conditions"][cond]
        d50 = "" if not base else f"  d {c['rho_50'] - base['rho_50']:+.4f}"
        dsr = "" if not base else f"  d {c['mean_solve_rate'] - base['mean_solve_rate']:+.4f}"
        print(f"  {cond:22s} rho_50 {c['rho_50']:.4f} ({c['rho_50_flag']:6s}){d50}   "
              f"mean solve {c['mean_solve_rate']:.4f}{dsr}   "
              f"well_formed {c['mean_well_formed_rate']:.3f}")
    print(f"\nwrote {args.out_json}" + (f" and {args.out_csv}" if args.out_csv else ""))


if __name__ == "__main__":
    main()
