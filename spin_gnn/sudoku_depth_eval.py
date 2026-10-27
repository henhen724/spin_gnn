"""solve_rate binned by DD reasoning depth T at a FIXED hint density rho.

This exists to break a confound that every result in the project so far is
subject to. In the removal-path ensemble, hints are removed only while DD can
still recover the full coloring, so removing a hint is *how* the deduction
chain gets longer: rho and T move together by construction. The transition
sweep measures r(solve_rate, T) = -0.966 and r(solve_rate, rho) = +0.944 on
the same instances, which is not two findings but one, and it cannot say
whether the model is limited by how much it has to deduce (depth) or by how
little it is told (density).

eval_solve_by_rho varies rho across bins, so it cannot separate them. This
module holds rho fixed and bins the SAME-density instances by their own T,
which varies across instances because the graph and the removal order are
random. Any solve_rate gradient across those bins is attributable to depth
alone, because density is pinned.

The validity of that claim rests entirely on one check, which is why
`realized_rho` is reported per bin and the spread across bins is printed as a
headline number rather than buried: a greedy removal path cannot go below its
own rho_min, so the *requested* rho is not always the *realized* one, and if
deep-T instances systematically realize a lower density than shallow-T ones
then the confound has simply moved inside the bins. Read realized_rho_spread
before reading anything else here. If it is not small compared to the rho
range the transition sweep covers, this measurement does not do its job.

Usage:
    python -m spin_gnn.sudoku_depth_eval --checkpoint <dir> --rho 0.6 \\
        --n-pool 600 --out-csv depth.csv
"""
import argparse
import csv
import os
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .train_sudoku_transformer import build_instances, load_checkpoint, solve_rate


def bin_by_depth(instances, min_per_bin):
    """Group instances by T = len(trace["rounds"]), dropping thin bins.

    Thin bins are dropped rather than merged: T is small-integer valued, so
    merging adjacent values would blur exactly the axis being measured, and a
    bin of five instances has a solve_rate standard error near 0.2 anyway.
    """
    by_t = defaultdict(list)
    for inst in instances:
        by_t[len(inst[3]["rounds"])].append(inst)
    kept = {t: v for t, v in by_t.items() if len(v) >= min_per_bin}
    dropped = sum(len(v) for t, v in by_t.items() if len(v) < min_per_bin)
    return dict(sorted(kept.items())), dropped


def eval_solve_by_depth(
    model, vocab, device, sudoku_cfg: SudokuReasoningConfig, rho, n_pool, seed_start,
    max_new_tokens, min_per_bin=40,
):
    """Returns (records, diagnostics). One record per T bin, all at one rho."""
    # rho and rho_max pinned to the same value: no per-instance density spread.
    pool_cfg = replace(sudoku_cfg, rho=float(rho), rho_max=float(rho))
    pool = build_instances(pool_cfg, n_pool, seed_start)
    bins, n_dropped = bin_by_depth(pool, min_per_bin)
    if not bins:
        raise SystemExit(
            f"no T bin reached {min_per_bin} instances out of {n_pool} at rho={rho}; "
            "raise --n-pool or lower --min-per-bin"
        )

    n_nodes = sudoku_cfg.n_nodes
    records = []
    for t, insts in bins.items():
        metrics = solve_rate(model, insts, vocab, device, max_new_tokens)
        records.append({
            "rho_requested": float(rho),
            "T": int(t),
            "n": len(insts),
            "realized_rho": sum(len(h) for _, _, h, _ in insts) / (len(insts) * n_nodes),
            **metrics,
        })

    solve = np.array([r["solve_rate"] for r in records], dtype=float)
    depth = np.array([r["T"] for r in records], dtype=float)
    realized = np.array([r["realized_rho"] for r in records], dtype=float)
    diagnostics = {
        "n_pool": n_pool,
        "n_bins": len(records),
        "n_dropped_thin_bins": n_dropped,
        # The number that decides whether this measurement is worth anything.
        "realized_rho_spread": float(realized.max() - realized.min()),
        "realized_rho_mean": float(realized.mean()),
        "r_solve_T_at_fixed_rho": float(np.corrcoef(solve, depth)[0, 1]) if len(records) > 2 else float("nan"),
        # If density really is pinned, this should be noise, not structure. A
        # zero-variance realized_rho is the BEST case, not a missing value, so
        # it is reported as an exact pin rather than left as the NaN that
        # corrcoef returns when it divides by a zero standard deviation.
        "r_realized_rho_T": (
            float(np.corrcoef(realized, depth)[0, 1])
            if len(records) > 2 and realized.std() > 0 else float("nan")
        ),
        "realized_rho_exactly_pinned": bool(realized.std() == 0),
    }
    return records, diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--rho", type=float, nargs="+", default=[0.6],
                    help="one fixed density per sweep; several lets the depth gradient "
                         "itself be compared across densities")
    ap.add_argument("--n-pool", type=int, default=600,
                    help="instances generated per rho before binning; the bins are "
                         "whatever T happens to produce, so this has to be generous")
    ap.add_argument("--min-per-bin", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=30_000_000_000)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ckpt in args.checkpoint:
        model, vocab, config = load_checkpoint(ckpt, device=device)
        sudoku_cfg = SudokuReasoningConfig(
            n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
            rho=config["rho"], rho_max=config["rho_max"],
            hint_mode=config.get("hint_mode", "removal"),
            scratchpad=config.get("scratchpad", False),
            shuffle_adj=config.get("shuffle_adj", False),
            shuffle_events=config.get("shuffle_events", False),
        )
        for rho in args.rho:
            records, diag = eval_solve_by_depth(
                model, vocab, device, sudoku_cfg, rho, args.n_pool, args.seed_start,
                config["max_len"], args.min_per_bin,
            )
            name = os.path.basename(ckpt.rstrip("/"))
            print(f"\n=== {name}  rho={rho}  ({diag['n_bins']} bins, "
                  f"{diag['n_dropped_thin_bins']} instances in bins too thin to keep) ===")
            print(f"  {'T':>3} {'n':>5} {'real_rho':>9} {'solve':>7} {'outcome':>8}")
            for r in records:
                print(f"  {r['T']:>3} {r['n']:>5} {r['realized_rho']:9.4f} "
                      f"{r['solve_rate']:7.3f} {r['outcome_accuracy']:8.3f}")
                rows.append({"checkpoint": name, **r})
            print(f"  realized_rho spread across bins: {diag['realized_rho_spread']:.4f} "
                  f"(mean {diag['realized_rho_mean']:.4f})")
            print(f"  r(solve_rate, T) at fixed rho:   {diag['r_solve_T_at_fixed_rho']:+.3f}")
            if diag["realized_rho_exactly_pinned"]:
                print("  r(realized_rho, T):              n/a -- realized_rho is IDENTICAL "
                      "in every bin, the clean case")
            else:
                print(f"  r(realized_rho, T):              {diag['r_realized_rho_T']:+.3f}  "
                      "<- near 0 means density really is pinned; a large value means the "
                      "confound moved inside the bins and this rho must be discounted")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
