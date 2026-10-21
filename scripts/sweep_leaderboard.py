"""Leaderboard over every sweep.csv, on both solve-rate metrics at once.

This exists because of a specific mistake. The project spent weeks quoting
agg_solve_rate = 0.767 (d0128 @900ep) as the best result while
sudoku_hparam_sweep/sweep.csv already contained 0.847 (d0256 @300ep). Nothing
was wrong with the data: the ranking had been done by eye when
mean_o1_solve_rate was the headline metric -- d0064 wins that at 0.980 -- and
was never redone when the headline changed. Rows are appended to sweep.csv in
run order and nothing in the pipeline ever sorts them, so there was no moment
at which the stale ranking would have been contradicted.

The two metrics do not measure the same thing and the difference is not a
detail:

  agg_solve_rate      solve_rate on the held-out val instances over
                      rho in [0.3, 0.95] -- most of the hints removed, so
                      something has to be deduced.
  mean_o1_solve_rate  mean over k in 0..4 of solve_rate at
                      rho = (n_nodes - k)/n_nodes, i.e. 1.000, 0.958, 0.917,
                      0.875, 0.833 for n_nodes=24 -- nearly fully hinted.
                      k=0 is rho=1.0: nothing to deduce at all.

The two rho ranges are disjoint, so a model can top one and sit mid-table on
the other, and d0256 vs d0064 is exactly that. Across the sweep as a whole
the two metrics mostly agree (r = +0.94, 31 runs) -- the conflict is local to
the top of the table, where the agg leader is also the single worst model at
rho = 1.0 (k0 = 0.24 against d0064's 1.00). That is what makes it worth
printing both side by side with the disagreement marked. This script
deliberately does not nominate a single winner: which range is the target of
record is an open question, and not one a sort order should quietly answer.

Usage:
    python scripts/sweep_leaderboard.py                 # all sweeps
    python scripts/sweep_leaderboard.py --top 5
"""
import argparse
import csv
import glob
import os

SCRATCH = os.environ.get("SCRATCH", "")
DEFAULT_GLOB = os.path.join(SCRATCH, "spin_gnn", "sudoku_hparam_sweep*", "sweep.csv")


def load(paths):
    rows = []
    for path in paths:
        sweep = os.path.basename(os.path.dirname(path))
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if not r.get("checkpoint_dir"):
                    continue
                r["sweep"] = sweep
                r["tag"] = os.path.basename(r["checkpoint_dir"])
                for key in ("agg_solve_rate", "mean_o1_solve_rate", "solve_rate_k0",
                            "final_val_nll", "epochs", "n_train_instances", "n_params"):
                    try:
                        r[key] = float(r.get(key, "") or "nan")
                    except ValueError:
                        r[key] = float("nan")
                rows.append(r)
    return rows


def table(rows, metric, top):
    ranked = sorted(rows, key=lambda r: (-r[metric] if r[metric] == r[metric] else 1))
    print(f"\n=== top {top} by {metric} ===")
    print(f"  {'#':>2} {'agg':>6} {'o1':>6} {'k0':>5} {'val_nll':>7} "
          f"{'ep':>5} {'n_tr':>6}  tag  [sweep]")
    for i, r in enumerate(ranked[:top], 1):
        print(f"  {i:>2} {r['agg_solve_rate']:6.3f} {r['mean_o1_solve_rate']:6.3f} "
              f"{r['solve_rate_k0']:5.2f} {r['final_val_nll']:7.3f} "
              f"{int(r['epochs']):5d} {int(r['n_train_instances']):6d}  "
              f"{r['tag']}  [{r['sweep']}]")
    return ranked


def correlations(rows):
    """What, across the whole sweep, actually tracks agg_solve_rate.

    Worth computing rather than eyeballing: the top of the leaderboard is a
    biased slice. Reading the top six rows suggests final_val_nll is
    ANTI-correlated with agg_solve_rate (the three best agg models have
    val_nll 2.41/1.68/2.16 while rank 5-6 sit at 1.17/1.20), which would be a
    striking result. Over all runs it is not there: r = +0.17, i.e. val_nll
    is close to uninformative about the metric of record. That is still a
    useful fact, just a different one -- it means a run cannot be called an
    improvement on the strength of its validation loss.
    """
    import numpy as np
    a = np.array([r["agg_solve_rate"] for r in rows])
    print("\n=== what tracks agg_solve_rate, over ALL runs ===")
    for key in ("mean_o1_solve_rate", "solve_rate_k0", "final_val_nll",
                "n_params", "epochs", "n_train_instances"):
        v = np.array([r[key] for r in rows])
        m = np.isfinite(a) & np.isfinite(v)
        if m.sum() < 3 or v[m].std() == 0:
            print(f"  r(agg, {key:18s}) =  n/a  (constant or too few runs)")
            continue
        print(f"  r(agg, {key:18s}) = {np.corrcoef(a[m], v[m])[0, 1]:+.3f}   n={int(m.sum())}")
    print("  Single seed per configuration and a sweep that was never a designed")
    print("  factorial, so these are descriptive, not estimates of an effect.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no sweep.csv matched {args.glob}")
    rows = load(paths)
    print(f"{len(rows)} runs from {len(paths)} sweep.csv files")

    by_agg = table(rows, "agg_solve_rate", args.top)
    by_o1 = table(rows, "mean_o1_solve_rate", args.top)
    correlations(rows)

    print("\n=== metric disagreement ===")
    a, o = by_agg[0], by_o1[0]
    if a["tag"] == o["tag"] and a["sweep"] == o["sweep"]:
        print(f"  none: {a['tag']} leads both.")
    else:
        print(f"  agg leader {a['tag']} [{a['sweep']}]: "
              f"agg={a['agg_solve_rate']:.3f}  o1={a['mean_o1_solve_rate']:.3f}  "
              f"k0={a['solve_rate_k0']:.2f}")
        print(f"   o1 leader {o['tag']} [{o['sweep']}]: "
              f"agg={o['agg_solve_rate']:.3f}  o1={o['mean_o1_solve_rate']:.3f}  "
              f"k0={o['solve_rate_k0']:.2f}")
        print("  The two score disjoint rho ranges (see this file's docstring);")
        print("  pick the target of record deliberately, do not read a winner off")
        print("  whichever column was sorted last.")


if __name__ == "__main__":
    main()
