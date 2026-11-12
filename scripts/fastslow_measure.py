"""Re-measure the ensemble side of the fast/slow transition (finding 43).

This exists so the phases deck can draw FIGURES. Finding 43's numbers are
archived as prose in FINDINGS.md and SUDOKU_DD.md -- `rho_min` 0.475 -> 0.309,
T ~ N^0.727, the 0.3827 inclusion-exclusion density -- but the per-size arrays
behind them were written by `$SCRATCH/fall_diag.py` on 2026-11-02 and never
landed in the tree. Typing archived scalars into a deck is a table nobody can
re-run, so this re-derives them with the project's own generator and writes one
JSON the builder reads.

WHAT IS MEASURED, and why each piece is on a slide:

  sizes      `rho_min` and the deduction depth T at the end of a greedy removal
             path, per ensemble, over six sizes. The random ensemble's floor
             converging ABOVE rho_c ~ 0.46 while the fall ensemble's drops
             below it is the whole point: one ensemble cannot reach the slow
             phase at any N, the other does.
  degrees    The permanently-undeducible fraction against the inclusion-
             exclusion prediction sum_{k=1}^{q-1} (-1)^{k+1} C(q-1,k)
             (1-k/(q-1))^d. This is the MECHANISM page -- the floor is not a
             sampler artifact, it is a node count fixed by (q, d) alone.
  path       Mean T along the removal path at N=48 on a common rho grid, per
             ensemble. The training-distribution page: where the two paths
             stop and how steeply T rises before they do.
  tokens     Sequence length and round count for real N=48 training instances
             of both ensembles. The cost page: the fall ensemble reasons for
             twice as many rounds at the same sequence length, because a trace
             holds one event per DEDUCED NODE (a count set by rho) and depth
             only adds BEGIN_ROUND markers.

Nothing here touches a trained model; this is the task, not the solver. The
grids the deck reads for the model side were scored on 2026-11-02 and are not
recomputed.

Run in a job (numba JIT plus the N=17196 paths):
    python3 -u -m scripts.fastslow_measure --out <dir> [--quick]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/users/hshunt/spin_gnn")

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_dd import on_grid, planted_regular_graph, removal_path
from spin_gnn.sudoku_tokens import Vocab, encode_instance
from spin_gnn.train_sudoku_transformer import build_instances

Q = 4
D = 5

# rho_c for q=4, d=5 in the thermodynamic limit, read off Dario's Sudoku_SC
# slides. Quoted, never fitted here -- this script measures where OUR greedy
# paths stop, and the slides' threshold is the line they have to cross.
RHO_C = 0.46

# Sizes, with seed counts that fall off as the paths get slower. N=48 keeps
# finding 43's 200 seeds because the deck quotes its mean +- sd.
#
# The top counts are deliberately generous. The first run of this script (job
# 52042656, the whole sweep in 49 s) used 5 and 3 seeds at the top two sizes and
# the local exponent came out 0.730 then 0.647, against finding 43's 0.7266 and
# 0.7273 -- a 3-seed mean of T_end at N=17196 has a standard error of ~21 on a
# value of 804, which is most of that wobble. The deck's claim is that the
# exponent is the SAME across independent intervals, so the intervals have to
# be measured well enough to make that falsifiable. At ~4.5 s per fall path at
# N=17196 the seed count is nearly free.
SIZES = [(48, 400), (96, 200), (252, 100), (996, 50), (4140, 30), (17196, 20),
         (49992, 8)]
QUICK = [(48, 24), (96, 12), (252, 6), (996, 3)]

# Degrees for the mechanism page. d = q-1 = 3 is the smallest degree at which a
# fall colouring can exist at all; d=5 and d=7 are the two the slides use.
DEGREES = [3, 4, 5, 6, 7, 8]
DEG_N, DEG_SEEDS = 996, 5

# The eval/training instance config, copied from a fall-scan checkpoint's own
# config.json rather than retyped: rho in [0.30, 0.95], removal-path hints,
# shuffled adjacency.
RHO, RHO_MAX = 0.30, 0.95
N_TOKEN_INSTANCES = 200


def noncolourful_prediction(q, d):
    """Inclusion-exclusion density of nodes missing a colour from their
    neighbourhood. Such a node can never be deduced, so it is a PERMANENT hint
    and greedy removal halts on the set of them."""
    return sum((-1) ** (k + 1) * math.comb(q - 1, k) * (1 - k / (q - 1)) ** d
               for k in range(1, q))


def noncolourful_fraction(adj, colors, q):
    """Measured counterpart: fraction whose closed neighbourhood misses a colour."""
    n = len(colors)
    seen = np.zeros((n, q), dtype=bool)
    seen[np.repeat(np.arange(n), adj.shape[1]), colors[adj].ravel()] = True
    seen[np.arange(n), colors] = True
    return float((~seen.all(axis=1)).mean())


def draw(N, d, q, seed, fall, attempts=30):
    """One planted graph. The fall construction reserves a mandatory stub per
    ordered colour pair and can still dead-end on the random remainder, so a
    RuntimeError is a failed draw, not a bug -- retry the stream."""
    for a in range(attempts):
        try:
            return planted_regular_graph(N, d, q, np.random.default_rng(seed + a * 7_919),
                                         fall=fall)
        except RuntimeError:
            continue
    raise SystemExit("no graph for N=%d d=%d fall=%s in %d attempts" % (N, d, fall, attempts))


def stats(xs):
    xs = [float(x) for x in xs]
    n = len(xs)
    mean = sum(xs) / n
    sd = (sum((x - mean) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "n": n, "min": min(xs), "max": max(xs)}


def measure_sizes(sizes, save=None):
    """rho_min, T_end and the non-colourful fraction, per size per ensemble.

    `save` is called after every row. The largest fall paths are the slow part
    of this script (that is the finding), so a wall-clock kill must leave the
    sizes measured so far on disk rather than nothing.
    """
    out = []
    for N, n_seeds in sizes:
        for name, fall in (("random", False), ("fall", True)):
            t0 = time.time()
            rmin, tend, nonc, mend = [], [], [], []
            for s in range(n_seeds):
                adj, colors = draw(N, D, Q, 1_000 + s, fall)
                nonc.append(noncolourful_fraction(adj, colors, Q))
                p = removal_path(adj, colors, Q, np.random.default_rng(2_000 + s))
                rmin.append(p["rho_min"])
                tend.append(p["T"][-1])
                mend.append(p["m"][-1])
            out.append({"N": N, "degree": D, "ensemble": name, "seeds": n_seeds,
                        "rho_min": stats(rmin), "T_end": stats(tend),
                        "noncolourful": stats(nonc), "m_at_floor": stats(mend),
                        "secs": round(time.time() - t0, 1)})
            r, t = out[-1]["rho_min"]["mean"], out[-1]["T_end"]["mean"]
            print("  N=%-6d %-6s rho_min %.4f  T_end %7.1f  nonc %.4f  (%.0fs)"
                  % (N, name, r, t, out[-1]["noncolourful"]["mean"], out[-1]["secs"]),
                  flush=True)
            if save:
                save(out)
    return out


def gammas(sizes_rows, ensemble):
    """Local power-law exponent of T_end between consecutive sizes.

    Reported per interval rather than as one fit, because that is the claim:
    finding 43 quotes gamma = 0.7266 and 0.7273 over two INDEPENDENT intervals,
    and two intervals agreeing is what makes it a power law rather than a line
    drawn through curvature.
    """
    rows = sorted([r for r in sizes_rows if r["ensemble"] == ensemble],
                  key=lambda r: r["N"])
    out = []
    for a, b in zip(rows, rows[1:]):
        g = (math.log(b["T_end"]["mean"]) - math.log(a["T_end"]["mean"])) \
            / (math.log(b["N"]) - math.log(a["N"]))
        out.append({"from": a["N"], "to": b["N"], "gamma": g})
    return out


def measure_degrees():
    """The mechanism: predicted vs measured permanent-hint density by degree."""
    out = []
    for d in DEGREES:
        row = {"degree": d, "predicted": noncolourful_prediction(Q, d),
               "N": DEG_N, "seeds": DEG_SEEDS}
        for name, fall in (("random", False), ("fall", True)):
            fr, rm = [], []
            for s in range(DEG_SEEDS):
                adj, colors = draw(DEG_N, d, Q, 3_000 + s, fall)
                fr.append(noncolourful_fraction(adj, colors, Q))
                rm.append(removal_path(adj, colors, Q,
                                       np.random.default_rng(4_000 + s))["rho_min"])
            row[name] = stats(fr)
            row[name + "_rho_min"] = stats(rm)
        out.append(row)
        print("  d=%d  predicted %.4f  random %.4f  fall %.4f  rho_min %.3f -> %.3f"
              % (d, row["predicted"], row["random"]["mean"], row["fall"]["mean"],
                 row["random_rho_min"]["mean"], row["fall_rho_min"]["mean"]), flush=True)
    return out


def measure_path(N=48, n_seeds=200):
    """Mean T along the removal path at a training size, on a common rho grid.

    NaN above a path's own start or below its floor, so the mean at each rho is
    over the paths that REACH it -- and the count is kept, because the random
    ensemble's curve ending at 0.47 is the figure's point and a mean over three
    surviving paths would otherwise look like a continuation.
    """
    grid = [round(x, 4) for x in np.arange(0.95, 0.28, -0.01)]
    out = {"N": N, "seeds": n_seeds, "rho": grid}
    for name, fall in (("random", False), ("fall", True)):
        Ts = []
        for s in range(n_seeds):
            adj, colors = draw(N, D, Q, 5_000 + s, fall)
            p = removal_path(adj, colors, Q, np.random.default_rng(6_000 + s))
            T, _m = on_grid(p, grid)
            Ts.append(T)
        A = np.array(Ts, dtype=float)
        reach = (~np.isnan(A)).sum(axis=0)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(A, axis=0)
        out[name] = {"T_mean": [None if not np.isfinite(v) else float(v) for v in mean],
                     "n_reaching": [int(c) for c in reach]}
        print("  path %-6s reaches rho=%.2f with %d/%d paths"
              % (name, grid[-1], reach[-1], n_seeds), flush=True)
    return out


def measure_tokens(N=48, n=N_TOKEN_INSTANCES):
    """What the model is fed: sequence length and round count, both ensembles.

    Through build_instances and encode_instance, the training path's own
    functions, so 'the sequence barely got longer' is a measurement of the real
    encoder and not of an assumption about it.
    """
    out = {"N": N, "n_instances": n}
    for name, fall in (("random", False), ("fall", True)):
        cfg = SudokuReasoningConfig(n_nodes=N, degree=D, n_colors=Q, rho=RHO,
                                    rho_max=RHO_MAX, hint_mode="removal",
                                    scratchpad=False, shuffle_adj=True, fall=fall)
        insts = build_instances(cfg, n, seed_start=7_000_000)
        vocab = Vocab(N, Q, D, scratchpad=False)
        lens, rounds, events = [], [], []
        for adj, _colors, _hints, trace in insts:
            lens.append(len(encode_instance(adj, trace, vocab)))
            rounds.append(len(trace["rounds"]))
            events.append(sum(len(r) for r in trace["rounds"]))
        out[name] = {"tokens": stats(lens), "rounds": stats(rounds),
                     "events": stats(events), "vocab_size": vocab.size}
        print("  tokens %-6s mean %.1f  max %d  rounds mean %.1f  max %d"
              % (name, out[name]["tokens"]["mean"], out[name]["tokens"]["max"],
                 out[name]["rounds"]["mean"], out[name]["rounds"]["max"]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="small sizes and few seeds, for a shape check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sizes = QUICK if args.quick else SIZES
    t0 = time.time()
    out = os.path.join(args.out, "fastslow.json")
    doc = {"q": Q, "degree": D, "rho_c": RHO_C, "quick": args.quick,
           "sizes": [], "degrees": [], "path": {}, "tokens": {}, "gamma": {},
           "complete": False}

    def save():
        doc["secs"] = round(time.time() - t0, 1)
        with open(out, "w") as fh:
            json.dump(doc, fh, indent=1)

    print("sizes (d=%d, q=%d):" % (D, Q), flush=True)
    doc["sizes"] = measure_sizes(sizes, save=lambda rows: (doc.__setitem__("sizes", rows),
                                                           save()))
    doc["gamma"] = {"fall": gammas(doc["sizes"], "fall"),
                    "random": gammas(doc["sizes"], "random")}
    save()
    print("degrees at N=%d:" % DEG_N, flush=True)
    doc["degrees"] = measure_degrees()
    save()
    print("removal paths at N=48:", flush=True)
    doc["path"] = measure_path(n_seeds=24 if args.quick else 200)
    save()
    print("tokenized instances at N=48:", flush=True)
    doc["tokens"] = measure_tokens(n=40 if args.quick else N_TOKEN_INSTANCES)
    doc["complete"] = True
    save()
    print("\nwrote %s in %.0fs" % (out, doc["secs"]))
    for g in doc["gamma"]["fall"]:
        print("  fall gamma %d->%d: %.4f" % (g["from"], g["to"], g["gamma"]))


if __name__ == "__main__":
    main()
