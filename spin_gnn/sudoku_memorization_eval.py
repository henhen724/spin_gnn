"""What is the model actually fitting: its training instances, or the encoding?

Two measurements that the project has never run, both eval-only.

PART A -- MEMORIZATION. Every instance is a fresh planted graph seeded by index
(train_sudoku_transformer.build_instances: graph seed = seed_start + i), and
train/val/pool seed ranges are disjoint, so no instance is ever *shared*. But
`train_instances` is built ONCE, before the epoch loop, so a 150-epoch run sees
the same 20000-80000 instances 150 times and can memorize them. Training index
i is reproducible as build_instances(cfg, 1, i), so a prefix of the training set
is recoverable exactly: we score the model on its OWN training instances and on
held-out ones. Any solve_rate gap is memorization, measured directly rather than
inferred from NLL curves.

Note on why the NLL curves could not answer this: `train_nll` is
`epoch_loss / epoch_n`, averaged over the epoch WHILE the weights update, while
`val_nll` uses end-of-epoch weights. The resulting upward bias on train_nll is
worth +0.33 early in training and decays as the curve flattens, so the sign of a
+-0.03 "train - val gap" says nothing -- especially against n_val=300 whose own
difficulty draw is worth a few hundredths.

PART B -- IS THE ANSWER IN THE TOKEN ORDER? planted_regular_graph appends edges
in class-pair block order (0,1),(0,2),(0,3),(1,2),... and fills adj[v] in that
order, so for a node of class a the blocks containing it are visited in
increasing order of the other class index. Measured: adj[v] lists v's
neighbours sorted by their TRUE COLOUR for 100.0% of nodes at N=24, 48 and 96,
and encode_instance emits them in exactly that order. For 61-67% of nodes the
neighbours cover all q-1 other colours, so the one absent colour block is v's
own. The model may therefore read neighbour colours off POSITION instead of
deducing them -- which would generalize to every fresh instance from this
generator and so be invisible to any train/val diagnostic.

Four orderings, all leaving the edge SET, the colouring, the hints and the DD
trace identical -- only the token order changes:
  identity      the training order (sorted by neighbour colour)
  within_block  permute only among neighbours of the SAME colour. Colour code
                intact, order changed. Within-block order is already arbitrary
                in training (sorted by node id in 36-39% of runs, i.e. chance),
                so this is IN-DISTRIBUTION and is the control that separates
                "order changed" from "colour code destroyed".
  shuffle       permute the whole list. Colour code destroyed.
  byid          sort by node id. Consistent, colour-free.
If within_block holds and shuffle collapses, the model is reading colour off
adjacency position. A drop under shuffle ALONE would be confounded with plain
distribution shift, which is exactly what within_block controls for; settling it
properly still needs a retrain under a colour-free order, and this job only says
whether that retrain is worth its GPU time.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .sudoku_agg_eval import mcnemar_exact_p, wilson_interval
from .sudoku_trace import solve_with_dd_trace
from .train_sudoku_transformer import build_instances, load_checkpoint, solve_rate

ORDERINGS = ("identity", "within_block", "shuffle", "byid")


def reorder(instances, mode, rng, n_colors, verify=True):
    """Reorder adj[v] within each node. Returns new instances; colors, hints and
    trace are passed through unchanged.

    With verify=True the DD trace is recomputed on the reordered adjacency and
    checked against the original, which is what licenses calling this a
    semantics-preserving counterfactual rather than a different problem.
    """
    out = []
    for adj, colors, hints, trace in instances:
        a = adj.copy()
        if mode == "identity":
            pass
        elif mode == "shuffle":
            for v in range(a.shape[0]):
                a[v] = rng.permutation(a[v])
        elif mode == "byid":
            a = np.sort(a, axis=1)
        elif mode == "within_block":
            for v in range(a.shape[0]):
                cs = colors[a[v]]
                for c in np.unique(cs):
                    sel = np.flatnonzero(cs == c)
                    a[v, sel] = rng.permutation(a[v, sel])
        else:
            raise ValueError(f"unknown ordering {mode!r}")
        if verify and mode != "identity":
            # same edge set per node, and DD derives the same rounds
            assert set(map(int, a[0])) == set(map(int, adj[0]))
            t2 = solve_with_dd_trace(a, colors, hints, n_colors)
            assert t2["resolved"] == trace["resolved"], mode
            assert [sorted(r) for r in t2["rounds"]] == [sorted(r) for r in trace["rounds"]], mode
        out.append((a, colors, hints, trace))
    return out


def two_proportion_p(k1, n1, k2, n2):
    """Unpaired two-sided z-test. Train-sample and held-out are DIFFERENT
    instances, so McNemar does not apply to that comparison."""
    from math import erfc, sqrt
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return erfc(abs(p1 - p2) / (se * sqrt(2)))


def cfg_from(config):
    return SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        rho=config["rho"], rho_max=config["rho_max"],
        hint_mode=config.get("hint_mode") or "removal",
        scratchpad=bool(config.get("scratchpad", False)),
    )


def score(model, insts, vocab, device, max_len, label):
    m = solve_rate(model, insts, vocab, device, max_len, return_per_instance=True)
    solved = m["solved_per_instance"]
    lo, hi = wilson_interval(int(solved.sum()), len(solved))
    print(f"    {m['solve_rate']:.4f} [{lo:.4f}, {hi:.4f}]  "
          f"trunc {m['truncated_rate']:.4f}  wf {m['well_formed_rate']:.4f}  {label}",
          flush=True)
    return m, solved, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--n-memorization", type=int, default=600,
                    help="instances per side of the train-vs-heldout comparison; 0 skips part A")
    ap.add_argument("--n-ordering", type=int, default=0,
                    help="instances for the ordering counterfactual; 0 skips part B")
    ap.add_argument("--heldout-seed-start", type=int, default=50_000_000_000)
    ap.add_argument("--seed", type=int, default=0, help="seed for the permutations")
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ckpt in args.checkpoint:
        model, vocab, config = load_checkpoint(ckpt, device=device)
        cfg = cfg_from(config)
        name = os.path.basename(ckpt.rstrip("/"))
        n_train = config.get("n_train_instances")
        print(f"\n=== {name}  N={cfg.n_nodes} L={config['num_layers']} "
              f"ff={config.get('dim_feedforward')} n_train={n_train} "
              f"epochs={config.get('epochs')} ===", flush=True)
        base = {"checkpoint": name, "run_dir": ckpt, "n_nodes": cfg.n_nodes,
                "num_layers": config["num_layers"],
                "dim_feedforward": config.get("dim_feedforward"),
                "n_train_instances": n_train, "epochs": config.get("epochs")}

        if args.n_memorization:
            n = args.n_memorization
            if n_train is not None and n > n_train:
                raise SystemExit(f"n_memorization={n} exceeds n_train={n_train}")
            print("  PART A: its own training instances vs held-out", flush=True)
            # A prefix of the training set: training index i is reproducible as
            # build_instances(cfg, 1, i), so indices 0..n-1 are exactly the first
            # n instances the model trained on. Difficulty is iid in i, so a
            # prefix is a fair sample and needs no extra randomization.
            # Checkpoints written before 38d503a have seed/epochs/n_train present
            # but null, so `.get("seed", 0)` returns None rather than the default.
            # Every run in the project trained at seed=0.
            tr = build_instances(cfg, n, config.get("seed") or 0)
            ho = build_instances(cfg, n, args.heldout_seed_start)
            mt, st, tlo, thi = score(model, tr, vocab, device, config["max_len"], "TRAIN sample")
            mh, sh, hlo, hhi = score(model, ho, vocab, device, config["max_len"], "held-out")
            gap = mt["solve_rate"] - mh["solve_rate"]
            p = two_proportion_p(int(st.sum()), n, int(sh.sum()), n)
            print(f"    memorization gap {gap:+.4f}  unpaired p={p:.3g}", flush=True)
            rows.append({**base, "part": "memorization", "ordering": "identity",
                         "n_instances": n, "train_solve_rate": mt["solve_rate"],
                         "heldout_solve_rate": mh["solve_rate"], "gap": gap, "p": p,
                         "train_wilson_lo": tlo, "train_wilson_hi": thi,
                         "heldout_wilson_lo": hlo, "heldout_wilson_hi": hhi})

        if args.n_ordering:
            n = args.n_ordering
            print("  PART B: held-out under four adjacency orderings", flush=True)
            ho = build_instances(cfg, n, args.heldout_seed_start)
            ref = None
            for mode in ORDERINGS:
                rng = np.random.default_rng(args.seed)
                insts = reorder(ho, mode, rng, cfg.n_colors)
                m, s, lo, hi = score(model, insts, vocab, device, config["max_len"], mode)
                row = {**base, "part": "ordering", "ordering": mode, "n_instances": n,
                       "heldout_solve_rate": m["solve_rate"],
                       "heldout_wilson_lo": lo, "heldout_wilson_hi": hi,
                       "well_formed_rate": m["well_formed_rate"],
                       "truncated_rate": m["truncated_rate"]}
                if ref is None:
                    ref = s
                else:
                    # same instances in the same order, so this one IS paired
                    b = int((ref & ~s).sum())
                    c = int((~ref & s).sum())
                    row["gap"] = m["solve_rate"] - float(ref.mean())
                    row["p"] = mcnemar_exact_p(b, c)
                    print(f"      vs identity {row['gap']:+.4f}  {c}-{b}  "
                          f"paired p={row['p']:.3g}", flush=True)
                rows.append(row)

    if args.out_csv and rows:
        keys = sorted({k for r in rows for k in r})
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
