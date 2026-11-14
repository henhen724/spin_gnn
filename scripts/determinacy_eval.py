"""X3's control: how well does a checkpoint's readout already answer "which
open nodes are deducible here", on instances no cell trained on?

WHY THIS EXISTS RATHER THAN READING history.json. The per-epoch
`determinacy_auroc` a stage-X cell records is measured on ITS OWN validation
set -- `build_instances(cfg, n_val, seed + n_train)`, so seed 0, 1 and 2 see
three DIFFERENT 300-instance draws -- and a stage-S cell, trained before the
term existed, records no such column at all. Neither is a control. This
script builds ONE fixed instance set from a fixed seed and scores every
checkpoint handed to it on exactly those instances, so stage X and stage S
differ in the model and in nothing else.

AND WHY THE CONTROL IS NOT 0.5. The main loss already lifts the logit of the
node about to be emitted, and the node about to be emitted is always
deducible. So a cell that never saw the determinacy term is expected to score
ABOVE chance here, and the honest question X3 asks is whether the term buys
margin over that -- not whether the number beats a coin.

Usage:
    python3 -m scripts.determinacy_eval \
        --checkpoint $SCRATCH/spin_gnn/<run>/<tag> [...] \
        --n-instances 600 --out-csv .../determinacy.csv
"""
import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spin_gnn.sat_dataset2 import SATTokenDataset
from spin_gnn.sudoku_determinacy import determinacy_targets, summarize
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import (
    build_instances, evaluate_determinacy, load_checkpoint,
)
from scripts.rho_transition_eval import cfg_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--n-instances", type=int, default=600,
                    help="held-out instances. 600 at 13 scored positions and 7.5 "
                         "open nodes each is about 58,000 scored cells, enough "
                         "that the AUROC's own noise is well under the effect X3 "
                         "is looking for.")
    ap.add_argument("--seed-start", type=int, default=70_000_000_000,
                    help="fixed, and deliberately far from any training or "
                         "rho-grid seed, so every checkpoint sees the SAME "
                         "instances and none of them is scored on something it "
                         "trained on")
    ap.add_argument("--out-csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  n_instances={args.n_instances}  seed_start={args.seed_start}")

    rows = []
    shared = None
    for ckpt in args.checkpoint:
        model, vocab, config = load_checkpoint(ckpt, device=device)
        sudoku_cfg = cfg_from_checkpoint(config)
        name = os.path.basename(os.path.normpath(ckpt))
        # The instance distribution is read from the checkpoint, exactly as
        # rho_transition_eval does it. If two checkpoints disagree on it they
        # are not scoring the same task and must not be compared, so say so
        # rather than printing two numbers side by side.
        key = (sudoku_cfg.n_nodes, sudoku_cfg.degree, sudoku_cfg.n_colors,
               sudoku_cfg.fall, sudoku_cfg.shuffle_adj, sudoku_cfg.hint_mode,
               sudoku_cfg.scratchpad, sudoku_cfg.shuffle_events)
        if shared is None:
            shared = key
            instances = build_instances(sudoku_cfg, args.n_instances, args.seed_start)
            tokens = [encode_instance(adj, trace, vocab) for adj, _c, _h, trace in instances]
            aux = [determinacy_targets(adj, trace, vocab, tok)
                   for (adj, _c, _h, trace), tok in zip(instances, tokens)]
            stats = summarize([t for _p, t in aux])
            print("instances: %d  positive_rate %.4f  scored_fraction %.4f  "
                  "(%s positive / %s negative / %s masked)"
                  % (args.n_instances, stats["positive_rate"], stats["scored_fraction"],
                     format(stats["positives"], ","), format(stats["negatives"], ","),
                     format(stats["masked"], ",")))
            dataset = SATTokenDataset(tokens, max_len=config["max_len"], aux=aux)
            indices = list(range(len(dataset)))
        elif key != shared:
            raise SystemExit(
                "%s was trained on a different instance distribution (%s) than "
                "the first checkpoint (%s); these AUROCs are not comparable"
                % (name, key, shared))

        out = evaluate_determinacy(model, dataset, indices, device, vocab)
        dw = float(config.get("determinacy_weight", 0.0) or 0.0)
        print("\n=== %s ===" % name)
        print("  arch=%s L=%d d_model=%d seed=%s lambda=%g%s"
              % (config.get("arch_mode", "full"), config["num_layers"],
                 config["d_model"], config.get("seed"), dw,
                 "  <- stage-S control (the term was never on)" if dw == 0 else ""))
        print("  determinacy AUROC %.4f   BCE %.4f"
              % (out["determinacy_auroc"], out["determinacy_bce"]))
        rows.append({
            "checkpoint": name, "path": ckpt,
            "arch_mode": config.get("arch_mode", "full"),
            "num_layers": config["num_layers"], "d_model": config["d_model"],
            "seed": config.get("seed"), "determinacy_weight": dw,
            "n_instances": args.n_instances, "seed_start": args.seed_start,
            "determinacy_auroc": out["determinacy_auroc"],
            "determinacy_bce": out["determinacy_bce"],
        })

    if len(rows) > 1:
        print("\n=== X3 reads these two groups, on the identical instances above ===")
        for dw in sorted({r["determinacy_weight"] for r in rows}):
            group = sorted(r["determinacy_auroc"] for r in rows
                           if r["determinacy_weight"] == dw)
            print("  lambda=%-4g  AUROC %s" % (dw, " / ".join("%.4f" % v for v in group)))

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nwrote %s" % args.out_csv)


if __name__ == "__main__":
    main()
