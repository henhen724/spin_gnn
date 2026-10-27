"""Model-size hyperparameter sweep for DD-Sudoku's SATReasoningTransformer,
targeting the near-trivial "O(1) uncolored nodes" regime specifically (not
the full training-time rho range) -- see STATUS.md: solve_rate plateaus at
~0.3 even at rho=1.0 (0 uncolored nodes) despite outcome_accuracy/
well_formed_rate already at 1.0, i.e. a capacity/precision ceiling on exact
24-node copying, not a difficulty-positioning issue. This sweep scales model
size (num_layers / dim_feedforward / d_model) to find the smallest config
that actually reaches ~100% solve_rate on that near-trivial end.

"O(1) uncolored nodes" is mapped to a fixed small-k grid of rho values at
the configured n_nodes (rho = (n_nodes - k) / n_nodes for k=0..4) rather
than a training-time density range -- k is a small constant independent of
n_nodes, so this is the right notion of "near-trivial" to hold fixed while
scaling the model.

Trains one (d_model, num_layers, nhead, dim_feedforward, lr) config, evals
solve_rate at each k via eval_solve_by_rho, appends a CSV row, and saves a
checkpoint.

Usage: python -m spin_gnn.sudoku_hparam_sweep --d-model 64 --num-layers 4 \\
    --dim-feedforward 512 --out-dir $SCRATCH/...
"""
import argparse
import csv
import os

from .config import SudokuReasoningConfig, SudokuReasoningModelConfig, SudokuReasoningTrainConfig
from .train_sudoku_transformer import eval_solve_by_rho, save_checkpoint, train_sudoku_model


def run_one(args):
    sudoku_cfg = SudokuReasoningConfig(
        n_nodes=args.n_nodes, degree=args.degree, n_colors=args.n_colors, rho=args.rho, rho_max=args.rho_max,
        seed=args.seed, scratchpad=args.scratchpad, hint_mode=args.hint_mode,
        shuffle_adj=args.shuffle_adj, shuffle_events=args.shuffle_events,
    )
    model_cfg = SudokuReasoningModelConfig(
        d_model=args.d_model, num_layers=args.num_layers, nhead=args.nhead, dim_feedforward=args.dim_feedforward,
        norm_first=not args.no_norm_first,
    )
    train_cfg = SudokuReasoningTrainConfig(
        n_train_instances=args.n_train_instances, n_val_instances=args.n_val_instances,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
        lr_schedule=args.lr_schedule, lr_min_factor=args.lr_min_factor,
    )
    result = train_sudoku_model(sudoku_cfg, model_cfg, train_cfg)

    o1_ks = [0, 1, 2, 3, 4]
    o1_rho_grid = [(args.n_nodes - k) / args.n_nodes for k in o1_ks]
    o1_records = eval_solve_by_rho(
        result["model"], result["vocab"], next(result["model"].parameters()).device, sudoku_cfg, o1_rho_grid,
        args.n_per_bin, args.eval_seed_start, max_new_tokens=result["model_max_len"],
    )
    for k, rec in zip(o1_ks, o1_records):
        print(f"k={k} (rho={rec['rho']:.4f}): solve_rate={rec['solve_rate']:.3f}")
    mean_o1_solve_rate = sum(rec["solve_rate"] for rec in o1_records) / len(o1_records)

    resolved_ff = result["model"].encoder.layers[0].linear1.out_features
    norm_tag = "pre" if model_cfg.norm_first else "post"
    tag = (
        f"d{args.d_model:04d}_h{args.nhead:02d}_L{args.num_layers:02d}"
        f"_ff{resolved_ff:05d}_lr{args.lr:.0e}_{norm_tag}"
        + ("_sp" if args.scratchpad else "")
        + ("_rand" if args.hint_mode == "random" else "")
        # The leak fix is a different instance distribution, not a different
        # hyperparameter, so it gets its own tag segment -- a _shuf cell must
        # never be mistaken for a comparable rerun of a sorted-order cell.
        + ("_shuf" if args.shuffle_adj else "")
        # Same argument for the within-round event order (finding 23's control):
        # a different instance distribution, so a cell that trained on permuted
        # rounds must never share a tag with one that trained on sorted rounds.
        + ("_ev" if args.shuffle_events else "")
        # Only suffixed when non-default, so every tag written before 2026-10-25
        # stays byte-identical -- the decks and the agg_eval checkpoint paths
        # address cells by this string.
        + ("_cos" if args.lr_schedule != "constant" else "")
    )
    out_dir = os.path.join(args.out_dir, tag)
    save_checkpoint(
        out_dir, sudoku_cfg, model_cfg, result["model"], result["vocab"], result["model_max_len"],
        result["baseline_accuracy"], result["solve_metrics"], result["history"], train_cfg=train_cfg,
    )

    row = {
        "d_model": args.d_model, "nhead": args.nhead, "num_layers": args.num_layers,
        "dim_feedforward": resolved_ff, "norm_first": model_cfg.norm_first, "lr": args.lr,
        "n_train_instances": args.n_train_instances, "n_val_instances": args.n_val_instances,
        "epochs": args.epochs, "n_params": sum(p.numel() for p in result["model"].parameters()),
        "baseline_accuracy": result["baseline_accuracy"],
        "final_val_accuracy": result["history"][-1]["accuracy"],
        "final_val_nll": result["history"][-1]["nll"],
        "agg_solve_rate": result["solve_metrics"]["solve_rate"],
        "agg_outcome_accuracy": result["solve_metrics"]["outcome_accuracy"],
        "mean_o1_solve_rate": mean_o1_solve_rate,
        **{f"solve_rate_k{k}": rec["solve_rate"] for k, rec in zip(o1_ks, o1_records)},
        "scratchpad": args.scratchpad,
        "hint_mode": args.hint_mode,
        "shuffle_adj": args.shuffle_adj,
        "shuffle_events": args.shuffle_events,
        "max_len": result["model_max_len"],
        "evidence_row_accuracy": result["solve_metrics"].get("evidence_row_accuracy"),
        "evidence_slot_accuracy": result["solve_metrics"].get("evidence_slot_accuracy"),
        "lr_schedule": args.lr_schedule,
        "checkpoint_dir": out_dir,
    }

    # sweep.csv is append-only so a scan can be extended a cell at a time. That
    # means the header on disk, not `row`, defines the column order: writing
    # list(row.keys()) after adding a field would silently emit ragged rows into
    # every sweep.csv written before that field existed. Reuse the existing
    # header and say out loud which fields it cannot hold.
    csv_path = os.path.join(args.out_dir, "sweep.csv")
    fieldnames, write_header = list(row.keys()), True
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            existing = next(csv.reader(f), None)
        if existing:
            fieldnames, write_header = existing, False
            dropped = [k for k in row if k not in existing]
            if dropped:
                print(f"note: {csv_path} predates {', '.join(dropped)}; "
                      f"not recorded there (the checkpoint dir and config.json have it)")
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"wrote {csv_path}: mean_o1_solve_rate={mean_o1_solve_rate:.4f} agg_solve_rate={row['agg_solve_rate']:.4f} "
          f"n_params={row['n_params']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=24)
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--n-colors", type=int, default=4)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--rho-max", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scratchpad", action="store_true",
                    help="restate each deduction's evidence before it (sudoku_tokens scratchpad grammar)")
    ap.add_argument("--shuffle-adj", action="store_true",
                    help="permute each node's neighbor list, removing the generator's "
                         "color-sorted adjacency order (see SudokuReasoningConfig.shuffle_adj); "
                         "required for any run that is measuring deduction rather than decoding")
    ap.add_argument("--shuffle-events", action="store_true",
                    help="permute the events within each DD round, removing the generator's "
                         "ascending-node-index order inside a round (see "
                         "SudokuReasoningConfig.shuffle_events); the round partition, T and "
                         "every DD predicate are unchanged, only the presentation order is")
    ap.add_argument("--hint-mode", type=str, default="removal", choices=["removal", "random"],
                    help="'random' samples hints independently so DD can fail, which is required for a "
                         "hint-density solvability transition to exist in the data at all")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dim-feedforward", type=int, default=0)
    ap.add_argument("--no-norm-first", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    # "constant" is what every run before 2026-10-25 used; keep it the default so
    # re-running an old cell reproduces it.
    ap.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    ap.add_argument("--lr-min-factor", type=float, default=0.01)
    ap.add_argument("--n-train-instances", type=int, default=5000)
    ap.add_argument("--n-val-instances", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n-per-bin", type=int, default=100)
    ap.add_argument("--eval-seed-start", type=int, default=20_000_000_000)
    ap.add_argument("--out-dir", type=str, required=True)
    parsed = ap.parse_args()
    if parsed.d_model % parsed.nhead != 0:
        raise ValueError(f"d_model={parsed.d_model} must be divisible by nhead={parsed.nhead}")
    run_one(parsed)
