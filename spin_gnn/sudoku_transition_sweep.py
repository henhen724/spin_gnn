"""Traces solve_rate across the easy (high rho, shallow) <-> hard (low rho,
deep) DD-Sudoku difficulty transition, for a handful of checkpoints spaced
across training -- the point is to see the whole learning curve shift, not
just the final epoch's number (which train_sudoku_transformer.py already
reports via `solve_rate` on the training-time rho range).

Trains once with checkpoint_epochs snapshots (train_sudoku_transformer.
train_sudoku_model), then for each snapshot, evaluates eval_solve_by_rho
over a fixed rho grid with pinned-density bins (each bin's instances all
target that one rho, not the training [rho, rho_max] spread), and writes
one CSV row per (epoch, rho) pair.

Two different transitions, selected by --hint-mode:
  "removal" (default) -- the easy/hard *difficulty* transition. Instances are
    resolvable by construction at every density, so rho sets reasoning depth
    only and there is no solvability threshold to see.
  "random" -- the hint-density *solvability* transition. Hints are drawn
    independently, so DD itself fails below threshold; each output row then
    carries dd_resolved_frac/dd_frac_recovered measured on the very same
    instances, which is the reference curve the model is compared against.

Usage: python -m spin_gnn.sudoku_transition_sweep --out-csv $SCRATCH/....csv
"""
import argparse
import csv

from .config import SudokuReasoningConfig, SudokuReasoningModelConfig, SudokuReasoningTrainConfig
from .sat_transformer2 import SATReasoningTransformer
from .train_sudoku_transformer import eval_solve_by_rho, train_sudoku_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=24)
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--n-colors", type=int, default=4)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--rho-max", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=1)
    ap.add_argument("--dim-feedforward", type=int, default=0)
    ap.add_argument("--no-norm-first", action="store_true")
    ap.add_argument("--n-train-instances", type=int, default=2000)
    ap.add_argument("--n-val-instances", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--checkpoint-epochs", type=str, default="0,5,10,20,30,40,50")
    ap.add_argument("--rho-grid", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
    ap.add_argument("--n-per-bin", type=int, default=50)
    ap.add_argument("--eval-seed-start", type=int, default=10_000_000_000)
    ap.add_argument("--hint-mode", type=str, default="removal", choices=["removal", "random"],
                    help="'random' samples hints independently so DD can fail; required for a "
                         "hint-density solvability transition to exist in the data at all")
    ap.add_argument("--scratchpad", action="store_true",
                    help="restate each deduction's evidence before it (sudoku_tokens scratchpad grammar)")
    ap.add_argument("--out-csv", type=str, required=True)
    args = ap.parse_args()

    checkpoint_epochs = [int(e) for e in args.checkpoint_epochs.split(",")]
    rho_grid = [float(r) for r in args.rho_grid.split(",")]

    sudoku_cfg = SudokuReasoningConfig(
        n_nodes=args.n_nodes, degree=args.degree, n_colors=args.n_colors, rho=args.rho, rho_max=args.rho_max,
        seed=args.seed, hint_mode=args.hint_mode, scratchpad=args.scratchpad,
    )
    model_cfg = SudokuReasoningModelConfig(
        d_model=args.d_model, num_layers=args.num_layers, nhead=args.nhead, dim_feedforward=args.dim_feedforward,
        norm_first=not args.no_norm_first,
    )
    train_cfg = SudokuReasoningTrainConfig(
        n_train_instances=args.n_train_instances, n_val_instances=args.n_val_instances,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
    )
    result = train_sudoku_model(sudoku_cfg, model_cfg, train_cfg, checkpoint_epochs=checkpoint_epochs)
    vocab = result["vocab"]
    device = next(iter(result["model"].parameters())).device

    rows = []
    for epoch in sorted(result["checkpoints"]):
        model = SATReasoningTransformer(
            vocab.size, d_model=model_cfg.d_model, num_layers=model_cfg.num_layers, nhead=model_cfg.nhead,
            max_len=result["model_max_len"], dim_feedforward=model_cfg.dim_feedforward or None,
            norm_first=model_cfg.norm_first,
        ).to(device)
        model.load_state_dict(result["checkpoints"][epoch])
        model.eval()
        records = eval_solve_by_rho(
            model, vocab, device, sudoku_cfg, rho_grid, args.n_per_bin,
            args.eval_seed_start + epoch * 1_000_000, max_new_tokens=result["model_max_len"],
        )
        for r in records:
            rows.append({"epoch": epoch, **r})
        print(f"checkpoint epoch={epoch}: " + ", ".join(f"rho={r['rho']:.2f}:sr={r['solve_rate']:.2f}" for r in records))

    # Dynamic fieldnames: eval_solve_by_rho's dict grew DD-ground-truth and
    # (under --scratchpad) evidence-accuracy keys, and hardcoding the list
    # would silently drop them.
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
