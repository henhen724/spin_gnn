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
from .sat_transformer2 import ARCH_MODES
from .train_sudoku_transformer import eval_solve_by_rho, save_checkpoint, train_sudoku_model

ARCH_TAG = {"full": "", "linear": "_lin", "rf_mlp": "_rfm", "rf_all": "_rfa", "no_mlp": "_noff",
            # A tied cell has the same (d_model, nhead, L, ff) as its untied
            # twin and a ninth of the parameters, so it needs its own segment
            # for the reason every other one here exists: it must never share a
            # directory name with J2.
            "tied_no_mlp": "_noff_tied"}
assert set(ARCH_TAG) == set(ARCH_MODES), "every arch_mode needs a tag segment"


def run_one(args):
    sudoku_cfg = SudokuReasoningConfig(
        n_nodes=args.n_nodes, degree=args.degree, n_colors=args.n_colors, rho=args.rho, rho_max=args.rho_max,
        seed=args.seed, scratchpad=args.scratchpad, hint_mode=args.hint_mode,
        shuffle_adj=args.shuffle_adj, shuffle_events=args.shuffle_events,
        fall=args.fall,
    )
    model_cfg = SudokuReasoningModelConfig(
        d_model=args.d_model, num_layers=args.num_layers, nhead=args.nhead, dim_feedforward=args.dim_feedforward,
        norm_first=not args.no_norm_first, arch_mode=args.arch_mode,
        n_blocks=args.n_blocks,
    )
    train_cfg = SudokuReasoningTrainConfig(
        n_train_instances=args.n_train_instances, n_val_instances=args.n_val_instances,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
        lr_schedule=args.lr_schedule, lr_min_factor=args.lr_min_factor,
        warmup_epochs=args.warmup_epochs, max_len_margin=args.max_len_margin,
        deterministic=args.deterministic,
        determinacy_weight=args.determinacy_weight,
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
        # Same argument again, and the strongest case of the three: the fall
        # constraint changes the GRAPH ensemble, so a fall cell's instances
        # are not drawn from the same distribution at all and must never pool
        # with a default cell (it is in agg_eval's INSTANCE_KEYS for exactly
        # that reason).
        + ("_fall" if args.fall else "")
        # The one segment here that is about the MODEL rather than the
        # instances, and it has to be in the tag for the same reason: a cell
        # whose MLP is linear, frozen or absent is a different architecture at
        # identical (d_model, nhead, L, ff) and must never share a directory
        # name with the full one. Only present when non-default, so every tag
        # written before 2026-11-04 is unchanged.
        + ARCH_TAG[args.arch_mode]
        # Only suffixed when non-default, so every tag written before 2026-10-25
        # stays byte-identical -- the decks and the agg_eval checkpoint paths
        # address cells by this string.
        + ("_cos" if args.lr_schedule != "constant" else "")
        # Same rule as _cos: only present when non-default, so no tag written
        # before 2026-10-28 changes, and a warmed-up cell can never collide
        # with the cold cell it is meant to be compared against.
        + (f"_w{args.warmup_epochs:02d}" if args.warmup_epochs else "")
        # Same rule again, and the first case where it protects a REPLICATE
        # rather than a variant: the seed moves the weight init and the batch
        # order, so a second-seed cell is a different draw of the same training
        # procedure and must not share a tag with the first. Without this,
        # sudoku_agg_eval and sudoku_depth_eval see one basename for two models
        # and can only tell them apart by their parent directory. Only present
        # when non-default, so every tag written before 2026-11-01 is unchanged.
        # Same rule, and the same reason as ARCH_TAG: k distinct blocks at
        # identical (d_model, nhead, L, ff, arch_mode) is a different
        # architecture -- k=2 holds twice the weights of k=1 and composes a
        # different map -- so it must never share a directory name with the
        # shared-block cell. Only present when k != 1, so every tag written
        # before 2026-11-11 is byte-identical.
        + (f"_k{args.n_blocks:02d}" if args.n_blocks != 1 else "")
        # Same rule once more. A pinned-kernel cell trains the same
        # architecture on the same data from the same init, and still ends at
        # different weights than an unpinned one at the same seed -- finding
        # 66b is the measurement of exactly that -- so the two must not share a
        # directory name. Only present when the flag is on, so every tag
        # written before 2026-11-13 is byte-identical.
        + ("_det" if args.deterministic else "")
        # Same rule once more, and this is the segment that makes stage X a
        # stage at all: the architecture, the parameter count and the state
        # dict are S1's exactly, so the objective is the ONLY thing a
        # directory name could carry. Without it a stage-X cell and its
        # stage-S control collide on disk. Only present when the term is on,
        # so every tag written before 2026-11-13 is byte-identical.
        + (f"_dw{args.determinacy_weight:g}" if args.determinacy_weight else "")
        + (f"_s{args.seed:02d}" if args.seed else "")
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
        "arch_mode": args.arch_mode,
        "n_blocks": args.n_blocks,
        "n_trainable_params": result["model"].trainable_parameter_count(),
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
        "fall": args.fall,
        "max_len": result["model_max_len"],
        "evidence_row_accuracy": result["solve_metrics"].get("evidence_row_accuracy"),
        "evidence_slot_accuracy": result["solve_metrics"].get("evidence_slot_accuracy"),
        "lr_schedule": args.lr_schedule,
        "warmup_epochs": args.warmup_epochs,
        "deterministic": args.deterministic,
        "determinacy_weight": args.determinacy_weight,
        "final_determinacy_auroc": result["history"][-1].get("determinacy_auroc"),
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
    ap.add_argument("--fall", action="store_true",
                    help="draw the planted colouring as a FALL colouring -- every node has a "
                         "neighbour of every other colour (SudokuReasoningConfig.fall). Removes "
                         "the structural floor under rho_min (0.47 -> 0.33 at N=48) that "
                         "otherwise keeps every run in the fast phase.")
    ap.add_argument("--hint-mode", type=str, default="removal", choices=["removal", "random"],
                    help="'random' samples hints independently so DD can fail, which is required for a "
                         "hint-density solvability transition to exist in the data at all")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dim-feedforward", type=int, default=0)
    ap.add_argument("--no-norm-first", action="store_true")
    ap.add_argument("--n-blocks", type=int, default=1,
                    help="k: how many DISTINCT blocks a tied stack cycles through, each "
                         "applied num_layers/k times. 1 (default) is the shared-block cell; "
                         "k must divide num_layers and is only valid with "
                         "--arch-mode tied_no_mlp. The network is still one map iterated "
                         "num_layers/k times, with k primitive maps inside each step.")
    ap.add_argument("--arch-mode", type=str, default="full", choices=list(ARCH_MODES),
                    help="how much machinery each block keeps: full (trained ReLU MLP, the "
                         "default and every run before 2026-11-04), linear (identity "
                         "activation), rf_mlp (linear1 frozen at init, linear2 trains), "
                         "rf_all (everything frozen except the readout -- convex), no_mlp "
                         "(attention only). See SudokuReasoningModelConfig.arch_mode.")
    ap.add_argument("--lr", type=float, default=1e-3)
    # "constant" is what every run before 2026-10-25 used; keep it the default so
    # re-running an old cell reproduces it.
    ap.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    ap.add_argument("--lr-min-factor", type=float, default=0.01)
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="linear lr ramp over the first N epochs before --lr-schedule takes over; "
                         "0 (default) reproduces every run before 2026-10-28")
    ap.add_argument("--deterministic", action="store_true",
                    help="pin the GPU kernels' reduction order so two runs of this cell at "
                         "one seed are the same run (torch.use_deterministic_algorithms, "
                         "cudnn.benchmark off). Needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in "
                         "the launcher's environment and costs speed. OFF by default, so "
                         "every run up to stage W is reproduced exactly.")
    ap.add_argument("--determinacy-weight", type=float, default=0.0,
                    help="stage X: weight on a multi-label BCE training the readout's NODE "
                         "logits, at each position about to emit a node, to say which "
                         "still-uncoloured nodes are deducible now. Adds no parameters. "
                         "0.0 (default) reproduces every run up to stage W exactly, "
                         "including the batch order.")
    ap.add_argument("--n-train-instances", type=int, default=5000)
    ap.add_argument("--n-val-instances", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--max-len-margin", type=int, default=-1,
                    help="generation headroom above the longest training sequence; "
                         "-1 = auto (2*n_nodes+2), 0 = pre-2026-11-02 behaviour")
    ap.add_argument("--n-per-bin", type=int, default=100)
    ap.add_argument("--eval-seed-start", type=int, default=20_000_000_000)
    ap.add_argument("--out-dir", type=str, required=True)
    parsed = ap.parse_args()
    if parsed.d_model % parsed.nhead != 0:
        raise ValueError(f"d_model={parsed.d_model} must be divisible by nhead={parsed.nhead}")
    run_one(parsed)
