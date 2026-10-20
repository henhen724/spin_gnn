"""Next-token-prediction training for a transformer over
spin_gnn.sudoku_trace's event-based DD-Sudoku reasoning traces.

Mirrors train_sat_transformer2.py's structure closely (same model class,
same generic dataset/loss code, same seeding + eval philosophy) -- applying
what the k-SAT v2 work established about tokenization directly to a new
problem rather than rediscovering it:
  - event-based trace (what got deduced, in what round) instead of a
    full-board-snapshot-per-step encoding (ksat.py vs ksat2.py's lesson).
  - explicit structural tags (BEGIN_ROUND) instead of leaving state
    transitions for the model to infer purely by position/attention
    (ksat2.py's SATISFIED/REACTIVATED lesson).
  - pre-norm (norm_first=True) from the start, not discovered after a
    depth-collapse failure (SudokuReasoningModelConfig's default, unlike
    SATReasoningModelConfig's norm_first=False).
  - torch.manual_seed(sudoku_cfg.seed) before model construction from the
    start, not discovered after a seed-variance investigation.

Reports, alongside train/val cross-entropy and per-token accuracy:
- `baseline_accuracy`: always-predict-the-majority-token accuracy, for
  context.
- `solve_rate`/`outcome_accuracy`/`well_formed_rate`: prompt with
  hints+graph only, let the model freely generate its own reasoning trace
  and final answer, then check it against the true instance -- the actual
  end-to-end capability, not just next-token accuracy under teacher
  forcing (which can look deceptively high while free-running generation
  fails, per the k-SAT copy_vs_delta_breakdown finding).

Run via Slurm (scripts/slurm/sudoku2_*.sbatch), not the Sherlock login node.
"""
import copy
import json
import os
from dataclasses import replace

import numpy as np
import torch

from .config import SudokuReasoningConfig, SudokuReasoningModelConfig, SudokuReasoningTrainConfig
from .sat_dataset2 import SATTokenDataset, iterate_token_batches
from .sat_transformer2 import SATReasoningTransformer, sat2_accuracy, sat2_nll_loss
from .sudoku_dd import planted_regular_graph, removal_path
from .sudoku_tokens import Vocab, check_evidence, check_solution, decode, encode_instance
from .sudoku_trace import solve_with_dd_trace

# torch defaults to one thread per visible CPU, which is a severe pessimization
# for a model this tiny (e.g. 64 threads on a Sherlock login node for a d_model=16
# batch-1 forward pass): per-call thread-pool synchronization overhead dwarfs the
# actual compute, observed as a ~200x slowdown (0.9ms -> 190ms/forward) that turns
# solve_rate's autoregressive generate() loop into a multi-minute stall. Capped
# once here, at import time, since every entry point (training script and tests)
# goes through this module.
torch.set_num_threads(min(4, os.cpu_count() or 1))


def build_vocab(sudoku_cfg: SudokuReasoningConfig):
    return Vocab(
        sudoku_cfg.n_nodes, sudoku_cfg.n_colors, sudoku_cfg.degree,
        scratchpad=getattr(sudoku_cfg, "scratchpad", False),
    )


def build_instances(sudoku_cfg: SudokuReasoningConfig, n_instances, seed_start):
    """Each element: (adj, colors, hints, trace) -- trace is
    sudoku_trace.solve_with_dd_trace's return dict.

    Hints are sampled via the slides' own protocol (sudoku_dd.removal_path:
    start fully hinted, remove hints one at a time in a random order, only
    accepting a removal if DD still recovers the full coloring without it) --
    not independent random hint sampling. This guarantees every instance is
    fully resolvable by construction (there is no "unresolved" case, exactly
    matching the slides' methodology), varying only in reasoning depth. The
    target density `rho` (drawn uniformly in [sudoku_cfg.rho, rho_max]) picks
    where along that path to stop; if a path's own minimal density (rho_min)
    is above the target (not every density is reachable by every path), it
    falls back to that path's sparsest reachable point instead."""
    instances = []
    for i in range(n_instances):
        # planted_regular_graph's multi-edge removal occasionally fails to find a
        # simple pairing for a given seed (more often at small N); retry with a
        # perturbed seed rather than letting one bad draw kill the whole run.
        for attempt in range(20):
            rng = np.random.default_rng(seed_start + i + attempt * 100_000_000)
            try:
                adj, colors = planted_regular_graph(
                    sudoku_cfg.n_nodes, sudoku_cfg.degree, sudoku_cfg.n_colors, rng,
                )
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError(f"could not build a planted graph for instance {i} after 20 attempts")

        rho_rng = np.random.default_rng(seed_start + i + 10_000_000)
        target_rho = (
            rho_rng.uniform(sudoku_cfg.rho, sudoku_cfg.rho_max)
            if sudoku_cfg.rho_max > sudoku_cfg.rho else sudoku_cfg.rho
        )
        path_rng = np.random.default_rng(seed_start + i + 20_000_000)
        path = removal_path(adj, colors, sudoku_cfg.n_colors, path_rng, snapshot_rhos=(target_rho,))
        t_final = path["snapshots"].get(target_rho, path["final_t"])
        hints = {int(v): int(colors[v]) for v in np.flatnonzero(t_final == 0)}

        trace = solve_with_dd_trace(adj, colors, hints, sudoku_cfg.n_colors)
        assert trace["resolved"], "removal_path-derived hint sets must always be fully resolvable"
        instances.append((adj, colors, hints, trace))
    return instances


def baseline_accuracy(dataset: SATTokenDataset, vocab_size):
    counts = np.zeros(vocab_size, dtype=np.int64)
    for seq in dataset.sequences:
        target = seq[1:]
        counts += np.bincount(target, minlength=vocab_size)
    return float(counts.max() / counts.sum())


def evaluate(model, dataset, indices, device, batch_size=32):
    model.eval()
    total_loss, total_n = 0.0, 0
    total_correct, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y, mask in iterate_token_batches(dataset, indices, batch_size, shuffle=False):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, valid_mask=mask)
            loss = sat2_nll_loss(logits, y, mask)
            acc = sat2_accuracy(logits, y, mask)
            n_valid = mask.sum().item()
            total_loss += loss.item() * n_valid
            total_correct += acc * n_valid
            total_tokens += n_valid
            total_n += n_valid
    return {"nll": total_loss / total_n, "accuracy": total_correct / total_tokens}


def generate(model, prompt_tokens, vocab, max_new_tokens, device="cpu", temperature=0.0, rng=None):
    """Autoregressively continue `prompt_tokens` (everything through
    BEGIN_REASONING) for up to max_new_tokens steps. Stops early right
    after SOLVED, or once UNRESOLVED has been generated and at least
    vocab.n_nodes further tokens have followed (the unresolved-node listing
    can never exceed one token per node)."""
    model.eval()
    rng = rng or np.random.default_rng()
    max_len = model.pos_embed.shape[0]
    seq = torch.from_numpy(prompt_tokens.astype(np.int64)).unsqueeze(0).to(device)
    unresolved_seen_at = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if seq.shape[1] >= max_len:
                break
            logits = model(seq)[0, -1]
            if temperature <= 0:
                next_id = int(logits.argmax(dim=-1).item())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
            seq = torch.cat([seq, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == vocab.SOLVED:
                break
            if next_id == vocab.UNRESOLVED and unresolved_seen_at is None:
                unresolved_seen_at = seq.shape[1]
            if unresolved_seen_at is not None and seq.shape[1] - unresolved_seen_at >= vocab.n_nodes:
                break
    return seq[0].cpu().numpy()


def solve_rate(model, instances, vocab, device, max_new_tokens, temperature=0.0, rng=None):
    """instances: list of (adj, colors, hints, trace) tuples. For each,
    prompts the model with everything through BEGIN_REASONING (hints +
    true graph) and lets it freely generate the rest -- can it derive its
    own correct deduction trace and final answer, not just predict one
    token given perfect ground-truth history.

    Returns a dict: solve_rate (fraction reaching a fully correct, checked
    solution), outcome_accuracy (fraction whose SOLVED/UNRESOLVED call is
    right regardless of whether the colors it lists are valid),
    well_formed_rate (fraction that parse without a grammar violation)."""
    rng = rng or np.random.default_rng()
    n_solved = n_outcome_correct = n_well_formed = 0
    ev_rows = ev_rows_correct = ev_slots = ev_slots_correct = 0
    for adj, colors, hints, trace in instances:
        full_tokens = encode_instance(adj, trace, vocab)
        reasoning_start = int(np.flatnonzero(full_tokens == vocab.BEGIN_REASONING)[0]) + 1
        prompt = full_tokens[:reasoning_start]
        generated = generate(model, prompt, vocab, max_new_tokens, device=device, temperature=temperature, rng=rng)
        decoded = decode(generated, vocab)

        if decoded["well_formed"]:
            n_well_formed += 1
        true_outcome = "SOLVED" if trace["resolved"] else "UNRESOLVED"
        if decoded["outcome"] == true_outcome:
            n_outcome_correct += 1
            if check_solution(decoded, colors, vocab.n_nodes):
                n_solved += 1

        if getattr(vocab, "scratchpad", False):
            ev = check_evidence(decoded)
            ev_rows += ev["n_rows"]
            ev_rows_correct += ev["n_rows_correct"]
            ev_slots += ev["n_slots"]
            ev_slots_correct += ev["n_slots_correct"]

    n = len(instances)
    metrics = {
        "solve_rate": n_solved / n if n else None,
        "outcome_accuracy": n_outcome_correct / n if n else None,
        "well_formed_rate": n_well_formed / n if n else None,
        "n_instances": n,
    }
    if getattr(vocab, "scratchpad", False):
        # Separates "can it copy the evidence it already wrote down" from "is
        # the deduction right" -- if solve_rate stays flat, this says which
        # half is at fault.
        metrics["evidence_row_accuracy"] = ev_rows_correct / ev_rows if ev_rows else None
        metrics["evidence_slot_accuracy"] = ev_slots_correct / ev_slots if ev_slots else None
        metrics["n_evidence_rows"] = ev_rows
    return metrics


def eval_solve_by_rho(
    model, vocab, device, sudoku_cfg: SudokuReasoningConfig, rho_grid, n_per_bin, seed_start, max_new_tokens,
):
    """solve_rate, binned by a fixed target hint density rather than the
    training-time [rho, rho_max] spread -- each bin's instances are built
    with rho_max pinned to rho (per-instance target density == the bin's
    rho, not a range), so a single number per bin isolates "solve accuracy
    at this difficulty" instead of averaging over a mix. Used to trace the
    easy (high rho, shallow) <-> hard (low rho, deep) transition for one
    checkpoint; call once per checkpoint to get a family of curves.

    Returns a list of dicts, one per rho in rho_grid, each the dict
    returned by solve_rate() plus a "rho" key."""
    records = []
    for rho in rho_grid:
        bin_cfg = replace(sudoku_cfg, rho=float(rho), rho_max=float(rho))
        instances = build_instances(bin_cfg, n_per_bin, seed_start)
        metrics = solve_rate(model, instances, vocab, device, max_new_tokens)
        records.append({"rho": float(rho), **metrics})
    return records


def train_sudoku_model(
    sudoku_cfg: SudokuReasoningConfig, model_cfg: SudokuReasoningModelConfig, train_cfg: SudokuReasoningTrainConfig,
    checkpoint_epochs=(),
):
    device = train_cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(sudoku_cfg.seed)
    vocab = build_vocab(sudoku_cfg)

    train_instances = build_instances(sudoku_cfg, train_cfg.n_train_instances, sudoku_cfg.seed)
    val_instances = build_instances(sudoku_cfg, train_cfg.n_val_instances, sudoku_cfg.seed + train_cfg.n_train_instances)

    train_tokens = [encode_instance(adj, trace, vocab) for adj, _, _, trace in train_instances]
    val_tokens = [encode_instance(adj, trace, vocab) for adj, _, _, trace in val_instances]

    max_len = train_cfg.max_len or None
    train_dataset = SATTokenDataset(train_tokens, max_len=max_len)
    val_dataset = SATTokenDataset(val_tokens, max_len=max_len)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"empty dataset: train={len(train_dataset)}, val={len(val_dataset)}")
    model_max_len = max(seq.shape[0] for seq in train_dataset.sequences + val_dataset.sequences)

    baseline_acc = baseline_accuracy(val_dataset, vocab.size)
    print(f"vocab_size={vocab.size} majority-class baseline accuracy (val): {baseline_acc:.4f}")

    model = SATReasoningTransformer(
        vocab.size, d_model=model_cfg.d_model, num_layers=model_cfg.num_layers,
        nhead=model_cfg.nhead, max_len=model_max_len,
        dim_feedforward=model_cfg.dim_feedforward or None, norm_first=model_cfg.norm_first,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    rng = np.random.default_rng(sudoku_cfg.seed)

    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))
    checkpoint_epochs = set(checkpoint_epochs)
    checkpoints = {}

    def snapshot():
        return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

    if 0 in checkpoint_epochs:
        checkpoints[0] = snapshot()

    history = []
    for epoch in range(train_cfg.epochs):
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        for x, y, mask in iterate_token_batches(train_dataset, train_indices, train_cfg.batch_size, rng=rng):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, valid_mask=mask)
            loss = sat2_nll_loss(logits, y, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            n_valid = mask.sum().item()
            epoch_loss += loss.item() * n_valid
            epoch_n += n_valid

        val_metrics = evaluate(model, val_dataset, val_indices, device, batch_size=train_cfg.batch_size)
        record = {"epoch": epoch, "train_nll": epoch_loss / epoch_n, **val_metrics}
        history.append(record)
        print(f"epoch {epoch}: train_nll={record['train_nll']:.4f} val_nll={val_metrics['nll']:.4f} "
              f"val_accuracy={val_metrics['accuracy']:.4f}")
        if (epoch + 1) in checkpoint_epochs:
            checkpoints[epoch + 1] = snapshot()

    solve_metrics = solve_rate(model, val_instances, vocab, device, max_new_tokens=model_max_len)
    print(f"solve_rate={solve_metrics['solve_rate']:.4f} outcome_accuracy={solve_metrics['outcome_accuracy']:.4f} "
          f"well_formed_rate={solve_metrics['well_formed_rate']:.4f}")

    return {
        "model": model, "vocab": vocab, "history": history, "baseline_accuracy": baseline_acc,
        "solve_metrics": solve_metrics, "model_max_len": model_max_len, "val_instances": val_instances,
        "checkpoints": checkpoints,
    }


def save_checkpoint(out_dir, sudoku_cfg, model_cfg, model, vocab, model_max_len, baseline_acc, solve_metrics, history):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    config = {
        "n_nodes": sudoku_cfg.n_nodes, "degree": sudoku_cfg.degree, "n_colors": sudoku_cfg.n_colors,
        "rho": sudoku_cfg.rho, "rho_max": sudoku_cfg.rho_max, "vocab_size": vocab.size,
        "scratchpad": getattr(sudoku_cfg, "scratchpad", False),
        "d_model": model_cfg.d_model, "num_layers": model_cfg.num_layers, "nhead": model_cfg.nhead,
        "dim_feedforward": model.encoder.layers[0].linear1.out_features, "norm_first": model.norm_first,
        "max_len": model_max_len, "baseline_accuracy": baseline_acc, "solve_metrics": solve_metrics,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


def load_checkpoint(out_dir, device="cpu"):
    with open(os.path.join(out_dir, "config.json")) as f:
        config = json.load(f)
    vocab = Vocab(
        config["n_nodes"], config["n_colors"], config["degree"],
        scratchpad=config.get("scratchpad", False),
    )
    model = SATReasoningTransformer(
        vocab.size, d_model=config["d_model"], num_layers=config["num_layers"],
        nhead=config["nhead"], max_len=config["max_len"], dim_feedforward=config.get("dim_feedforward"),
        norm_first=config.get("norm_first", True),
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(out_dir, "model.pt"), map_location=device, weights_only=True))
    model.eval()
    return model, vocab, config


if __name__ == "__main__":
    import argparse

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
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    sudoku_cfg = SudokuReasoningConfig(
        n_nodes=args.n_nodes, degree=args.degree, n_colors=args.n_colors, rho=args.rho, rho_max=args.rho_max,
        seed=args.seed,
    )
    model_cfg = SudokuReasoningModelConfig(
        d_model=args.d_model, num_layers=args.num_layers, nhead=args.nhead, dim_feedforward=args.dim_feedforward,
        norm_first=not args.no_norm_first,
    )
    train_cfg = SudokuReasoningTrainConfig(
        n_train_instances=args.n_train_instances, n_val_instances=args.n_val_instances,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
    )
    result = train_sudoku_model(sudoku_cfg, model_cfg, train_cfg)
    if args.out_dir:
        save_checkpoint(
            args.out_dir, sudoku_cfg, model_cfg, result["model"], result["vocab"], result["model_max_len"],
            result["baseline_accuracy"], result["solve_metrics"], result["history"],
        )
