"""One-off diagnostic: for the best qk-sweep checkpoint, teacher-force each
satisfiable val instance up through the second-to-last token (exactly like
train_sat_transformer.satisfying_completion_rate) and split the model's
single next-token assignment prediction into two groups of coordinates:

- "copy": variables unchanged between the previous assignment token in
  context and the true final assignment -- getting these right requires
  only locating the most recent assignment token via attention and copying
  its values forward, no new reasoning.
- "delta": the variable(s) that actually changed on the final step --
  requires correctly deriving the unit-propagation/branching/backtrack
  result, real reasoning, not copying.

Answers: is the ~1-3% joint success rate low because copying itself often
fails (small per-coordinate slips compounding over ~23 coordinates), or
because the model reliably copies but fails the one genuinely hard
coordinate?
"""
import os

import numpy as np
import torch

from spin_gnn.config import SATConfig, SATTrainConfig
from spin_gnn.train_sat_transformer import build_traces, load_checkpoint

CHECKPOINT_DIR = os.path.join(
    os.environ["SCRATCH"], "spin_gnn", "sat_hparam_sweep", "d0256_h16_L04_ff01024_qk0512"
)


def copy_vs_delta_breakdown(model, traces, n_vars, device):
    model.eval()
    max_len = model.pos_embed.shape[0]
    sat_traces = [
        t for t in traces
        if t["satisfiable"] and 2 <= t["tokens"].shape[0] and t["tokens"].shape[0] - 1 <= max_len
    ]
    if not sat_traces:
        return None

    copy_correct = copy_total = 0
    copy_decided_correct = copy_decided_total = 0  # copy AND true value != 0 (already committed)
    copy_unassigned_correct = copy_unassigned_total = 0  # copy AND true value == 0 (never touched)
    delta_correct = delta_total = 0
    copy_all_correct = 0
    full_exact_match = 0
    n_changed_hist = {}

    with torch.no_grad():
        for trace in sat_traces:
            tokens = trace["tokens"]
            x = torch.from_numpy(tokens[:-1]).unsqueeze(0).to(device)
            logits = model(x)
            pred = logits[0, -1].argmax(dim=-1).cpu().numpy() - 1
            pred_assignment = pred[:n_vars].astype(np.int8)

            final_assignment = tokens[-1][:n_vars].astype(np.int8)
            prev_idx = np.flatnonzero(tokens[:-1, -1] < 0)[-1]
            prev_assignment = tokens[prev_idx][:n_vars].astype(np.int8)

            changed = final_assignment != prev_assignment
            n_changed_hist[int(changed.sum())] = n_changed_hist.get(int(changed.sum()), 0) + 1
            copy_mask = ~changed

            copy_correct += int((pred_assignment[copy_mask] == final_assignment[copy_mask]).sum())
            copy_total += int(copy_mask.sum())

            decided_mask = copy_mask & (final_assignment != 0)
            unassigned_mask = copy_mask & (final_assignment == 0)
            copy_decided_correct += int((pred_assignment[decided_mask] == final_assignment[decided_mask]).sum())
            copy_decided_total += int(decided_mask.sum())
            copy_unassigned_correct += int((pred_assignment[unassigned_mask] == final_assignment[unassigned_mask]).sum())
            copy_unassigned_total += int(unassigned_mask.sum())

            delta_correct += int((pred_assignment[changed] == final_assignment[changed]).sum())
            delta_total += int(changed.sum())
            copy_all_correct += int(np.all(pred_assignment[copy_mask] == final_assignment[copy_mask]))
            full_exact_match += int(np.all(pred_assignment == final_assignment))

    return {
        "n_instances": len(sat_traces),
        "copy_accuracy": copy_correct / copy_total if copy_total else None,
        "copy_decided_accuracy": copy_decided_correct / copy_decided_total if copy_decided_total else None,
        "copy_decided_total": copy_decided_total,
        "copy_unassigned_accuracy": copy_unassigned_correct / copy_unassigned_total if copy_unassigned_total else None,
        "copy_unassigned_total": copy_unassigned_total,
        "delta_accuracy": delta_correct / delta_total if delta_total else None,
        "copy_all_correct_rate": copy_all_correct / len(sat_traces),
        "full_exact_match_rate": full_exact_match / len(sat_traces),
        "n_changed_histogram": n_changed_hist,
    }


if __name__ == "__main__":
    model, model_config = load_checkpoint(CHECKPOINT_DIR, device="cpu")
    print("loaded:", CHECKPOINT_DIR, "->", model_config)

    sat_cfg = SATConfig(n_vars=model_config["n_vars"], k=model_config["k"], alpha=model_config["alpha"])
    train_cfg = SATTrainConfig(n_train_instances=10000, n_val_instances=500)
    val_traces = build_traces(sat_cfg, train_cfg.n_val_instances, sat_cfg.seed + train_cfg.n_train_instances)

    result = copy_vs_delta_breakdown(model, val_traces, sat_cfg.n_vars, device="cpu")
    print(result)
