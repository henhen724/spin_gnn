"""Which TYPE of reasoning step is hardest for the v2 (event-based) model to
predict, under teacher forcing (true history, so this isolates "can it get
this kind of token right given perfect context" from the free-running
compounding-error story)?

Labels every target token in the validation set with a role based on where
it sits in the grammar (which section, and -- inside the reasoning section
-- which event kind: decision, unit_prop, or backtrack, and which field of
that event: the step-kind marker itself, the clause reference, the
variable, or the value), then reports per-token-prediction accuracy
grouped by role.
"""
import os

import numpy as np
import torch

from spin_gnn.config import SATReasoningConfig, SATReasoningTrainConfig
from spin_gnn.ksat2_tokens import encode_instance
from spin_gnn.train_sat_transformer2 import build_instances, load_checkpoint

CHECKPOINT_DIR = os.path.join(
    os.environ["SCRATCH"], "spin_gnn", "sat2_hparam_sweep", "m020_d0128_h02_L16_ff00512_lr3e-04"
)


def label_roles(tokens, vocab):
    """tokens: (T,) array of token ids. Returns a list of length T-1, one
    role string per TARGET token (tokens[1:], matching how the dataset
    pairs (input, target) -- tokens[0] is never itself a prediction target)."""
    roles = []
    n = len(tokens)
    phase = "hints"
    current_kind = None
    idx = 1
    while idx < n:
        t = int(tokens[idx])
        if phase == "hints":
            if t == vocab.BEGIN_STATEMENT:
                roles.append("control:BEGIN_STATEMENT")
                phase = "statement"
            else:
                roles.append("hints:content")
        elif phase == "statement":
            if t == vocab.BEGIN_REASONING:
                roles.append("control:BEGIN_REASONING")
                phase = "reasoning"
            else:
                roles.append("statement:content")
        elif phase == "reasoning":
            if t in (vocab.SAT, vocab.UNSAT):
                roles.append("control:outcome")
                phase = "final"
            elif t == vocab.NO_UNIT:
                roles.append("reasoning:step_kind")
                current_kind = "decision"
            elif t == vocab.IS_UNIT:
                roles.append("reasoning:step_kind")
                current_kind = "unit_prop"
            elif t == vocab.BACKTRACK:
                roles.append("reasoning:step_kind")
                current_kind = "backtrack"
            elif vocab.is_clause_token(t) and current_kind == "unit_prop":
                roles.append("reasoning:unit_prop:clause_ref")
            elif t == vocab.SET:
                roles.append(f"reasoning:{current_kind}:SET_marker")
            elif vocab.is_var_token(t):
                roles.append(f"reasoning:{current_kind}:var")
            elif t in (vocab.TRUE, vocab.FALSE):
                roles.append(f"reasoning:{current_kind}:value")
            else:
                roles.append("reasoning:other")
        else:  # final
            roles.append("final:content")
        idx += 1
    return roles


if __name__ == "__main__":
    model, vocab, config = load_checkpoint(CHECKPOINT_DIR, device="cpu")
    print("loaded:", CHECKPOINT_DIR, "->", config)

    sat_cfg = SATReasoningConfig(n_vars=config["n_vars"], k=config["k"], alpha=config["alpha"], n_hints=config["n_hints"])
    train_cfg = SATReasoningTrainConfig(n_train_instances=3000, n_val_instances=300)
    val_instances = build_instances(sat_cfg, train_cfg.n_val_instances, sat_cfg.seed + train_cfg.n_train_instances)

    correct_by_role = {}
    total_by_role = {}
    model.eval()
    with torch.no_grad():
        for clauses, n_vars, hints, trace in val_instances:
            tokens = encode_instance(clauses, n_vars, hints, trace, vocab)
            roles = label_roles(tokens, vocab)
            x = torch.from_numpy(tokens[:-1]).unsqueeze(0)
            target = tokens[1:]
            logits = model(x)[0]
            pred = logits.argmax(dim=-1).numpy()
            correct = pred == target
            for role, is_correct in zip(roles, correct):
                total_by_role[role] = total_by_role.get(role, 0) + 1
                correct_by_role[role] = correct_by_role.get(role, 0) + int(is_correct)

    print(f"\n{'role':<35} {'accuracy':>10} {'n':>8}")
    rows = [(role, correct_by_role[role] / total_by_role[role], total_by_role[role]) for role in total_by_role]
    rows.sort(key=lambda r: r[1])
    for role, acc, n in rows:
        print(f"{role:<35} {acc:>10.4f} {n:>8d}")
