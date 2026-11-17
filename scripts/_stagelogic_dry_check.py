"""Invariants stage LOGIC rests on, checked before a GPU hour is spent.

Everything here can be silently wrong without crashing, and would show up as a
plausible curve with the wrong shape rather than as an error:

 1. THE FORCED-SET IDENTITY. Measurement L reads F off the trace -- F at a node
    position is "the rest of the current round" -- instead of recomputing DD.
    That is only valid because sudoku_trace's round r IS the set of nodes
    forced by hints plus rounds < r. Checked by running `forced_mask` from
    scratch at the start of every round and demanding set equality with the
    round's node list. If this fails the cheap path is wrong and the whole
    measurement is scored against the wrong permitted set.

 2. THE SCHEMA. `logic_schema`'s true-token sequence must equal
    `encode_instance`'s reasoning section token for token. One token of drift
    shifts every curve.

 3. THE LOGICAL RULE CAN ONLY FORGIVE. The ground-truth node at every node
    position must itself be in F, so a strictly-correct argmax is always
    logically correct and p_logic >= p_strict is an identity, not a hope.

 4. THE WALK. `walk_forced` on the GROUND-TRUTH continuation must find every
    pair in F, with no pair outside it. A tracker that cannot certify a perfect
    trace as fully forced cannot be trusted on an imperfect one.

 5. A PREMATURE NODE EXISTS. The "premature" class must be reachable: at a node
    position there must be uncoloured nodes outside F, otherwise the guard
    question is vacuous. Counted, not assumed.

Runs on CPU in seconds, on the smallest successful checkpoint.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.rho_transition_eval import cfg_from_checkpoint
from scripts.stagepos_cells import SEED_START, successful_runs
from scripts.stagelogic_score import (forced_mask, hint_asg, logic_schema,
                                      walk_forced)
from scripts.stagepos_score import true_pairs
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import build_instances, load_checkpoint

from dataclasses import replace

N_INST = 8
RHOS = [0.85, 0.60, 0.40]


def main():
    runs = successful_runs(verbose=False)
    rec = min((r for r in runs if r["in79"]), key=lambda r: r["n_trainable_params"])
    print("dry check on %s  %s seed %d" % (rec["run_id"], rec["cell"], rec["seed"]))
    _model, vocab, config = load_checkpoint(rec["path"], device="cpu")
    cfg = cfg_from_checkpoint(config)
    q = int(vocab.n_colors)

    n_rounds = n_pos = n_prem_avail = 0
    for rho in RHOS:
        inst = build_instances(replace(cfg, rho=rho, rho_max=rho), N_INST, SEED_START)
        for adj, colors, hints, trace in inst:
            # ---- 1 --------------------------------------------------------
            asg = hint_asg(adj.shape[0], hints)
            for r, ev in enumerate(trace["rounds"]):
                want = {int(nd) for nd, _c in ev}
                got = set(np.flatnonzero(forced_mask(adj, asg, q)).tolist())
                if got != want:
                    raise SystemExit(
                        "FORCED-SET IDENTITY FAILED at rho=%.2f round %d: DD forces "
                        "%d nodes, the trace round holds %d, symmetric difference %d"
                        % (rho, r, len(got), len(want), len(got ^ want)))
                for nd, c in ev:
                    asg[int(nd)] = int(c)
                n_rounds += 1

            # ---- 2 and 3 ---------------------------------------------------
            toks = encode_instance(adj, trace, vocab)
            rs = int(np.flatnonzero(toks == vocab.BEGIN_REASONING)[0]) + 1
            schema = logic_schema(trace, vocab)
            rebuilt = np.array([s[-1] for s in schema], dtype=np.int64)
            if not np.array_equal(rebuilt, toks[rs:]):
                raise SystemExit("SCHEMA MISMATCH: logic_schema does not reproduce "
                                 "the reasoning section of encode_instance")
            colored = {int(v) for v in trace["hints"]}
            for tok_type, j, _r, allowed, true_tok in schema:
                if tok_type != "node":
                    continue
                u = vocab.node_index(true_tok)
                if u not in allowed:
                    raise SystemExit("THE TRUE NODE IS NOT IN F at j=%d -- the "
                                     "logical rule would mark a correct argmax "
                                     "wrong" % j)
                free = adj.shape[0] - len(colored) - len(allowed)
                n_prem_avail += free > 0
                colored.add(u)
                n_pos += 1

            # ---- 4 ---------------------------------------------------------
            per_pair, err = walk_forced(toks[rs:], vocab, adj, q, hints, colors, -1)
            n = len(true_pairs(trace))
            if len(per_pair) != n or not all(p["in_F"] for p in per_pair) or err:
                bad = sum(1 for p in per_pair if not p["in_F"])
                raise SystemExit("WALK_FORCED FAILED on the ground-truth trace: "
                                 "%d/%d pairs, %d not in F, err=%r"
                                 % (len(per_pair), n, bad, err))
    print("  1  DD forced set == the trace's round, on %d rounds" % n_rounds)
    print("  2  logic_schema reproduces encode_instance")
    print("  3  the true node is in F at all %d node positions" % n_pos)
    print("  4  walk_forced certifies the ground-truth trace fully forced")
    # ---- 5 -------------------------------------------------------------------
    if n_prem_avail == 0:
        raise SystemExit("NO PREMATURE NODE IS EVER AVAILABLE -- the guard "
                         "question is vacuous on these instances")
    print("  5  an uncoloured node outside F exists at %d/%d node positions (%.1f%%)"
          % (n_prem_avail, n_pos, 100.0 * n_prem_avail / n_pos))
    print("DRY CHECK OK")


if __name__ == "__main__":
    main()
