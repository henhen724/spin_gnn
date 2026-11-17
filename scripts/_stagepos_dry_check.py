"""Invariants stage POS rests on, checked before a GPU hour is spent.

Four things can be silently wrong in this measurement and none of them would
show up as a crash -- they would show up as a plausible curve with the wrong
shape:

  1. THE POSITION SCHEMA. `stagepos_score.trace_schema` reconstructs
     (token type, deduction index, round) for every position of the reasoning
     section from the trace alone. If it is off by one token, every curve in
     measurement A is shifted and the whole question -- does error rise with
     position -- is answered against the wrong x-axis. Checked by rebuilding
     the token stream from the schema and comparing it to
     `encode_instance`'s output, token for token.

  2. THE TRACE LENGTH IDENTITY. hshunt's framing uses 2n + T + 1 for the
     number of reasoning tokens. Checked directly.

  3. BATCHED GREEDY == `generate`. Measurement B is only affordable batched,
     and batching changes the floating-point reduction order, so the two can
     in principle diverge at a near-tie. Checked against the reference
     one-at-a-time `train_sudoku_transformer.generate` on real instances: the
     sequences must agree token for token, and any disagreement is only
     tolerated where the reference's own top-two logit gap at that step was
     below 1e-3, i.e. a genuine tie rather than a bug.

  4. THE WALK. `walk_continuation` on the GROUND-TRUTH continuation must find
     exactly n pairs, all of them correct, with no error and no stop event. A
     walker that cannot score a perfect trace as perfect cannot be trusted on
     an imperfect one.

Runs on CPU in seconds, on the smallest successful checkpoint.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.rho_transition_eval import cfg_from_checkpoint
from scripts.stagepos_cells import SEED_START, successful_runs
from scripts.stagepos_score import (rollout_batch, trace_schema, true_pairs,
                                    walk_continuation)
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import build_instances, generate, load_checkpoint

from dataclasses import replace

N_INST = 6
RHO = 0.60
TIE_TOL = 1e-3


def rebuild_from_schema(trace, schema, vocab):
    toks, k = [], 0
    pairs = true_pairs(trace)
    for tok_type, j, _rnd in schema:
        if tok_type == "round":
            toks.append(vocab.BEGIN_ROUND)
        elif tok_type == "node":
            toks.append(vocab.node_token(pairs[j][0]))
        elif tok_type == "color":
            toks.append(vocab.color_token(pairs[j][1]))
        elif tok_type == "term":
            toks.append(vocab.SOLVED if trace["resolved"] else vocab.UNRESOLVED)
        else:
            raise AssertionError("unknown token type %r" % tok_type)
        k += 1
    return np.array(toks, dtype=np.int64)


def main():
    runs = successful_runs(verbose=False)
    # smallest trainable parameter count among the successful runs: the
    # cheapest model that is still a real one.
    rec = min((r for r in runs if r["in79"]), key=lambda r: r["n_trainable_params"])
    print("dry check on %s  %s seed %d  ntp=%d"
          % (rec["run_id"], rec["cell"], rec["seed"], rec["n_trainable_params"]))

    model, vocab, config = load_checkpoint(rec["path"], device="cpu")
    cfg = cfg_from_checkpoint(config)
    max_len = int(config["max_len"])
    inst = build_instances(replace(cfg, rho=RHO, rho_max=RHO), N_INST, SEED_START)

    # ---- 1 and 2 -----------------------------------------------------------
    for adj, _colors, hints, trace in inst:
        toks = encode_instance(adj, trace, vocab)
        rs = int(np.flatnonzero(toks == vocab.BEGIN_REASONING)[0]) + 1
        schema = trace_schema(trace, vocab)
        rebuilt = rebuild_from_schema(trace, schema, vocab)
        if not np.array_equal(rebuilt, toks[rs:]):
            raise SystemExit("SCHEMA MISMATCH: trace_schema does not reproduce the "
                             "reasoning section of encode_instance")
        n = len(true_pairs(trace))
        T = len(trace["rounds"])
        if len(toks) - rs != 2 * n + T + 1:
            raise SystemExit("LENGTH IDENTITY FAILED: %d reasoning tokens, "
                             "2n+T+1 = %d" % (len(toks) - rs, 2 * n + T + 1))
        if n + len(hints) != cfg.n_nodes:
            raise SystemExit("n blanks + n hints != N")
    print("  1,2  schema reproduces encode_instance, and 2n+T+1 holds, on %d instances"
          % len(inst))

    # ---- 4 -----------------------------------------------------------------
    for adj, colors, hints, trace in inst:
        toks = encode_instance(adj, trace, vocab)
        rs = int(np.flatnonzero(toks == vocab.BEGIN_REASONING)[0]) + 1
        pairs, term = walk_continuation(toks[rs:], vocab, adj, colors, hints,
                                        true_pairs(trace))
        n = len(true_pairs(trace))
        bad = [p for p in pairs if p["kind"] != "ok"]
        if len(pairs) != n or bad or term != "SOLVED":
            raise SystemExit("WALK FAILED on the ground-truth trace: %d/%d pairs, "
                             "%d bad, terminated=%r" % (len(pairs), n, len(bad), term))
        if not all(p["matches_true"] and p["color_proper"] for p in pairs):
            raise SystemExit("WALK: the ground-truth trace is not matches_true "
                             "and color_proper at every pair")
    print("  4    walk scores the ground-truth trace as %d/%d perfect pairs, SOLVED"
          % (len(pairs), len(pairs)))

    # ---- 3 -----------------------------------------------------------------
    prompts = []
    for adj, _colors, _hints, trace in inst:
        full = encode_instance(adj, trace, vocab)
        rs = int(np.flatnonzero(full == vocab.BEGIN_REASONING)[0]) + 1
        prompts.append(full[:rs])
    ref = [generate(model, p, vocab, max_len, device="cpu") for p in prompts]
    bat, _ = rollout_batch(model, prompts, vocab, "cpu", max_len, max_len)
    n_tol = 0
    for i, (a, b) in enumerate(zip(ref, bat)):
        if len(a) == len(b) and np.array_equal(a, b):
            continue
        # first divergence; a tie there is tolerable, anything else is not
        k = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        with torch.no_grad():
            lg = model(torch.from_numpy(a[:k].astype(np.int64)).unsqueeze(0))
            if isinstance(lg, tuple):
                lg = lg[0]
            top2 = torch.topk(lg[0, -1], 2).values
            gap = float(top2[0] - top2[1])
        if gap > TIE_TOL:
            raise SystemExit(
                "BATCHED GREEDY DIVERGED from generate() at instance %d position %d "
                "with a top-two logit gap of %.5f -- that is not a tie, it is a bug "
                "in rollout_batch" % (i, k, gap))
        n_tol += 1
    print("  3    batched greedy == generate() on %d/%d sequences (%d tolerated ties)"
          % (len(ref) - n_tol, len(ref), n_tol))
    print("DRY CHECK OK")


if __name__ == "__main__":
    main()
