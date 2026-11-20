"""Stage WIDTH -- is finding 111's per-step size penalty mediated by ROUND
WIDTH, and specifically by position WITHIN a round?

WHY.  Stage SCALE (finding 111) falsified C2: at matched DD depth T, N=96's
per-step `reason_node` accuracy sits on a lower plateau than N=48's (0.7604 vs
0.9129).  That killed the accumulation account -- depth alone does not explain
the gap -- but it does not say what about "bigger graph" costs the accuracy.
There is an obvious candidate that is NOT graph size as such:

  n_blank = N(1 - rho) cells are spread over T rounds, so at matched (T, rho)
  a doubling of N doubles the number of cells forced PER ROUND.

And `sudoku_trace.py:83` emits each round's cells SORTED by index, so writing
the j-th cell of a round means naming the smallest forced cell not among the
j-1 already named.  That is a within-round bookkeeping load which grows with
round width and has nothing to do with N per se.

If the gap closes at matched within-round position j, finding 111's penalty is
a round-width effect and `N` was a proxy for it.  If it survives at matched j,
the penalty is genuinely about locating a neighbourhood in a larger graph.
Those two make different predictions for the fast-slow regime, where BOTH T and
round width grow with N, so separating them is the point.

Imports the labelling and instance machinery from stage SCALE rather than
copying it: that script produced published numbers and is not edited here.

Run:  python3 scripts/stagewidth_probe.py --task I --ntasks K
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.rho_transition_eval import cfg_from_checkpoint
from scripts.stagescale_probe import (RUNS, RHOS, SCRATCH, instance_cache,
                                      run_path, section_labels)
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import load_checkpoint

OUT = os.path.join(SCRATCH, "spin_gnn/runs/stagewidth_2026-11-20")

# Coarse joint bins.  j and T are ANTI-correlated at fixed N (a deep instance
# has narrow rounds), so matching on j alone re-introduces a depth confound in
# the opposite direction to C2's.  The joint bin is what settles it.
T_BANDS = [(1, 4), (5, 9), (10, 19), (20, 10 ** 6)]
J_BANDS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 10 ** 6)]


def band(x, bands):
    for lo, hi in bands:
        if lo <= x <= hi:
            return "%d-%d" % (lo, hi) if hi < 10 ** 6 else "%d+" % lo
    raise AssertionError(x)


def round_meta(trace):
    """For each reason_node token, in emission order, its (width, position).

    Mirrors section_labels' walk of the reasoning section: per round a
    `round_mark`, then (reason_node, reason_color) per event.  Returned in the
    same order section_labels emits `reason_node`, so the two zip positionally.
    """
    out = []
    for events in trace["rounds"]:
        w = len(events)
        for j in range(w):
            out.append((w, j))
    return out


def probe_run(rec, rows_out):
    run_id, n_nodes, pair_key, rel = rec
    model, vocab, config = load_checkpoint(run_path(n_nodes, rel), device="cpu")
    model.eval()
    cfg = cfg_from_checkpoint(config)
    degree = cfg.degree
    max_len = int(config.get("max_len", 10 ** 9))
    assert cfg.n_nodes == n_nodes, (cfg.n_nodes, n_nodes)
    assert cfg.fall, "%s was not trained with --fall" % run_id

    for rho in RHOS:
        inst = instance_cache(cfg, rho)
        by_w, by_j, by_tj = {}, {}, {}
        n_tok = n_cor = n_skip = w_bad = 0
        sum_w = sum_inst = 0

        with torch.no_grad():
            for adj, _colors, _hints, trace in inst:
                if not trace["rounds"]:
                    continue
                toks = np.asarray(encode_instance(adj, trace, vocab),
                                  dtype=np.int64)
                if len(toks) > max_len:
                    n_skip += 1
                    continue
                lab = section_labels(adj, trace, vocab.n_nodes, degree)
                if len(lab) != len(toks):
                    continue
                meta = round_meta(trace)
                T = len(trace["rounds"])

                x = torch.from_numpy(toks[:-1]).unsqueeze(0)
                y = torch.from_numpy(toks[1:]).unsqueeze(0)
                mask = torch.ones_like(x, dtype=torch.bool)
                logits = model(x, valid_mask=mask)
                correct = (logits.argmax(dim=-1) == y).view(-1).numpy()

                tgt_lab = lab[1:]
                k = 0
                for i, s in enumerate(tgt_lab):
                    if s != "reason_node":
                        continue
                    if k >= len(meta):
                        w_bad += 1
                        break
                    w, j = meta[k]
                    k += 1
                    c = int(correct[i])
                    for d, key in ((by_w, w), (by_j, j),
                                   (by_tj, band(T, T_BANDS) + "|"
                                    + band(j, J_BANDS))):
                        cell = d.setdefault(key, [0, 0])
                        cell[0] += c
                        cell[1] += 1
                    n_tok += 1
                    n_cor += c
                # W-diagnostic: the walk must consume the trace exactly.
                if k != len(meta):
                    w_bad += 1
                sum_w += sum(len(e) for e in trace["rounds"])
                sum_inst += T

        rows_out.append({
            "run_id": run_id, "n_nodes": n_nodes, "pair_key": pair_key or "",
            "rho": "%.4f" % rho, "n_tokens": n_tok, "n_correct": n_cor,
            "accuracy": "%.8f" % (n_cor / n_tok) if n_tok else "",
            "mean_width": "%.4f" % (sum_w / max(sum_inst, 1)),
            "w_bad": w_bad, "n_skip": n_skip,
            "by_w": json.dumps({str(k): v for k, v in sorted(by_w.items())}),
            "by_j": json.dumps({str(k): v for k, v in sorted(by_j.items())}),
            "by_tj": json.dumps({k: v for k, v in sorted(by_tj.items())}),
        })
        print("  %s rho %.2f  reason_node acc %.4f  mean round width %.2f  "
              "w_bad %d" % (run_id, rho, n_cor / max(n_tok, 1),
                            sum_w / max(sum_inst, 1), w_bad), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--only", default="")
    ap.add_argument("--rho", type=float, nargs="*", help="smoke test only")
    args = ap.parse_args()

    # Smoke override only.  N_PER_BIN is NOT overridable here on purpose: these
    # instances are stage SCALE's cached 200-per-bin ones and must stay the
    # same puzzles, so the two stages are comparable token for token.
    global RHOS
    if args.rho:
        RHOS = list(args.rho)

    os.makedirs(args.out, exist_ok=True)
    recs = [r for r in RUNS if not args.only or r[0] == args.only]
    mine = recs[args.task::args.ntasks]
    print("task %d/%d: %d of %d runs" % (args.task, args.ntasks, len(mine),
                                         len(recs)), flush=True)

    rows = []
    for rec in mine:
        probe_run(rec, rows)

    path = os.path.join(args.out, "width_%dof%d.csv" % (args.task, args.ntasks))
    with open(path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
