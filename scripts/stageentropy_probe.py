"""Stage ENTROPY -- is the next-token predictive distribution collapsing to a
point on the deduction tokens?  Nothing is trained; checkpoints are loaded
read-only; no closed knob is touched.

THE QUESTION (hshunt, 2026-11-20).  A model that had learned direct deduction
exactly would be CERTAIN at every reasoning step, so its predictive entropy
there would be zero.  Is the measured entropy concentrating toward zero?

WHY ENTROPY MUST BE READ PER SECTION, AND A POOLED NUMBER IS MEANINGLESS.
The sequence mixes two kinds of token and only one of them has a zero floor:

  * DETERMINISTIC GIVEN THE PREFIX -- `reason_node`, `reason_color`,
    `round_mark`, `term`.  `sudoku_trace.py:83` appends `sorted(...)` per round,
    so the next cell is the smallest not-yet-emitted forced cell, and DD fixes
    its colour.  A perfect model puts mass 1 on one symbol: H = 0 exactly.
  * GENUINELY RANDOM -- `hint_node`, `hint_color`, `graph_lead`,
    `graph_nbr_novel`.  The graph is a fresh random d-regular draw and the hint
    set is a random subset, so no model can do better than the draw's own
    entropy.  H has a POSITIVE floor here and driving it down would be
    memorisation, not learning.
  * `graph_nbr_copy` sits in between: the EDGE is already determined by the
    earlier mention, but `shuffle_adj=True` randomises the order of the five
    slots within a row, so an ordering entropy survives.

Finding 109 is the same point in the loss: `val_nll`'s floor rises with N
(1.371 / 1.625 / 1.861 at N = 48 / 96 / 192) entirely because of the graph
section, `f_graph = (5/16) log N`.  So a falling total entropy would prove
nothing and a flat total entropy would disprove nothing.  Everything below is
keyed on the deterministic sections.

ENTROPY IS NOT NLL.  `H = -sum_v p_v log p_v` is the model's OWN uncertainty;
`NLL = -log p(target)` is whether it was right.  A confidently wrong model has
low H and high NLL.  Both are recorded at every position so the gap can be read
directly -- that gap, not either number alone, is what says whether a collapsed
distribution has collapsed onto the CORRECT symbol.

REGISTERED PREDICTIONS.  Scored on the nine SCALE cells at five densities.
"Best cell" means the highest teacher-forced `reason_node` accuracy.

  E1  mean H on `reason_color`, best cell       < 0.05 nats   falsifier > 0.30
  E2  mean H on `reason_node`, best cell        < 0.15 nats   falsifier > 0.60
  E3  CONCENTRATION, the actual question: fraction of `reason_node` tokens
      with H < 0.10 nats at the best cell       >= 0.70       falsifier < 0.30
  E4  the random floor does NOT collapse: mean H on `graph_nbr_novel`
      > 1.0 nats in EVERY cell                                falsifier < 0.5
  E5  entropy tracks competence: Spearman(mean reasoning H, `reason_node`
      accuracy) over the nine cells <= -0.70    falsifier > -0.30
  E6  harder instances are less certain: reasoning H at rho 0.35 exceeds
      rho 0.90 in every cell                                  falsifier: fails
      in more than two cells
  E7  the N penalty shows up as uncertainty: at matched `pair_key`, N=96
      reasoning H > N=48 reasoning H in all three pairs        falsifier: 0 of 3

  Diagnostics (must pass or the numbers are void):
  DE1 token counts per section agree with stage SCALE's, cell for cell.
  DE2 H <= log(vocab_size) at every position, and H >= 0.
  DE3 on a UNIFORM distribution the estimator returns log V to 1e-9.

E4 and E6 are controls: if they fail, the probe is measuring something other
than what it claims.  E3 is the one hshunt asked for and is reported whichever
way it falls.

Stdlib + torch.  Reuses stage SCALE's RUNS/RHOS/cache/section_labels by import,
deliberately: SCALE produced published numbers and is not edited here.

Run:  python3 scripts/stageentropy_probe.py --task I --ntasks K
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.rho_transition_eval import cfg_from_checkpoint
from scripts.stagescale_probe import (RHOS, RUNS, SECTIONS, instance_cache,
                                      run_path, section_labels)
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import load_checkpoint

SCRATCH = "/scratch/users/hshunt"
OUT = os.path.join(SCRATCH, "spin_gnn/runs/stageentropy_2026-11-20")

# The deterministic sections -- the only ones with a zero entropy floor.
DETERMINISTIC = ("reason_node", "reason_color", "round_mark", "term")
# Bin edges for the concentration histogram, in nats.  The first bin is the
# "effectively certain" one that E3 is scored on.
H_EDGES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00]


def h_bin_counts(values):
    """Counts falling under each edge, plus a final >= last-edge bucket."""
    out = [0] * (len(H_EDGES) + 1)
    for v in values:
        for i, e in enumerate(H_EDGES):
            if v < e:
                out[i] += 1
                break
        else:
            out[-1] += 1
    return out


def probe_run(rec, rows_out, limit):
    run_id, n_nodes, pair_key, rel = rec
    model, vocab, config = load_checkpoint(run_path(n_nodes, rel), device="cpu")
    model.eval()
    cfg = cfg_from_checkpoint(config)
    degree = cfg.degree
    max_len = int(config.get("max_len", 10 ** 9))
    vocab_size = None

    for rho in RHOS:
        inst = instance_cache(cfg, rho)
        if limit:
            inst = inst[:limit]
        h_sum = {s: 0.0 for s in SECTIONS}
        h_sq = {s: 0.0 for s in SECTIONS}
        nll_sum = {s: 0.0 for s in SECTIONS}
        p1_sum = {s: 0.0 for s in SECTIONS}
        n_tok = {s: 0 for s in SECTIONS}
        n_cor = {s: 0 for s in SECTIONS}
        h_vals = {s: [] for s in DETERMINISTIC}
        h_max_seen, n_inst, n_skip = 0.0, 0, 0

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
                    n_skip += 1
                    continue
                x = torch.from_numpy(toks[:-1]).unsqueeze(0)
                y = torch.from_numpy(toks[1:]).unsqueeze(0)
                mask = torch.ones_like(x, dtype=torch.bool)
                logits = model(x, valid_mask=mask)
                b, l, v = logits.shape
                vocab_size = v
                logp = F.log_softmax(logits, dim=-1)
                p = logp.exp()
                # H = -sum p log p, per position.  Computed from log_softmax
                # rather than log(softmax) so an underflowing tail contributes
                # 0 * -inf as 0 rather than nan.
                ent = -(p * logp).sum(dim=-1).view(l).numpy()
                nll = F.cross_entropy(logits.reshape(b * l, v),
                                      y.reshape(b * l),
                                      reduction="none").view(l).numpy()
                top1 = p.max(dim=-1).values.view(l).numpy()
                correct = (logits.argmax(dim=-1) == y).view(l).numpy()

                for i, s in enumerate(lab[1:]):
                    e = float(ent[i])
                    h_sum[s] += e
                    h_sq[s] += e * e
                    nll_sum[s] += float(nll[i])
                    p1_sum[s] += float(top1[i])
                    n_tok[s] += 1
                    n_cor[s] += int(correct[i])
                    if s in DETERMINISTIC:
                        h_vals[s].append(e)
                    if e > h_max_seen:
                        h_max_seen = e
                n_inst += 1

        det_tok = sum(n_tok[s] for s in DETERMINISTIC)
        det_h = sum(h_sum[s] for s in DETERMINISTIC)
        for s in SECTIONS:
            n = n_tok[s]
            mean_h = h_sum[s] / n if n else float("nan")
            var = (h_sq[s] / n - mean_h ** 2) if n else float("nan")
            rows_out.append({
                "run_id": run_id, "n_nodes": n_nodes,
                "pair_key": pair_key or "", "rho": "%.4f" % rho,
                "section": s, "n_tokens": n,
                "mean_H": "%.8f" % mean_h if n else "",
                "sd_H": "%.8f" % math.sqrt(max(var, 0.0)) if n else "",
                "mean_nll": "%.8f" % (nll_sum[s] / n) if n else "",
                "mean_top1": "%.8f" % (p1_sum[s] / n) if n else "",
                "accuracy": "%.8f" % (n_cor[s] / n) if n else "",
                "h_bins": json.dumps(h_bin_counts(h_vals[s]))
                          if s in DETERMINISTIC else "",
                "reasoning_mean_H": "%.8f" % (det_h / det_tok) if det_tok else "",
                "n_instances": n_inst, "n_skip": n_skip,
                "vocab_size": vocab_size,
                "max_H_seen": "%.8f" % h_max_seen,
                "log_vocab": "%.8f" % math.log(vocab_size) if vocab_size else "",
            })
        print("  %-12s rho %.2f  H(reason_node) %.4f  H(graph_novel) %.4f  "
              "acc %.4f" % (run_id, rho,
                            h_sum["reason_node"] / max(n_tok["reason_node"], 1),
                            h_sum["graph_nbr_novel"]
                            / max(n_tok["graph_nbr_novel"], 1),
                            n_cor["reason_node"] / max(n_tok["reason_node"], 1)),
              flush=True)


def selftest():
    """DE3 -- the estimator returns log V on a uniform distribution."""
    v = 58
    logits = torch.zeros(1, 3, v)
    logp = F.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(dim=-1)
    err = float((ent - math.log(v)).abs().max())
    print("DE3 uniform-distribution selftest: max |H - log V| = %.3e  %s"
          % (err, "PASS" if err < 1e-9 else "FAIL"))
    return err < 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap instances per bin; 0 = all cached")
    args = ap.parse_args()

    if not selftest():
        return 1
    os.makedirs(args.out, exist_ok=True)
    recs = [r for r in RUNS if not args.only or r[0] == args.only]
    mine = recs[args.task::args.ntasks]
    print("task %d/%d: %d of %d runs, limit=%s"
          % (args.task, args.ntasks, len(mine), len(recs), args.limit or "all"),
          flush=True)

    rows = []
    for rec in mine:
        probe_run(rec, rows, args.limit)

    path = os.path.join(args.out, "entropy_%dof%d.csv" % (args.task,
                                                          args.ntasks))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
