"""Stage SECTION -- per-section teacher-forced NLL and accuracy on the 79
existing N=48 checkpoints.  Nothing is trained; checkpoints are loaded
read-only; no closed knob is touched.

WHY THIS EXISTS.  Finding 109 showed that the metric this project ranks
architectures on is mostly not about deduction: the graph section is 72-73% of
tokens at every N and the reasoning section only ~19%, so ~81% of `val_nll` is
spent transmitting the instance.  The standing non-negotiable says "validation
token accuracy FIRST" and `scripts/acc_vs_solve.py` built the whole minimality
frame on it -- on a number that is three-quarters graph slots.  The trainer has
no per-section metric (`evaluate` returns one nll/accuracy over ALL tokens), so
the decomposition has never been measured.  It needs no training to measure.

SCOPE, AND THE JUDGEMENT IS MINE AND FLAGGED RATHER THAN HIDDEN.  Like stage
HEADS I judged this inside standing scope because it trains nothing, touches no
closed knob, and is orthogonal to the four questions waiting on hshunt.  Unlike
HEADS it does not even load attention -- it is one forward pass per instance.

WHAT IT MEASURES.  `sat2_nll_loss` is a per-position cross-entropy averaged
over `valid_mask`, so a per-section NLL is the identical computation under a
different mask -- no reimplementation of the loss, and the section NLLs
reconstruct the published total exactly (diagnostic E2).  Sections follow
`sudoku_tokens.encode_instance`'s layout:

    BEGIN_HINTS (NODE COLOR)*                      -> hint_node, hint_color
    BEGIN_GRAPH (NODE (NODE){degree})*             -> graph_lead, graph_nbr_*
    BEGIN_REASONING (BEGIN_ROUND (NODE COLOR)+)*   -> round_mark, reason_*
    SOLVED                                         -> term

THE GRAPH SECTION IS SPLIT INTO NOVEL AND COPY SLOTS, which is the point.  A
neighbour slot in row `i` naming `j` is a COPY when the edge was already
written in row `j < i` and NOVEL otherwise; every undirected edge is written
twice, so the split is exactly half.  109's closed form prices only the novel
half, `f_graph = [d/(2(d+3))] log N`, and this probe is where that estimate
meets a measurement.

A COPY SLOT IS NOT DETERMINED, WHICH IS EASY TO GET WRONG AND IS WHY S1 IS
BANDED LOW.  With `--shuffle-adj` the order WITHIN a row is random, so even
though the SET of already-seen neighbours of `i` is recoverable from earlier
rows, which one fills the next slot is not.  The ceiling on a copy slot is the
within-row ordering entropy, not 1.0.

Run:  python3 scripts/stagesection_probe.py --task I --ntasks K
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.stagepos_cells import successful_runs
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import load_checkpoint

OUT = "/scratch/users/hshunt/spin_gnn/runs/stagesection_2026-11-20"
BOUND = "/scratch/users/hshunt/spin_gnn/runs/stagebound_2026-11-17/bound_runs.csv"
INST = "/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17/instances"
RHOS = [0.90, 0.75, 0.60]
SLICE0, SLICE1 = 200, 400          # the held-out half, as BYPASS used
N_NODES = 48
DEGREE = 5
GATE = 0.740

SECTIONS = ["hint_node", "hint_color", "graph_lead", "graph_nbr_novel",
            "graph_nbr_copy", "round_mark", "reason_node", "reason_color",
            "term", "begin"]


def load_instances(rho):
    path = os.path.join(INST, "inst_rho%05d_n400_s0.pkl" % round(rho * 10000))
    with open(path, "rb") as fh:
        return pickle.load(fh)[SLICE0:SLICE1]


def rho50_by_run():
    out = {}
    with open(BOUND) as fh:
        for r in csv.DictReader(fh):
            if r["flag_pub"] == "ok":
                out[r["run_id"]] = float(r["rho50_pub"])
    return out


def section_labels(adj, trace, n_nodes):
    """One label per token, mirroring encode_instance's layout exactly.  Built
    from the instance rather than by re-encoding, so E1 can check it against
    the encoded stream token for token."""
    lab = ["begin"]                                   # BEGIN_HINTS
    for _node, _c in sorted(trace["hints"].items()):
        lab += ["hint_node", "hint_color"]
    lab.append("begin")                               # BEGIN_GRAPH
    seen = set()
    for i in range(n_nodes):
        lab.append("graph_lead")
        for j in adj[i]:
            e = (min(i, int(j)), max(i, int(j)))
            lab.append("graph_nbr_copy" if e in seen else "graph_nbr_novel")
            seen.add(e)
    lab.append("begin")                               # BEGIN_REASONING
    for events in trace["rounds"]:
        lab.append("round_mark")
        for _ev in events:
            lab += ["reason_node", "reason_color"]
    lab.append("term")                                # SOLVED / UNRESOLVED
    return lab


def probe_run(rec, rho50, rho, inst, rows_out):
    model, vocab, config = load_checkpoint(rec["path"], device="cpu")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    nll_sum = {s: 0.0 for s in SECTIONS}
    n_tok = {s: 0 for s in SECTIONS}
    n_cor = {s: 0 for s in SECTIONS}
    e1_bad = 0
    n_inst = 0

    with torch.no_grad():
        for adj, _colors, _hints, trace in inst:
            if not trace["rounds"]:
                continue
            toks = np.asarray(encode_instance(adj, trace, vocab), dtype=np.int64)
            lab = section_labels(adj, trace, vocab.n_nodes)

            # E1 -- the label walk and the encoder must agree on length, and on
            # every graph row leader being that row's own node token.
            if len(lab) != len(toks):
                e1_bad += 1
                continue

            x = torch.from_numpy(toks[:-1]).unsqueeze(0)
            y = torch.from_numpy(toks[1:]).unsqueeze(0)
            mask = torch.ones_like(x, dtype=torch.bool)
            logits = model(x, valid_mask=mask)
            b, l, v = logits.shape
            per_pos = F.cross_entropy(logits.reshape(b * l, v),
                                      y.reshape(b * l),
                                      reduction="none").view(l).numpy()
            correct = (logits.argmax(dim=-1) == y).view(l).numpy()

            # target i predicts token i+1, so the target's section is the
            # section of the token being PREDICTED.
            tgt_lab = lab[1:]
            for i, s in enumerate(tgt_lab):
                nll_sum[s] += float(per_pos[i])
                n_tok[s] += 1
                n_cor[s] += int(correct[i])
            n_inst += 1

    tot_nll = sum(nll_sum.values())
    tot_tok = sum(n_tok.values())
    tot_cor = sum(n_cor.values())
    for s in SECTIONS:
        rows_out.append({
            "run_id": rec["run_id"], "cell": rec["cell"], "n_params": n_params,
            "rho50_pub": "%.6f" % rho50, "gate_pass": int(rho50 <= GATE),
            "rho": "%.4f" % rho, "n_instances": n_inst, "section": s,
            "n_tokens": n_tok[s], "nll_sum": "%.8f" % nll_sum[s],
            "n_correct": n_cor[s],
            "nll": "%.8f" % (nll_sum[s] / n_tok[s]) if n_tok[s] else "",
            "accuracy": "%.8f" % (n_cor[s] / n_tok[s]) if n_tok[s] else "",
            "total_nll": "%.8f" % (tot_nll / tot_tok),
            "total_accuracy": "%.8f" % (tot_cor / tot_tok),
            "total_tokens": tot_tok, "e1_bad": e1_bad,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-inst", type=int, default=0,
                    help="smoke-test only: cap instances per density")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rho50 = rho50_by_run()
    recs = [r for r in successful_runs(verbose=False) if r["run_id"] in rho50]
    recs.sort(key=lambda r: r["run_id"])
    mine = recs[args.task::args.ntasks]
    print("task %d/%d: %d of %d runs" % (args.task, args.ntasks, len(mine), len(recs)),
          flush=True)

    rows = []
    for rho in RHOS:
        inst = load_instances(rho)
        if args.max_inst:
            inst = inst[:args.max_inst]
        for k, rec in enumerate(mine):
            probe_run(rec, rho50[rec["run_id"]], rho, inst, rows)
            print("  rho %.2f  %d/%d  %s" % (rho, k + 1, len(mine), rec["run_id"]),
                  flush=True)

    path = os.path.join(args.out, "section_%dof%d.csv" % (args.task, args.ntasks))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)), flush=True)


if __name__ == "__main__":
    main()
