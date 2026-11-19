"""Stage BYPASS (107) -- the measurement half.

Re-runs the five section-level ablations from stage ABLATE's exploratory ladder
on the 36 runs that stage ABLATE never probed, to test out of sample whether
surviving the loss of the neighbours' hint colours (ALLHINT) tracks competence
or architecture family.  Bars and falsifiers are registered in
runs/stagebypass_2026-11-19/REGISTERED.md, committed before this file existed.

Unlike stageablate_probe.py this script needs no head ranking: every condition
here blocks a set of KEY POSITIONS at every (layer, head), so there is nothing
to rank and the held-out runs -- which stage HEADS never measured -- are usable.

Torch-only and writes no figures: the py-pytorch and py-matplotlib module sets
unload each other on this cluster, so scoring lives in scripts/stagebypass_fit.py
and the two cannot be one program.

Nothing is trained.  Checkpoints are loaded read-only.
"""
import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.stagepos_cells import successful_runs
from spin_gnn.sat_transformer2 import applied_blocks
from spin_gnn.sudoku_attention_distance import (forward_with_ablation,
                                                token_positions)
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import load_checkpoint

OUT = "/scratch/users/hshunt/spin_gnn/runs/stagebypass_2026-11-19"
BOUND = "/scratch/users/hshunt/spin_gnn/runs/stagebound_2026-11-17/bound_runs.csv"
ABLATE = "/scratch/users/hshunt/spin_gnn/runs/stageablate_2026-11-19/ablate_runs.csv"
INST = "/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17/instances"
RHOS = [0.90, 0.75, 0.60]
SLICE0, SLICE1 = 200, 400          # the held-out half; HEADS used [0:200]
GATE = 0.740
N_NODES = 48
DEGREE = 5
SUMTOL = 1e-4
E3TOL = 1e-6
E3_ROWS = 8

# All five block key positions at every (layer, head).  INTACT first so that
# E1 is measured on the same instances as everything else.
CONDITIONS = ["INTACT", "ALLHINT", "ALLPAIRS", "ALLROWS6", "ALLSEC"]


def norm_cell(c):
    return " ".join(c.split())


def load_instances(rho):
    path = os.path.join(INST, "inst_rho%05d_n400_s0.pkl" % round(rho * 10000))
    with open(path, "rb") as fh:
        return pickle.load(fh)[SLICE0:SLICE1]


def heldout_runs():
    """The 36 runs in bound_runs.csv that stage ABLATE did not probe, keyed by
    run_id and carrying PER-RUN rho_50 (not the cell median ABLATE used)."""
    with open(ABLATE) as fh:
        seen = {r["run_id"] for r in csv.DictReader(fh)}
    out = {}
    with open(BOUND) as fh:
        for r in csv.DictReader(fh):
            if r["run_id"] not in seen and r["flag_pub"] == "ok":
                out[r["run_id"]] = float(r["rho50_pub"])
    return out


def select_runs():
    held = heldout_runs()
    recs = [r for r in successful_runs(verbose=False) if r["run_id"] in held]
    recs.sort(key=lambda r: r["run_id"])
    return [(r, held[r["run_id"]]) for r in recs]


def key_positions(cond, adj, pos, v, own_row, nbr_hint, rows6, all_sec):
    if cond == "INTACT":
        return []
    if cond == "ALLPAIRS":
        return own_row
    if cond == "ALLHINT":
        return nbr_hint
    if cond == "ALLROWS6":
        return rows6
    if cond == "ALLSEC":
        return all_sec
    raise ValueError(cond)


def probe_run(rec, rho50, rho, inst, rows_out):
    model, vocab, config = load_checkpoint(rec["path"], device="cpu")
    model.eval()
    depth = len(applied_blocks(model))
    heads = model.encoder.layers[0].self_attn.num_heads
    all_pairs = [(l, h) for l in range(depth) for h in range(heads)]

    n_q = 0
    correct = {c: 0 for c in CONDITIONS}
    mass = {c: 0.0 for c in CONDITIONS}
    e2_bad = e3_bad = e4_bad = 0
    checked = False

    for i, (adj, colors, hints, trace) in enumerate(inst):
        if not trace["rounds"]:
            continue
        v = int(trace["rounds"][0][0][0])
        true_c = int(trace["rounds"][0][0][1])
        toks = encode_instance(adj, trace, vocab)
        pos = token_positions(adj, trace, vocab)
        q = pos["round_color_pos"][v] - 1
        row_v = pos["graph_row_pos"][v]

        # E4 -- token_positions reimplements encode_instance's layout.
        if int(toks[row_v]) != vocab.node_token(v) or \
           int(toks[q + 1]) != vocab.color_token(true_c) or \
           int(toks[q]) != vocab.node_token(v):
            e4_bad += 1
            continue

        x = torch.from_numpy(np.asarray(toks, dtype=np.int64)).unsqueeze(0)
        own_row = list(range(row_v, row_v + 1 + DEGREE))
        nbr_hint = [pos["hint_color_pos"][int(u)] for u in adj[v]
                    if int(u) in pos["hint_color_pos"]]
        # everything before BEGIN_REASONING: the whole hints section and the
        # whole graph section, markers included.  ALLSEC's positive control.
        # q itself is never blocked, so no attention row is ever fully masked.
        all_sec = list(range(0, pos["graph_row_pos"][N_NODES - 1]
                             + 1 + len(adj[N_NODES - 1]) + 1))
        # the SIX rows that mention v: its own, plus the row of each of its 5
        # neighbours, where v appears as a neighbour entry.  The encoding
        # writes v's adjacency six times, which is the redundancy ALLPAIRS
        # leaves untouched.
        rows6 = list(own_row)
        for u in adj[v]:
            ru = pos["graph_row_pos"][int(u)]
            rows6 += list(range(ru, ru + 1 + len(adj[int(u)])))
        want = vocab.color_token(true_c)

        # the sham: same per-head mask shape, nothing blocked.  E3's reference,
        # so that E3(c) tests locality and not a kernel-dispatch difference.
        sham_attn = None
        if not checked:
            _, sham_attn = forward_with_ablation(
                model, x, q=q, blocked={(l, 0): [] for l in range(depth)},
                need_attn=True)
            sham = np.stack([t.numpy() for t in sham_attn])  # (depth,heads,L,L)

        for cond in CONDITIONS:
            keys = key_positions(cond, adj, pos, v, own_row, nbr_hint,
                                 rows6, all_sec)
            blocked = {p: keys for p in all_pairs} if keys else {}
            logits, _ = forward_with_ablation(
                model, x, q=q, blocked=blocked, need_attn=False)
            correct[cond] += int(int(logits[0, q].argmax()) == want)

            if not checked:
                # E2 -- how much row-q attention mass this condition actually
                # removes, measured on the UNABLATED forward and averaged over
                # every (layer, head).  ABLATE only ever recorded this for the
                # ranked conditions; the registration makes it an E-check here,
                # so a condition that silently blocks nothing cannot pass.
                mass[cond] = float(sham[:, :, q, keys].sum(axis=2).mean()) \
                    if keys else 0.0

            if (not checked) and blocked:
                # An EXTRA forward purely for the checks, padded so that every
                # application carries a per-head mask exactly as the sham does.
                # Without the padding, E3(b) would compare a 2-D mask path at
                # the non-ablated applications against the sham's 3-D one and
                # report a kernel-dispatch difference as a locality failure.
                # The metric above is never read from this forward.
                padded = dict(blocked)
                for l in range(depth):
                    padded.setdefault((l, 0), [])
                _, attn = forward_with_ablation(
                    model, x, q=q, blocked=padded, need_attn=True)
                a = np.stack([t.numpy() for t in attn])
                # E2(b) -- the block landed and the row still normalises
                for (l, h) in blocked:
                    if np.abs(a[l, h, q, keys]).max() > 0:
                        e2_bad += 1
                    if abs(a[l, h, q, :q + 1].sum() - 1.0) > SUMTOL:
                        e2_bad += 1
                # E3(a) -- rows p < q are unchanged at every application
                if q > 0:
                    rs = np.random.RandomState(abs(hash((rec["run_id"], cond))) % 2**31)
                    rows = rs.choice(q, min(E3_ROWS, q), replace=False)
                    for l in range(depth):
                        # a[l][:, rows, :] in two steps: a[l, :, rows, :] would
                        # separate two advanced indices by a slice, which numpy
                        # answers by moving the row axis to the front (torch
                        # does not), so the two sides stop being comparable.
                        d = np.abs(a[l][:, rows, :] - sham[l][:, rows, :]).max()
                        if d > E3TOL:
                            e3_bad += 1
                # E3(b) -- ALLHINT's key set, recomputed from the adjacency
                # alone, must be exactly the hint-colour tokens of v's
                # neighbours: no hint of a non-neighbour, no neighbour missed.
                # This is the locality check the registration names.
                if cond == "ALLHINT":
                    nbrs = {int(u) for u in adj[v]}
                    want_keys = sorted(p for u, p in pos["hint_color_pos"].items()
                                       if int(u) in nbrs)
                    if sorted(keys) != want_keys:
                        e3_bad += 1
                    for u, p in pos["hint_color_pos"].items():
                        if p in keys and int(u) not in nbrs:
                            e3_bad += 1
        checked = True
        n_q += 1

    if n_q == 0:
        return
    for cond in CONDITIONS:
        rows_out.append({
            "run_id": rec["run_id"], "cell": norm_cell(rec["cell"]),
            "seed": rec["seed"], "rho50": "%.4f" % rho50,
            "group": "gate" if rho50 < GATE else "below",
            "rho": "%.2f" % rho, "cond": cond,
            "depth": depth, "heads": heads, "n_pairs": depth * heads,
            "d_model": rec["d_model"], "arch_mode": rec["arch_mode"],
            "params": rec["n_trainable_params"], "val_acc": "%.6f" % rec["val_acc"],
            "n_q": n_q, "n_correct": correct[cond],
            "acc": "%.6f" % (correct[cond] / n_q),
            "mass_removed": "%.6f" % mass[cond],
            "e2_bad": e2_bad, "e3_bad": e3_bad, "e4_bad": e4_bad,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n-inst", type=int, default=0, help="0 = the full held-out half")
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--rhos", default="")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    rhos = [float(s) for s in args.rhos.split(",")] if args.rhos else RHOS

    runs = select_runs()
    runs = [r for i, r in enumerate(runs) if i % args.nshards == args.shard]
    if args.max_runs:
        runs = runs[:args.max_runs]
    print("shard %d/%d: %d runs, rhos %s, conditions %d, instances [%d:%d]"
          % (args.shard, args.nshards, len(runs), rhos, len(CONDITIONS),
             SLICE0, SLICE1), flush=True)

    cache = {}
    for rho in rhos:
        inst = load_instances(rho)
        cache[rho] = inst[:args.n_inst] if args.n_inst else inst

    rows_out = []
    t0 = time.time()
    for i, (rec, rho50) in enumerate(runs):
        for rho in rhos:
            probe_run(rec, rho50, rho, cache[rho], rows_out)
        print("  [%2d/%2d] %-4s %-28s s%d rho50=%.3f  %.1f s elapsed"
              % (i + 1, len(runs), rec["run_id"], norm_cell(rec["cell"]),
                 rec["seed"], rho50, time.time() - t0), flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "bypass_runs_%dof%d.csv" % (args.shard, args.nshards))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("wrote %s (%d rows)\ntotal %.1f s" % (path, len(rows_out), time.time() - t0),
          flush=True)


if __name__ == "__main__":
    main()
