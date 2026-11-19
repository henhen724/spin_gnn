"""Stage ABLATE (106) -- the measurement half.

Blocks the top-m own-row (layer, head) pairs from reading the named cell's
BEGIN_GRAPH row at the j=0 query and re-measures argmax-colour accuracy, against
a same-count random-pair control and a same-pairs wrong-row control.  The
conditions, bars and falsifiers are registered in
runs/stageablate_2026-11-19/REGISTERED.md, committed before this file existed.

Selection of the top-m pairs is read from stage HEADS's byhead CSVs, which were
measured on instances [0:200].  This script evaluates on [200:400], so the
selection is out of sample with respect to the evaluation.

Torch-only and writes no figures: the py-pytorch and py-matplotlib module sets
unload each other on this cluster, so scoring lives in scripts/stageablate_fit.py
and the two cannot be one program.

Nothing is trained.  Checkpoints are loaded read-only.
"""
import argparse
import csv
import glob
import os
import pickle
import sys
import time
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.stagepos_cells import successful_runs
from spin_gnn.sat_transformer2 import applied_blocks
from spin_gnn.sudoku_attention_distance import (forward_with_ablation,
                                                token_positions)
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import load_checkpoint

OUT = "/scratch/users/hshunt/spin_gnn/runs/stageablate_2026-11-19"
HEADS = "/scratch/users/hshunt/spin_gnn/runs/stageheads_2026-11-19"
INST = "/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17/instances"
RHOS = [0.90, 0.75, 0.60]
SLICE0, SLICE1 = 200, 400          # the held-out half; HEADS used [0:200]
GATE = 0.740
RAND_POOL_EXCLUDES = 6             # RAND-m draws from pairs outside the top 6
N_NODES = 48
DEGREE = 5
SUMTOL = 1e-4
E3TOL = 1e-6
E3_ROWS = 8

# (name, m).  Registered: INTACT, TOP 1/2/3/6, RAND 1/3/6, OTHERROW 3.
# The three ALL* conditions are EXPLORATORY and carry no registered bar; they
# were added after a 12-instance pilot showed TOP-6 (90% of own-row mass on that
# run) moving the logits at q by 4.6 on average without ever changing the
# argmax.  They ask the sharper question that makes possible: whether the row
# and the neighbours' hints are read at the query position at all.
CONDITIONS = [("INTACT", 0), ("TOP", 1), ("TOP", 2), ("TOP", 3), ("TOP", 6),
              ("RAND", 1), ("RAND", 3), ("RAND", 6), ("OTHERROW", 3),
              ("ALLPAIRS", 0), ("ALLHINT", 0), ("ALLBOTH", 0),
              ("ALLROWS6", 0), ("ALLADJ", 0), ("ALLSEC", 0)]
EXPLORATORY = {"ALLPAIRS", "ALLHINT", "ALLBOTH", "ALLROWS6", "ALLADJ", "ALLSEC"}


def norm_cell(c):
    return " ".join(c.split())


def load_instances(rho):
    path = os.path.join(INST, "inst_rho%05d_n400_s0.pkl" % round(rho * 10000))
    with open(path, "rb") as fh:
        return pickle.load(fh)[SLICE0:SLICE1]


def cell_rho50():
    """cell -> published rho_50, from the HEADS merged run table."""
    out = {}
    with open(os.path.join(HEADS, "heads_runs.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["in79"] == "True":
                out[norm_cell(r["cell"])] = float(r["rho50_pub_cell"])
    return out


def heads_ranking():
    """(cell, seed, rho) -> [(layer, head), ...] by descending own-row mass,
    plus the own-row mass share each prefix removes."""
    rank, mass = {}, {}
    acc = {}
    for path in sorted(glob.glob(os.path.join(HEADS, "heads_byhead_*of*.csv"))):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                key = (norm_cell(r["cell"]), int(r["seed"]), "%.2f" % float(r["rho"]))
                acc.setdefault(key, []).append(
                    (float(r["own_row"]), int(r["layer"]), int(r["head"])))
    for key, v in acc.items():
        v.sort(key=lambda t: -t[0])
        tot = sum(t[0] for t in v) or 1.0
        rank[key] = [(l, h) for _, l, h in v]
        mass[key] = np.cumsum([t[0] for t in v]) / tot
    return rank, mass


def select_runs(gate_only=False):
    """18 gate-passing runs (all seeds, all untied) + the 25 below-gate cells at
    their lowest seed, which carry A6 only."""
    r50 = cell_rho50()
    recs = [r for r in successful_runs(verbose=False)
            if r["in79"] and norm_cell(r["cell"]) in r50]
    out, seen = [], {}
    for rec in sorted(recs, key=lambda r: (norm_cell(r["cell"]), r["seed"])):
        cell = norm_cell(rec["cell"])
        passing = r50[cell] <= GATE
        if passing:
            out.append((rec, "gate", r50[cell]))
        elif not gate_only and cell not in seen:
            seen[cell] = True
            out.append((rec, "below", r50[cell]))
    return out


def rng_for(run_id, rho, cond, m, i):
    seed = zlib.crc32(("%s|%s|%s|%d|%d" % (run_id, rho, cond, m, i)).encode())
    return np.random.RandomState(seed)


def blocked_for(cond, m, pairs, pool, all_pairs, own_row, other_row, nbr_hint,
                all_sec, rows6, rs):
    if cond == "INTACT":
        return {}
    if cond == "TOP":
        return {p: own_row for p in pairs[:m]}
    if cond == "RAND":
        take = pool if len(pool) <= m else [pool[j] for j in
                                            rs.choice(len(pool), m, replace=False)]
        return {p: own_row for p in take}
    if cond == "OTHERROW":
        return {p: other_row for p in pairs[:m]}
    if cond == "ALLPAIRS":
        return {p: own_row for p in all_pairs}
    if cond == "ALLHINT":
        return {p: nbr_hint for p in all_pairs} if nbr_hint else {}
    if cond == "ALLBOTH":
        return {p: own_row + nbr_hint for p in all_pairs}
    if cond == "ALLROWS6":
        return {p: rows6 for p in all_pairs}
    if cond == "ALLADJ":
        return {p: rows6 + nbr_hint for p in all_pairs}
    if cond == "ALLSEC":
        return {p: all_sec for p in all_pairs}
    raise ValueError(cond)


def probe_run(rec, group, rho50, rho, inst, rank, mass, rows_out):
    model, vocab, config = load_checkpoint(rec["path"], device="cpu")
    model.eval()
    depth = len(applied_blocks(model))
    heads = model.encoder.layers[0].self_attn.num_heads
    key = (norm_cell(rec["cell"]), rec["seed"], "%.2f" % rho)
    if key not in rank:
        return
    pairs = rank[key]
    cum = mass[key]
    pool = pairs[RAND_POOL_EXCLUDES:] or pairs[:]

    n_q = 0
    correct = {c: 0 for c in CONDITIONS}
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
        all_pairs = [(l, h) for l in range(depth) for h in range(heads)]
        # everything before BEGIN_REASONING: the whole hints section and the
        # whole graph section, markers included.  ALLSEC's positive control.
        # q itself is never blocked, so no attention row is ever fully masked.
        all_sec = list(range(0, pos["graph_row_pos"][N_NODES - 1]
                             + 1 + len(adj[N_NODES - 1]) + 1))
        # the SIX rows that mention v: its own, plus the row of each of its 5
        # neighbours, where v appears as a neighbour entry.  The encoding
        # writes v's adjacency six times, which is the redundancy ALLPAIRS and
        # ALLBOTH leave untouched.
        rows6 = list(own_row)
        for u in adj[v]:
            ru = pos["graph_row_pos"][int(u)]
            rows6 += list(range(ru, ru + 1 + len(adj[int(u)])))
        want = vocab.color_token(true_c)

        # the sham: same per-head mask shape, nothing blocked.  E3's reference,
        # so that (b) tests locality and not a kernel-dispatch difference.
        sham_attn = None
        if not checked:
            _, sham_attn = forward_with_ablation(
                model, x, q=q, blocked={(l, 0): [] for l in range(depth)},
                need_attn=True)

        for cond, m in CONDITIONS:
            rs = rng_for(rec["run_id"], "%.2f" % rho, cond, m, i)
            u = v
            while u == v:
                u = int(rs.randint(N_NODES))
            other_row = list(range(pos["graph_row_pos"][u],
                                   pos["graph_row_pos"][u] + 1 + DEGREE))
            blocked = blocked_for(cond, m, pairs, pool, all_pairs, own_row,
                                  other_row, nbr_hint, all_sec, rows6, rs)
            logits, _ = forward_with_ablation(
                model, x, q=q, blocked=blocked, need_attn=False)
            correct[(cond, m)] += int(int(logits[0, q].argmax()) == want)

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
                a = torch.stack([t for t in attn]).numpy()   # (depth, heads, L, L)
                # E2 -- the block landed and the row still normalises
                for (l, h), keys in blocked.items():
                    if np.abs(a[l, h, q, keys]).max() > 0:
                        e2_bad += 1
                    if abs(a[l, h, q, :q + 1].sum() - 1.0) > SUMTOL:
                        e2_bad += 1
                # E3(a) -- application 0, row q, non-ablated heads
                hit0 = {h for (l, h) in blocked if l == 0}
                for h in range(heads):
                    if h in hit0:
                        continue
                    if np.abs(a[0, h, q] - sham_attn[0][h, q].numpy()).max() > E3TOL:
                        e3_bad += 1
                # E3(b) -- rows p < q are unchanged at every application
                if q > 0:
                    rows = rs.choice(q, min(E3_ROWS, q), replace=False)
                    for l in range(depth):
                        # a[l][:, rows, :] in two steps: a[l, :, rows, :] would
                        # separate two advanced indices by a slice, which numpy
                        # answers by moving the row axis to the front (torch
                        # does not), so the two sides stop being comparable.
                        d = np.abs(a[l][:, rows, :] -
                                   sham_attn[l][:, rows, :].numpy()).max()
                        if d > E3TOL:
                            e3_bad += 1
        checked = True
        n_q += 1

    if n_q == 0:
        return
    for (cond, m) in CONDITIONS:
        # own-row mass the condition removes; OTHERROW removes none of it by
        # construction, which is the whole point of that control.
        removed = float(cum[m - 1]) if cond == "TOP" else 0.0
        rows_out.append({
            "run_id": rec["run_id"], "cell": norm_cell(rec["cell"]),
            "seed": rec["seed"], "group": group, "rho50_pub_cell": "%.4f" % rho50,
            "rho": "%.2f" % rho, "cond": cond, "m": m,
            "depth": depth, "heads": heads, "n_pairs": depth * heads,
            "d_model": rec["d_model"], "arch_mode": rec["arch_mode"],
            "params": rec["n_trainable_params"], "val_acc": "%.6f" % rec["val_acc"],
            "n_q": n_q, "n_correct": correct[(cond, m)],
            "acc": "%.6f" % (correct[(cond, m)] / n_q),
            "mass_removed": "%.6f" % removed,
            "registered": cond not in EXPLORATORY,
            # RAND redraws per instance, so no single pair set describes it.
            "top_pairs": ";".join("%d.%d" % p for p in pairs[:m])
                         if cond in ("TOP", "OTHERROW") else "",
            "e2_bad": e2_bad, "e3_bad": e3_bad, "e4_bad": e4_bad,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n-inst", type=int, default=0, help="0 = the full held-out half")
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--rhos", default="")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    rhos = [float(s) for s in args.rhos.split(",")] if args.rhos else RHOS

    rank, mass = heads_ranking()
    runs = select_runs(gate_only=args.gate_only)
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
    for i, (rec, group, rho50) in enumerate(runs):
        for rho in rhos:
            probe_run(rec, group, rho50, rho, cache[rho], rank, mass, rows_out)
        print("  [%2d/%2d] %-28s s%d %-5s %.1f s elapsed"
              % (i + 1, len(runs), norm_cell(rec["cell"]), rec["seed"], group,
                 time.time() - t0), flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "ablate_runs_%dof%d.csv" % (args.shard, args.nshards))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("wrote %s (%d rows)\ntotal %.1f s" % (path, len(rows_out), time.time() - t0),
          flush=True)


if __name__ == "__main__":
    main()
