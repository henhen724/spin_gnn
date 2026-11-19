"""Stage HEADS (105) -- the measurement half.

Reads attention at the one query position whose hidden state must emit the
colour of the first cell DD deduces, and attributes it to the token groups
registered in runs/stageheads_2026-11-19/REGISTERED.md:

    OWNROW   -- the named cell's own row of BEGIN_GRAPH (header + 5 neighbours)
    ROWS     -- all 48 graph rows
    OWNNBR   -- the 5 neighbour tokens of OWNROW, split CRITICAL/DROPPABLE/BLANK
    NBRHINT  -- the hint COLOUR tokens of the named cell's assigned neighbours
    HINTS    -- all hint COLOUR tokens

This script is torch-only and writes no figures: on this cluster the
py-pytorch and py-matplotlib module sets unload each other, so scoring lives
in scripts/stageheads_fit.py and the two cannot be one program.

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
from spin_gnn.sudoku_attention_distance import (forward_capture_attention,
                                                token_positions)
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import load_checkpoint

OUT = "/scratch/users/hshunt/spin_gnn/runs/stageheads_2026-11-19"
INST = "/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17/instances"
RHOS = [0.90, 0.75, 0.60]
N_INST = 200
N_NODES = 48
DEGREE = 5
SUMTOL = 1e-4


def load_instances(rho):
    path = os.path.join(INST, "inst_rho%05d_n400_s0.pkl" % round(rho * 10000))
    with open(path, "rb") as fh:
        return pickle.load(fh)[:N_INST]


def neighbour_classes(adj, asg, v):
    """Per the registration: every j=0 cell is DD-forced, so its k assigned
    neighbours carry exactly 3 distinct colours.  Returns one label per slot of
    adj[v], in the order the tokens are emitted (encode_instance iterates
    adj[i] directly, which is what token_positions mirrors)."""
    cols = [int(asg[int(u)]) for u in adj[v]]
    cols = [None if c < 0 else c for c in cols]
    assigned = [c for c in cols if c is not None]
    out = []
    for c in cols:
        if c is None:
            out.append("blank")
        elif assigned.count(c) == 1:
            out.append("crit")
        else:
            out.append("drop")
    return out, len(assigned)


def probe_run(rec, rho, inst, byhead_rows, run_rows):
    model, vocab, config = load_checkpoint(rec["path"], device="cpu")
    model.eval()
    depth = len(applied_blocks(model))
    heads = model.encoder.layers[0].self_attn.num_heads

    # accumulators over instances, per (layer, head)
    acc = {k: np.zeros((depth, heads)) for k in
           ("own_row", "own_hdr", "own_nbr", "rows", "nbrhint", "hints")}
    # neighbour-class mass is summed over (l,h); counts are token counts
    cls = {k: 0.0 for k in ("crit_dc", "drop_dc", "assigned_ab", "blank_ab",
                            "crit_all", "drop_all", "blank_all")}
    cnt = {k: 0 for k in cls}
    n_q = n_hint_tok = k_tot = 0
    rank1 = 0
    rank_sum = 0
    e1_color_right = e1_is_color = 0
    e2_bad = e3_bad = 0

    for adj, colors, hints, trace in inst:
        if not trace["rounds"]:
            continue
        v = int(trace["rounds"][0][0][0])
        true_c = int(trace["rounds"][0][0][1])
        asg = np.full(adj.shape[0], -1, dtype=np.int8)   # hint_asg, inlined:
        for node, col in trace["hints"].items():         # stagecand_fit imports
            asg[int(node)] = int(col)                    # matplotlib, which
        toks = encode_instance(adj, trace, vocab)        # unloads torch here
        pos = token_positions(adj, trace, vocab)
        q = pos["round_color_pos"][v] - 1
        row_v = pos["graph_row_pos"][v]

        # E3 -- token_positions reimplements encode_instance's layout rather
        # than reading it, so check the two agree on every query.
        if int(toks[row_v]) != vocab.node_token(v) or \
           int(toks[q + 1]) != vocab.color_token(true_c) or \
           int(toks[q]) != vocab.node_token(v):
            e3_bad += 1
            continue

        x = torch.from_numpy(np.asarray(toks, dtype=np.int64)).unsqueeze(0)
        logits, attn = forward_capture_attention(model, x)
        # keep only the query row: (depth, heads, L)
        a = torch.stack([m[:, q, :] for m in attn]).numpy()

        # E2 -- a proper causal distribution
        s = a.sum(axis=2)
        if np.abs(s - 1.0).max() > SUMTOL or (a[:, :, q + 1:] != 0).any():
            e2_bad += 1

        # E1
        pred = int(logits[0, q].argmax())
        e1_is_color += int(pred in vocab._color_token_set)
        e1_color_right += int(pred == vocab.color_token(true_c))

        # --- group masses, per (layer, head) -------------------------------
        own_idx = np.arange(row_v, row_v + 1 + DEGREE)
        acc["own_row"] += a[:, :, own_idx].sum(axis=2)
        acc["own_hdr"] += a[:, :, row_v]
        acc["own_nbr"] += a[:, :, row_v + 1:row_v + 1 + DEGREE].sum(axis=2)

        row_starts = np.array([pos["graph_row_pos"][i] for i in range(N_NODES)])
        all_idx = (row_starts[:, None] + np.arange(1 + DEGREE)[None, :])
        per_row = a[:, :, all_idx].sum(axis=3)           # (depth, heads, 48)
        acc["rows"] += per_row.sum(axis=2)

        hint_pos = np.array(sorted(pos["hint_color_pos"].values()))
        acc["hints"] += a[:, :, hint_pos].sum(axis=2)
        nbr_hint = np.array([pos["hint_color_pos"][int(u)] for u in adj[v]
                             if int(u) in pos["hint_color_pos"]])
        if nbr_hint.size:
            acc["nbrhint"] += a[:, :, nbr_hint].sum(axis=2)
        n_hint_tok += len(hint_pos)

        # --- rank of the named cell's row among 48 -------------------------
        rowmass = per_row.sum(axis=(0, 1))               # (48,)
        r = int(1 + (rowmass > rowmass[v]).sum())
        rank1 += int(r == 1)
        rank_sum += r

        # --- neighbour-slot classes ----------------------------------------
        labels, k = neighbour_classes(adj, asg, v)
        k_tot += k
        slot_mass = a[:, :, row_v + 1:row_v + 1 + DEGREE].sum(axis=(0, 1))
        for t, lab in enumerate(labels):
            cls[lab + "_all"] += float(slot_mass[t])
            cnt[lab + "_all"] += 1
        if k >= 4:                       # has a droppable neighbour
            for t, lab in enumerate(labels):
                if lab in ("crit", "drop"):
                    cls[lab + "_dc"] += float(slot_mass[t])
                    cnt[lab + "_dc"] += 1
        if k <= 4:                       # has a blank neighbour
            for t, lab in enumerate(labels):
                if lab == "blank":
                    cls["blank_ab"] += float(slot_mass[t])
                    cnt["blank_ab"] += 1
                else:
                    cls["assigned_ab"] += float(slot_mass[t])
                    cnt["assigned_ab"] += 1
        n_q += 1

    if n_q == 0:
        return

    for l in range(depth):
        for h in range(heads):
            byhead_rows.append({
                "run": "%s|s%d" % (rec["cell"], rec["seed"]),
                "cell": rec["cell"], "seed": rec["seed"], "rho": "%.2f" % rho,
                "layer": l, "head": h, "n_q": n_q,
                "own_row": "%.8g" % acc["own_row"][l, h],
                "own_hdr": "%.8g" % acc["own_hdr"][l, h],
                "own_nbr": "%.8g" % acc["own_nbr"][l, h],
                "rows": "%.8g" % acc["rows"][l, h],
                "nbrhint": "%.8g" % acc["nbrhint"][l, h],
                "hints": "%.8g" % acc["hints"][l, h],
            })

    g = acc["own_row"].ravel()
    order = np.sort(g)[::-1]
    tot = order.sum()
    n50 = int(np.searchsorted(np.cumsum(order), 0.5 * tot) + 1) if tot > 0 else -1
    top1 = float(order[0] / tot) if tot > 0 else float("nan")

    row = {
        "run_id": rec["run_id"], "cell": rec["cell"], "rho": "%.2f" % rho,
        "depth": depth, "heads": heads, "d_model": rec["d_model"],
        "arch_mode": rec["arch_mode"], "num_layers": rec["num_layers"],
        "params": rec["n_trainable_params"], "in79": rec["in79"],
        "val_acc": "%.6f" % rec["val_acc"], "n_q": n_q,
        "m_own_row": "%.8g" % acc["own_row"].sum(),
        "m_own_hdr": "%.8g" % acc["own_hdr"].sum(),
        "m_own_nbr": "%.8g" % acc["own_nbr"].sum(),
        "m_rows": "%.8g" % acc["rows"].sum(),
        "m_nbrhint": "%.8g" % acc["nbrhint"].sum(),
        "m_hints": "%.8g" % acc["hints"].sum(),
        "k_tot": k_tot, "n_hint_tok": n_hint_tok,
        "n50": n50, "top1_frac": "%.6f" % top1,
        "rank1": rank1, "rank_sum": rank_sum,
        "e1_color_right": e1_color_right, "e1_is_color": e1_is_color,
        "e2_bad": e2_bad, "e3_bad": e3_bad,
    }
    for k in cls:
        row["mass_" + k] = "%.8g" % cls[k]
        row["n_" + k] = cnt[k]
    run_rows.append(row)


def main():
    global N_INST
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n-inst", type=int, default=N_INST)
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    N_INST = args.n_inst

    runs = successful_runs(verbose=False)
    runs = [r for i, r in enumerate(runs) if i % args.nshards == args.shard]
    if args.max_runs:
        runs = runs[:args.max_runs]
    print("shard %d/%d: %d runs, rhos %s, %d instances each"
          % (args.shard, args.nshards, len(runs), RHOS, N_INST), flush=True)

    cache = {rho: load_instances(rho) for rho in RHOS}
    byhead_rows, run_rows = [], []
    t0 = time.time()
    for i, rec in enumerate(runs):
        for rho in RHOS:
            probe_run(rec, rho, cache[rho], byhead_rows, run_rows)
        print("  [%2d/%2d] %-34s %.1f s elapsed"
              % (i + 1, len(runs), rec["cell"], time.time() - t0), flush=True)

    os.makedirs(args.out, exist_ok=True)
    tag = "_%dof%d" % (args.shard, args.nshards)
    for name, rows in (("heads_byhead", byhead_rows), ("heads_runs", run_rows)):
        path = os.path.join(args.out, name + tag + ".csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote %s (%d rows)" % (path, len(rows)), flush=True)
    print("total %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
