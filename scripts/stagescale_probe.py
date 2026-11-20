"""Stage SCALE -- the per-section decomposition at N=48 AND N=96, on matched
cells.  Nothing is trained; checkpoints are loaded read-only; no closed knob.

WHY THIS EXISTS.  Finding 109 argued ANALYTICALLY that `val_nll`'s floor rises
with `N` because the graph section carries `[d/(2(d+3))] log N` nats/token, and
stage SECTION then MEASURED at N=48 that 97.8-99.85% of the loss is not
deduction.  Both halves are needed to read the N=192 ladder: its 2.19 against
N=48's 1.64 is only a category error if the reasoning section's own cost is
`N`-invariant.  That has never been measured at two values of `N`.

THE QUESTION, AND BOTH ANSWERS ARE LIVE.  Finding 108 measured a real
per-deduction penalty in FREE generation at matched DD depth (`a_eff` 0.9940 at
N=48 against 0.9122/0.9628 at N=96, T=12.0).  Stage POS's P3 found teacher-forced
accuracy FLAT in position where free generation fails, and 98 showed the two
regimes score different events.  So:

  * if teacher-forced `reason_node` accuracy is flat in `N` at matched depth,
    108's penalty is ACCUMULATION over a longer trace, not per-step competence,
    and the ladder's rising `val_nll` says nothing about deduction;
  * if it falls, the penalty is PER-STEP and shows up in the loss itself.

THE FLOOR FOR `reason_node` IS ZERO AT EVERY N, which is what makes the
comparison clean.  `sudoku_trace.py:83` appends `sorted(...)` per round, so the
next cell is the smallest not-yet-emitted forced cell -- a convention the model
can learn exactly (the best N=48 run reaches 0.9950).  There is no `log N`
ordering entropy to subtract, unlike the graph section's copy slots.  ACCURACY
is the cross-N statistic rather than NLL, because the cell alphabet doubles from
48 to 96 and chance alone would cost `log 2` in nats.

INSTANCES COME FROM EACH CHECKPOINT'S OWN CONFIG.  `cfg_from_checkpoint` reads
`n_nodes`, `degree`, `n_colors`, `hint_mode`, `shuffle_adj` and **`fall`** from
`config.json`, so the instance distribution matches TRAINING by construction and
cannot silently differ in the fall key.  Only `rho`/`rho_max` are overridden, to
the bin.

Run:  python3 scripts/stagescale_probe.py --task I --ntasks K
"""
import argparse
import csv
import json
import os
import pickle
import sys
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.rho_transition_eval import cfg_from_checkpoint
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import build_instances, load_checkpoint

SCRATCH = "/scratch/users/hshunt"
OUT = os.path.join(SCRATCH, "spin_gnn/runs/stagescale_2026-11-20")
CACHE = os.path.join(OUT, "instances")
SCAN_48 = os.path.join(SCRATCH, "spin_gnn/sudoku_n48_fall_scan_2026-11-02")
SCAN_96 = os.path.join(SCRATCH, "spin_gnn/sudoku_n96_fall_scan_2026-11-02")

# Five densities, not SECTION's three.  The two low ones exist ONLY so the DD
# depths overlap across N: at rho 0.60 the N=48 instances sit at T ~ 5 and the
# N=96 ones deeper, so a matched-rho comparison is not a matched-DEPTH one, and
# 108's penalty was measured at matched depth.
RHOS = [0.90, 0.75, 0.60, 0.45, 0.35]
N_PER_BIN = 200
SEED_START = 70_000_000_000     # disjoint from the eval grid's 60e9 and from
                                # stagepos's cache, so these instances are new

SECTIONS = ["hint_node", "hint_color", "graph_lead", "graph_nbr_novel",
            "graph_nbr_copy", "round_mark", "reason_node", "reason_color",
            "term", "begin"]

# (run_id, n_nodes, pair_key, path).  pair_key is None where the cell exists at
# only one N; the registered claims are keyed on the three pairs that exist at
# BOTH, which holds depth, width, recipe, token budget per node and SEED fixed
# and varies only N.
RUNS = [
    ("n48_L08_s0", 48, "L08_s0", "t05_L08_ff01024_n240000_s0/d0256_h04_L08_ff01024_lr1e-03_pre_shuf_fall"),
    ("n48_L12_s0", 48, "L12_s0", "t08_L12_ff01024_n240000_s0/d0256_h04_L12_ff01024_lr1e-03_pre_shuf_fall"),
    ("n48_L08_s1", 48, "L08_s1", "t11_L08_ff01024_n240000_s1/d0256_h04_L08_ff01024_lr1e-03_pre_shuf_fall_s01"),
    ("n48_L04_s0", 48, None, "t02_L04_ff01024_n240000_s0/d0256_h04_L04_ff01024_lr1e-03_pre_shuf_fall"),
    ("n96_L08_s0", 96, "L08_s0", "t01_L08_ff01024_n240000_s0/d0256_h04_L08_ff01024_lr1e-03_pre_shuf_fall"),
    ("n96_L12_s0", 96, "L12_s0", "t00_L12_ff01024_n240000_s0/d0256_h04_L12_ff01024_lr1e-03_pre_shuf_fall"),
    ("n96_L08_s1", 96, "L08_s1", "t05_L08_ff01024_n240000_s1/d0256_h04_L08_ff01024_lr1e-03_pre_shuf_fall_s01"),
    ("n96_L12_s1", 96, None, "t04_L12_ff01024_n240000_s1/d0256_h04_L12_ff01024_lr1e-03_pre_shuf_fall_s01"),
    ("n96_L16_s0", 96, None, "t02_L16_ff01024_n240000_s0/d0256_h04_L16_ff01024_lr1e-03_pre_shuf_fall"),
]


def run_path(n_nodes, rel):
    """The run DIRECTORY -- load_checkpoint reads config.json beside model.pt."""
    root = SCAN_48 if n_nodes == 48 else SCAN_96
    return os.path.join(root, rel)


def section_labels(adj, trace, n_nodes, degree):
    """One label per token, mirroring encode_instance's layout exactly.

    Identical to stagesection_probe.section_labels except that `degree` is read
    from the instance rather than assumed -- the row length is `1 + degree`, and
    hard-coding 5 here would silently mislabel any future (d, q) rung.
    """
    lab = ["begin"]                                   # BEGIN_HINTS
    for _node, _c in sorted(trace["hints"].items()):
        lab += ["hint_node", "hint_color"]
    lab.append("begin")                               # BEGIN_GRAPH
    seen = set()
    for i in range(n_nodes):
        lab.append("graph_lead")
        row = adj[i]
        assert len(row) == degree, (len(row), degree)
        for j in row:
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


def instance_cache(cfg, rho):
    """Built once per (n_nodes, degree, n_colors, rho) and shared by every run
    in that family at that N.

    Atomic replace, as stagepos_score.instance_cache does, so two array tasks
    racing on the same bin cannot tear the file.

    degree and n_colors are IN THE KEY, not just in `cfg`: they change the
    instance family (and rho_c, and the analytic length N(degree+3)+T+4), so a
    checkpoint trained at another (degree, q) must not be handed cached
    instances from this one.  Every run here is d=5 q=4, so the key is constant
    in practice -- it exists so that stops being an assumption.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "inst_n%d_d%d_q%d_rho%05d_n%d.pkl"
                        % (cfg.n_nodes, cfg.degree, cfg.n_colors,
                           round(rho * 10000), N_PER_BIN))
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    bin_cfg = replace(cfg, rho=float(rho), rho_max=float(rho))
    inst = build_instances(bin_cfg, N_PER_BIN, SEED_START)
    tmp = path + ".%d.tmp" % os.getpid()
    with open(tmp, "wb") as fh:
        pickle.dump(inst, fh, protocol=4)
    os.replace(tmp, path)
    return inst


def probe_run(rec, rows_out):
    run_id, n_nodes, pair_key, rel = rec
    model, vocab, config = load_checkpoint(run_path(n_nodes, rel), device="cpu")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg = cfg_from_checkpoint(config)
    degree = cfg.degree
    max_len = int(config.get("max_len", 10 ** 9))
    assert cfg.n_nodes == n_nodes, (cfg.n_nodes, n_nodes)
    assert cfg.fall, "%s was not trained with --fall" % run_id

    for rho in RHOS:
        inst = instance_cache(cfg, rho)
        nll_sum = {s: 0.0 for s in SECTIONS}
        n_tok = {s: 0 for s in SECTIONS}
        n_cor = {s: 0 for s in SECTIONS}
        # Per-depth accuracy on the deduction tokens, which is the matched-DEPTH
        # comparison 108 made in free generation.  Keyed by T so the fit can bin
        # across N without re-running anything.
        by_T = {}
        d1_bad = d3_bad = n_skip = 0
        sum_T = 0

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
                    d1_bad += 1
                    continue
                T = len(trace["rounds"])
                # D3 -- 109's closed form for the sequence length, at this N and
                # this degree.  The rho dependence cancels because every cell
                # costs two tokens either as a hint or as a deduction.
                if len(toks) != n_nodes * (degree + 3) + T + 4:
                    d3_bad += 1

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
                rn = by_T.setdefault(T, [0, 0])
                for i, s in enumerate(tgt_lab):
                    nll_sum[s] += float(per_pos[i])
                    n_tok[s] += 1
                    n_cor[s] += int(correct[i])
                    if s == "reason_node":
                        rn[0] += int(correct[i])
                        rn[1] += 1
                sum_T += T

        tot_nll = sum(nll_sum.values())
        tot_tok = sum(n_tok.values())
        tot_cor = sum(n_cor.values())
        n_inst = sum(1 for _ in inst) - n_skip
        for s in SECTIONS:
            rows_out.append({
                "run_id": run_id, "n_nodes": n_nodes,
                "pair_key": pair_key or "", "n_params": n_params,
                "degree": degree, "rho": "%.4f" % rho, "n_instances": n_inst,
                "mean_T": "%.4f" % (sum_T / max(n_inst, 1)),
                "section": s, "n_tokens": n_tok[s],
                "nll_sum": "%.8f" % nll_sum[s], "n_correct": n_cor[s],
                "nll": "%.8f" % (nll_sum[s] / n_tok[s]) if n_tok[s] else "",
                "accuracy": "%.8f" % (n_cor[s] / n_tok[s]) if n_tok[s] else "",
                "total_nll": "%.8f" % (tot_nll / tot_tok),
                "total_accuracy": "%.8f" % (tot_cor / tot_tok),
                "total_tokens": tot_tok,
                "d1_bad": d1_bad, "d3_bad": d3_bad, "n_skip": n_skip,
                "by_T": json.dumps({str(k): v for k, v in sorted(by_T.items())}),
            })
        print("  %s rho %.2f  nll %.4f  reason_node acc %.4f  meanT %.2f"
              % (run_id, rho, tot_nll / tot_tok,
                 n_cor["reason_node"] / max(n_tok["reason_node"], 1),
                 sum_T / max(n_inst, 1)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--only", default="", help="smoke test: one run_id")
    ap.add_argument("--n-per-bin", type=int, default=0, help="smoke test only")
    ap.add_argument("--rho", type=float, nargs="*", help="smoke test only")
    args = ap.parse_args()

    # Smoke overrides.  N_PER_BIN is in the cache filename, so a smoke cannot
    # poison the real bins' instances.
    global N_PER_BIN, RHOS
    if args.n_per_bin:
        N_PER_BIN = args.n_per_bin
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

    path = os.path.join(args.out, "scale_%dof%d.csv" % (args.task, args.ntasks))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)), flush=True)


if __name__ == "__main__":
    main()
