"""Why did the chained-case census return zero? Count survivors per filter.

Diagnostic only -- no model, no GPU, nothing written. Run on CPU.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/users/hshunt/spin_gnn")

from scripts.ddguard_score import CKPT
from spin_gnn.sudoku_chain4 import _stays_proper, revealed_before
from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.train_sudoku_transformer import build_instances

N_INST = int(os.environ.get("N_INST", "200"))
Q = 4


def main():
    import json
    with open(os.path.join(CKPT["S1"][0], "config.json")) as fh:
        config = json.load(fh)
    cfg = SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"],
        n_colors=config["n_colors"], rho=config["rho"], rho_max=config["rho_max"],
        scratchpad=config.get("scratchpad", False),
        hint_mode=config.get("hint_mode", "removal"),
        shuffle_adj=bool(config.get("shuffle_adj", False)),
        shuffle_events=bool(config.get("shuffle_events", False)),
        fall=bool(config.get("fall", False)))
    print("cfg n=%d d=%d q=%d rho=%s fall=%s" % (cfg.n_nodes, cfg.degree,
                                                 cfg.n_colors, cfg.rho, cfg.fall))

    rng = np.random.default_rng(0)
    n = {k: 0 for k in ("inst", "multiround", "vk", "covered", "deduced_any",
                        "single", "single_deduced", "rule_proper", "has_b",
                        "placebo", "ok")}
    nrounds, ndeduced, nknown, ncovered = [], [], [], []
    for adj, colors, hints, trace in build_instances(cfg, N_INST, 600_000_000):
        n["inst"] += 1
        nrounds.append(len(trace["rounds"]))
        if len(trace["rounds"]) < 2:
            continue
        n["multiround"] += 1
        for k, rnd in enumerate(trace["rounds"]):
            if k < 1:
                continue
            for v, c in rnd:
                v, c = int(v), int(c)
                n["vk"] += 1
                avail, deduced = revealed_before(trace, hints, v)
                ndeduced.append(len(deduced))
                nbrs = [int(u) for u in adj[v]]
                known = {u: avail[u] for u in nbrs if u in avail}
                covered = sorted(set(known.values()))
                nknown.append(len(known))
                ncovered.append(len(covered))
                if len(covered) != Q - 1 or c in covered:
                    continue
                n["covered"] += 1
                if any(u in deduced for u in known):
                    n["deduced_any"] += 1
                singles = [(col, [u for u in known if known[u] == col])
                           for col in covered]
                sing = [(col, us[0]) for col, us in singles if len(us) == 1]
                if sing:
                    n["single"] += 1
                bind = [(col, u) for col, u in sing if u in deduced]
                if not bind:
                    continue
                n["single_deduced"] += 1
                pick = None
                for a, w in bind:
                    if not _stays_proper(adj, avail, w, c):
                        continue
                    n["rule_proper"] += 1
                    others = [b for b in covered
                              if b != a and _stays_proper(adj, avail, w, b)]
                    if not others:
                        continue
                    n["has_b"] += 1
                    pick = (a, w, int(rng.choice(sorted(others))))
                    break
                if pick is None:
                    continue
                a, w, b = pick
                far = sorted(u for u in deduced
                             if u not in nbrs and u != v and u != w)
                for p in far:
                    opts = [col for col in range(Q)
                            if col != avail[p] and _stays_proper(adj, avail, p, col)]
                    if opts:
                        n["placebo"] += 1
                        break
                else:
                    continue
                n["ok"] += 1

    print()
    for k in ("inst", "multiround", "vk", "covered", "deduced_any", "single",
              "single_deduced", "rule_proper", "has_b", "placebo", "ok"):
        print("  %-16s %d" % (k, n[k]))
    print()
    for name, xs in (("rounds/instance", nrounds), ("|deduced| at v", ndeduced),
                     ("|known nbrs|", nknown), ("|covered|", ncovered)):
        if xs:
            print("  %-16s min %d  median %d  max %d"
                  % (name, min(xs), sorted(xs)[len(xs) // 2], max(xs)))


if __name__ == "__main__":
    main()
