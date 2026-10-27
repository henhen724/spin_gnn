"""Where the depth cost lives: picking the node, or picking its color.

48863464 showed solve_rate falling with DD round count T at pinned hint
density, while 48861799 showed the frontier predicate decoded at 0.863 AUROC
from epoch 5 onward and never improving. Both cannot be the whole story: a
model that already knows which nodes are deducible, and whose knowledge of
that does not change over training, should not care how many rounds an
instance needs -- unless what degrades with depth is something else.

The reasoning grammar emits alternating (node, color) pairs inside
BEGIN_ROUND-delimited rounds, so every target token in that section is either
a SELECTION slot (which node to deduce next) or a COLORING slot (what color
that node takes, given the chain so far). Those are the two halves of the
candidate explanation, and one teacher-forced pass over an instance labels
every slot with its round index. If the cost is in chaining colors, coloring
accuracy falls with round index while selection accuracy does not.

Teacher forcing is the point, not a convenience: the prefix is always the
correct chain, so a drop at round r is the model failing to use a correct
r-1-round chain, not error compounding from its own earlier mistakes. That
makes this a strictly harder test of the chaining hypothesis than generation
would be -- compounding alone would produce the same curve under generation
with no per-round weakness at all.

THE CONFOUND, and the one number that decides whether this works: round index
and position in the reasoning section advance together, so a decline with
round index could be nothing but a decline with sequence position. What
rescues the measurement is that rounds have different sizes across instances,
so the same reasoning position occurs at different round indices. Accuracy is
therefore also reported within position bins, and `round_effect_within_position`
is the slope of accuracy against round index after position is held fixed. If
that is flat while the marginal slope is not, the marginal slope is position,
not depth, and the depth reading is void. Read it before the marginal curve.

That control covers a second confound for free, which is why position and not
round count is the thing held fixed. The number of still-uncolored nodes the
model is choosing among is n_nodes - n_hints - (pairs emitted so far), and at a
pinned rho the hint count is constant, so the candidate count is a
deterministic function of reasoning position within every instance. Holding
position fixed therefore holds the candidate count fixed too, and a selection
accuracy that rises with round index -- which it does, marginally -- cannot be
dismissed or explained as simply having fewer nodes left to choose from.

Usage:
    python -m spin_gnn.sudoku_depth_slots --checkpoint <dir> --rho 0.6 0.75 \\
        --n-instances 400 --out-csv slots.csv
"""
import argparse
import csv
import os
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .sudoku_tokens import encode_instance
from .train_sudoku_transformer import build_instances, build_vocab, load_checkpoint

SELECTION, COLORING = "selection", "coloring"


def label_reasoning_slots(tokens, vocab):
    """Label every reasoning-section target token with its slot and round.

    Returns a list of dicts with keys: target_index (index into `tokens` of the
    token being predicted), slot, round_idx (1-based), pos_in_reasoning
    (0-based count of reasoning targets before it), is_first_in_round.

    `is_first_in_round` is tracked because within a round the emission order is
    sorted by node id, so every selection but the first is nearly determined by
    the previous one; the first selection of a round is the only one that
    requires knowing the round ended and which nodes the new frontier contains.
    Pooling the two would average a near-deterministic slot with the one slot
    that actually asks the frontier question.

    Walks the emitted token stream rather than the trace, so the labels are
    exactly the positions the model is scored at. Only the no-scratchpad
    grammar is handled; scratchpad sequences interleave evidence tokens and
    are rejected rather than mislabeled.
    """
    if getattr(vocab, "scratchpad", False):
        raise ValueError("scratchpad grammar has no plain (node, color) slot structure")
    try:
        start = int(np.flatnonzero(tokens == vocab.BEGIN_REASONING)[0])
    except IndexError:
        raise ValueError("sequence has no BEGIN_REASONING token")

    slots, round_idx, expect, fresh_round = [], 0, SELECTION, False
    for i in range(start + 1, len(tokens)):
        tok = int(tokens[i])
        if tok == vocab.BEGIN_ROUND:
            round_idx += 1
            fresh_round = True
            # A round boundary resets the pairing; a half-finished pair here
            # would mean the grammar is not what this function assumes.
            if expect is not SELECTION:
                raise ValueError(f"BEGIN_ROUND mid-pair at token {i}")
            continue
        if tok in (vocab.SOLVED, vocab.UNRESOLVED):
            break
        slots.append({
            "target_index": i,
            "slot": expect,
            "round_idx": round_idx,
            "pos_in_reasoning": len(slots),
            "is_first_in_round": bool(fresh_round),
        })
        # Cleared only once the round's first (node, color) pair is complete,
        # so both members of that pair carry the flag.
        if expect is COLORING:
            fresh_round = False
        expect = COLORING if expect is SELECTION else SELECTION
    if expect is not SELECTION:
        raise ValueError("reasoning section ended mid-pair")
    return slots


def score_slots(model, vocab, device, instances, n_position_bins=6):
    """Teacher-forced per-slot correctness over `instances`. Returns rows."""
    rows = []
    model.eval()
    with torch.no_grad():
        for adj, _colors, _hints, trace in instances:
            tokens = encode_instance(adj, trace, vocab)
            slots = label_reasoning_slots(tokens, vocab)
            if not slots:
                continue
            x = torch.as_tensor(tokens[:-1], dtype=torch.long, device=device)[None, :]
            mask = torch.ones_like(x, dtype=torch.bool)
            pred = model(x, valid_mask=mask)[0].argmax(-1).cpu().numpy()
            n_reasoning = len(slots)
            for s in slots:
                i = s["target_index"]
                rows.append({
                    **s,
                    "T": len(trace["rounds"]),
                    "n_reasoning_slots": n_reasoning,
                    # Shared axis across instances of different length, so
                    # position can be held fixed when rounds cannot.
                    "frac_through_reasoning": s["pos_in_reasoning"] / n_reasoning,
                    "correct": int(pred[i - 1] == tokens[i]),
                })
    return rows


def by_first_in_round(rows, min_cell=30):
    """Accuracy per (slot, is_first_in_round, round_idx).

    Split out separately rather than folded into `summarize` because it asks a
    different question: not whether depth costs accuracy, but whether whatever
    cost exists is concentrated in the one slot per round that has to locate a
    new frontier instead of continuing a sorted run.
    """
    cells = defaultdict(list)
    for r in rows:
        cells[(r["slot"], bool(r["is_first_in_round"]), r["round_idx"])].append(r["correct"])
    out = []
    for (slot, first, rnd), c in sorted(cells.items(), key=lambda kv: (kv[0][0], not kv[0][1], kv[0][2])):
        if len(c) >= min_cell:
            out.append({"slot": slot, "is_first_in_round": first, "round_idx": rnd,
                        "n": len(c), "accuracy": float(np.mean(c))})
    return out


def _slope(x, y):
    """Least-squares slope of y on x, NaN when x has no spread."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or x.std() == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def summarize(rows, n_position_bins=6, min_cell=30):
    """Marginal accuracy by (slot, round), plus the position-controlled slope.

    The controlled slope is a within-position-bin average: inside each bin of
    frac_through_reasoning, accuracy is regressed on round index, and those
    slopes are averaged weighted by bin size. Bins where round index has no
    spread, or cells thinner than `min_cell`, contribute nothing -- a slope
    fitted to two round indices with nine samples is noise with a sign.
    """
    by_slot = defaultdict(list)
    for r in rows:
        by_slot[r["slot"]].append(r)

    marginal, controlled = [], {}
    for slot, rs in by_slot.items():
        acc_by_round = defaultdict(list)
        for r in rs:
            acc_by_round[r["round_idx"]].append(r["correct"])
        for rnd in sorted(acc_by_round):
            c = acc_by_round[rnd]
            if len(c) >= min_cell:
                marginal.append({
                    "slot": slot, "round_idx": rnd, "n": len(c),
                    "accuracy": float(np.mean(c)),
                })
        rounds = [r["round_idx"] for r in rs if len(acc_by_round[r["round_idx"]]) >= min_cell]
        accs = [r["correct"] for r in rs if len(acc_by_round[r["round_idx"]]) >= min_cell]
        marginal_slope = _slope(rounds, accs)

        edges = np.linspace(0, 1, n_position_bins + 1)
        bin_of = np.clip(np.digitize([r["frac_through_reasoning"] for r in rs], edges[1:-1]),
                         0, n_position_bins - 1)
        slopes, weights = [], []
        for b in range(n_position_bins):
            sel = [r for r, bb in zip(rs, bin_of) if bb == b]
            cells = defaultdict(list)
            for r in sel:
                cells[r["round_idx"]].append(r["correct"])
            keep = {k: v for k, v in cells.items() if len(v) >= min_cell}
            if len(keep) < 2:
                continue
            xs = np.concatenate([[k] * len(v) for k, v in keep.items()])
            ys = np.concatenate([v for v in keep.values()])
            s = _slope(xs, ys)
            if not np.isnan(s):
                slopes.append(s)
                weights.append(len(ys))
        controlled[slot] = {
            "marginal_slope": marginal_slope,
            "round_effect_within_position": (
                float(np.average(slopes, weights=weights)) if slopes else float("nan")
            ),
            "n_position_bins_usable": len(slopes),
            "n": len(rs),
            "accuracy": float(np.mean([r["correct"] for r in rs])),
        }
    return marginal, controlled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--rho", type=float, nargs="+", default=[0.6])
    ap.add_argument("--n-instances", type=int, default=400)
    ap.add_argument("--min-cell", type=int, default=30)
    ap.add_argument("--n-position-bins", type=int, default=6)
    ap.add_argument("--seed-start", type=int, default=40_000_000_000)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_rows = []
    for ckpt in args.checkpoint:
        model, vocab, config = load_checkpoint(ckpt, device=device)
        base_cfg = SudokuReasoningConfig(
            n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
            rho=config["rho"], rho_max=config["rho_max"],
            hint_mode=config.get("hint_mode", "removal"),
            scratchpad=config.get("scratchpad", False),
            # Analyzing a _shuf checkpoint without this feeds it the
            # color-sorted adjacency it was trained to ignore.
            shuffle_adj=bool(config.get("shuffle_adj", False)),
        )
        name = os.path.basename(ckpt.rstrip("/"))
        for rho in args.rho:
            cfg = replace(base_cfg, rho=float(rho), rho_max=float(rho))
            instances = build_instances(cfg, args.n_instances, args.seed_start)
            rows = score_slots(model, vocab, device, instances)
            marginal, controlled = summarize(rows, args.n_position_bins, args.min_cell)

            print(f"\n=== {name}  rho={rho}  ({len(rows)} slots, "
                  f"{len(instances)} instances) ===")
            for slot in (SELECTION, COLORING):
                c = controlled.get(slot)
                if c is None:
                    continue
                print(f"  {slot:>9}  overall acc {c['accuracy']:.4f}  n={c['n']}")
                print(f"             marginal slope vs round        {c['marginal_slope']:+.5f}")
                print(f"             slope at FIXED position        "
                      f"{c['round_effect_within_position']:+.5f}  "
                      f"({c['n_position_bins_usable']} usable position bins)")
                cells = [m for m in marginal if m["slot"] == slot]
                print("             by round: " + "  ".join(
                    f"r{m['round_idx']}={m['accuracy']:.3f}(n={m['n']})" for m in cells))
            split = by_first_in_round(rows, args.min_cell)
            for slot in (SELECTION, COLORING):
                for first in (True, False):
                    cells = [m for m in split if m["slot"] == slot and m["is_first_in_round"] is first]
                    if cells:
                        tag = "first-of-round" if first else "continuing     "
                        print(f"  {slot:>9} {tag}: " + "  ".join(
                            f"r{m['round_idx']}={m['accuracy']:.3f}(n={m['n']})" for m in cells))
            # One flat schema for both views, so the CSV can be read without
            # knowing which view produced a row.
            blank = {"view": "", "slot": "", "round_idx": "", "is_first_in_round": "",
                     "n": "", "accuracy": "", "marginal_slope": "",
                     "round_effect_within_position": "", "n_position_bins_usable": ""}
            for m in marginal:
                c = controlled[m["slot"]]
                out_rows.append({
                    "checkpoint": name, "rho": float(rho), **blank, "view": "by_round", **m,
                    "marginal_slope": c["marginal_slope"],
                    "round_effect_within_position": c["round_effect_within_position"],
                    "n_position_bins_usable": c["n_position_bins_usable"],
                })
            for m in split:
                out_rows.append({"checkpoint": name, "rho": float(rho), **blank,
                                 "view": "first_split", **m})

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
