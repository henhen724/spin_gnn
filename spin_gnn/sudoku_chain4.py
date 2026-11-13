"""Does the shared block eliminate against facts it DERIVED? Finding 69, moved off round 0.

WHAT FINDING 69 LEFT OPEN, IN ITS OWN WORDS. Its scope page says "Round 0 only":
both of its edits require every known neighbour of v to be a HINT, so every fact
the model eliminated against was one it was GIVEN. Direct deduction is iterative,
and a node forced in round k >= 1 is forced partly by colours the model itself
emitted in earlier rounds. Applying the rule once to given facts and running the
algorithm over its own output are different claims, and only the first is
measured. Behaviourally the model must be integrating rounds somehow -- finding
23 has every depth bin at 1.0000, and a node that only becomes determined after
five rounds cannot be got right otherwise -- but that is an inference from
accuracy, not a causal measurement, and this project has been burned by exactly
that step before.

THE TEST is finding 69's, with one thing changed: the edited token is a colour
the model DEDUCED, not a colour it was given. Take v deduced in round k >= 1
whose visible neighbours cover q-1 = 3 colours, where the freed colour a is held
by exactly one visible neighbour w AND w was itself emitted as a deduction
earlier in the prefix. Write the same four one-token variants, three of them at
the SAME token index, all of them inside the reasoning section:

  clean     nothing changed                     v forced to c; a excluded by w
  rule      deduced w: a -> c                   v forced to a; c now excluded
  broken    deduced w: a -> b  (b also covered) v NOT forced; candidates {c, a}
  placebo   another deduced non-neighbour       v forced to c; nothing about v moves

and read BOTH positions on each, with finding 69's own `_read`, imported rather
than reimplemented so the two measurements cannot drift:

  node position   (v's node-token index - 1)   next-token prediction is v's identity
  colour position (v's node-token index)       next-token prediction is v's colour

WHAT THE COMPARISON IS. Finding 69 measured rule-flip 1.0000 in all six cells and
p_freed(broken) 0.5253 / 0.5228 / 0.5354 for the tied arm. If the chained numbers
match, the rule runs over derived facts exactly as it runs over given ones, and
"it chains" stops being an inference from accuracy. If the chained rule-flip is
materially lower, the provenance of the eliminated fact matters, which would be
the first thing this project has found that distinguishes the two.

The comparison is CELL-LEVEL, not paired per case: these cases are round k >= 1
with a deduced binding neighbour, so by construction they are different cases
from finding 69's. Same checkpoints, same readout, same bars -- different case
list, and nothing here may be reported as a paired difference.

AMENDED 2026-11-13 AFTER THE CENSUS RETURNED ZERO, AND WHY THAT IS NOT P-HACKING.
The first version required the edited node to stay LOCALLY PROPER, as finding 69
does. Job 52231126's census found 0 cases in 1200 instances, and the reason is a
theorem, not a thin tail: 1926 of 1997 candidate deductions reached that filter
and every single one failed it. A node w is deduced to colour a precisely
because its visible neighbours already cover every colour but a, so EVERY
alternative colour for w is held by at least one neighbour of w, and later
rounds only add more. Requiring the edited node to be derived and requiring it
to stay locally proper are mutually exclusive. The set of locally consistent
single-token edits to the reasoning section is EMPTY.

The amendment is therefore forced, and it was made without seeing any model
output: `--cases-only` runs no forward pass, so the census is a property of the
instance generator alone and the amended construction cannot have been tuned to
any checkpoint's behaviour. What changed is three lines -- the edit minimises
its visible clashes instead of forbidding them, the count is recorded, and the
PLACEBO IS MATCHED TO THAT COUNT. Everything scored stays as registered.

HONEST CAVEAT, STATED UP FRONT AND CARRIED INTO THE DECK. Editing an emitted
deduction shows the model a reasoning step it would not itself have written, so
the prefix is inconsistent in a way a hint edit is not -- globally against the
planted colouring, and now, unavoidably, locally at the edited node too. A model
that notices that and reverts to its own value would score as "does not chain"
while chaining perfectly in free generation, so a LOW flip rate is ambiguous.
Two things bound it. `rule_reverts` says whether the model falls back on c: that
is the signature of detection, a third colour is neither. And the placebo is now
the same KIND of object as the edit -- a derived token recoloured, clashing with
the same number of visible neighbours, just nowhere near v -- so if the model
merely reacts to a corrupted prefix, the placebo moves too and C4 fails. A HIGH
flip rate has no ambiguity at all: it can only mean the edited fact was used.

NOTHING IS TRAINED. Six existing checkpoints, read four times each.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .sudoku_chain import reasoning_node_positions
from .sudoku_dissoc import VARIANTS, _mean, _read
from .sudoku_tokens import encode_instance
from .train_sudoku_transformer import build_instances, load_checkpoint


def revealed_before(trace, hints, node):
    """(everything visible before node's token, the DEDUCED part of it).

    sudoku_skip._revealed_before stops after round 0, which is exactly the
    restriction this module exists to lift. Order is token order: rounds in
    sequence, and within a round the order the trace emits them, because that is
    what the causal mask lets the model see at node's position.
    """
    avail, deduced = dict(hints), {}
    for rnd in trace["rounds"]:
        for u, c in rnd:
            u, c = int(u), int(c)
            if u == node:
                return avail, deduced
            avail[u] = c
            deduced[u] = c
    return avail, deduced


def _clashes(adj, avail, node, color):
    """How many visible neighbours of `node` already hold `color`.

    THE CENSUS THEOREM (job 52231126, zero cases from 1200 instances). For a node
    the model DEDUCED this count is never zero, whatever colour you pick. A node
    w is deduced to colour a precisely because its visible neighbours cover every
    other colour, so each alternative has at least one holder sitting next to w,
    and later rounds only reveal more. Requiring the edited node to stay locally
    proper and requiring it to be derived are therefore MUTUALLY EXCLUSIVE: the
    set of locally consistent single-token edits to the reasoning section is
    empty.

    So the edit cannot avoid a visible clash and the measurement stops pretending
    to. It minimises the count instead, records it, and matches the placebo to
    it -- see find_chained_quads.
    """
    return sum(1 for u in adj[node] if avail.get(int(u)) == color)


def _stays_proper(adj, avail, node, color):
    """Kept for the census script, which uses it to re-derive the theorem."""
    return _clashes(adj, avail, node, color) == 0


def find_chained_quads(adj, colors, hints, trace, vocab, rng, max_per_round=2):
    """Round k>=1 deductions that one edit to an EARLIER DEDUCTION can flip or break.

    Every returned case carries all four variants' ingredients, so a case either
    supports the whole 2x2 or is not a case. Cases are collected in trace order,
    so the list is a deterministic function of the instance stream.
    """
    q = vocab.n_colors
    out = []
    if len(trace["rounds"]) < 2:
        return out
    for k, rnd in enumerate(trace["rounds"]):
        if k < 1:
            continue
        taken = 0
        for rank, (v, c) in enumerate(rnd):
            if taken >= max_per_round:
                break
            v, c = int(v), int(c)
            avail, deduced = revealed_before(trace, hints, v)
            nbrs = [int(u) for u in adj[v]]
            known = {u: avail[u] for u in nbrs if u in avail}
            covered = sorted(set(known.values()))
            if len(covered) != q - 1 or c in covered:
                continue
            # the binding neighbour must be one the model DEDUCED, not one it was
            # given -- that single condition is the whole difference from finding 69
            singles = [(col, [u for u in known if known[u] == col]) for col in covered]
            bind = [(col, us[0]) for col, us in singles if len(us) == 1 and us[0] in deduced]
            # A visible clash at w is unavoidable -- see _clashes. Take the
            # binding neighbour whose rule edit is least visibly wrong, ties by
            # node id, so the case list stays a deterministic function of the
            # instance stream and no rng decides what gets measured.
            if not bind:
                continue
            a, w = min(bind, key=lambda aw: (_clashes(adj, avail, aw[1], c), aw[1]))
            rule_cl = _clashes(adj, avail, w, c)
            others = sorted(b for b in covered if b != a)
            if not others:
                continue
            b = min(others, key=lambda col: (_clashes(adj, avail, w, col), col))
            broken_cl = _clashes(adj, avail, w, b)
            # The placebo is another DEDUCED node, so it is the same kind of
            # token in the same section as the edit it is the null for -- AND,
            # since the real edit must clash, the placebo is matched on how many
            # neighbours it clashes with. That is what makes it a null for
            # "the model noticed a corrupted prefix" and not only for relevance.
            far = sorted(u for u in deduced if u not in nbrs and u != v and u != w)
            cands = []
            for p in far:
                for col in range(q):
                    if col != avail[p]:
                        cands.append((p, col, _clashes(adj, avail, p, col)))
            if not cands:
                continue
            p, pcol, placebo_cl = min(cands, key=lambda t: (abs(t[2] - rule_cl),
                                                            t[2], t[0], t[1]))
            placebo = (p, pcol)
            out.append({
                "node": v, "round": k, "round_rank": rank, "true_color": c,
                "freed_color": a, "new_color": b, "recolored_nbr": w,
                "nbr_round": next(kk for kk, rr in enumerate(trace["rounds"])
                                  if any(int(u) == w for u, _ in rr)),
                "placebo_node": placebo[0], "placebo_color": placebo[1],
                "rule_clashes": rule_cl, "broken_clashes": broken_cl,
                "placebo_clashes": placebo_cl,
                "clash_matched": int(placebo_cl == rule_cl),
                "n_known": len(known), "n_colors": q,
                "v_first_in_round": rank == 0,
            })
            # A per-ROUND cap rather than a per-instance one. The old cap of 4
            # per instance filled up inside round 1 and the depth figure got
            # r3=15 out of 611, because rounds are scanned in order. Capping
            # each round separately spends the same budget across depths, which
            # is the one axis this measurement adds over finding 69. Decided
            # from the case census alone -- no forward pass, no model output.
            taken += 1
    return out


def build_quad_chain(adj, trace, vocab, case):
    """{variant: tokens} plus the node position. Every edit is one token, inside
    the reasoning section, strictly before v's own node token."""
    clean = encode_instance(adj, trace, vocab)
    pmap = reasoning_node_positions(clean, vocab)
    v_idx = pmap[case["node"]][0]
    pos = v_idx - 1
    w_pos = pmap[case["recolored_nbr"]][0] + 1
    pl_pos = pmap[case["placebo_node"]][0] + 1
    assert w_pos < v_idx and pl_pos < v_idx, "every edit must be in v's prefix"
    assert w_pos != pl_pos
    assert vocab.is_color_token(clean[w_pos]) and vocab.is_color_token(clean[pl_pos])
    assert int(vocab.color_index(clean[w_pos])) == case["freed_color"]
    assert int(vocab.color_index(clean[pl_pos])) != case["placebo_color"]
    assert vocab.is_node_token(clean[pos + 1])
    assert int(vocab.node_index(clean[pos + 1])) == case["node"]
    out = {"clean": clean}
    for name, (p, col) in (("placebo", (pl_pos, case["placebo_color"])),
                           ("rule", (w_pos, case["true_color"])),
                           ("broken", (w_pos, case["new_color"]))):
        t = clean.copy()
        t[p] = vocab.color_token(col)
        assert (clean != t).sum() == 1, "%s must be a one-token change" % name
        out[name] = t
    return out, pos


def run_chain4(model, vocab, device, cases):
    rows = []
    for adj, colors, hints, trace, case in cases:
        quad, pos = build_quad_chain(adj, trace, vocab, case)
        v_tok = vocab.node_token(case["node"])
        c, a = case["true_color"], case["freed_color"]
        with torch.no_grad():
            reads = {}
            for name in VARIANTS:
                lg = model(torch.from_numpy(quad[name]).unsqueeze(0).to(device))
                reads[name] = _read(lg, pos, vocab, v_tok, c, a)
        row = {
            "node": case["node"], "round": case["round"],
            "round_rank": case["round_rank"], "binding_nbr": case["recolored_nbr"],
            "nbr_round": case["nbr_round"], "n_known": case["n_known"],
            "true_color": c, "freed_color": a, "new_color": case["new_color"],
            "placebo_node": case["placebo_node"],
            "v_first_in_round": case["v_first_in_round"],
            # how visibly wrong each edit is: a derived token cannot be edited
            # without clashing with at least one visible neighbour (_clashes),
            # so these are >= 1 by construction and the placebo is matched
            "rule_clashes": case["rule_clashes"],
            "broken_clashes": case["broken_clashes"],
            "placebo_clashes": case["placebo_clashes"],
            "clash_matched": case["clash_matched"],
        }
        for name in VARIANTS:
            for key, val in reads[name].items():
                row["%s_%s" % (name, key)] = val
        row["rule_flips"] = reads["rule"]["color"] == a           # C1
        row["rule_reverts"] = reads["rule"]["color"] == c         # C3, detection
        row["broken_frees"] = (reads["broken"]["p_freed"]         # C2, the sign test
                               > reads["placebo"]["p_freed"])
        row["dissoc"] = bool(reads["broken"]["emits_v"]) and bool(row["broken_frees"])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n-instances", type=int, default=1200)
    ap.add_argument("--n-cases", type=int, default=300)
    ap.add_argument("--seed-start", type=int, default=600_000_000,
                    help="findings 68/69's stream, so the chained cases come from "
                         "the same instances as the round-0 ones")
    ap.add_argument("--cases-only", action="store_true",
                    help="build the case list, report the yield and its round "
                         "profile, run no forward pass and write nothing -- the "
                         "census step of the launcher, so a thin yield costs a "
                         "minute instead of six cells")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    model, vocab, config = load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    sudoku_cfg = SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        rho=config["rho"], rho_max=config["rho_max"],
        scratchpad=config.get("scratchpad", False), hint_mode=config.get("hint_mode", "removal"),
        shuffle_adj=bool(config.get("shuffle_adj", False)),
        shuffle_events=bool(config.get("shuffle_events", False)),
        fall=bool(config.get("fall", False)),
    )

    # Same rng discipline as sudoku_dissoc.main: one stream, seeded 0, advanced
    # only by the finder, so the case list is a function of the instances alone.
    rng = np.random.default_rng(0)
    cases = []
    for adj, colors, hints, trace in build_instances(sudoku_cfg, args.n_instances,
                                                     args.seed_start):
        for case in find_chained_quads(adj, colors, hints, trace, vocab, rng):
            cases.append((adj, colors, hints, trace, case))
    print("%d chained cases (round>=1, binding neighbour DEDUCED) from %d instances"
          % (len(cases), args.n_instances))
    if args.cases_only:
        depth = {}
        for _, _, _, _, case in cases[:args.n_cases]:
            depth[case["round"]] = depth.get(case["round"], 0) + 1
        print("  by round: " + "  ".join("r%d=%d" % (k, depth[k]) for k in sorted(depth)))
        return
    if len(cases) < args.n_cases:
        raise SystemExit("only %d cases, need %d -- raise --n-instances"
                         % (len(cases), args.n_cases))
    cases = cases[:args.n_cases]
    os.makedirs(args.out_dir, exist_ok=True)

    rows = run_chain4(model, vocab, args.device, cases)
    n = len(rows)

    print("\nCHAINED, ONE EDIT TO A DERIVED FACT -- %d cases" % n)
    print("  variant    emits_v   p_true   p_freed   p_cand   argmax=freed")
    for name in VARIANTS:
        ev, _ = _mean(rows, "%s_emits_v" % name)
        pt, _ = _mean(rows, "%s_p_true" % name)
        pf, _ = _mean(rows, "%s_p_freed" % name)
        pc, _ = _mean(rows, "%s_p_cand" % name)
        af = float(np.mean([r["%s_color" % name] == r["freed_color"] for r in rows]))
        print("  %-9s  %7.4f  %7.4f  %8.4f  %7.4f  %11.4f" % (name, ev, pt, pf, pc, af))

    print("\n  C1 rule_flips    %.4f   (bar 0.60, chance 0.25, round-0 was 1.0000)"
          % _mean(rows, "rule_flips")[0])
    print("  C3 rule_reverts  %.4f   (the 'detected the edit' signature)"
          % _mean(rows, "rule_reverts")[0])
    d = [r["broken_p_freed"] - r["placebo_p_freed"] for r in rows]
    b = sum(1 for x in d if x > 0)
    c_ = sum(1 for x in d if x < 0)
    print("  C2 p_freed broken - placebo: median %+.4f  mean %+.4f  b=%d c=%d"
          % (float(np.median(d)), float(np.mean(d)), b, c_))
    print("     dissoc (names v AND frees a)   %.4f" % _mean(rows, "dissoc")[0])
    pl = [r["placebo_p_freed"] - r["clean_p_freed"] for r in rows]
    print("  C4 p_freed placebo - clean:  mean %+.4f  (|.| <= 0.05)" % float(np.mean(pl)))

    with open(os.path.join(args.out_dir, "chain4_behavior.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"checkpoint": args.checkpoint, "n_cases": n, "seed_start": args.seed_start,
               "rule_flips": _mean(rows, "rule_flips")[0],
               "rule_reverts": _mean(rows, "rule_reverts")[0],
               "dissoc": _mean(rows, "dissoc")[0], "sign_b": b, "sign_c": c_,
               "median_freed_diff": float(np.median(d)),
               "placebo_minus_clean": float(np.mean(pl))}
    for name in VARIANTS:
        for key in ("emits_v", "p_true", "p_freed", "p_cand"):
            summary["%s_%s" % (name, key)] = _mean(rows, "%s_%s" % (name, key))[0]
    with open(os.path.join(args.out_dir, "chain4_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote %s/chain4_behavior.csv (%d rows)" % (args.out_dir, n))


if __name__ == "__main__":
    main()
