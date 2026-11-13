"""One edit, two positions: the same-edit dissociation finding 68 said it was missing.

WHAT FINDING 68 LEFT OPEN, IN ITS OWN WORDS. The rule test recolours the binding
hint w from the freed colour `a` to v's forced colour `c`, and reads the COLOUR
position: the answer follows, 350/350, in both arms. The guard test recolours the
same hint from `a` to an already-covered colour `b`, and reads the NODE position:
the tied model keeps naming v, 1.0000 at every seed, where the untied model stops.
Those are two DIFFERENT edits read at two DIFFERENT positions. The honest reading
of the pair is therefore "responds at the colour position, not at the node
position", which is weaker than a dissociation, because nothing rules out that the
w->b edit simply never reaches the tied model at all. If it does not, the right
conclusion is not "no guard" but "insensitive to this edit", and finding 68's
headline would be bounded accordingly.

THE TEST. Take finding 68's OWN cases -- the same case finder, the same rng, the
same instance stream, so the case list is byte-identical and every number here is
paired against a published one -- and write FOUR one-token variants of the one
sequence, three of them at the SAME token position:

  clean     nothing changed                    v forced to c; a excluded by w
  rule      hint w: a -> c                     v forced to a; c now excluded
  broken    hint w: a -> b  (b already covered) v NOT forced; candidates {c, a}
  placebo   a non-neighbour hint, proper       v forced to c; nothing about v moves

Each differs from clean in exactly one token, asserted. Then read BOTH positions
on each of the four:

  node position   (node_token_position)      next-token prediction is v's identity
  colour position (that index + 1)           next-token prediction is v's colour

v's node token is physically present in all four sequences -- the edits are to
hint colours earlier in the prefix -- so the colour readout is teacher-forced on
"v is the node", which is exactly the conditional DD's rule computes.

WHAT EACH ARM OF THE 2x2 PREDICTS. The quantity is `p_freed`, the model's
probability on colour `a` at v's colour position. Under DD:

  clean     p_freed ~ 0      a is held by w, so a is eliminated
  placebo   p_freed ~ 0      the edit is irrelevant to v
  rule      p_freed ~ 1      a is the unique survivor; this is finding 27 again
  broken    0 < p_freed      a is a live candidate again, sharing mass with c

THE HEADLINE is the broken cell at the colour position, because it is the one
measurement that no existing run has made. If the tied model's p_freed RISES under
the same edit whose node readout it ignores, then one edit moves the elimination
and leaves the schedule untouched: a same-edit dissociation, and finding 68's
decomposition survives at full strength. If p_freed does NOT rise, the edit never
reached the colour computation either, the tied model is simply insensitive to
w->b, and finding 68's claim has to be narrowed to the weaker reading. Both
outcomes are reportable and the second is the falsifier, pre-registered in
scripts/dissoc_score.py so that a null here cannot be re-described later as a
different question.

NOTHING IS TRAINED. Six existing checkpoints, read twice each; the construction,
the instances and the case list come from sudoku_skip unchanged.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .sudoku_skip import (_is_forced, _revealed_before, find_unforceable_deductions,
                          node_token_position)
from .sudoku_tokens import encode_instance
from .train_sudoku_transformer import build_instances, load_checkpoint

# The four one-token variants, in the order every table and figure prints them.
# clean first because it is the reference, then the two controls, then the edit
# the finding turns on -- so a reader meets the null before the headline.
VARIANTS = ("clean", "placebo", "rule", "broken")


def rule_edit_valid(adj, trace, hints, case):
    """Can hint w be recoloured to v's forced colour c on THIS case?

    The rule edit is only defined when it keeps the revealed colouring proper (no
    revealed neighbour of w already shows c) and actually leaves v forced to the
    freed colour a. Cases that fail are kept in the CSV with rule_valid false and
    excluded from the rule statistics by the scorer, never dropped here: a case
    list that changes with the question is not a paired case list.
    """
    w, c, a = case["recolored_nbr"], case["true_color"], case["freed_color"]
    avail = _revealed_before(trace, hints, case["node"])
    if c in {avail[u] for u in (int(x) for x in adj[w]) if u in avail}:
        return False
    edited = dict(avail)
    edited[w] = c
    if not _is_forced(adj, edited, case["node"], case["n_colors"]):
        return False
    # the survivor must be a, i.e. the edit flips the answer rather than keeping it
    covered = {edited[u] for u in (int(x) for x in adj[case["node"]]) if u in edited}
    return a not in covered


def build_quad(adj, trace, hints, vocab, case):
    """{variant: tokens} plus the node position. Each variant is clean with
    exactly one token rewritten; rule and broken rewrite the SAME token."""
    clean = encode_instance(adj, trace, vocab)
    pos = node_token_position(clean, vocab, case["node"])
    assert vocab.is_node_token(clean[pos + 1]), "colour position must follow a node token"
    assert int(vocab.node_index(clean[pos + 1])) == case["node"]
    out = {"clean": clean}
    edits = {
        "placebo": (case["placebo_color_pos"], case["placebo_color"]),
        "rule": (case["hint_color_pos"], case["true_color"]),
        "broken": (case["hint_color_pos"], case["new_color"]),
    }
    for name, (p, col) in edits.items():
        assert vocab.is_color_token(clean[p]), "expected a hint colour token"
        t = clean.copy()
        t[p] = vocab.color_token(col)
        assert (clean != t).sum() == 1, "%s must be a one-token change" % name
        out[name] = t
    assert case["hint_color_pos"] != case["placebo_color_pos"]
    return out, pos


def _read(logits, pos, vocab, v_tok, c, a):
    """The two readouts at one variant: who gets named, and what colour v gets."""
    node_row = logits[0, pos]
    col_row = logits[0, pos + 1]
    probs = torch.softmax(col_row.float(), dim=-1)
    col_toks = [vocab.color_token(i) for i in range(vocab.n_colors)]
    col_probs = [float(probs[t]) for t in col_toks]
    top = int(np.argmax(col_probs))
    return {
        "emits_v": int(node_row.argmax()) == v_tok,
        "v_logit": float(node_row[v_tok]),
        "color": top,
        "p_true": col_probs[c],
        "p_freed": col_probs[a],
        # mass on the two colours DD leaves live under the broken hints. Under
        # clean/placebo/rule only one of them is live, so this is 1 - (mass on the
        # colours still covered) and says whether elimination ran at all.
        "p_cand": col_probs[c] + col_probs[a],
        "p_color": float(sum(col_probs)),
    }


def run_dissoc(model, vocab, device, cases):
    rows = []
    for adj, colors, hints, trace, case in cases:
        quad, pos = build_quad(adj, trace, hints, vocab, case)
        v_tok = vocab.node_token(case["node"])
        c, a = case["true_color"], case["freed_color"]
        with torch.no_grad():
            reads = {}
            for name in VARIANTS:
                lg = model(torch.from_numpy(quad[name]).unsqueeze(0).to(device))
                reads[name] = _read(lg, pos, vocab, v_tok, c, a)
        row = {
            "node": case["node"], "round0_rank": case["round0_rank"],
            "binding_nbr": case["recolored_nbr"], "n_known": case["n_known"],
            "true_color": c, "freed_color": a, "new_color": case["new_color"],
            "rule_valid": case["rule_valid"],
        }
        for name in VARIANTS:
            for k, val in reads[name].items():
                row["%s_%s" % (name, k)] = val
        # The three registered per-case events.
        row["rule_flips"] = reads["rule"]["color"] == a            # H1
        row["broken_frees"] = (reads["broken"]["p_freed"]          # H2, the sign test
                               > reads["placebo"]["p_freed"])
        row["dissoc"] = bool(reads["broken"]["emits_v"]) and bool(row["broken_frees"])
        rows.append(row)
    return rows


def _mean(rows, key, cond=None):
    g = [r for r in rows if cond is None or r[cond]]
    return (float(np.mean([r[key] for r in g])), len(g)) if g else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n-instances", type=int, default=400)
    ap.add_argument("--n-cases", type=int, default=350)
    ap.add_argument("--seed-start", type=int, default=600_000_000,
                    help="finding 68's stream, unchanged, so the case list is identical")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=str, required=True)
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

    # Identical to sudoku_skip.main: same rng, same stream, same finder, same cap.
    rng = np.random.default_rng(0)
    cases = []
    for adj, colors, hints, trace in build_instances(sudoku_cfg, args.n_instances, args.seed_start):
        for case in find_unforceable_deductions(adj, colors, hints, trace, vocab, rng):
            case["n_colors"] = vocab.n_colors
            case["rule_valid"] = rule_edit_valid(adj, trace, hints, case)
            cases.append((adj, colors, hints, trace, case))
    print("%d breakable round-0 cases from %d instances" % (len(cases), args.n_instances))
    if not cases:
        raise SystemExit("no breakable cases found")
    cases = cases[:args.n_cases]
    os.makedirs(args.out_dir, exist_ok=True)

    rows = run_dissoc(model, vocab, args.device, cases)
    n = len(rows)
    n_rule = sum(r["rule_valid"] for r in rows)

    print("\nONE EDIT, TWO POSITIONS -- %d cases (%d with a defined rule edit)" % (n, n_rule))
    print("  variant    emits_v   p_true   p_freed   p_cand   argmax=freed")
    for name in VARIANTS:
        ev, _ = _mean(rows, "%s_emits_v" % name)
        pt, _ = _mean(rows, "%s_p_true" % name)
        pf, _ = _mean(rows, "%s_p_freed" % name)
        pc, _ = _mean(rows, "%s_p_cand" % name)
        af = float(np.mean([r["%s_color" % name] == r["freed_color"] for r in rows]))
        print("  %-9s  %7.4f  %7.4f  %8.4f  %7.4f  %11.4f" % (name, ev, pt, pf, pc, af))

    flips, _ = _mean(rows, "rule_flips", cond="rule_valid")
    print("\n  H1 rule_flips (rule-valid cases only)   %.4f   (bar 0.60, chance 0.25)" % flips)

    d = [r["broken_p_freed"] - r["placebo_p_freed"] for r in rows]
    b = sum(1 for x in d if x > 0)
    c_ = sum(1 for x in d if x < 0)
    print("  H2 p_freed broken - placebo: median %+.4f  mean %+.4f  b=%d c=%d"
          % (float(np.median(d)), float(np.mean(d)), b, c_))
    print("     dissoc (names v AND frees a)         %.4f" % _mean(rows, "dissoc")[0])
    pl = [r["placebo_p_freed"] - r["clean_p_freed"] for r in rows]
    print("  H4 p_freed placebo - clean:  mean %+.4f  (specificity, |.| <= 0.05)"
          % float(np.mean(pl)))

    with open(os.path.join(args.out_dir, "dissoc_behavior.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"checkpoint": args.checkpoint, "n_cases": n, "n_rule_valid": n_rule,
               "seed_start": args.seed_start, "rule_flips": flips,
               "dissoc": _mean(rows, "dissoc")[0],
               "sign_b": b, "sign_c": c_,
               "median_freed_diff": float(np.median(d)),
               "placebo_minus_clean": float(np.mean(pl))}
    for name in VARIANTS:
        for k in ("emits_v", "p_true", "p_freed", "p_cand"):
            summary["%s_%s" % (name, k)] = _mean(rows, "%s_%s" % (name, k))[0]
    with open(os.path.join(args.out_dir, "dissoc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote %s/dissoc_behavior.csv (%d rows)" % (args.out_dir, n))


if __name__ == "__main__":
    main()
