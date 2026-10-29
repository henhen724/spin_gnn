"""The converse of finding 27: does it WITHHOLD a deduction that stops being forced?

Finding 27 settled one direction. Recolour the neighbour doing the forcing so
DD's answer changes c -> a, and the leak-free models follow on 350/350 round-0
cases and 0.980 on chained ones. But every case there is a node DD *does*
emit, selected because its forcing set covers exactly q-1 colours with a unique
binding neighbour. That is DD's elimination rule by construction, and it is the
caveat finding 27 carries: the test only ever asks the model to fire.

DD is not just a rule, it is a rule plus a GUARD -- emit v only when v is
actually determined. A model that wrote a colour for whichever unknown node
came next in index order would score 1.000 on finding 27's intervention and
still not be running DD, because it would be guessing on the undetermined ones.
Behaviourally that model cannot exist at solve_rate 1.0000, but "cannot exist"
is an inference from accuracy, which is precisely the step this project has
been burned by twice (findings 15 and 23).

THE TEST. Take finding 27's round-0 case: v forced because its known
neighbours cover q-1 colours, with colour a held by exactly one hint w. Instead
of recolouring w to c (which keeps v forced and flips the answer to a), recolour
w to b, another colour ALREADY covered by a different neighbour. Then a is held
by nobody, the forcing set drops to q-2 colours, and v has two candidates left:
DD does not emit v in this round at all. One token changes, as before.

The measurement moves one position left. Finding 27 reads the logits AT v's
node token, which predict v's colour. This reads the logits at the token BEFORE
it, which predict the node identity itself:

  clean_emits_v     the model names v next, as DD does           (expect ~1.0)
  corrupt_emits_v   it still names v after v stopped being forced  <- HEADLINE
  pred_is_forced    when it names someone else, that node really is forced
                    under the corrupt hints

SPECIFICITY CONTROL, built in. A one-token edit anywhere might perturb the node
stream for reasons that have nothing to do with determinacy. So every case also
gets a PLACEBO: recolour a hint that is NOT a neighbour of v, to a colour that
keeps the revealed colouring proper. v stays forced, DD's round is unchanged,
and the model should still name v. `placebo_emits_v` is therefore the upper
baseline that `corrupt_emits_v` has to be read against; without it a low flip
rate would be uninterpretable.

A LOW `corrupt_emits_v` with a HIGH `placebo_emits_v` means the guard is real
and causal: the model tracks which nodes are determined, not just what colour a
determined node takes. A HIGH `corrupt_emits_v` in both means the node schedule
is blind -- index order with no determinacy check -- which would bound the
verdict to "DD's rule, not DD's search" far more sharply than the `min_unknown`
hand rule does, since that is a probe result and this would be a causal one.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .sudoku_tokens import encode_instance
from .train_sudoku_transformer import build_instances, load_checkpoint


def _revealed_before(trace, hints, node):
    """Everything the prefix has revealed by the time v's node token is due:
    the hints, plus the round-0 deductions emitted strictly before v."""
    avail = dict(hints)
    for u, c in trace["rounds"][0]:
        if int(u) == node:
            break
        avail[int(u)] = int(c)
    return avail


def _forced_colors(adj, avail, node, n_colors):
    """Colours of node's known neighbours, as a {colour: count}."""
    counts = {}
    for u in adj[node]:
        u = int(u)
        if u in avail:
            counts[avail[u]] = counts.get(avail[u], 0) + 1
    return counts


def _is_forced(adj, avail, node, n_colors):
    return node not in avail and len(_forced_colors(adj, avail, node, n_colors)) == n_colors - 1


def find_unforceable_deductions(adj, colors, hints, trace, vocab, rng, max_per_instance=4):
    """Round-0 nodes whose forcing can be BROKEN by one hint recolour, with a
    placebo recolour attached to each."""
    if not trace["rounds"]:
        return []
    q = vocab.n_colors
    hint_nodes = sorted(hints)
    hint_rank = {v: i for i, v in enumerate(hint_nodes)}
    out = []
    for rank, (node, c) in enumerate(trace["rounds"][0]):
        node, c = int(node), int(c)
        nbrs = [int(u) for u in adj[node]]
        # Forcing must be computed from EVERYTHING THE PREFIX REVEALS, not just
        # the hints. DD determines round 0 from hints alone, but the model reads
        # a prefix that also shows the round-0 deductions emitted before v, and a
        # neighbour among those can hold the colour we are about to free -- in
        # which case breaking the hint leaves v forced to the reader. 18 of 350
        # cases hit exactly that on the first run of this job.
        avail = _revealed_before(trace, hints, node)
        known = {u: avail[u] for u in nbrs if u in avail}
        covered = sorted(set(known.values()))
        if len(covered) != q - 1 or c in covered:
            continue
        counts = {col: sum(1 for x in known.values() if x == col) for col in covered}
        singles = [col for col, k in counts.items() if k == 1]
        # the unique holder has to be a HINT, so there is a hint colour token to
        # edit; a unique holder that is itself a deduction is a different test
        a = next((col for col in singles
                  if next(u for u, cl in known.items() if cl == col) in hints), None)
        if a is None:
            continue
        w = next(u for u, col in known.items() if col == a)
        # b: another covered colour, so recolouring w to b leaves a uncovered
        # and v with two candidates. It must keep the REVEALED colouring proper,
        # i.e. no neighbour of w already shows b.
        w_shown = {avail[u] for u in (int(x) for x in adj[w]) if u in avail}
        cands = [col for col in covered if col != a and col not in w_shown]
        if not cands:
            continue
        b = int(cands[0])
        # belt and braces: the edit must actually un-force v
        broken = dict(avail)
        broken[w] = b
        if _is_forced(adj, broken, node, q):
            continue
        # placebo: a hint that is NOT a neighbour of v and not v itself, whose
        # colour can be changed while keeping the revealed colouring proper.
        nbr_set = set(nbrs) | {node}
        placebo = None
        for p in rng.permutation(hint_nodes):
            p = int(p)
            if p in nbr_set:
                continue
            p_shown = {avail[u] for u in (int(x) for x in adj[p]) if u in avail}
            alt = [col for col in range(q) if col != hints[p] and col not in p_shown]
            if alt:
                placebo = (p, int(alt[0]))
                break
        if placebo is None:
            continue
        out.append({
            "node": node, "round0_rank": rank, "true_color": c,
            "freed_color": a, "recolored_nbr": w, "new_color": b,
            "placebo_nbr": placebo[0], "placebo_color": placebo[1],
            "n_known": len(known),
            "hint_color_pos": 1 + 2 * hint_rank[w] + 1,
            "placebo_color_pos": 1 + 2 * hint_rank[placebo[0]] + 1,
        })
        if len(out) >= max_per_instance:
            break
    return out


def node_token_position(tokens, vocab, node):
    """Index of the token BEFORE node's round-0 node token -- the position whose
    next-token prediction is the node identity. One left of
    sudoku_circuits.reasoning_color_position."""
    start = int(np.flatnonzero(tokens == vocab.BEGIN_REASONING)[0]) + 1
    assert tokens[start] == vocab.BEGIN_ROUND
    i = start + 1
    while i + 1 < len(tokens) and vocab.is_node_token(tokens[i]):
        if int(vocab.node_index(tokens[i])) == node:
            return i - 1
        i += 2
    raise ValueError(f"node {node} not in round 0")


def build_triple(adj, trace, vocab, case):
    """(clean, broken, placebo, predict_pos). Each edit is one token."""
    clean = encode_instance(adj, trace, vocab)
    out = [clean]
    for pos_key, col_key, old in (("hint_color_pos", "new_color", case["freed_color"]),
                                  ("placebo_color_pos", "placebo_color", None)):
        pos = case[pos_key]
        assert vocab.is_color_token(clean[pos]), "expected a hint colour token"
        if old is not None:
            assert int(vocab.color_index(clean[pos])) == old
        t = clean.copy()
        t[pos] = vocab.color_token(case[col_key])
        assert (clean != t).sum() == 1, "counterfactual must be a one-token change"
        out.append(t)
    return out[0], out[1], out[2], node_token_position(clean, vocab, case["node"])


def _best_other(row, tok):
    """Largest logit other than tok's, so the margin is signed: positive means
    tok is the argmax."""
    r = row.clone()
    r[tok] = float("-inf")
    return float(r.max())


def run_skip_behavior(model, vocab, device, cases):
    rows = []
    for adj, colors, hints, trace, case in cases:
        clean, broken, placebo, pos = build_triple(adj, trace, vocab, case)
        with torch.no_grad():
            lc, lb, lp = (model(torch.from_numpy(t).unsqueeze(0).to(device))
                          for t in (clean, broken, placebo))
        v_tok = vocab.node_token(case["node"])
        pred_c, pred_b, pred_p = (int(l[0, pos].argmax()) for l in (lc, lb, lp))
        # is the node it names instead actually forced, given the broken hints
        # plus everything the prefix already revealed?
        avail = _revealed_before(trace, hints, case["node"])
        avail_broken = dict(avail)
        avail_broken[case["recolored_nbr"]] = case["new_color"]
        pred_node = (int(vocab.node_index(pred_b)) if vocab.is_node_token(pred_b) else -1)
        rows.append({
            "node": case["node"], "round0_rank": case["round0_rank"],
            "binding_nbr": case["recolored_nbr"], "n_known": case["n_known"],
            "clean_emits_v": pred_c == v_tok,
            "corrupt_emits_v": pred_b == v_tok,
            "placebo_emits_v": pred_p == v_tok,
            "v_still_forced_broken": _is_forced(adj, avail_broken, case["node"], vocab.n_colors),
            "pred_node": pred_node,
            "pred_is_node_token": pred_node >= 0,
            # when it is NOT a node token this is usually BEGIN_ROUND, i.e. "the
            # round is over" -- which is DD's own answer when nothing else is
            # determined, so the name has to be recorded, not just the flag
            "pred_token": vocab.name(pred_b),
            "pred_is_forced": (pred_node >= 0
                               and _is_forced(adj, avail_broken, pred_node, vocab.n_colors)),
            "pred_is_unknown": pred_node >= 0 and pred_node not in avail_broken,
            # how hard the clean model wanted v (v's logit over the best rival),
            # and how much each edit moved it
            "clean_v_margin": float(lc[0, pos][v_tok] - _best_other(lc[0, pos], v_tok)),
            "corrupt_v_logit_drop": float(lc[0, pos][v_tok] - lb[0, pos][v_tok]),
            "placebo_v_logit_drop": float(lc[0, pos][v_tok] - lp[0, pos][v_tok]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n-instances", type=int, default=400)
    ap.add_argument("--n-cases", type=int, default=350)
    ap.add_argument("--seed-start", type=int, default=600_000_000,
                    help="same stream as sudoku_circuits/sudoku_chain, so the fire and "
                         "withhold tests draw on the same instances")
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
    )

    rng = np.random.default_rng(0)
    cases = []
    for adj, colors, hints, trace in build_instances(sudoku_cfg, args.n_instances, args.seed_start):
        for case in find_unforceable_deductions(adj, colors, hints, trace, vocab, rng):
            cases.append((adj, colors, hints, trace, case))
    print(f"{len(cases)} breakable round-0 deductions (with a placebo available) "
          f"from {args.n_instances} instances")
    if not cases:
        raise SystemExit("no breakable cases found")
    cases = cases[:args.n_cases]
    os.makedirs(args.out_dir, exist_ok=True)

    rows = run_skip_behavior(model, vocab, args.device, cases)
    n = len(rows)
    still = sum(r["v_still_forced_broken"] for r in rows)
    assert still == 0, f"{still} cases still forced after the edit -- the construction is wrong"
    clean = sum(r["clean_emits_v"] for r in rows) / n
    broke = sum(r["corrupt_emits_v"] for r in rows) / n
    plac = sum(r["placebo_emits_v"] for r in rows) / n
    skipped = [r for r in rows if not r["corrupt_emits_v"]]
    forced_of_skipped = (sum(r["pred_is_forced"] for r in skipped) / len(skipped)) if skipped else float("nan")
    unknown_of_skipped = (sum(r["pred_is_unknown"] for r in skipped) / len(skipped)) if skipped else float("nan")
    print(f"\nWITHHOLDING over {n} determinacy-breaking counterfactuals")
    print(f"  clean_emits_v        {clean:.3f}   (DD names v here)")
    print(f"  corrupt_emits_v      {broke:.3f}   <- HEADLINE: low = it checks the guard")
    print(f"  placebo_emits_v      {plac:.3f}   (specificity control, should match clean)")
    print(f"  of the {len(skipped)} skips: names a FORCED node {forced_of_skipped:.3f}, "
          f"an unknown node {unknown_of_skipped:.3f}")
    tok_counts = {}
    for r in skipped:
        if not r["pred_is_node_token"]:
            tok_counts[r["pred_token"]] = tok_counts.get(r["pred_token"], 0) + 1
    node_skips = [r for r in skipped if r["pred_is_node_token"]]
    if node_skips:
        print(f"    restricted to the {len(node_skips)} that name a node at all: forced "
              f"{sum(r['pred_is_forced'] for r in node_skips) / len(node_skips):.3f}")
    if tok_counts:
        print(f"    non-node predictions: {tok_counts}  "
              f"(BEGIN_ROUND = 'round over', DD's own answer when nothing else is forced)")
    print(f"  mean logit drop on v   broken {np.mean([r['corrupt_v_logit_drop'] for r in rows]):+.3f}"
          f"   placebo {np.mean([r['placebo_v_logit_drop'] for r in rows]):+.3f}")

    by_rank = {}
    for r in rows:
        by_rank.setdefault(min(r["round0_rank"], 3), []).append(r)
    print("\n  by v's position in round 0 (3 = 3rd or later)")
    for k in sorted(by_rank):
        g = by_rank[k]
        print(f"    rank {k}: n={len(g):4d}  clean {sum(x['clean_emits_v'] for x in g) / len(g):.3f}"
              f"  broken {sum(x['corrupt_emits_v'] for x in g) / len(g):.3f}"
              f"  placebo {sum(x['placebo_emits_v'] for x in g) / len(g):.3f}")

    with open(os.path.join(args.out_dir, "skip_behavior.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, "skip_summary.json"), "w") as f:
        json.dump({"checkpoint": args.checkpoint, "n_cases": n,
                   "clean_emits_v": clean, "corrupt_emits_v": broke,
                   "placebo_emits_v": plac,
                   "n_skipped": len(skipped),
                   "skipped_pred_is_forced": forced_of_skipped,
                   "skipped_pred_is_unknown": unknown_of_skipped,
                   "skipped_names_a_node": (sum(r["pred_is_node_token"] for r in skipped) / len(skipped)
                                            if skipped else float("nan")),
                   "skipped_forced_given_node": (sum(r["pred_is_forced"] for r in node_skips) / len(node_skips)
                                                 if node_skips else float("nan")),
                   "skipped_non_node_tokens": tok_counts,
                   "by_round0_rank": {str(k): {"n": len(g),
                                               "clean": sum(x["clean_emits_v"] for x in g) / len(g),
                                               "broken": sum(x["corrupt_emits_v"] for x in g) / len(g),
                                               "placebo": sum(x["placebo_emits_v"] for x in g) / len(g)}
                                      for k, g in by_rank.items()},
                   "seed_start": args.seed_start}, f, indent=2)
    print(f"\nwrote {args.out_dir}/skip_behavior.csv ({n} rows)")


if __name__ == "__main__":
    main()
