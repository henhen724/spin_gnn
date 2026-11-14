"""Is the untied model's determinacy GUARD specific to facts it was GIVEN?

WHAT FINDING 70 LEFT ON THE TABLE, AND WHY IT IS NOT YET A FINDING. Finding 68
measured the guard -- does the model stop naming v once v stops being determined
-- on 350 round-0 cases, and got S1 1.0000 at every seed (never withholds)
against J2 0.0000 / 0.0000 / 0.8444 (withholds, except seed 2). Finding 70's
read-back then noticed, POST HOC, that J2's guard rate on the chained cases is
0.9903 / 0.8619 / 0.9830: on those cases the untied model keeps naming v almost
always, like the tied one. No hypothesis asked that, and the comparison is
cell-level -- two different case lists -- so it cannot be believed as it stands.

THREE THINGS DIFFER BETWEEN THOSE TWO CASE LISTS AT ONCE.
  (a) PROVENANCE. The round-0 edit recolours a HINT; the chained edit recolours a
      fact the model DEDUCED earlier in its own output.
  (b) DEPTH. Round-0 cases put v in the first deduction round, a few tokens into
      the reasoning section; chained cases put it in rounds 1-15.
  (c) CLASH. By the clash theorem (finding 70), an edit to a derived fact always
      conflicts with at least one visible neighbour, while finding 68 REQUIRED
      its hint edit to keep the revealed colouring proper -- zero clashes. So the
      chained prefix is visibly corrupted and the round-0 one is not.
Any of the three could produce the post-hoc gap, and (c) is not even about the
model's reasoning.

THE TEST ISOLATES (a) BY PAIRING INSIDE ONE CASE. Take a node v the trace
deduces, whose visible neighbours cover exactly q-1 = 3 colours, and require TWO
of those colours to have exactly one visible holder each: one holder a HINT
(w_g, holding a_g) and one holder a DEDUCTION the model emitted earlier (w_d,
holding a_d). Both are single holders, so breaking either one frees exactly one
colour and leaves v with two candidates -- v stops being determined in precisely
the same way. Then write six one-token variants of the one sequence:

  clean           nothing changed              v forced to c
  placebo         a far node recoloured        v forced to c; nothing about v moves
  rule_given      hint w_g:     a_g -> c       v forced to a_g
  rule_derived    deduced w_d:  a_d -> c       v forced to a_d
  broken_given    hint w_g:     a_g -> b_g     v NOT forced; candidates {c, a_g}
  broken_derived  deduced w_d:  a_d -> b_d     v NOT forced; candidates {c, a_d}

and read the NODE position (v's node-token index - 1), whose next-token
prediction is v's identity: does the model still name v? Same v, same round, same
prefix length, same readout, same forward-pass count. (b) is held fixed by
construction, and the two broken edits differ only in where the fact they break
came from.

(c) IS HELD FIXED BY MATCHING, NOT BY WISHING. The derived edit must clash -- the
theorem leaves no choice -- so its clash count is computed first and the GIVEN
edit's new colour b_g is chosen to clash the same number of times, out of the
colours available to it. `clash_matched` records whether that succeeded and the
scorer's G6 repeats the headline on the matched subset alone. The placebo is
matched to the same count, as in finding 70, so it is a null for "the model is
reacting to a corrupted prefix" and not only for relevance.

WHY BOTH RULE EDITS ARE HERE AND COST TWO MORE FORWARD PASSES. If the model keeps
naming v after the derived edit, the dull explanation is that the derived token
never reached it. The rule edit at the SAME token is the control that rules that
out: if recolouring w_d to c flips the emitted colour to a_d, the model read the
token. Findings 69 and 70 each measured one of these two rates on its own case
list; measuring both here, on one list, is what makes the guard comparison
interpretable rather than suggestive.

WHAT A NULL WOULD MEAN, STATED BEFORE THE RUN. If the two guard rates match at
fixed v, provenance is NOT what moved the post-hoc numbers, and the effect must
live in (b) or (c) -- which the depth table (G5, descriptive) is in the file to
show. That outcome is as publishable as the positive one and it relocates the
question rather than closing it.

NOTHING IS TRAINED. Six existing checkpoints, read six times each.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .sudoku_chain import reasoning_node_positions
from .sudoku_chain4 import _clashes, revealed_before
from .sudoku_dissoc import _mean, _read
from .sudoku_skip import _is_forced
from .sudoku_tokens import encode_instance
from .train_sudoku_transformer import build_instances, load_checkpoint

# The six one-token variants, in the order every table and figure prints them:
# the reference, then the null, then the two reads, then the two the finding
# turns on -- so a reader meets the controls before the headline.
VARIANTS = ("clean", "placebo", "rule_given", "rule_derived",
            "broken_given", "broken_derived")
# Which side's freed colour each variant is scored against. clean and placebo
# are read against BOTH, off the same logits, because they are the reference for
# each side and no forward pass is repeated to get it.
SIDE_OF = {"rule_given": "g", "broken_given": "g",
           "rule_derived": "d", "broken_derived": "d"}


def best_broken(adj, avail, covered, w, a):
    """(b, clashes) for the least visibly wrong way to break w's colour.

    b is another colour already covered by a different neighbour of v, so after
    the edit b is still covered, a is held by nobody, and v drops from q-1
    covered colours to q-2 -- two candidates, not determined. Ties by colour
    index, so the case list stays a deterministic function of the instance
    stream and no rng decides what gets measured.
    """
    opts = [col for col in covered if col != a]
    b = min(opts, key=lambda x: (_clashes(adj, avail, w, x), x))
    return b, _clashes(adj, avail, w, b)


def find_provenance_pairs(adj, colors, hints, trace, vocab, max_per_round=2):
    """Deductions v that can be un-determined EITHER by breaking a given fact OR
    by breaking a derived one, with both edits built and clash-matched.

    Every returned case carries all six variants' ingredients, so a case either
    supports the whole comparison or is not a case. Cases are collected in trace
    order and capped per ROUND rather than per instance, because depth is G5's
    axis and a per-instance cap fills up inside the first round.
    """
    q = vocab.n_colors
    out = []
    for k, rnd in enumerate(trace["rounds"]):
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
            holders = {col: [u for u in known if known[u] == col] for col in covered}
            singles = {col: us[0] for col, us in holders.items() if len(us) == 1}
            givens = sorted((col, u) for col, u in singles.items() if u in hints)
            derivs = sorted((col, u) for col, u in singles.items() if u in deduced)
            if not givens or not derivs:
                continue

            # The derived side is chosen first and the given side is matched to
            # it, not the other way round: the derived edit's clash count is
            # forced by the theorem, so it is the constraint, and the given edit
            # -- which has a free choice of b -- is the one that can move.
            cand_d = [(a, w) + best_broken(adj, avail, covered, w, a) for a, w in derivs]
            a_d, w_d, b_d, cl_d = min(cand_d, key=lambda t: (t[3], t[1]))
            cand_g = [(a, w, b, _clashes(adj, avail, w, b))
                      for a, w in givens for b in covered if b != a]
            a_g, w_g, b_g, cl_g = min(cand_g,
                                      key=lambda t: (abs(t[3] - cl_d), t[3], t[1], t[2]))

            # Belt and braces: each edit on its own must actually un-determine v.
            # Cheap, and it is the property the whole measurement rests on.
            ok = True
            for w, b in ((w_g, b_g), (w_d, b_d)):
                edited = dict(avail)
                edited[w] = b
                if _is_forced(adj, edited, v, q):
                    ok = False
            if not ok:
                continue

            # The placebo: a node far from v recoloured, clash-matched to the
            # derived edit. A DEDUCED far node is preferred, so the null is the
            # same kind of object as the harder edit; round-0 cases often have
            # none yet, and there a far hint keeps the case rather than losing
            # the shallow end of the depth axis.
            far = sorted(u for u in avail if u not in nbrs and u not in (v, w_g, w_d))
            cands = [(p, col, _clashes(adj, avail, p, col), 0 if p in deduced else 1)
                     for p in far for col in range(q) if col != avail[p]]
            if not cands:
                continue
            p, pcol, pl_cl, pl_hint = min(
                cands, key=lambda t: (abs(t[2] - cl_d), t[3], t[2], t[0], t[1]))

            # Re-derived from the trace rather than from the `w in hints` test
            # that selected it, so the scorer's guard 3 is an independent check
            # and not a constant this function wrote down about itself.
            emitted = {int(u): kk for kk, rr in enumerate(trace["rounds"])
                       for u, _ in rr}
            out.append({
                "node": v, "round": k, "round_rank": rank, "true_color": c,
                "given_round": emitted.get(w_g, -1),
                "n_known": len(known), "n_colors": q,
                "v_first_in_round": rank == 0,
                # the given side
                "w_given": w_g, "freed_given": a_g, "new_given": b_g,
                "broken_clashes_given": cl_g,
                "rule_clashes_given": _clashes(adj, avail, w_g, c),
                # the derived side
                "w_derived": w_d, "freed_derived": a_d, "new_derived": b_d,
                "broken_clashes_derived": cl_d,
                "rule_clashes_derived": _clashes(adj, avail, w_d, c),
                "derived_round": emitted[w_d],
                "clash_matched": int(cl_g == cl_d),
                # the null
                "placebo_node": p, "placebo_color": pcol, "placebo_clashes": pl_cl,
                "placebo_is_deduced": int(pl_hint == 0),
                "placebo_matched": int(pl_cl == cl_d),
            })
            taken += 1
    return out


def build_variants(adj, trace, hints, vocab, case):
    """{variant: tokens} plus the node position. Every edit is one token, in v's
    prefix; the two broken edits sit at two DIFFERENT tokens, which is the whole
    point -- one in the givens section, one in the reasoning section."""
    clean = encode_instance(adj, trace, vocab)
    pmap = reasoning_node_positions(clean, vocab)
    hint_rank = {u: i for i, u in enumerate(sorted(hints))}
    v_idx = pmap[case["node"]][0]
    pos = v_idx - 1

    def cpos(node):
        """Where this node's colour token is: the givens section for a hint, the
        reasoning section for a fact the model emitted."""
        if node in hint_rank:
            return 1 + 2 * hint_rank[node] + 1
        return pmap[node][0] + 1

    g_pos, d_pos, pl_pos = (cpos(case["w_given"]), cpos(case["w_derived"]),
                            cpos(case["placebo_node"]))
    assert case["w_given"] in hint_rank, "the given side must be a hint"
    assert case["w_derived"] not in hint_rank, "the derived side must be emitted"
    for p, who, want in ((g_pos, "given", case["freed_given"]),
                         (d_pos, "derived", case["freed_derived"]),
                         (pl_pos, "placebo", None)):
        assert p < v_idx, "the %s edit must be in v's prefix" % who
        assert vocab.is_color_token(clean[p]), "%s edit is not a colour token" % who
        if want is not None:
            assert int(vocab.color_index(clean[p])) == want
    assert len({g_pos, d_pos, pl_pos}) == 3, "the three edits must be three tokens"
    assert vocab.is_node_token(clean[pos + 1])
    assert int(vocab.node_index(clean[pos + 1])) == case["node"]

    out = {"clean": clean}
    for name, (p, col) in (("placebo", (pl_pos, case["placebo_color"])),
                           ("rule_given", (g_pos, case["true_color"])),
                           ("rule_derived", (d_pos, case["true_color"])),
                           ("broken_given", (g_pos, case["new_given"])),
                           ("broken_derived", (d_pos, case["new_derived"]))):
        t = clean.copy()
        t[p] = vocab.color_token(col)
        assert (clean != t).sum() == 1, "%s must be a one-token change" % name
        out[name] = t
    return out, pos


def run_prov(model, vocab, device, cases):
    rows = []
    for adj, colors, hints, trace, case in cases:
        variants, pos = build_variants(adj, trace, hints, vocab, case)
        v_tok = vocab.node_token(case["node"])
        c = case["true_color"]
        a_g, a_d = case["freed_given"], case["freed_derived"]
        with torch.no_grad():
            reads = {}
            for name in VARIANTS:
                lg = model(torch.from_numpy(variants[name]).unsqueeze(0).to(device))
                for side, a in (("g", a_g), ("d", a_d)):
                    reads[(name, side)] = _read(lg, pos, vocab, v_tok, c, a)
        row = {k: case[k] for k in (
            "node", "round", "round_rank", "true_color", "n_known",
            "v_first_in_round", "given_round", "w_given", "freed_given", "new_given",
            "broken_clashes_given", "rule_clashes_given", "w_derived",
            "freed_derived", "new_derived", "broken_clashes_derived",
            "rule_clashes_derived", "derived_round", "clash_matched",
            "placebo_node", "placebo_clashes", "placebo_is_deduced",
            "placebo_matched")}
        for name in VARIANTS:
            rg, rd = reads[(name, "g")], reads[(name, "d")]
            # emits_v, the colour argmax and p_true are properties of the one
            # forward pass, not of the side it is read against; only the freed
            # colour's mass is side-specific.
            row["%s_emits_v" % name] = rg["emits_v"]
            row["%s_color" % name] = rg["color"]
            row["%s_p_true" % name] = rg["p_true"]
            row["%s_p_freed_g" % name] = rg["p_freed"]
            row["%s_p_freed_d" % name] = rd["p_freed"]
            row["%s_p_cand_g" % name] = rg["p_cand"]
            row["%s_p_cand_d" % name] = rd["p_cand"]
        # The registered per-case events. `names_v_*` is finding 68's guard
        # quantity: HIGH means the model kept naming v after v stopped being
        # determined, i.e. no guard. The scorer divides it by clean_emits_v the
        # way finding 68 did, and prints the unconditional rate beside it.
        row["names_v_given"] = reads[("broken_given", "g")]["emits_v"]
        row["names_v_derived"] = reads[("broken_derived", "d")]["emits_v"]
        row["rule_flips_given"] = reads[("rule_given", "g")]["color"] == a_g
        row["rule_flips_derived"] = reads[("rule_derived", "d")]["color"] == a_d
        row["rule_reverts_given"] = reads[("rule_given", "g")]["color"] == c
        row["rule_reverts_derived"] = reads[("rule_derived", "d")]["color"] == c
        row["prov_split"] = (bool(row["names_v_derived"])
                             and not bool(row["names_v_given"]))
        row["prov_split_rev"] = (bool(row["names_v_given"])
                                 and not bool(row["names_v_derived"]))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n-instances", type=int, default=1200)
    ap.add_argument("--n-cases", type=int, default=400)
    ap.add_argument("--seed-start", type=int, default=600_000_000,
                    help="findings 68/69/70's stream, so these cases come from the "
                         "same instances as every earlier probe's")
    ap.add_argument("--cases-only", action="store_true",
                    help="build the case list, report the yield, its round profile "
                         "and its clash-match rate, run no forward pass and write "
                         "nothing -- the census step of the launcher, so a thin "
                         "yield costs a minute instead of six cells")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    model, vocab, config = load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    sudoku_cfg = SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        rho=config["rho"], rho_max=config["rho_max"],
        scratchpad=config.get("scratchpad", False),
        hint_mode=config.get("hint_mode", "removal"),
        shuffle_adj=bool(config.get("shuffle_adj", False)),
        shuffle_events=bool(config.get("shuffle_events", False)),
        fall=bool(config.get("fall", False)),
    )

    cases = []
    for adj, colors, hints, trace in build_instances(sudoku_cfg, args.n_instances,
                                                     args.seed_start):
        for case in find_provenance_pairs(adj, colors, hints, trace, vocab):
            cases.append((adj, colors, hints, trace, case))
    print("%d provenance-paired cases (one hint holder AND one deduced holder) "
          "from %d instances" % (len(cases), args.n_instances))
    if args.cases_only:
        take = cases[:args.n_cases]
        depth = {}
        for _, _, _, _, case in take:
            depth[case["round"]] = depth.get(case["round"], 0) + 1
        print("  by round: " + "  ".join("r%d=%d" % (k, depth[k]) for k in sorted(depth)))
        if take:
            m = sum(cs["clash_matched"] for *_, cs in take) / len(take)
            pm = sum(cs["placebo_matched"] for *_, cs in take) / len(take)
            print("  clash-matched %.4f of the first %d, placebo-matched %.4f"
                  % (m, len(take), pm))
        return
    if len(cases) < args.n_cases:
        raise SystemExit("only %d cases, need %d -- raise --n-instances"
                         % (len(cases), args.n_cases))
    cases = cases[:args.n_cases]
    os.makedirs(args.out_dir, exist_ok=True)

    rows = run_prov(model, vocab, args.device, cases)
    n = len(rows)

    print("\nPROVENANCE-PAIRED GUARD -- %d cases" % n)
    print("  variant          emits_v   p_true   p_freed(g)  p_freed(d)")
    for name in VARIANTS:
        ev, _ = _mean(rows, "%s_emits_v" % name)
        pt, _ = _mean(rows, "%s_p_true" % name)
        pg, _ = _mean(rows, "%s_p_freed_g" % name)
        pd, _ = _mean(rows, "%s_p_freed_d" % name)
        print("  %-15s  %7.4f  %7.4f  %10.4f  %10.4f" % (name, ev, pt, pg, pd))

    clean = [r for r in rows if r["clean_emits_v"]]
    nd = len(clean)
    gg = float(np.mean([r["names_v_given"] for r in clean])) if clean else float("nan")
    gd = float(np.mean([r["names_v_derived"] for r in clean])) if clean else float("nan")
    print("\n  G1 still names v | clean named v (n=%d):  given %.4f   derived %.4f"
          % (nd, gg, gd))
    b = sum(1 for r in clean if r["prov_split"])
    c_ = sum(1 for r in clean if r["prov_split_rev"])
    print("     McNemar b=%d (derived only) c=%d (given only)" % (b, c_))
    print("  G2 rule flips:  given %.4f   derived %.4f   (bar 0.60, chance 0.25)"
          % (_mean(rows, "rule_flips_given")[0], _mean(rows, "rule_flips_derived")[0]))
    pl = float(np.mean([r["placebo_emits_v"] for r in clean])) if clean else float("nan")
    print("  G3 placebo still names v | clean named v:  %.4f   (floor 0.85)" % pl)

    with open(os.path.join(args.out_dir, "prov_behavior.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"checkpoint": args.checkpoint, "n_cases": n, "seed_start": args.seed_start,
               "clean_n": nd, "guard_given": gg, "guard_derived": gd,
               "mcnemar_b": b, "mcnemar_c": c_, "placebo_cond": pl,
               "rule_flips_given": _mean(rows, "rule_flips_given")[0],
               "rule_flips_derived": _mean(rows, "rule_flips_derived")[0],
               "clash_matched": _mean(rows, "clash_matched")[0]}
    for name in VARIANTS:
        for key in ("emits_v", "p_true", "p_freed_g", "p_freed_d"):
            summary["%s_%s" % (name, key)] = _mean(rows, "%s_%s" % (name, key))[0]
    with open(os.path.join(args.out_dir, "prov_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote %s/prov_behavior.csv (%d rows)" % (args.out_dir, n))


if __name__ == "__main__":
    main()
