"""Causal circuit analysis: does the model USE the DD forcing rule?

Probes and SAEs answer whether DD state is decodable or represented. Neither
answers whether it is used -- a predicate can be linearly readable from a
residual stream that the model's output never consults. This module
intervenes instead of reading.

The counterfactual comes straight out of DD's forcing rule. A node u is
forced exactly when its known neighbours cover n_colors - 1 distinct colors,
leaving one color c free; DD then assigns c. Pick the neighbour w that is the
*unique* holder of some other color a among u's known neighbours, and change
w's stated color from a to c. Now u's neighbours cover the same number of
distinct colors but a is the free one, so DD's answer for u flips c -> a.

That is a ONE TOKEN change to the hint section that provably flips the
correct answer for u, with the token for u itself, the graph table and the
whole reasoning prefix left untouched. A model implementing the forcing rule
must flip with it; a model doing anything insensitive to neighbour colors
(copying, position heuristics, memorized statistics) cannot.

The corrupted instance is deliberately not required to be a globally proper
coloring -- recoloring w may clash with w's own neighbours. That is fine and
is the point: the question is whether the model applies the LOCAL forcing
rule at u, and only u's prediction is ever scored. Requiring global validity
would mean changing several hints at once and would destroy the one-token
isolation that makes the result interpretable.

Restricted to deductions in the FIRST reasoning round, where every known
neighbour of u is a hint. In later rounds some neighbours are known from
earlier deductions, whose colors live in the reasoning section, so a single
hint edit would no longer control u's evidence.

Three measurements, in increasing strength:

  behavior  -- does the predicted color for u flip c -> a? Establishes the
               model is sensitive to the forcing rule at all. If flip_rate is
               at chance there is no circuit to localize and the rest is void.
  patching  -- run the corrupted input but splice the clean residual stream
               back in at one (layer, position). The (layer, position) cells
               that restore the clean answer are where the deduction is
               carried. This localizes the computation.
  ablation  -- zero one attention head's contribution and measure the loss of
               the flip. Names the components, not just the sites.

Scoring uses logit_diff = logit[a] - logit[c] at the position that predicts
u's color, the standard patching metric: it is signed, continuous, and keeps
partial effects visible where an argmax flip would round them away.
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


def find_flippable_deductions(adj, colors, hints, trace, vocab, max_per_instance=4):
    """Nodes deduced in round 0 for which a single hint recolor flips DD's
    answer. Returns dicts with the node, its true color c, the flipped color
    a, and the neighbour w to recolor."""
    if not trace["rounds"]:
        return []
    hint_nodes = sorted(hints)
    hint_rank = {v: i for i, v in enumerate(hint_nodes)}
    out = []
    for node, c in trace["rounds"][0]:
        node, c = int(node), int(c)
        nbrs = [int(u) for u in adj[node]]
        # Round 0 => every known neighbour is a hint.
        known = {u: hints[u] for u in nbrs if u in hints}
        covered = sorted({col for col in known.values()})
        if len(covered) != vocab.n_colors - 1 or c in covered:
            continue  # not forced, or inconsistent with the trace
        # a must be held by exactly one neighbour, else recoloring that one
        # neighbour does not free a.
        counts = {col: sum(1 for v in known.values() if v == col) for col in covered}
        singles = [col for col, k in counts.items() if k == 1]
        if not singles:
            continue
        a = singles[0]
        w = next(u for u, col in known.items() if col == a)
        out.append({
            "node": node, "true_color": c, "flipped_color": a, "recolored_nbr": w,
            # position of w's COLOR token: BEGIN_HINTS, then (node, color) pairs
            "hint_color_pos": 1 + 2 * hint_rank[w] + 1,
        })
        if len(out) >= max_per_instance:
            break
    return out


def reasoning_color_position(tokens, vocab, node):
    """Index of the token that PRECEDES node's color in round 0, i.e. the
    position whose next-token prediction is that color. That is the node
    token itself."""
    start = int(np.flatnonzero(tokens == vocab.BEGIN_REASONING)[0]) + 1
    assert tokens[start] == vocab.BEGIN_ROUND
    i = start + 1
    while i + 1 < len(tokens) and vocab.is_node_token(tokens[i]):
        if int(vocab.node_index(tokens[i])) == node:
            return i
        i += 2  # (node, color) pair
    raise ValueError(f"node {node} not found in round 0")


def build_pair(adj, colors, hints, trace, vocab, case):
    """(clean_tokens, corrupt_tokens, predict_pos). The two differ in exactly
    one token."""
    clean = encode_instance(adj, trace, vocab)
    corrupt = clean.copy()
    pos = case["hint_color_pos"]
    assert vocab.is_color_token(clean[pos]), "hint_color_pos must point at a color token"
    assert vocab.color_index(clean[pos]) == case["flipped_color"]
    corrupt[pos] = vocab.color_token(case["true_color"])
    assert (clean != corrupt).sum() == 1, "counterfactual must be a one-token change"
    return clean, corrupt, reasoning_color_position(clean, vocab, case["node"])


def _logit_diff(logits, pos, vocab, case):
    """logit[flipped] - logit[true] at the predicting position. Rises when the
    model moves toward the counterfactual answer."""
    row = logits[0, pos]
    return float(row[vocab.color_token(case["flipped_color"])] - row[vocab.color_token(case["true_color"])])


def run_behavior(model, vocab, device, cases):
    """Does one neighbour-color token flip the predicted color?"""
    rows = []
    for adj, colors, hints, trace, case in cases:
        clean, corrupt, pos = build_pair(adj, colors, hints, trace, vocab, case)
        with torch.no_grad():
            lc = model(torch.from_numpy(clean).unsqueeze(0).to(device))
            lk = model(torch.from_numpy(corrupt).unsqueeze(0).to(device))
        pred_clean = int(lc[0, pos].argmax())
        pred_corrupt = int(lk[0, pos].argmax())
        rows.append({
            "node": case["node"], "true_color": case["true_color"], "flipped_color": case["flipped_color"],
            "clean_correct": pred_clean == vocab.color_token(case["true_color"]),
            "corrupt_flipped": pred_corrupt == vocab.color_token(case["flipped_color"]),
            "clean_logit_diff": _logit_diff(lc, pos, vocab, case),
            "corrupt_logit_diff": _logit_diff(lk, pos, vocab, case),
        })
    return rows


def _capture_clean(model, tokens, device):
    """Residual stream after every layer for the clean run."""
    acts = {}
    handles = [
        model.encoder.layers[i].register_forward_hook(
            lambda _m, _i, o, i=i: acts.__setitem__(i, o.detach().clone())
        )
        for i in range(len(model.encoder.layers))
    ]
    try:
        with torch.no_grad():
            model(torch.from_numpy(tokens).unsqueeze(0).to(device))
    finally:
        for h in handles:
            h.remove()
    return acts


def run_patching(model, vocab, device, cases, positions_of_interest):
    """Splice clean activations into the corrupted run at one (layer,
    position) and measure how much of the clean answer returns.

    recovery = 1.0 means that single cell fully restores the clean behavior,
    i.e. the deduction is carried there; 0.0 means it is irrelevant."""
    n_layers = len(model.encoder.layers)
    rows = []
    for adj, colors, hints, trace, case in cases:
        clean, corrupt, pos = build_pair(adj, colors, hints, trace, vocab, case)
        clean_acts = _capture_clean(model, clean, device)
        corrupt_t = torch.from_numpy(corrupt).unsqueeze(0).to(device)
        with torch.no_grad():
            base = _logit_diff(model(corrupt_t), pos, vocab, case)
            ref = _logit_diff(model(torch.from_numpy(clean).unsqueeze(0).to(device)), pos, vocab, case)
        if abs(base - ref) < 1e-6:
            continue  # corruption had no effect; nothing to localize
        sites = positions_of_interest(clean, vocab, case, pos)
        for layer in range(n_layers):
            for site_name, site_pos in sites.items():
                def hook(_m, _i, o, layer=layer, site_pos=site_pos):
                    o = o.clone()
                    o[:, site_pos] = clean_acts[layer][:, site_pos]
                    return o
                h = model.encoder.layers[layer].register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        patched = _logit_diff(model(corrupt_t), pos, vocab, case)
                finally:
                    h.remove()
                rows.append({
                    "node": case["node"], "layer": layer, "site": site_name,
                    "patched_logit_diff": patched,
                    "recovery": (patched - base) / (ref - base),
                })
    return rows


def run_head_ablation(model, vocab, device, cases):
    """Zero one attention head's contribution (its slice of out_proj's input,
    which is the only place heads are still separable) and measure the loss of
    the counterfactual effect.

    effect_retained = 1.0 means the head is irrelevant to the flip; 0.0 means
    ablating it destroys the flip entirely."""
    n_layers = len(model.encoder.layers)
    nhead = model.encoder.layers[0].self_attn.num_heads
    d_model = model.encoder.layers[0].self_attn.embed_dim
    d_head = d_model // nhead
    rows = []
    for adj, colors, hints, trace, case in cases:
        clean, corrupt, pos = build_pair(adj, colors, hints, trace, vocab, case)
        clean_t = torch.from_numpy(clean).unsqueeze(0).to(device)
        corrupt_t = torch.from_numpy(corrupt).unsqueeze(0).to(device)
        with torch.no_grad():
            base_effect = (_logit_diff(model(corrupt_t), pos, vocab, case)
                           - _logit_diff(model(clean_t), pos, vocab, case))
        if abs(base_effect) < 1e-6:
            continue
        for layer in range(n_layers):
            for head in range(nhead):
                def pre_hook(_m, inputs, head=head):
                    x = inputs[0].clone()
                    x[..., head * d_head:(head + 1) * d_head] = 0.0
                    return (x,)
                h = model.encoder.layers[layer].self_attn.out_proj.register_forward_pre_hook(pre_hook)
                try:
                    with torch.no_grad():
                        eff = (_logit_diff(model(corrupt_t), pos, vocab, case)
                               - _logit_diff(model(clean_t), pos, vocab, case))
                finally:
                    h.remove()
                rows.append({
                    "node": case["node"], "layer": layer, "head": head,
                    "ablated_effect": eff, "effect_retained": eff / base_effect,
                })
    return rows


def default_sites(tokens, vocab, case, predict_pos):
    """The four places the flipped color could be read from or written to."""
    graph_start = int(np.flatnonzero(tokens == vocab.BEGIN_GRAPH)[0]) + 1
    return {
        # the edited token itself
        "edited_hint": case["hint_color_pos"],
        # u's own row in the graph table, where its neighbour list is stated
        "graph_row_u": graph_start + case["node"] * (1 + vocab.degree),
        # the recolored neighbour's row
        "graph_row_w": graph_start + case["recolored_nbr"] * (1 + vocab.degree),
        # the position whose next-token prediction is u's color
        "predict_pos": predict_pos,
    }


def collect_cases(sudoku_cfg, vocab, n_instances, seed_start, max_per_instance=2):
    cases = []
    for adj, colors, hints, trace in build_instances(sudoku_cfg, n_instances, seed_start):
        for case in find_flippable_deductions(adj, colors, hints, trace, vocab, max_per_instance):
            cases.append((adj, colors, hints, trace, case))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n-instances", type=int, default=400)
    ap.add_argument("--n-cases", type=int, default=120, help="cap on counterfactuals actually run")
    ap.add_argument("--seed-start", type=int, default=600_000_000,
                    help="disjoint from training, probes (5e8/9e8), series (7e8) and transition (8e8)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    model, vocab, config = load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    sudoku_cfg = SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        rho=config["rho"], rho_max=config["rho_max"],
        scratchpad=config.get("scratchpad", False), hint_mode=config.get("hint_mode", "removal"),
    )
    cases = collect_cases(sudoku_cfg, vocab, args.n_instances, args.seed_start)
    print(f"{len(cases)} flippable round-0 deductions from {args.n_instances} instances")
    cases = cases[:args.n_cases]
    os.makedirs(args.out_dir, exist_ok=True)

    beh = run_behavior(model, vocab, args.device, cases)
    n = len(beh)
    clean_acc = sum(r["clean_correct"] for r in beh) / n
    flip = sum(r["corrupt_flipped"] for r in beh) / n
    both = sum(r["clean_correct"] and r["corrupt_flipped"] for r in beh) / n
    print(f"\nBEHAVIOR over {n} counterfactuals")
    print(f"  clean_correct         {clean_acc:.3f}")
    print(f"  corrupt_flipped       {flip:.3f}   (chance ~{1 / vocab.n_colors:.3f})")
    print(f"  both (uses the rule)  {both:.3f}")
    print(f"  mean logit_diff  clean {np.mean([r['clean_logit_diff'] for r in beh]):+.3f}"
          f"  corrupt {np.mean([r['corrupt_logit_diff'] for r in beh]):+.3f}")

    _write(os.path.join(args.out_dir, "behavior.csv"), beh)

    # Gate on the EFFECT SIZE, not the flip rate. Patching and ablation are
    # scored in logit_diff and normalized by (clean - corrupt), so they are
    # well-conditioned whenever that gap is large -- even if it never crosses
    # the argmax boundary. An earlier version gated on the flip rate and so
    # skipped localization on a run where the corruption moved the logit gap
    # by 12.3 (-16.70 -> -4.44) at a flip rate of 0.267: a large, perfectly
    # localizable causal effect that happened to stay sub-threshold. A small
    # gap is the real degenerate case, because then the denominator is noise.
    effect = abs(np.mean([r["corrupt_logit_diff"] - r["clean_logit_diff"] for r in beh]))
    print(f"  mean |effect| on the logit gap  {effect:.3f}")
    if effect < 0.5:
        print("\nthe corruption barely moves the logit gap, so patching would divide by noise. "
              "Stopping here.")
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump({"clean_correct": clean_acc, "corrupt_flipped": flip, "both": both,
                       "logit_effect": float(effect), "localized": False}, f, indent=2)
        return
    if both < 2.0 / vocab.n_colors:
        print("\nNOTE: the flip rate is near chance, so the model does not APPLY the forcing rule "
              "at decision level. The logit effect above is still large, so what gets localized "
              "below is where that sub-threshold sensitivity to neighbour colors lives -- not a "
              "circuit that determines the answer.")

    patch = run_patching(model, vocab, args.device, cases, default_sites)
    _write(os.path.join(args.out_dir, "patching.csv"), patch)
    print("\nPATCHING: mean recovery by (layer, site)")
    sites = list(dict.fromkeys(r["site"] for r in patch))
    print("  layer " + "".join(f"{s:>14}" for s in sites))
    for layer in sorted({r["layer"] for r in patch}):
        cells = [np.mean([r["recovery"] for r in patch if r["layer"] == layer and r["site"] == s] or [np.nan])
                 for s in sites]
        print(f"  {layer:5d} " + "".join(f"{c:14.3f}" for c in cells))

    abl = run_head_ablation(model, vocab, args.device, cases)
    _write(os.path.join(args.out_dir, "head_ablation.csv"), abl)
    print("\nABLATION: 10 heads whose removal most destroys the flip")
    by_head = {}
    for r in abl:
        by_head.setdefault((r["layer"], r["head"]), []).append(r["effect_retained"])
    ranked = sorted(((np.mean(v), k) for k, v in by_head.items()))[:10]
    for retained, (layer, head) in ranked:
        print(f"  L{layer:02d}H{head}  effect_retained={retained:+.3f}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"clean_correct": clean_acc, "corrupt_flipped": flip, "both": both,
                   "logit_effect": float(effect), "localized": True, "n_cases": n,
                   "top_heads": [{"layer": l, "head": h, "effect_retained": float(r)}
                                 for r, (l, h) in ranked]}, f, indent=2)


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
