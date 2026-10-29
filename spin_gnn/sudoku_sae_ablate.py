"""Do the `justifies_next` features CAUSE the colour decision, or merely sit
beside it?

Findings 19/21 established that the leak-free model REPRESENTS, at `pre_color`
positions, the predicate `justifies_next[u]` = "u is a known neighbour of the
node whose colour is being emitted" -- 69 features at L8, best F1 0.743,
against the leaked twin's 0.247. That is a decoding result. A linear probe
finding a predicate does not show the model computes with it; finding 3 (the
frontier probe measured `d_model`, not competence) and the `next_node` baseline
(a no-deduction hand rule beat the model's own features) are two cases in this
project where exactly that inference failed.

This module intervenes. At the `pre_color` position of a node that DD forces,
the residual stream at layer L is replaced by its SAE reconstruction with a
chosen feature set zeroed, and the forward pass continues. Everything is scored
against the `recon` condition rather than `clean`, so SAE reconstruction error
is netted out of every number.

THE CONDITIONS, and what each one rules out:

  clean            no intervention. Establishes the case set is solvable.
  recon            full SAE reconstruction, nothing zeroed. The baseline every
                   effect below is measured against; its gap to `clean` is the
                   reconstruction error and is reported, not hidden.
  zero_jn_all      every interpretable `justifies_next` feature. The blunt
                   version of the question.
  zero_jn_bind     ONLY the features labelled `justifies_next[w]`, where w is
                   the one neighbour whose colour is doing the forcing -- the
                   neighbour that, recoloured, flips DD's answer. THE TEST.
  zero_jn_nonbind  the same number of features, but for a known neighbour u of
                   the same node whose colour is NOT the binding constraint.
                   The selectivity control: if the model is computing the
                   elimination, bind must hurt and nonbind must not.
  zero_random      count-matched random live features from outside the family.
                   The magnitude control: removing any 60-odd features from a
                   reconstruction degrades it somewhat, and that floor has to
                   be measured rather than assumed away.

The pair that carries the claim is zero_jn_bind vs zero_jn_nonbind. Both remove
the same number of features, from the same family, at the same position, in the
same layer, on the same instances; they differ only in WHICH neighbour the
feature refers to. No reconstruction-error or feature-count story explains a
gap between them -- only the model actually using that neighbour's colour to
eliminate a candidate does.

Case source is `sudoku_circuits.find_flippable_deductions`, so "forced" and
"binding neighbour" mean exactly what they mean in the counterfactual job, and
the two runs are about the same cases.
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .sudoku_circuits import collect_cases, reasoning_color_position
from .sudoku_sae import SparseAutoencoder
from .sudoku_tokens import encode_instance
from .train_sudoku_transformer import load_checkpoint

JN = "justifies_next"
_JN_NODE = re.compile(r"justifies_next\[(\d+)\]")


def load_sae(sae_dir, layer, d_model, device="cpu"):
    """The dictionary plus the per-dimension standardization it was fit under.

    train_sae standardizes its inputs, so the dictionary lives in
    (x - mu) / sigma space; splicing a reconstruction back into the residual
    stream without inverting that is meaningless. norm_layer*.pt was added to
    sudoku_sae.py on 10-28 for exactly this -- dictionaries written before then
    cannot be used here, and saying so loudly beats silently reconstructing
    garbage."""
    sd = torch.load(os.path.join(sae_dir, f"sae_layer{layer:02d}.pt"), map_location=device)
    sae = SparseAutoencoder(d_model, int(sd["b_enc"].shape[0]))
    sae.load_state_dict(sd)
    sae.to(device).eval()

    norm_path = os.path.join(sae_dir, f"norm_layer{layer:02d}.pt")
    if not os.path.exists(norm_path):
        raise SystemExit(
            f"{norm_path} missing. That SAE run predates the norm-stat fix, so its "
            f"dictionary cannot be inverted into residual-stream coordinates. "
            f"Re-run spin_gnn.sudoku_sae for this checkpoint."
        )
    nrm = torch.load(norm_path, map_location=device)
    return sae, nrm["mu"].reshape(1, -1).to(device), nrm["sigma"].reshape(1, -1).to(device)


def feature_sets(sae_dir, layer, kind="pre_color", select="interpretable"):
    """(all justifies_next features, {node: [features]}, live features outside
    the family).

    Prefers the per-kind scoring, because that is where finding 21's counts
    come from: at L8 the pre_color file lists 69 `justifies_next` features and
    the pooled file only 7, since pooling dilutes a predicate that is only
    defined while a colour is being emitted. Selecting from the pooled file
    would ablate a tenth of the set the finding is about.

    select="beats_null" widens the set to every feature above its own
    position-rolled null rather than only those also clearing min_f1, which
    buys per-node coverage -- with one feature per node the selective
    conditions have almost no cases where both sides are non-empty."""
    path = os.path.join(sae_dir, f"features_layer{layer:02d}_{kind}.csv")
    if not os.path.exists(path):
        path = os.path.join(sae_dir, f"features_layer{layer:02d}.csv")
        print(f"  NOTE: no per-kind file, falling back to POOLED labels ({path}); "
              f"expect far fewer {JN} features than findings 19/21 report")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    def keep(r):
        if select == "beats_null":
            return r["beats_null"] == "True" and r["is_alive"] == "True"
        return r["is_interpretable"] == "True" and r["beats_null"] == "True"

    jn_all, by_node, other_alive = [], defaultdict(list), []
    for r in rows:
        idx = int(r["feature"])
        if r["best_predicate_family"] == JN and keep(r):
            jn_all.append(idx)
            m = _JN_NODE.match(r["best_predicate"])
            if m:
                by_node[int(m.group(1))].append(idx)
        elif r["is_alive"] == "True":
            other_alive.append(idx)
    return sorted(jn_all), dict(by_node), sorted(other_alive)


def _ablation_hook(sae, mu, sigma, pos, kill):
    """Replace the residual at `pos` with its reconstruction, minus `kill`.

    kill = [] is the reconstruction baseline. Batch-first (B, T, d_model),
    matching sudoku_circuits.run_patching's o[:, site_pos] convention."""
    kill = torch.as_tensor(kill, dtype=torch.long) if len(kill) else None

    def hook(_m, _i, o):
        o = o.clone()
        z = (o[:, pos] - mu) / sigma
        f = sae.encode(z)
        if kill is not None:
            f[:, kill.to(f.device)] = 0.0
        z_hat = f @ sae.W_dec.T + sae.b_dec
        o[:, pos] = z_hat * sigma + mu
        return o

    return hook


def _score(logits, pos, vocab, true_color):
    """(predicted the true colour, margin over the best competing colour)."""
    row = logits[0, pos]
    ctoks = [vocab.color_token(c) for c in range(vocab.n_colors)]
    true_tok = vocab.color_token(true_color)
    others = [t for t in ctoks if t != true_tok]
    return bool(int(row.argmax()) == true_tok), float(row[true_tok] - max(row[t] for t in others))


def run_case(model, sae, mu, sigma, vocab, device, layer, adj, trace, case,
             jn_all, by_node, other_alive, hints, rng):
    """One forced node, every condition, at that node's pre_color position."""
    tokens = encode_instance(adj, trace, vocab)
    pos = reasoning_color_position(tokens, vocab, case["node"])
    tok_t = torch.from_numpy(tokens).unsqueeze(0).to(device)
    node, w, true_color = case["node"], case["recolored_nbr"], case["true_color"]

    bind = by_node.get(w, [])
    # A known neighbour of the same node that is NOT the binding constraint,
    # preferring one with a comparable number of features so the two selective
    # conditions remove a similar amount.
    known_others = [u for u in (int(x) for x in adj[node])
                    if u in hints and u != w and by_node.get(u)]
    nonbind = []
    if known_others and bind:
        best = min(known_others, key=lambda u: abs(len(by_node[u]) - len(bind)))
        nonbind = by_node[best][:len(bind)] if len(by_node[best]) > len(bind) else by_node[best]

    pool = [i for i in other_alive if i not in set(jn_all)]
    rand = list(rng.choice(pool, size=min(len(jn_all), len(pool)), replace=False)) if pool else []

    conds = {
        "clean": None,
        "recon": [],
        "zero_jn_all": jn_all,
        "zero_jn_bind": bind,
        "zero_jn_nonbind": nonbind,
        "zero_random": rand,
    }

    out = {"node": node, "binding_nbr": w, "true_color": true_color,
           "n_bind": len(bind), "n_nonbind": len(nonbind),
           "n_jn_all": len(jn_all), "n_random": len(rand)}
    for name, kill in conds.items():
        if kill is None:
            with torch.no_grad():
                logits = model(tok_t)
        else:
            h = model.encoder.layers[layer].register_forward_hook(
                _ablation_hook(sae, mu, sigma, pos, kill))
            try:
                with torch.no_grad():
                    logits = model(tok_t)
            finally:
                h.remove()
        ok, margin = _score(logits, pos, vocab, true_color)
        out[f"{name}_correct"] = ok
        out[f"{name}_margin"] = margin
    return out


def mcnemar_exact(rows, a, b):
    """Paired exact test on `correct` between two conditions. Returns
    (n_a_only, n_b_only, two-sided p)."""
    from math import comb
    x = sum(1 for r in rows if r[f"{a}_correct"] and not r[f"{b}_correct"])
    y = sum(1 for r in rows if r[f"{b}_correct"] and not r[f"{a}_correct"])
    n = x + y
    if n == 0:
        return x, y, 1.0
    k = min(x, y)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return x, y, min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--sae-dir", type=str, required=True,
                    help="a spin_gnn.sudoku_sae out-dir, with sae_layerNN.pt and norm_layerNN.pt")
    ap.add_argument("--layer", type=int, nargs="+", default=[8],
                    help="residual-stream layer(s) to ablate at; 8 is where findings 19/21 peak")
    ap.add_argument("--n-instances", type=int, default=1200)
    ap.add_argument("--n-cases", type=int, default=300)
    ap.add_argument("--seed-start", type=int, default=600_000_000,
                    help="same stream as sudoku_circuits, so both jobs see the same cases")
    ap.add_argument("--seed", type=int, default=0, help="for the random-feature control")
    ap.add_argument("--kind", type=str, default="pre_color", choices=["pre_color", "pre_node"],
                    help="which per-kind feature scoring to select from; pre_color is where "
                         "finding 21 puts all the justifies_next structure")
    ap.add_argument("--select", type=str, default="interpretable",
                    choices=["interpretable", "beats_null"],
                    help="beats_null widens the set for per-node coverage")
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
    cases = collect_cases(sudoku_cfg, vocab, args.n_instances, args.seed_start)
    print(f"{len(cases)} flippable round-0 deductions from {args.n_instances} instances")
    cases = cases[:args.n_cases]
    os.makedirs(args.out_dir, exist_ok=True)

    d_model = model.encoder.layers[0].self_attn.embed_dim
    summary = []
    for layer in args.layer:
        sae, mu, sigma = load_sae(args.sae_dir, layer, d_model, device=args.device)
        jn_all, by_node, other_alive = feature_sets(args.sae_dir, layer,
                                                    kind=args.kind, select=args.select)
        print(f"\nlayer {layer}: {len(jn_all)} interpretable {JN} features over "
              f"{len(by_node)} nodes, {len(other_alive)} other live features")
        if not jn_all:
            print(f"  no {JN} features at this layer -- nothing to ablate, skipping")
            continue

        rng = np.random.default_rng(args.seed + layer)
        rows = []
        for i, (adj, colors, hints, trace, case) in enumerate(cases):
            rows.append(run_case(model, sae, mu, sigma, vocab, args.device, layer,
                                 adj, trace, case, jn_all, by_node, other_alive, hints, rng))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(cases)} cases")

        # Only cases where BOTH selective conditions had features to remove can
        # speak to the bind-vs-nonbind contrast; reporting it over cases where
        # one side was a no-op would dilute it toward zero.
        paired = [r for r in rows if r["n_bind"] > 0 and r["n_nonbind"] > 0]
        with open(os.path.join(args.out_dir, f"ablate_layer{layer:02d}.csv"), "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wri.writeheader()
            wri.writerows(rows)

        names = ["clean", "recon", "zero_jn_all", "zero_jn_bind", "zero_jn_nonbind", "zero_random"]
        print(f"\n  ACCURACY on the forced colour  (n={len(rows)}, paired subset n={len(paired)})")
        rec = {}
        for nm in names:
            use = paired if nm in ("zero_jn_bind", "zero_jn_nonbind") else rows
            acc = sum(r[f"{nm}_correct"] for r in use) / max(1, len(use))
            marg = float(np.mean([r[f"{nm}_margin"] for r in use]))
            rec[nm] = (acc, marg)
            print(f"    {nm:<17} acc {acc:.3f}   mean margin {marg:+.3f}")

        x, y, p = mcnemar_exact(paired, "zero_jn_nonbind", "zero_jn_bind")
        d_bind = rec["zero_jn_bind"][0] - rec["recon"][0]
        d_rand = rec["zero_random"][0] - rec["recon"][0]
        print(f"\n  THE TEST  bind vs nonbind (paired, n={len(paired)}): "
              f"{rec['zero_jn_bind'][0]:.3f} vs {rec['zero_jn_nonbind'][0]:.3f}, "
              f"{x} nonbind-only / {y} bind-only, p={p:.3g}")
        print(f"  vs reconstruction baseline: bind {d_bind:+.3f}, random {d_rand:+.3f}")
        summary.append({
            "layer": layer, "n_cases": len(rows), "n_paired": len(paired),
            "n_jn_features": len(jn_all),
            **{f"{nm}_acc": rec[nm][0] for nm in names},
            **{f"{nm}_margin": rec[nm][1] for nm in names},
            "bind_vs_nonbind_p": p, "bind_minus_recon": d_bind, "random_minus_recon": d_rand,
        })

    if summary:
        with open(os.path.join(args.out_dir, "ablate_summary.csv"), "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            wri.writeheader()
            wri.writerows(summary)
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump({"checkpoint": args.checkpoint, "sae_dir": args.sae_dir,
                       "layers": args.layer, "n_instances": args.n_instances,
                       "n_cases": args.n_cases, "seed_start": args.seed_start,
                       "seed": args.seed, "kind": args.kind, "select": args.select}, f, indent=2)


if __name__ == "__main__":
    main()
