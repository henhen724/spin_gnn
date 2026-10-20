"""Sparse autoencoders on the DD-Sudoku transformer's residual stream, with
features scored automatically against ground-truth DD predicates.

An SAE on its own yields a dictionary nobody has time to read. The thing that
makes it decisive here is that the target algorithm is known: for every
position we have DD's own state (spin_gnn.sudoku_activations), so each
learned feature can be matched automatically against a concrete, named
hypothesis:

  frontier[v]     -- node v is deducible right now (the DD predicate itself)
  known[v]        -- node v's color is known (bookkeeping)
  color[v]=c      -- node v is known to be color c
  next_node[v]    -- v is the node about to be emitted
  nbr_count[v]=j  -- exactly j of v's neighbors are known (DD's own
                     intermediate quantity, which it thresholds at
                     n_colors-1)

Each feature gets the best-matching predicate by F1 over positions. A
dictionary in which high-F1 frontier[v] features exist is direct evidence
the model represents the DD frontier in superposition; one where only
known[v] and color[v]=c features appear says it is doing bookkeeping and
getting the deductions some other way.

Usage:
    python -m spin_gnn.sudoku_sae --checkpoint <dir> --layer 8 --out-dir <dir>
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .config import SudokuReasoningConfig
from .sudoku_activations import collect_dataset
from .train_sudoku_transformer import build_instances, load_checkpoint


class SparseAutoencoder(torch.nn.Module):
    """f = ReLU(W_enc (x - b_dec) + b_enc);  x_hat = W_dec f + b_dec.

    Decoder columns are renormalized to unit norm after every step, the
    standard guard against the trivial route to a small L1 penalty (shrink
    feature activations, grow decoder norms, leave the reconstruction
    unchanged)."""

    def __init__(self, d_model, d_sae, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(d_sae, d_model, generator=g) / np.sqrt(d_model)
        self.W_enc = torch.nn.Parameter(w.clone())
        self.W_dec = torch.nn.Parameter(w.clone().T.contiguous())
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)

    def forward(self, x):
        f = self.encode(x)
        return f @ self.W_dec.T + self.b_dec, f


def train_sae(acts, d_sae, l1_coeff=0.1, epochs=1500, batch_size=2048, lr=3e-3, device="cpu", seed=0,
              dead_feature_window=50):
    """acts: (P, d_model) float tensor of residual activations.

    Inputs are standardized per-dimension, so l1_coeff means the same thing
    across layers (residual norm grows substantially with depth in a pre-norm
    stack, and an unnormalized penalty would effectively be a different
    sparsity target at every layer).

    Defaults were calibrated on a planted sparse code (see
    tests/test_sudoku_sae.py::test_sae_recovers_a_planted_sparse_code): with
    l1_coeff=0.1 the sparsity term only bites slowly, so an earlier default of
    400 epochs produced a dense dictionary (L0~21 against a true sparsity of
    0.9) that reconstructed almost perfectly and would have been uninterpretable
    while looking healthy on FVE alone. 1500 epochs reaches L0~2.2 at
    FVE~0.999. Check the reported L0 against the number of DD predicates you
    expect before trusting any feature labels."""
    x = acts.to(device).float()
    mu, sigma = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6)
    x = (x - mu) / sigma
    total_var = x.var(0).sum().item()

    sae = SparseAutoencoder(x.shape[1], d_sae, seed=seed).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    fired_recently = torch.zeros(d_sae, dtype=torch.bool, device=device)

    for epoch in range(epochs):
        perm = torch.randperm(x.shape[0], generator=gen).to(device)
        ep_recon = ep_l0 = ep_n = 0.0
        if epoch % dead_feature_window == 0:
            fired_recently = torch.zeros(d_sae, dtype=torch.bool, device=device)
        for i in range(0, x.shape[0], batch_size):
            xb = x[perm[i : i + batch_size]]
            xhat, f = sae(xb)
            recon = (xhat - xb).pow(2).sum(-1).mean()
            loss = recon + l1_coeff * f.abs().sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            with torch.no_grad():
                fired_recently |= (f > 0).any(0)
                ep_recon += recon.item() * xb.shape[0]
                ep_l0 += (f > 0).float().sum(-1).sum().item()
                ep_n += xb.shape[0]
        if epoch % 25 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                xhat, f = sae(x)
                resid_var = (x - xhat).var(0).sum().item()
            history.append({
                "epoch": epoch,
                "recon_mse": ep_recon / ep_n,
                "l0": ep_l0 / ep_n,
                "frac_variance_explained": 1.0 - resid_var / max(1e-9, total_var),
                "n_dead_features": int((~fired_recently).sum()),
            })
            h = history[-1]
            print(f"  sae epoch {epoch:4d}: mse={h['recon_mse']:.4f} L0={h['l0']:.1f} "
                  f"FVE={h['frac_variance_explained']:.3f} dead={h['n_dead_features']}")

    with torch.no_grad():
        _, f = sae(x)
    return sae, f.cpu(), history, (mu.cpu(), sigma.cpu())


def train_sae_to_target_l0(acts, d_sae, target_l0=32.0, l1_start=0.1, max_tries=7,
                           tolerance=2.0, **kwargs):
    """Search l1_coeff geometrically until L0 lands at or below `target_l0`.

    A fixed l1_coeff does not transport across activation widths. The
    reconstruction term is a SUM over dimensions, so it grows with d_model
    while the L1 term does not; l1_coeff=0.1, calibrated on a 12-dimensional
    planted code, is therefore ~10x too weak at d_model=128 and yields
    L0 = 84..278 of 2048 features with ZERO dead features -- a dense
    dictionary whose feature labels mean nothing, while still reporting
    FVE=0.992 and so looking healthy on reconstruction alone. That is the
    exact failure this function exists to prevent.

    Returns the first fit meeting the target, or the sparsest one tried, with
    the search path attached so a dictionary that never got sparse is visible
    in the output rather than silently reported as a result.
    """
    path = []
    best = None
    l1 = l1_start
    for _ in range(max_tries):
        sae, feats, history, norm = train_sae(acts, d_sae, l1_coeff=l1, **kwargs)
        l0, fve = history[-1]["l0"], history[-1]["frac_variance_explained"]
        path.append({"l1_coeff": l1, "l0": l0, "frac_variance_explained": fve,
                     "n_dead_features": history[-1]["n_dead_features"]})
        print(f"  l1={l1:.4g} -> L0={l0:.1f} FVE={fve:.3f}")
        if best is None or l0 < best[2][-1]["l0"]:
            best, best_l1 = (sae, feats, history, norm), l1
        if l0 <= target_l0 + tolerance:
            break
        # L0 responds roughly logarithmically to l1, so step geometrically and
        # grow the step when still far off.
        l1 *= 4.0 if l0 > 4 * target_l0 else 2.0
    sae, feats, history, norm = best
    history[-1]["l1_search_path"] = path
    history[-1]["l1_coeff"] = best_l1  # the l1 that produced THIS fit, not the last tried
    history[-1]["hit_target_l0"] = history[-1]["l0"] <= target_l0 + tolerance
    return sae, feats, history, norm


def dd_predicates(labels, n_nodes, n_colors, max_nbr_count=None):
    """Named boolean predicates over positions, as (name, (P,) bool array)."""
    preds = []
    for v in range(n_nodes):
        preds.append((f"frontier[{v}]", labels["frontier_mask"][:, v].astype(bool)))
        preds.append((f"remaining[{v}]", labels["remaining_mask"][:, v].astype(bool)))
        preds.append((f"known[{v}]", labels["known_mask"][:, v].astype(bool)))
        preds.append((f"next_node[{v}]", labels["next_node"] == v))
        for c in range(n_colors):
            preds.append((f"color[{v}]={c}", labels["node_color"][:, v] == c))
    for c in range(n_colors):
        preds.append((f"next_color={c}", labels["next_color"] == c))
    hi = max_nbr_count if max_nbr_count is not None else int(labels["n_known_nbrs"].max())
    for j in range(hi + 1):
        # Aggregated over nodes: "some node has exactly j known neighbors" is
        # not useful, so this is the count for the node about to be emitted,
        # which is the one DD is thresholding.
        nxt = labels["next_node"]
        cnt = labels["n_known_nbrs"][np.arange(len(nxt)), nxt]
        preds.append((f"next_node_nbr_count={j}", cnt == j))
    preds.append(("kind_is_node", labels["kind_is_node"].astype(bool)))
    return preds


def score_features(feature_acts, labels, n_nodes, n_colors, min_f1=0.3):
    """Best-matching DD predicate per feature, by F1 of (feature fires) vs
    (predicate holds). Features are binarized at >0, which is the natural
    threshold for a ReLU dictionary."""
    fired = (feature_acts > 0).numpy()
    preds = dd_predicates(labels, n_nodes, n_colors)
    names = [n for n, _ in preds]
    mat = np.stack([p for _, p in preds], axis=1)  # (P, n_pred)

    fired_f = fired.astype(np.float32)
    mat_f = mat.astype(np.float32)
    tp = fired_f.T @ mat_f                       # (n_feat, n_pred)
    n_fire = fired_f.sum(0)[:, None]
    n_pred = mat_f.sum(0)[None, :]
    f1 = 2 * tp / np.maximum(1e-9, n_fire + n_pred)

    rows = []
    for i in range(f1.shape[0]):
        j = int(np.argmax(f1[i]))
        rows.append({
            "feature": i,
            "firing_rate": float(fired_f[:, i].mean()),
            "mean_activation_when_firing": float(
                feature_acts[:, i][feature_acts[:, i] > 0].mean()
            ) if fired[:, i].any() else 0.0,
            "best_predicate": names[j],
            "best_f1": float(f1[i, j]),
            "best_predicate_family": names[j].split("[")[0].split("=")[0],
            "is_interpretable": bool(f1[i, j] >= min_f1),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--layer", type=int, nargs="+", default=[4, 8, 12],
                    help="0 = token+position embedding, i>=1 = output of block i-1")
    ap.add_argument("--expansion", type=int, default=16, help="d_sae = expansion * d_model")
    ap.add_argument("--l1-coeff", type=float, default=0.1,
                    help="with --target-l0, the starting point of the search")
    ap.add_argument("--target-l0", type=float, default=32.0,
                    help="search l1_coeff until mean L0 reaches this; 0 disables the search and "
                         "uses --l1-coeff as-is. A dense dictionary makes every feature label "
                         "meaningless while still reporting a high FVE, so leave this on.")
    ap.add_argument("--sae-epochs", type=int, default=1500)
    ap.add_argument("--n-instances", type=int, default=1500)
    ap.add_argument("--max-positions-per-instance", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, config = load_checkpoint(args.checkpoint, device=device)
    sudoku_cfg = SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        rho=config["rho"], rho_max=config["rho_max"],
        hint_mode=config.get("hint_mode", "removal"), scratchpad=config.get("scratchpad", False),
    )
    instances = build_instances(sudoku_cfg, args.n_instances, seed_start=600_000_000 + args.seed)
    acts, labels = collect_dataset(
        model, instances, vocab, device, args.max_positions_per_instance, np.random.default_rng(args.seed),
    )
    print(f"collected {acts.shape[1]} positions from {args.n_instances} instances, {acts.shape[0]} layers")

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for layer in args.layer:
        if layer >= acts.shape[0]:
            print(f"skipping layer {layer}: model has {acts.shape[0] - 1} blocks")
            continue
        print(f"layer {layer}: training SAE (d_sae={args.expansion * acts.shape[2]})")
        if args.target_l0 > 0:
            sae, feats, history, _ = train_sae_to_target_l0(
                acts[layer], args.expansion * acts.shape[2], target_l0=args.target_l0,
                l1_start=args.l1_coeff, epochs=args.sae_epochs, device=device, seed=args.seed,
            )
        else:
            sae, feats, history, _ = train_sae(
                acts[layer], args.expansion * acts.shape[2], l1_coeff=args.l1_coeff,
                epochs=args.sae_epochs, device=device, seed=args.seed,
            )
        rows = score_features(feats, labels, sudoku_cfg.n_nodes, sudoku_cfg.n_colors)

        torch.save(sae.state_dict(), os.path.join(args.out_dir, f"sae_layer{layer:02d}.pt"))
        with open(os.path.join(args.out_dir, f"features_layer{layer:02d}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(args.out_dir, f"history_layer{layer:02d}.json"), "w") as f:
            json.dump(history, f, indent=2)

        alive = [r for r in rows if r["firing_rate"] > 0]
        interp = [r for r in alive if r["is_interpretable"]]
        fam = {}
        for r in interp:
            fam[r["best_predicate_family"]] = fam.get(r["best_predicate_family"], 0) + 1
        best_frontier = max((r["best_f1"] for r in rows if r["best_predicate"].startswith("frontier")),
                            default=0.0)
        summary.append({
            "checkpoint": os.path.basename(args.checkpoint), "layer": layer,
            "d_sae": args.expansion * acts.shape[2], "l1_coeff": args.l1_coeff,
            "n_positions": int(acts.shape[1]),
            "frac_variance_explained": history[-1]["frac_variance_explained"],
            "l0": history[-1]["l0"], "n_dead_features": history[-1]["n_dead_features"],
            # False means the dictionary stayed dense and the feature labels in
            # features_layer*.csv must not be read as semantics.
            "hit_target_l0": history[-1].get("hit_target_l0", None),
            "l1_final": history[-1].get("l1_coeff", args.l1_coeff),
            "n_alive": len(alive), "n_interpretable": len(interp),
            "best_frontier_f1": best_frontier,
            "family_counts": json.dumps(fam, sort_keys=True),
        })
        print(f"  layer {layer}: {len(interp)}/{len(alive)} alive features matched a DD predicate "
              f"(F1>=0.3); best frontier F1={best_frontier:.3f}; families={fam}")

    with open(os.path.join(args.out_dir, "sae_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {args.out_dir}/sae_summary.csv")


if __name__ == "__main__":
    main()
