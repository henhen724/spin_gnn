"""Stage Z: is the installed determinacy signal the PREDICATE, or a COUNT?

Registered in scripts/stagez_score.py, which was committed before this ran.
Two measurements, both on existing checkpoints, neither of which trains
anything.

THE SETUP. Stages X and Y each installed a determinacy readout scoring
AUROC ~= 0.95 against DD's own predicate, and neither produced a guard: break
node v's determinacy with a one-token edit and the model names v anyway, at
the placebo's rate. The surviving explanation is that what got installed is not
the predicate but a CORRELATE of it, and sudoku_skip's corruption is built in
exactly the way that tells the two apart:

    forced(v) <=> v uncoloured AND |{colours among v's revealed nbrs}| == q-1

The corruption RECOLOURS a revealed neighbour w of v. So across the edit

    n_known_nbrs(v)    INVARIANT      -- w is revealed before and after
    n_known_colors(v)  q-1 -> q-2     -- which is what destroys forcing

A readout tracking the count cannot see the edit, however it is coupled to the
emission. That would explain both null results for a reason no re-coupling can
repair.

Z1  COUNTERFACTUAL RESPONSE OF THE CHANNEL (stage Y only). On the same 350
    guard cases, read stage Y's own determinacy score g[p, v] on the clean,
    corrupted and placebo inputs. The corruption destroys v's determinacy; the
    placebo leaves v forced. A predicate must fall under the first and not the
    second. Reported as the win rate P(|dg_corrupt| > |dg_placebo|), whose
    chance value is 0.50 because both edits are one-token hint recolours.

    Stage Y only because `return_determinacy=True` raises on a model built
    without a channel (sat_transformer2.py:263-267). Stage X has no separate
    score -- its representation IS the node logit, whose counterfactual
    response is already in skip_behavior.csv and is at chance.

Z2  THE COVARIATE CEILING ON THE AUROC. No model at all: compute the AUROC of
    the model-free count n_known_nbrs over exactly the cells
    determinacy_targets scores. If a count alone approaches 0.95, the
    published 0.95 is not evidence that anything computes the predicate.

    THE TRAP THIS MODULE EXISTS TO AVOID. determinacy_targets' known set at a
    position is round-start PLUS the deductions already emitted this round
    (sudoku_determinacy.py:129-130), which is NOT sudoku_skip._revealed_before's
    prefix-exact convention. Covariates computed on the wrong convention would
    be scored against labels from the other one and the comparison would be
    silently invalid. `_covariates` therefore reproduces that loop line for
    line and ASSERTS that its recomputed target equals the module's, so the
    two cannot drift apart without the job failing.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .config import SudokuReasoningConfig
from .sudoku_determinacy import MASKED, auroc, determinacy_targets
from .sudoku_skip import _best_other, build_triple, find_unforceable_deductions
from .sudoku_tokens import encode_instance
from .train_sudoku_transformer import build_instances, load_checkpoint


# ---------------------------------------------------------------------------
# Z1
# ---------------------------------------------------------------------------
def run_cfresp(model, vocab, device, cases):
    """Three forwards per case, reading the channel instead of the behaviour.

    `model(t, return_determinacy=True)` returns (z, g) where z is the emission
    AFTER the gate (z[node block] += log sigmoid(g)), so the behavioural
    columns computed here are the same quantities run_skip_behavior writes and
    must reproduce them case for case -- which is how we know the case list is
    the published one.
    """
    rows = []
    for adj, colors, hints, trace, case in cases:
        clean, broken, placebo, pos = build_triple(adj, trace, vocab, case)
        with torch.no_grad():
            (zc, gc), (zb, gb), (zp, gp) = (
                model(torch.from_numpy(t).unsqueeze(0).to(device), return_determinacy=True)
                for t in (clean, broken, placebo))
        v = case["node"]
        v_tok = vocab.node_token(v)

        # the channel's own score for v at the deciding position
        g_clean, g_broken, g_placebo = (float(g[0, pos, v]) for g in (gc, gb, gp))
        # ...against the spread the AUROC is computed over: g across all nodes
        # at this same position. Absolute units of g mean nothing on their own.
        g_all = gc[0, pos].float().cpu().numpy()
        g_spread = float(g_all.std())
        # where v sits in that ranking, clean. A predicate should put a forced
        # node near the top, so this is the AUROC's claim at the case level.
        g_rank_v = float((g_all < g_clean).mean())

        # what actually reaches the emission
        ls = lambda x: float(F.logsigmoid(torch.tensor(x, dtype=torch.float64)))
        dls_corrupt = ls(g_clean) - ls(g_broken)
        dls_placebo = ls(g_clean) - ls(g_placebo)

        margin = float(zc[0, pos][v_tok] - _best_other(zc[0, pos], v_tok))
        dz_corrupt = float(zc[0, pos][v_tok] - zb[0, pos][v_tok])
        dz_placebo = float(zc[0, pos][v_tok] - zp[0, pos][v_tok])
        rows.append({
            "node": v, "round0_rank": case["round0_rank"], "n_known": case["n_known"],
            # behavioural columns, for the cross-check against skip_behavior.csv
            "clean_emits_v": int(zc[0, pos].argmax()) == v_tok,
            "corrupt_emits_v": int(zb[0, pos].argmax()) == v_tok,
            "placebo_emits_v": int(zp[0, pos].argmax()) == v_tok,
            "clean_v_margin": margin,
            "corrupt_v_logit_drop": dz_corrupt,
            "placebo_v_logit_drop": dz_placebo,
            # Z1 proper
            "g_clean": g_clean, "g_broken": g_broken, "g_placebo": g_placebo,
            "g_spread": g_spread, "g_rank_v": g_rank_v,
            # positive = the channel's score for v FELL, which is the direction
            # a guard needs
            "dg_corrupt": g_clean - g_broken,
            "dg_placebo": g_clean - g_placebo,
            "dg_corrupt_norm": (g_clean - g_broken) / g_spread if g_spread > 0 else float("nan"),
            "dlogsig_corrupt": dls_corrupt,
            "dlogsig_placebo": dls_placebo,
            # could the channel's own response have changed the argmax, holding
            # everything else fixed? First-order -- the rivals' gates move too --
            # but it is the behavioural scale the response has to reach.
            "channel_response_exceeds_margin": dls_corrupt >= margin,
            # how much of the (small) total logit response the channel accounts
            # for; nan when the total is ~0 and the ratio is meaningless
            "channel_share_of_drop": (dls_corrupt / dz_corrupt
                                      if abs(dz_corrupt) > 1e-9 else float("nan")),
        })
    return rows


def summarize_cfresp(rows):
    n = len(rows)
    a = {k: np.array([r[k] for r in rows], dtype=np.float64)
         for k in ("dg_corrupt", "dg_placebo", "dg_corrupt_norm", "dlogsig_corrupt",
                   "dlogsig_placebo", "g_spread", "g_rank_v", "clean_v_margin",
                   "corrupt_v_logit_drop", "channel_share_of_drop")}
    share = a["channel_share_of_drop"]
    # THE DENOMINATOR. "Large enough to have changed the argmax" is only a
    # question where v IS the clean argmax: when it is not, clean_v_margin is
    # negative and any positive response trivially exceeds it. So flip_frac is
    # conditional on clean_emits_v, exactly as the published guard rate is, and
    # the denominator is written down rather than assumed.
    flips = [r["channel_response_exceeds_margin"] for r in rows if r["clean_emits_v"]]
    return {
        "n_cases": n,
        # the registered primary statistic; chance is 0.50
        "win_rate": float((np.abs(a["dg_corrupt"]) > np.abs(a["dg_placebo"])).mean()),
        "norm_drop": float(np.median(a["dg_corrupt_norm"])),
        "flip_frac": float(np.mean(flips)) if flips else float("nan"),
        "flip_denom": len(flips),
        "median_dg_corrupt": float(np.median(a["dg_corrupt"])),
        "median_dg_placebo": float(np.median(a["dg_placebo"])),
        "median_abs_dg_corrupt": float(np.median(np.abs(a["dg_corrupt"]))),
        "median_abs_dg_placebo": float(np.median(np.abs(a["dg_placebo"]))),
        "median_g_spread": float(np.median(a["g_spread"])),
        "median_g_rank_v": float(np.median(a["g_rank_v"])),
        "median_dlogsig_corrupt": float(np.median(a["dlogsig_corrupt"])),
        "median_clean_v_margin": float(np.median(a["clean_v_margin"])),
        "median_channel_share_of_drop": float(np.nanmedian(share)) if np.isfinite(share).any() else float("nan"),
        # behavioural, for the cross-check
        "clean_emits_v": float(np.mean([r["clean_emits_v"] for r in rows])),
        "corrupt_emits_v": float(np.mean([r["corrupt_emits_v"] for r in rows])),
        "placebo_emits_v": float(np.mean([r["placebo_emits_v"] for r in rows])),
    }


# ---------------------------------------------------------------------------
# Z2
# ---------------------------------------------------------------------------
def _covariates(adj, trace, vocab, tokens):
    """Model-free per-(position, node) covariates on determinacy_targets' own
    positions, convention and masking.

    Deliberately a line-for-line copy of determinacy_targets' loop rather than
    a call into it: the point of the measurement is that the covariates are
    scored on exactly the cells the labels come from, and the only way to be
    sure of that is to recompute the labels here and assert they match.

    Returns (pos, target, {name: (P, N) float array}).
    """
    from .sudoku_activations import reasoning_positions

    n_nodes, n_colors = vocab.n_nodes, vocab.n_colors
    adj = np.asarray(adj)
    positions = [p for p in reasoning_positions(tokens, vocab) if p["kind"] == "pre_node"]
    rounds = [[(int(u), int(c)) for u, c in rd] for rd in trace["rounds"]]

    pre_round_known = []
    known = {int(u): int(c) for u, c in trace["hints"].items()}
    for rd in rounds:
        pre_round_known.append(dict(known))
        known.update(dict(rd))

    P = len(positions)
    pos = np.zeros(P, dtype=np.int64)
    target = np.zeros((P, n_nodes), dtype=np.int8)
    n_known_nbrs = np.zeros((P, n_nodes), dtype=np.float64)
    n_known_colors_out = np.zeros((P, n_nodes), dtype=np.float64)
    seen = np.zeros((n_nodes, n_colors + 1), dtype=bool)
    row_ix = np.repeat(np.arange(n_nodes), adj.shape[1])

    for p, info in enumerate(positions):
        pos[p] = info["pos"]
        cur = dict(pre_round_known[info["round"]])
        cur.update(dict(rounds[info["round"]][: info["k"]]))

        node_color = np.full(n_nodes, n_colors, dtype=np.int64)
        if cur:
            idx = np.fromiter(cur.keys(), dtype=np.int64, count=len(cur))
            val = np.fromiter(cur.values(), dtype=np.int64, count=len(cur))
            node_color[idx] = val
        seen[:] = False
        seen[row_ix, node_color[adj].ravel()] = True
        n_known_colors = seen[:, :n_colors].sum(1)

        is_known = node_color < n_colors
        target[p] = (n_known_colors == n_colors - 1).astype(np.int8)
        target[p][is_known] = MASKED

        # THE COVARIATE. How many neighbours are revealed, counting a repeated
        # colour twice -- invariant under sudoku_skip's corruption, which only
        # changes WHICH colour a revealed neighbour shows.
        n_known_nbrs[p] = (node_color[adj] < n_colors).sum(1)
        # The predicate's own ingredient, as a ceiling: the target is a
        # threshold of this, so its AUROC is 1.0 by construction. Printed to
        # show what separating the two covariates is worth, not as a baseline.
        n_known_colors_out[p] = n_known_colors
    return pos, target, {"n_known_nbrs": n_known_nbrs,
                         "n_known_colors": n_known_colors_out}


def covariate_auroc(instances, vocab):
    """AUROC of model-free covariates over determinacy_targets' scored cells."""
    labels, cols = [], {}
    n_cells = n_scored = 0
    for adj, colors, hints, trace in instances:
        tokens = encode_instance(adj, trace, vocab)
        ref_pos, ref_target = determinacy_targets(adj, trace, vocab, tokens)
        pos, target, cov = _covariates(adj, trace, vocab, tokens)
        # the guarantee the whole measurement rests on
        assert np.array_equal(pos, ref_pos), "covariate positions drifted from the target's"
        assert np.array_equal(target, ref_target), "covariate loop no longer reproduces the target"

        scored = target != MASKED
        n_cells += target.size
        n_scored += int(scored.sum())
        labels.append((target[scored] == 1))
        for k, v in cov.items():
            cols.setdefault(k, []).append(v[scored])

    y = np.concatenate(labels)
    out = {k: float(auroc(np.concatenate(v), y)) for k, v in cols.items()}
    return out, {
        "n_instances": len(instances),
        "n_cells": n_cells, "n_scored": n_scored,
        "frac_scored": n_scored / n_cells,
        "positive_rate": float(y.mean()),
    }


# ---------------------------------------------------------------------------
# Z3
# ---------------------------------------------------------------------------
def within_stratum_auroc(scores, labels, strata):
    """P(a positive outranks a negative WITH THE SAME stratum value), pooled
    over strata by pair count.

    The pooled AUROC mixes two different rankings: "a node with more revealed
    neighbours over one with fewer" and "a forced node over an unforced one at
    the same degree of revelation". Only the second is the predicate.
    Conditioning on the covariate removes the first by construction, which is
    why the covariate's own value here is exactly 0.5 -- it is constant inside
    its own stratum, so every pair ties and `auroc` averages them.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    strata = np.asarray(strata)
    num = den = 0.0
    per = {}
    for s in np.unique(strata):
        m = strata == s
        y = labels[m]
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        rec = {"n": int(m.sum()), "n_pos": n_pos, "n_neg": n_neg}
        if n_pos and n_neg:
            a = float(auroc(scores[m], y))
            rec["auroc"] = a
            # weight by the number of (positive, negative) pairs the stratum
            # contributes, so the pooled value is still a probability over
            # pairs and a stratum with three cells cannot swing it
            num += a * n_pos * n_neg
            den += n_pos * n_neg
        else:
            rec["auroc"] = None      # no pair to rank; contributes nothing
        per[int(s)] = rec
    return (num / den if den else float("nan")), per


def run_stratified(model, vocab, device, instances):
    """Z3 for one checkpoint, plus the two covariates that calibrate the scale.

    One forward per instance rather than the batched path evaluate_determinacy
    uses: the covariates are aligned to determinacy_targets' (P, N) layout, and
    reproducing make_token_batch_aux's padding to keep that alignment would be
    a second place for the two to drift. The pooled AUROC is returned so it can
    be checked against the published figure, which is the proof that this reads
    the same cells.
    """
    scores, labels, strata, colors = [], [], [], []
    for adj, c, h, trace in instances:
        tokens = encode_instance(adj, trace, vocab)
        ref_pos, ref_target = determinacy_targets(adj, trace, vocab, tokens)
        pos, target, cov = _covariates(adj, trace, vocab, tokens)
        assert np.array_equal(pos, ref_pos) and np.array_equal(target, ref_target)
        with torch.no_grad():
            _z, g = model(torch.from_numpy(tokens).unsqueeze(0).to(device),
                          return_determinacy=True)
        gp = g[0].float().cpu().numpy()[pos]          # (P, N), same gather as evaluate_determinacy
        scored = target != MASKED
        scores.append(gp[scored])
        labels.append(target[scored] == 1)
        strata.append(cov["n_known_nbrs"][scored])
        colors.append(cov["n_known_colors"][scored])

    s = np.concatenate(scores)
    y = np.concatenate(labels)
    k = np.concatenate(strata)
    kc = np.concatenate(colors)
    model_within, per_stratum = within_stratum_auroc(s, y, k)
    # the two ends of the scale, both exact by construction
    nbrs_within, _ = within_stratum_auroc(k, y, k)
    cols_within, _ = within_stratum_auroc(kc, y, k)
    return {
        "model": model_within,
        "model_pooled": float(auroc(s, y)),
        "per_stratum": per_stratum,
        "tautologies": {"n_known_nbrs": nbrs_within, "n_known_colors": cols_within},
    }


# ---------------------------------------------------------------------------
def _cfg_from(config):
    return SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        rho=config["rho"], rho_max=config["rho_max"],
        scratchpad=config.get("scratchpad", False), hint_mode=config.get("hint_mode", "removal"),
        shuffle_adj=bool(config.get("shuffle_adj", False)),
        shuffle_events=bool(config.get("shuffle_events", False)),
        fall=bool(config.get("fall", False)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, nargs="+", required=True,
                    help="stage Y cells; each must have a determinacy channel")
    ap.add_argument("--n-instances", type=int, default=400)
    ap.add_argument("--n-cases", type=int, default=350)
    ap.add_argument("--seed-start", type=int, default=600_000_000)
    ap.add_argument("--auroc-n-instances", type=int, default=600)
    ap.add_argument("--auroc-seed-start", type=int, default=70_000_000_000)
    ap.add_argument("--skip-dir", type=str, nargs="*", default=[],
                    help="stage Y's skip_behavior.csv dirs, one per checkpoint, to "
                         "cross-check that this rebuilt the published case list")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    per_seed, cases, vocab, sudoku_cfg = [], None, None, None
    det_instances, strat_seeds = None, []
    for i, ckpt in enumerate(args.checkpoint):
        model, vocab, config = load_checkpoint(ckpt, device=args.device)
        model.eval()
        if not config.get("determinacy_channel"):
            raise SystemExit(f"{ckpt} has no determinacy channel; Z1 is stage Y only")
        sudoku_cfg = _cfg_from(config)

        if cases is None:
            # one case list, shared across cells, built exactly as sudoku_skip
            # builds it: same rng seed, same stream, same truncation
            rng = np.random.default_rng(0)
            cases = []
            for adj, colors, hints, trace in build_instances(
                    sudoku_cfg, args.n_instances, args.seed_start):
                for case in find_unforceable_deductions(adj, colors, hints, trace, vocab, rng):
                    cases.append((adj, colors, hints, trace, case))
            print(f"{len(cases)} breakable round-0 deductions from "
                  f"{args.n_instances} instances", flush=True)
            cases = cases[:args.n_cases]

        rows = run_cfresp(model, vocab, args.device, cases)
        s = summarize_cfresp(rows)
        s["checkpoint"] = ckpt

        # Did we rebuild the published case list? The behavioural columns are
        # the same quantities sudoku_skip wrote, so they must agree exactly.
        if i < len(args.skip_dir):
            with open(os.path.join(args.skip_dir[i], "skip_summary.json")) as fh:
                ref = json.load(fh)
            for k in ("clean_emits_v", "corrupt_emits_v", "placebo_emits_v"):
                if k in ref:
                    assert abs(ref[k] - s[k]) < 1e-9, (
                        f"{k}: rebuilt {s[k]} vs published {ref[k]} -- different case list")
            s["cross_check_vs_skip_summary"] = "exact"

        with open(os.path.join(args.out_dir, f"cfresp_s{i}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        per_seed.append(s)
        print(f"\n[{ckpt}]")
        print(f"  win_rate   {s['win_rate']:.4f}   (chance 0.50)  <- HEADLINE")
        print(f"  norm_drop  {s['norm_drop']:+.4f}  (median dg/spread; bar 0.25)")
        print(f"  flip_frac  {s['flip_frac']:.4f}  over {s['flip_denom']} cases where "
              f"v IS the clean argmax (of {s['n_cases']})")
        print(f"  median |dg| corrupt {s['median_abs_dg_corrupt']:.4f} vs "
              f"placebo {s['median_abs_dg_placebo']:.4f}, g spread {s['median_g_spread']:.3f}")
        print(f"  v's clean percentile in g: {s['median_g_rank_v']:.4f}")
        print(f"  emits_v clean/corrupt/placebo "
              f"{s['clean_emits_v']:.4f}/{s['corrupt_emits_v']:.4f}/{s['placebo_emits_v']:.4f}",
              flush=True)

        # Z3, on the determinacy set rather than the guard cases. Built once
        # and shared: it is the same 600 instances the published AUROCs were
        # measured on, and rebuilding it per checkpoint is the expensive part.
        if det_instances is None:
            det_instances = list(build_instances(
                sudoku_cfg, args.auroc_n_instances, args.auroc_seed_start))
        st = run_stratified(model, vocab, args.device, det_instances)
        st["checkpoint"] = ckpt
        strat_seeds.append(st)
        print(f"  Z3 within-stratum AUROC {st['model']:.4f}   "
              f"(pooled {st['model_pooled']:.4f}; 0.5 = the count, 1.0 = the predicate)",
              flush=True)

    print("\n=== Z2: covariate ceiling ===", flush=True)
    cov, meta = covariate_auroc(det_instances, vocab)
    print(f"  {meta['n_scored']} scored cells of {meta['n_cells']} "
          f"({meta['frac_scored']:.3f}), positive rate {meta['positive_rate']:.3f}")
    for k, v in cov.items():
        print(f"  AUROC({k:>16s}) = {v:.4f}")

    taut = strat_seeds[0]["tautologies"]
    print("\n=== Z3: within-stratum AUROC, given n_known_nbrs ===")
    print(f"  scale ends (asserted): n_known_nbrs {taut['n_known_nbrs']:.4f} "
          f"(must be 0.5000), n_known_colors {taut['n_known_colors']:.4f} (must be 1.0000)")
    assert abs(taut["n_known_nbrs"] - 0.5) < 1e-9, "the stratification is not what it claims"
    assert abs(taut["n_known_colors"] - 1.0) < 1e-9, "scored cells are not the uncoloured ones"
    for st in strat_seeds:
        print(f"  {st['model']:.4f}   (pooled {st['model_pooled']:.4f})")
    # the proof that Z3 reads the same cells as the published AUROC column
    print("  pooled vs published 0.9493/0.9478/0.9480: "
          + "/".join(f"{st['model_pooled']:.4f}" for st in strat_seeds))

    out = {"per_seed": per_seed, "covariate_auroc": cov, "covariate_meta": meta,
           "guard_case_set": {"n_instances": args.n_instances, "n_cases": len(cases),
                              "seed_start": args.seed_start},
           "stratified": {"per_seed": strat_seeds, "tautologies": taut,
                          "n_instances": len(det_instances),
                          "seed_start": args.auroc_seed_start}}
    with open(os.path.join(args.out_dir, "cfresp_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out_dir}/cfresp_summary.json", flush=True)


if __name__ == "__main__":
    main()
