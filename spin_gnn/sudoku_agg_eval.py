"""Compare checkpoints on ONE shared held-out instance set, paired.

Two problems with reading agg_solve_rate out of sweep.csv, both of which bite
exactly now that the top models are near the ceiling.

First, the sets are not the same. train_sudoku_model builds its validation
instances at seed_start = seed + n_train_instances, so a run with
n_train=20000 validates on a different 300 instances than a run with
n_train=5000. The protocol matches (300 instances, the training rho range),
and both are unbiased draws from the same distribution, so the comparison is
not wrong -- but it is not the identical instance set, and the difference
carries the sampling noise of two independent draws. Worse, the data4x runs
train on seeds seed+5000..seed+19999, which *contains* the 5000-instance runs'
validation seeds, so those two sets are not even independent in the direction
one would hope.

Second, 300 instances cannot resolve the differences that are left. At a rate
near 0.98 the standard error is about 0.008, so two models four points apart
are a coin flip. Nothing above ~0.97 can be ranked at that resolution, and
reporting a new best from it is reporting noise.

This module fixes both. One instance pool, built from a seed region disjoint
from every training set used so far, is scored by every checkpoint, and the
comparison is paired: only instances where two models disagree carry
information about their difference, so the test is a McNemar exact binomial on
the discordant pairs rather than a difference of two independent rates. Near
the ceiling that is dramatically more sensitive -- two models that each solve
98% of 1500 instances may disagree on only 40, and those 40 are the entire
evidence.

The instance-defining config (n_nodes, degree, n_colors, rho, rho_max,
hint_mode, scratchpad) must match across checkpoints or the comparison is
meaningless, so a mismatch is refused rather than reported.

Usage:
    python -m spin_gnn.sudoku_agg_eval --checkpoint <ref> <other> ... \\
        --n-instances 1500 --out-csv agg.csv
"""
import argparse
import csv
import json
import math
import os

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .train_sudoku_transformer import build_instances, load_checkpoint, solve_rate

# Keys that define the instance distribution; models differing on any of these
# are not being asked the same question. The two with defaults were added to
# config.json after the earliest sweeps, so a checkpoint from before then has
# no such key -- that means it used the default, not that it used something
# different, and reading it as a mismatch would wrongly block every comparison
# against the original sweep.
INSTANCE_KEYS = ("n_nodes", "degree", "n_colors", "rho", "rho_max", "hint_mode", "scratchpad")
INSTANCE_KEY_DEFAULTS = {"hint_mode": "removal", "scratchpad": False}


def mcnemar_exact_p(b, c):
    """Two-sided exact binomial p-value on the discordant pairs.

    b = reference solved and the other failed, c = the reverse. Under the null
    that the two models are equally likely to win a disagreement, b ~
    Binomial(b + c, 0.5). Exact rather than chi-square because the discordant
    count is small by construction when both models are near the ceiling,
    which is precisely the regime this module exists for.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def wilson_interval(k, n, z=1.96):
    """Wilson score interval, which unlike k/n +- z*SE stays inside [0, 1].

    At 0.99 over 1500 trials the normal interval runs past 1.0, so it cannot
    describe the models this is used on.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def instance_signature(config):
    sig = {}
    for k in INSTANCE_KEYS:
        v = config.get(k)
        sig[k] = INSTANCE_KEY_DEFAULTS[k] if v is None and k in INSTANCE_KEY_DEFAULTS else v
    return sig


def check_comparable(signatures, names):
    ref = signatures[0]
    for sig, name in zip(signatures[1:], names[1:]):
        diff = {k: (ref[k], sig[k]) for k in INSTANCE_KEYS if ref[k] != sig[k]}
        if diff:
            raise SystemExit(
                f"{name} was trained on a different instance distribution than {names[0]}: "
                f"{diff} -- a shared-instance comparison would be meaningless"
            )
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True,
                    help="the first is the reference every other is compared against")
    ap.add_argument("--n-instances", type=int, default=1500)
    # Far above every training seed used so far (max 20000) and every other
    # eval module's base, so this pool has never been trained or tuned on.
    ap.add_argument("--seed-start", type=int, default=50_000_000_000)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded, names = [], []
    for ckpt in args.checkpoint:
        model, vocab, config = load_checkpoint(ckpt, device=device)
        loaded.append((model, vocab, config))
        names.append(os.path.basename(ckpt.rstrip("/")))

    # Disambiguate identically-named checkpoints from different sweeps, which
    # is the normal case here (the same tag appears in several sweep dirs).
    parents = [os.path.basename(os.path.dirname(os.path.abspath(c.rstrip("/"))))
               for c in args.checkpoint]
    if len(set(names)) < len(names):
        names = [f"{n} [{p}]" for n, p in zip(names, parents)]

    sig = check_comparable([instance_signature(c) for _, _, c in loaded], names)
    cfg = SudokuReasoningConfig(
        n_nodes=sig["n_nodes"], degree=sig["degree"], n_colors=sig["n_colors"],
        rho=sig["rho"], rho_max=sig["rho_max"],
        hint_mode=sig["hint_mode"] or "removal", scratchpad=bool(sig["scratchpad"]),
    )
    print(f"shared pool: {args.n_instances} instances, rho in "
          f"[{cfg.rho}, {cfg.rho_max}], seed_start {args.seed_start}")
    instances = build_instances(cfg, args.n_instances, args.seed_start)

    results = []
    for (model, vocab, config), name in zip(loaded, names):
        m = solve_rate(model, instances, vocab, device, config["max_len"],
                       return_per_instance=True)
        results.append({"name": name, **m})
        lo, hi = wilson_interval(int(m["solved_per_instance"].sum()), args.n_instances)
        print(f"  {m['solve_rate']:.4f}  [{lo:.4f}, {hi:.4f}]  truncated "
              f"{m['truncated_rate']:.4f}  {name}")

    ref = results[0]
    print(f"\n=== paired against {ref['name']} (same {args.n_instances} instances) ===")
    rows = []
    for r in results:
        a, b_arr = ref["solved_per_instance"], r["solved_per_instance"]
        b = int((a & ~b_arr).sum())   # reference solved, this one did not
        c = int((~a & b_arr).sum())   # this one solved, reference did not
        p = mcnemar_exact_p(b, c)
        delta = r["solve_rate"] - ref["solve_rate"]
        if r is not ref:
            print(f"  {delta:+.4f}  discordant {b + c:4d} ({c} won / {b} lost)  "
                  f"exact p = {p:.4g}  {r['name']}")
        lo, hi = wilson_interval(int(b_arr.sum()), args.n_instances)
        rows.append({
            "checkpoint": r["name"], "n_instances": args.n_instances,
            "solve_rate": r["solve_rate"], "wilson_lo": lo, "wilson_hi": hi,
            "truncated_rate": r["truncated_rate"],
            "outcome_accuracy": r["outcome_accuracy"],
            "vs_reference": ref["name"], "delta": delta,
            "discordant_won": c, "discordant_lost": b, "mcnemar_exact_p": p,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out_csv}")

    # The CSV only holds comparisons against the reference, so a question like
    # "are the two 5000-instance models actually tied?" would need the whole
    # job re-run. Save the raw solved/not matrix instead: every pair can then
    # be tested afterwards from a file that costs nothing to keep.
    npz_path = os.path.splitext(args.out_csv)[0] + "_per_instance.npz"
    np.savez_compressed(
        npz_path,
        solved=np.stack([r["solved_per_instance"] for r in results]),
        names=np.array(names, dtype=object),
        seed_start=args.seed_start,
    )
    print(f"wrote the {len(results)}x{args.n_instances} solved matrix to {npz_path}")
    print(json.dumps({"shared_instance_signature": sig}, default=str))


if __name__ == "__main__":
    main()
