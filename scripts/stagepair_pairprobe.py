"""Did the satisfied margin learn the PREDICATE, or an edit detector?

This is stage PAIR's registered risk P2, and finding 80b is what unblocked it:
there is now a model whose margin is satisfied, and nothing yet distinguishes the
two things that could have satisfied it.

    a determinacy representation      g_clean - g_placebo ~= 0
    an edit detector                  g_clean - g_placebo large, like corrupt

The corrupted prompt differs from the clean one by ONE token in ~400, and the
token it changes is a hint that removes v's forcing. A function that fires on
"a token changed near v" satisfies L_pair perfectly and is worth nothing. The
PLACEBO is the discriminator and it is free: it recolours a hint that is NOT a
neighbour of v, so v stays forced. It is never trained on and appears only here.

NOT A STAGE. One seed, not three; it reads the registered bars but cannot score
the arm, and `check_p1` will say so itself rather than my saying it for it.

THE CONVERGENCE RULE AND THIS ARM COLLIDE, and that is worth knowing before the
numbers. `stagepair_score.converged` admits a seed at `final_train_nll <= 1.600 +
0.030`; 80b's lambda=1 arm finished at 1.6688. STUCK_TOL was registered to
exclude a seed stuck at uniform (3.95), not to exclude one paying a 0.06 tax for
an auxiliary term it satisfied. So this arm is NOT interpretable under the rule
as written. The numbers are still measured and printed -- deciding whether the
rule or the arm is wrong changes the registered stage and is hshunt's call.

WHY THREE MODELS AND NOT ONE. Measured on one case list at one eval stream:

  Y         stage Y seed 0, untouched -- finding 78's model, the 0.49 to beat
  warm L0   3,000 steps from Y at lambda=0 on the same fresh data
  warm L1   3,000 steps from Y at lambda=1, the satisfied margin

Without `warm L0`, any movement in `warm L1` could be the 30,000 extra instances
rather than the margin. Finding 78's published numbers are NOT a usable control
here: they were measured at seed_start 600,000,000 on 350 cases, and this probe
runs the registered PAIR_EVAL_SEED_START on 1,000. Comparing across those would
be comparing two case lists, so stage Y is re-measured here instead.

THE GUARD RATE IS CONDITIONAL. Its denominator is the number of cases where the
clean prompt emits v at all -- stage Y's were 236/228/217 of 350, pooling to the
681 behind "fires once in 681". It is counted from the rows here, never assumed;
`check_p4` raises if it ever comes out as the case count.

COST. Two arms x 3,000 steps is the price of the checkpoint 52441962 deliberately
did not keep; both are written to disk this time so the next probe is eval-only.
"""

import json
import os
import shutil
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/spin_gnn/scripts")

import stagepair_score as SP
from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sat_dataset2 import (SATTokenDataset, iterate_token_batches,
                                   make_token_batch_pair)
from spin_gnn.sat_transformer2 import SATReasoningTransformer, sat2_nll_loss
from spin_gnn.sudoku_cfresp import (run_cfresp, run_stratified,
                                    summarize_cfresp)
from spin_gnn.sudoku_determinacy import (determinacy_loss_from_channel,
                                         determinacy_targets)
from spin_gnn.sudoku_pair import build_pair_index
from spin_gnn.sudoku_skip import find_unforceable_deductions
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab

CKPT_DIR = ("/scratch/users/hshunt/spin_gnn/sudoku_n48_determinacy_Y_2026-11-14b/"
            "t00_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s0/"
            "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_chan")
OUT_DIR = "/scratch/users/hshunt/spin_gnn/runs/stagepair_pairprobe_2026-11-15"

# Training: the same numbers 52441962 ran, so arm L1 reproduces its trajectory.
N_INSTANCES = 30_000
BATCH = 32
STEPS = 3_000
LOG_EVERY = 500
SEED = 0
DATA_SEED = SEED + 777
MARGIN = SP.MARGIN
MAX_LEN = 512
LR = 1e-3
# What 52441962 finished at, for the reproduction check. GPU reductions are not
# bit-reproducible across runs, so this is printed and compared loosely, never
# asserted -- a hard assert here would fail on noise and block the eval.
REPRO = {"warm L0": 1.5865, "warm L1": 1.6688}

# Evaluation: imported, never retyped. These are the registered streams.
EVAL_SEED_START = SP.PAIR_EVAL_SEED_START
EVAL_N_CASES = SP.PAIR_EVAL_N_CASES
AUROC_SEED_START = SP.AUROC_SEED_START
# 1,000 cases need more than the 400 instances stage Z drew for 350 of them.
EVAL_N_INSTANCES = 1_400
AUROC_N_INSTANCES = 600

dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"device={dev}  eval seed_start={EVAL_SEED_START:,}  "
      f"cases={EVAL_N_CASES:,}  steps={STEPS:,}/arm", flush=True)

with open(os.path.join(CKPT_DIR, "config.json")) as fh:
    config = json.load(fh)
assert config.get("determinacy_channel"), "stage Y checkpoint has no channel"
assert config["max_len"] == MAX_LEN, config["max_len"]

sudoku_cfg = SudokuReasoningConfig(n_nodes=48, degree=5, n_colors=4,
                                   shuffle_adj=True, fall=True, seed=DATA_SEED)
vocab = build_vocab(sudoku_cfg)
state = torch.load(os.path.join(CKPT_DIR, "model.pt"), map_location="cpu",
                   weights_only=True)


def fresh_from_Y():
    m = SATReasoningTransformer(
        vocab.size, d_model=320, num_layers=16, nhead=4, max_len=MAX_LEN,
        dim_feedforward=1024, norm_first=True, arch_mode="tied_no_mlp",
        n_blocks=1, determinacy_channel=True,
        node_offset=vocab.node_offset, n_nodes=vocab.n_nodes)
    m.load_state_dict(state, strict=True)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == SP.N_TRAINABLE, ("not the registered cell: %s trainable"
                                 % f"{n:,}")
    m = m.to(dev)
    s = float(m.det_scale.norm())
    assert s > 0.05, ("det_scale is %.3e -- dead channel, probe is vacuous" % s)
    return m


# ---------------------------------------------------------------------------
# the two warm arms
# ---------------------------------------------------------------------------
print("\nbuilding the 30,000 fine-tuning instances (data seed "
      f"{DATA_SEED})", flush=True)
inst = build_instances(sudoku_cfg, N_INSTANCES, sudoku_cfg.seed)
tokens = [encode_instance(adj, trace, vocab) for adj, _, _, trace in inst]
aux = [determinacy_targets(adj, trace, vocab, tok)
       for (adj, _c, _h, trace), tok in zip(inst, tokens)]
pair = build_pair_index(inst, tokens, vocab, sudoku_cfg.seed + 500_000_000)
ds = SATTokenDataset(tokens, max_len=MAX_LEN, aux=aux, pair=pair)
idx = list(range(len(ds)))
print(f"dataset={len(ds):,}", flush=True)


def train_arm(pair_w, label):
    """3,000 steps from stage Y. Returns (model, final_main, last_hinge)."""
    print(f"\n=== {label}: lambda={pair_w:g} " + "=" * 34, flush=True)
    torch.manual_seed(SEED)
    model = fresh_from_Y()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=LR)
    rng = np.random.default_rng(SEED)
    print(f"{'step':>5} {'main':>7} {'hinge':>7} {'gap':>8} {'sat':>5} {'|s|':>8}",
          flush=True)
    step, main_v, hinge_v, gap_v, sat_v = 0, float("nan"), float("nan"), \
        float("nan"), float("nan")
    tail = []
    while step < STEPS:
        for batch in iterate_token_batches(ds, idx, BATCH, rng=rng,
                                           collate=make_token_batch_pair):
            if step >= STEPS:
                break
            x, y, mask = (t.to(dev) for t in batch[:3])
            pr = batch[5].to(dev)
            logits, g = model(x, valid_mask=mask, return_determinacy=True)
            main = sat2_nll_loss(logits, y, mask)
            loss = main
            det = determinacy_loss_from_channel(g, batch[3].to(dev),
                                                batch[4].to(dev))
            if det is not None:
                loss = loss + det
            if pair_w > 0:
                sel = (pr[:, 0] >= 0).nonzero(as_tuple=True)[0]
                if sel.numel():
                    xb = x.clone()
                    xb[sel, pr[sel, 0]] = pr[sel, 1]
                    _zb, gb = model(xb, valid_mask=mask,
                                    return_determinacy=True)
                    gap = (g[sel, pr[sel, 2], pr[sel, 3]]
                           - gb[sel, pr[sel, 2], pr[sel, 3]])
                    hinge = torch.clamp(MARGIN - gap, min=0)
                    loss = loss + pair_w * hinge.mean()
                    hinge_v, gap_v = float(hinge.mean()), float(gap.mean())
                    sat_v = float((gap >= MARGIN).float().mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            main_v = float(main)
            # The last 250 steps, because one batch of 32 is a noisy estimate of
            # train nll and the convergence rule is read against a threshold
            # 0.03 wide. Both are reported.
            if step >= STEPS - 250:
                tail.append(main_v)
            if step % LOG_EVERY == 0:
                print(f"{step:5d} {main_v:7.4f} {hinge_v:7.4f} {gap_v:+8.4f} "
                      f"{sat_v:5.2f} {float(model.det_scale.norm()):8.4f}",
                      flush=True)
            step += 1
    mean_tail = float(np.mean(tail))
    print(f"{'end':>5} {main_v:7.4f} {hinge_v:7.4f} {gap_v:+8.4f} {sat_v:5.2f} "
          f"{float(model.det_scale.norm()):8.4f}   "
          f"(mean of last 250 steps: {mean_tail:.4f}; 52441962 ended at "
          f"{REPRO[label]:.4f})", flush=True)
    d = os.path.join(OUT_DIR, label.replace(" ", "_"))
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, "model.pt"))
    shutil.copy(os.path.join(CKPT_DIR, "config.json"),
                os.path.join(d, "config.json"))
    print(f"  checkpoint kept: {d}", flush=True)
    return model, main_v, mean_tail, hinge_v, sat_v


models = {}
train = {}
for w, label in [(0.0, "warm L0"), (1.0, "warm L1")]:
    m, fin, tail_mean, hv, sv = train_arm(w, label)
    m.eval()
    models[label] = m
    train[label] = {"final_train_nll": fin, "mean_train_nll_last250": tail_mean,
                    "final_hinge": hv, "final_sat": sv, "lambda": w}
models["Y"] = fresh_from_Y()
models["Y"].eval()
# Stage Y's own published final train nll, not re-measured here: the arm it is a
# control for is the one being scored, and Y's number is already registered.
train["Y"] = {"final_train_nll": 1.5610, "mean_train_nll_last250": 1.5610,
              "final_hinge": float("nan"), "final_sat": float("nan"),
              "lambda": None}

# ---------------------------------------------------------------------------
# one case list, all three models
# ---------------------------------------------------------------------------
print(f"\nbuilding the eval case list at seed_start {EVAL_SEED_START:,}",
      flush=True)
rng = np.random.default_rng(0)
cases = []
for adj, colors, hints, trace in build_instances(sudoku_cfg, EVAL_N_INSTANCES,
                                                 EVAL_SEED_START):
    for case in find_unforceable_deductions(adj, colors, hints, trace, vocab,
                                            rng):
        cases.append((adj, colors, hints, trace, case))
print(f"{len(cases):,} breakable round-0 deductions from "
      f"{EVAL_N_INSTANCES:,} instances", flush=True)
assert len(cases) >= EVAL_N_CASES, (
    "only %d cases; raise EVAL_N_INSTANCES rather than lowering the "
    "registered PAIR_EVAL_N_CASES" % len(cases))
cases = cases[:EVAL_N_CASES]

det_instances = list(build_instances(sudoku_cfg, AUROC_N_INSTANCES,
                                     AUROC_SEED_START))

per_arm = {}
for label in ("Y", "warm L0", "warm L1"):
    print(f"\n--- probing {label} " + "-" * 40, flush=True)
    rows = run_cfresp(models[label], vocab, dev, cases)
    s = summarize_cfresp(rows)
    st = run_stratified(models[label], vocab, dev, det_instances)

    # THE DENOMINATOR, counted and not assumed: cases where the clean prompt
    # emits v at all. The guard has fired when the corrupted prompt no longer
    # emits v, which is the behaviour stage Y produced once in 681.
    clean = [r for r in rows if r["clean_emits_v"]]
    guard_denom = len(clean)
    guard_rate = (float(np.mean([not r["corrupt_emits_v"] for r in clean]))
                  if clean else float("nan"))

    p = dict(train[label])
    p.update({
        "arm": label,
        "pair_auroc_corrupt": SP.matched_pair_auroc(
            [r["dg_corrupt"] for r in rows]),
        "pair_auroc_placebo": SP.matched_pair_auroc(
            [r["dg_placebo"] for r in rows]),
        "pooled_auroc": st["model_pooled"],
        "within_stratum_auroc": st["model"],
        "guard_rate": guard_rate,
        "guard_denom": guard_denom,
        "n_cases": len(rows),
        "median_abs_dg_corrupt": s["median_abs_dg_corrupt"],
        "median_abs_dg_placebo": s["median_abs_dg_placebo"],
        "median_g_spread": s["median_g_spread"],
        "win_rate": s["win_rate"],
        "clean_emits_v": s["clean_emits_v"],
        "corrupt_emits_v": s["corrupt_emits_v"],
        "placebo_emits_v": s["placebo_emits_v"],
    })
    per_arm[label] = p
    print(f"  pair AUROC corrupt {p['pair_auroc_corrupt']:.4f}   "
          f"placebo {p['pair_auroc_placebo']:.4f}   "
          f"(chance 0.50; stage Z 0.4457/0.4800)")
    print(f"  pooled AUROC {p['pooled_auroc']:.4f}  within-stratum "
          f"{p['within_stratum_auroc']:.4f}  (Y published 0.9493 / 0.8059)")
    print(f"  guard {guard_rate:.4f} of {guard_denom} cases where the clean "
          f"prompt emits v (NOT of {len(rows)})")
    print(f"  median |dg| corrupt {p['median_abs_dg_corrupt']:.4f} vs placebo "
          f"{p['median_abs_dg_placebo']:.4f}, g spread "
          f"{p['median_g_spread']:.3f}", flush=True)

# ---------------------------------------------------------------------------
# the registered bars, read by the registered code
# ---------------------------------------------------------------------------
with open(os.path.join(OUT_DIR, "pair_summary.json"), "w") as fh:
    json.dump({"per_arm": per_arm,
               "eval": {"seed_start": EVAL_SEED_START, "n_cases": EVAL_N_CASES,
                        "n_instances": EVAL_N_INSTANCES,
                        "auroc_seed_start": AUROC_SEED_START,
                        "auroc_n_instances": AUROC_N_INSTANCES},
               "train": {"steps": STEPS, "data_seed": DATA_SEED, "lr": LR,
                         "margin": MARGIN, "seed": SEED}},
              fh, indent=2)
print(f"\nwrote {OUT_DIR}/pair_summary.json")

print("\n" + "=" * 72)
print("the registered checks, on the lambda=1 arm alone. One seed, so P1 is")
print("EXPECTED to refuse -- MIN_SEEDS is 2 and that is the stage's rule, not")
print("a failure of the arm.")
only = [per_arm["warm L1"]]
ok = SP.converged(only)
print(f"  converged (<= {SP.FLOOR + SP.STUCK_TOL:.3f}): {ok} at "
      f"final_train_nll {only[0]['final_train_nll']:.4f}")
for name, fn in (("P1 pair", SP.check_p1), ("P2 placebo", SP.check_p2),
                 ("P3 pooled", SP.check_p3), ("P4 guard", SP.check_p4)):
    print(f"  {name:<11} {json.dumps(fn(only, ok), sort_keys=True)}")

print("\n" + "=" * 72)
print(f"{'arm':<9} {'corrupt':>8} {'placebo':>8} {'pooled':>7} {'guard':>7} "
      f"{'denom':>6} {'train nll':>10}")
for label in ("Y", "warm L0", "warm L1"):
    p = per_arm[label]
    print(f"{label:<9} {p['pair_auroc_corrupt']:8.4f} "
          f"{p['pair_auroc_placebo']:8.4f} {p['pooled_auroc']:7.4f} "
          f"{p['guard_rate']:7.4f} {p['guard_denom']:6d} "
          f"{p['final_train_nll']:10.4f}")
print(f"\nBars, for reading the table and not for me to apply: P1 corrupt >= "
      f"{SP.PAIR_BAR}, P2 placebo in {list(SP.PLACEBO_BAND)},")
print(f"P3 pooled >= {SP.POOLED_BAR}, P4 guard >= {SP.GUARD_BAR}. The placebo "
      f"is what separates a predicate")
print("from an edit detector: an edit detector scores HIGH on both columns.")
print("No verdict is printed.")
