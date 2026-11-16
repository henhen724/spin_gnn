"""Does the predicate appear BEFORE the churn does?

THE QUESTION, AND WHY IT NEEDS NO DECISION FROM ANYONE. Findings 82-83 leave one
thing conspicuously open: the lambda=1 arm bought its predicate (matched-pair
AUROC 0.8710 / 0.8620 / 0.8710 against placebo 0.4360 / 0.4450 / 0.4420) while
one-token edits came to flip whether v is emitted in 142 / 175 / 177 cases of
1,000, against stage Y's 4. Nothing measured so far says whether that churn is
NECESSARY or merely concurrent. Every other open item on stage PAIR changes a
registered quantity -- the P4 denominator reading, the 1.630 convergence rule,
the form of `L_pair` -- and is therefore not mine to settle. This one changes
nothing: it re-runs an arm that already ran, with the same recipe, and looks at
it more often.

THE INSTRUMENT. One lambda=1 warm arm, the identical recipe `stagepair_pairprobe`
ran (3,000 steps from that seed's stage Y body, batch 32, lr 1e-3, lambda 1.0,
margin 1.0, 30,000 instances, `DATA_SEED = TRAJ_SEED + 777` -- copied from the
probe, so seed 0 is 52452867's arm and seeds 1-2 are 52456376's, each on the same
data stream its published row was measured on). `TRAJ_SEED` selects the body:
seed 0 is finding 84, seeds 1 and 2 are the replication registered below. State
dicts are snapshotted at 0, 125, 250, 500, 750,
1,000, 1,500, 2,000, 2,500 and 3,000 steps, kept on the CPU, and evaluated AFTER
training finishes -- so the training trajectory is a plain run and not one
perturbed by ten eval-mode interleavings. Every snapshot then meets the SAME
registered 1,000-case list at `PAIR_EVAL_SEED_START`, so the columns are
comparable down the page.

THE REGISTERED HYPOTHESIS, WRITTEN BEFORE THE NUMBERS EXIST. The hinge pushes on
the SIGNED difference g_clean - g_broken, while the churn is a change in the
LEVEL of g, so I predict they separate in time and the predicate comes first:

  * corrupt AUROC first crosses the 0.65 bar between step 250 and step 750;
  * at that snapshot, total flips are 20-70, i.e. at most half the final 142;
  * the placebo stays inside (0.40, 0.60) at EVERY snapshot, step 0 included --
    if it ever leaves the band upward, an edit detector formed first and the
    whole trajectory reading is about something else;
  * the raw guard rate rises monotonically with the flip count, because on this
    reading they are two views of the same loudness.

THE READING THAT FALSIFIES THE SLACK CLAIM: flips already above 100 at the first
snapshot where corrupt clears 0.65. That would say the churn is paid up front and
no amount of stopping early avoids it. A second falsifier is a crossing that
never happens inside 3,000 steps even though 52452867 ended at 0.8710, which
would mean this rerun did not reproduce and nothing here is readable.

WHAT A HIT WOULD AND WOULD NOT LICENSE. It would NOT license stopping the arm
early: training length is part of the registered stage and changing it is
hshunt's call, the same as the other three. It would license saying that the
churn and the predicate are separable in time on this body, which is the first
evidence either way. And a snapshot can show fewer flips simply by having a
quieter channel overall, so the flip count is never read alone -- the placebo
AUROC, the McNemar excess and ||det_scale|| are printed beside it.

REPRODUCTION ANCHORS, printed and compared loosely, never asserted. GPU
reductions are not bit-reproducible across runs, so step 3,000 is expected near
but not at the published row, and step 0 IS the stage Y body for that seed and
should land on finding 82's Y row almost exactly (it is the same weights meeting
the same cases). All three seeds' anchors are in `ANCHOR_BY_SEED`, read out of
the `pairprobe` and `guardspec` artifacts rather than retyped from the deck.

HOW CLOSE "NEAR" IS, MEASURED RATHER THAN GUESSED (finding 84, seed 0). The same
code, recipe and cases reproduced the matched-pair AUROCs to 0.0010 but moved the
raw guard 0.1087 -> 0.1399, its denominator 506 -> 529 and the flips 142 -> 160.
Emission is an ARGMAX, so a logit shift far below ranking resolution still flips
a discrete decision. Expect the step-3,000 AUROCs on the anchor and the
step-3,000 guard up to ~0.03 off it; step 0 should agree exactly, because no
training intervenes.

SEEDS 1 AND 2: THE SECOND REGISTERED HYPOTHESIS, WRITTEN 2026-11-15 BEFORE EITHER
RUN EXISTS. Finding 84 is one body. Its headline -- the predicate is in place at
step 125 of 3,000 at 29 flips of the final 160, while the raw guard there is
0.0140 and so FAILS P4's 0.10 bar -- is exactly the shape of claim that finding
83 got wrong by reading three bodies off one. The two remaining bodies end with
MORE churn than seed 0 (175 and 177 flips against 142) and seed 1's stage Y body
starts closer to the bar than the others (matched-pair corrupt 0.5870 against
0.4560 and 0.4640), which is why seed 1 is the interesting one:

  * both seeds' corrupt AUROC first clears 0.65 at step 125 or step 250 -- i.e.
    inside the first 8% of training, as seed 0 did, not in 84's originally
    predicted 250-750 window, which that run already missed low;
  * at that crossing the flips are at most 35% of that seed's own PUBLISHED final
    count, i.e. <= 61 on both (0.35 x 175 and 0.35 x 177, floored; seed 0 came in
    at 29 against a published 142 and a rerun 160, i.e. 18%);
  * the raw guard at that same crossing snapshot is BELOW 0.10 on both seeds, so
    the "passes P1/P2, fails P4" split is a property of the stage and not of
    seed 0 -- this is the load-bearing prediction and the reason to run this;
  * the placebo stays inside (0.40, 0.60) at all ten snapshots on both seeds;
  * raw guard and flips rank together at Spearman >= +0.7 on both seeds, as they
    did at +0.879 on seed 0 -- registered as a rank statistic and not as
    monotonicity, because 84 predicted monotone and got a non-monotone +0.879.

WHAT FALSIFIES IT. A seed whose raw guard is already at or above 0.10 at the
snapshot where corrupt first clears 0.65. That would say the predicate and the
guard arrive together on that body, and 84's separation is seed 0's alone rather
than the stage's. A placebo above 0.60 at any trained snapshot would instead say
an edit detector formed first on that body, which would put P2 back in play.

STILL NOT A LICENCE TO STOP EARLY, even on 3/3. Training length is part of the
registered stage and changing it is hshunt's call, the same as the P4
denominator, the 1.630 convergence rule and the form of `L_pair`.

NO VERDICT IS PRINTED.
"""

import json
import os
import sys
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/spin_gnn/scripts")

import stagepair_score as SP
from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sat_dataset2 import (SATTokenDataset, iterate_token_batches,
                                   make_token_batch_pair)
from spin_gnn.sat_transformer2 import SATReasoningTransformer, sat2_nll_loss
from spin_gnn.sudoku_cfresp import run_cfresp
from spin_gnn.sudoku_determinacy import (determinacy_loss_from_channel,
                                         determinacy_targets)
from spin_gnn.sudoku_pair import build_pair_index
from spin_gnn.sudoku_skip import find_unforceable_deductions
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab

_Y = "/scratch/users/hshunt/spin_gnn/sudoku_n48_determinacy_Y_2026-11-14b"
_CELL = "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_chan"
Y_DIRS = {
    0: f"{_Y}/t00_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s0/{_CELL}",
    1: f"{_Y}/t01_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s1/{_CELL}_s01",
    2: f"{_Y}/t02_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s2/{_CELL}_s02",
}

# The recipe, copied from `stagepair_pairprobe.py` so the arm is the same arm.
N_INSTANCES = 30_000
BATCH = 32
STEPS = 3_000
LOG_EVERY = 500
MAX_LEN = 512
LR = 1e-3
MARGIN = SP.MARGIN
SEED = int(os.environ.get("TRAJ_SEED", "0"))
assert SEED in Y_DIRS, "no stage Y checkpoint for seed %d" % SEED
CKPT_DIR = Y_DIRS[SEED]
OUT_DIR = ("/scratch/users/hshunt/spin_gnn/runs/stagepair_trajectory_2026-11-15"
           + ("" if SEED == 0 else f"_s{SEED}"))
DATA_SEED = SEED + 777

# Step 0 is stage Y itself, which is why it is in the list: it anchors the flip
# count at the value finding 82 measured (4 of 1,000) rather than at an assumed
# zero. The grid is dense early because the hypothesis says the crossing is early.
SNAP_STEPS = (0, 125, 250, 500, 750, 1_000, 1_500, 2_000, 2_500, 3_000)

# Registered eval streams, imported and never retyped.
EVAL_SEED_START = SP.PAIR_EVAL_SEED_START
EVAL_N_CASES = SP.PAIR_EVAL_N_CASES
EVAL_N_INSTANCES = 1_400

# A SMOKE RUN IS NOT A MEASUREMENT. `TRAJ_SMOKE=1` shrinks every size so the code
# path runs end to end in ~2 minutes and a crash costs that instead of a GPU hour.
# It changes the data, the step count and the case count, so NOTHING it prints is
# a number about the model; it exists only to find the `KeyError` before the real
# run does. The banner says so in the log, and it writes to its own directory.
SMOKE = os.environ.get("TRAJ_SMOKE") == "1"
if SMOKE:
    print("=" * 72)
    print("TRAJ_SMOKE=1 -- SHAPES ONLY. Every size below is wrong on purpose "
          "and no number\nprinted by this run is a measurement of anything.")
    print("=" * 72, flush=True)
    N_INSTANCES, STEPS, LOG_EVERY = 400, 20, 10
    SNAP_STEPS = (0, 10, 20)
    EVAL_N_CASES, EVAL_N_INSTANCES = 40, 80
    OUT_DIR += "_smoke"

# The published rows this trajectory should pass through, for the loose
# reproduction check: step 0 is that seed's stage Y body and step 3,000 is its
# `warm L1` arm. Read out of `runs/stagepair_pairprobe_2026-11-15*/` and
# `runs/stagepair_guardspec_2026-11-15*/rows_warm_L1.jsonl` on 2026-11-15 --
# `runs/` is gitignored and $SCRATCH purges, so these constants are the durable
# copy. `flips` uses `make_predicate_web.mcnemar`'s definition: clean != corrupt
# OR clean != placebo. NEVER asserted; the guard column in particular carries the
# ~0.03 of argmax noise finding 84 measured.
ANCHOR_BY_SEED = {
    0: {0: {"corrupt": 0.4560, "placebo": 0.5140, "guard": 0.0015,
            "denom": 656, "flips": 4},
        3_000: {"corrupt": 0.8710, "placebo": 0.4360, "guard": 0.1087,
                "denom": 506, "flips": 142}},
    1: {0: {"corrupt": 0.5870, "placebo": 0.4730, "guard": 0.0000,
            "denom": 637, "flips": 1},
        3_000: {"corrupt": 0.8620, "placebo": 0.4450, "guard": 0.2072,
                "denom": 502, "flips": 175}},
    2: {0: {"corrupt": 0.4640, "placebo": 0.5060, "guard": 0.0017,
            "denom": 584, "flips": 7},
        3_000: {"corrupt": 0.8710, "placebo": 0.4420, "guard": 0.1632,
                "denom": 429, "flips": 177}},
}
ANCHOR = ANCHOR_BY_SEED[SEED]

# THE REGISTERED PREDICTIONS, one entry per seed, written into this file before
# the corresponding run and printed back against the table at the end. Seed 0's
# is finding 84's, left EXACTLY as it was registered even though it missed low
# (crossing 125, not 250-750) -- rewriting it after the fact is the one thing
# this block exists to prevent. Seeds 1-2 are the replication registered in the
# docstring; `flip_cap` is derived from that seed's published final count so the
# bound is 35% of the right denominator, and `guard_falsifier` is the
# load-bearing one: a guard already at or above P4's bar when the predicate
# arrives would mean 84's separation is seed 0's alone.
FLIP_SHARE_CAP = 0.35
PREDICTION = {
    0: {"cross": (250, 750), "flip_cap": 70, "flip_floor": 20},
    1: {"cross": (125, 250), "flip_cap": int(FLIP_SHARE_CAP * 175),
        "flip_floor": 0},
    2: {"cross": (125, 250), "flip_cap": int(FLIP_SHARE_CAP * 177),
        "flip_floor": 0},
}[SEED]
PREDICTION["guard_falsifier"] = SP.GUARD_BAR   # 0.10, imported, never retyped

dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"device={dev}  SEED={SEED}  steps={STEPS:,}  snapshots={len(SNAP_STEPS)}",
      flush=True)
print(f"eval seed_start={EVAL_SEED_START:,}  cases={EVAL_N_CASES:,}", flush=True)
print(f"stage Y body: {CKPT_DIR}\nout: {OUT_DIR}", flush=True)

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
    return m.to(dev)


# ---------------------------------------------------------------------------
# train once, snapshotting
# ---------------------------------------------------------------------------
print(f"\nbuilding the {N_INSTANCES:,} fine-tuning instances (data seed "
      f"{DATA_SEED})", flush=True)
inst = build_instances(sudoku_cfg, N_INSTANCES, sudoku_cfg.seed)
tokens = [encode_instance(adj, trace, vocab) for adj, _, _, trace in inst]
aux = [determinacy_targets(adj, trace, vocab, tok)
       for (adj, _c, _h, trace), tok in zip(inst, tokens)]
pair = build_pair_index(inst, tokens, vocab, sudoku_cfg.seed + 500_000_000)
ds = SATTokenDataset(tokens, max_len=MAX_LEN, aux=aux, pair=pair)
idx = list(range(len(ds)))
print(f"dataset={len(ds):,}", flush=True)

torch.manual_seed(SEED)
model = fresh_from_Y()
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                       lr=LR)
rng = np.random.default_rng(SEED)

snaps = {}
hist = {}


def snapshot(step, main_v, hinge_v, gap_v, sat_v, window):
    snaps[step] = {k: v.detach().cpu().clone()
                   for k, v in model.state_dict().items()}
    hist[step] = {"step": step, "train_nll": main_v, "hinge": hinge_v,
                  "gap": gap_v, "sat": sat_v,
                  # the mean over the preceding window, because one batch of 32
                  # is a noisy read and the whole convergence argument turns on
                  # differences of 0.03
                  "train_nll_mean": (float(np.mean(window[-250:]))
                                     if window else main_v),
                  "det_scale_norm": float(model.det_scale.norm())}


print(f"\n{'step':>5} {'main':>7} {'hinge':>7} {'gap':>8} {'sat':>5} {'|s|':>8}",
      flush=True)
step = 0
main_v = float("nan")
hinge_v = gap_v = sat_v = float("nan")
window = []
snapshot(0, main_v, hinge_v, gap_v, sat_v, window)
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
        sel = (pr[:, 0] >= 0).nonzero(as_tuple=True)[0]
        if sel.numel():
            xb = x.clone()
            xb[sel, pr[sel, 0]] = pr[sel, 1]
            _zb, gb = model(xb, valid_mask=mask, return_determinacy=True)
            gap = (g[sel, pr[sel, 2], pr[sel, 3]]
                   - gb[sel, pr[sel, 2], pr[sel, 3]])
            hinge = torch.clamp(MARGIN - gap, min=0)
            loss = loss + hinge.mean()
            hinge_v, gap_v = float(hinge.mean()), float(gap.mean())
            sat_v = float((gap >= MARGIN).float().mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        main_v = float(main)
        window.append(main_v)
        step += 1
        if step % LOG_EVERY == 0:
            print(f"{step:5d} {main_v:7.4f} {hinge_v:7.4f} {gap_v:+8.4f} "
                  f"{sat_v:5.2f} {float(model.det_scale.norm()):8.4f}",
                  flush=True)
        if step in SNAP_STEPS:
            snapshot(step, main_v, hinge_v, gap_v, sat_v, window)

assert set(snaps) == set(SNAP_STEPS), sorted(set(SNAP_STEPS) - set(snaps))
print(f"\n{len(snaps)} snapshots held in memory", flush=True)

# ---------------------------------------------------------------------------
# one case list, every snapshot
# ---------------------------------------------------------------------------
print(f"\nbuilding the eval case list at seed_start {EVAL_SEED_START:,}",
      flush=True)
crng = np.random.default_rng(0)
cases = []
for adj, colors, hints, trace in build_instances(sudoku_cfg, EVAL_N_INSTANCES,
                                                 EVAL_SEED_START):
    for case in find_unforceable_deductions(adj, colors, hints, trace, vocab,
                                            crng):
        cases.append((adj, colors, hints, trace, case))
print(f"{len(cases):,} breakable round-0 deductions from "
      f"{EVAL_N_INSTANCES:,} instances", flush=True)
assert len(cases) >= EVAL_N_CASES, (
    "only %d cases; raise EVAL_N_INSTANCES rather than lowering the registered "
    "PAIR_EVAL_N_CASES" % len(cases))
cases = cases[:EVAL_N_CASES]

probe = fresh_from_Y()
rows_out = []
for st in SNAP_STEPS:
    probe.load_state_dict(snaps[st], strict=True)
    probe.eval()
    rows = run_cfresp(probe, vocab, dev, cases)

    # THE DENOMINATOR, counted and not assumed -- `check_p4` raises on 350 and
    # the three lambda=1 arms came out at 506 / 502 / 429, so it moves with the
    # model and here it moves with the STEP.
    clean = [r for r in rows if r["clean_emits_v"]]
    denom = len(clean)
    guard = (float(np.mean([not r["corrupt_emits_v"] for r in clean]))
             if clean else float("nan"))
    # the control P4 never had: the placebo leaves v forced, so a turn-off there
    # is a withdrawal the predicate does not license
    placebo_guard = (float(np.mean([not r["placebo_emits_v"] for r in clean]))
                     if clean else float("nan"))
    b = sum(1 for r in clean
            if not r["corrupt_emits_v"] and r["placebo_emits_v"])
    c = sum(1 for r in clean
            if r["corrupt_emits_v"] and not r["placebo_emits_v"])
    excess = (b - c) / denom if denom else float("nan")
    se = np.sqrt(b + c) / denom if denom else float("nan")
    # same definition as `make_predicate_web.mcnemar`: EITHER edit changing
    # whether v is emitted, which is the 142 / 175 / 177 column
    flips = sum(1 for r in rows
                if r["clean_emits_v"] != r["corrupt_emits_v"]
                or r["clean_emits_v"] != r["placebo_emits_v"])

    rec = dict(hist[st])
    rec.update({
        "pair_auroc_corrupt": SP.matched_pair_auroc(
            [r["dg_corrupt"] for r in rows]),
        "pair_auroc_placebo": SP.matched_pair_auroc(
            [r["dg_placebo"] for r in rows]),
        "guard_rate": guard, "guard_denom": denom,
        "placebo_guard_rate": placebo_guard,
        "excess": excess, "excess_lo": excess - 1.96 * se,
        "excess_hi": excess + 1.96 * se,
        "corrupt_only": b, "placebo_only": c,
        "p": stats.binomtest(b, b + c, 0.5).pvalue if b + c else float("nan"),
        "total_flips": flips, "n_cases": len(rows),
    })
    rows_out.append(rec)
    print(f"  step {st:5d}: corrupt {rec['pair_auroc_corrupt']:.4f}  placebo "
          f"{rec['pair_auroc_placebo']:.4f}  guard {guard:.4f} of {denom}  "
          f"flips {flips:3d}  |s| {rec['det_scale_norm']:.3f}", flush=True)

with open(os.path.join(OUT_DIR, "trajectory_summary.json"), "w") as fh:
    json.dump({"per_step": rows_out,
               "snap_steps": list(SNAP_STEPS),
               "train": {"steps": STEPS, "data_seed": DATA_SEED, "lr": LR,
                         "margin": MARGIN, "seed": SEED, "lambda": 1.0,
                         "n_instances": N_INSTANCES, "batch": BATCH},
               "eval": {"seed_start": EVAL_SEED_START, "n_cases": EVAL_N_CASES,
                        "n_instances": EVAL_N_INSTANCES},
               "anchors": {str(k): v for k, v in ANCHOR.items()}},
              fh, indent=2)
print(f"\nwrote {OUT_DIR}/trajectory_summary.json")

# ---------------------------------------------------------------------------
# the table, and the hypothesis read against it
# ---------------------------------------------------------------------------
print("\n" + "=" * 86)
print(f"{'step':>5} {'corrupt':>8} {'placebo':>8} {'guard':>7} {'denom':>6} "
      f"{'flips':>6} {'excess':>8} {'|s|':>7} {'nll(250)':>9}")
for r in rows_out:
    print(f"{r['step']:5d} {r['pair_auroc_corrupt']:8.4f} "
          f"{r['pair_auroc_placebo']:8.4f} {r['guard_rate']:7.4f} "
          f"{r['guard_denom']:6d} {r['total_flips']:6d} {r['excess']:+8.4f} "
          f"{r['det_scale_norm']:7.3f} {r['train_nll_mean']:9.4f}")

print("\nreproduction, printed and compared loosely (GPU reductions are not "
      "bit-reproducible):")
for st, a in ANCHOR.items():
    # a smoke run has no step 3,000, so the anchor is skipped rather than raising
    r = next((x for x in rows_out if x["step"] == st), None)
    if r is None:
        print(f"  step {st:5d}  not in this run's snapshots, anchor skipped")
        continue
    print(f"  step {st:5d}  corrupt {r['pair_auroc_corrupt']:.4f} vs "
          f"{a['corrupt']:.4f}   placebo {r['pair_auroc_placebo']:.4f} vs "
          f"{a['placebo']:.4f}   guard {r['guard_rate']:.4f} of "
          f"{r['guard_denom']} vs {a['guard']:.4f} of {a['denom']}   flips "
          f"{r['total_flips']} vs {a['flips']}")

cross = [r for r in rows_out if r["pair_auroc_corrupt"] >= SP.PAIR_BAR]
clo, chi = PREDICTION["cross"]
print(f"\nthe hypothesis, read against the table (bars imported: P1 "
      f">= {SP.PAIR_BAR}, P2 in {list(SP.PLACEBO_BAND)}, P4 "
      f">= {SP.GUARD_BAR}):")
if cross:
    f = cross[0]
    print(f"  corrupt first clears {SP.PAIR_BAR} at step {f['step']} "
          f"(predicted {clo}-{chi}), where flips are {f['total_flips']} "
          f"(predicted {PREDICTION['flip_floor']}-{PREDICTION['flip_cap']}; "
          f"this run's final was {rows_out[-1]['total_flips']} and the "
          f"published one {ANCHOR[3_000]['flips']})")
    # the load-bearing one for seeds 1-2: is the guard already firing when the
    # predicate arrives? on seed 0 it was 0.0140 against a 0.10 bar.
    print(f"  raw guard AT that crossing: {f['guard_rate']:.4f} of "
          f"{f['guard_denom']} -- P4's bar is {SP.GUARD_BAR}, so the guard "
          f"{'IS' if f['guard_rate'] >= SP.GUARD_BAR else 'is NOT'} already "
          f"firing where the predicate arrives "
          f"({'falsifies' if f['guard_rate'] >= SP.GUARD_BAR else 'replicates'}"
          f" finding 84's separation on this body)")
else:
    print(f"  corrupt NEVER clears {SP.PAIR_BAR} in {STEPS:,} steps -- the "
          f"second falsifier: this rerun did not reproduce the published "
          f"{ANCHOR[3_000]['corrupt']:.4f} and nothing above is readable as a "
          f"trajectory")
lo, hi = SP.PLACEBO_BAND
out_of_band = [r["step"] for r in rows_out
               if not (lo < r["pair_auroc_placebo"] < hi)]
print(f"  placebo outside {list(SP.PLACEBO_BAND)} at steps: "
      f"{out_of_band if out_of_band else 'none'}")
gr = [r["guard_rate"] for r in rows_out]
fl = [r["total_flips"] for r in rows_out]
print(f"  guard monotone in step: "
      f"{all(b >= a - 1e-9 for a, b in zip(gr, gr[1:]))}; flips monotone: "
      f"{all(b >= a for a, b in zip(fl, fl[1:]))}; Spearman(guard, flips) "
      f"{stats.spearmanr(gr, fl).statistic:+.3f}")
print("\nStopping early would change the registered stage's training length, "
      "which is hshunt's\ncall exactly as the other three are. "
      "NO VERDICT IS PRINTED.")
