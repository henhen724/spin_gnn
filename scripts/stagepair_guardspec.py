"""Is stage PAIR's guard TARGETED, or did the model just get less willing to emit?

52452867 produced the one number I had registered a prediction against and got
wrong: P4's guard rate came out 0.1087 on the lambda=1 arm against stage Y's
0.0015, where `PRED_GUARD_FIRES = False` said it would not fire at all. It is
above the 0.10 bar. But the guard rate is CONDITIONAL on the clean prompt
emitting v, and that denominator moved underneath it:

    arm        clean   corrupt  placebo   guard (corrupt | clean)
    Y         0.6560   0.6560   0.6580    0.0015 of 656
    warm L0   0.6480   0.6470   0.6510    0.0031 of 648
    warm L1   0.5060   0.4790   0.5230    0.1087 of 506

The lambda=1 model emits v on 15 points fewer clean prompts. So some of 0.1087
could be a general loss of emission confidence -- the model is simply shakier
about emitting anything, and one changed token anywhere tips it over -- rather
than a guard that reads the predicate.

THE MISSING CONTROL IS THE PLACEBO ARM OF THE SAME CONDITIONAL. 52452867 printed
the placebo's MARGINAL emission rate (0.5230) but never the conditional

    P(placebo stops emitting v | clean emits v)

which is the exact analogue of the guard rate. The marginals cannot settle it:
clean 506 -> corrupt 479 is a net -27 while the guard counted 55 turn-offs, so
28 cases turned ON under corruption. Flips go both ways and only the paired
table separates them.

HYPOTHESIS, stated before the numbers. If the guard is targeted, the placebo
conditional should be far below 0.1087 -- the placebo leaves v forced, so a
model reading the predicate has no reason to withdraw. If instead it comes out
near 0.1087, then "the guard fired" means only that this model's emissions are
fragile under any one-token edit, and P4's crossing says nothing about the
predicate. I predict placebo-off at 0.02-0.05: below the corrupt rate, but above
Y's 0.0015, because the channel's magnitude response to the placebo did grow
(median |dg| 0.0111 -> 0.6686) even though its sign did not (AUROC 0.4360).

THE OVERLAP IS THE SHARPEST READING. Among clean-emitting cases, counting how
many turn off under corrupt only, placebo only, or BOTH distinguishes a guard
from a fragility: a guard turns off a set of cases that the placebo leaves
alone, a fragility turns off the same fragile cases either way.

EVAL ONLY, because 52452867 kept both checkpoints. No training, same three
models, the same registered case list rebuilt from the same seeds, and the
per-case rows are written to disk this time so no later question needs the GPU.

NOT A STAGE, and it changes nothing registered: no new loss term, no new
threshold, no re-reading of a bar. It adds one conditional that should have been
in the first probe.
"""

import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/spin_gnn/scripts")

import stagepair_score as SP
from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sat_transformer2 import SATReasoningTransformer
from spin_gnn.sudoku_cfresp import run_cfresp, summarize_cfresp
from spin_gnn.sudoku_skip import find_unforceable_deductions
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab

Y_DIR = ("/scratch/users/hshunt/spin_gnn/sudoku_n48_determinacy_Y_2026-11-14b/"
         "t00_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s0/"
         "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_chan")
PROBE_DIR = "/scratch/users/hshunt/spin_gnn/runs/stagepair_pairprobe_2026-11-15"
OUT_DIR = "/scratch/users/hshunt/spin_gnn/runs/stagepair_guardspec_2026-11-15"

ARMS = [("Y", Y_DIR),
        ("warm L0", os.path.join(PROBE_DIR, "warm_L0")),
        ("warm L1", os.path.join(PROBE_DIR, "warm_L1"))]

# The case list is rebuilt, not reloaded: same registered stream, same seeds,
# same CPU code path, so it is the same 1,000 cases 52452867 measured. The guard
# rates below reproducing 0.0015/0.0031/0.1087 is the check that it is.
SEED = 0
DATA_SEED = SEED + 777
MAX_LEN = 512
EVAL_SEED_START = SP.PAIR_EVAL_SEED_START
EVAL_N_CASES = SP.PAIR_EVAL_N_CASES
EVAL_N_INSTANCES = 1_400
REPRO_GUARD = {"Y": 0.0015, "warm L0": 0.0031, "warm L1": 0.1087}
REPRO_DENOM = {"Y": 656, "warm L0": 648, "warm L1": 506}

dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"device={dev}  eval seed_start={EVAL_SEED_START:,}  "
      f"cases={EVAL_N_CASES:,}  EVAL ONLY, no training", flush=True)

sudoku_cfg = SudokuReasoningConfig(n_nodes=48, degree=5, n_colors=4,
                                   shuffle_adj=True, fall=True, seed=DATA_SEED)
vocab = build_vocab(sudoku_cfg)


def load_arm(d):
    with open(os.path.join(d, "config.json")) as fh:
        config = json.load(fh)
    assert config.get("determinacy_channel"), "no channel in %s" % d
    assert config["max_len"] == MAX_LEN, config["max_len"]
    m = SATReasoningTransformer(
        vocab.size, d_model=320, num_layers=16, nhead=4, max_len=MAX_LEN,
        dim_feedforward=1024, norm_first=True, arch_mode="tied_no_mlp",
        n_blocks=1, determinacy_channel=True,
        node_offset=vocab.node_offset, n_nodes=vocab.n_nodes)
    m.load_state_dict(torch.load(os.path.join(d, "model.pt"),
                                 map_location="cpu", weights_only=True),
                      strict=True)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == SP.N_TRAINABLE, "not the registered cell: %s" % f"{n:,}"
    m = m.to(dev).eval()
    assert float(m.det_scale.norm()) > 0.05, "dead channel, probe is vacuous"
    return m


print(f"\nbuilding the eval case list at seed_start {EVAL_SEED_START:,}",
      flush=True)
rng = np.random.default_rng(0)
cases = []
for adj, colors, hints, trace in build_instances(sudoku_cfg, EVAL_N_INSTANCES,
                                                 EVAL_SEED_START):
    for case in find_unforceable_deductions(adj, colors, hints, trace, vocab,
                                            rng):
        cases.append((adj, colors, hints, trace, case))
assert len(cases) >= EVAL_N_CASES, "only %d cases" % len(cases)
cases = cases[:EVAL_N_CASES]
print(f"{len(cases):,} cases", flush=True)


def flip_table(rows, key):
    """Conditional turn-off and turn-on rates for one perturbation.

    Both directions, because the marginal rate is a net of the two and the net
    hid 28 turn-ons in 52452867.
    """
    on = [r for r in rows if r["clean_emits_v"]]
    off = [r for r in rows if not r["clean_emits_v"]]
    return {
        "denom_clean_on": len(on),
        "turns_off": int(sum(not r[key] for r in on)),
        "rate_off_given_on": (float(np.mean([not r[key] for r in on]))
                              if on else float("nan")),
        "denom_clean_off": len(off),
        "turns_on": int(sum(r[key] for r in off)),
        "rate_on_given_off": (float(np.mean([r[key] for r in off]))
                              if off else float("nan")),
    }


per_arm = {}
for label, d in ARMS:
    print(f"\n--- {label} " + "-" * 44, flush=True)
    model = load_arm(d)
    rows = run_cfresp(model, vocab, dev, cases)
    s = summarize_cfresp(rows)

    with open(os.path.join(OUT_DIR, "rows_%s.jsonl"
                           % label.replace(" ", "_")), "w") as fh:
        for r in rows:
            fh.write(json.dumps({k: (float(v) if isinstance(v, (float, np.floating))
                                     else int(v) if isinstance(v, (bool, int, np.integer))
                                     else v)
                                 for k, v in r.items()}, sort_keys=True) + "\n")

    cor = flip_table(rows, "corrupt_emits_v")
    pla = flip_table(rows, "placebo_emits_v")

    # THE OVERLAP. Among clean-emitting cases: which perturbation turned each
    # one off. A targeted guard puts its mass in "corrupt only"; a fragility
    # puts it in "both", because the same shaky cases tip either way.
    on = [r for r in rows if r["clean_emits_v"]]
    both = sum(1 for r in on
               if not r["corrupt_emits_v"] and not r["placebo_emits_v"])
    cor_only = sum(1 for r in on
                   if not r["corrupt_emits_v"] and r["placebo_emits_v"])
    pla_only = sum(1 for r in on
                   if r["corrupt_emits_v"] and not r["placebo_emits_v"])

    p = {
        "arm": label,
        "n_cases": len(rows),
        "corrupt": cor,
        "placebo": pla,
        "overlap": {"corrupt_only": cor_only, "placebo_only": pla_only,
                    "both": both, "denom_clean_on": len(on)},
        "pair_auroc_corrupt": SP.matched_pair_auroc(
            [r["dg_corrupt"] for r in rows]),
        "pair_auroc_placebo": SP.matched_pair_auroc(
            [r["dg_placebo"] for r in rows]),
        "clean_emits_v": s["clean_emits_v"],
        "corrupt_emits_v": s["corrupt_emits_v"],
        "placebo_emits_v": s["placebo_emits_v"],
        "median_abs_dg_corrupt": s["median_abs_dg_corrupt"],
        "median_abs_dg_placebo": s["median_abs_dg_placebo"],
    }
    per_arm[label] = p
    print(f"  guard  (corrupt off | clean on) {cor['rate_off_given_on']:.4f} "
          f"of {cor['denom_clean_on']}   "
          f"[52452867: {REPRO_GUARD[label]:.4f} of {REPRO_DENOM[label]}]")
    print(f"  PLACEBO (placebo off | clean on) {pla['rate_off_given_on']:.4f} "
          f"of {pla['denom_clean_on']}   <-- the missing control")
    print(f"  turn-ons: corrupt {cor['turns_on']}/{cor['denom_clean_off']}, "
          f"placebo {pla['turns_on']}/{pla['denom_clean_off']}")
    print(f"  of the clean-on cases that went off: corrupt only {cor_only}, "
          f"placebo only {pla_only}, both {both}", flush=True)

with open(os.path.join(OUT_DIR, "guardspec_summary.json"), "w") as fh:
    json.dump({"per_arm": per_arm,
               "eval": {"seed_start": EVAL_SEED_START,
                        "n_cases": EVAL_N_CASES,
                        "n_instances": EVAL_N_INSTANCES},
               "source": {"Y": Y_DIR, "warm": PROBE_DIR}}, fh, indent=2)
print(f"\nwrote {OUT_DIR}/guardspec_summary.json")

print("\n" + "=" * 76)
print(f"{'arm':<9} {'guard':>7} {'placebo':>8} {'ratio':>6} {'cOnly':>6} "
      f"{'pOnly':>6} {'both':>5} {'denom':>6}")
for label, _d in ARMS:
    p = per_arm[label]
    gr = p["corrupt"]["rate_off_given_on"]
    pr = p["placebo"]["rate_off_given_on"]
    ratio = (gr / pr) if pr > 0 else float("inf")
    o = p["overlap"]
    print(f"{label:<9} {gr:7.4f} {pr:8.4f} {ratio:6.1f} {o['corrupt_only']:6d} "
          f"{o['placebo_only']:6d} {o['both']:5d} {o['denom_clean_on']:6d}")
print("\nP4's bar is on the GUARD column only (>= %.2f) and this script does not"
      % SP.GUARD_BAR)
print("re-read it. The placebo column is the control it had no denominator for.")
print("Guard >> placebo means targeted; guard ~= placebo means the lambda=1")
print("model is merely fragile and P4's crossing is uninformative. No verdict is")
print("printed.")
