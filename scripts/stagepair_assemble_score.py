"""Assemble the three seeds into the artifact the REGISTERED scorer expects, and
run that scorer unmodified.

WHY THIS EXISTS. `stagepair_score.score()` wants a `pair_summary.json` with a
`per_seed` list of length `N_SEEDS = 3`. The pair probe writes `per_arm` -- three
ARMS (stage Y, the lambda=0 control, the lambda=1 arm) for ONE seed -- because it
was built to isolate the cause, not to be scored. 52452867 and 52456376 together
now cover all three seeds, so the artifact the stage asked for can finally be
built, and the verdict can come from the registered code rather than from me
reading bars off a table.

WHAT IT DOES AND DOES NOT DO. It copies the `warm L1` entry of each seed's probe
summary into `per_seed[seed]` and writes the result. It computes nothing, changes
no bar, and touches neither `stagepair_score.py` nor the probe outputs. Every
number in the output traces to a job that ran on a GPU.

THE EXPECTED OUTPUT IS ALREADY KNOWN AND IS NOT A PASS. All three seeds end above
`FLOOR + STUCK_TOL = 1.630` (1.6636 / 1.6582 / 1.7004), so `converged` is
[False, False, False], fewer than `MIN_SEEDS`, and every check returns pass:false
with an empty `scored` list -- the same refusal seed 0 got, now with all three
seeds present so the refusal is about the threshold alone and not about a missing
seed. Running it is how that statement becomes the scorer's words instead of mine.
"""
import json
import os
import sys

sys.path.insert(0, "/home/users/hshunt/spin_gnn/scripts")
import stagepair_score as SP

ROOT = "/scratch/users/hshunt/spin_gnn"
PROBES = {0: f"{ROOT}/runs/stagepair_pairprobe_2026-11-15",
          1: f"{ROOT}/runs/stagepair_pairprobe_2026-11-15_s1",
          2: f"{ROOT}/runs/stagepair_pairprobe_2026-11-15_s2"}
OUT_DIR = f"{ROOT}/runs/stagepair_score_2026-11-15"

NEEDED = ("pair_auroc_corrupt", "pair_auroc_placebo", "pooled_auroc",
          "guard_rate", "guard_denom", "final_train_nll")

per_seed = []
for s in sorted(PROBES):
    with open(os.path.join(PROBES[s], "pair_summary.json")) as fh:
        arm = json.load(fh)["per_arm"]["warm L1"]
    for k in NEEDED:
        assert k in arm, "seed %d is missing %s" % (s, k)
    assert arm["lambda"] == 1.0, (s, arm["lambda"])
    assert arm["n_cases"] == SP.PAIR_EVAL_N_CASES, arm["n_cases"]
    per_seed.append({k: arm[k] for k in NEEDED})
    print("seed %d: corrupt %.4f  placebo %.4f  pooled %.4f  guard %.4f of %d"
          "  nll %.4f" % (s, arm["pair_auroc_corrupt"],
                          arm["pair_auroc_placebo"], arm["pooled_auroc"],
                          arm["guard_rate"], arm["guard_denom"],
                          arm["final_train_nll"]))

# The denominator trap, asserted rather than trusted: P4 raises on 350, and the
# registered Y denominators are (236, 228, 217), so a probe denominator near
# either would mean the wrong conditional got written into the artifact.
assert all(150 < p["guard_denom"] < 1000 for p in per_seed), per_seed
assert len(per_seed) == SP.N_SEEDS, len(per_seed)

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "pair_summary.json"), "w") as fh:
    json.dump({"per_seed": per_seed,
               "assembled_from": {str(k): v for k, v in PROBES.items()},
               "jobs": ["52452867", "52456376"],
               "note": "assembled, not measured here; see the module docstring"},
              fh, indent=2)

print("\nthe registered scorer, unmodified, on all three seeds:")
print(json.dumps(SP.score(OUT_DIR), indent=2, sort_keys=True))
