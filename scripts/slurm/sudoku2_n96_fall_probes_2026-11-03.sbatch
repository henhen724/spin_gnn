#!/bin/bash
#SBATCH --job-name=sud2N96Prb
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out
#
# WHERE DOES DD STATE LIVE WHEN THE REASONING IS ACTUALLY DEEP?
#
# Every mechanistic result in this project -- findings 27-30 (the causal
# elimination rule and the guard), 36 (no single sublayer write matters), 37-39
# (the SAE structure and where it peaks) -- was measured at N=24, where DD
# finishes in a couple of rounds. The six N=96 fall cells of 50860571 are the
# only checkpoints in the project that sit in the SLOW phase: dd_mean_T runs
# 1.19 -> 21.32 across the rho grid (finding 45). If network depth is doing
# reasoning depth, this is the regime where it has to show.
#
# Nothing here is blocked by the seed problem that blocks the depth scan. The
# headline is a WITHIN-model layer profile, and the cross-model comparisons
# below carry their own measured yardsticks rather than borrowing rho_50's.
#
# WHAT IS MEASURED. sudoku_probes, unchanged protocol (400/160 instances, 20
# positions, 600 probe epochs, probe seeds 0 1 2 -- identical to 48861799 and
# to the competence ladder, so the layer profiles are comparable to the N=24
# ones). Per layer: frontier_mask (DD's deducibility predicate), known_mask
# (bookkeeping control, trackable by copying alone), next_node, n_known_nbrs
# (the count DD thresholds at n_colors-1). Layer 0 is the embedding and is the
# control: a target that already scores at layer 0 is being read off the
# tokenization, not computed.
#
# THE SIX CELLS, and what each contrast is for:
#   t00 L=12 240k s0 |
#   t01 L=08 240k s0 | depth, at fixed data and fixed seed
#   t02 L=16 240k s0 |
#   t04 L=12 240k s1   seed replicate of t00  -> training-seed yardstick at L=12
#   t05 L=08 240k s1   seed replicate of t01  -> training-seed yardstick at L=8
#   t03 L=12 120k s0   data halved, same steps -> secondary
#
# PRE-REGISTERED.
#   P1 (headline) The frontier_mask peak sits at the same ABSOLUTE layer index
#      (within +/-1) across L=8/12/16, not at the same FRACTION of depth. That
#      is what finding 11 ("network depth does not track reasoning depth") and
#      finding 36 (no single sublayer write matters) predict: a fixed-size
#      circuit with the extra layers as slack. I predict that. The informative
#      alternative is peak_layer/L constant within +/-0.12, which would say the
#      computation stretches to fill the depth it is given -- and would make
#      depth-vs-reasoning-depth a measurable thing at last.
#   P2 (the yardsticks, and they come first) Two noise terms, both measured
#      here, before any depth claim is allowed:
#        (a) probe-fit noise = spread across probe seeds 0/1/2 within one
#            checkpoint;
#        (b) training-seed noise = |t01 - t05| at L=8 and |t00 - t04| at L=12,
#            in the same statistic.
#      A depth effect is reportable only if it exceeds (b), and (b) must
#      exceed (a) or the training seeds are not even resolved. If (b) swallows
#      the depth difference the way it swallowed rho_50 (0.138 against a 0.05
#      bar, finding 45), THAT is the result and it gets reported as such: it
#      would mean every single-seed mechanistic claim in this project rests on
#      a noise term nobody has measured.
#   P3 (control) known_mask does NOT show the same depth pattern. If the
#      pattern is the same in both, it is bookkeeping and not deduction.
#   P4 (task baseline) frontier is a subset of not-known, so a reader that
#      recovers only known_mask scores on frontier for free. That free score is
#      a property of the ENSEMBLE, and the N=24 value does not transfer, so it
#      is recomputed here at N=96 fall. Any frontier number at or below it
#      means nothing. This is why frontier_from_known_baseline.py needed the
#      ensemble flags it did not have.
#   Secondary: t03 vs t00. rho_50 could not resolve 120k from 240k (0.069
#      against a 0.138 yardstick). If the probe profile separates them, the
#      probe channel is the quieter instrument and the data question should be
#      asked there instead.
#
# ARRAY BY PROBE SEED, one task each, so a timeout costs one seed instead of
# all three: sudoku_probes writes its CSV only after every checkpoint is done.
# The three CSVs concatenate on (run, seed, layer, target).
#
# COST. The path was smoke-tested on this exact checkpoint family: 12 N=96
# instances through an L=8 cell took 24.8 s on one CPU core at 3 probe epochs,
# and instance generation is 29 ms each. On a GPU, 560 instances x 6
# checkpoints at 600 probe epochs is ~40 min expected; --time 4:00:00 is the
# margin for the L=16 cell and for a cold filesystem.

set -euo pipefail

TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

S="${SCRATCH}/spin_gnn"
SCAN="${S}/sudoku_n96_fall_scan_2026-11-02"
OUT_DIR="${S}/sudoku_n96_fall_probes_2026-11-03"
mkdir -p "$OUT_DIR"

CKPTS=(
    "${SCAN}/t00_L12_ff01024_n240000_s0/d0256_h04_L12_ff01024_lr1e-03_pre_shuf_fall"
    "${SCAN}/t01_L08_ff01024_n240000_s0/d0256_h04_L08_ff01024_lr1e-03_pre_shuf_fall"
    "${SCAN}/t02_L16_ff01024_n240000_s0/d0256_h04_L16_ff01024_lr1e-03_pre_shuf_fall"
    "${SCAN}/t03_L12_ff01024_n120000_s0/d0256_h04_L12_ff01024_lr1e-03_pre_shuf_fall"
    "${SCAN}/t04_L12_ff01024_n240000_s1/d0256_h04_L12_ff01024_lr1e-03_pre_shuf_fall_s01"
    "${SCAN}/t05_L08_ff01024_n240000_s1/d0256_h04_L08_ff01024_lr1e-03_pre_shuf_fall_s01"
)

echo "=== smoke check ==="
python3 - "${CKPTS[@]}" <<'PY'
import json, os, sys
import torch
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0), "torch", torch.__version__)
# The whole point is that these are the FALL cells at N=96; a checkpoint that
# is not one would be probed off-distribution and would look like a null.
for d in sys.argv[1:]:
    c = json.load(open(os.path.join(d, "config.json")))
    want_L = int(os.path.basename(d).split("_L")[1][:2])
    assert c["n_nodes"] == 96 and c["fall"] and c["shuffle_adj"], (d, c["n_nodes"], c["fall"])
    assert c["num_layers"] == want_L, (d, c["num_layers"], want_L)
    print(f"  ok L={c['num_layers']:2d} max_len={c['max_len']} {os.path.basename(os.path.dirname(d))}")
PY

echo
echo "=== probes, probe seed ${TASK} ==="
python3 -m spin_gnn.sudoku_probes \
    --checkpoint "${CKPTS[@]}" \
    --n-train-instances 400 \
    --n-test-instances 160 \
    --max-positions-per-instance 20 \
    --probe-epochs 600 \
    --seed "${TASK}" \
    --out-csv "${OUT_DIR}/probes_seed${TASK}.csv"

if [ "${TASK}" = "0" ]; then
    echo
    echo "=== label-only frontier baseline, N=96 fall (P4; no model involved) ==="
    # -m, not the path: without the venv the repo root reaches sys.path only
    # through module form, and `python3 scripts/...py` cannot import spin_gnn.
    python3 -m scripts.frontier_from_known_baseline \
        --n-nodes 96 --degree 5 --n-colors 4 --fall --shuffle-adj \
        --n-train 250 --n-test 150 --probe-epochs 300 \
        --out-json "${OUT_DIR}/frontier_baseline_n96_fall.json"
fi
