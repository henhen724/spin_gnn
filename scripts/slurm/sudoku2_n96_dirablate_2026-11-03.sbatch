#!/bin/bash
#SBATCH --job-name=sud2N96DirAbl
#SBATCH --partition=sganguli
#SBATCH --array=0-1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=%x_%A_%a.out
#
# DOES THE N=96 MODEL *USE* THE DD PREDICATE IT REPRESENTS?
#
# Finding 46 is a probe result and a probe result has a hard ceiling: it shows
# that DD's deducibility predicate is linearly present in the stream
# (frontier_mask AUROC 0.731-0.772 against a label-only free score of 0.6734),
# and that how strongly it is present says nothing about how well the cell
# solves (Spearman -0.086 with rho_50), while next_node top-1 says almost
# everything (-0.943). Correlational both ways. This job deletes each subspace
# from the residual stream and re-measures rho_50 on the identical eval, which
# is the only way to turn "the network represents DD state" into "the network
# runs DD" or into "it doesn't, and the predicate is a byproduct".
#
# WHERE THE DELETION HAPPENS. At EVERY block output, each layer using its own
# layer's probe (--ablate-from 1). Ablating only the last block would be the
# obvious choice -- it is where both probes peak -- and it would be nearly
# meaningless: nothing attends to the last block's output, so removing a
# subspace there changes the logits and no computation at all. Ablating
# everywhere forces layers j+1..L to work without the information.
#
# PRE-REGISTERED. All five conditions remove the SAME k dimensions at the SAME
# layers; k is fixed by the rule in choose_k (smallest k in {2,4,8,16,32} whose
# next_node ablation drops the clean next_node probe to within 0.05 of the
# majority baseline 0.0149) and the whole sweep is printed regardless.
#
#   P1 (the knife is sharp) ablate_next_node raises rho_50 and lowers mean
#      solve_rate well past ablate_random. If it does not, the manipulation is
#      too weak at every k in the sweep and NOTHING else in this job can be
#      read; that is the first number to look at.
#   P2 (THE TEST) ablate_frontier_mask vs ablate_random. If the predicate is
#      load-bearing, deleting it costs materially more than deleting a
#      count-matched random subspace. If the two are indistinguishable, the
#      model represents DD's predicate and does not use it -- which is the
#      answer finding 46's -0.086 predicts, and it would make the standing
#      "does the network run DD" question a NO on the causal axis despite the
#      YES on the decodability axis.
#   P3 (not just logit proximity) ablate_next_color is a second output-adjacent
#      target. If next_node and next_color cost the same, "it sat near the
#      readout" explains the next_node effect and nothing about selection does.
#   P4 (generality) the ordering of P1-P3 is the same on t04 (the best cell,
#      rho_50 0.5635) and t05 (the worst, 0.8290). t05 also gives a
#      dose-response read: the cell with the weakest next_node probe should
#      have the least left to lose by ablating it.
#
# WHAT WOULD INVALIDATE THE RUN. Every condition's read-back is printed before
# its curve. An ablation that does not move its OWN probe's score did not
# remove what it was aimed at. An ablation whose well_formed_rate collapses
# broke the grammar rather than the reasoning, and its rho_50 is about syntax.
#
# PAIRING. --seed-start 60000000000 and n-per-bin 400 on the same 14-point grid
# as the 11-02 fall scan, so the clean condition should reproduce that scan's
# rho_50 for the cell (t04 0.5635, t05 0.8290) to within binomial noise. That
# reproduction is itself a check on the harness; if clean does not match, stop.
#
# PROBE PROTOCOL. 400/160 instances at 20 positions each, matching 50979059
# exactly, NOT sudoku_probes' function defaults. The first submission of this
# job (51013090, cancelled 5 min in) used the defaults and t04's next_node
# probe came back at top-1 0.3986 against the 0.8704 the probe job published
# -- 1436 training positions is not enough for a 256 -> 96 readout. The
# subspace this job deletes is only as good as the probe it comes from, and an
# underfit probe points the knife slightly wrong in every condition, which
# pushes all of P1-P3 toward "no effect".
#
# COST. One 14-point x 400-instance curve measured 18 min on one GPU
# (sud2N96FallScan_50860571_0.out, 06:53:18 -> 07:11:25). Five conditions is
# ~90 min, and ablated models truncate more often, so generations run nearer
# the 1008-token budget: budget 2x, plus probe fitting and the k sweep.
# --time 5:00:00.

set -euo pipefail

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${SLURM_SUBMIT_DIR}"

SCAN="${SCRATCH}/spin_gnn/sudoku_n96_fall_scan_2026-11-02"
OUT_DIR="${SCRATCH}/spin_gnn/sudoku_n96_dirablate_2026-11-03"
mkdir -p "$OUT_DIR"

# tag:checkpoint. t04 is the best cell in the fall grid and t05 the worst, and
# they differ by 0.224 on next_node and by nothing on the predicate -- the two
# ends of exactly the contrast this job is testing.
declare -A CELLS=(
    [0]="t04:${SCAN}/t04_L12_ff01024_n240000_s1/d0256_h04_L12_ff01024_lr1e-03_pre_shuf_fall_s01"
    [1]="t05:${SCAN}/t05_L08_ff01024_n240000_s1/d0256_h04_L08_ff01024_lr1e-03_pre_shuf_fall_s01"
)

SPEC="${CELLS[$SLURM_ARRAY_TASK_ID]}"
TAG="${SPEC%%:*}"
CKPT="${SPEC#*:}"

echo "task ${SLURM_ARRAY_TASK_ID}: ${TAG}"
echo "checkpoint ${CKPT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 -m scripts.direction_ablation_rho \
    --checkpoint "${CKPT}" \
    --conditions clean ablate_next_node ablate_frontier_mask ablate_next_color ablate_random \
    --k auto \
    --ablate-from 1 \
    --probe-epochs 600 \
    --probe-seed 0 \
    --n-train-instances 400 --n-test-instances 160 --max-positions-per-instance 20 \
    --rho-grid 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5 0.45 0.4 0.35 0.32 \
    --n-per-bin 400 \
    --seed-start 60000000000 \
    --out-json "${OUT_DIR}/${TAG}_dirablate.json" \
    --out-csv  "${OUT_DIR}/${TAG}_dirablate.csv"

echo
echo "done ${TAG}; wrote:"
ls -1 "${OUT_DIR}"
