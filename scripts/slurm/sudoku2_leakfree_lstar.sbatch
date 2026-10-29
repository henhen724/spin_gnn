#!/bin/bash
#SBATCH --job-name=sud2LFlstar
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#
# DOES NETWORK DEPTH TRACK REASONING DEPTH, ONCE THE SHORTCUT IS GONE?
#
# ===========================================================================
# NOT READY TO SUBMIT AS WRITTEN. Three variables must be set in the
# submitting environment, and the script aborts in the first second if they
# are not:
#
#   NTRAIN  training instances per cell. Set from job 49705064, which is
#           measuring whether leak-free N=24 clears 0.90 at all and at what
#           budget. Submitting this series at a budget where even L=12 scores
#           0.854 would return L* UNDEFINED in every bin and cost five cells
#           to say nothing -- the exact error of finding 15 at N=96.
#   EPOCHS  epochs per cell, likewise from 49705064 (its cells A and B
#           separate data from compute at 300 vs 100).
#   LAYERS  the `num_layers` ladder, space-separated, in order of information
#           (a timeout truncates the tail). The full ladder is 2 3 4 6 8;
#           see COST below for why it is meant to be submitted in two halves.
#
#   e.g.  NTRAIN=240000 EPOCHS=300 LAYERS="4 3 2" sbatch scripts/slurm/sudoku2_leakfree_lstar.sbatch
#         NTRAIN=240000 EPOCHS=300 LAYERS="6 8"   sbatch scripts/slurm/sudoku2_leakfree_lstar.sbatch
#
# NNODES defaults to 24. The same script runs the N=48 half of "a scan at
# each N" with NNODES=48, but only after N=24 has fixed the budget and the
# ladder -- at N=48 every cell costs ~2x more and the leak-free baseline
# there has never been measured.
# ===========================================================================
#
# THE QUESTION. hshunt's plan: "a hyper-parameter scan at each N to figure out
# if there is a relationship between reasoning depth required and for example
# MLP depth." L*(T) is that relationship stated as a measurement: the smallest
# `num_layers` that reaches 0.90 on instances of DD depth T, at FIXED realized
# density so the answer is about depth and not about how constrained the
# instance is.
#
# WHY IT IS WORTH REDOING. Finding 11 measured L*(T) across five series, 20
# bins, and found it FLAT: slope +0.00 layers per round against a
# pre-registered +3, with L*(24)=3 at every T. That result was measured on the
# leaked encoding (finding 16), where `adj[v]` arrives sorted by the
# neighbour's true colour and the record model collapses 0.9975 -> 0.1450 when
# that order is permuted. On a task solved by decoding token order, depth does
# not bind, so a flat L*(T) was close to a foregone conclusion. Finding 17
# then showed that leak-free, solve_rate falls monotonically with T at fixed
# density (rho=0.60: 0.946/0.913/0.871/0.827 over T=2..5, r=-0.998). THIS is
# the experiment finding 11 was trying to be.
#
# ONE CHANGE PER CELL. Every cell is the 49659502 config -- N=24, degree=5,
# q=4, rho=0.3, rho_max=0.95, removal hints, d_model=256, nhead=4, ff=1024,
# pre-norm, lr=1e-3 constant, batch=32, seed=0, --shuffle-adj -- with
# `num_layers` as the only difference, at the NTRAIN/EPOCHS budget that
# 49705064 establishes. `num_layers` is not an INSTANCE_KEY, so every cell
# pools with 49659502's 0.8540 and with 49705064's two cells, and all of them
# are paired-testable on the same n=1500 pool and on the same depth bins.
#
# PRE-REGISTERED (to be read against finding 11's flat result):
#  (a) L*(T) RISES with T. Concretely: L*(T_max) > L*(T_min) by at least one
#      rung of the ladder at rho=0.60. This is the prediction finding 11
#      falsified under the leak and that finding 17's monotone depth cost
#      revives. Moderate confidence -- it is the same prediction I got wrong
#      before, and the reason I now expect it (the task is different) is a
#      reason I should have to state in advance.
#      Falsifier: L*(T) flat again, within one rung across the whole T range.
#      That would be the more important result of the two: it would say the
#      knee is a fixed cost of the problem and not of the deduction chain,
#      which is finding 11's "L=2 scores 0.686 at T=1" observation promoted
#      from a curiosity to the main effect.
#  (b) L*(24) leak-free is LARGER than the leaked L*(24)=3. Reason: the clean
#      task costs 0.142 solve_rate at L=12 (finding 17), so the shallow end of
#      the ladder has further to climb. Falsifier: L=3 clears 0.90 leak-free,
#      which would mean the extra difficulty is spread over all depths rather
#      than concentrated where serial computation is needed.
#  (c) The ordering of finding 13 survives: at matched parameters the deeper,
#      narrower cell still wins. Not tested directly here (this ladder is at
#      fixed ff=1024) but L=4 vs the ff=256 cells of the N=48 scan will be,
#      and if (a) holds then depth is the currency that buys T.
#  (d) Low confidence, stated so a surprise counts: the per-T solve_rate
#      curves of the DEEP cells are flatter in T than those of the shallow
#      cells, i.e. the interaction has the sign that "more layers buys more
#      rounds" implies. Falsifier: parallel curves, which would make the depth
#      cost an additive difficulty term with no architectural remedy.
#
# WHAT WOULD INVALIDATE THE WHOLE SERIES, and is checked first: if 49705064
# shows the depth cost flattening at the larger budget (its pre-reg (d), r at
# rho=0.60 above -0.5), then finding 17's depth signal was a data artefact and
# there is no T-dependence left to explain. Do not submit this series before
# reading that result.
#
# THE N=48 SUBMISSION (10-29 01:50): NNODES=48 NTRAIN=240000 EPOCHS=100
# LAYERS="6 4 3". Three things fix those values and none of them are guesses:
#
#  * The budget is finding 23's. 240000 x 100ep is the cheap cell that tied
#    240000 x 300ep instance-for-instance on all 1500 pool instances, so the
#    binding constraint was instance COVERAGE, not gradient steps. EPOCHS=300
#    would spend 3x the GPU for a measured tie.
#  * N=24 is OUT, which is why this runs at 48 at all. Finding 23 put every
#    rho=0.60 depth bin at the leak-free N=24 task at exactly 1.0000, so L*(T)
#    there is 3 at every T by saturation and the pool cannot separate cells.
#    Finding 22 says the same thing from the other side: the T-rise is visible
#    only at a cell sitting ON the 0.90 gate.
#  * THE LADDER IS ORDERED 6, 4, 3 FOR A REASON. At N=48 the leak-free task has
#    never been measured, so the first question is not where the knee is but
#    whether this budget clears the gate anywhere; L=6 is the anchor that
#    answers it (leaked L*(48) = 4 at ff=1024, and leak-free can only be
#    higher). 4 and 3 then bracket the knee from above and below. If L=6 misses
#    the gate in every bin, L* is UNDEFINED rather than large and the series
#    says nothing about T -- finding 15's error at N=96 -- so READ L=6 FIRST
#    and do not write up 4 and 3 on their own.
#
# A timeout therefore costs the L=3 cell, which is the one rung that only
# matters if 4 and 6 both clear. The three checkpoints land on disk as they
# finish, so a timeout inside the evals is recoverable by re-running the two
# eval commands below on the saved cells, not by retraining.
#
# COST, at NTRAIN=240000 / EPOCHS=300 and N=24. 49659502 trained L=12 at
# 80000x300 in ~4.5 h, so at 3x the data a cell is roughly 1.1 h per layer:
# L=2 ~2.3 h, L=3 ~3.4 h, L=4 ~4.5 h, L=6 ~6.8 h, L=8 ~9 h. The full ladder is
# ~26 h of training plus ~2 h of evals, which is why LAYERS exists: submit
# "4 3 2" (~12 h with evals) and "6 8" (~18 h with evals) as two jobs rather
# than one 30 h job whose tail is the least informative part. The 20 h limit
# covers either half.

set -euo pipefail
NTRAIN="${NTRAIN:?set NTRAIN from job 49705064's budget, e.g. NTRAIN=240000}"
EPOCHS="${EPOCHS:?set EPOCHS from job 49705064's budget, e.g. EPOCHS=300}"
LAYERS="${LAYERS:?set LAYERS, the num_layers ladder in order of information, e.g. LAYERS=\"4 3 2\"}"
NNODES="${NNODES:-24}"

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT="${SCRATCH}/spin_gnn/sudoku_n${NNODES}_leakfree_lstar_$(date +%Y-%m-%d)"
mkdir -p "$OUT"
echo "N=${NNODES} layers=[${LAYERS}] n_train=${NTRAIN} epochs=${EPOCHS} -> ${OUT}"

COMMON=(--n-nodes "$NNODES" --degree 5 --n-colors 4 --shuffle-adj
        --d-model 256 --nhead 4 --dim-feedforward 1024
        --n-train-instances "$NTRAIN" --epochs "$EPOCHS"
        --n-val-instances 300 --batch-size 32 --n-per-bin 100)

CKPTS=()
for L in $LAYERS; do
    cell="${OUT}/L$(printf '%02d' "$L")"
    echo
    echo "=== cell L=${L} ==="
    python -m spin_gnn.sudoku_hparam_sweep "${COMMON[@]}" \
        --num-layers "$L" --out-dir "$cell"
    CKPTS+=("${cell}/d0256_h04_L$(printf '%02d' "$L")_ff01024_lr1e-03_pre_shuf")
done

# The L=12 cells already measured leak-free at this N, added as references so
# every comparison is paired on the same instances. Guarded: at N=48 neither
# exists yet and their absence must not fail the evals.
for ref in "${SCRATCH}/spin_gnn/sudoku_n24_leakfree_2026-10-26/d0256_h04_L12_ff01024_lr1e-03_pre_shuf" \
           "${SCRATCH}/spin_gnn/sudoku_n24_data_2026-10-26/data240k_ep300/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"; do
    if [ "$NNODES" = 24 ] && [ -d "$ref" ]; then CKPTS+=("$ref"); else echo "skipping reference $ref"; fi
done

echo
echo "=== pre-regs (a)+(d): L*(T), all cells on ONE set of depth bins ==="
# n_pool 1500 with min 60 per bin: a bin at 0.90 then has a Wilson half-width
# of ~0.07, which is the resolution the threshold claim needs. rho=0.60 is the
# wide-depth sweep (T=2..5 at N=24); rho=0.75 is the shallow control.
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${CKPTS[@]}" \
    --rho 0.75 0.60 --n-pool 1500 --min-per-bin 60 \
    --out-csv "${OUT}/depth_lstar_${SLURM_JOB_ID}.csv"

echo
echo "=== pre-reg (b): the shared n=1500 pool, disjoint from every training set ==="
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "${CKPTS[@]}" \
    --n-instances 1500 --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500_${SLURM_JOB_ID}.csv"

echo
echo "=== summaries ==="
cat "${OUT}/sharedpool_n1500_${SLURM_JOB_ID}.csv"
