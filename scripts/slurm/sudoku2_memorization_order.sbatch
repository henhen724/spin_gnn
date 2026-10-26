#!/bin/bash
#SBATCH --job-name=sud2memord
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.out
#
# WHAT ARE THESE NETWORKS ACTUALLY FITTING? Eval-only, no training.
#
# The project has been inferring memorization from NLL curves, and those curves
# cannot support it. `train_nll` is epoch_loss/epoch_n -- averaged over the epoch
# WHILE the weights update -- while `val_nll` uses end-of-epoch weights, so
# train_nll carries an upward bias worth +0.3263 at epoch 10 and decaying as the
# curve flattens (measured on the N=96 control). Against n_val=300, whose own
# difficulty draw is worth a few hundredths, the SIGN of a +-0.03 "train - val
# gap" is not interpretable. Finding 15's mechanism story -- "cell B is the only
# N=96 run with a positive gap, so it fits then generalizes" -- rests on +0.0239
# and does not survive this. (Finding 15's RESULT, data as the binding lever at
# +0.1007 on 1500 pool instances, is untouched; only the explanation is.)
#
# PART A measures memorization directly instead. Training index i is reproducible
# as build_instances(cfg, 1, i), so the first 600 training instances are
# recoverable exactly; score the model on its OWN training set and on held-out
# instances from the shared-pool range. The training set is built once, before
# the epoch loop, so a 150-epoch run sees the same instances 150 times and CAN
# memorize them -- fresh-per-instance generation prevents overlap between splits,
# not memorization of a fixed split.
#   Checkpoints span the data budget, since the hypothesis is that data closes
#   the gap: N=48 L=12 at 20000 (final train-val -0.1666, the largest NLL gap in
#   the project), then the N=96 control at 20000 against cell B at 80000.
#
# PART B asks whether the ANSWER IS IN THE TOKEN ORDER, which would not be
# memorization at all and would be invisible to every train/val diagnostic.
# planted_regular_graph appends edges in class-pair block order and fills adj[v]
# in that order, so for a node of class a the blocks containing it are visited in
# increasing order of the other class index. MEASURED on real training instances:
# adj[v] lists v's neighbours sorted by their TRUE COLOUR for 100.0% of nodes at
# N=24, 48 and 96, and encode_instance emits adj[i] in raw table order. For
# 61-67% of nodes the neighbours cover all q-1 other colours, so the single
# absent colour block identifies v's own colour. Round events are also emitted in
# ascending node index for 100.0% of rounds, which makes `next_node` partly a
# sorting task -- the 'sorted-order confound' already noted at layer 0 in the SAE
# work, now traced to its source.
#
# Four orderings, all preserving the edge set, the colouring, the hints and the
# DD trace (the trace is RECOMPUTED on the reordered adjacency and asserted equal,
# so this is a semantics-preserving counterfactual, not a different problem):
#   identity / within_block / shuffle / byid.
# within_block permutes only among neighbours of the SAME colour, so the colour
# code survives and only the order changes; within-block order is already
# arbitrary in training (sorted by node id in 36-39% of runs, i.e. chance), so it
# is IN-DISTRIBUTION. It is the control that separates "the order changed" from
# "the colour code was destroyed" -- a drop under shuffle alone would be
# confounded with plain distribution shift.
#   Checkpoints: the N=24 RECORD MODEL at 0.9960 from $OAK (the most decisive
#   single number here -- if a 0.9960 model collapses when adj[v] is shuffled,
#   the headline result is substantially a decoding of token order), cell B at
#   N=96, and the new best N=48 thin-deep cell.
#
# PRE-REGISTERED:
#  (a) The memorization gap is POSITIVE at 20000 and SMALLER at 80000. Reason:
#      the N=48 20000 run ends 0.1666 below its val curve, which is far too large
#      to be the averaging bias. Falsifier: no resolvable gap at 20000, meaning
#      the NLL gap is teacher-forcing-only and does not reach free generation --
#      in which case "memorization" is the wrong frame for all of it and the
#      150ep-beats-300ep result (+0.0280) needs another explanation.
#  (b) within_block does NOT differ resolvably from identity (p>0.05). This is a
#      sanity check on the whole design; if it fails, the counterfactual is
#      measuring distribution shift and part B cannot be read at all.
#  (c) shuffle is resolvably WORSE than identity on every checkpoint. Falsifier:
#      no drop, which clears the encoding -- the colour sort is present but
#      unused, findings 2/8 stand as they are, and no retrain is needed.
#  (d) The drop under shuffle is LARGER at N=96 than at N=24. Reason: longer
#      adjacency lists and deeper DD traces make genuine deduction costlier
#      relative to reading position, so the shortcut should carry more of the
#      work where the problem is harder. This one is a guess about mechanism
#      rather than a load-bearing prediction; note that my record on mechanism
#      guesses in this project is 0 for 1 (finding 15).
#
# WHAT THIS JOB DOES NOT SETTLE. Even if (c) holds, the clean attribution needs a
# RETRAIN under a colour-free adjacency order, because shuffle at eval time is
# off-distribution by construction and within_block controls for that only
# approximately. This job decides whether that retrain is worth its GPU time, and
# at ~4 h it is far cheaper than the ~8 h N=96 training run it would gate.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT="${SCRATCH}/spin_gnn/sudoku_memorization_2026-10-26"
mkdir -p "$OUT"

N48SCAN="${SCRATCH}/spin_gnn/sudoku_layer_scan_2026-10-24_n48/d0256_h04_L12_ff01024_lr1e-03_pre"
N96CTRL="${SCRATCH}/spin_gnn/sudoku_n96_baseline_2026-10-25/d0256_h04_L12_ff01024_lr1e-03_pre"
N96CELLB="${SCRATCH}/spin_gnn/sudoku_n96_plateau_2026-10-26/d0256_h04_L12_ff01024_lr1e-03_pre"
N48BEST="${SCRATCH}/spin_gnn/sudoku_ff_depth_tradeoff_2026-10-25_n48/d0256_h04_L08_ff00256_lr1e-03_pre"
RECORD="${OAK}/hshunt/checkpoints/spin_gnn/sudoku_dd_d0256_L12_ntrain80000_ep300_sharedpool0.9960_job48912548"

echo "=== PART A: own training instances vs held-out, across the data budget ==="
python -m spin_gnn.sudoku_memorization_eval \
    --checkpoint "$N48SCAN" "$N96CTRL" "$N96CELLB" \
    --n-memorization 600 \
    --n-ordering 0 \
    --out-csv "${OUT}/memorization_gap.csv"

echo
echo "=== PART B: is the answer in the token order? ==="
python -m spin_gnn.sudoku_memorization_eval \
    --checkpoint "$RECORD" "$N96CELLB" "$N48BEST" \
    --n-memorization 0 \
    --n-ordering 400 \
    --out-csv "${OUT}/ordering_counterfactual.csv"
