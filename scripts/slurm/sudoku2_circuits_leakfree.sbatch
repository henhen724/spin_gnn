#!/bin/bash
#SBATCH -J sud2circLF
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -o %x-%j.out

# THE VERDICT RUN: does the LEAK-FREE model USE the DD forcing rule?
#
# hshunt presents Friday 8am PST and needs a yes/no with causal evidence, not
# more decoding scores. This is the causal half.
#
# WHY THIS HAS NEVER BEEN ASKED PROPERLY. Finding 2 ("the model is not running
# DD -- probes, algorithm-ID, counterfactual, SAE all agree") rests on job
# 48583444/49... run on 2026-10-19, against
# sudoku_hparam_sweep_2026-10-19_ep900/d0128_h04_L12_ff01024_lr1e-03_pre. That
# checkpoint is (i) LEAKED-ORDER -- trained a week before finding 16 found the
# adjacency-order colour code -- and (ii) d_model=128, not the d0256 record. So
# the standing "not running DD" verdict is about a narrow model solving a task
# whose answer was readable off token order. It says nothing about the
# leak-free network.
#
# Meanwhile findings 19/21 found, in the leak-free model only, 69
# `justifies_next` features at `pre_color` (best F1 0.743 at L8, against the
# leaked twin's 0.247) -- "u is a KNOWN neighbour of the node whose colour is
# being emitted", which is exactly the input to DD's elimination rule. That is
# a DECODING result: the information is present. Whether the model COMPUTES
# with it is what this job tests, and nothing in the repo has tested it.
#
# THE COUNTERFACTUAL (sudoku_circuits.find_flippable_deductions). Node u is
# forced when its known neighbours cover q-1 colours leaving c free. Recolour
# the one neighbour uniquely holding colour a to c; now a is the free colour,
# so DD's answer for u flips c -> a. Exactly ONE token changes (a hint), and
# u's token, the graph table and the reasoning prefix do not.
#
# A CONFOUND THAT IS ACTUALLY A CONTROL. In the LEAKED encoding `adj[v]` is
# sorted by true colour, and build_pair edits only the hint token -- it does
# not regenerate the graph table. So after the edit the leaked model's graph
# row still encodes the ORIGINAL colouring and now contradicts the hint. A
# model reading colour off adjacency order should therefore KEEP the original
# answer and NOT flip. That makes arm C a sharp negative control rather than a
# nuisance: low flip rate there is the leak doing its job, and it is the
# baseline the leak-free arms must beat.
#
# PRE-REGISTERED PREDICTIONS, with falsifiers.
#
# (a) HEADLINE. Arm A (leak-free, pool 1.0000) has clean_correct >= 0.95 and
#     corrupt_flipped WELL ABOVE the 1/q = 0.25 chance rate -- I will call
#     >= 0.60 a yes. Reason: it solves the honest task perfectly with a zero
#     memorization gap, so the forcing rule is the only route to that score,
#     and findings 19/21 show the needed predicate is represented at exactly
#     the position where the colour is chosen. FALSIFIER: flip rate at or
#     near 0.25, which would mean the perfect solve_rate is produced by
#     something that does not respond to the single fact DD says decides the
#     answer -- a clean "no, it is doing something else", and the most
#     important possible outcome of this run.
#
# (b) Arm A flips MORE than arm C (leaked 0.9960). Falsifier: C >= A, which
#     would say the flip is not about the leak at all and would undercut the
#     whole leak-free-vs-leaked contrast in findings 19/21.
#
# (c) Patching recovery concentrates in MIDDLE layers at the graph row for the
#     recoloured neighbour, i.e. the neighbour table is joined to the known
#     colours there. Falsifier: recovery only at the edited hint token (the
#     model is copying, not joining) or only at the final layer (which is the
#     built-in control cell and must read 1.0 by construction).
#
# (d) Arm B (leak-free 0.854) flips above chance but BELOW arm A. Reason: same
#     mechanism, less of it -- this is the checkpoint findings 19/21 actually
#     opened up, so a monotone A > B > C is what "the mechanism sharpens as the
#     model solves the task" looks like. Falsifier: B >= A, which would mean
#     the last 0.146 of solve_rate came from something other than the forcing
#     rule, and would make the 0.854 model the right interpretability target
#     after all.
#
# ORDER IS DELIBERATE: A, then C, then B. A is the verdict; C is the contrast
# that makes (b) testable and is the sharper of the two controls; B only buys
# the monotonicity in (d). If the 10 h limit truncates anything it must be B,
# never the verdict. Each arm writes its own out-dir, so partial results are
# complete for whichever arms finished.
#
# Same --seed-start in all three arms, so the underlying graphs, colourings and
# hints are the same instances across arms; only the per-model encoding differs
# (shuffle_adj comes from each checkpoint's own config.json, line 299), which
# is the same "each arm on its own distribution" convention findings 19/21 used.
#
# COST. The 10-19 run did 600 instances / 150 cases on d0128 inside 4 h. This
# is d0256 (2x width) at 350/250/250 cases, so ~5x that work. Patching is the
# long pole: 12 layers x 4 sites per case. 10 h is sized for arm A plus both
# controls; arm A alone should land inside 4 h.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

LF1000="${SCRATCH}/spin_gnn/sudoku_n24_data_2026-10-26/data240k_ep100/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"
LF0854="${SCRATCH}/spin_gnn/sudoku_n24_leakfree_2026-10-26/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"
LEAKED="${OAK}/hshunt/checkpoints/spin_gnn/sudoku_dd_d0256_L12_ntrain80000_ep300_sharedpool0.9960_job48912548"
OUT="${SCRATCH}/spin_gnn/sudoku_circuits_leakfree_2026-10-28"
mkdir -p "$OUT"

echo "=== ARM A: leak-free, shared-pool 1.0000 (finding 23) -- THE VERDICT ==="
python -m spin_gnn.sudoku_circuits \
    --checkpoint "$LF1000" \
    --n-instances 1200 \
    --n-cases 350 \
    --out-dir "${OUT}/A_leakfree_1000"

echo
echo "=== ARM C: leaked, shared-pool 0.9960 -- negative control ==="
python -m spin_gnn.sudoku_circuits \
    --checkpoint "$LEAKED" \
    --n-instances 1200 \
    --n-cases 250 \
    --out-dir "${OUT}/C_leaked_0996"

echo
echo "=== ARM B: leak-free, shared-pool 0.8540 -- the checkpoint findings 19/21 opened ==="
python -m spin_gnn.sudoku_circuits \
    --checkpoint "$LF0854" \
    --n-instances 1200 \
    --n-cases 250 \
    --out-dir "${OUT}/B_leakfree_0854"
