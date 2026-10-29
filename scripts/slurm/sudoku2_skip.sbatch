#!/bin/bash
#SBATCH -J sud2skip
#SBATCH -p sganguli,normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -o %x-%j.out

# DOES IT WITHHOLD? The converse of finding 27, and the guard half of DD.
#
# 50075339 and 50077245 both ask the model to FIRE: they take a node DD emits,
# recolour the neighbour doing the forcing, and check that the colour follows.
# Every case is selected because its forcing set covers exactly q-1 colours
# with a unique binding neighbour -- DD's elimination rule by construction.
# That is the caveat finding 27 carries, and it is the one a referee reaches
# for first: a model that wrote a colour for whichever unknown node came next
# in index order would pass that test and still not be running DD.
#
# DD is a rule plus a GUARD: emit v only when v is actually determined. This
# job tests the guard. Same round-0 cases, same one-token discipline, but the
# hint w is recoloured to a colour ALREADY covered by another neighbour, so the
# freed colour a is held by nobody, the forcing set drops to q-2, and v has two
# candidates left -- DD does not emit v in this round at all. The measurement
# moves one token left, from the logits at v's node token (which predict its
# colour) to the logits before it (which predict the node identity).
#
# EVERY CASE CARRIES ITS OWN PLACEBO: a second one-token edit on a hint that is
# NOT a neighbour of v, chosen to keep the revealed colouring proper. v stays
# forced, DD's round is unchanged, and the model should still name v. That is
# the upper baseline `corrupt_emits_v` is read against; without it a low rate
# would just say "edits perturb the node stream".
#
# NO GPU ON PURPOSE, as in 50077245: ~1050 forwards per arm is minutes of CPU
# and the real cost is DD trace generation, which is CPU-bound. Job 50075499
# (the SAE on the 1.0000 checkpoint) is using the GPU this would compete for.
#
# PRE-REGISTERED PREDICTIONS, with falsifiers. Stated knowing an 8-case CPU
# validation already returned 1.000 clean / 0.000 broken / 1.000 placebo with a
# +12.6 logit drop on v against the placebo's -0.06, so (a) is expected to hold;
# this job is the powered version with the arms and the rank breakdown attached.
#
# (a) HEADLINE. `corrupt_emits_v` <= 0.20 for both leak-free arms while
#     `placebo_emits_v` >= 0.95. The model tracks which nodes are DETERMINED,
#     not merely what colour a determined node takes, so DD's guard is causally
#     present and the verdict's "no for search" shrinks to "within a round the
#     order is ascending index" -- which is DD's own convention, not a gap.
#     FALSIFIER: `corrupt_emits_v` stays high with the placebo high too. That
#     is the blind-schedule model, and it would sharpen the negative half far
#     past the `min_unknown` hand rule, which is only a probe result.
#     SPOILER CASE: placebo also collapses => the edit perturbs the stream
#     non-specifically and the test says nothing; report it as uninformative
#     rather than as a result.
#
# (b) When it skips v, it names a node that is genuinely forced under the
#     broken hints at >= 0.70 (`skipped_pred_is_forced`), i.e. it does not just
#     drop v, it moves to the next node DD would have taken. Falsifier: it names
#     unforced or already-known nodes, which would mean "stopped wanting v"
#     without "found the right next one" -- a weaker but still real guard.
#
# (c) The leaked 0.9960 model keeps emitting v. Its adjacency rows still encode
#     the ORIGINAL colouring (finding 16), so the edited hint cannot un-determine
#     anything for it, exactly as in 50077245 where it reverted 0.992 of the
#     time. Falsifier: it withholds too, which would mean the guard survives the
#     shortcut and is not part of what the leak replaced.
#
# (d) Arm B (leak-free 0.8540) matches arm A, continuing finding 27's A = B >> C.
#     Falsifier: a gap, which would finally locate the last 0.146 of solve_rate
#     -- the one thing finding 27 could not do.
#
# (e) No decay with v's position in round 0 (`by_round0_rank`), i.e. the guard
#     is not something that only works on the first node of a round.

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
OUT="${SCRATCH}/spin_gnn/sudoku_skip_2026-10-28"
mkdir -p "$OUT"

# Arm order and case counts match 50075339/50077245 so the fire and withhold
# rates sit in one table.
echo "=== ARM A: leak-free 1.0000 (finding 23) -- THE GUARD VERDICT ==="
python -m spin_gnn.sudoku_skip --checkpoint "$LF1000" \
    --n-instances 400 --n-cases 350 --device cpu \
    --out-dir "${OUT}/A_leakfree_1000"

echo
echo "=== ARM C: leaked 0.9960 -- negative control (prediction c) ==="
python -m spin_gnn.sudoku_skip --checkpoint "$LEAKED" \
    --n-instances 400 --n-cases 250 --device cpu \
    --out-dir "${OUT}/C_leaked_0996"

echo
echo "=== ARM B: leak-free 0.8540 -- the checkpoint findings 19/21 opened ==="
python -m spin_gnn.sudoku_skip --checkpoint "$LF0854" \
    --n-instances 400 --n-cases 250 --device cpu \
    --out-dir "${OUT}/B_leakfree_0854"

echo
echo "=== summaries ==="
for d in "${OUT}"/*/; do
    echo "--- $d"
    python -c "import json; d=json.load(open('${d}skip_summary.json')); print(f\"  clean {d['clean_emits_v']:.3f}  broken {d['corrupt_emits_v']:.3f}  placebo {d['placebo_emits_v']:.3f}  of skips forced {d['skipped_pred_is_forced']:.3f}  n={d['n_cases']}\")" 2>/dev/null || echo "  (none)"
done
