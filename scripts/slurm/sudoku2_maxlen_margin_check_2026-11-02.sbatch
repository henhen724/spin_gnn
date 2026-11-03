#!/bin/bash
#SBATCH --job-name=sud2MaxLenMargin
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#
# VERIFIES THE max_len MARGIN, finding 44's one open defect.
#
#   sbatch scripts/slurm/sudoku2_maxlen_margin_check_2026-11-02.sbatch
#
# THE DEFECT. The positional table was sized flush to the longest TRAINING
# sequence, and generate() hard-stops at pos_embed.shape[0]. So a model that
# emits one event more than the longest trace it ever saw is guillotined
# mid-sequence and scored as a failure, with no error anywhere. Finding 44
# measured it: at N=48 max_len came out 414 against a longest true fall trace
# of 408, and the two L=12 cells truncated in up to 0.0275 of the
# deepest-density bins -- which is why their tail numbers are lower bounds.
# At N=96, where traces are ~2x longer, it would bind everywhere.
#
# THE FIX (config.py, train_sudoku_transformer.py, sudoku_hparam_sweep.py):
# max_len = longest training sequence + max_len_margin, with the margin
# defaulting to 2*n_nodes+2 -- the trace grammar's own units, a deduction
# costing 2 tokens and a round marker 1, so it is room to re-emit every node
# once more and still reach SOLVED. 98 rows at N=48, 194 at N=96; at d_model
# 256 that is 25k-50k parameters on a ~9.6M model. `--max-len-margin 0`
# reproduces every run before today exactly.
#
# THE PRICE, STATED UP FRONT: rows above the longest training sequence are
# never trained. They are only ever reached by a generation that has already
# run past every trace in the training set, where the status quo is a
# guaranteed truncation rather than a chance to finish -- so this is weakly
# positive on the metric and cannot be negative, but it is not free and the
# first N=96 run should report truncated_rate to confirm.
#
# WHY A JOB FOR SOMETHING THIS SMALL. The same script was killed on a login
# node after ~13 minutes (buffered stdout lost, stderr intact, which is the
# signature). Instance generation plus DD is the cost, not the model.
#
# WHAT IT ASSERTS -- see scripts/maxlen_margin_check.py:
#   1. margin 0 reproduces the pre-change table exactly (= longest sequence)
#   2. the auto margin is 2*n_nodes+2 above that, and pos_embed agrees
#   3. an explicit margin is honoured
#   4. config.json records the RESOLVED margin, not the -1 sentinel
#   5. generation can actually run past the longest training sequence, which
#      is the whole point; under margin 0 it is hard-stopped there
# Exit 0 and "ALL CHECKS PASSED" is the pass condition; any assert is a fail.

set -euo pipefail

# Module stack only -- $SCRATCH/spin_gnn_venv was eaten by the 90-day purge
# (STATUS open problems). Never piped: a subshell silently loses numpy.
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"

cd "${SLURM_SUBMIT_DIR}"
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

python3 -u scripts/maxlen_margin_check.py
