#!/bin/bash
#SBATCH --job-name=sud2DQsplit
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/users/hshunt/spin_gnn/runs/stagedq_2026-11-21/slurm-%A_%a.out
#
# STAGE DQ -- the per-section split. Inference only, CPU, no training, no
# checkpoint written.
#
# WHY THIS JOB EXISTS. The stage-DQ training array 53243550 produces the
# density sweep (D4) and the trainability read (D5) but NO per-section NLL --
# and D1, the discriminator the whole stage exists for, is a per-section
# number. Stage NSPLIT had the same split of labour (training array, then
# 53203962 for the sections); this is the (d=4, q=3) analogue. The pointer at
# the foot of sudoku2_dq_2026-11-21.sbatch originally named
# stagensplit_fit.py, which is WRONG for this ensemble -- see below.
#
#   # chained behind the training array:
#   sbatch --dependency=afterany:53243550 \
#       scripts/slurm/sudoku2_dq_split_2026-11-21.sbatch
#
#   # machinery check, needs no DQ checkpoint, runs while training:
#   DQ_SPLIT_ARGS=--plumbing-only sbatch --array=0-0 --time=00:30:00 \
#       scripts/slurm/sudoku2_dq_split_2026-11-21.sbatch
#
# afterANY, NOT afterok, DELIBERATELY. If one training cell dies, afterok
# leaves this sitting in the queue as DependencyNeverSatisfied forever and
# nobody is told. With afterany the probe runs, measures every cell that DID
# land, writes MISSING.txt, and stagedq_fit.py then refuses any prediction
# whose cells are not all present. Measure what exists; score only what is
# complete.
#
# WHY NOT stagensplit_fit.py. It is hard-wired to q=4 in two load-bearing
# ways: FLOOR = {48: 1.371, 96: 1.625, 192: 1.861} is keyed on N ALONE, so it
# raises KeyError at N=120 and -- worse -- at N=96 it would silently apply a
# q=4 val_nll floor to a q=3 cell with nothing looking wrong; and
# TA3_TARGET = (5/16) log 2 is finding 109's q=4 constant. Both are on this
# stage's banned list. scripts/stagedq_fit.py carries neither and scores
# against ln 3 = 1.098612.
#
# WHAT IS CHECKED BEFORE ANY DQ CHECKPOINT IS TOUCHED
#   DQ-P0  the entropy estimator returns log V on a uniform distribution
#          (inherited from stage ENTROPY, unchanged).
#   DQ-P1  PLUMBING. stagedq_probe redirects two of probe_run's globals
#          (run_path, which KeyErrors at N=120, and RHOS, a q=4 grid). With
#          those redirects in place it re-measures the ALREADY PUBLISHED
#          d5q4 N=96 L12_s0 cell at rho=0.90 and must reproduce
#          nsplit_0of1.csv to 1e-6 on every section. A code check, not a
#          science comparison -- it runs on the q=4 grid against a q=4 cell
#          precisely so any disagreement is attributable to the redirect.
#   DQ-P2  every DQ cell reports vocab_size == n_nodes + 6 + 3. Sections are
#          labelled by POSITION, so a q=4 checkpoint globbed in by accident
#          would otherwise be scored against ln 3 and look spectacular.
# The scorer's own --dry-run was run on 2026-11-21 before submit and passed
# 4/4, including a check that the q-guard really does fire on a d5q4 curve.
#
# DENSITY GRID, fixed before any cell landed. The q=4 grid
# [0.90 0.75 0.60 0.45 0.35] bottoms at 0.75x of rho_c = 0.465. The same rule
# at rho_c = 0.25 gives 0.1875, and rho_min here is ~0.150, so the DQ grid is
# [0.90 0.70 0.50 0.30 0.18] -- 0.30 just above rho_c, 0.18 just below, with
# margin over the floor. D1 is reported per density AS WELL AS pooled, so the
# pooling cannot be accused of hiding a density structure.
#
set -u
module purge --force
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312

OUT=/scratch/users/hshunt/spin_gnn/runs/stagedq_2026-11-21
mkdir -p "$OUT"
cd /home/users/hshunt/spin_gnn

echo "=== commit ==="
git log --oneline -1
hostname
date +'start %F %T'

python3 -u scripts/stagedq_probe.py \
    --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --ntasks "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --out "$OUT" ${DQ_SPLIT_ARGS:-}

date +'end   %F %T'

# LAST ONE OUT WRITES THE VERDICT. Every task tries, but only the task that
# finds all the sibling CSVs already on disk goes ahead -- so whichever
# finishes last produces a COMPLETE report and the earlier ones produce none.
# Scoring unconditionally would emit three partial VERDICT.txt files and one
# correct one, with nothing in the file saying which it was. No polling, no
# extra job, no dependency: the check is a directory listing. The scorer
# writes --out atomically, so a dead heat cannot tear the file.
if [ -z "${DQ_SPLIT_ARGS:-}" ]; then
    NCSV=$(ls "$OUT"/dqsplit_*.csv 2>/dev/null | wc -l)
    NTASK=${SLURM_ARRAY_TASK_COUNT:-1}
    if [ "$NCSV" -ge "$NTASK" ]; then
        echo
        echo "=== all $NCSV/$NTASK section CSVs present: scoring D1-D5 ==="
        python3 -u scripts/stagedq_fit.py --out "$OUT/VERDICT.txt" || true
    else
        echo
        echo "$NCSV/$NTASK section CSVs present; a later task will score."
    fi
fi
