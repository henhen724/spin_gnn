#!/bin/bash
#SBATCH --job-name=s4Refit
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/s4Refit_%j.out
#
# DOES THE KEY RELABELLING MOVE A NUMBER THAT SHIPPED? rung_key now writes the
# eval-T tail as "T24" instead of 24 (commit from job 51932519). The grouping is
# supposed to be identical -- a relabelling, not a regrouping -- and job
# 51932519 ran acc_vs_solve's CLI, which reports 17/20 well-determined RUNGS
# over its own dir list and therefore does NOT speak to the three numbers the
# stage S deck actually shipped: 81 held-out CELLS, 46 discriminating, worst
# miss +0.0729 at tied d192 seed 0. That commit message claims those were
# verified. They were not, by that job. This is the job that checks them.
#
# It re-renders the deck THROUGH THE SAME CODE PATH, with the same builder sha
# passed in, but writes to $SCRATCH so the committed artifact (673301e, stamped
# d9b177f, already uploaded) is not touched. Only the printed diagnostics are
# read from it.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

OUT="${SCRATCH}/spin_gnn/s4_refit_check_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
echo "out=$OUT  (NOT notebooks/stages -- the shipped PDF stays as committed)"

python3 -u -m notebooks.make_stages_deck --commit d9b177f --out "$OUT"
echo "deck rc=$?"

echo
echo "=== what shipped, for the comparison =================================="
echo "  S4 fit: 81 held-out, 46 discriminating"
echo "  worst miss +0.0729  seed 0  rung (16, 4, 'tied_no_mlp', 192, 1024, 0)"
echo "  tied d320 residuals +0.0165 / -0.0034 / +0.0065"
echo "  (the T=24 rung should now print 'T24' where it printed 24)"
echo "end $(date '+%F %T')"
