#!/bin/bash
#
# PRICE the pair-margin arm. STATUS.md next step 1 says to price it before
# registering it; this is that, and only that.
#
# NOTHING IS TRAINED, nothing is registered and no bar is set. The job builds
# instances, scans them for matched clean/corrupted pairs, and times one fixed
# batch through the stage-Y shape with and without the second forward the
# margin needs. No checkpoint is read and none is written.
#
# What it answers, none of which can be guessed:
#   A  the fraction of training INSTANCES that carry a pair at all -- the
#      number that decides how much of an epoch the term even applies to.
#      Stage Z's "400 instances, 350 cases" is a case count at
#      max_per_instance=4 and says nothing about this.
#   B  the scan's own cost, separated from build_instances, which every run
#      already pays.
#   C  the real step-time ratio at d320 L16 tied_no_mlp with the channel on,
#      including the corrupted forward's unread readout.
#
# A 60-instance login-node check already showed the scan is ~0.08 ms/instance
# and the yield ~0.73; this job settles both on 20,000 instances and measures
# the part a login node must not measure, which is the GPU step.
#
# One GPU because the ratio has to come from the real shape at the real batch.
# ~20 min: the scan over 20,000 instances is the bulk of it (build_instances
# runs about 13 ms/instance), the timing itself is under two minutes.
#
#SBATCH --job-name=stagepairPrice
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache

OUT=/scratch/users/hshunt/spin_gnn/runs/2026-11-15_stagepair_price
mkdir -p "$OUT"

echo
echo "=== compile ======================================================"
python3 -m py_compile scripts/stagepair_price.py || { echo "py_compile FAILED"; exit 1; }

echo
echo "=== price ========================================================"
# seed 9e8: disjoint from training (0..240299), from stage Y's probe and from
# stage Z's two streams (6e8 for the cases, 7e10 for the AUROC instances), so
# the yield is measured on instances no checkpoint has seen. It does not
# matter for a timing, and it does matter for the yield.
python3 -u -m scripts.stagepair_price \
    --n-instances 20000 --seed 900000000 --steps 300 --warmup 30 \
    --out "$OUT/price.json"
RC=$?
echo "price rc=$RC"

echo "end $(date '+%F %T')"
exit $RC
