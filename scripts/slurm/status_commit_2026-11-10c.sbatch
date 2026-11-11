#!/bin/bash
#SBATCH --job-name=statusCommit
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/statusCommit_%j.out
#
# One-line STATUS.md update: the d448 warmup diagnostic's first two epochs. A job
# and not a login node for the usual reason ($HOME I/O wedges in
# cl_sync_io_wait); never delete .git/index.lock, never kill the process.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists; nothing staged or committed."
    ls -l .git/index.lock
    exit 1
fi

git add STATUS.md scripts/slurm/status_commit_2026-11-10c.sbatch
echo "add rc=$?"
git diff --cached --name-only

if git diff --cached --quiet; then
    echo "nothing staged -- already committed"
else
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
status: the d448 warmup ramp clears the epoch-0 blowup, not yet called

Diagnostic 51860479 (d448 seed 1, --warmup-epochs 0 -> 5, the one knob) ends its
first two epochs at train NLL 2.6923 then 1.9230, val token accuracy 0.3517 then
0.4420. The same seed under warmup 0 was at 3.8873 by the end of epoch 0 and
flat at the uniform predictor after that, so the ramp does prevent the blowup.

NOT CALLED YET, and the script says so: the ramp only reaches the full 1e-3 at
epoch 5, which is where a warmup that merely DELAYS the instability would show
it. The job prints its own verdict at epoch 7 against the threshold written into
it before it ran.
MSG
    echo "commit rc=$?"
fi
git log --oneline -1

module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$? (non-fatal: the commit is the record)"
echo "end $(date '+%F %T')"
