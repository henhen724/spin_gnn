#!/bin/bash
#SBATCH --job-name=statusCommit
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/statusCommit_%j.out
#
# End-of-session housekeeping commit: the rewritten STATUS.md, today's third and
# last LOG.md line, and the d448 warmup diagnostic script that job 51860479 is
# running. A JOB AND NOT A LOGIN NODE because $HOME I/O on sh02-ln02 wedges in
# cl_sync_io_wait; never delete .git/index.lock, never kill the process.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists -- it belongs to whichever"
    echo "process holds it. Nothing staged, nothing committed."
    ls -l .git/index.lock
    exit 1
fi

git add STATUS.md LOG.md scripts/slurm/sudoku2_n48_tiedwarm_Sdiag_2026-11-10.sbatch \
    scripts/slurm/status_commit_2026-11-10b.sbatch
echo "add rc=$?"
echo "--- staged ---"
git diff --cached --name-only

for f in STATUS.md LOG.md scripts/slurm/sudoku2_n48_tiedwarm_Sdiag_2026-11-10.sbatch \
         scripts/slurm/status_commit_2026-11-10b.sbatch; do
    if ! git diff --cached --name-only | grep -qx "$f"; then
        echo "NOT STAGED: $f"; exit 1
    fi
done

if git diff --cached --quiet; then
    echo "nothing staged -- already committed"
else
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
status: stages O+P closed, stage S's d448 arm diverged at 2 of 3 seeds

STATUS.md rewritten in place (42 lines) and LOG.md's third and last line for
today appended.

WHAT CHANGED IN THE RUNNING PICTURE. Stage S (array 51853969) has d448 seeds 1
and 2 sitting at train NLL 3.956 / 3.970 with val accuracy 0.031 -- the uniform
predictor over the whole vocabulary, reached inside epoch 0 and flat since --
while d320's three seeds and d448 seed 0 are healthy at 1.63-1.69. Same width as
the healthy seed, so this is not structural; it is the early-step instability
warmup exists for. d448 is the only arm that can discriminate (S3), so S1 --
registered to be read before any level -- does not fit its own branches as
written: "the arm failed to optimize" is a fourth outcome and saying so is
better than forcing it into "flat".

The two diverged tasks are deliberately NOT cancelled. S3 and guard 6 pre-empt
exactly the one-arm cancellation that left stage O uncallable, and a cell with
no final epoch cannot be scored at all.

sudoku2_n48_tiedwarm_Sdiag_2026-11-10.sbatch is the bounded diagnostic that went
out instead (job 51860479, 1 GPU, 8 epochs, its own dated directory, NOT a scored
cell). One knob: --warmup-epochs 0 -> 5 at d448 seed 1. Both readings are written
into the script before it ran -- train NLL below 2.0 means an init instability
and S5's schedule screen is the follow-up; still near 3.9 means the next knob is
the lr (S5's 3e-4 cosine arm), not a longer ramp. The screen itself is ~2
GPU-days and is hshunt's call, not something to start unasked overnight.
MSG
    echo "commit rc=$?"
fi
git log --oneline -1

module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$? (non-fatal: the commit is the record)"
echo "end $(date '+%F %T')"
