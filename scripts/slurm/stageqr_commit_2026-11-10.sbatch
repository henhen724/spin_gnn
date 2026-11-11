#!/bin/bash
#SBATCH --job-name=stageqrCommit
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stageqrCommit_%j.out
#
# Commit the stage Q + R write-up, after the PNG read-back. A JOB AND NOT A
# LOGIN NODE for the same reason stage M's commit was: $HOME I/O on sh02-ln02
# wedges intermittently in cl_sync_io_wait. Never delete .git/index.lock, never
# kill the process -- move the step into a job and let it finish.
#
# WHAT IS BEING COMMITTED. All five paths are untracked, so this commit brings
# in the whole write-up: the builder (with the three read-back fixes), the
# notebook builder, the render sbatch, this sbatch, the 20-page PDF and the
# executed notebook. The page PNGs are gitignored on purpose -- they are a
# read-back device, not a deliverable.
#
# NOT COMMITTED HERE: STATUS.md. It is rewritten once at the end of the session,
# not once per artifact, so that its "running jobs" section is true when it
# lands rather than true for ten minutes.
#
# NO UPLOAD. claude-results-upload.sh is run by hand afterwards, so a failed
# transfer is read rather than retried in a loop.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

MSG=/scratch/users/hshunt/commit_msg_1110_stageqr_deck.txt
PDF=notebooks/stageqr/stageqr_slides.pdf
NB=notebooks/sudoku_n48_stageqr_2026-11-10_results.ipynb

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged, nothing"
    echo "committed, and the lock is left alone -- it belongs to whichever"
    echo "process holds it."
    ls -l .git/index.lock
    exit 1
fi

for f in "$PDF" "$NB" notebooks/make_stageqr_deck.py; do
    if [ ! -s "$f" ]; then echo "MISSING OR EMPTY: $f"; exit 1; fi
done
sz=$(stat -c %s "$PDF")
echo "pdf bytes=$sz  (upload cap is 2 MB)"
if [ "$sz" -gt 2000000 ]; then echo "OVER THE 2 MB CAP -- not committing"; exit 1; fi

git add "$PDF" "$NB" notebooks/make_stageqr_deck.py notebooks/build_stageqr_nb.py \
    scripts/slurm/stageqr_writeup_2026-11-10.sbatch \
    scripts/slurm/stageqr_commit_2026-11-10.sbatch
echo "add rc=$?"
echo "--- staged ---"
git diff --cached --name-only

# A `git add` on an ignored path says nothing at all, and the PDF lives inside
# notebooks/stageqr/ whose page*.png are ignored. Check the artifacts actually
# staged rather than trusting the add.
for f in "$PDF" "$NB"; do
    if ! git diff --cached --name-only | grep -qx "$f"; then
        echo "NOT STAGED (ignored? already committed?): $f"
        exit 1
    fi
done

if git diff --cached --quiet; then
    echo "nothing staged -- already committed"
else
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -F "$MSG" -q
    echo "commit rc=$?"
fi
git log --oneline -1

module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$? (non-fatal: the commit is the record)"
echo "end $(date '+%F %T')"
echo
echo "NEXT, BY HAND: ~/.local/bin/claude-results-upload.sh spin-gnn $PDF \\"
echo "  \"stage Q+R counterfactual audit of no_mlp L=16 d192\" \\"
echo "  \"flip 0.994 on derived facts, -0.006 vs round-0\""
