#!/bin/bash
#SBATCH --job-name=stagemCommit
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagemCommit_%j.out
#
# Commit the stage M write-up. A JOB AND NOT A LOGIN NODE because $HOME I/O on
# sh02-ln02 wedges intermittently -- on 11-09 a `git commit` sat 33 minutes in
# cl_sync_io_wait holding a 0-byte .git/index.lock and then completed, exit 0,
# intact. The rule that follows from that is: move the step into a job, never
# delete the lock, never kill the process.
#
# WHAT IS BEING COMMITTED. notebooks/make_stagem_deck.py already went in with
# d325907 (the stage R job staged it for the KeyError fix, and that commit
# happened to catch the five cosmetic fixes too), so this commit is the
# ARTIFACTS: the 13-page PDF, the executed 14-cell notebook, the builder, the
# render sbatch, this sbatch, and STATUS.md reconciled with findings 58 and 59.
# The page PNGs are gitignored on purpose -- they are a read-back device, not a
# deliverable, and 13 of them would bloat a history that already carries 267 MB
# of model.pt.
#
# NOT COMMITTED HERE: scripts/slurm/stagep_score_2026-11-10.sbatch, which is
# untracked but belongs with stage P's results. One stage per commit.
#
# NO UPLOAD. claude-results-upload.sh is run by hand afterwards, so that a
# failed transfer is read and not retried in a loop.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

MSG=/scratch/users/hshunt/commit_msg_1110_stagem_deck.txt
PDF=notebooks/stagem/stagem_slides.pdf
NB=notebooks/sudoku_n96_stagem_2026-11-09_results.ipynb

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged, nothing"
    echo "committed, and the lock is left alone -- it belongs to whichever"
    echo "process holds it."
    ls -l .git/index.lock
    exit 1
fi

# The deck must exist and be shippable before it is recorded as shipped.
for f in "$PDF" "$NB"; do
    if [ ! -s "$f" ]; then echo "MISSING OR EMPTY: $f"; exit 1; fi
done
sz=$(stat -c %s "$PDF")
echo "pdf bytes=$sz  (upload cap is 2 MB)"
if [ "$sz" -gt 2000000 ]; then echo "OVER THE 2 MB CAP -- not committing"; exit 1; fi

git add "$PDF" "$NB" notebooks/build_stagem_nb.py \
    scripts/slurm/stagem_writeup_2026-11-10.sbatch \
    scripts/slurm/stagem_commit_2026-11-10.sbatch STATUS.md
echo "add rc=$?"
git diff --cached --name-only

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
echo "NEXT, BY HAND: claude-results-upload.sh spin-gnn $PDF ..."
