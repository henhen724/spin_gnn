#!/bin/bash
#SBATCH --job-name=statusFixSha
#SBATCH --partition=sganguli
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/statusFixSha_%j.out
#
# One correction, committed on its own so the record shows it.
#
# STATUS.md and the LOG.md line both credited the published deck to builder
# commit 084409b. That was the INTERMEDIATE render (52027787, which died on a
# 27-word note and published nothing). The deck that is live was drawn by
# 52028860 from e8d173e, and e8d173e is what deck.md's own provenance line
# prints. A reader who checked out 084409b would get the builder with the
# biggest_step(first=2) bug that render was fixing, so this is not cosmetic.
#
# From a compute node like every git step here. Refuses to run if
# .git/index.lock exists; never deletes it.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    exit 1
fi

echo "deck provenance says: $(grep -o 'commit [0-9a-f]\{7\}' \
    /home/users/hshunt/slides/decks/spin-gnn/deck.md | head -1)"
echo "STATUS.md says:       commit $(grep -o 'commit \`[0-9a-f]\{7\}\`' STATUS.md \
    | head -1 | tr -d '`' | cut -d' ' -f2)"

git add STATUS.md LOG.md scripts/slurm/status_commit_2026-11-11d.sbatch
git diff --cached --name-only | sed 's/^/  /'

if git diff --cached --quiet; then
    echo "nothing staged -- no commit made"
else
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -m "Correct the published deck's builder commit: e8d173e, not 084409b

084409b was the intermediate render that FAILED on a 27-word note and published
nothing. The live deck was drawn by 52028860 from e8d173e, which is what
deck.md's own provenance line prints. Checking out 084409b would hand a reader
the builder with the biggest_step(first=2) bug that render existed to fix."
    echo "commit rc=$?"
fi

module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$?"
git log --oneline -1

echo "end $(date '+%F %T')"
