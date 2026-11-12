#!/bin/bash
#SBATCH --job-name=statusCommitU
#SBATCH --partition=sganguli
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/statusCommitU_%j.out
#
# Commit the stage U handoff: STATUS.md rewritten in place, two dated LOG.md
# lines.
#
# The LOG.md lines were appended BEFORE this job (two lines, under the standing
# cap of three, and nothing that is already a finding): stage U's scored result
# and its publication, which is a delivery event plus the read-back defects the
# next session should expect to have to hunt. This job therefore only commits --
# appending again would duplicate them.
#
# STATUS.md's pointer line is corrected here too: it claimed FINDINGS.md held
# findings 1-64 in full, and FINDINGS.md has lagged since stage K at 54. It now
# says what is true -- 1-54 in FINDINGS.md, 55-64 in LOG.md and the decks.
#
# From a compute node like every git step here: $HOME I/O on sh02-ln02 wedges in
# cl_sync_io_wait. Refuses to run if .git/index.lock exists; never deletes it.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    ls -l .git/index.lock
    exit 1
fi

echo "STATUS.md is $(wc -l < STATUS.md) lines (cap 60)"
if [ "$(wc -l < STATUS.md)" -gt 60 ]; then
    echo "FAILING CLOSED: STATUS.md is over the 60-line cap."
    exit 1
fi

git add STATUS.md LOG.md scripts/slurm/status_commit_2026-11-11c.sbatch \
        notebooks/make_stageu_web.py
echo "add rc=$?"
git diff --cached --name-only | sed 's/^/  /'

if git diff --cached --quiet; then
    echo "nothing staged -- no commit made"
else
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - << 'MSG'
Hand off 2026-11-11: stage U scored, published, and read back twice

STATUS.md rewritten in place (under the 60-line cap) and two dated LOG.md lines
appended.

Stage U's registered gate prediction is FALSIFIED -- k=2 reads 0.752 and k=4
reads 0.755 against the control's 0.757, all 3/3 crossing and all inside 0.01,
which is the competing hypothesis' prediction -- and the train NLL floor is flat
across k, so the floor belongs to the recipe and not to the role count.

The result worth keeping is the bimodality the bound-aware median hides: one
seed in EACH k > 1 arm left the plateau (k=2 seed 2 at rho_50 0.603, k=4 seed 1
at 0.579, token accuracy 0.4929 and 0.4948) while no k=1 seed did in three tight
seeds. U4 puts both ON the N=48 accuracy-to-solve curve, so nothing
architectural happened: what binds is which basin the optimizer finds. One seed
at 0.579 does not displace J2's three-seed 0.580 and no improvement is claimed.

Deck published at slides-hshunt.pages.dev/spin-gnn/ (18 pages, builder 084409b,
rendered by 52028860, slides-repo commit b579821, check-gate.sh prints "gate
holds"). The last fix is in this commit: biggest_step() took first=2 and was
reporting the tail of the initial collapse, so the learning-curve note printed
"two plateau seeds step as sharply (2 and 2)" where the figure shows steps at
epochs 8 and 23. first=5 is past the collapse.

STATUS.md now names stage W -- 100 epochs on stage U's own k=2 cell -- as the
arm the data motivates, because both basin seeds were still descending four to
ten times faster than the plateau seeds when the fifty-epoch budget ended.
MSG
    echo "commit rc=$?"
fi

module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$? (non-fatal: the commit is the record)"
git log --oneline -2

echo
echo "end $(date '+%F %T')"
