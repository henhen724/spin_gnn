#!/bin/bash
#SBATCH --job-name=statusCommitV
#SBATCH --partition=sganguli
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/statusCommitV_%j.out
#
# Append this session's LOG.md lines and commit the rewritten STATUS.md.
#
# Three dated lines, the standing cap, and nothing that is already a finding:
# the stage T web deck (a DELIVERY change, not a result), stage U's mid-run read
# (numbers that will be superseded by its scorer in two hours, so they belong in
# the log and not in FINDINGS.md), and stage V's registration.
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

cat >> LOG.md << 'LOG'

- 2026-11-11 — **Stage T shipped as a password-gated WEB deck**, the delivery `~/CLAUDE.md` now asks for (PDF is ask-only). Builder `notebooks/make_staget_web.py` (`WebDeck` mirrors the PDF page API; an `expect`/`savefig` hook stands in for `PdfPages` so the tokenization and trace pages work unchanged), 16 pages + 7 figures at `~/slides/decks/spin-gnn/`, rendered by `51957571`, one wrong bullet caught in the read-back and fixed by `51957848` (T\* was stated as being within 0.01 of itself), published by `51957914`. `tools/check-gate.sh spin-gnn` prints **gate holds** — including the test that matters, figures 401 without the cookie. Live at `https://slides-hshunt.pages.dev/spin-gnn/`.
- 2026-11-11 — **Stage U mid-run check at epoch 33/50 (array `51922710`): healthy, no epoch-0 divergence, 0 restarts.** Paired by seed, val token accuracy k=1 → k=2 → k=4: seed 0 0.4826/0.4826/0.4829, seed 1 0.4825/0.4838/0.4854, seed 2 0.4834/0.4864/0.4850 (medians 0.4826 → 0.4838 → 0.4850). Marginal and seed-dependent against the 0.0011 the gate needs — as the registration said in advance, accuracy cannot adjudicate stage U alone. Not a finding until `stageu_score` runs on the pooled eval.
- 2026-11-11 — **Stage V registered `ca11c2b` (job `51962711`), T=24 arm `51962715` queued `--dependency=afterok`.** Trains the shared block at a LONGER schedule: 613,818 trainable at T=16, 24 AND 32, so the one knob in this programme that costs no parameters. Motivated by the generator's own depth distribution (median 5 rounds, **range 1–19**) against 16 applications. V-a (val acc ≥ 0.4841 in ≥2/3) is the adjudicator and is read first; V-e closes the pair stage T set up but could not reach — frozen-at-24 vs *trained*-at-24, same count, same applications, same grid. Four routes closed on existing evidence before picking this one: width (saturated), cosine (finding 47), more data (`n_train_instances` is not in `rung_key`, so it would pool into stage S's rung), best-val (final-epoch checkpoints only).
LOG
echo "append rc=$?  LOG.md is now $(wc -l < LOG.md) lines"

git add STATUS.md LOG.md scripts/slurm/status_commit_2026-11-11b.sbatch
echo "add rc=$?"
git diff --cached --name-only | sed 's/^/  /'

if git diff --cached --quiet; then
    echo "nothing staged -- no commit made"
else
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - << 'MSG'
Hand off 2026-11-11: stage T on the web, stage U running, stage V registered

STATUS.md rewritten in place (under the 60-line cap) and three dated LOG.md
lines appended.

Stage T's delivery now matches ~/CLAUDE.md: a password-gated web deck at
slides-hshunt.pages.dev/spin-gnn/, with check-gate.sh confirming figures are 401
without the cookie. The PDF route is ask-only from here.

Stage U is healthy at epoch 33/50 with 0 restarts; its paired accuracy deltas are
marginal and seed-dependent, which is what the registration predicted would
happen and why the call falls to U0's 3/3 crossing rather than to accuracy.

Stage V is registered and its T=24 arm is queued behind the registration's
afterok. The T=32 arm is deliberately held until T=24 is read: sganguli is three
nodes, stage U holds six GPUs, and V-d needs both rungs but not concurrently.

STATUS.md also records the four routes now closed on evidence rather than taste
(width, cosine, more data, best-val selection), so the next session does not
re-derive them -- one of them, cosine, was most of a GPU-day the first time.
MSG
    echo "commit rc=$?"
fi

module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$? (non-fatal: the commit is the record)"
git log --oneline -2

echo
echo "end $(date '+%F %T')"
