#!/bin/bash
#SBATCH --job-name=deckRender
#SBATCH --partition=sganguli
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/deckRender_%j.out
#
# Commit the two new deck builders, then render whichever decks are asked for on
# the command line. hshunt asked for two decks aimed at someone who has not been
# following: the fast/slow transition (spin-gnn-phases) and the simplest
# architectures that still learn direct deduction (spin-gnn-minimal).
#
#   sbatch scripts/slurm/decks_render_2026-11-12.sbatch minimal
#   sbatch scripts/slurm/decks_render_2026-11-12.sbatch phases
#   sbatch scripts/slurm/decks_render_2026-11-12.sbatch minimal phases
#
# NEITHER TOUCHES decks/spin-gnn, the live stage-U deck. New slugs, and
# claude-slides-publish.sh stages only decks/<project>.
#
# In a job because both builders import torch through deck_tokenization (for the
# real instance the tokenization and trace pages show) and because $HOME I/O
# from sh02-ln02 wedges. The commit runs first so the provenance line on every
# page names a sha that actually contains the builder that drew it -- 52027787
# published a deck crediting the wrong sha and the correction cost its own job.
#
# Rendering only. Publishing is deliberately NOT here: every page gets read back
# as a PNG, twice, before it goes out.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID  decks=$*"
echo "start $(date '+%F %T')"

if [ "$#" -eq 0 ]; then
    echo "usage: sbatch $0 <minimal|phases> ..."; exit 1
fi

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache

for b in notebooks/make_minarch_web.py notebooks/make_phases_web.py \
         scripts/fastslow_measure.py; do
    python3 -m py_compile "$b" || { echo "py_compile FAILED: $b"; exit 1; }
done

# -- commit the builders, so --commit below names real code
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    exit 1
fi
git add notebooks/make_minarch_web.py notebooks/make_phases_web.py \
        scripts/fastslow_measure.py scripts/slurm/fastslow_measure_2026-11-12.sbatch \
        scripts/slurm/decks_render_2026-11-12.sbatch
git diff --cached --name-only | sed 's/^/  staged /'
if ! git diff --cached --quiet; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -m "Two newcomer-facing decks: the fast/slow transition, and the architecture ladder

make_phases_web.py draws the phases deck from a measurement taken today
(scripts/fastslow_measure.py) rather than from finding 43's archived scalars,
whose per-size arrays were written by a \$SCRATCH script that no longer exists.
The task side is measured; the model side is read from the two grids scored
11-02 and never differenced, since they are not paired and the default cell had
twice the epochs.

make_minarch_web.py draws the architecture ladder through the shared scorer's
own loaders over all fifteen N=48 scan directories, so a rung in the deck is the
rung the stage scorers printed. N=96 dirs and the eval-T sweep are excluded by
construction. Both builders assert every printed sentence before writing a page."
    echo "commit rc=$?"
fi
SHA=$(git rev-parse --short HEAD)
echo "builder commit: $SHA"
echo

RC=0
for deck in "$@"; do
    case "$deck" in
      minimal)
        echo "=== spin-gnn-minimal ==="
        python3 -u -m notebooks.make_minarch_web --commit "$SHA" || RC=1 ;;
      phases)
        echo "=== spin-gnn-phases ==="
        python3 -u -m notebooks.make_phases_web --commit "$SHA" || RC=1 ;;
      *) echo "unknown deck: $deck"; RC=1 ;;
    esac
    echo
done

echo "render rc=$RC"
echo "end $(date '+%F %T')"
exit $RC
