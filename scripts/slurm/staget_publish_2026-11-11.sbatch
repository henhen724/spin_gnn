#!/bin/bash
#SBATCH --job-name=stagetPub
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagetPub_%j.out
#
# PUBLISH the stage T web deck. This is the deploy: the helper mints the password
# on first publish, builds locally, commits and pushes ~/slides, and the push is
# what Cloudflare Pages renders. It also sends the ntfy alert with the link and
# the password, which is the only thing that tells hshunt a deck exists.
#
# Run AFTER the read-back, never with it. deck.md and all seven figures were read
# back off render 51957571, which found one bullet wrong (T* stated as being
# within 0.01 of itself), and again off 51957848 after the fix.
#
# In a job because the helper writes and pushes from $HOME, where sh02-ln02's
# I/O wedges. It loads its own git and python modules, so this script loads none.
#
# Not idempotent in one respect worth knowing: re-running is safe and KEEPS the
# existing password (newpass.py refuses to overwrite without --rotate), so a link
# already shared keeps working. It will simply send a second ntfy alert.

set -u
cd /home/users/hshunt/slides || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

deck=/home/users/hshunt/slides/decks/spin-gnn/deck.md
if [ ! -f "$deck" ]; then
    echo "FAILING CLOSED: no deck at $deck"
    exit 1
fi
echo "deck pages: $(( $(grep -c '^---$' "$deck") - 1 ))"
echo "figures: $(ls -1 /home/users/hshunt/slides/decks/spin-gnn/figures/ | wc -l)"

echo
~/.local/bin/claude-slides-publish.sh spin-gnn \
    "stage T sweep of applications T=4..24 on frozen tied d320 weights, 3 seeds" \
    "rho_50 bottoms at the trained T=16 (0.757) and rises to 0.819 at T=24"
rc=$?
echo "publish rc=$rc"

echo
echo "end $(date '+%F %T')"
exit "$rc"
