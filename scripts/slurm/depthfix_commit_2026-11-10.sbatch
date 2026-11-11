#!/bin/bash
#SBATCH --job-name=depthfixCommit
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/depthfixCommit_%j.out
#
# Commits the tied-depth fix, its tests, the revived test runner, the prepared
# (unsubmitted) S5 array, and the STATUS/LOG rewrite. A job and not a login node
# for the usual reason: $HOME I/O on sh02-ln02 wedges in cl_sync_io_wait and a
# commit once held a 0-byte .git/index.lock for 33 minutes. Never delete that
# lock, never kill the process -- fail closed and let it drain.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists; nothing staged or committed."
    ls -l .git/index.lock
    exit 1
fi

PATHS="
spin_gnn/sat_transformer2.py
spin_gnn/sudoku_activations.py
spin_gnn/sudoku_direction_ablate.py
spin_gnn/sudoku_sae_ablate.py
spin_gnn/sudoku_attention_distance.py
spin_gnn/sudoku_circuits.py
scripts/direction_ablation_rho.py
scripts/run_tests.py
scripts/slurm/run_tests.sbatch
scripts/slurm/sudoku2_n48_tiedwidth_S5warm_2026-11-11.sbatch
tests/test_tied_depth.py
tests/test_arch_modes.py
STATUS.md
LOG.md
scripts/slurm/depthfix_commit_2026-11-10.sbatch
"

# shellcheck disable=SC2086
git add $PATHS
echo "add rc=$?"
git diff --cached --name-only

if git diff --cached --quiet; then
    echo "nothing staged -- already committed"
    git log --oneline -1
    echo "end $(date '+%F %T')"
    exit 0
fi

# Verify every intended path actually landed in the index before committing.
# A silent `git add` miss (a typo, a .gitignore rule) is how a commit ends up
# describing files it does not contain.
missing=0
staged=$(git diff --cached --name-only)
for p in $PATHS; do
    if ! printf '%s\n' "$staged" | grep -qx -- "$p"; then
        echo "MISSING FROM INDEX: $p"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "FAILING CLOSED: not committing a partial set."
    exit 1
fi

git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
    commit -q -F - <<'MSG'
fix: index the block APPLICATION, not its slot in encoder.layers

A weight-tied stack holds one block object and applies it n_iters times, so
encoder.layers has length 1 whatever the depth. Five analysis modules read the
depth off that list and therefore described a depth-1 network while returning
plausible numbers and no error: ResidualCapture handed back a 2-row stack
(embedding + last round) for a depth-16 model, SubspaceAblation bounds-checked
against 1, sudoku_sae_ablate raised IndexError above layer 1 and fired its edit
in every round at layer 1, and sudoku_attention_distance -- which replicates
forward() by hand -- scored a different network entirely.

The rule now lives once, in sat_transformer2.py, next to the attributes it
reads: is_tied / model_depth / applied_blocks, plus ApplicationHooks, a context
manager that keys forward hooks by application index rather than module
identity (a forward_pre_hook resets the counter, so two passes do not drift).
sudoku_circuits' local copy of the same rule is deleted in favour of the
import; it was written there first for its own fix, and four more modules then
needed it, which is how a second copy becomes a second chance to be wrong.
Its weight-space head/sublayer ablations still refuse on tied, because zeroing
a weight cannot be made per-round.

tests/test_tied_depth.py (11 cases) pins both properties that failed silently:
the layer axis IS the depth, and an intervention named for round k lands on
round k alone. The strongest two compare against hand-applied loops rather than
against shapes -- ResidualCapture's rows must equal `h = blk(h, src_mask=mask)`
row by row, and forward_capture_attention's logits must match model(x) -- the
latter being the first check that the hand-rolled forward reproduces the real
one on either architecture.

Running the suite meant repairing the runner. run_tests.sbatch was dead code:
it sourced the long-purged $SCRATCH venv, set an inline PYTHONPATH that shadows
the py-* modules' own numpy/torch, and ran a `python -m pytest` that does not
exist for py312 here, so it exited before collecting anything. scripts/run_tests.py
drives the suite through its own _pytest_shim collector and treats an import
error as a failure rather than a skip. The suite is 27 files / 142 cases, and
that honesty immediately showed 10 files that have never been collectable on
this machine (module-level `import pytest`) -- including why
test_arch_modes::test_same_initialization_across_modes had been failing unseen:
it predates the tied mode and asserted a state_dict key equality a tied stack
cannot satisfy. Now asserted as full's keys minus blocks 1.., which keeps the
test's actual purpose (same init draw as full's block 0).

Also here, not yet submitted on purpose:
scripts/slurm/sudoku2_n48_tiedwidth_S5warm_2026-11-11.sbatch.
Diagnostic 51860479 printed its own pre-registered verdict, WARMUP
FIXES IT -- d448 seed 1 reaches 1.6228 train NLL / 0.4804 val accuracy by epoch
7 with a 5-epoch ramp, against 3.9563 / 0.0313 at warmup 0, and epochs 5-7 run
at the full 1e-3, so the ramp is not merely postponing the blow-up. That makes
stage S's two dead d448 seeds an optimization failure rather than a property of
the cell, and stage S's only discriminating arm unscoreable as it stands. The
replacement is a 6-cell one-knob array with d192 as the inertness control; it
stays unsubmitted because ~1.6 GPU-days is hshunt's call and its scorer is not
written, and the discipline is scorer first, array --dependency=afterok.
MSG
echo "commit rc=$?"
git log --oneline -1

module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$? (non-fatal: the commit is the record)"
echo "end $(date '+%F %T')"
