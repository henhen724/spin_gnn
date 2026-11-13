#!/bin/bash
#SBATCH --job-name=ddgRegister
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/ddgRegister_%j.out
#
# REGISTER THE FIRE-AND-GUARD PROBE BEFORE ITS FIRST EVAL CAN START.
#
# scripts/slurm/ddguard_eval_2026-11-13.sbatch is submitted --dependency=afterok
# on THIS job, so both steps fail closed:
#
#   1. scripts/ddguard_score.py --register must exit 0. That is guard 1: all six
#      checkpoints exist, each says the arch_mode / d_model / num_layers /
#      n_trainable_params / seed of the cell it is filed under, and the two arms
#      agree on every instance-generating key. If they did not agree, the case
#      lists would differ and the whole probe would be unpaired -- so the check
#      that makes the design valid runs before any GPU is spent.
#   2. the commit must happen. The docstring of scripts/ddguard_score.py IS the
#      registration -- four hypotheses written as the expressions that will be
#      computed, their bars, their falsifiers, and the caveats -- so it is in git
#      before any number exists to tempt it. Finding 39 is why: a registration
#      written as a sentence plus a remembered number failed as written even
#      though the science replicated.
#
# NOTHING IS TRAINED by this job or by the eval it gates. Six existing
# checkpoints are re-read. No knob moves and no published number can change.
#
# From a compute node, like every git step on this project: $HOME I/O on the
# login nodes wedges intermittently in cl_sync_io_wait. The commit refuses to run
# if a .git/index.lock exists and never deletes it -- the lock belongs to
# whichever process holds it, and ~/slides and this repo are both shared.

set -uo pipefail
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
date +'start %F %T'

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache
export PYTHONUNBUFFERED=1

echo
echo "=== 1. compile ===================================================="
python3 -m py_compile scripts/ddguard_score.py || { echo "py_compile FAILED"; exit 1; }
echo "compile: PASS"

echo
echo "=== 2. guard 1: the six checkpoints are the cells they claim ======"
python3 -u scripts/ddguard_score.py --register || {
    echo "GUARD 1 FAILED -- the eval's afterok is never satisfied and no GPU is spent"
    exit 1; }

echo
echo "=== 3. commit the registration ===================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    exit 1
fi
git add scripts/ddguard_score.py \
        scripts/slurm/ddguard_register_2026-11-13.sbatch \
        scripts/slurm/ddguard_eval_2026-11-13.sbatch
git diff --cached --name-only | sed 's/^/  staged /'
if ! git diff --cached --quiet; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
Register the fire-and-guard probe: does one shared block run DD's GUARD too?

Not a stage. Nothing is trained, six existing checkpoints are re-read, no knob
moves and no published number can change. Every registered knob is closed and
this reopens none of them.

The brief is the smallest or most analytically tractable architecture that still
LEARNS DIRECT DEDUCTION, and rho_50 is a behavioural proxy for the second half of
that sentence. Findings 27/28 built causal tests that ask it directly and neither
has ever been run on a tied model. S1 -- tied_no_mlp T=16 d320, 613,818
trainable, one weight matrix set to read instead of sixteen -- is the most
analytically tractable object this project has, and it misses the free line by
0.017 of rho_50.

Two measurements on the same round-0 cases, both from findings 27/28's own
modules, unchanged:

  rule  = #{clean_correct and corrupt_flipped} / #{clean_correct}
          recolour the binding hint so DD's answer flips c -> a; does the
          prediction follow? (spin_gnn/sudoku_circuits.py, behaviour only)
  guard = #{clean_emits_v and corrupt_emits_v} / #{clean_emits_v}
          recolour it so v stops being forced at all; does the model still name
          v? LOW is good. (spin_gnn/sudoku_skip.py)

The guard is the discriminating one and that is not a guess. Finding 28 is the
only test in this project that has ever separated two models with the same rule
score: arms A and B both flip at 1.000, and the 0.146 of solve_rate between them
lived entirely in the guard (0.000 vs 0.176, McNemar p = 1.14e-13). S1 against J2
is the comparable situation.

It is also the cheap half of candidate (b), CHANGE THE MEASUREMENT. Finding 67
put the old noise floor at sigma_d 0.01412 against an effect of 0.0011 -- 2x to
22x the effect, which is why S through W all read "within noise". These two
statistics are PAIRED PER CASE on byte-identical case lists, so the arms differ
in the network and in nothing else. A different instrument, not a different arm,
and one short job instead of GPU-weeks.

Every instance-generating key is identical in the two config.json files
(n_nodes 48, degree 5, n_colors 4, rho 0.3, rho_max 0.95, hint_mode removal,
shuffle_adj True, shuffle_events False, fall True, scratchpad False), both evals
run at the same --seed-start and --n-instances, and build_instances plus case
selection are pure functions of those and np.random.default_rng(0). Guard 2
asserts the resulting case lists match row for row rather than trusting it, and
exits non-zero if they do not, because an unpaired McNemar is worse than no
McNemar.

H2's effect size is deliberately NOT barred. Finding 28's 0.176 was measured
against a 0.146 solve_rate difference; here the behavioural difference is 0.177
of rho_50, a different quantity on a different scale, and there is no honest
conversion. The bar is on detectability -- two of three seed pairs significant
at 0.05 after Holm, with the discordance in the predicted direction -- and the
median paired difference is reported as a description.

The registered falsifier is real and finding 28 says so: its LEAKED arm, which
flips at 0.008 and never learned the rule, still withheld at 0.008. If the guard
is that cheap to learn, both arms score ~0 here, H2 is FALSIFIED, and the result
is that the guard is NOT what tying costs. That closes a door and is worth the
job on its own.

Also corrects STATUS.md, which said the guard third of findings 27/28 has no
module in the tree. It does: spin_gnn/sudoku_skip.py, which is what finding 28
was measured with. The line was stale.
MSG
    echo "commit rc=$?"
fi
git rev-parse --short HEAD | sed 's/^/registration commit: /'

date +'end %F %T'
