#!/bin/bash
#
# STAGE Z. Is the determinacy signal stages X and Y installed the PREDICATE, or
# a COUNT that correlates with it?
#
# Registered in scripts/stagez_score.py, committed before this job was
# submitted -- not inside it, so `git log` against this job's SubmitTime shows
# the bars and the prediction were fixed first.
#
# WHY. X and Y each installed a determinacy readout at AUROC ~= 0.95 and
# neither produced a guard: stage Y's fired once in 681 cases. Two
# explanations are already refuted (the readout carrying two objectives, a
# monotone gate that can only suppress). The third has never been tested, and
# it comes from how sudoku_skip's corruption is built. The corruption
# RECOLOURS a revealed neighbour w of v, so
#
#     n_known_nbrs(v)    INVARIANT     -- w is revealed before and after
#     n_known_colors(v)  q-1 -> q-2    -- which is what destroys forcing
#
# A readout tracking the plain count cannot see the edit, however it is coupled
# to the emission. That explains both null results at once, for a reason no
# amount of re-coupling can repair.
#
# This was already visible in stage Y's own stored CSVs, which is what prompted
# the job: against the placebo's noise floor, the corruption's effect on the
# deciding logit is a coin flip. P(|corrupt drop| > |placebo drop|) is
# 0.489/0.500/0.483 for stage Y and 0.497/0.494/0.603 for stage X, against
# chance 0.500, and skip_summary.json has clean == corrupt == placebo ==
# 0.6743 exactly, inside every round0_rank stratum. Stage Z asks whether the
# REPRESENTATION is equally blind, and whether a count explains the AUROC.
#
#   Z1  the channel's own score g[p, v] on clean / corrupted / placebo inputs,
#       on stage Y's same 350 cases. Stage Y only: return_determinacy=True
#       raises without a channel, and stage X's "representation" is the node
#       logit, already measured above.
#   Z2  the AUROC of the model-free count n_known_nbrs over exactly the cells
#       determinacy_targets scores. No model loaded.
#   Z3  the same AUROC STRATIFIED by n_known_nbrs, added after the first run
#       (52413267) returned the unregistered outcome: the channel does not
#       respond at all (win rate 0.4914/0.5171/0.4571, flip fraction 0.0000
#       over 236/228/217) and yet the plain count tops out at 0.8857 against
#       the model's 0.9493, so the model knows something past the count. Z3
#       measures how much on a scale whose ends are fixed by construction --
#       n_known_nbrs is constant inside its own stratum so it scores exactly
#       0.5, n_known_colors is the predicate's threshold so it scores exactly
#       1.0, and both are ASSERTED. Registered before this run as check_z3.
#
# This run SUPERSEDES 52413267 and writes the same directory. Z1 and Z2 are
# recomputed identically from the same streams; "stratified" is the new key.
#
# NOTHING IS TRAINED. Three checkpoints are read; no weight is written. There
# is no fine-tune path in this repo in any case -- load_checkpoint restores
# weights only and calls model.eval().
#
# Aggregated into ONE job per Rule 10: three Z1 cells, the Z2 baseline and the
# scoring, rather than five short submissions. It writes only to
# runs/2026-11-15_stagez_cfresp/ and commits only the two new source files.
#
#SBATCH --job-name=stagezCfresp
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache

OUT=/scratch/users/hshunt/spin_gnn/runs/2026-11-15_stagez_cfresp
Y=/scratch/users/hshunt/spin_gnn/sudoku_n48_determinacy_Y_2026-11-14b
G=/scratch/users/hshunt/spin_gnn/runs/2026-11-14b_stagey_guard/guard
mkdir -p "$OUT"

echo
echo "=== 1. compile and the registered scorer's self-test ============="
python3 -m py_compile spin_gnn/sudoku_cfresp.py scripts/stagez_score.py || {
    echo "py_compile FAILED"; exit 1; }
python3 scripts/stagez_score.py --self-test || { echo "self-test FAILED"; exit 1; }
echo "registration commit: $(git log -1 --format=%h -- scripts/stagez_score.py)"

echo
echo "=== 2. Z1 + Z2 ==================================================="
# --skip-dir makes the job FAIL if the rebuilt case list is not the published
# one: the behavioural columns here are the same quantities sudoku_skip wrote,
# so they have to agree to the last digit or the comparison is meaningless.
python3 -u -m spin_gnn.sudoku_cfresp \
    --checkpoint \
      "$Y/t00_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s0/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_chan" \
      "$Y/t01_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s1/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_chan_s01" \
      "$Y/t02_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s2/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_chan_s02" \
    --skip-dir "$G/stagey_s0" "$G/stagey_s1" "$G/stagey_s2" \
    --n-instances 400 --n-cases 350 --seed-start 600000000 \
    --auroc-n-instances 600 --auroc-seed-start 70000000000 \
    --out-dir "$OUT"
RC=$?
echo "probe rc=$RC"
[ $RC -ne 0 ] && { echo "probe FAILED -- nothing scored, nothing committed"; exit $RC; }

echo
echo "=== 3. score against the pre-registered bars ====================="
python3 scripts/stagez_score.py "$OUT" | tee "$OUT/stagez_verdict.json"

echo
echo "=== 4. commit ===================================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    exit 1
fi
# runs/ is gitignored and stays that way -- results live on $SCRATCH and in the
# deck, never in the repo. Only source is committed here.
git add spin_gnn/sudoku_cfresp.py scripts/stagez_score.py \
        scripts/slurm/stagez_cfresp_2026-11-15.sbatch
git diff --cached --name-only | sed 's/^/  staged /'
if ! git diff --cached --quiet; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
Stage Z: does the installed determinacy signal respond to the counterfactual?

Two measurements on existing checkpoints, no training. The registration, with
the bars and the prediction, is scripts/stagez_score.py and was committed
before this job was submitted.

The question is the one X and Y both leave open. Each installed a determinacy
readout at AUROC ~= 0.95 and neither produced a guard. The surviving
explanation is that the signal is a CORRELATE of determinacy rather than the
predicate, and sudoku_skip's corruption is built in exactly the way that
separates them: it recolours a revealed neighbour, so the count of revealed
neighbours is invariant across the edit while the distinct-colour count, which
is what forcing depends on, drops from q-1 to q-2.

Z1 reads stage Y's own channel score g[p, v] on the clean, corrupted and
placebo inputs over the same 350 cases, and reports the win rate
P(|dg_corrupt| > |dg_placebo|) whose chance value is 0.50 because both edits
are one-token hint recolours. Z2 computes the AUROC of the model-free count
n_known_nbrs over exactly the cells determinacy_targets scores, with no model
loaded at all.

Two things the module is built to prevent. _covariates reproduces
determinacy_targets' loop line for line and asserts its recomputed target
equals the module's -- the known set there is round-start plus the deductions
already emitted this round, which is NOT sudoku_skip._revealed_before's
prefix-exact convention, and covariates on the wrong convention would be
scored against labels from the other one and compare silently wrong. And
--skip-dir cross-checks the rebuilt case list against the published
skip_summary.json emit rates, so a drifted stream fails the job instead of
producing a number that looks fine.

The prompt for the job was in stage Y's own stored CSVs: against the placebo's
noise floor the corruption's effect on the deciding logit is a coin flip, at
0.489/0.500/0.483 for Y and 0.497/0.494/0.603 for X.

Z3 was added after the first run (52413267) returned the registered-but-
unexpected third outcome, "neither predicate nor count". The channel does not
respond to the corruption at all -- win rate 0.4914/0.5171/0.4571 against
chance 0.500, median |dg| 0.0103 under the corruption versus 0.0102 under the
placebo against a spread of 3.5, and flip fraction 0.0000 over 236/228/217 --
and yet the plain count reaches only 0.8857 against the model's 0.9493. So the
mechanism this stage was written to test is not the whole story, and Z3 asks
what the remaining 0.064 of AUROC is by conditioning on the count.

The stratified scale calibrates itself at both ends and the probe asserts
both: n_known_nbrs is constant inside its own stratum so every pair ties and
it scores exactly 0.5, and n_known_colors is the predicate's own threshold so
it scores exactly 1.0. A mis-stratified run therefore fails the job instead of
reporting a plausible middle value. The pooled AUROC is reported alongside so
it can be checked against the published 0.9493/0.9478/0.9480, which is the
proof that Z3 reads the same cells.
MSG
    echo "commit rc=$?"
fi
echo "commit: $(git rev-parse --short HEAD)"

echo
echo "=== 5. push ======================================================"
module load system git/2.45.1
git push origin sudoku-dd
echo "push rc=$?"

echo "end $(date '+%F %T')"
exit $RC
