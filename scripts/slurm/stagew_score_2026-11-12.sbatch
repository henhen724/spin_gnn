#!/bin/bash
#SBATCH --job-name=stagewScore
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagewScore_%j.out
#
# Score stage W the moment its array lands, as a DEPENDENCY rather than a poll
# (--dependency=afterok:52053221; slurm.md prefers dependencies to polling, and
# /etc/agents/AGENTS.md rule 10 forbids the loop).
#
# WHY THIS EXISTS AS ITS OWN JOB. Stage V was scored interactively and its
# output was never written to disk -- grepping "STAGE V --" across the project's
# .out files finds only the registration. The scorer then had to be re-run
# inside the deck job to manufacture a record after the fact. This job is that
# record, written before anyone reads a number: one file, one timestamp, one
# checkout, produced whether or not an agent session is awake when the cells
# finish.
#
# NO GPU: the scorer reads the per-cell rho_curve CSVs and history.json only.
# It takes no flags -- its --scan-dir default is [SCAN_W] + BASELINE_DIRS +
# SCAN_DIRS, and guard 3 inside it asserts SCAN_W never leaks into the pooled
# yardstick. It also runs score_sufficiency() itself, so the accuracy-to-solve
# fit that stages S4/U4/V needed is in this same output.
#
# READ W-a FIRST. Registered bar: final val token accuracy >= 0.4841 in at least
# 2 of 3 seeds -- the control's 0.4830 plus the 0.0011 the gate needs, IMPORTED
# from stagev_score rather than retyped. If W-a misses, the budget route is dead
# on the statistic that cannot censor, and no reading of rho_50 rescues it.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache

echo
echo "=== 1. what the array actually did ==============================="
sacct -j 52053221 -X -o JobID,JobName%16,State,Elapsed,ExitCode,NodeList
echo
for t in 0 1 2; do
    f="/scratch/users/hshunt/spin_gnn/sud2N48BudgetW_52053221_${t}.out"
    echo "task ${t} final epoch: $(grep '^epoch ' "$f" | tail -n 1)"
done
echo "control, stage S 50 ep, for the pair: train_nll 1.5851/1.5915/1.5864, acc 0.4830, rho_50 0.757 (0.754/0.757/0.765), mean_solve 0.3148"

echo
echo "=== 2. the registered scorer ====================================="
python3 -u -m scripts.stagew_score
RC=$?
echo "scorer rc=$RC"

echo
echo "=== 3. efficiency, for the next budget estimate =================="
seff 52053221_0 2>/dev/null | head -12

echo "end $(date '+%F %T')"
exit $RC
