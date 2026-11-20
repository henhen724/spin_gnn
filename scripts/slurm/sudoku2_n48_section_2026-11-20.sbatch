#!/bin/bash
#SBATCH --job-name=sect110
#SBATCH --output=/home/users/hshunt/spin_gnn/sect110_%A_%a.out
#SBATCH --partition=sganguli
#SBATCH --array=0-7%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#
# STAGE SECTION -- what fraction of `val_nll` is deduction, per section, on the
# 79 existing N=48 checkpoints.  INFERENCE ONLY: nothing is trained, no
# checkpoint is written, no closed knob is touched, no GPU is requested.
#
# WHY NOW.  Finding 109 priced the sequence analytically and found the graph
# section is 72-73% of tokens at every N while the reasoning section is ~19%,
# so ~81% of `val_nll` is instance transmission.  The standing non-negotiable
# ranks architectures on "validation token accuracy FIRST" and
# scripts/acc_vs_solve.py built the minimality frame on exactly that number.
# If the ranking metric is three-quarters graph slots then the standing
# architecture ranking may be ordering the wrong thing, and the restated brief
# -- the smallest or most analytically tractable architecture that still learns
# DIRECT DEDUCTION -- makes that question the live one.  The trainer has no
# per-section metric to answer it with and never has.
#
# SCOPE, AND THE JUDGEMENT IS MINE AND IS FLAGGED RATHER THAN HIDDEN.  Like
# stage HEADS I judged this inside standing scope: it trains nothing, touches
# no closed knob, and is orthogonal to the four registered questions waiting on
# hshunt.  It does not even load attention.  2 h x 8 CPU tasks, priced by
# measurement: 95 s per run on a login-node timing of 1 run x 3 densities x 25
# instances (18.9 s including ~8 s of import and model load), so ~16 min/task.
#
# ---------------------------------------------------------------------------
# WHAT THE SMOKE ALREADY SHOWED, DISCLOSED IN FULL BEFORE REGISTERING.
# Stage HEADS disclosed H1's sign from a 3-instance smoke; this is the same
# move and the bands below are set AROUND what was seen, not against it.  One
# run (r00, `L=3 h=4 full`, 2,530,618 params, rho_50 0.758 -- it FAILS the
# gate), 4 instances, rho 0.90:
#
#   section            ntok      nll      acc
#   hint_node           172   0.4088   0.9128
#   hint_color          172   1.3426   0.3140     (log 4 = 1.3863)
#   graph_lead          192   0.0000   1.0000
#   graph_nbr_novel     480   3.4835   0.0354     (log 48 = 3.8712)
#   graph_nbr_copy      480   0.9810   0.5521
#   round_mark            4   0.0000   1.0000
#   reason_node          20   0.1966   0.9000
#   reason_color         20   0.0000   1.0000
#   term                  4   0.0000   1.0000
#   begin                 8   0.0000   1.0000
#   TOTAL nll 1.5774, acc 0.4762 over 1552 tokens; E1 0 bad; E2 exact.
#
# So the DIRECTION of S1, S2, S5 and E3 is already near-certain and is
# registered as such.  What is live is every number that one run cannot give:
# the distribution over 79 architectures, the COUNTS, the gate separation, and
# above all S3 -- whether the section metric ORDERS architectures better than
# the total, on which a single run says exactly nothing.
# ---------------------------------------------------------------------------
#
# REGISTERED PREDICTIONS.  Scored by scripts/stagesection_fit.py, committed in
# this same commit, before the probe ran.
#
# S1. The deduction share of the loss -- (nll_sum[reason_node] +
#     nll_sum[reason_color]) / total -- has median BELOW 0.02 at rho 0.90 and
#     BELOW 0.10 at rho 0.60.  FALSIFIER: median above 0.25 at any density.
#     Rationale: at rho 0.90 only ~5 cells are blank and the smoke puts the
#     share at 0.0016; at rho 0.60 there are ~19 and both reason_node's error
#     rate (94, 98) and the section's token count rise, so the rho 0.60 half
#     is the live one and is banded five times looser.
#
# S2. reason_color accuracy >= 0.99 on at least 70 of 79 runs at every density.
#     FALSIFIER: fewer than 50.  Rationale: 103 found the colour step exact in
#     77 of 79 runs by free generation; this is the teacher-forced token-level
#     version of the same claim, and the smoke hits 1.0000 on a run that FAILS
#     the gate.  The count is what is live -- 103's two exceptions were the two
#     smallest architectures, so a count of 77 would replicate and 79 would not.
#
# S3. THE DECISION-RELEVANT ONE, AND NOTHING SEEN SO FAR BEARS ON IT.
#     |Spearman(reason_node accuracy, rho50_pub)| >= |Spearman(total accuracy,
#     rho50_pub)| - 0.05 over the 79 runs at rho 0.60.  FALSIFIER: the section
#     metric is weaker by more than 0.05, i.e. the total -- despite being ~81%
#     graph slots -- orders competence BETTER than the deduction tokens do.
#     That outcome is live and would be the more interesting one: it would mean
#     the graph section is a proxy for competence rather than a dilutant, and
#     that acc_vs_solve.py's frame survives 109 intact.
#
# S4. The gate separates on the section metric and not on the total: at
#     rho 0.60, median(reason_node acc | gate pass) - median(... | fail) >=
#     0.05 while |the same difference on total accuracy| <= 0.02.
#     FALSIFIER: either half reversed in sign.
#
# S5. graph_nbr_novel NLL sits within 0.5 nats of log 48 = 3.8712 on the median
#     run -- the model is near the information-theoretic ceiling on the section
#     that dominates the loss, which is WHY total `val_nll` is pinned and rises
#     with N.  FALSIFIER: median below 2.5, which would mean novel slots are far
#     more predictable than 109's closed form assumes and would put 109's floor
#     (1.371 / 1.625 / 1.861) in question.
#
# DIAGNOSTICS, and every S above is VOID if one fails.
# E1. The section label walk and encode_instance agree on length for every
#     instance of every run: 0 bad.
# E2. The section NLLs reconstruct the published total to 1e-6 on every cell --
#     this is what makes the decomposition a partition and not an estimate.
# E3. graph_lead, round_mark, term and begin are at accuracy 1.0000 and nll
#     0.0000 on every run: a graph row leader is determined by sorted node
#     order and the markers are grammatically forced, so anything else is a
#     labelling bug, not a finding.  Confirmed on the smoke; registered so a
#     failure anywhere in the 79 is caught rather than absorbed.
#
# WHAT THIS CANNOT SHOW.  Teacher-forced accuracy is not solve rate -- stage
# POS's P3 already found teacher-forced accuracy flat in position while free
# generation fails, and 98 showed the two regimes score different events.  A
# section metric that orders architectures is a better RANKING instrument; it
# is not evidence about the mechanism, and nothing here may be reported as
# clearing or moving the 0.740 gate.
#
set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 \
    py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312
cd /home/users/hshunt/spin_gnn
python3 scripts/stagesection_probe.py \
    --task "${SLURM_ARRAY_TASK_ID}" --ntasks 8
