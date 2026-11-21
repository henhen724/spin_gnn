#!/bin/bash
#SBATCH --job-name=sud2DQprobe
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#
# STAGE DQ TIMING PROBE -- the measurement that has to land before the real
# array is sized.
#
#   sbatch scripts/slurm/sudoku2_dq_probe_2026-11-21.sbatch
#
# Project convention is that walltime is MEASURED, NOT EXTRAPOLATED (the
# N=192 rung sized itself off job 53061369 the same way). This probe does
# that for the two candidate stage-DQ cells, and it is also the first time
# (d=4, q=3) has ever touched a GPU here: every cell to date is d=5, q=4, so
# the vocabulary (size N+9, node_offset 9 instead of 10) and the training
# loop have only been checked on CPU by scripts/stagedq_datacheck.py.
#
# It trains NOTHING that will be reported. 3 epochs x 4800 instances per
# size, purely to get s/epoch and the longest training sequence.
#
# WHY (d=4, q=3) AT ALL -- hshunt chose this lever on 2026-11-21 over the
# trace-format change and a fourth size rung.
#
# Finding 116: at N=192 the model's reason_color sits at ln 4 = 1.3863 nats
# with accuracy 1/q, at every density and on both seeds -- it never learns
# the colouring rule. At N<=96 it learns it essentially perfectly (0.0036
# nats, 0.9991). The open question is whether that collapse is driven by N
# or by the DD depth that comes with N. (d=4, q=3) separates them: it buys
# depth from the LOCAL parameters, so we can hold N at 96 -- where the rule
# is known to be learnable -- and still exceed the depth of any q=4 cell
# below N=192.
#
# THE 6x IN FINDING 10 DOES NOT SURVIVE AND THIS STAGE DOES NOT CLAIM IT.
# runs/stagedq_ensemble_2026-11-21/ measures both ensembles at both fall
# settings, 24 seeds per cell. Finding 10's "(d=4,q=3) at N=60 reaches mean
# depth 14.5, which (5,4) needs N~333 for" came from sweep 49509101, whose
# scorer never passes `fall`. Reproduced on that ensemble I get 5.00x in N
# and 5.48x in tokens, consistent with the published ~6x. On fall=True --
# the ensemble EVERY training cell uses -- the same measurement gives
# 1.58x in N and 1.77x in tokens. The fall constraint removes the nodes that
# can never be deduced, which is what caps depth near the floor, and it
# helps the q=4 d=5 reference far more than it helps q=3 d=4: at N=192
# turning fall on moves rho_min 0.4729 -> 0.3134 for the reference but only
# 0.1944 -> 0.1437 for the lever. FINDING 10 NEEDS AMENDING TO SAY WHICH
# ENSEMBLE ITS FACTOR IS MEASURED ON.
#
# 1.58x is still worth having, but it is not the reason to run this. The
# reason is the depth-vs-N separation above, and that is what the real
# array's predictions will be registered on.
#
# WHAT THE ENSEMBLE PRE-CHECK ESTABLISHED (runs/stagedq_ensemble_2026-11-21/)
#   * rho_c MOVES: ~0.465 at d5q4, ~0.25 at d4q3, read off where T/log N
#     stops collapsing across N=48..384. The same diagnostic recovers the
#     PUBLISHED 0.465 on the d5q4 arm, so it is calibrated, not fitted.
#   * rho_min moves too: ~0.314 at d5q4 N=192, ~0.150 at d4q3 N=96.
#   * NO q=4 CONSTANT CARRIES OVER. Not rho_c, not the free line 0.740, not
#     the val_nll floors 1.371/1.625/1.861, not s* 0.1484/0.119, not the
#     chance level (ln 3 = 1.0986, not ln 4 = 1.3863). Stage DQ measures its
#     own and quotes none of those.
#   * depth at matched N, deepest decile, from stagedq_datacheck:
#       d4q3 N=96   T=32.8   mean seq len 687
#       d4q3 N=120  T=35.0   mean seq len 856
#       d5q4 N=96   T=22.5   mean seq len 781   <- the matched control
#     So d4q3 N=96 is DEEPER than d5q4 N=96 while being SHORTER, and
#     d4q3 N=120 matches the depth of the d5q4 N=192 cell that failed, at
#     1.8x fewer tokens.
#
# PREFLIGHT ALREADY PASSED ON CPU (scripts/stagedq_datacheck.py, 40
# instances per cell): F1 every deduction forced by its prior-round
# neighbours 0 bad, F2 deduced colour == planted 0 bad, F3 forcing context
# already emitted 0 bad, F4 node tokens round-trip under the moved
# node_offset 0 bad, on d4q3 N=96, d4q3 N=120 and the d5q4 N=96 control.
# Measured marginal on reason_color 1.0985/1.0986 against ln 3 = 1.098612,
# so a trained d4q3 cell landing there is finding 116's failure repeating.
#
set -euo pipefail

cd /home/users/hshunt/spin_gnn

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 \
    py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312

echo "=== commit at submit ==="
git log --oneline -1
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PROBE=$SCRATCH/spin_gnn/sudoku_dq_probe_2026-11-21
mkdir -p "$PROBE"

echo
echo "=== sizing: max_len margin, vocab, legality for the new ensemble ==="
python3 - <<'PY'
import torch
from spin_gnn.config import SudokuReasoningConfig, SudokuReasoningTrainConfig
from spin_gnn.sudoku_tokens import Vocab
from spin_gnn.train_sudoku_transformer import resolve_max_len_margin
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
for n, d, q in ((96, 4, 3), (120, 4, 3), (96, 5, 4)):
    assert n % q == 0 and (d * n) % (q * (q - 1)) == 0, (n, d, q)
    m = resolve_max_len_margin(SudokuReasoningTrainConfig(),
                               SudokuReasoningConfig(n_nodes=n, degree=d,
                                                     n_colors=q))
    v = Vocab(n, q, d)
    print("N=%3d d=%d q=%d  margin %4d  vocab %4d  node_offset %2d"
          % (n, d, q, m, v.size, v.node_offset))
PY

for SPEC in "96:4:3:0.14" "120:4:3:0.14"; do
    N="${SPEC%%:*}"; REST="${SPEC#*:}"
    D="${REST%%:*}"; REST="${REST#*:}"
    Q="${REST%%:*}"; R0="${REST##*:}"
    echo
    echo "================= timing N=$N d=$D q=$Q rho0=$R0 ================="
    date +'start %F %T'
    python3 -u -m spin_gnn.sudoku_hparam_sweep \
        --n-nodes "$N" --degree "$D" --n-colors "$Q" --shuffle-adj --fall \
        --rho "$R0" --rho-max 0.95 \
        --d-model 256 --nhead 4 --dim-feedforward 1024 \
        --num-layers 12 --seed 0 \
        --n-train-instances 4800 --epochs 3 \
        --n-val-instances 300 --batch-size 32 --n-per-bin 20 \
        --out-dir "$PROBE/n${N}_d${D}_q${Q}"
    date +'end   %F %T'
done

echo
echo "=== READ THIS TO SIZE THE ARRAY ==="
echo "s/epoch at 4800 x (50 epochs x 240000/4800) = total per cell."
echo "Cell timings are the 'start'/'end' pairs above; the per-epoch line in"
echo "each sweep log is the number to scale, NOT the wall span, which"
echo "includes one-off instance generation and the eval."
