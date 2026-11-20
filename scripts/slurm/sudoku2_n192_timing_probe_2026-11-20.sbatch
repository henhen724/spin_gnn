#!/bin/bash
#SBATCH --job-name=sud2N192Timing
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#
# PRICE N=192 ON THE GPU BEFORE BUYING IT. Not an experiment -- no claim is
# scored here and nothing from this job may be quoted as a result.
#
#   sbatch scripts/slurm/sudoku2_n192_timing_probe_2026-11-20.sbatch
#
# WHY N=192. Finding 44's N=96 sbatch pre-registered the next size rung in
# writing: "if the best cell again fails to cross 0.5 at rho=0.32 ... the next
# move is N=192, NOT another N=96 grid". H1 PASSED at N=96 (all six cells
# crossed inside the grid, finding 45), so N=96 is not too small for H1 -- but
# the FAST/SLOW question is a different one, and it is the one that needs size.
#
# WHAT THE SIZING PROBE ALREADY SETTLED (52ab5eb, CPU, fall=True, 40 paths):
#
#   N    rho_min        T_end      max_len   cost vs N=96 (linear..quadratic)
#   48   0.333+-0.016   13.5+-3.9      511   0.51x .. 0.26x
#   96   0.322+-0.019   22.4+-4.3      998   1.00x .. 1.00x
#  192   0.320+-0.013   34.6+-7.9     1976   1.98x .. 3.92x
#  384   0.314+-0.008   57.2+-12.7    3928   3.94x .. 15.49x
#
# Sequence length is 8N + T + 4 (the rho-dependence cancels), so N is the
# expensive axis and depth at fixed N is nearly free. N=192 doubles max_len
# against N=96, which is somewhere between 2.0x (position-wise terms) and
# 3.9x (attention) per epoch. At N=96 the L=12/240k/50ep headline cell cost
# 13.6 h, so N=192 is 27-53 h and the honest answer is "measure it" -- a 26 h
# walltime like the N=96 scan's may not even hold. THAT is what this job is
# for: it trains 3 epochs on 4800 instances and reports min/epoch, so the real
# grid gets a --time from a measurement rather than from an exponent guess.
#
# WHY THE FAST/SLOW QUESTION NEEDS A BIGGER N THAN THE SOLVE-RATE QUESTION DID.
# DD has two phases (finding 1+34, Dario's Sudoku_SC slides): T ~ log N above
# rho_c ~ 0.46, T ~ N^gamma below it, gamma -> ~0.727. The measured exponent
# between adjacent rungs of the fall ladder (fastslow.json) is
#   48->96  0.643   96->252  0.670   252->996  0.750   996->4140  0.690
# so the 48->96 interval reads 12% below asymptotic: the N=48/N=96 pair is
# too short a lever to call a power law. The grid table shows both phases
# plainly in the INSTANCES -- median T at rho=0.70 is 3/4/5/5/5/5 across
# N=48..384 (flat, log-like) against 13/23/27/32/36/51 at rho=0.33 (a clean
# power law, 8x in N giving 3.9x in T against 8^0.727 = 4.4).
#
# THE TENSION THIS JOB DOES NOT RESOLVE, STATED UP FRONT. The instances move
# INTO the slow phase as N grows, but the models have been moving OUT of it:
# best rho_50 went 0.460 at N=48 (finding 44) to 0.564 at N=96 (finding 45),
# i.e. away from rho_c ~ 0.46 and up into the fast phase. A model whose
# half-solve density sits at 0.56 never operates where T ~ N^0.727. So "train
# on larger instances until the fast/slow transition is visible" may make the
# ensemble span the transition while the model retreats from it. The N=192
# grid must therefore be read on the SHAPE of solve-vs-T, not only on rho_50's
# crossing -- and if rho_50 degrades again, that degradation is the result and
# gets reported as such rather than retried at a larger N.
#
# =====================================================================
# THE CELL. Identical to finding 45's N=96 headline cell except n_nodes,
# n_train and epochs -- the protocol change is size, as before.
#   N=192 d=5 q=4, --fall --shuffle-adj, L=12, d_model 256, nhead 4,
#   ff 1024, pre-norm, lr 1e-3 constant, batch 32, no warmup.
#
# WHAT IS CHECKED (all printed; none of it is a finding):
#   1. the generator runs end to end at N=192 with fall=True
#   2. max_len as BUILT vs the 1976 the sizing probe predicted
#   3. trainable parameter count, with max_len's positional table in it
#   4. peak GPU memory, against the 64G/1-GPU allocation
#   5. seconds per epoch at 4800 instances -> projected h for 240k x 50ep
#
# pipefail matters here: the trainer's output is piped through a timestamper,
# and without it a crash in training would be masked by the timestamper's
# clean exit and the job would report success.
set -euo pipefail

cd /home/users/hshunt/spin_gnn

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 \
    py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312

OUT=$SCRATCH/spin_gnn/runs/n192_timing_2026-11-20
mkdir -p "$OUT"

echo "=== commit at submit ==="
git log --oneline -1
echo "=== node / gpu ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Environment smoke check, same shape as the N=96 scan's: turn an environment
# or max_len failure into a 30-second job rather than a 2-hour one that dies
# in the eval. The margin assert is load-bearing -- at N=192 a flush
# positional table would truncate the deepest bins silently (finding 44's
# max_len = 414 against a 408-token trace, caught only in truncated_rate).
python3 - <<'PY'
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep  # noqa: F401
from spin_gnn.config import SudokuReasoningConfig, SudokuReasoningTrainConfig
from spin_gnn.train_sudoku_transformer import resolve_max_len_margin
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
m = resolve_max_len_margin(SudokuReasoningTrainConfig(),
                           SudokuReasoningConfig(n_nodes=192, degree=5, n_colors=4))
assert m == 386, f"expected a 2*192+2 generation margin, got {m}"
print("max_len margin", m)
PY

echo "=== N=192 timing probe: 3 epochs x 4800 instances ==="
echo "start $(date -Is)"
# Prefix every line with elapsed seconds so the epoch boundaries the trainer
# prints ("epoch N: train_nll=...") become a per-epoch rate without touching
# the trainer. python3 rather than awk's systime(), which is a gawk extension.
/usr/bin/time -v python3 -u -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 192 --degree 5 --n-colors 4 \
    --rho 0.30 --rho-max 0.95 \
    --fall --shuffle-adj \
    --arch-mode full --num-layers 12 --d-model 256 --nhead 4 \
    --dim-feedforward 1024 \
    --lr 1e-3 --lr-schedule constant --warmup-epochs 0 \
    --batch-size 32 --n-train-instances 4800 --n-val-instances 100 \
    --epochs 3 \
    --n-per-bin 20 --eval-seed-start 90000000000 \
    --seed 0 \
    --out-dir "$OUT" 2>&1 | python3 -u -c "
import sys, time
t0 = time.time()
for line in sys.stdin:
    sys.stdout.write('[%7.1fs] %s' % (time.time() - t0, line))
    sys.stdout.flush()
"
echo "end $(date -Is)"

echo "=== built max_len vs the 1976 the sizing probe predicted ==="
python3 - <<'PY'
import json, os
p = os.path.join(os.environ["SCRATCH"], "spin_gnn/runs/n192_timing_2026-11-20",
                 "config.json")
if os.path.exists(p):
    c = json.load(open(p))
    print("max_len", c.get("max_len"), "(predicted 1976)")
    print("n_nodes", c.get("n_nodes"), "fall", c.get("fall"),
          "shuffle_adj", c.get("shuffle_adj"))
    print("n_params", c.get("n_params"), "trainable", c.get("n_trainable"))
else:
    print("no config.json at", p)
PY

echo "=== done; out dir ==="
ls -la "$OUT"
