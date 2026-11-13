#!/usr/bin/env python3
"""How many paired replicates would it take to read stage S's 0.0011?

WHY THIS EXISTS. Stages S, T, U, V and W each moved one knob and each came back
"within noise". That phrase was doing a lot of work without ever being given a
number, and finding 66b supplies the number for free: the 100-epoch cells of
stage W and their 50-epoch controls are independent RUNS of one configuration
(identical config but for the epoch count, identical data -- the stored baseline
accuracy agrees to 16 digits), so the difference between them at the control's
last epoch is a draw from the run-to-run noise with NO knob in it at all.

Three seeds is three such draws. That is enough to estimate a standard deviation
badly, and estimating it badly is still better than the phrase "within noise".

WHAT IS BEING RESOLVED. Stage S4 put all three seeds on the N=48
accuracy-to-solve curve, which converts the 0.017 of rho_50 still separating S1
from the 0.740 gate into about 0.0011 of validation token accuracy. That 0.0011
is the effect size every stage since S has been trying to measure.

NO NEW DATA. The pairing, the log parse and the epoch convention all come from
scripts.step_epoch_census, the module that registered them. MID_EP is the
control's last epoch, 0-based the way the training log numbers epochs, which is
the same convention the stage-W deck prints.

Run it anywhere -- stdlib only, so the login node's python3 is enough:
    python3 scripts/noise_floor.py
"""

import os
import sys
from statistics import mean, stdev

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.step_epoch_census import ROOT, config_of, history

W_LOGS = "sud2N48BudgetW_52053221_%d.out"
CONTROL_LOGS = "sud2N48TiedWidthS_51853969_%d.out"

# The control ran 50 epochs; the log numbers them from 0, so its last is 49.
MID_EP = 49
# What stage S is still short of the gate by, in validation token accuracy
# (S4's accuracy-to-solve curve converts 0.017 of rho_50 into this).
NEED = 0.0011
# Two-sided alpha 0.05 at 80% power, normal approximation: (1.96 + 0.8416)^2.
# A t-based number would be larger, so this is the optimistic side of the
# calculation and the conclusion below only gets stronger.
Z2 = (1.959964 + 0.841621) ** 2
# One 50-epoch cell of S1, measured: 474 min from the stage S array.
HOURS_PER_REPLICATE = 474 / 60.0


def by_seed(pattern):
    out = {}
    for task in (0, 1, 2):
        path = os.path.join(ROOT, pattern % task)
        if not os.path.exists(path):
            continue
        seed = config_of(path).get("seed")
        if seed is None:
            raise AssertionError("no seed= line in %s" % path)
        out[seed] = history(path)
    return out


def at(hist, epoch):
    for e, nll, acc in hist:
        if e == epoch:
            return nll, acc
    raise AssertionError("no epoch %d in a history of %d rows" % (epoch, len(hist)))


def required_n(sd):
    """Paired replicates per arm to resolve NEED at 80% power."""
    return Z2 * (sd / NEED) ** 2


def main():
    w, c = by_seed(W_LOGS), by_seed(CONTROL_LOGS)
    seeds = sorted(set(w) & set(c))
    if not seeds:
        raise SystemExit("no seed appears in both arrays")

    print("Two runs of ONE configuration, read at the control's last epoch "
          "(epoch %d)." % MID_EP)
    print("The epoch count is the only thing that differs, and at epoch %d "
          "neither arm" % MID_EP)
    print("has used it yet, so these differences contain no knob.")
    print()
    print("  seed |   arm nll |  ctrl nll |   d nll   |  arm acc |  ctrl acc |  d acc")
    dn, da = [], []
    for s in seeds:
        wn, wa = at(w[s], MID_EP)
        cn, ca = at(c[s], MID_EP)
        dn.append(wn - cn)
        da.append(wa - ca)
        print("  %4d | %9.4f | %9.4f | %+9.4f | %8.4f | %9.4f | %+.4f"
              % (s, wn, cn, wn - cn, wa, ca, wa - ca))

    print()
    print("paired accuracy difference: mean %+.4f, sd %.4f over %d seeds"
          % (mean(da), stdev(da), len(da)))
    print("paired train NLL difference: mean %+.4f, sd %.4f"
          % (mean(dn), stdev(dn)))
    print()
    print("Every single pair is larger in magnitude than the %.4f being chased."
          % NEED)
    print()

    # Two readings of the same three numbers. The honest one uses all three
    # seeds. The optimistic one drops the seed whose control stepped into a
    # basin and whose twin did not -- which is not a justified exclusion, only a
    # lower bound on how good the instrument could possibly be.
    full = stdev(da)
    tight = [d for s, d in zip(seeds, da) if s != 2]
    for label, sd in (("all %d seeds" % len(da), full),
                      ("dropping seed 2 (its control stepped, its twin did not)",
                       stdev(tight) if len(tight) > 1 else None)):
        if sd is None:
            continue
        n = required_n(sd)
        print("%-56s sd %.5f -> %6.0f replicates per arm, %7.0f GPU-hours"
              % (label, sd, n, n * HOURS_PER_REPLICATE))

    print()
    print("Three seeds is a terrible sd estimate and the two readings differ by")
    print("20x, which is the point: under EITHER of them the replicate count is")
    print("out of reach, so the answer is not more replicates. Lower the noise")
    print("floor instead -- runs/2026-11-13_determinism measures whether pinning")
    print("the GPU kernels does that.")


if __name__ == "__main__":
    main()
