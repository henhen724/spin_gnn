#!/usr/bin/env python3
"""Numbers for the correction pages 136 owes three already-published decks.

Stage SIB's per-run table quotes h_rule pooled over all deduction indices j,
but `/spin-gnn/stopping-condition/` classifies cells by h_rule(0) -- j = 0
alone. Quoting one against the other would be the index-pooling mistake finding
128 already caught once. So this prints BOTH arms for all six cells, admitted
seed alone and pooled over all three seeds, and the correction pages copy from
here rather than from a human's arithmetic.

Nothing is measured here: it reads the same shards stagesib_fit.py reads.

  module purge --force; module load devel math viz; module load python/3.12.1
  module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
  python3 scripts/stagesib_corr.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stagesib_fit as F  # noqa: E402


def main():
    v, vj0, _, _ = F.measure()
    missing = [r for r in F.ALL_RUNS if r not in v]
    if missing:
        print("shards missing for %s" % sorted(missing))
        return 1

    print("BAR = %g,  blind guess = 0.75,  all 14 densities, 400 instances/bin"
          % F.BAR)
    for arm, lab in ((v, "h_rule, pooled over all j"),
                     (vj0, "h_rule(0), j = 0 only")):
        print()
        print("=== %s ===" % lab)
        print("  %-26s %6s  %-22s %-22s %s"
              % ("cell", "params", "admitted seed", "pooled 3 seeds", "ratio"))
        for c in F.CELLS:
            a = arm[c["adm"]]
            k, n, h = F.pooled(c, arm)
            ratio = (h / a["h_rule"]) if a["h_rule"] > 0 else float("inf")
            print("  %-26s %6d  %.6f (%d/%d)%s %.6f (%d/%d)%s %s"
                  % (c["cell"], c["params"],
                     a["h_rule"], a["k"], a["F"],
                     " " * max(0, 22 - len("%.6f (%d/%d)"
                                           % (a["h_rule"], a["k"], a["F"]))),
                     h, k, n,
                     " " * max(0, 22 - len("%.6f (%d/%d)" % (h, k, n))),
                     ("%.1fx" % ratio) if ratio != float("inf") else "0 -> >0"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
