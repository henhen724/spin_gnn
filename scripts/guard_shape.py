"""When the guard fires, what does the model emit INSTEAD? Finding 68's own
data, re-read; nothing is run and nothing is trained.

Finding 68 measured WHETHER the untied model stops naming a node whose
determinacy has been destroyed. It never asked what it names instead, and
that distinction decides the shape of any intervention built on it:

  STOP-SHAPED.  The guard suppresses the node block as a whole and the model
      emits a terminal marker. An intervention that gates the node logits
      monotonically -- z -> z + log sigma(a(z - tau)) -- produces exactly
      this, costs two scalars, and breaks the softmax's additive-constant
      gauge, which finding 73 says is where the missing information lives.

  RERANK-SHAPED.  The guard moves the argmax to a DIFFERENT node that is
      still forced. A monotone gate cannot produce this at all: z + log
      sigma(a(z - tau)) is strictly increasing in z, so it leaves the order
      WITHIN the node block exactly as it was. Only something that changes
      one node's logit relative to another's can.

The two call for different follow-ups, so the question is settled from data
already on disk before either is built.

    python3 scripts/guard_shape.py [--run runs/2026-11-13_ddguard/guard]

Columns are finding 68's: clean_emits_v / corrupt_emits_v are the guard pair,
and pred_* describe what the corrupted prompt actually produced. The
denominator is clean_emits_v, the same CONDITIONAL denominator finding 68
published -- the unconditional rate is a different number and this project
has twice confused the two.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ddguard_score import truthy

DEFAULT = "runs/2026-11-13_ddguard/guard"


def shape(rows):
    den = [r for r in rows if truthy(r["clean_emits_v"])]
    fired = [r for r in den if not truthy(r["corrupt_emits_v"])]
    if not fired:
        return len(den), 0, None
    frac = lambda col: sum(truthy(r[col]) for r in fired) / len(fired)  # noqa
    return len(den), len(fired), {
        "node_token": frac("pred_is_node_token"),
        "still_forced": frac("pred_is_forced"),
        "names_v_again": sum(r["pred_node"] == r["node"] for r in fired)
        / len(fired),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT)
    a = ap.parse_args()
    print("denominator is clean_emits_v, the CONDITIONAL one finding 68 "
          "published")
    print("%-10s %7s %7s  %s" % ("cell", "fired", "of", "of the fired cases"))
    verdict = []
    for cell in sorted(os.listdir(a.run)):
        p = os.path.join(a.run, cell, "skip_behavior.csv")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            den, n, s = shape(list(csv.DictReader(f)))
        if s is None:
            print("%-10s %7d %7d  the guard never fired" % (cell, 0, den))
            continue
        print("%-10s %7d %7d  node token %.3f | still forced %.3f | "
              "names v again %.3f"
              % (cell, n, den, s["node_token"], s["still_forced"],
                 s["names_v_again"]))
        if n >= 30:
            verdict.append(s["node_token"])
    if verdict:
        print()
        if min(verdict) >= 0.9:
            print("RERANK-SHAPED in every cell where the guard fires at all "
                  "(node token >= 0.90).")
            print("A monotone gate on the node logits CANNOT produce this: "
                  "z + log sigma(a(z-tau))")
            print("is strictly increasing in z and leaves the order within "
                  "the node block alone.")
        elif max(verdict) <= 0.1:
            print("STOP-SHAPED: the guard leaves the node block entirely.")
        else:
            print("MIXED: the two shapes coexist, so neither follow-up is "
                  "ruled out by this.")


if __name__ == "__main__":
    main()
