"""Is the standing minimum actually the cheapest thing on this grid?

The brief since 11-04 is the SMALLEST architecture that still learns direct
deduction. Every rung so far earned that title by a route that only looks at a
few cells at a time: `no_mlp` d256 is "the standing minimum" because it is the
cheapest rung whose rho_50 crosses inside the band. That is a statement about
the rungs somebody thought to compare, not about the grid.

This script asks the question of all of them at once, and it is DESCRIPTIVE: it
registers no prediction, scores no falsifier, and introduces no new cell. It
sorts measurements that already exist.

WHY TOKEN ACCURACY IS THE QUALITY AXIS. rho_50 censors -- a third of this grid
has no number at all on it -- and mean_solve is near zero across the whole cheap
end, where the interesting comparisons are. Finding 4+6's 11-04 update measured
validation token accuracy to be a sufficient statistic for mean_solve over this
grid (15/15 well-determined rungs, worst discriminating miss 0.0091 against a
0.0423 leave-one-out bar), and it never censors. mean_solve is printed beside
every row so the two can be read against each other.

THE RULE, WRITTEN BEFORE IT WAS RUN. A rung R is DOMINATED if some rung Q is
strictly cheaper in trainable parameters AND matches it on accuracy to within
ACC_BAR:

    trainable(Q) < trainable(R)  and  acc(Q) >= acc(R) - ACC_BAR

ACC_BAR is not chosen here. It is built the same way s* was: the largest
WITHIN-rung across-seed accuracy spread anywhere in the grid, over rungs with at
least three seeds. A gap smaller than the spread one architecture shows against
itself is not a gap, and the frontier is only as sharp as that bar -- a point
that survives by less than ACC_BAR survives by nothing.

ELIGIBILITY, AND AN HONEST NOTE ABOUT WHEN IT WAS ADDED. The first version of
this script only MARKED rungs that never escaped the 1.64 plateau. Run that way
it returned a frontier of three rungs, all of them at 0 escapes out of 3, with
`no_mlp` d128 -- mean_solve 0.0291, rho_50 censored -- "matching" the hub and
dominating eighteen rungs that actually solve. That is not a surprising result
to be reported; it is the wrong question. The brief asks for the smallest
architecture that STILL LEARNS DIRECT DEDUCTION, and a cell that never left the
plateau learned nothing, which is the programme's own rule ("0 escapes -> did
not train in 375k steps, never evidence in either direction"). So eligibility
is now a majority of seeds below the plateau. The rule changed AFTER seeing the
output, so both frontiers are printed and labelled: the one specified first,
and the one the brief actually asks for.

Default scan dirs are SCAN_DIRS, which excludes SCAN_E for the usual reason:
stage E's rungs carry entirely default hyperparameters and would pool into the
cells every published verdict was scored against. Pass --scan-dir to add it.
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.acc_vs_solve import val_accuracy  # noqa: E402
from scripts.minarch_analysis import (SCAN_DIRS, cell_key, cell_metrics,  # noqa: E402
                                      load_all, median, rung_label, spread)
from scripts.stagec_score import (HUB, NO_MLP_D256, PLATEAU, S_MEAN,  # noqa: E402
                                   escapes, rho_txt,
                                   summarize)

MIN_SEEDS = 2  # a one-seed rung is a seed, not an architecture (findings 41/42)


def collect(scan_dirs, boot):
    by = defaultdict(list)
    for r in load_all(scan_dirs):
        by[cell_key(r)].append(r)
    return {k: dict(cell_metrics(v, boot), path=v[0]["path"]) for k, v in by.items()}


def rungs(cells):
    """One record per rung with at least MIN_SEEDS seeds on disk."""
    by = defaultdict(list)
    for key, m in cells.items():
        by[key[:-1]].append(m)
    out = []
    for key, ms in by.items():
        if len(ms) < MIN_SEEDS:
            continue
        accs = [val_accuracy(m["path"]) for m in ms]
        if any(a != a for a in accs):  # unreadable history
            continue
        s = summarize(ms)
        n_esc, _nll = escapes(ms)
        out.append({
            "key": key, "name": rung_label(*key), "n": len(ms),
            "trainable": s["trainable"], "acc": median(accs), "acc_spread": spread(accs),
            "mean_solve": s["mean_solve"], "rho50": rho_txt(s), "n_esc": n_esc,
        })
    return sorted(out, key=lambda r: r["trainable"])


def acc_bar(rs):
    """The largest within-rung across-seed accuracy spread, s*'s construction."""
    wide = [r for r in rs if r["n"] >= 3]
    worst = max(wide, key=lambda r: r["acc_spread"])
    return worst["acc_spread"], worst


def frontier(rs, bar, axis="acc", field="dominated_by"):
    """Rungs no cheaper rung matches on `axis` to within `bar`."""
    keep = []
    for r in rs:
        dom = [q for q in rs
               if q["trainable"] < r["trainable"] and q[axis] >= r[axis] - bar]
        if not dom:
            keep.append(r)
        r[field] = min(dom, key=lambda q: q["trainable"]) if dom else None
    return keep


def row(r, bar_note=""):
    esc = f"{r['n_esc']}/{r['n']}"
    flag = "  <- never escaped 1.64" if r["n_esc"] == 0 else bar_note
    return (f"  {r['name']:<24} {r['trainable']:>10,}  acc {r['acc']:.4f} "
            f"[{r['acc_spread']:.4f}]  mean_solve {r['mean_solve']:.4f}  "
            f"rho_50 {r['rho50']:>8}  esc {esc}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", nargs="+", default=SCAN_DIRS)
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()

    cells = collect(a.scan_dir, a.boot)
    rs = rungs(cells)
    print("THE CHEAPEST ARCHITECTURE AT EACH LEVEL OF QUALITY -- descriptive, no verdict")
    print(f"  {len(cells)} cells from {len(a.scan_dir)} directories -> {len(rs)} rungs "
          f"with >={MIN_SEEDS} seeds")
    bar, worst = acc_bar(rs)
    print(f"  ACC_BAR = {bar:.4f}, the largest within-rung across-seed accuracy spread in the")
    print(f"  grid ({worst['name']}, {worst['n']} seeds). Built, not chosen; a gap smaller than")
    print("  this is not a gap. For scale, s* in mean_solve is 0.2918.")

    print("\n=== AS FIRST SPECIFIED, WITH UNTRAINED RUNGS LEFT IN -- the wrong question ===")
    for r in frontier(rs, bar):
        print(row(r))
    print("  Every rung on it is 0/3 escapes. A cell that never left the plateau has not")
    print("  learned direct deduction, so this frontier answers 'what is cheap', not 'what")
    print("  is the cheapest thing that works'. Kept visible because the rule changed after")
    print("  it was seen.")

    live = [r for r in rs if r["n_esc"] * 2 > r["n"]]
    front = frontier(live, bar)
    print(f"\n=== ELIGIBLE: {len(live)} of {len(rs)} rungs trained "
          f"(majority of seeds below {PLATEAU}) ===")
    print(f"=== THE FRONTIER AMONG THEM: {len(front)} rungs ===")
    for r in front:
        print(row(r))

    dom = [r for r in live if r["dominated_by"] is not None]
    print(f"\n=== DOMINATED: {len(dom)} rungs, each with its cheapest matcher ===")
    for r in dom:
        q = r["dominated_by"]
        print(f"  {r['name']:<24} {r['trainable']:>10,}  acc {r['acc']:.4f}   <- "
              f"{q['name']} at {q['trainable']:,} ({q['trainable'] / r['trainable']:.2f}x) "
              f"matches it, acc {q['acc']:.4f}")

    print("\n=== THE SAME FRONTIER IN mean_solve, WHERE THE BAR IS REGISTERED ===")
    mfront = frontier(live, S_MEAN, axis="mean_solve", field="m_dom")
    print(f"  bar = s* = {S_MEAN}, the registered seed spread on this axis, not a bar built here")
    for r in mfront:
        print(row(r))
    print("  Why do this twice: ACC_BAR is calibrated in accuracy, and over the eligible")
    span = max(r["acc"] for r in live) - min(r["acc"] for r in live)
    mspan = max(r["mean_solve"] for r in live) - min(r["mean_solve"] for r in live)
    print(f"  rungs accuracy spans only {span:.4f} while mean_solve spans {mspan:.4f} -- a slope of")
    print(f"  ~{mspan / span:.0f} solve per unit accuracy, so ACC_BAR {bar:.4f} is worth about")
    print(f"  {bar * mspan / span:.2f} of mean_solve, roughly {bar * mspan / span / S_MEAN:.1f} s*. A rung can 'match' another on")
    print("  accuracy while deducing far worse. Accuracy is a sufficient statistic for")
    print("  mean_solve over this grid (finding 4+6) but NOT a sensitive one, and the two")
    print("  frontiers above are the price of that. Read mean_solve for size of effect,")
    print("  accuracy only for rungs where rho_50 censors.")

    print("\n=== HOW MUCH OF THIS THE GRID CAN ACTUALLY RESOLVE ===")
    print(f"  accuracy spans {span:.4f} across every eligible rung, against ACC_BAR "
          f"{bar:.4f}")
    print(f"  -- a ratio of {span / bar:.2f}. mean_solve spans {mspan:.4f} against s* "
          f"{S_MEAN}, a ratio of {mspan / S_MEAN:.2f}.")
    print("  In accuracy the seed-to-seed bar is two thirds of the entire range the grid")
    print("  covers, so the accuracy frontier can separate almost nothing; in mean_solve")
    print("  there are 2.7 bars of room and it separates rungs. Same resolution limit the")
    print("  rung-collapse permutation test ran into (p=0.569 inside the band, p=0.045 only")
    print("  across the depth ladder's full range).")

    print("\n=== WHAT SETS THE FLOOR: TRAINABILITY, NOT CAPACITY ===")
    cheap = [r for r in rs if r["trainable"] < min(q["trainable"] for q in live)]
    print(f"  Every eligible rung is at or above {min(q['trainable'] for q in live):,} trainable. The "
          f"{len(cheap)} rungs below that line")
    print("  are not excluded for deducing badly -- they are excluded for never leaving the")
    print("  1.64 plateau in 375k steps, which is an optimization outcome, not a capacity one:")
    for r in cheap:
        print(row(r))
    print("  So the frontier question 'is anything cheaper as good?' has no cheaper candidate")
    print("  LEFT to ask it of, and the standing minimum is on the frontier by default rather")
    print("  than by beating anything. The binding constraint on 'smallest architecture that")
    print("  still learns direct deduction' is now the word LEARNS. Note how far the floor")
    print("  could move: no_mlp d128 is 3.6x below the standing minimum and already reaches")
    print("  accuracy 0.4558 against its 0.4812 while sitting ON the plateau -- the plateau is")
    print("  a partial-learning state, not a dead model (rf_all sits at 0.11).")

    print("\n=== THE QUESTION THE BRIEF ASKS ===")
    std = next((r for r in live if r["key"] == NO_MLP_D256), None)
    hub = next((r for r in live if r["key"] == HUB), None)
    if std is None:
        print("  the standing minimum is not on disk under its own key -- nothing to say")
        return
    print(f"  standing minimum  {std['name']}, {std['trainable']:,} trainable, "
          f"acc {std['acc']:.4f}, mean_solve {std['mean_solve']:.4f}")
    if std in front:
        print("  IT IS ON THE FRONTIER: no rung in this grid that trained is cheaper and")
        print(f"  matches its accuracy to within {bar:.4f}.")
    else:
        q = std["dominated_by"]
        print(f"  IT IS NOT ON THE FRONTIER: {q['name']} at {q['trainable']:,} "
              f"({q['trainable'] / std['trainable']:.2f}x) matches it, acc {q['acc']:.4f}.")
        print(f"  Read that against the bar: the two differ by {std['acc'] - q['acc']:.4f} in")
        print(f"  accuracy and {std['mean_solve'] - q['mean_solve']:+.4f} in mean_solve, so the")
        print("  claim is 'not separated at this resolution', NOT 'equally good'.")
    if hub:
        print(f"\n  for reference the hub is {hub['trainable']:,} at acc {hub['acc']:.4f}, "
              f"mean_solve {hub['mean_solve']:.4f}")
    print("\n  Descriptive. Nothing here is a registered result, every row is a median over")
    print(f"  {MIN_SEEDS}-3 seeds, and the frontier is only as sharp as ACC_BAR = {bar:.4f}.")


if __name__ == "__main__":
    main()
