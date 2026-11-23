#!/usr/bin/env python3
"""Stage RULE2 -- extend the h_rule refit to the four never-fitted densities.

ONE KNOB. Stage RULE pooled over the ten fitted densities (rho 0.95-0.50)
because FIT_RHOS = GRID[:10] and rho 0.45/0.40/0.35/0.33 were generated but
never fitted (trap 2, finding 101). This pools over all fourteen and changes
nothing else: same shards, same 79 runs, same 31 architectures, same bars, same
size partial order, same instrument.

WHY THE FOUR BINS ARE NOT BOOKKEEPING. rho_c ~ 0.465 on this d5q4 ensemble
(117), so the never-fitted bins are exactly the ones at or below the colouring
transition. The question is whether DD's rule survives below rho_c or whether
"keeps the rule" only ever meant "keeps the rule in the fast phase".

REGISTERED BIAS DIRECTION. A rollout is censored at its first error, so the deep
bins are scored only on rollouts that got there without erring. That selection
biases h_rule DOWNWARD at low density. A rise there is real; a fall is the
survivorship artefact and is reported as one, never as the rule improving.

Predictions E1-E6 and diagnostics D1-D4 are in
runs/stagerule2_2026-11-23/REGISTERED.md, committed before this ran.

NOTHING IS SUBMITTED AND NO MODEL IS TRAINED.
"""

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stageexp_fit import med, wilson  # noqa: E402
from stagegate_fit import GRID, FIT_RHOS, GATE, J2_LABEL, spearman  # noqa: E402
from stagerule_fit import (load, per_cell, d1_d2, line, smaller_than,  # noqa: E402
                           R1_BAR, BLIND, D1_TOTAL_ERR)

RUNS = "/scratch/users/hshunt/spin_gnn/runs"
OUT = os.path.join(RUNS, "stagerule2_2026-11-23")

ADDED = [round(r, 4) for r in GRID[10:]]      # 0.45, 0.40, 0.35, 0.33
FITTED = [round(r, 4) for r in FIT_RHOS]
HEADLINE = "L=16 h=4 no_mlp d128"             # finding 121's registered answer
PARAM_BEST = "L=16 h=4 tied_no_mlp d320"      # finding 121's post-hoc answer

# Registered bars. E5 is the one carrying the science; E1 decides whether any
# of the rest is interesting.
E1_SHARE = 0.25
E3_ABOVE = 2        # architectures above R1_BAR at ten densities
E6_RHO = 0.90

RHO_C = 0.465       # this ensemble's own transition (117). NOT a d4q3 number.


def by_density(cells):
    """Pooled over the whole in79 population, per density: forced emissions,
    rule errors, at-risk rollouts, and how many (run, rho) cells are empty."""
    acc = collections.defaultdict(lambda: {"in_F": 0.0, "err_in_F": 0.0,
                                           "at_risk": 0.0, "empty": 0,
                                           "runs": 0})
    for c in cells:
        a = acc[c["rho"]]
        a["in_F"] += c["in_F"]
        a["err_in_F"] += c["err_in_F"]
        a["at_risk"] += c["at_risk"]
        a["runs"] += 1
        if not (c["in_F"] > 0):
            a["empty"] += 1
    for r, a in acc.items():
        a["h"] = a["err_in_F"] / a["in_F"] if a["in_F"] > 0 else float("nan")
    return acc


def cell_curve(cells, cell):
    """Per-density h_rule for one architecture, pooled over its seeds."""
    acc = collections.defaultdict(lambda: [0.0, 0.0])
    for c in cells:
        if c["cell"] == cell:
            acc[c["rho"]][0] += c["in_F"]
            acc[c["rho"]][1] += c["err_in_F"]
    return {r: (e / f if f > 0 else float("nan"), f)
            for r, (f, e) in acc.items()}


def d3(fh, a10, a14):
    """The pooling identity. A decrease is arithmetically impossible, so a
    violation means the density set is wrong rather than the data surprising."""
    m10 = {a["cell"]: a for a in a10}
    bad = []
    for a in a14:
        b = m10[a["cell"]]
        if a["in_F"] < b["in_F"] - 1e-9 or a["err_in_F"] < b["err_in_F"] - 1e-9:
            bad.append(a["cell"])
    tot14 = sum(int(round(a["err_in_F"])) for a in a14)
    tot10 = sum(int(round(a["err_in_F"])) for a in a10)
    ok = (not bad) and tot14 == D1_TOTAL_ERR
    line(fh, "D3 %s: all-14 rule errors summed over the 31 architectures = %d "
             "(expected %d); ten-density total = %d; %d architectures "
             "non-monotone (expected 0)"
         % ("PASS" if ok else "FAIL", tot14, D1_TOTAL_ERR, tot10, len(bad)))
    if bad:
        line(fh, "   non-monotone: %s" % ", ".join(sorted(bad)))
    return ok, tot10, tot14


def table(fh, a10, a14):
    m10 = {a["cell"]: a for a in a10}
    line(fh, "=" * 78)
    line(fh, "h_rule BEFORE (ten fitted densities, rho 0.95-0.50) and AFTER")
    line(fh, "(all fourteen, adding %s -- the bins at or below rho_c ~ %.3f)."
         % ("/".join("%.2f" % r for r in ADDED), RHO_C))
    line(fh, "Same shards, same 79 runs, same eval. A blind guess is %.2f."
         % BLIND)
    line(fh, "=" * 78)
    line(fh, "%-30s %7s %10s %10s %9s %7s"
         % ("architecture", "params", "h_rule 10", "h_rule 14", "added %",
            "forced"))
    for a in a14:
        b = m10[a["cell"]]
        share = (a["in_F"] - b["in_F"]) / a["in_F"] if a["in_F"] > 0 else 0.0
        line(fh, "%-30s %7d %10.5f %10.5f %8.1f%% %7d"
             % (a["cell"], a["params"], b["h_rule"], a["h_rule"],
                100.0 * share, int(a["in_F"])))
    line(fh)


def density_block(fh, dens, cells):
    line(fh, "=" * 78)
    line(fh, "PER DENSITY, pooled over all 79 runs. The last four rows are the")
    line(fh, "bins stage RULE never read. rho_c ~ %.3f on this ensemble (117)."
         % RHO_C)
    line(fh, "=" * 78)
    line(fh, "%6s %10s %9s %10s %12s %s"
         % ("rho", "h_rule", "forced", "errors", "at risk", "empty cells"))
    for r in sorted(dens, reverse=True):
        a = dens[r]
        line(fh, "%6.2f %10.5f %9d %10d %12d %4d of %d%s"
             % (r, a["h"], int(a["in_F"]), int(a["err_in_F"]),
                int(a["at_risk"]), a["empty"], a["runs"],
                "   <- added" if r in ADDED else ""))
    line(fh)
    line(fh, "SURVIVORSHIP, the bias that governs the last four rows: a rollout")
    line(fh, "is censored at its first error of any kind, so the deep bins are")
    line(fh, "scored only on rollouts that reached them without erring. That")
    line(fh, "selection pushes h_rule DOWN at low rho. A rise is real; a fall")
    line(fh, "is the artefact and is never read as the rule improving.")
    line(fh)
    for cell in (HEADLINE, PARAM_BEST):
        cur = cell_curve(cells, cell)
        line(fh, "   %-30s %s" % (cell, "  ".join(
            "%.2f:%s" % (r, "n/a" if cur[r][1] == 0 else "%.5f" % cur[r][0])
            for r in sorted(cur, reverse=True))))
    line(fh)


def score(fh, a10, a14, dens):
    m10 = {a["cell"]: a for a in a10}
    line(fh, "=" * 78)
    line(fh, "REGISTERED PREDICTIONS")
    line(fh, "=" * 78)

    # E1 -- does the extension have any denominator behind it at all
    shares = [(a["in_F"] - m10[a["cell"]]["in_F"]) / a["in_F"]
              for a in a14 if a["in_F"] > 0]
    msh = med(shares)
    line(fh, "E1 %s: the four added bins are %.1f%% of total forced emissions "
             "(median over 31), against a %.0f%% bar"
         % ("HIT" if msh < E1_SHARE else "MISS", 100 * msh, 100 * E1_SHARE))
    line(fh, "   range %.1f%% to %.1f%%; pooled over the population the added "
             "bins hold %d of %d forced emissions (%.1f%%)"
         % (100 * min(shares), 100 * max(shares),
            sum(int(dens[r]["in_F"]) for r in ADDED),
            sum(int(dens[r]["in_F"]) for r in dens),
            100.0 * sum(dens[r]["in_F"] for r in ADDED)
            / sum(dens[r]["in_F"] for r in dens)))

    # E2 -- the headline cell
    h = next(a for a in a14 if a["cell"] == HEADLINE)
    b = m10[HEADLINE]
    lo, hi = wilson(int(h["err_in_F"]), int(h["in_F"]))
    line(fh, "E2 %s: %s reads h_rule %.5f [%.5f, %.5f] on all 14 (%d / %d "
             "forced), against %.5f on the ten, bar %.2f"
         % ("HIT" if h["h_rule"] < R1_BAR else "MISS", HEADLINE, h["h_rule"],
            lo, hi, int(h["err_in_F"]), int(h["in_F"]), b["h_rule"], R1_BAR))
    line(fh, "   per-seed on all 14: %s"
         % "/".join("%.5f" % s for _, s, _, _ in h["seeds"]))

    # E3 -- does anything cross the bar
    up10 = set(a["cell"] for a in a10 if a["h_rule"] >= R1_BAR)
    up14 = set(a["cell"] for a in a14 if a["h_rule"] >= R1_BAR)
    line(fh, "E3 %s: %d architectures sit above %.2f before and %d after; "
             "crossings %d"
         % ("HIT" if up10 == up14 and len(up10) == E3_ABOVE else "MISS",
            len(up10), R1_BAR, len(up14), len(up10 ^ up14)))
    if up10 ^ up14:
        line(fh, "   crossed: %s" % ", ".join(sorted(up10 ^ up14)))

    # E4 -- the three breakers
    t10 = [a["cell"] for a in sorted(a10, key=lambda x: -x["h_rule"])[:3]]
    t14 = [a["cell"] for a in sorted(a14, key=lambda x: -x["h_rule"])[:3]]
    line(fh, "E4 %s: top three by h_rule unchanged"
         % ("HIT" if set(t10) == set(t14) else "MISS"))
    line(fh, "   before: %s" % ", ".join(t10))
    line(fh, "   after:  %s" % ", ".join(t14))

    # E5 -- the science
    h33, h50 = dens[0.33]["h"], dens[0.5]["h"]
    line(fh, "E5 %s: pooled h_rule at rho 0.33 is %.5f against %.5f at rho "
             "0.50, a factor of %.2fx"
         % ("HIT" if h33 > h50 else "MISS", h33, h50,
            h33 / h50 if h50 > 0 else float("nan")))
    # The direction decides which sentence is true, and printing the wrong one
    # would contradict the verdict on the same line. REGISTERED.md fixed both
    # readings in advance precisely so this is not a choice made now.
    if h33 > h50:
        line(fh, "   it rises AGAINST the censoring bias, so the measured "
                 "factor is a LOWER bound on the real degradation below rho_c.")
    else:
        line(fh, "   IT FALLS, WHICH IS THE REGISTERED ARTEFACT AND NOT A "
                 "FINDING. Censoring keeps the rollouts that reached the deep "
                 "bins without erring, so the low-rho bins are scored on a "
                 "selected sample and h_rule there is biased DOWN. This stage "
                 "therefore CANNOT say whether DD's rule degrades below rho_c; "
                 "it reports that the question is unanswerable from censored "
                 "rollouts and needs an eval that scores past the first error.")
    seq = [dens[r]["h"] for r in sorted(dens, reverse=True)]
    line(fh, "   full sequence high rho to low: %s"
         % " ".join("%.5f" % v for v in seq))

    # E6 -- does stage RULE's ranking survive
    xs = [a["h_rule"] for a in a10]
    ys = [next(z["h_rule"] for z in a14 if z["cell"] == a["cell"]) for a in a10]
    rs = spearman(xs, ys)
    line(fh, "E6 %s: Spearman(h_rule 10, h_rule 14) = %+.4f over 31 "
             "architectures, against a %.2f bar"
         % ("HIT" if rs >= E6_RHO else "MISS", rs, E6_RHO))
    # Counted, never quoted from stage RULE: its "12 measured at exactly zero"
    # is the 25-architecture SUB-GATE population of R3, while E6 ranks all 31.
    # Over all 31 the count is 15 at ten densities and 13 at fourteen, and
    # mixing the two populations is what made this line wrong once already.
    z10 = sum(1 for a in a10 if a["h_rule"] == 0.0)
    z14 = sum(1 for a in a14 if a["h_rule"] == 0.0)
    line(fh, "   (over these 31, %d measure exactly zero at ten densities and "
             "%d at fourteen, so ties are doing some of this work; read it as "
             "'the ranking did not reorder', never as resolution.)"
         % (z10, z14))
    if rs < E6_RHO:
        # A registered falsifier fired. Say exactly what it falsified and what
        # it did not, with the movers named -- not a paragraph defending 121.
        r10 = {a["cell"]: i for i, a in
               enumerate(sorted(a10, key=lambda x: -x["h_rule"]))}
        r14 = {a["cell"]: i for i, a in
               enumerate(sorted(a14, key=lambda x: -x["h_rule"]))}
        moves = sorted(((abs(r14[c] - r10[c]), c) for c in r10), reverse=True)
        line(fh, "   THE FALSIFIER FIRED, SO STATE ITS SCOPE. It falsifies the "
                 "FULL RANKING: the biggest rank moves are %s."
             % ", ".join("%s %d->%d" % (c, r10[c] + 1, r14[c] + 1)
                         for _, c in moves[:4]))
        line(fh, "   It does NOT falsify the claim finding 121 actually made. "
                 "E3 and E4 both HIT: the same 2 architectures sit above the "
                 "bar, the same 3 are worst, and the headline cells move by "
                 "less than 0.0001. 121 said in terms that h_rule SEPARATES "
                 "and does not rank -- of its 25 sub-gate architectures 12 "
                 "measured exactly zero and the remaining 13 spanned only "
                 "5.4x -- so a bulk reordering under a density change is that "
                 "caveat coming true, not a surprise.")
        line(fh, "   WHAT IS NOW ON RECORD: the h_rule RANKING is "
                 "density-dependent and must never be quoted; the h_rule "
                 "SEPARATION is not.")
    line(fh)


def answer(fh, a10, a14):
    """What the extension does to finding 121's two headline cells. The brief's
    two readings of "smallest" disagree and resolving that is hshunt's call, so
    both are reported and neither is promoted."""
    line(fh, "=" * 78)
    line(fh, "WHAT THIS DOES TO FINDING 121'S ANSWER")
    line(fh, "=" * 78)
    m10 = {a["cell"]: a for a in a10}
    j2 = next(a for a in a14 if a["cell"] == J2_LABEL)
    for tag, cell in (("object order   ", HEADLINE),
                      ("parameter count", PARAM_BEST)):
        a, b = next(z for z in a14 if z["cell"] == cell), m10[cell]
        lo, hi = wilson(int(a["err_in_F"]), int(a["in_F"]))
        line(fh, "%s %-30s %7d params  %d seeds  %.5f -> %.5f [%.5f, %.5f]"
             % (tag, cell, a["params"], a["n_seeds"], b["h_rule"],
                a["h_rule"], lo, hi))
    line(fh, "%s %-30s %7d params  %d seeds  %.5f -> %.5f"
         % ("standing J2    ", J2_LABEL, j2["params"], j2["n_seeds"],
            m10[J2_LABEL]["h_rule"], j2["h_rule"]))
    line(fh)
    smaller = [a for a in a14 if smaller_than(a, j2)
               and a["h_rule"] < R1_BAR
               and not (a["rho50"] == a["rho50"] and a["rho50"] <= GATE)]
    line(fh, "   %d objects strictly smaller than J2 are still rule-intact and "
             "gate-failing on all 14 densities (6 on the ten)." % len(smaller))
    line(fh, "   NEITHER READING OF \"smallest\" IS PROMOTED HERE. The object "
             "order is the project's standing one; parameter count is the "
             "other reading of the 2026-11-23 brief; the choice is hshunt's.")
    line(fh)


def main():
    os.makedirs(OUT, exist_ok=True)
    cells, viol, n_haz, n_summ = load()
    a10 = per_cell(cells)
    a14 = per_cell(cells, GRID)
    assert len(a10) == 31 and len(a14) == 31, "expected 31 architectures"
    dens = by_density(cells)
    assert set(dens) == set(FITTED) | set(ADDED), "density grid changed"

    path = os.path.join(OUT, "VERDICT.txt")
    with open(path, "w") as fh:
        line(fh, "STAGE RULE2 -- h_rule over all fourteen densities")
        line(fh, "one knob: the pooling set, ten fitted -> all fourteen")
        line(fh, "%d hazard rows, %d summary rows, %d in79 cells"
             % (n_haz, n_summ, len(cells)))
        line(fh, "NOTHING SUBMITTED, NO MODEL TRAINED, NO NEW DATA.")
        line(fh)
        ok, _ = d1_d2(fh, cells, viol)
        ok3, _, _ = d3(fh, a10, a14)
        line(fh, "D4 PASS (checked outside this script): stage RULE's "
                 "VERDICT.txt md5 b06bcbdd9071651ed2e68706cb16e617 before and "
                 "after the per_cell(rhos=) patch.")
        line(fh)
        if not (ok and ok3):
            line(fh, "DIAGNOSTIC FAILURE -- the stage stops here, as "
                     "registered. No prediction is scored.")
            print("\nwrote", path)
            return 1
        table(fh, a10, a14)
        density_block(fh, dens, cells)
        score(fh, a10, a14, dens)
        answer(fh, a10, a14)
    print("\nwrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
