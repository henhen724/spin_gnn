#!/usr/bin/env python3
"""Stage RULE -- rank the 31 architectures by DD's RULE, not by the solving gate.

A refit of stage LOGIC's hazard shards. Nothing is submitted and no model is
trained. Predictions R1-R5 are registered in
runs/stagerule_2026-11-23/REGISTERED.md and were written before this ran.

THE INSTRUMENT. Per (run_id, rho, j) the shards carry n_at_risk, n_in_F
(emissions naming a cell DD has genuinely FORCED), n_err and n_err_in_F (first
errors that named a forced cell -- i.e. the colour was wrong on a forced cell).
So

    h_rule = sum_j n_err_in_F / sum_j n_in_F = Pr(colour wrong | cell forced)

is DD's rule with the stopping condition divided out. Finding 98 established
that the published gate rho_50 never charges for naming an UNFORCED cell, so
every version of that gate is a solving score dominated by the stopping
condition -- the part the 2026-11-23 brief drops. This is the part it keeps.

h_rule IS NOT A DENSITY THRESHOLD and is never converted into one.
"""

import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stageexp_fit import fnum, tobool, read_shards, med  # noqa: E402
from stagegate_fit import (GRID, FIT_RHOS, GATE, J2_LABEL,  # noqa: E402
                           rho50, spearman)

RUNS = "/scratch/users/hshunt/spin_gnn/runs"
LOGIC = os.path.join(RUNS, "stagelogic_2026-11-17")
OUT = os.path.join(RUNS, "stagerule_2026-11-23")

# Registered bars. Changing one of these after reading a number is the thing
# the registration exists to prevent.
R1_BAR = 0.01          # "has DD's rule"
R2_FACTOR = 10.0       # finding 100's three, against the gate-passing median
R3_SPAN = 5.0          # resolution below the gate
R3_MIN_DENOM = 100     # a cell needs this many forced emissions to be ranked
R4_RHO = 0.50          # |Spearman| at or above this means the gate already
#                        encodes the rule and this instrument adds nothing
BLIND = 0.75           # a blind colour guess at q=4: 1 - 1/q

# Finding 100's published counts, for D1. A mismatch stops the stage.
D1_TOTAL_ERR = 1988
D1_AFFECTED = 63
D1_CELLS = 1106
D1_THREE = {"tied_no_mlp d192": 977, "no_mlp d128 ep100": 677,
            "no_mlp d160": 307}

# THE SIZE PARTIAL ORDER, stated so it can be disagreed with. Rank is "how much
# machinery the object has": tying shares one block's weights across depth, and
# no_mlp deletes the MLP sublayer, so each is a strictly smaller OBJECT than
# what it is derived from. linear and rf_mlp are NOT on this chain -- linear
# removes the attention nonlinearity and rf_mlp freezes the MLP, which are
# different axes, not smaller points on this one -- so they are reported and
# never called smaller or larger.
MACHINERY = {"tied_no_mlp": 0, "no_mlp": 1, "full": 2}
OFF_CHAIN = {"linear", "rf_mlp"}


def line(fh, s=""):
    print(s)
    fh.write(s + "\n")


def smaller_than(a, b):
    """Strictly smaller OBJECT: no more machinery, no more depth, no more
    width, and at least one of the three strictly less. Returns False for
    anything off the machinery chain rather than guessing."""
    if a["mode"] in OFF_CHAIN or b["mode"] in OFF_CHAIN:
        return False
    ma, mb = MACHINERY[a["mode"]], MACHINERY[b["mode"]]
    le = ma <= mb and a["L"] <= b["L"] and a["d"] <= b["d"]
    lt = ma < mb or a["L"] < b["L"] or a["d"] < b["d"]
    return le and lt


def load():
    """Per (run_id, rho): forced-emission and rule-error totals, plus the
    measured solve rate and the architecture the run belongs to."""
    haz = read_shards(LOGIC, "F_hazard_all_*of10.csv", keep=[
        "run_id", "in79", "rho", "j", "n_at_risk", "n_in_F", "n_err",
        "n_err_in_F"])
    summ = read_shards(LOGIC, "LF_summary_all_*of10.csv", keep=[
        "run_id", "cell", "arch_mode", "num_layers", "d_model",
        "n_trainable_params", "val_acc", "in79", "rho", "solve_rate"])

    # D2 runs over EVERY in79 row before anything is aggregated.
    viol = {"in_F_gt_at_risk": 0, "err_in_F_gt_err": 0}
    acc = collections.defaultdict(lambda: {"in_F": 0.0, "err_in_F": 0.0,
                                           "at_risk": 0.0, "err": 0.0,
                                           "njs": 0, "ddf": 0.0, "joint": 0.0,
                                           "z_ddf": False, "z_joint": False})
    for h in haz:
        if not tobool(h, "in79"):
            continue
        ar, inf = fnum(h, "n_at_risk"), fnum(h, "n_in_F")
        e_all, e_inf = fnum(h, "n_err"), fnum(h, "n_err_in_F")
        if inf > ar:
            viol["in_F_gt_at_risk"] += 1
        if e_inf > e_all:
            viol["err_in_F_gt_err"] += 1
        if not (ar > 0):
            continue
        p = acc[(h["run_id"], round(fnum(h, "rho"), 4))]
        p["njs"] += 1
        p["in_F"] += inf
        p["err_in_F"] += e_inf
        p["at_risk"] += ar
        p["err"] += e_all
        # D1 needs finding 100's OWN quantity, which is not "any rule error"
        # but "the rule errors drop S_joint below S_ddf". Rebuild both survival
        # products with the identical log1p + zero-flag logic as
        # stagegate_fit.load(), so a mismatch cannot come from the arithmetic.
        for tag, hv in (("ddf", (ar - inf) / ar),
                        ("joint", ((ar - inf) + e_inf) / ar)):
            if hv >= 1.0:
                p["z_" + tag] = True
            elif hv > 0:
                p[tag] += np.log1p(-hv)

    cells = []
    for s in summ:
        if not tobool(s, "in79"):
            continue
        key = (s["run_id"], round(fnum(s, "rho"), 4))
        if key not in acc:
            continue
        p = acc[key]
        g = lambda t: 0.0 if p["z_" + t] else float(np.exp(p[t]))
        cells.append({
            "S_ddf": g("ddf"), "S_joint": g("joint"),
            "run_id": s["run_id"], "cell": " ".join(s["cell"].split()),
            "mode": s["arch_mode"], "L": int(float(s["num_layers"])),
            "d": int(float(s["d_model"])),
            "params": int(fnum(s, "n_trainable_params")),
            "rho": round(fnum(s, "rho"), 4),
            "S": fnum(s, "solve_rate"),
            "in_F": p["in_F"], "err_in_F": p["err_in_F"],
            "at_risk": p["at_risk"], "err": p["err"], "njs": p["njs"],
        })
    return cells, viol, len(haz), len(summ)


def d1_d2(fh, cells, viol):
    """Reproduce finding 100's counts and check the columns mean what the
    registration assumes. Either failing stops the stage."""
    line(fh, "=" * 72)
    line(fh, "DIAGNOSTICS -- if either fails, nothing below is reported")
    line(fh, "=" * 72)

    ok2 = viol["in_F_gt_at_risk"] == 0 and viol["err_in_F_gt_err"] == 0
    line(fh, "D2 %s: n_in_F > n_at_risk on %d rows; n_err_in_F > n_err on %d"
             % ("PASS" if ok2 else "FAIL", viol["in_F_gt_at_risk"],
                viol["err_in_F_gt_err"]))

    # Finding 100's counts are over ALL 14 densities (1,106 = 79 runs x 14),
    # not the ten fitted ones, so D1 pools the whole grid and says so.
    total = int(round(sum(c["err_in_F"] for c in cells)))
    # Finding 100's "63 of 1,106" is cells where the rule errors DROP S_joint
    # BELOW S_ddf -- not cells that merely have one. The two differ exactly
    # where S_ddf is already floored at zero (some j had n_in_F == 0, so its
    # hazard hit 1.0): a product already at 0 cannot be dropped further. My
    # first D1 counted the second quantity and "failed" at 77 vs 63 for that
    # reason alone, with every raw COUNT reproducing exactly -- the same shape
    # of defect as finding 98's D1, which summed over the joined j. Both are
    # printed so the gap is visible rather than argued.
    affected = sum(1 for c in cells if c["S_joint"] < c["S_ddf"])
    any_err = sum(1 for c in cells if c["err_in_F"] > 0)
    floored = sum(1 for c in cells
                  if c["err_in_F"] > 0 and not (c["S_joint"] < c["S_ddf"]))
    by_arch = collections.Counter()
    for c in cells:
        if c["err_in_F"] > 0:
            by_arch[c["cell"]] += int(round(c["err_in_F"]))

    line(fh, "D1 totals over all %d densities: n_err_in_F %d (expected %d), "
             "cells where S_joint < S_ddf %d of %d (expected %d of %d)"
         % (len(GRID), total, D1_TOTAL_ERR, affected, len(cells),
            D1_AFFECTED, D1_CELLS))
    line(fh, "   %d cells carry a rule error at all; %d of those sit at "
             "S_ddf = 0 already, where no error can lower the product -- that "
             "difference, not drift, is why a naive count reads %d."
         % (any_err, floored, any_err))

    # Finding 100 wrote "no_mlp d160" without a depth, and the ladder has both
    # L=12 and L=16 at d=160. Resolve it by the count it published rather than
    # guessing, and fail loudly if neither matches.
    named, resolved = [], True
    for frag, want in sorted(D1_THREE.items(), key=lambda kv: -kv[1]):
        hits = [lab for lab, n in by_arch.items()
                if frag in lab and n == want]
        if len(hits) == 1:
            named.append(hits[0])
            line(fh, "   resolved '%s' -> %-30s %d errors" % (frag, hits[0], want))
        else:
            resolved = False
            cand = {lab: n for lab, n in by_arch.items() if frag in lab}
            line(fh, "   COULD NOT resolve '%s' at %d errors; candidates %s"
                 % (frag, want, cand))

    three_cells = sum(1 for c in cells
                      if c["S_joint"] < c["S_ddf"] and c["cell"] in named)
    three_err = sum(by_arch[l] for l in named)
    line(fh, "   the three named architectures hold %d of %d affected cells "
             "and %d of %d errors" % (three_cells, affected, three_err, total))

    ok1 = (total == D1_TOTAL_ERR and affected == D1_AFFECTED
           and len(cells) == D1_CELLS and resolved)
    line(fh, "D1 %s" % ("PASS" if ok1 else "FAIL"))
    line(fh)
    return ok1 and ok2, named


def per_cell(cells):
    """Pool over the TEN FITTED densities only (trap 2), per run first so the
    per-seed parts survive, then per architecture."""
    fit = set(round(r, 4) for r in FIT_RHOS)
    runs = collections.defaultdict(lambda: {"in_F": 0.0, "err_in_F": 0.0,
                                            "curve": {}})
    meta = {}
    for c in cells:
        r = runs[(c["cell"], c["run_id"])]
        r["curve"][c["rho"]] = c["S"]
        if c["rho"] in fit:
            r["in_F"] += c["in_F"]
            r["err_in_F"] += c["err_in_F"]
        meta[c["cell"]] = {"mode": c["mode"], "L": c["L"], "d": c["d"],
                           "params": c["params"]}

    arch = collections.defaultdict(lambda: {"in_F": 0.0, "err_in_F": 0.0,
                                            "seeds": [], "rho50s": []})
    for (cell, run_id), r in runs.items():
        a = arch[cell]
        a["in_F"] += r["in_F"]
        a["err_in_F"] += r["err_in_F"]
        h = r["err_in_F"] / r["in_F"] if r["in_F"] > 0 else float("nan")
        a["seeds"].append((run_id, h, r["in_F"], r["err_in_F"]))
        a["rho50s"].append(rho50(r["curve"])[0])

    out = []
    for cell, a in arch.items():
        out.append({
            "cell": cell, **meta[cell],
            "h_rule": (a["err_in_F"] / a["in_F"] if a["in_F"] > 0
                       else float("nan")),
            "in_F": a["in_F"], "err_in_F": a["err_in_F"],
            "seeds": sorted(a["seeds"]),
            "rho50": med(a["rho50s"]),
            "n_seeds": len(a["seeds"]),
        })
    return sorted(out, key=lambda x: x["params"])


def report(fh, arch):
    line(fh, "=" * 72)
    line(fh, "THE RULE HAZARD PER ARCHITECTURE, pooled over the ten fitted")
    line(fh, "densities (rho 0.95-0.50). A blind colour guess is %.2f." % BLIND)
    line(fh, "rho_50 IS A COLOURING CRITERION, NEVER A DD ONE (98): it is")
    line(fh, "printed only as the thing h_rule is compared against.")
    line(fh, "=" * 72)
    line(fh, "%-30s %7s %6s %9s %10s %7s %s"
         % ("architecture", "params", "gate", "rho_50", "h_rule",
            "forced", "per-seed h_rule"))
    for a in arch:
        clears = a["rho50"] == a["rho50"] and a["rho50"] <= GATE
        hs = " ".join("nan" if h != h else "%.4f" % h
                      for _, h, _, _ in a["seeds"])
        line(fh, "%-30s %7d %6s %9s %10s %7d  %s"
             % (a["cell"], a["params"], "PASS" if clears else "--",
                "censored" if a["rho50"] != a["rho50"] else "%.4f" % a["rho50"],
                "undef" if a["h_rule"] != a["h_rule"] else "%.5f" % a["h_rule"],
                int(a["in_F"]), hs))
    line(fh)


def score(fh, arch, named):
    gate = [a for a in arch if a["rho50"] == a["rho50"] and a["rho50"] <= GATE]
    below = [a for a in arch if a not in gate]
    j2 = next((a for a in arch if a["cell"] == J2_LABEL), None)

    line(fh, "=" * 72)
    line(fh, "REGISTERED PREDICTIONS")
    line(fh, "=" * 72)

    # R1
    bad = [a for a in gate if not (a["h_rule"] < R1_BAR)]
    line(fh, "R1 %s: %d architectures clear the %.3f gate; %d of them fail "
             "h_rule < %.2f%s"
         % ("HIT" if not bad else "MISS", len(gate), GATE, len(bad), R1_BAR,
            "" if not bad else " (" + ", ".join(
                "%s %.5f" % (a["cell"], a["h_rule"]) for a in bad) + ")"))
    line(fh, "   gate-passing h_rule: " + ", ".join(
        "%s %.5f" % (a["cell"], a["h_rule"]) for a in gate))

    # R2
    gmed = med([a["h_rule"] for a in gate])
    three = [a for a in arch if a["cell"] in named]
    ratios = [(a["cell"], a["h_rule"] / gmed if gmed > 0 else float("inf"))
              for a in three]
    worst = min(r for _, r in ratios) if ratios else float("nan")
    line(fh, "R2 %s: finding 100's three sit at %s the gate-passing median "
             "(%.2e); the weakest is %.0fx against a %.0fx bar"
         % ("HIT" if worst >= R2_FACTOR else "MISS",
            "/".join("%.0fx" % r for _, r in ratios), gmed, worst, R2_FACTOR))
    # The ratio is secure in DIRECTION but its magnitude divides by a median of
    # ~3e-6, so "21806x" is not a measured figure. Quote the absolutes.
    line(fh, "   QUOTE THE ABSOLUTES, NOT THE RATIO: %s against ~%.0e for the "
             "median gate-passing architecture -- three to four orders of "
             "magnitude, and the ratio's precision is an artefact of a "
             "near-zero denominator."
         % (", ".join("%.4f" % a["h_rule"] for a in
                      sorted(three, key=lambda x: -x["h_rule"])), gmed))
    for cell, r in ratios:
        a = next(x for x in arch if x["cell"] == cell)
        line(fh, "   %-30s h_rule %.5f  (%d errors / %d forced)  %.1fx"
             % (cell, a["h_rule"], int(a["err_in_F"]), int(a["in_F"]), r))

    # R3 -- a zero denominator is not a zero hazard (trap 6)
    thin = [a for a in below if a["in_F"] < R3_MIN_DENOM]
    rank = [a for a in below if a["in_F"] >= R3_MIN_DENOM
            and a["h_rule"] == a["h_rule"] and a["h_rule"] > 0]
    zero = [a for a in below if a["in_F"] >= R3_MIN_DENOM
            and a["h_rule"] == 0.0]
    if rank:
        lo = min(a["h_rule"] for a in rank)
        hi = max(a["h_rule"] for a in rank)
        span = hi / lo if lo > 0 else float("inf")
    else:
        lo = hi = span = float("nan")
    line(fh, "R3 %s: below the gate h_rule spans %.5f-%.5f = %.1fx over %d "
             "rankable architectures, against a %.0fx bar"
         % ("HIT" if span >= R3_SPAN else "MISS", lo, hi, span, len(rank),
            R3_SPAN))
    line(fh, "   excluded: %d with fewer than %d forced emissions%s; "
             "%d measured at exactly zero"
         % (len(thin), R3_MIN_DENOM,
            "" if not thin else " (" + ", ".join(
                "%s %d" % (a["cell"], int(a["in_F"])) for a in thin) + ")",
            len(zero)))
    line(fh, "   compare finding 99: DD-faithful rho_50 spans 0.0173 here, "
             "'narrower than one s*', and the ranking there is noise.")

    # R4
    xs = [a["h_rule"] for a in arch if a["h_rule"] == a["h_rule"]
          and a["rho50"] == a["rho50"]]
    ys = [a["rho50"] for a in arch if a["h_rule"] == a["h_rule"]
          and a["rho50"] == a["rho50"]]
    rs = spearman(xs, ys)
    line(fh, "R4 %s: Spearman(h_rule, published rho_50) = %+.4f over %d "
             "architectures with both defined, against a |%.2f| bar"
         % ("HIT" if abs(rs) < R4_RHO else "MISS", rs, len(xs), R4_RHO))
    line(fh, "   (ties are broken by argsort, the project's existing "
             "spearman; with %d zero-hazard cells that inflates |rho_s| if "
             "anything, so a HIT here is conservative.)"
         % sum(1 for a in arch if a["h_rule"] == 0.0))

    # R5 -- the prediction that answers the brief
    if j2 is None:
        line(fh, "R5 VOID: J2 label %r not found" % J2_LABEL)
    else:
        small = [a for a in arch if smaller_than(a, j2)]
        hits = [a for a in small
                if a["h_rule"] == a["h_rule"] and a["h_rule"] < R1_BAR
                and not (a["rho50"] == a["rho50"] and a["rho50"] <= GATE)]
        line(fh, "R5 %s: %d architectures are strictly smaller objects than "
                 "J2 (%s, %d params); %d of them have h_rule < %.2f while "
                 "FAILING the %.3f gate"
             % ("HIT" if hits else "MISS", len(small), J2_LABEL, j2["params"],
                len(hits), R1_BAR, GATE))
        line(fh, "   J2 itself: h_rule %.5f, rho_50 %.4f, %d forced emissions"
             % (j2["h_rule"], j2["rho50"], int(j2["in_F"])))
        for a in small:
            flag = "RULE-INTACT, GATE-FAILING" if a in hits else ""
            line(fh, "   %-30s %7d params  %d seed%s  h_rule %-9s rho_50 %-9s %s"
                 % (a["cell"], a["params"], a["n_seeds"],
                    " " if a["n_seeds"] == 1 else "s",
                    "undef" if a["h_rule"] != a["h_rule"] else "%.5f" % a["h_rule"],
                    "censored" if a["rho50"] != a["rho50"] else "%.4f" % a["rho50"],
                    flag))
        if hits:
            line(fh)
            # NO SINGLE-SEED RUNG IS A MEASUREMENT OF ITS ARCHITECTURE
            # (findings 41/42, re-taught by 119/120). The project's own
            # non-negotiable is 3 seeds with the spread quoted, so the headline
            # is the smallest SEEDED cell and anything thinner is suggestive
            # only. Reporting the raw smallest without this split is how a
            # lucky single seed becomes a published minimum.
            raw = min(hits, key=lambda a: a["params"])
            seeded = [a for a in hits if a["n_seeds"] >= 3]
            best = min(seeded, key=lambda a: a["params"]) if seeded else None
            if best is None:
                line(fh, "   NO rule-intact gate-failing architecture has 3+ "
                         "seeds; nothing here is a measurement of its "
                         "architecture and no minimum is claimed.")
            else:
                line(fh, "   SMALLEST SEEDED OBJECT WITH DD'S RULE INTACT: %s, "
                         "%d params, %d seeds, h_rule %.5f (per-seed %s) -- it "
                         "FAILS the solving gate at rho_50 %.4f."
                     % (best["cell"], best["params"], best["n_seeds"],
                        best["h_rule"],
                        "/".join("%.4f" % h for _, h, _, _ in best["seeds"]),
                        best["rho50"]))
                line(fh, "   Against J2 (%d params) that is %.2fx smaller as an "
                         "object, on a criterion J2 also satisfies (%.5f)."
                     % (j2["params"], j2["params"] / best["params"],
                        j2["h_rule"]))
            if raw["n_seeds"] < 3:
                line(fh, "   SUGGESTIVE ONLY, NOT THE HEADLINE: %s is smaller "
                         "still at %d params with h_rule %.5f, but it has %d "
                         "seed -- findings 41/42 measured 0.1627 of seed "
                         "spread at fixed architecture, and 119/120 showed a "
                         "single seed flipping a size ordering. It is a "
                         "candidate to seed, not a result."
                     % (raw["cell"], raw["params"], raw["h_rule"],
                        raw["n_seeds"]))
            # The spread within an architecture is the error bar on all of this.
            spreads = [(a["cell"], min(h for _, h, _, _ in a["seeds"]),
                        max(h for _, h, _, _ in a["seeds"]))
                       for a in arch if a["n_seeds"] >= 3]
            wide = max(spreads, key=lambda t: t[2] - t[1])
            line(fh)
            line(fh, "   WIDEST WITHIN-ARCHITECTURE SPREAD among 3+-seed "
                     "cells: %s, %.4f to %.4f. The rule is not a property of "
                     "the architecture alone." % wide)
    line(fh)


def main():
    os.makedirs(OUT, exist_ok=True)
    cells, viol, n_haz, n_summ = load()
    path = os.path.join(OUT, "VERDICT.txt")
    with open(path, "w") as fh:
        line(fh, "STAGE RULE -- the smallest architecture that still learns DD")
        line(fh, "refit of %s" % LOGIC)
        line(fh, "%d hazard rows, %d summary rows, %d in79 cells"
             % (n_haz, n_summ, len(cells)))
        line(fh, "NOTHING SUBMITTED, NO MODEL TRAINED, NO NEW DATA.")
        line(fh)
        ok, named = d1_d2(fh, cells, viol)
        if not ok:
            line(fh, "DIAGNOSTIC FAILURE -- the stage stops here, as "
                     "registered. No prediction is scored.")
            print("\nwrote", path)
            return 1
        arch = per_cell(cells)
        assert len(arch) == 31, "expected 31 cell labels, got %d" % len(arch)
        report(fh, arch)
        score(fh, arch, named)
    print("\nwrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
