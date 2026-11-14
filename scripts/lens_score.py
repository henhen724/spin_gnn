"""REGISTRATION AND SCORER for finding 72: where in the iteration is it decided?

Committed before the measurement job is submitted. Every bar below was written
down here first; the scorer is the registration, so a bar cannot be chosen
after the cells land.

THE QUESTION. Findings 68-71 all read the model's output. STATUS has carried
"no localisation" as a caveat on four decks because WEIGHT-space ablation is
undefined on a tied block: one set of weights shared by 16 applications, so
zeroing a head zeroes it in every round at once. ACTIVATION interventions do
not have that problem -- the state after the t-th application exists in both
arms, and both arms have depth 16, so the application index is a shared axis.
This asks where along it the two things findings 68 and 69 measured appear.

THE MEASUREMENT (spin_gnn/sudoku_lens.py, nothing trained). Finding 68's
byte-identical 350 round-0 cases. Each case has a clean prompt, in which v is
forced to c, and a broken prompt, one hint colour token different, in which v
has two candidates {c, a} and DD must not name v at all. Two readouts, both as
finding 69 defined them: emits_v at the node position (is v the argmax of who
gets named next) and p_freed at the colour position (the mass on a).

  THE LENS. One forward per (case, variant), capturing the state after every
  application and applying the model's OWN final norm and readout to the two
  read positions. Gives emits_v(t), p_freed(t), p_cand(t).

  THE PATCH. Run the BROKEN prompt and splice the CLEAN state back in at one
  (application t, site). recovery(t) is how often the clean behaviour returns.
  Two sites: `edit`, the hint colour token the edit rewrote -- repairing it
  asks when the broken evidence has to be right; and `read`, the node position
  -- repairing it asks when the decision is still movable.

GATES, both fail-closed inside the measurement module as well as here.

  L1  LENS FIDELITY. At t = depth-1 the lens must reproduce the model's own
      output: emits_v agreement 1.0000 and max |p_freed| gap <= 1e-4, every
      case, every variant. The lens is the final computation at t = depth-1 by
      construction, so anything less means the readout is being applied wrongly
      and NOTHING here may be read.
  A2  THE SPLICE IS A NO-OP ON ITS OWN SOURCE. Patching the clean state into
      the CLEAN run must change nothing: 1.0000 of rows with emits_v unchanged
      and |p_freed| drift <= 1e-4. A hook that shifted a position or cloned the
      wrong tensor would otherwise manufacture a recovery curve out of its own
      bug.
  A3  THE GROUPING KEY IS A KEY. (case_id, variant, t) and (case_id, site, t)
      must be unique, and the case_id count must match the metadata. The first
      version of this scorer grouped on `node` -- a node index in 0..47 that
      recurs in dozens of the hundreds of cases the stream yields -- which
      pooled unrelated cases without erroring. The assertion exists so that
      class of bug cannot recur silently.

L3, THE HEADLINE, UNTIED ARM ONLY, and the reason is finding 68: the untied
model is the one with a guard to lose (0.0000 / 0.0000 / 0.8444), and the tied
model never withholds, so on the tied arm recovery of emits_v has nothing to
recover and is not a test. Scored on DISCORDANT cases only -- clean names v,
broken does not -- because those are the cases where the guard acted. At least
DISC_MIN = 50 of them, or the cell is reported THIN and not scored.

  REGISTERED CLAIM: the guard is LOCALISED IN TIME. There is an application t
  with recovery(t) >= 0.80 and recovery(t + SHARP_GAP) <= 0.20, i.e. repair
  stops working across a window of SHARP_GAP = 2 applications. Required in at
  least 2 of 3 untied seeds, at the `edit` site.
  FALSIFIER, and it is a result and not a failure: recovery declines gradually
  over more than that, which says the guard is distributed across the stack.
  Reported as DISTRIBUTED, with the profile, and the deck says so.

L2, L4, L5 are DESCRIPTIVE. They are read whatever L3 does, and no bar is
attached to them, because nothing in the literature or in this project fixes
what a decision time should be.

  L2  DECISION TIME, both arms. t_dec is the last application at which the
      lens's emits_v changes, so the decision is fixed from t_dec on. Median
      per cell, clean and broken.
  L4  WHERE THE AMBIGUITY APPEARS, both arms, and the one the project wants.
      Finding 69: the tied model computes the candidate set (p_cand exactly
      1.0000, p_freed ~0.52) and schedules as if certain. Recovery here is on
      p_freed rather than on emits_v,
          rec_p(t) = (p_freed_patched(t) - p_freed_broken)
                     / (p_freed_clean - p_freed_broken),
      on cases where that denominator is at least FREED_MIN = 0.05 in absolute
      value, since below that it is noise over noise. The deadline is the last
      t with mean rec_p >= 0.50. On the tied arm this is a statement about the
      ITERATED MAP -- the candidate set is resolved by application k -- which
      is what a theory of this object would have to reproduce.
  L5  CONVERGENCE, clean prompt. delta(t) is the relative movement the t-th
      application produced at a read position, and tail8 is the share of it
      happening at t >= 8. A small tail8 means the later applications move
      little. That BEARS ON finding 63 (T*=16 did not help) and explains it;
      it does NOT reopen it. Stage T stays closed.

NOT CLAIMED, and the deck must carry all of it.
  - Nothing is trained. rho_50 does not move and no published number changes.
  - The arms still differ 4.1x in trainable parameters, so every between-arm
    gap here is tying AND budget, exactly as in findings 68-71.
  - A lens is not a causal claim. L2 and L5 are read off intermediate states
    with a readout trained only on the last one; L3 and L4 are the causal
    half, and where the two disagree the patch wins.
  - Round 0 only. This is finding 68's case list, so finding 71's result --
    that the guard is weaker on derived facts and weaker still with depth --
    is OUT OF SCOPE here and cannot be localised by this run.
  - One site at a time. A decision carried redundantly across two positions
    would show as low recovery at both, and this design cannot tell that from
    a decision carried somewhere else entirely.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ddguard_score import ARMS, SEEDS, rows_of, truthy  # noqa: E402

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
# ..._lens holds the first attempt, cancelled 10 minutes in when A3's bug was
# found; its partial S1_s0 cell is left in place rather than deleted, and this
# run writes somewhere else so a stale cell can never be scored by accident.
RUN = os.path.join(_SCR, "spin_gnn", "runs", "2026-11-13_lens2")

DEPTH = 16               # both arms; asserted per cell against lens_meta.json
DISC_MIN = 50            # L3 is not scored on a thinner cell than this
SHARP_HI = 0.80          # recovery still works here ...
SHARP_LO = 0.20          # ... and no longer works here ...
SHARP_GAP = 2            # ... this many applications later
HALF = 0.50              # deadline level, for both recovery kinds
FREED_MIN = 0.05         # minimum |clean - broken| p_freed for an L4 case
LENS_GAP = 1e-4          # L1 and A2 tolerance
SITES = ("edit", "read")


def cell_dir(arm, seed):
    return os.path.join(RUN, "%s_s%d" % (arm, seed))


def _num(x):
    """Every CSV cell is a string and booleans arrive as 'True'; float() raises
    on those. Same helper as dissoc_score, copied rather than rediscovered."""
    s = str(x).strip()
    if s.lower() in ("true", "false"):
        return 1.0 if s.lower() == "true" else 0.0
    return float(s)


def profile(rows, key, cases=None):
    """mean(key) at each t, over the given cases (all of them if None).

    Grouped on `case_id`, never on `node`: `node` is an index in 0..47 and the
    stream yields hundreds of cases over 400 instances, so node 7 recurs in
    dozens of them and a node-keyed set would pool unrelated cases.
    """
    out = []
    for t in range(DEPTH):
        v = [_num(r[key]) for r in rows
             if int(r["t"]) == t and (cases is None or int(r["case_id"]) in cases)]
        out.append(float(np.mean(v)) if v else float("nan"))
    return out


def deadline(prof, level=HALF):
    """The LAST application at which the profile is still at or above level.

    Last rather than first: recovery is expected to fall with t, so the
    quantity of interest is how late a repair can still land, and reading the
    first crossing would report 0 for any profile that starts high.
    """
    hits = [t for t, v in enumerate(prof) if v == v and v >= level]
    return max(hits) if hits else -1


def sharp(prof):
    """The registered localisation test: a drop from SHARP_HI to SHARP_LO
    across SHARP_GAP applications. Returns the t it happens at, or None."""
    for t in range(DEPTH - SHARP_GAP):
        a, b = prof[t], prof[t + SHARP_GAP]
        if a == a and b == b and a >= SHARP_HI and b <= SHARP_LO:
            return t
    return None


def decision_time(traj_rows, variant):
    """L2: per case, the last t at which the lens emits_v changes value.

    0 means the lens had already settled after the first application. Returned
    as the median over cases, which is the only summary quoted, because the
    distribution is integer-valued and small.
    """
    by_case = {}
    for r in traj_rows:
        if r["variant"] != variant:
            continue
        by_case.setdefault(int(r["case_id"]), {})[int(r["t"])] = int(_num(r["emits_v"]))
    out = []
    for seq in by_case.values():
        v = [seq[t] for t in range(DEPTH) if t in seq]
        last = 0
        for t in range(1, len(v)):
            if v[t] != v[t - 1]:
                last = t
        out.append(last)
    return (float(np.median(out)), len(out)) if out else (float("nan"), 0)


def tail_share(traj_rows, key, cut=8):
    """L5: share of a case's total post-first-application movement at t >= cut."""
    by_case = {}
    for r in traj_rows:
        if r["variant"] != "clean":
            continue
        d = _num(r[key])
        if d != d:            # nan at t = 0, by construction
            continue
        by_case.setdefault(int(r["case_id"]), []).append((int(r["t"]), d))
    out = []
    for seq in by_case.values():
        tot = sum(d for _, d in seq)
        if tot > 0:
            out.append(sum(d for t, d in seq if t >= cut) / tot)
    return (float(np.mean(out)), len(out)) if out else (float("nan"), 0)


def score_cell(arm, seed):
    d = cell_dir(arm, seed)
    with open(os.path.join(d, "lens_meta.json")) as fh:
        meta = json.load(fh)
    assert meta["depth"] == DEPTH, "cell %s s%d has depth %d" % (arm, seed, meta["depth"])
    assert meta["tied"] == (arm == "S1"), "cell %s s%d tied=%s" % (arm, seed, meta["tied"])
    traj = rows_of(os.path.join(d, "lens_traj.csv"))
    patch = rows_of(os.path.join(d, "lens_patch.csv"))
    noop = rows_of(os.path.join(d, "lens_noop.csv"))

    # A3: the grouping key has to be a key. Checked rather than assumed,
    # because the first version of this scorer grouped on `node` -- an index in
    # 0..47 that recurs across instances -- and silently pooled unrelated cases.
    assert len({(r["case_id"], r["variant"], r["t"]) for r in traj}) == len(traj)
    assert len({(r["case_id"], r["site"], r["t"]) for r in patch}) == len(patch)
    assert len({int(r["case_id"]) for r in traj}) == meta["n_cases"]
    assert len({int(r["case_id"]) for r in patch}) == meta["n_patch"]

    # ---- the two gates, recomputed here rather than trusted from the module
    last = [r for r in traj if int(r["t"]) == DEPTH - 1]
    l1_agree = float(np.mean([_num(r["emits_v"]) == _num(r["final_emits_v"])
                              for r in last]))
    l1_gap = max(abs(_num(r["p_freed"]) - _num(r["final_p_freed"])) for r in last)
    a2_same = float(np.mean([truthy(r["same_emits_v"]) for r in noop]))
    a2_drift = max(_num(r["abs_freed_drift"]) for r in noop)

    # ---- L3, the discordant cases: clean names v, broken does not
    disc = {int(r["case_id"]) for r in patch
            if truthy(r["clean_emits_v"]) and not truthy(r["broken_emits_v"])}
    # ---- L4's cases: the two prompts have to separate on p_freed at all
    freed = {int(r["case_id"]) for r in patch
             if abs(_num(r["clean_p_freed"]) - _num(r["broken_p_freed"])) >= FREED_MIN}

    cell = {"arm": arm, "seed": seed, "n_cases": meta["n_cases"],
            "n_patch": meta["n_patch"], "n_disc": len(disc), "n_freed": len(freed),
            "L1_agree": l1_agree, "L1_gap": l1_gap,
            "A2_same": a2_same, "A2_drift": a2_drift,
            "L1_pass": bool(l1_agree >= 1.0 and l1_gap <= LENS_GAP),
            "A2_pass": bool(a2_same >= 1.0 and a2_drift <= LENS_GAP)}

    for site in SITES:
        srows = [r for r in patch if r["site"] == site]
        rec = profile(srows, "emits_v", disc) if len(disc) >= DISC_MIN else [float("nan")] * DEPTH
        cell["rec_%s" % site] = rec
        cell["deadline_%s" % site] = deadline(rec)
        cell["sharp_%s" % site] = sharp(rec)

        # L4: recovery of p_freed, normalised per case so cases with a small
        # separation cannot dominate the mean through their numerator.
        per_t = {t: [] for t in range(DEPTH)}
        base = {}
        for r in srows:
            cid = int(r["case_id"])
            if cid in freed:
                base[cid] = (_num(r["clean_p_freed"]), _num(r["broken_p_freed"]))
        for r in srows:
            cid = int(r["case_id"])
            if cid not in base:
                continue
            cl, br = base[cid]
            per_t[int(r["t"])].append((_num(r["p_freed"]) - br) / (cl - br))
        recp = [float(np.mean(per_t[t])) if per_t[t] else float("nan")
                for t in range(DEPTH)]
        cell["recp_%s" % site] = recp
        cell["deadline_p_%s" % site] = deadline(recp)

    for variant in ("clean", "broken"):
        m, n = decision_time(traj, variant)
        cell["tdec_%s" % variant] = m
        cell["tdec_n"] = n
    for key, lab in (("delta_node", "node"), ("delta_col", "col")):
        s, n = tail_share(traj, key)
        cell["tail8_%s" % lab] = s
        cell["tail8_n"] = n
    return cell


def main():
    cells = {}
    for arm in ARMS:
        for s in SEEDS:
            cells["%s_s%d" % (arm, s)] = score_cell(arm, s)

    gates_ok = all(c["L1_pass"] and c["A2_pass"] for c in cells.values())
    untied = [cells["J2_s%d" % s] for s in SEEDS]
    thin = [c for c in untied if c["n_disc"] < DISC_MIN]
    sharp_hits = [c for c in untied if c["sharp_edit"] is not None]

    if not gates_ok:
        verdict = "lens_broken"
    elif len(thin) == len(untied):
        verdict = "thin"
    elif len(sharp_hits) >= 2:
        verdict = "localised"
    else:
        verdict = "distributed"

    print("=" * 78)
    print("FINDING 72: where in the iteration is the decision made?")
    print("=" * 78)
    print("\nGATES")
    print("  cell    L1 agree   L1 gap    A2 same   A2 drift")
    for k in sorted(cells):
        c = cells[k]
        print("  %-6s  %8.4f  %8.1e  %8.4f  %8.1e"
              % (k, c["L1_agree"], c["L1_gap"], c["A2_same"], c["A2_drift"]))

    print("\nL3  recovery of emits_v, discordant cases, site=edit")
    print("  cell    n_disc  deadline  sharp at   recovery t=0,4,8,12,15")
    for k in sorted(cells):
        c = cells[k]
        r = c["rec_edit"]
        print("  %-6s  %6d  %8d  %8s   %s"
              % (k, c["n_disc"], c["deadline_edit"],
                 "-" if c["sharp_edit"] is None else c["sharp_edit"],
                 " ".join("%.2f" % r[t] for t in (0, 4, 8, 12, 15))))

    print("\nL4  recovery of p_freed, site=edit (the ambiguity's deadline)")
    print("  cell    n_freed  deadline  rec_p t=0,4,8,12,15")
    for k in sorted(cells):
        c = cells[k]
        r = c["recp_edit"]
        print("  %-6s  %7d  %8d  %s"
              % (k, c["n_freed"], c["deadline_p_edit"],
                 " ".join("%.2f" % r[t] for t in (0, 4, 8, 12, 15))))

    print("\nL2  lens decision time (median last change)   L5  tail share t>=8")
    print("  cell    clean  broken      node    colour")
    for k in sorted(cells):
        c = cells[k]
        print("  %-6s  %5.1f  %6.1f    %6.3f    %6.3f"
              % (k, c["tdec_clean"], c["tdec_broken"],
                 c["tail8_node"], c["tail8_col"]))

    print("\nVERDICT: %s" % verdict)
    if verdict == "localised":
        print("  L3 CONFIRMED: recovery drops %.2f -> %.2f within %d applications "
              "in %d of 3 untied seeds." % (SHARP_HI, SHARP_LO, SHARP_GAP,
                                            len(sharp_hits)))
    elif verdict == "distributed":
        print("  L3 FALSIFIED as registered: no %d-application window takes "
              "recovery from %.2f to %.2f in 2 of 3 untied seeds. The guard is "
              "distributed, which is the registered alternative and is reported "
              "as the finding." % (SHARP_GAP, SHARP_HI, SHARP_LO))
    elif verdict == "thin":
        print("  Fewer than %d discordant cases in every untied seed; L3 is not "
              "scored and the run is reported as bounded." % DISC_MIN)
    else:
        print("  A gate failed. The lens or the splice is wrong, so nothing "
              "below the gate is read.")

    out = {"verdict": verdict, "gates_ok": gates_ok,
           "n_sharp": len(sharp_hits), "cells": cells,
           "bars": {"DISC_MIN": DISC_MIN, "SHARP_HI": SHARP_HI,
                    "SHARP_LO": SHARP_LO, "SHARP_GAP": SHARP_GAP,
                    "HALF": HALF, "FREED_MIN": FREED_MIN, "LENS_GAP": LENS_GAP}}
    path = os.path.join(RUN, "lens_summary.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
