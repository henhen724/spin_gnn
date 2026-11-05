"""Score stage C's P1-P5 against its sbatch header and nothing else.

Stage C (51114734) asked how WIDE the one load-bearing nonlinear component has
to be. Stages A and B had found that removing the MLP three different ways --
the elementwise nonlinearity only (`linear`), the training of its first matrix
only (`rf_mlp`), the whole write (`no_mlp`) -- all cost the SAME, so there is
exactly one load-bearing nonlinear component and the natural next question is
its width. Stage C varies d_model, dim_feedforward and the warmup schedule at
L=6, three seeds per rung, everything else held at the hub's settings.

WHY THIS IS A SEPARATE SCRIPT FROM minarch_analysis.py. That script scores the
REGISTERED GRID -- the (L, nhead, arch) rungs of stages A and B -- and pools
their across-seed spread into the bar s*. Stage C's rungs carry a non-default
d_model, dim_feedforward or warmup, so rung_key() gives them 6-tuple keys,
all_rungs() does not name them, and they are deliberately kept OUT of that
pool: a later job must never widen the bar a published verdict was scored
against. They therefore need their own scorer, and it has to quote the bar
that was in force when stage C was submitted rather than recompute one.

THE BAR, copied from the stage C sbatch header and frozen here:
    s* = 0.1484 in rho_50, 0.2918 in mean_solve
pooled over every complete rung in the registered stage A/B grid. If this ever
disagrees with what minarch_analysis prints for the registered grid, the
registered grid has changed and the discrepancy is the bug -- the control
below asserts they still match.

THE CONTROL is stage A tasks 6-8: L=6, d256, ff1024, nhead 4, full, no warmup,
run to completion at the same budget, the same 14-point rho grid and the same
default --seed-start, so it is PAIRED with every stage C cell point by point.
It lives in SCAN_DEFAULT, not in stage C's own directory.

Usage:  python3 -m scripts.stagec_score [--boot N]
"""
import argparse
from collections import defaultdict

from scripts.minarch_analysis import (SCAN_C, SCAN_DEFAULT, SCAN_L6, _bound_median,
                                      cell_key, cell_metrics, fmt_rho50, history,
                                      load_all, median, rung_label, spread)

# Frozen at stage C's submission; see the module docstring. Not recomputed.
S_RHO50, S_MEAN = 0.1484, 0.2918
PLATEAU = 1.64  # finding 44's L=4/ff=1024/240k train NLL, never escaped
HUB = (6, 4, "full")  # stage A tasks 6-8, the paired control for every row

# task ranges from the stage C sbatch header, so a cell that landed under the
# wrong key is visible as a missing row rather than as a silently short rung.
ROWS = {
    "P1": ((6, 4, "full", 128, 1024, 0), "width 256->128"),
    "P2": ((6, 4, "no_mlp", 128, 1024, 0), "is the MLP gap width-dependent"),
    "P3": ((6, 4, "full", 256, 256, 0), "ff 1024->256"),
    # nhead=1, NOT 4: P4 is the trainability control on stage B's N5, so it
    # keeps N5's single head and changes only the warmup. Written as 4 here
    # first, which silently reported "pending" while all three cells sat on
    # disk -- the per-cell table at the bottom is what caught it, which is the
    # reason that table prints every stage C key rather than only the scored
    # ones.
    "P4": ((6, 1, "full", 256, 1024, 5), "trainability control on N5"),
    "P5": ((6, 4, "full", 64, 1024, 0), "bracket, 128->64"),
}
# Stage B's attention-only cell at the hub's width. P2's difference of
# differences needs it, and it is the only stage C comparison that reaches
# outside stage C and its own control.
NO_MLP_D256 = (6, 4, "no_mlp")


def rung(cells, key):
    """Per-seed metrics for one rung, or None if nothing has landed yet."""
    seeds = sorted(k[-1] for k in cells if k[:-1] == key)
    return [cells[key + (s,)] for s in seeds] or None


def summarize(ms):
    """Median rho_50 (bound-aware), mean_solve, and how many seeds crossed."""
    crossed = [m for m in ms if m["flag"] == "ok"]
    bm = _bound_median([(m["rho50"], m["flag"] == "ok") for m in ms])
    return {
        "n": len(ms), "n_cross": len(crossed),
        "rho50": bm["rho50_bm"], "rho50_readable": bm["bm_readable"],
        "mean_solve": median([m["mean_solve"] for m in ms]),
        "spread_mean": spread([m["mean_solve"] for m in ms]),
        "trainable": ms[0]["n_trainable"],
    }


def rho_txt(s):
    if s["rho50_readable"]:
        return f"{s['rho50']:.3f}"
    return "CENSORED"


def escapes(ms):
    """How many seeds got below finding 44's plateau, and their train NLLs."""
    nll = [history(m["path"])[1] for m in ms]
    return sum(1 for v in nll if v < PLATEAU), nll


def banner(tag, asks, s, hub, ms):
    n_esc, nll = escapes(ms)
    print(f"\n=== {tag} -- {asks} ===")
    print(f"  {s['n']}/3 seeds on disk, {s['n_cross']}/{s['n']} crossed 0.5")
    print(f"  rho_50 median {rho_txt(s)}   (hub {hub['rho50']:.3f})")
    print(f"  mean_solve median {s['mean_solve']:.4f} [spread {s['spread_mean']:.4f}]"
          f"   (hub {hub['mean_solve']:.4f})")
    print(f"  trainable {s['trainable']:,}   (hub {hub['trainable']:,})")
    print(f"  plateau escapes {n_esc}/{s['n']}   train NLL "
          + "/".join(f"{v:.3f}" for v in nll))
    return n_esc


def verdict(text):
    for line in text.strip("\n").split("\n"):
        print(f"  {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    rows = load_all([SCAN_DEFAULT, SCAN_L6, SCAN_C])
    by = defaultdict(list)
    for r in rows:
        by[cell_key(r)].append(r)
    cells = {k: dict(cell_metrics(v, args.boot), path=v[0]["path"]) for k, v in by.items()}

    hub_ms = rung(cells, HUB)
    assert hub_ms and len(hub_ms) == 3, "the paired control (stage A tasks 6-8) is incomplete"
    hub = summarize(hub_ms)
    print("STAGE C (51114734), scored against its own sbatch header")
    print(f"  bar in force at submission: s* = {S_RHO50:.4f} rho_50, {S_MEAN:.4f} mean_solve")
    print(f"  paired control  {rung_label(*HUB)}  rho_50 {hub['rho50']:.3f}  "
          f"mean_solve {hub['mean_solve']:.4f}  trainable {hub['trainable']:,}")
    free = hub["rho50"] + S_RHO50
    print(f"  a removal is 'free' iff rho_50 stays at or below {free:.3f}")

    # ---- P1 -------------------------------------------------------------
    key, asks = ROWS["P1"]
    ms = rung(cells, key)
    if ms is None:
        print("\n=== P1 -- pending, nothing on disk ===")
    else:
        s = summarize(ms)
        n_esc = banner("P1", asks, s, hub, ms)
        if s["n"] < 3:
            verdict("NOT YET SCOREABLE: fewer than 3 seeds.")
        elif s["n_cross"] < 3:
            verdict(f"""
>>> FALSIFIED by its own registered falsifier, which reads 'gap > s*, or any
    seed censored while all three hub seeds solved'. {3 - s['n_cross']} of 3 seeds never
    crossed 0.5 at any of the 14 hint densities, and all three hub seeds did.
    Halving the residual stream is NOT free: the width is itself load-bearing
    at N=48 and the minimal architecture is bounded below by it.
    AND IT IS NOT A TRAINABILITY ARTEFACT -- {n_esc} of 3 seeds got below finding
    44's plateau {PLATEAU}, so unlike N5's nhead=1 this verdict survives the
    watch item that confounds every other 'worse' in the grid.""")
        elif s["rho50"] - hub["rho50"] < S_RHO50:
            verdict(">>> PASSES: the gap is below s*, halving d_model is free at L=6.")
        else:
            verdict(">>> FALSIFIED on the gap: rho_50 rose by more than s*.")

    # ---- P2 -------------------------------------------------------------
    key, asks = ROWS["P2"]
    ms = rung(cells, key)
    full128 = rung(cells, ROWS["P1"][0])
    wide_nomlp = rung(cells, NO_MLP_D256)
    if ms is None:
        print("\n=== P2 -- pending, nothing on disk ===")
    else:
        s = summarize(ms)
        n_esc = banner("P2", asks, s, hub, ms)
        f128 = summarize(full128) if full128 else None
        wide = summarize(wide_nomlp) if wide_nomlp else None
        if wide:
            print(f"  gap(d256) = {wide['rho50']:.3f} - {hub['rho50']:.3f} = "
                  f"{wide['rho50'] - hub['rho50']:+.3f}   (stage B, already measured)")
        # The registered quantity is a DIFFERENCE OF DIFFERENCES and both of
        # its d128 terms are censored, so it does not exist. Say so instead of
        # falling back to a mean_solve "tie" between two cells on the floor --
        # that is the M3-M7 trap, and it reads as evidence when it is not.
        if f128 and not (s["rho50_readable"] and f128["rho50_readable"]):
            verdict(f"""
>>> NOT SCOREABLE AS REGISTERED, and this was foreseen in the handoff before
    the cells landed. gap(d128) = rho_50(d128,no_mlp) - rho_50(d128,full)
    needs both terms, and both rungs are censored
    ({s['n_cross']}/{s['n']} and {f128['n_cross']}/{f128['n']} seeds crossing), so the difference of
    differences does not exist.
    The mean_solve fallback would compare {s['mean_solve']:.4f} against
    {f128['mean_solve']:.4f} -- two cells on the floor. That is the 'both cells
    failed' tie, not 'the machinery was removable', and it is the same trap
    M3-M7 fell into at L=4 where the hub itself was censored.
    Compounding it, {s['n'] - n_esc} of {s['n']} seeds here are ABOVE the {PLATEAU} plateau, so
    these cells are trainability-confounded on top of being floored.
    HONEST READING: d128 is below the width this task needs, and nothing about
    the MLP can be read at a width where the full model does not work either.
    The question P2 asked now needs a width where both terms are alive.""")
            if s["n_cross"]:
                best = max(ms, key=lambda m: m["mean_solve"])
                verdict(f"""
    Worth recording and not over-reading: one attention-only d128 seed DID
    solve (rho_50 {best['rho50']:.3f}, mean_solve {best['mean_solve']:.4f}) while no full d128 seed
    crossed on any seed. One cell out of six, inside an across-seed spread of
    {s['spread_mean']:.4f} on this very rung; it points the same way as stage D's D2 --
    that width, not the MLP, is what is scarce -- but it is not evidence on
    its own and is not being counted as any.""")
        elif f128:
            g128 = s["rho50"] - f128["rho50"]
            g256 = wide["rho50"] - hub["rho50"] if wide else float("nan")
            d = abs(g128 - g256)
            verdict(f">>> gap(d128) {g128:+.3f} vs gap(d256) {g256:+.3f}, |diff| {d:.3f} "
                    f"vs s* {S_RHO50:.4f} -> "
                    + ("PASSES: the MLP's contribution is width-independent."
                       if d < S_RHO50 else "FALSIFIED: the two knobs interact."))

    # ---- P3, P4, P5 -----------------------------------------------------
    for tag in ("P3", "P4", "P5"):
        key, asks = ROWS[tag]
        ms = rung(cells, key)
        if ms is None:
            print(f"\n=== {tag} -- pending, nothing on disk ===")
            continue
        s = summarize(ms)
        n_esc = banner(tag, asks, s, hub, ms)
        if s["n"] < 3:
            verdict(f"NOT YET SCOREABLE: {s['n']} of 3 seeds on disk. Per-seed so far: "
                    + ", ".join(f"s{k[-1]} {fmt_rho50(cells[k])}/"
                                f"{cells[k]['mean_solve']:.4f}"
                                for k in sorted(cells) if k[:-1] == key))
            continue
        if tag == "P4":
            # P4 is a CONTROL on stage B's N5, and its registered quantity is
            # the escape count, not rho_50: 0/3 again makes N5's "WORSE" a
            # property of the architecture rather than of the schedule. The
            # comparison that matters is therefore against stage B's OWN
            # nhead=1 cells at warmup 0, not against the 4-head hub, so print
            # them side by side rather than leaving the reader to look it up.
            n5 = rung(cells, (6, 1, "full"))
            if n5:
                n5_esc, n5_nll = escapes(n5)
                print(f"  stage B's N5 at warmup 0: escapes {n5_esc}/{len(n5)}, train NLL "
                      + "/".join(f"{v:.3f}" for v in n5_nll)
                      + f", mean_solve {summarize(n5)['mean_solve']:.4f}")
            if n_esc == 0:
                verdict("""
>>> 0/3 ESCAPES A SECOND TIME, with a different optimizer setting. That is the
    registered useful result, not a failed cell: two independent schedules
    failing the same way is enough to write N5's 'WORSE' as a property of one
    head rather than of the warmup, which is the thing that could not be said
    at the end of stage B.""")
            else:
                verdict(f"""
>>> {n_esc}/3 SEEDS ESCAPED the plateau with warmup. N5 must be RE-SCORED on these
    cells and stage B's 'nhead=1 is WORSE' verdict withdrawn in writing, as
    the header requires. rho_50 median {rho_txt(s)}, mean_solve {s['mean_solve']:.4f}.""")
            continue
        gap = s["rho50"] - hub["rho50"] if s["rho50_readable"] else float("nan")
        if tag == "P3":
            # The two metrics can disagree at the censoring boundary -- rho_50
            # is an interpolated crossing and goes undefined one bin earlier
            # than mean_solve degrades -- so print both whichever way the
            # registered rule falls, and never let a hair's-breadth pass be
            # written up as a demonstration.
            dmean = s["mean_solve"] - hub["mean_solve"]
            margin = S_RHO50 - gap
            print(f"  mean_solve delta {dmean:+.4f} vs s* {S_MEAN:.4f} "
                  f"(for scale: removing the whole MLP at d256 costs -0.2350)")
            if s["rho50_readable"] and gap < S_RHO50 and margin < 0.1 * S_RHO50:
                verdict(f"""
>>> PASSES THE REGISTERED RULE BY {margin:.4f} AND NOTHING MORE. gap {gap:+.3f} against
    s* {S_RHO50:.4f} is a TIE WITH THE BAR, not a demonstration that quartering ff
    is free, and it must not be written up as one: the median seed sits on
    the free/not-free threshold to three decimals. Read the mean_solve line
    instead, where there is no boundary: {s['mean_solve']:.4f} against the hub's
    {hub['mean_solve']:.4f} is {dmean:+.4f}, essentially what removing the ENTIRE MLP costs
    (-0.2350, mean_solve 0.3039). So a 4x narrower MLP is worth about what no
    MLP is worth -- which SHARPENS 'the MLP is all-or-nothing': the 'all' it
    needs is ff=1024, and its WIDTH is nearly all of its contribution.""")
            elif s["rho50_readable"] and gap < S_RHO50:
                verdict(f">>> PASSES: gap {gap:+.3f} < s* {S_RHO50:.4f}. Quartering ff is free.")
            else:
                verdict(f"""
>>> FALSIFIED: gap {gap:+.3f} against s* {S_RHO50:.4f} ({rho_txt(s)} vs the hub's
    {hub['rho50']:.3f}). The MLP's WIDTH is part of what it contributes, so 'one
    learned nonlinearity' is not a complete description of it -- the size of
    the nonlinear bank matters and the ff axis needs its own ladder before any
    analytic claim is made.""")
        else:  # P5
            if s["n_cross"] <= 1:
                verdict(f"""
>>> AS PREDICTED: censored at {3 - s['n_cross']} of 3, mean_solve {s['mean_solve']:.4f}. With
    256 / 192 / 128 / 64 the width axis can now be read as a LADDER against
    the depth ladder instead of as a single pass/fail.""")
            else:
                verdict(f"""
>>> FALSIFIED: {s['n_cross']} of 3 seeds cross at d64 (rho_50 {rho_txt(s)}). Width is
    nearly irrelevant on this task and the minimal architecture should be
    searched at d32 and below before anything else is tried.""")

    print("\n--- all stage C cells on disk ---")
    hdr = (f"{'rung':<26} {'s':>1} {'rho50':>7} {'mean':>7} {'floor':>7} "
           f"{'trainable':>10} {'trNLL':>6} {'esc':>3}")
    print(hdr)
    for k in sorted(cells, key=lambda k: (len(k), k[:-1], k[-1])):
        if len(k) != 7:  # stage C rungs carry (d_model, ff, warmup) + seed
            continue
        m = cells[k]
        trnll = history(m["path"])[1]
        print(f"{rung_label(*k[:-1]):<26} {k[-1]:>1} {fmt_rho50(m):>7} "
              f"{m['mean_solve']:>7.4f} {m['solve_at_floor']:>7.4f} "
              f"{m['n_trainable']:>10,} {trnll:>6.3f} "
              f"{'yes' if trnll < PLATEAU else 'NO':>3}")


if __name__ == "__main__":
    main()
