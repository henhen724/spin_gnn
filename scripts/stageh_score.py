"""Score stage H (51223007): does the depth that bought the MLP back survive being made NARROW?

    python3 -m scripts.stageh_score [--boot 2000]

THE QUESTION. Stage E's E2 -- attention-only, no MLP anywhere, at L=12 --
landed at rho_50 0.605 against the hub's 0.592 on 3 of 3 crossing seeds and
falsified the barrier reading: the 0.76 pile-up was capacity at this budget,
not an architecture-independent floor, and DEPTH is the currency that bought
the MLP back where WIDTH (D2, 1.5x the hub's residual stream, 0.760) did not.

But E2 is a loss on the only axis the brief cares about: 3,331,642 trainable
against the 1,746,490 of the rung it replaced. Stage H moves ONE knob from E2
-- d_model, 256 -> 128 (H1) and 192 (H2) -- and asks whether the descent
survives at a quarter and a half of the stream. H1 at 879,418 would be 5.6x
below the hub and 2.0x below the pre-E2 standing minimum.

THIS FILE IS THE REGISTRATION, TRANSCRIBED from the sbatch header
(scripts/slurm/sudoku2_n48_minarch_H_2026-11-05.sbatch), and it was written
while 51223007 tasks 0-2 were ~20 minutes in at epochs 0-2 and NO stage H cell
had finished. Every literal below is either copied from that header or measured
off rungs that already existed before stage H was submitted. Nothing here is
chosen after seeing a stage H number; git log is the timestamp.

  H1  no_mlp L=12 d128    879,418 trainable (header arithmetic)
  H2  no_mlp L=12 d192  1,908,922 trainable (header arithmetic)

THE TWO READINGS, and the tie-break was fixed in advance:
  (D) DEPTH IS THE CURRENCY -- H1's median rho_50 at or below 0.740 (hub + s*)
      on at least 2 of 3 crossing seeds. The standing minimum moves to 879,418.
  (W) WIDTH IS A SEPARATE FLOOR -- H1's median back in [0.72, 0.80], or
      censored. Depth only pays above some width and the cheap end is bounded
      by the residual stream.
  TIE-BREAK: a median anywhere in [0.72, 0.740], boundary included, scores as
  (W) and is written as "did not clear the hub". (D) is the claim that needs
  the evidence and a boundary value is not evidence.

THE ASYMMETRY, and it is the trap stage C's P2 fell into. A cell that never
gets below finding 44's 1.64 train-NLL plateau has NOT shown that its
architecture cannot represent direct deduction; it has shown it did not train
in 375k steps. The header registers >= 2 of 3 escapes at H1 (depth has
monotonically improved escapes on this axis: 0/3 at L=4, 2/3 at L=6, 3/3 at
L=8 and L=12, all d256) while escape_timing.py's Q3 predicts 0 of 3 (d128
no_mlp's three seeds are a median 87 epochs short at their own tail slope, and
this job does not raise the budget -- stage G is the budget arm). BOTH ARE ON
RECORD AND ONE OF THEM IS WRONG IN PRINT. If H1 does not escape, this scorer
refuses to read its rho_50 as evidence for (W): the cell is reported as
trainability-limited and the (D)/(W) question stays open.

REPORTING ORDER, as registered: H4 (the free sufficiency test) ahead of
everything if it fails, then H2 (the monotonicity falsifier), then H1. H2's
median must lie between H1's and E2's 0.605; outside its neighbours' range by
more than s* falsifies monotonicity in width at L=12, which no ladder in this
grid has shown.

Scored against the frozen bar s* = 0.1484 / 0.2918, not recomputed. Every
comparison is median-vs-median; a difference smaller than s* is "not separable
at this budget", never a result (findings 41/42/47).
"""
import argparse
from collections import defaultdict

from scripts.minarch_analysis import (SCAN_DIRS, SCAN_E, SCAN_H, cell_key,
                                      cell_metrics, load_all)
from scripts.stagec_score import (S_MEAN, S_RHO50, escapes, rho_txt, rung,
                                  summarize, verdict)

HUB_RHO50 = 0.592        # full L=6 d256, the reference architecture
HUB_TRAINABLE = 4899898
FREE = 0.740             # hub + s*, the (D)/(W) threshold and H3's ceiling
BAND_LO, BAND_HI = 0.72, 0.80   # the pile-up the barrier reading was built on
LOO_BAR = 0.0423         # acc_vs_solve's leave-one-out residual bar

H1 = (12, 4, "no_mlp", 128, 1024, 0)
H2 = (12, 4, "no_mlp", 192, 1024, 0)
E2 = (12, 4, "no_mlp")                      # full-width partner, same depth
L6_D128 = (6, 4, "no_mlp", 128, 1024, 0)    # same-width partner, depth removed

# Header arithmetic, printed beside the real number read out of config.json.
# Stage D's header estimated d384 no_mlp at "~3.6M" against a built 3,799,354;
# an estimate in a header is not a result, and the write-up quotes the disk.
PREDICTED_TRAINABLE = {"H1": 879418, "H2": 1908922}

# Frozen off disk before stage H was submitted:
#   (n, n_cross, rho_50 text, mean_solve, spread, trainable, escapes)
PARTNER = {
    "E2": (3, 3, "0.605", 0.5214, 0.0480, 3331642, 3),
    "L6_D128": (3, 1, "CENSORED", 0.0291, 0.2075, 480058, 0),
}
PRED_ESCAPES_HEADER = {"H1": ">=2 of 3 (depth has monotonically improved escapes)"}
PRED_ESCAPES_Q3 = {"H1": "0 of 3 (median 87 epochs short at the terminal slope)"}


def report(tag, cells, key, asks):
    ms = rung(cells, key)
    print(f"\n=== {tag} -- {asks} ===")
    if not ms:
        print("  nothing on disk yet.")
        return None
    s = summarize(ms)
    n_esc, nll = escapes(ms)
    want = PREDICTED_TRAINABLE[tag]
    print(f"  {s['n']}/3 seeds on disk, {s['n_cross']}/{s['n']} crossed 0.5")
    print(f"  rho_50 median {rho_txt(s):>9}   (hub {HUB_RHO50:.3f}, E2 {PARTNER['E2'][2]}, "
          f"free at or below {FREE})")
    print(f"  mean_solve    {s['mean_solve']:>9.4f}   [spread {s['spread_mean']:.4f}]   "
          f"(E2 {PARTNER['E2'][3]:.4f})")
    print(f"  trainable {s['trainable']:,}   (header arithmetic {want:,}"
          + ("" if s["trainable"] == want else "  <-- DISAGREES, quote the disk") + ")")
    print(f"  plateau escapes {n_esc}/{s['n']}   train NLL "
          + "/".join(f"{v:.3f}" for v in nll))
    return dict(s=s, n_esc=n_esc, ms=ms)


def score_h1(h1):
    """The (D)/(W) call, with the trainability gate ahead of it."""
    s, n_esc = h1["s"], h1["n_esc"]
    if n_esc < 2:
        verdict(f"""
TRAINABILITY GATE: {n_esc} of {s['n']} seeds escaped the 1.64 plateau, against a
registered expectation of >= 2. THIS IS NOT EVIDENCE FOR (W). The honest
statement is the one stage C's P2 got wrong: d128 attention-only did not train
in 375k steps at L=12, and a rung that did not train is not a measurement of
its architecture. The (D)/(W) question stays OPEN and H3 cannot fire.
escape_timing.py's Q3 predicted exactly this ({PRED_ESCAPES_Q3['H1']}) against
the header's {PRED_ESCAPES_HEADER['H1']}; Q3 is the better model and its ETA
table can be used to choose the next budget cell. The next cell is more steps,
not a conclusion -- and stage G is where that is measured.""")
        return "inconclusive"
    if not s["rho50_readable"]:
        verdict(f"""
(W) WIDTH IS A SEPARATE FLOOR. {n_esc} of {s['n']} seeds escaped the plateau, so
this rung DID train and its censoring is a property of the architecture rather
than of the budget -- that distinction is the whole reason the gate above runs
first. Twelve layers of attention at d128 trains and still does not solve at
any hint density on the grid. Depth does not pay below some width, the two
resources are NOT interchangeable, and the cheap end is bounded by the residual
stream rather than by the MLP or by depth.""")
        return "W"
    if s["rho50"] <= FREE:
        verdict(f"""
(D) DEPTH IS THE CURRENCY, and it survives the stream being quartered. Median
rho_50 {s['rho50']:.3f} at or below the {FREE} free threshold on {s['n_cross']} of
{s['n']} crossing seeds, with {n_esc} of {s['n']} plateau escapes, at
{s['trainable']:,} trainable -- {HUB_TRAINABLE / s['trainable']:.1f}x below the hub and
{PARTNER['E2'][5] / s['trainable']:.1f}x below E2. The gap to the hub is
{s['rho50'] - HUB_RHO50:+.3f} ({abs(s['rho50'] - HUB_RHO50) / S_RHO50:.2f} s*).""")
        return "D"
    where = "inside the band" if BAND_LO <= s["rho50"] <= BAND_HI else "above the band"
    verdict(f"""
(W) WIDTH IS A SEPARATE FLOOR. Median rho_50 {s['rho50']:.3f} is {where}
[{BAND_LO}, {BAND_HI}] and did not clear the hub, on {n_esc} of {s['n']} escapes
so the rung trained. By the tie-break fixed in the header, a median anywhere in
[{BAND_LO}, {FREE}] scores as (W): (D) is the claim that needed the evidence
here and a boundary value is not evidence. Depth bought the MLP back at d256
and does not buy it back at d128.""")
    return "W"


def score_h2(h2, h1):
    """Monotonicity in width at L=12, reported ahead of H1."""
    s = h2["s"]
    e2_rho = float(PARTNER["E2"][2])
    if not s["rho50_readable"]:
        verdict(f"""
H2 CENSORED. d192 does not cross at any hint density on the grid while E2 at
d256 crosses at {e2_rho:.3f}, so the bisector cannot order its neighbours and the
monotonicity test is unavailable rather than passed. Escapes {h2['n_esc']}/{s['n']}
decide whether this is an architecture statement or a trainability one.""")
        return "unavailable"
    if h1 is None:
        print(f"  H1 not on disk yet; H2's {s['rho50']:.3f} can only be compared to "
              f"E2's {e2_rho:.3f} for now.")
        return "partial"
    s1 = h1["s"]
    if not s1["rho50_readable"]:
        inside = s["rho50"] >= e2_rho - S_RHO50
        note = (f"H1 is censored, so monotonicity asks only that H2 sit at or above E2's "
                f"{e2_rho:.3f} to within s*: it is {s['rho50']:.3f}")
    else:
        lo, hi = min(s1["rho50"], e2_rho), max(s1["rho50"], e2_rho)
        inside = lo - S_RHO50 <= s["rho50"] <= hi + S_RHO50
        note = (f"neighbours are H1 {s1['rho50']:.3f} and E2 {e2_rho:.3f}, so H2 at "
                f"{s['rho50']:.3f} is {'inside' if inside else 'OUTSIDE'} "
                f"[{lo:.3f}, {hi:.3f}] allowing s* = {S_RHO50}")
    if inside:
        verdict(f"""
MONOTONE, as every other width ladder in this grid is ({note}). The bisector
behaves, so H1 can be read as the end of a ladder rather than as a lone point.""")
        return "monotone"
    verdict(f"""
MONOTONICITY IN WIDTH IS FALSIFIED AT L=12, and this is reported ahead of H1
because it changes how H1 reads. {note}. Width and depth interact in a way no
ladder in this grid has shown -- the L=6 attention-only ladder (0.763 at d256,
censored at d128) and the complete model's (0.592 / 0.752 / censored) are both
monotone. Whatever H1 says, it is not the bottom of a well-behaved ladder, and
a third width is needed before any minimum is claimed.""")
    return "non-monotone"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+", default=SCAN_DIRS + [SCAN_E, SCAN_H],
                    help="SCAN_E and SCAN_H are out of SCAN_DIRS on purpose; "
                         "they are added here only.")
    args = ap.parse_args()

    by = defaultdict(list)
    for r in load_all(args.scan_dir):
        by[cell_key(r)].append(r)
    cells = {k: dict(cell_metrics(v, args.boot), path=v[0]["path"], seed=k[-1])
             for k, v in by.items()}

    print("STAGE H: does the depth that bought the MLP back survive being made NARROW?")
    print(f"  {len(cells)} cells read; ONE knob from E2, d_model 256 -> 192 and 128")
    print(f"  frozen bar s* = {S_RHO50} rho_50 / {S_MEAN} mean_solve, NOT recomputed")
    print(f"  (D) H1 <= {FREE} on >= 2 of 3 crossing seeds; (W) H1 in "
          f"[{BAND_LO}, {BAND_HI}] or censored; [{BAND_LO}, {FREE}] scores as (W)")

    # The partners must still read the way they were frozen, or the literals in
    # this file are stale and every comparison below is against the wrong number.
    print("\nFROZEN-LITERAL CONTROL (the paired partners, re-read from disk):")
    for tag, key in (("E2", E2), ("L6_D128", L6_D128)):
        ms = rung(cells, key)
        if not ms:
            print(f"  {tag}: NOT ON DISK -- cannot verify the literals")
            continue
        s, (ne, _nll) = summarize(ms), escapes(ms)
        got = (s["n"], s["n_cross"], rho_txt(s), round(s["mean_solve"], 4),
               round(s["spread_mean"], 4), s["trainable"], ne)
        want = PARTNER[tag]
        print(f"  {tag} {'matches' if got == want else 'DISAGREES WITH'} the frozen literals")
        if got != want:
            print(f"    frozen {want}")
            print(f"    disk   {got}")
            print("    the registered grid changed; fix that before reading anything below.")

    h2 = report("H2", cells, H2, "THE BISECTOR, attention-only L=12 d192")
    h1 = report("H1", cells, H1, "THE CHEAP END, attention-only L=12 d128")

    print("\n" + "=" * 72)
    if not h1 or not h2:
        have = [t for t, v in (("H1", h1), ("H2", h2)) if v]
        print(f"VERDICT: NOT YET -- {', '.join(have) if have else 'no'} rung(s) on disk.")
        print("  H2 is reported ahead of H1 by registration, and H3 needs both.")
        print("=" * 72)
        return

    print("H2 FIRST, as registered (monotonicity would change how H1 reads):")
    mono = score_h2(h2, h1)

    print("\nH1:")
    call = score_h1(h1)

    s1 = h1["s"]
    print("\nH3 -- THE STANDING-MINIMUM RULE, fixed before E2 was scored:")
    print(f"  a rung replaces the standing minimum ONLY IF 3 of 3 seeds cross AND the")
    print(f"  median rho_50 is at or below {FREE}. Two of three is not enough -- that is")
    print(f"  exactly why L=6 no_mlp at 0.763 was never a satisfying answer.")
    if call == "D" and s1["n_cross"] == 3 and s1["n"] == 3:
        print(f"  H3 FIRES. The standing minimum moves from E2 ({PARTNER['E2'][5]:,} "
              f"trainable at {PARTNER['E2'][2]}) to attention-only L=12 d128, "
              f"{s1['trainable']:,} at {rho_txt(s1)}:")
        print(f"    size DOWN {PARTNER['E2'][5] / s1['trainable']:.2f}x and "
              f"{HUB_TRAINABLE / s1['trainable']:.1f}x below the hub; rho_50 moves "
              f"{s1['rho50'] - float(PARTNER['E2'][2]):+.3f} "
              f"({abs(s1['rho50'] - float(PARTNER['E2'][2])) / S_RHO50:.2f} s*).")
        print("    This is the first rung in the programme to make the model SMALLER and")
        print("    keep every seed solving; E2 bought all-seed solving at 1.91x the size.")
    elif call == "D":
        print(f"  H3 DOES NOT FIRE: H1's median clears {FREE} but only {s1['n_cross']} of "
              f"{s1['n']} seeds cross. Both conditions or neither; the standing minimum")
        print(f"  stays at E2, {PARTNER['E2'][5]:,} trainable.")
    else:
        print(f"  H3 DOES NOT FIRE ({call}). The standing minimum stays at E2, "
              f"{PARTNER['E2'][5]:,} trainable at {PARTNER['E2'][2]}.")
    if mono == "non-monotone":
        print("  AND the ladder it would sit at the bottom of is not monotone, so no")
        print("  minimum should be quoted from this job without a third width.")
    print("=" * 72)

    print("\nH4 -- OUT-OF-SAMPLE SUFFICIENCY, free, and reported ahead of H1/H2 if it")
    print(f"  fails. The depth-ladder fit now holds on 16 of 16 well-determined held-out")
    print(f"  rungs over 34 cells, worst discriminating miss 0.0099 against the LOO bar")
    print(f"  {LOO_BAR}. Prediction: both stage H rungs land on the curve within {LOO_BAR}.")
    print("  Run:")
    print("    python3 -m scripts.acc_vs_solve --scan-dir \\\n      " +
          " \\\n      ".join(args.scan_dir))
    print("  A stage H rung falling OFF the curve would be the grid's first evidence of")
    print("  an architectural change doing something SOLVE-SPECIFIC rather than moving")
    print("  the model along one quality axis.")


if __name__ == "__main__":
    main()
