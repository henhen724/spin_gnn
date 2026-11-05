"""Score stage I (51246202): where exactly is the width trainability floor?

    python3 -m scripts.stagei_score [--boot 2000]

THE QUESTION, AND IT IS A BRACKET, NOT A DIRECTION. Stage H put two cells of
attention-only L=12 at d128 on disk and both of them are censored with final
train_nll 1.6768 and 1.6703, NEVER below finding 44's 1.64 plateau in 375k
steps. Its first d192 cell was at train_nll 1.5670 by epoch 40 and had already
escaped. So at this depth and this budget the model either finds the algorithm
or it does not, the switch is somewhere in d in [128, 192], and no cell in the
grid sits inside that factor of 1.5. Stage I is one rung, d160, and it is the
bisector.

THIS FILE IS THE REGISTRATION, TRANSCRIBED from the sbatch header
(scripts/slurm/sudoku2_n48_minarch_I_2026-11-05.sbatch), and it was written
while 51246202 was pending and 51223007 was still running. Every literal below
is either copied from that header or measured off cells that were already on
disk when stage I was submitted. Nothing here is chosen after seeing a stage I
number; git log is the timestamp.

  I1  no_mlp L=12 d160  1,345,018 trainable (header arithmetic:
      4*d^2 + 8*d per layer, 630*d + 58 fixed, confirmed by stage H's
      config.json at both d128 -> 879,418 and d192 -> 1,908,922)

THE PRIMARY IS A NUMBER, NOT A DIRECTION, because both ends of the bracket are
measured. Final train_nll at L=12, no_mlp, 375k steps:

    d128   1.6736   mean of 1.6768 / 1.6703, 0 of 2 escapes   (stage H, final)
    d160   ???      <- this job
    d192   ~1.560   seed 0 extrapolated from 1.5670 at epoch 40 at its own
                    0.0007/epoch tail rate -- AN EXTRAPOLATION OF A RUNNING
                    CELL at the time of registration, not a measurement
    d256   1.5330   mean of 1.5372 / 1.5243 / 1.5374, 3 of 3   (stage E, final)

REGISTERED: I1 escapes the plateau on >= 2 of 3 seeds with final train_nll in
[1.600, 1.620] -- linear in 1/d gives 1.605, linear in d gives 1.617, and both
clear 1.64 by 0.023 to 0.035. The interval is quoted so that being right about
the direction and wrong about the number stays visible.

THE FALSIFIER IS THE INTERESTING OUTCOME. 0 of 3 escapes, or a final train_nll
above 1.64, says the 128->192 transition is a CLIFF rather than a graded curve
in width, and the brief's instruction is to treat the breaking point as the
result. Note that "it just needs longer" is NOT free here: stage G spent 750k
steps on d128 no_mlp at L=6 and got 0 of 3 escapes, so the budget argument has
already been tried once at the cheap end and lost.

REPORTING ORDER, as registered: I4 (the free sufficiency test) ahead of
everything if it fails, then I1's trainability number, then the ladder, then
the standing-minimum rule.

Scored against the frozen bar s* = 0.1484 / 0.2918, not recomputed. Every
comparison is median-vs-median; a difference smaller than s* is "not separable
at this budget", never a result (findings 41/42/47).
"""
import argparse
from collections import defaultdict

from scripts.minarch_analysis import (SCAN_DIRS, SCAN_E, SCAN_H, SCAN_I,
                                      cell_key, cell_metrics, load_all)
from scripts.stagec_score import (S_MEAN, S_RHO50, escapes, rho_txt, rung,
                                  summarize, verdict)

HUB_RHO50 = 0.592        # full L=6 d256, the reference architecture
HUB_TRAINABLE = 4899898
FREE = 0.740             # hub + s*, and the standing-minimum rule's ceiling
BAND_LO, BAND_HI = 0.72, 0.80
LOO_BAR = 0.0423         # acc_vs_solve's leave-one-out residual bar
PLATEAU = 1.64           # finding 44

I1 = (12, 4, "no_mlp", 160, 1024, 0)
H1 = (12, 4, "no_mlp", 128, 1024, 0)
H2 = (12, 4, "no_mlp", 192, 1024, 0)
E2 = (12, 4, "no_mlp")                      # full-width partner, same depth
L6_D128 = (6, 4, "no_mlp", 128, 1024, 0)

PREDICTED_TRAINABLE = {"I1": 1345018}
PRED_NLL_LO, PRED_NLL_HI = 1.600, 1.620
PRED_ESCAPES = ">= 2 of 3"

# Frozen off disk before stage I was submitted:
#   (n, n_cross, rho_50 text, mean_solve, spread, trainable, escapes)
# E2 and L6_D128 were final and stage H's scorer verified them against disk.
PARTNER = {
    "E2": (3, 3, "0.605", 0.5214, 0.0480, 3331642, 3),
    "L6_D128": (3, 1, "CENSORED", 0.0291, 0.2075, 480058, 0),
}
# The two stage H cells the prediction was BUILT on. These were complete at
# registration time; the control below re-reads them so that if stage H's third
# d128 seed changes the rung's story, the fact that this job's interpolation
# used only two cells is visible rather than buried.
BRACKET_AT_REGISTRATION = {
    "d128": {"cells": 2, "nll": (1.6768, 1.6703), "escapes": 0, "final": True},
    "d192": {"cells": 1, "nll": (1.5670,), "escapes": 1, "final": False,
             "note": "epoch 40 of 50, still descending ~0.0007/epoch"},
    "d256": {"cells": 3, "nll": (1.5372, 1.5243, 1.5374), "escapes": 3, "final": True},
}


def report(tag, cells, key, asks, predicted_trainable=None):
    ms = rung(cells, key)
    print(f"\n=== {tag} -- {asks} ===")
    if not ms:
        print("  nothing on disk yet.")
        return None
    s = summarize(ms)
    n_esc, nll = escapes(ms)
    print(f"  {s['n']}/3 seeds on disk, {s['n_cross']}/{s['n']} crossed 0.5")
    print(f"  rho_50 median {rho_txt(s):>9}   (hub {HUB_RHO50:.3f}, E2 "
          f"{PARTNER['E2'][2]}, free at or below {FREE})")
    print(f"  mean_solve    {s['mean_solve']:>9.4f}   [spread {s['spread_mean']:.4f}]")
    if predicted_trainable is not None:
        print(f"  trainable {s['trainable']:,}   (header arithmetic "
              f"{predicted_trainable:,}"
              + ("" if s["trainable"] == predicted_trainable
                 else "  <-- DISAGREES, quote the disk") + ")")
    else:
        print(f"  trainable {s['trainable']:,}")
    print(f"  plateau escapes {n_esc}/{s['n']}   final train NLL "
          + "/".join(f"{v:.4f}" for v in nll))
    return dict(s=s, n_esc=n_esc, nll=nll, ms=ms)


def score_trainability(i1):
    """The primary. A registered interval meets a measurement."""
    s, n_esc, nll = i1["s"], i1["n_esc"], i1["nll"]
    med = sorted(nll)[len(nll) // 2]
    print(f"\nI1 -- THE REGISTERED NUMBER: final train_nll in "
          f"[{PRED_NLL_LO:.3f}, {PRED_NLL_HI:.3f}] on {PRED_ESCAPES} escapes")
    print(f"  observed median final train_nll {med:.4f} over {s['n']} seeds "
          f"({'/'.join(f'{v:.4f}' for v in nll)})")
    inside = PRED_NLL_LO <= med <= PRED_NLL_HI
    if inside:
        where = "as predicted"
    elif med < PRED_NLL_LO:
        where = "below, better than predicted"
    else:
        where = "above, worse than predicted"
    print(f"  interval: {'INSIDE' if inside else 'OUTSIDE'}  ({where})")
    if n_esc >= 2:
        verdict(f"""
THE FLOOR IS GRADED, NOT A CLIFF. {n_esc} of {s['n']} seeds at d160 got below the
{PLATEAU} plateau, median final train_nll {med:.4f}, so the width switch between
128 and 192 has a middle and the bracket closes to [128, 160]. The registered
interval [{PRED_NLL_LO:.3f}, {PRED_NLL_HI:.3f}] was {'HIT' if inside else 'MISSED'} at
{med:.4f}: {'the two interpolations of the bracketing segment were right about both the direction and the size' if inside else 'the direction was right and the size was not, so the curve in width is not linear in either d or 1/d'}.
Whether this rung also SOLVES is a separate question and is scored below; a
trained rung that censors is an architecture statement, which is exactly the
distinction d128 could not supply.""")
        return "graded"
    verdict(f"""
THE FLOOR IS A CLIFF, and by the brief's own instruction that breaking point is
the result. Only {n_esc} of {s['n']} seeds at d160 got below the {PLATEAU} plateau
(median final train_nll {med:.4f} against a registered [{PRED_NLL_LO:.3f}, {PRED_NLL_HI:.3f}]),
so halving the gap in width bought none of the gap in loss: the transition
between d128 and d192 is not graded and the bracket closes to [160, 192].
BOTH interpolations are falsified, which matters for anything built on them.
And the budget escape hatch is not available: stage G already spent 750k steps
on d128 no_mlp at L=6 and got 0 of 3. The honest statement is that attention-
only at L=12 needs a residual stream wider than 160 to find direct deduction in
375k steps, and rho_50 for this rung is NOT evidence about its architecture.""")
    return "cliff"


def score_ladder(i1, h1, h2):
    """Monotonicity in width at L=12, with the two stage H neighbours live."""
    s = i1["s"]
    e2_rho = float(PARTNER["E2"][2])
    print("\nI2 -- THE LADDER (monotone in width at L=12?)")
    if not s["rho50_readable"]:
        print(f"  I1 censored, so it cannot be ordered between its neighbours. "
              f"Escapes {i1['n_esc']}/{s['n']} decide whether that is an "
              f"architecture statement or a trainability one.")
        return "unavailable"
    neigh = []
    for tag, h in (("H1 d128", h1), ("H2 d192", h2)):
        if h and h["s"]["rho50_readable"]:
            neigh.append((tag, h["s"]["rho50"]))
        elif h:
            neigh.append((tag, None))
    neigh.append(("E2 d256", e2_rho))
    txt = ", ".join(f"{t} {'CENSORED' if v is None else f'{v:.3f}'}" for t, v in neigh)
    print(f"  neighbours: {txt}; I1 d160 {s['rho50']:.3f}")
    known = [v for _t, v in neigh if v is not None]
    if not known:
        print("  no readable neighbour; monotonicity is unavailable, not passed.")
        return "unavailable"
    lo, hi = min(known), max(known)
    inside = lo - S_RHO50 <= s["rho50"] <= hi + S_RHO50
    if inside:
        print(f"  MONOTONE within s* = {S_RHO50}: I1 sits in [{lo:.3f}, {hi:.3f}] "
              f"allowing the bar. The width ladder at L=12 behaves like every "
              f"other ladder in this grid.")
        return "monotone"
    verdict(f"""
MONOTONICITY IN WIDTH IS FALSIFIED AT L=12, and it is reported ahead of the
standing-minimum rule because it changes how I1 reads. I1 at {s['rho50']:.3f} is outside
[{lo:.3f}, {hi:.3f}] by more than s* = {S_RHO50}. No ladder in this grid has shown width
and depth interacting this way -- the L=6 attention-only ladder and the complete
model's are both monotone -- so I1 is not the bottom of a well-behaved ladder and
no minimum should be quoted from it without a fourth width.""")
    return "non-monotone"


def score_minimum(i1, call, mono):
    s = i1["s"]
    print("\nI3 -- THE STANDING-MINIMUM RULE, unchanged from stage H's H3:")
    print(f"  a rung replaces the standing minimum ONLY IF 3 of 3 seeds cross AND")
    print(f"  the median rho_50 is at or below {FREE}. Two of three is not enough,")
    print(f"  and finding 51 is why: stage G showed a 2-of-3 at 50 epochs was ONE")
    print(f"  UNFINISHED SEED, not a property of the rung.")
    if (call == "graded" and s["rho50_readable"] and s["rho50"] <= FREE
            and s["n_cross"] == 3 and s["n"] == 3):
        print(f"  I3 FIRES. The standing minimum moves from E2 "
              f"({PARTNER['E2'][5]:,} trainable at {PARTNER['E2'][2]}) to "
              f"attention-only L=12 d160, {s['trainable']:,} at {rho_txt(s)}:")
        print(f"    size DOWN {PARTNER['E2'][5] / s['trainable']:.2f}x and "
              f"{HUB_TRAINABLE / s['trainable']:.1f}x below the hub; rho_50 moves "
              f"{s['rho50'] - float(PARTNER['E2'][2]):+.3f} "
              f"({abs(s['rho50'] - float(PARTNER['E2'][2])) / S_RHO50:.2f} s*).")
    elif call == "cliff":
        print(f"  I3 DOES NOT FIRE: the rung did not train, so its rho_50 is not a")
        print(f"  measurement of its architecture. The standing minimum stays at E2,")
        print(f"  {PARTNER['E2'][5]:,} trainable at {PARTNER['E2'][2]}.")
    elif s["rho50_readable"] and s["rho50"] <= FREE:
        print(f"  I3 DOES NOT FIRE: the median clears {FREE} but only {s['n_cross']} of "
              f"{s['n']} seeds cross. Both conditions or neither.")
    else:
        print(f"  I3 DOES NOT FIRE: median {rho_txt(s)} did not clear {FREE}. The "
              f"standing minimum stays at E2, {PARTNER['E2'][5]:,} trainable.")
    if mono == "non-monotone":
        print("  AND the ladder it would sit on is not monotone, so no minimum should")
        print("  be quoted from this job without a fourth width.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+",
                    default=SCAN_DIRS + [SCAN_E, SCAN_H, SCAN_I],
                    help="SCAN_E, SCAN_H and SCAN_I are out of SCAN_DIRS on "
                         "purpose; they are added here only.")
    args = ap.parse_args()

    by = defaultdict(list)
    for r in load_all(args.scan_dir):
        by[cell_key(r)].append(r)
    cells = {k: dict(cell_metrics(v, args.boot), path=v[0]["path"], seed=k[-1])
             for k, v in by.items()}

    print("STAGE I: where exactly is the width trainability floor at L=12?")
    print(f"  {len(cells)} cells read; ONE rung, d160, bisecting [128, 192]")
    print(f"  frozen bar s* = {S_RHO50} rho_50 / {S_MEAN} mean_solve, NOT recomputed")
    print(f"  registered: escapes {PRED_ESCAPES}, final train_nll in "
          f"[{PRED_NLL_LO:.3f}, {PRED_NLL_HI:.3f}]")

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

    print("\nBRACKET CONTROL -- what the prediction was actually built on, against")
    print("what is on disk now. The d192 end was a RUNNING cell when stage I was")
    print("registered, so this is the line that says whether the interpolation was")
    print("anchored on a number that later moved.")
    for tag, key in (("d128", H1), ("d192", H2)):
        b = BRACKET_AT_REGISTRATION[tag]
        ms = rung(cells, key)
        at = (f"{b['cells']} cell(s), nll "
              + "/".join(f"{v:.4f}" for v in b["nll"])
              + f", {b['escapes']} escape(s)"
              + ("" if b["final"] else f"  [{b['note']}]"))
        if not ms:
            print(f"  {tag}: at registration {at}  |  disk: nothing")
            continue
        s, (ne, nll) = summarize(ms), escapes(ms)
        print(f"  {tag}: at registration {at}")
        print(f"        now  {s['n']} cell(s), nll "
              + "/".join(f"{v:.4f}" for v in nll) + f", {ne} escape(s)"
              + ("" if s["n"] == b["cells"] else "   <-- more cells than the prediction used"))

    i1 = report("I1", cells, I1, "THE BISECTOR, attention-only L=12 d160",
                PREDICTED_TRAINABLE["I1"])
    h2 = report("H2", cells, H2, "neighbour above, attention-only L=12 d192")
    h1 = report("H1", cells, H1, "neighbour below, attention-only L=12 d128")

    print("\n" + "=" * 72)
    if not i1:
        print("VERDICT: NOT YET -- no stage I cell on disk.")
        print("=" * 72)
        return

    call = score_trainability(i1)
    mono = score_ladder(i1, h1, h2)
    score_minimum(i1, call, mono)
    print("=" * 72)

    print("\nI4 -- OUT-OF-SAMPLE SUFFICIENCY, free, and reported ahead of I1 if it")
    print(f"  fails. The depth-ladder fit holds on 16 of 16 well-determined held-out")
    print(f"  rungs over 37 cells, worst discriminating miss 0.0099 against the LOO")
    print(f"  bar {LOO_BAR}. Prediction: I1 lands on the curve within {LOO_BAR}.")
    print("  Run:")
    print("    python3 -m scripts.acc_vs_solve --scan-dir \\\n      " +
          " \\\n      ".join(args.scan_dir))
    print("  I1 falling OFF the curve would be the grid's first evidence of an")
    print("  architectural change doing something SOLVE-SPECIFIC rather than moving")
    print("  the model along one quality axis.")


if __name__ == "__main__":
    main()
