"""Score stage G (51194118): is the cheap end excluded by CAPACITY or by the BUDGET?

    python3 -m scripts.stageg_score [--boot 2000]

THE QUESTION. For five stages this programme has written "0 of 3 seeds escaped
finding 44's 1.64 train-NLL plateau" as "did not train in 375k steps, never
evidence in either direction". That is the honest reading and it was treated as
a dead end. Finding 49 (scripts/pareto_min.py, scripts/escape_timing.py) showed
it is checkable instead: the floor on this grid is TRAINABILITY, not capacity --
only 12 of 24 rungs with at least two seeds ever escaped, the standing minimum
is merely the cheapest of those 12, and all 7 cheaper rungs are excluded at 0/3
escapes rather than for scoring worse.

Stage G moves ONE knob, epochs 50 -> 100 (750k gradient steps against 375k), at
a FIXED 240k pool with constant LR and warmup 0, so no schedule is reshaped and
no data is added. G1 is the cheapest rung in the grid; G2 is the standing
minimum, run as the PAIRED CONTROL inside the same job, because the budget
itself is what moves and nothing at 50 epochs is comparable across it.

THIS FILE IS THE REGISTRATION, TRANSCRIBED, and it was committed while tasks
0-1 were the only ones running and NO stage G cell had finished. Every number
below is either copied from the sbatch header (4a35a2e) or measured off the
50-epoch twins that already existed. Nothing here is chosen after seeing a
stage G result.

  G1  no_mlp d128   480,058 trainable   escapes >=2/3 under TRAINABILITY,
                                        0-1/3 under CAPACITY
  G2  no_mlp d256 1,746,490 trainable   3/3 escapes; if median rho_50 < 0.615
                                        the 0.76 band is a budget artifact

THE ASYMMETRY, stated in the header before the cells existed and repeated here
because it is the whole point: G1 ESCAPING proves the budget was binding. G1
NOT escaping does NOT prove capacity is binding -- it is equally consistent
with "needs 150 epochs". There is no symmetric reading and the scorer refuses
to print one.

EXACTLY TWO COMPARISONS ARE LEGITIMATE: G1 against G2 at the same budget, and
each rung against its own 50-epoch twin. G1's rho_50 against the frozen 0.76
band is CROSS-BUDGET and is not scored.

-----------------------------------------------------------------------------
THE PRE-REGISTERED QUANTITATIVE PREDICTION, AND IT CONTRADICTS THE HEADER

escape_timing.py's Q3, added 11-05 while stage G was running and before any G
cell landed, measured two things the header did not have:

  (a) LATE ESCAPES HAVE NO KNEE. All 14 escapes after epoch 15 crossed from a
      train NLL of 1.640-1.652 -- already sitting ON the line -- and 13 of 14
      were within FIVE epochs of the plateau at their own prior rate. A late
      escape is a drift across a line, not a cell that suddenly found the
      problem. So "escapes are still arriving at epoch 44" says nothing about
      how far a 0/3 cell is.

  (b) THE 0/3 CELLS ARE MOSTLY FAR AWAY. Over the 39 cells that never escaped,
      the median gap is 87 epochs at their own terminal rate and only 10 of 39
      would reach the plateau within 50 MORE epochs. NONE of those 10 is a
      stage G cell.

Extrapolating each twin's last-20% slope linearly to epoch 100 therefore
predicts, BEFORE THE FACT:

      G1 escapes 0 of 3  (gaps of 80, 123 and 230 epochs; none within 50)
      G2 escapes 2 of 3  (seed 1's gap is 72 epochs; seeds 0 and 2 are already
                          below the plateau at epoch 50)

G2 at 2/3 CONTRADICTS THE SBATCH HEADER'S OWN "3/3 escapes predicted", which
was written without this measurement. Both predictions are on the record and
the scorer prints both, because the interesting outcome is which one is right:
the header's was a guess, this one is an extrapolation with a stated model, and
a disagreement between them is resolved by the data rather than by me.

THE EXTRAPOLATION IS ITSELF OPTIMISTIC. These tails are flattening, not
straight, so each predicted epoch-100 train NLL below is a LOWER bound and the
observed value should land AT OR ABOVE it. A cell landing well BELOW its
prediction is the one genuinely surprising outcome available in this job: it
would mean the descent accelerates at longer budgets, every ETA in Q3's table
is an overestimate, and five stages' worth of "did not train" caveats hide more
trainable rungs than anyone thought.

WHAT STAGE G IS NOW FOR, given that its headline prediction is predicted to
fail. Three things survive, and they are why the job is worth its 25 GPU-h:
  1. G2 is a real measurement either way -- the standing minimum's solve rate
     and token accuracy at double the budget, which tests whether the 0.76 band
     is a budget artifact. Two of its seeds are already past the plateau and
     still descending.
  2. G1 at epoch 100 is the first direct test of the linear extrapolation, and
     hence of Q3's method, which the programme will otherwise keep using.
  3. The epoch-50 CHECKPOINT CHECK below is free and validates the pairing.

A DESIGN NOTE, recorded against myself: G1 and G2 were chosen for the PARETO
question (the cheapest rung and the standing minimum) before Q3 existed. The
cells closest to escaping -- rf_mlp seed 1 at 1 epoch, rf_mlp seed 2 at 3,
full d128 seed 1 at 14 -- are not in stage G, and would have been a cheaper and
far more likely test of the budget hypothesis. That is a worse grid than I
could have chosen with one more day of analysis, and it is the reason the
descriptive work came first in the next stage.
-----------------------------------------------------------------------------

THE EPOCH-50 CHECKPOINT CHECK, and it is a hard gate. With a constant LR, no
warmup, a fixed pool and the same seed, the first 50 epochs of a 100-epoch run
are the SAME optimization as the 50-epoch twin. Each G cell's epoch-50 train
NLL must therefore reproduce its twin's FINAL train NLL to within GPU
nondeterminism. Tolerance registered at 0.01 nats. If a cell misses it,
something other than the budget changed, the pairing is void, and the cell is
reported inconclusive rather than explained.

Scored against the frozen bar s* = 0.1484 / 0.2918, not recomputed.
"""
import argparse
import json
import os
from collections import defaultdict

from scripts.minarch_analysis import (SCAN_DIRS, SCAN_G, cell_key, cell_metrics,
                                      load_all)
from scripts.stagec_score import (S_MEAN, S_RHO50, escapes, rho_txt, rung, summarize,
                                  verdict)

EPOCHS_G = 100
TOL_CKPT = 0.01          # epoch-50 reproduction tolerance, nats
FREE = 0.740             # hub 0.592 + s*, the free/not-free threshold
G2_ARTIFACT = 0.615      # 0.763 - s*: below this, the 0.76 band is a budget artifact
LOO_BAR = 0.0423         # acc_vs_solve's leave-one-out residual

# The 50-epoch twins. Keys are the epoch-aware rung keys; a 100-epoch cell gets
# a distinct key by construction (251f35e), which is what stops a stage G cell
# from pooling into a published median.
G1 = (6, 4, "no_mlp", 128, 1024, 0, EPOCHS_G)
G2 = (6, 4, "no_mlp", 256, 1024, 0, EPOCHS_G)
G1_TWIN = (6, 4, "no_mlp", 128, 1024, 0)
G2_TWIN = (6, 4, "no_mlp")

# Frozen off disk 11-05, before any stage G cell existed. Per seed:
#   (final train NLL at epoch 50, final val token accuracy, last-20% slope/ep)
TWIN = {
    "G1": {0: (1.725, 0.4558, -0.00069),
           1: (1.738, 0.4540, -0.00042),
           2: (1.706, 0.4744, -0.00083)},
    "G2": {0: (1.631, 0.4812, -0.00039),
           1: (1.688, 0.4646, -0.00067),
           2: (1.597, 0.4834, -0.00056)},
}
# Rung-level, also frozen: (n, n_cross, rho_50 text, mean_solve, spread, trainable, escapes)
TWIN_RUNG = {
    "G1": (3, 1, "CENSORED", 0.0291, 0.2075, 480058, 0),
    "G2": (3, 2, "0.763", 0.3039, 0.2671, 1746490, 2),
}
PRED_ESCAPES = {"G1": 0, "G2": 2}        # from the extrapolation, see docstring
HEADER_ESCAPES = {"G1": ">=2 (trainability) / <=1 (capacity)", "G2": "3"}


def predicted_nll(tag, seed):
    """Linear extrapolation of the twin's terminal slope to epoch 100.

    A LOWER bound: these tails flatten, so the observed value should land at or
    above this. Landing below it is the surprising outcome.
    """
    nll, _acc, slope = TWIN[tag][seed]
    return nll + slope * (EPOCHS_G - 50)


def curve(path):
    with open(os.path.join(path, "history.json")) as fh:
        h = json.load(fh)
    return ([float(e["train_nll"]) for e in h], [float(e["accuracy"]) for e in h])


def integrity(tag, ms):
    """The hard gates: epochs, trainable count, and the epoch-50 reproduction."""
    ok = True
    want_tr = TWIN_RUNG[tag][5]
    if ms[0]["n_trainable"] != want_tr:
        print(f"  !! TRAINABLE COUNT {ms[0]['n_trainable']:,} != the twin's {want_tr:,}. "
              f"The architecture moved, not just the budget. PAIRING VOID.")
        ok = False
    print(f"  epoch-50 checkpoint check (tolerance {TOL_CKPT} nats, registered):")
    for m in ms:
        seed = cell_key_seed(m)
        tr, _acc = curve(m["path"])
        if len(tr) < 50:
            print(f"    seed {seed}: only {len(tr)} epochs on disk, cannot check yet")
            ok = False
            continue
        got, want = tr[49], TWIN[tag][seed][0]
        d = got - want
        mark = "ok" if abs(d) <= TOL_CKPT else "MISMATCH"
        print(f"    seed {seed}: epoch-50 train NLL {got:.3f} vs twin's final "
              f"{want:.3f}  ({d:+.3f})  {mark}")
        if abs(d) > TOL_CKPT:
            print("      the first 50 epochs are supposed to be the SAME optimization; "
                  "something other than the budget changed.")
            ok = False
    return ok


def cell_key_seed(m):
    """Seed of a cell, read back off its own config rather than its directory."""
    return int(m.get("seed", m.get("config_seed", -1)))


def report(tag, cells, asks):
    ms = rung(cells, G1 if tag == "G1" else G2)
    if not ms:
        print(f"\n=== {tag} -- {asks} ===")
        print("  nothing on disk yet.")
        return "pending", None
    s = summarize(ms)
    n_esc, nll = escapes(ms)
    n, n_cross, rho_t, msolve, spr, tr_n, esc0 = TWIN_RUNG[tag]

    print(f"\n=== {tag} -- {asks} ===")
    print(f"  {s['n']}/3 seeds on disk at {EPOCHS_G} epochs, {s['n_cross']}/{s['n']} "
          f"crossed 0.5")
    print(f"  trainable {s['trainable']:,}   (50-epoch twin {tr_n:,})")
    print(f"  THE ONLY LEGITIMATE WITHIN-RUNG COMPARISON, this cell vs its own twin:")
    print(f"    rho_50      {rho_txt(s):>9}   twin {rho_t:>9}")
    print(f"    mean_solve  {s['mean_solve']:>9.4f}   twin {msolve:>9.4f}   "
          f"[spread {s['spread_mean']:.4f} / {spr:.4f}]")
    print(f"    escapes     {n_esc:>4}/{s['n']}        twin {esc0:>4}/{n}")
    print(f"    train NLL   " + "/".join(f"{v:.3f}" for v in nll))

    print(f"  registered, sbatch header (4a35a2e): escapes {HEADER_ESCAPES[tag]}")
    print(f"  registered, extrapolation (this file): escapes {PRED_ESCAPES[tag]} of 3")
    print("  per-seed extrapolation, predicted as a LOWER bound on epoch-100 train NLL:")
    below = 0
    for m in ms:
        seed = cell_key_seed(m)
        if seed not in TWIN[tag]:
            print(f"    seed {seed}: no twin on record, cannot extrapolate")
            continue
        tr, acc = curve(m["path"])
        pred = predicted_nll(tag, seed)
        got = tr[-1]
        d = got - pred
        if d < -TOL_CKPT:
            below += 1
            note = "BELOW the bound -- descent ACCELERATED"
        elif d > 0:
            note = "at or above, as predicted (tails flatten)"
        else:
            note = "on the bound"
        print(f"    seed {seed}: epoch-{len(tr)} train NLL {got:.3f} vs predicted "
              f"{pred:.3f} ({d:+.3f})  {note}")
        print(f"             token accuracy {acc[-1]:.4f}   twin {TWIN[tag][seed][1]:.4f}  "
              f"({acc[-1] - TWIN[tag][seed][1]:+.4f})")
    return ("done", {"s": s, "n_esc": n_esc, "below": below, "ms": ms})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+", default=SCAN_DIRS + [SCAN_G],
                    help="SCAN_G is out of SCAN_DIRS on purpose; it is added here only.")
    args = ap.parse_args()

    by = defaultdict(list)
    for r in load_all(args.scan_dir):
        by[cell_key(r)].append(r)
    cells = {k: dict(cell_metrics(v, args.boot), path=v[0]["path"], seed=k[-1])
             for k, v in by.items()}

    print("STAGE G: is the cheap end excluded by CAPACITY, or by the BUDGET?")
    print(f"  {len(cells)} cells read; ONE knob, epochs 50 -> {EPOCHS_G}, fixed 240k pool")
    print(f"  frozen bar s* = {S_RHO50} rho_50 / {S_MEAN} mean_solve, NOT recomputed")
    print("  G1's rho_50 against the 0.76 band is CROSS-BUDGET and is not scored.")

    # The twins must still read the way they were frozen, or the literals above
    # are stale and every comparison in this file is against the wrong number.
    print("\nFROZEN-LITERAL CONTROL (the 50-epoch twins, re-read from disk):")
    for tag, key in (("G1", G1_TWIN), ("G2", G2_TWIN)):
        ms = rung(cells, key)
        if not ms:
            print(f"  {tag} twin: NOT ON DISK -- cannot verify the literals")
            continue
        s, (ne, _nll) = summarize(ms), escapes(rung(cells, key))
        want = TWIN_RUNG[tag]
        got = (s["n"], s["n_cross"], rho_txt(s), round(s["mean_solve"], 4),
               round(s["spread_mean"], 4), s["trainable"], ne)
        print(f"  {tag} twin {'matches' if got == want else 'DISAGREES WITH'} the frozen "
              f"literals")
        if got != want:
            print(f"    frozen {want}")
            print(f"    disk   {got}")
            print("    the registered grid changed; fix that before reading anything below.")

    out = {}
    for tag, asks in (("G2", "THE PAIRED CONTROL, standing minimum no_mlp d256 at 100 epochs"),
                      ("G1", "THE CHEAP END, no_mlp d128 at 100 epochs")):
        status, res = report(tag, cells, asks)
        out[tag] = res
        if res:
            integrity(tag, res["ms"])

    print("\n" + "=" * 72)
    g1, g2 = out.get("G1"), out.get("G2")
    if not g1 or not g2:
        have = [k for k, v in out.items() if v]
        print(f"VERDICT: NOT YET -- {', '.join(have) if have else 'no'} rung(s) on disk. "
              f"G2 is")
        print("  reported AHEAD of G1 by registration, and G3 needs both.")
        print("=" * 72)
        return

    # G2 first, by registration.
    s2 = g2["s"]
    print("G2 (reported first, as registered):")
    if g2["n_esc"] < 3:
        print(f"  escapes {g2['n_esc']}/3, so the SBATCH HEADER's 3/3 is WRONG and the")
        print(f"  extrapolation's {PRED_ESCAPES['G2']}/3 is the better model. Doubling the "
              f"budget did not")
        print("  rescue the seed that was 72 epochs away, which is what Q3 said.")
    else:
        print("  escapes 3/3, as the header predicted and against the extrapolation's 2/3:")
        print("  the straggler crossed a gap the terminal slope said needed 72 epochs, so")
        print("  the tails ACCELERATE and Q3's ETAs are overestimates.")
    if s2["rho50_readable"] and s2["rho50"] < G2_ARTIFACT:
        print(f"  rho_50 {s2['rho50']:.3f} is BELOW {G2_ARTIFACT}: THE 0.76 BAND IS PARTLY A")
        print("  BUDGET ARTIFACT. Every rung in the pile-up was scored at 375k steps, and")
        print("  the band moves when the steps do -- so the band is a property of the")
        print("  BUDGET as much as of the architectures, and finding 44's plateau language")
        print("  has to be reread across stages A-E. This does NOT by itself rescue the")
        print("  barrier-vs-capacity question, which is about rungs, not about the floor.")
    else:
        print(f"  rho_50 {rho_txt(s2)} is not below {G2_ARTIFACT}: the standing minimum does")
        print("  not move resolvably at double the budget, so the 0.76 band is NOT a budget")
        print("  artifact at this rung, and the band survives the one test that could have")
        print("  dissolved it.")

    # G1, with the asymmetry enforced.
    s1 = g1["s"]
    print("\nG1:")
    if g1["n_esc"] >= 2:
        print(f"  ESCAPES {g1['n_esc']}/3. THE BUDGET WAS BINDING at the cheap end, and it")
        print("  was binding against the extrapolation's prediction of 0/3 -- the cheapest")
        print("  rung in the grid is TRAINABLE and was only ever excluded by 375k steps.")
        print("  Five stages of '0/3 escapes' now need rereading: the exclusion was a")
        print("  measurement limit, not a property of the architecture.")
        if s1["rho50_readable"] and s2["rho50_readable"] and \
                abs(s1["rho50"] - s2["rho50"]) <= S_RHO50:
            print(f"  G3 FIRES: median rho_50 {s1['rho50']:.3f} is within s* = {S_RHO50} of")
            print(f"  G2's {s2['rho50']:.3f}, and G1 escaped >=2/3. THE STANDING MINIMUM "
                  f"DROPS TO")
            print(f"  {s1['trainable']:,} TRAINABLE, {TWIN_RUNG['G2'][5] / TWIN_RUNG['G1'][5]:.2f}x "
                  f"below the old one -- but AT 100 EPOCHS, and a")
            print("  minimum quoted at a different budget is a different claim. Say the")
            print("  budget in the same breath as the number, every time.")
        else:
            print("  G3 DOES NOT FIRE: G1 escaped but its rho_50 is not within s* of G2's, "
                  "so")
            print("  the standing minimum does NOT move. Both conditions or neither.")
    else:
        print(f"  escapes {g1['n_esc']}/3, which is what the extrapolation predicted "
              f"({PRED_ESCAPES['G1']}/3)")
        print("  from gaps of 80, 123 and 230 epochs against the 50 that were granted.")
        print("  THIS IS NOT EVIDENCE THAT CAPACITY IS BINDING. It is the registered")
        print("  asymmetry: escaping would have proved the budget was binding; not")
        print("  escaping is equally consistent with 'needs 150 epochs'. The honest")
        print("  statement is UNCHANGED from stage C's P2 -- d128 did not train, at 750k")
        print("  steps now rather than 375k -- and the next cell is 200 epochs, not a")
        print("  conclusion. G3 does not fire; the standing minimum is unchanged.")

    tot_below = g1["below"] + g2["below"]
    print("\nTHE METHOD TEST (this is the part that generalizes):")
    if tot_below:
        print(f"  {tot_below} of 6 seeds landed BELOW their extrapolated lower bound, so the")
        print("  descent ACCELERATES at longer budgets and every ETA in escape_timing's Q3")
        print("  is an overestimate. Rerun Q3's reading of stages A-E before quoting it.")
    else:
        print("  0 of 6 seeds landed below the extrapolated lower bound, so the linear")
        print("  extrapolation of the terminal slope is a sound (if optimistic) model of")
        print("  what another 50 epochs buys. Q3's table can be used to choose the next")
        print("  budget cell instead of guessing -- the 10 cells it puts within 50 epochs")
        print("  are the cheap tests, and none of them is in this job.")
    print("=" * 72)

    print(f"\nG4 -- OUT-OF-SAMPLE SUFFICIENCY AT A NEW BUDGET, free and reported ahead of")
    print(f"  the above if it fails. The accuracy-to-solve curve has held on 13 of 13")
    print(f"  held-out rungs within the LOO bar {LOO_BAR}, but never once at a budget it")
    print(f"  was not fitted at. Run:")
    print("    python3 -m scripts.acc_vs_solve --scan-dir \\\n      " +
          " \\\n      ".join(args.scan_dir))
    print("  A stage G rung falling OFF the curve would mean the curve is a statement")
    print("  about 375k steps rather than about model quality, and would retire it as a")
    print("  cross-budget tool.")


if __name__ == "__main__":
    main()
