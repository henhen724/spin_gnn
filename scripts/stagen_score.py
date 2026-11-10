"""Score stage N: does KB (`no_mlp L=16 d128`, N=48) replicate on fresh seeds?

REGISTERED BEFORE THE FIRST CELL LANDED. This docstring IS the registration --
committed by scripts/slurm/stagen_register_2026-11-10.sbatch, which the array
scripts/slurm/sudoku2_n48_kbrep_N_2026-11-10.sbatch depends on with
--dependency=afterok. No cell could start ahead of it.

WHAT STAGE K MEASURED (finding 54, array 51668331, scored 51699439)

    rung                 rho_50   trainable   cross   escapes   call
    KA no_mlp L=16 d160   0.757    1,759,738    3/3     3/3      past the free line
    KB no_mlp L=16 d128   0.766    1,145,658    3/3     2/3      past the free line

The standing minimum stayed `no_mlp L=16 d192`, 2,504,890 trainable at rho_50
0.580 (J2, finding 53), because the replacement rule H3 = I3 = J3 = K2 needs
3-of-3 crossing AND a median at or below the free line 0.740, and 0.766 misses
by +0.026 = +0.18 s*. A narrow miss on three seeds.

WHAT STAGE N CHANGES: THE SEED TRIPLE, AND NOTHING ELSE. Seeds 3/4/5 of the
identical KB cell -- same arch, shape, ff, nhead, warmup, 240k x 50ep budget,
`--shuffle-adj --fall` ensemble, 14-point grid to 0.33, 400 per bin. Fresh seeds
cannot collide with K's 0/1/2 in the pooled cell key, so the two triples pool
mechanically as well as statistically.

THE BARS ARE N=48 BARS, REUSED AND NOT REFROZEN: s* = 0.1484 (rho_50), 0.2918
(mean_solve), free line 0.740, plateau 1.64, LOO bar 0.0423. Stage M's N=96 bars
(0.117 / 0.1586 / 0.0079) are a different size and never appear here.

N0  REPLICATION OF THE LEVEL. The fresh triple's median rho_50 lands within one
    s* of 0.766, i.e. in [0.618, 0.914]. BELOW means 0.766 was pessimistic and
    the pooled median may go free; ABOVE means the bottom of the ladder is worse
    than finding 54 reports.

N1  REPLICATION OF THE CROSSING. All 3 fresh seeds return a MEASURED rho_50 on a
    grid whose floor is 0.33. A censor is this stage's result: KB's 3-of-3
    crossing at L=16 is what made d128's L=12 censor surprising, and K1's own
    registration withholds that surprise until fresh seeds confirm it.

N2  THE POOLED CALL, POOLING RULE FIXED AT REGISTRATION. Pool K's 0/1/2 with
    N's 3/4/5 and apply the replacement rule to all six: KB takes the standing
    minimum only if 6 OF 6 CROSS AND THE SIX-SEED MEDIAN rho_50 <= 0.740.
    Choosing the rule after seeing the numbers is how a narrow miss becomes a
    hit, so it is chosen here.
    PREDICTION: KB stays past the free line; the standing minimum is UNCHANGED.
    REGISTERED UPSIDE: if the six-seed median is free, KB at 1,145,658 REPLACES
    the standing minimum -- a 2.19x reduction.

N3  THE PLATEAU. Pooled escapes below train NLL 1.64 are at least 4 of 6. Fewer
    means d128 at L=16 is optimization-limited rather than capacity-limited,
    and the ~2M floor is a floor on what this recipe TRAINS, not on what the
    task needs.

N4  ACCURACY FIRST. Validation token accuracy is read before rho_50 because it
    never censors. All six KB cells sit on scripts/acc_vs_solve.py's N=48 fitted
    curve within the 0.0423 LOO bar. A cell falling OFF it is the interesting
    one and is the headline over N2. That curve is an N=48 object and applies
    only because this stage is N=48.

GUARDS, each one a mistake this programme has already made once
  1. N=96 bars are never used here, and no N=48 number is differenced against an
     N=96 one (finding 55 is the standing reason).
  2. rho_50 CENSORS. A censored fresh seed is a result, not a missing number,
     and the bound-aware median is what gets quoted.
  3. The pooling rule is fixed above, before any fresh number exists.
  4. SCAN_N stays OUT of SCAN_DIRS, as E/G/H/I/J/K all do, so a replicate cannot
     enter a pooled yardstick by accident.
  5. Every checkpoint is the final epoch, here as everywhere in this programme.
"""

import argparse
import os
from collections import defaultdict
from statistics import median

from scripts.acc_vs_solve import val_accuracy
from scripts.minarch_analysis import (SCAN_DIRS, cell_key, cell_metrics,
                                      load_all)
from scripts.stagec_score import (PLATEAU, S_MEAN, S_RHO50, escapes, rho_txt,
                                  rung, summarize)
from scripts.stagej_score import LOO_BAR, SUFFICIENCY_DIRS
from scripts.stagek_score import FREE, SCAN_K

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_N = os.path.join(_SCR, "spin_gnn", "sudoku_n48_kbrep_N_2026-11-10")

# Pooled on purpose: K holds seeds 0/1/2 of this cell, N holds 3/4/5.
SCAN_N_DIRS = [SCAN_N, SCAN_K]

KB = (16, 4, "no_mlp", 128, 1024, 0)      # the cell this stage replicates
KA = (16, 4, "no_mlp", 160, 1024, 0)      # its stage-K neighbour, for context
J2 = (16, 4, "no_mlp", 192, 1024, 0)      # the standing minimum

K_SEEDS = (0, 1, 2)
N_SEEDS = (3, 4, 5)

KB_TRAINABLE = 1145658
STANDING_TRAINABLE = 2504890
STANDING_RHO50 = 0.580
KB_K_RHO50 = 0.766                        # what stage K measured on 0/1/2
KB_K_ESCAPES = 2                          # of 3
PRED_LO, PRED_HI = KB_K_RHO50 - S_RHO50, KB_K_RHO50 + S_RHO50   # N0's bracket


def load_pooled(dirs, boot):
    """Every cell in `dirs`, keyed so that one rung's seeds land together.

    load_all takes a LIST of directories and iterates it; handing it a single
    directory string makes it iterate the characters and return nothing, which
    reads exactly like "the run has not landed yet".
    """
    present = []
    for d in dirs:
        if os.path.isdir(d):
            present.append(d)
        else:
            print(f"  (no such run dir yet: {d})")
    by = defaultdict(list)
    for r in load_all(present):
        by[cell_key(r)].append(r)
    return {k: dict(cell_metrics(v, boot), path=v[0]["path"], seed=k[-1])
            for k, v in by.items()}


def subset(ms, seeds):
    """The metrics for just these seeds, in seed order."""
    return [m for s in seeds for m in ms if m["seed"] == s]


def acc_of(ms):
    """Final validation token accuracy per seed. Read BEFORE rho_50 (N4).

    Via acc_vs_solve.val_accuracy, which reads history.json's "accuracy" key.
    NOT via minarch_analysis.history(), whose third element is the final val
    NLL -- a different quantity that would print under an accuracy label.
    """
    return [(m["seed"], val_accuracy(m["path"])) for m in ms]


def score_accuracy_first(ms):
    print("\n=== N4a -- VALIDATION TOKEN ACCURACY, READ FIRST ==================")
    print("  The standing non-negotiable: accuracy never censors, so it is read")
    print("  before rho_50 and rho_50 only confirms it.")
    rows = acc_of(ms)
    if not rows:
        print("  nothing on disk yet")
        return
    for seed, a in sorted(rows):
        where = "K (0/1/2)" if seed in K_SEEDS else "N (3/4/5), FRESH"
        print(f"    seed {seed}  val token acc {a:.4f}   from stage {where}")
    ks = [a for s, a in rows if s in K_SEEDS]
    ns = [a for s, a in rows if s in N_SEEDS]
    if ks and ns:
        dk, dn = median(ks), median(ns)
        print(f"  median accuracy  K triple {dk:.4f}   N triple {dn:.4f}   "
              f"difference {dn - dk:+.4f}")
        print("  A difference here with no difference in rho_50, or the reverse,")
        print("  is the thing N4 is for -- see the curve command at the end.")


def score_n1(fresh):
    print("\n=== N1 -- DID THE CROSSING REPLICATE? ============================")
    print("  registered: all 3 fresh seeds return a measured rho_50 (grid floor")
    print("  0.33). A censor is the result, not a gap.")
    if not fresh:
        print("  PENDING: no fresh cell on disk yet.")
        return None
    s = summarize(fresh)
    print(f"  {s['n']}/3 fresh seeds on disk, {s['n_cross']}/{s['n']} crossed 0.5")
    for m in fresh:
        flag = "measured" if m["flag"] == "ok" else f"CENSORED ({m['flag']})"
        print(f"    seed {m['seed']}  rho_50 {m['rho50']:.3f}  {flag}")
    if s["n"] < 3:
        print(f"  INCOMPLETE: {s['n']} of 3 fresh cells landed. N1 is not called")
        print("  on a partial triple; the missing cells are the first thing to fix.")
    elif s["n_cross"] == 3:
        print("  N1 HELD. KB's 3-of-3 crossing at L=16 replicates on fresh seeds,")
        print("  so K1's surprise -- d128 censoring at L=12 and crossing at L=16 --")
        print("  survives and may now be quoted outside the scorer.")
    else:
        print(f"  N1 FALSIFIED: {3 - s['n_cross']} of 3 fresh seeds censored.")
        print("  KB's crossing does NOT replicate. K1's surprise was a three-seed")
        print("  accident and finding 54's KB row is relabelled as such. This")
        print("  outranks N0 and N2 -- a level computed from a censoring rung is")
        print("  a bound, not a measurement.")
    return s


def score_n0(fresh):
    print("\n=== N0 -- DID THE LEVEL REPLICATE? ===============================")
    print(f"  registered: fresh median rho_50 in [{PRED_LO:.3f}, {PRED_HI:.3f}]")
    print(f"  (stage K's {KB_K_RHO50:.3f} plus or minus one s* = {S_RHO50}).")
    if not fresh:
        print("  PENDING: no fresh cell on disk yet.")
        return None
    s = summarize(fresh)
    txt = rho_txt(s)
    print(f"  fresh median rho_50 {txt}   mean_solve {s['mean_solve']:.4f} "
          f"[spread {s['spread_mean']:.4f}]")
    if not s["rho50_readable"]:
        print("  CENSORED -- N0 is not called on a bound. See N1.")
        return s
    d = s["rho50"] - KB_K_RHO50
    print(f"  against stage K's {KB_K_RHO50:.3f}: {d:+.3f} = {d / S_RHO50:+.2f} s*")
    if PRED_LO <= s["rho50"] <= PRED_HI:
        print("  N0 HELD. The level replicates inside the seed spread, which is")
        print("  what the 3-seeds-per-rung rule predicts and the weakest possible")
        print("  confirmation: it says the triples pool, not that KB is good.")
    elif s["rho50"] < PRED_LO:
        print("  N0 FALSIFIED, DIRECTION LOW. Stage K's 0.766 was pessimistic.")
        print("  Read N2 next: this is the direction in which the pooled median")
        print("  can cross the free line.")
    else:
        print("  N0 FALSIFIED, DIRECTION HIGH. Stage K's 0.766 was optimistic and")
        print("  the bottom of the ladder is worse than finding 54 reports. The")
        print("  honest reading is that a 3-seed level at this depth is not")
        print("  reproducible to one s*, which is a caveat on KA too.")
    return s


def score_n2(pooled, fresh):
    print("\n=== N2 -- THE POOLED STANDING-MINIMUM CALL =======================")
    print("  POOLING RULE, FIXED AT REGISTRATION: 6 of 6 seeds cross AND the")
    print(f"  six-seed median rho_50 <= {FREE:.3f}.")
    print(f"  PREDICTION: KB stays past {FREE:.3f}; standing minimum UNCHANGED at")
    print(f"  {STANDING_TRAINABLE:,} trainable / rho_50 {STANDING_RHO50:.3f}.")
    if not pooled:
        print("  PENDING: nothing on disk yet.")
        return
    s = summarize(pooled)
    print(f"  {s['n']} seeds pooled, {s['n_cross']}/{s['n']} crossed 0.5")
    print(f"  six-seed median rho_50 {rho_txt(s)}   mean_solve "
          f"{s['mean_solve']:.4f} [spread {s['spread_mean']:.4f}]")
    print(f"  trainable {s['trainable']:,}")
    if s["trainable"] != KB_TRAINABLE:
        print(f"  STOP: trainable is not KB's {KB_TRAINABLE:,}. These are not the")
        print("  same cell and they must not be pooled. Nothing below is valid.")
        return
    if s["n"] < 6:
        print(f"  INCOMPLETE: {s['n']} of 6 seeds. The rule is written for six and")
        print("  is not applied to fewer; a partial pool is reported, not called.")
        return
    if not s["rho50_readable"]:
        print("  CENSORED pooled median -- KB does not take the minimum, and the")
        print("  reason is a bound rather than a level.")
        return
    free = s["n_cross"] == 6 and s["rho50"] <= FREE
    gap = (s["rho50"] - FREE) / S_RHO50
    if free:
        print(f"  *** N2's PREDICTION FALSIFIED, AND IN THE REGISTERED UPSIDE ***")
        print(f"  {s['rho50']:.3f} is {gap:+.2f} s* relative to {FREE:.3f}, with")
        print("  6 of 6 crossing. KB REPLACES THE STANDING MINIMUM:")
        print(f"    {STANDING_TRAINABLE:,} -> {KB_TRAINABLE:,} trainable, "
              f"{STANDING_TRAINABLE / KB_TRAINABLE:.2f}x smaller.")
        print("  Before quoting it: this is an N=48 claim and finding 55 says the")
        print("  N=48 minimum does not transfer to N=96. Say N=48 in the sentence.")
    else:
        why = ("not all six crossed" if s["n_cross"] < 6
               else f"the median is {gap:+.2f} s* past {FREE:.3f}")
        print(f"  N2 HELD: KB does not take the minimum ({why}).")
        print(f"  Standing minimum UNCHANGED: {STANDING_TRAINABLE:,} trainable at")
        print(f"  rho_50 {STANDING_RHO50:.3f}. Before/after on the same eval:")
        print(f"    {STANDING_RHO50:.3f} at {STANDING_TRAINABLE:,} -> "
              f"{STANDING_RHO50:.3f} at {STANDING_TRAINABLE:,}, UNCHANGED.")
        print("  That is this stage's result and it is a negative one: six seeds")
        print("  of the smallest crossing rung still do not make it free.")


def score_n3(pooled, fresh):
    print("\n=== N3 -- THE PLATEAU ============================================")
    print(f"  registered: pooled escapes below train NLL {PLATEAU} are >= 4 of 6.")
    print(f"  Stage K's KB had {KB_K_ESCAPES} of 3.")
    if not pooled:
        print("  PENDING: nothing on disk yet.")
        return
    n_esc, nll = escapes(pooled)
    print(f"  pooled escapes {n_esc}/{len(pooled)}   train NLL "
          + "/".join(f"{v:.3f}" for v in nll))
    if fresh:
        fe, fnll = escapes(fresh)
        print(f"  fresh triple alone: {fe}/{len(fresh)}   "
              + "/".join(f"{v:.3f}" for v in fnll))
    if len(pooled) < 6:
        print("  INCOMPLETE: fewer than six seeds; not called.")
    elif n_esc >= 4:
        print("  N3 HELD. Most seeds train out of the plateau, so KB's level is")
        print("  about capacity rather than about optimization.")
    else:
        print(f"  N3 FALSIFIED: {n_esc} of 6. d128 at L=16 is OPTIMIZATION-limited,")
        print("  not capacity-limited, and that reinterprets the bottom of the")
        print("  whole ladder: the ~2M floor becomes a floor on what this recipe")
        print("  TRAINS, not on what the task needs. It also means a better")
        print("  schedule -- not a bigger model -- is the next thing to try, and")
        print("  that is a change of direction, so it outranks N2.")


def score_n4():
    print("\n=== N4b -- OUT-OF-SAMPLE SUFFICIENCY, free, and reported ahead of")
    print(f"  everything if it fails: all six KB cells land on the N=48")
    print(f"  accuracy-to-solve curve within the LOO bar {LOO_BAR}.")
    print("  A cell falling OFF it is a seed-specific failure mode and would be")
    print("  the headline over N2.")
    print("  Run (the nine sufficiency dirs, plus stage K and stage N):")
    print("    python3 -m scripts.acc_vs_solve --scan-dir \\\n      " +
          " \\\n      ".join(SUFFICIENCY_DIRS + [SCAN_K, SCAN_N]))


def show_cells(cells):
    print("\n=== every KB cell on disk, K and N together ======================")
    print("  (printed in full so a cell under the wrong key shows up as a")
    print("   missing row rather than as a silently short rung)")
    ms = rung(cells, KB)
    if not ms:
        print("    nothing yet")
        return
    for m in sorted(ms, key=lambda r: r["seed"]):
        stage = "K" if m["seed"] in K_SEEDS else "N"
        flag = "ok" if m["flag"] == "ok" else m["flag"].upper()
        print(f"    [{stage}] seed {m['seed']}  rho_50 {m['rho50']:.3f} {flag:>9}  "
              f"mean_solve {m['mean_solve']:.4f}  {m['n_trainable']:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+", default=SCAN_N_DIRS)
    args = ap.parse_args()

    # Guard 4: a replicate must not enter a pooled yardstick.
    assert SCAN_N not in SCAN_DIRS, "SCAN_N leaked into the pooled yardstick dirs"
    assert SCAN_K not in SCAN_DIRS, "SCAN_K leaked into the pooled yardstick dirs"

    print("STAGE N -- does KB replicate? `no_mlp L=16 d128`, N=48, fresh seeds")
    print("=" * 70)
    print(f"  scanning: {' '.join(args.scan_dir)}")
    print(f"  bars (N=48, reused not refrozen): s* {S_RHO50} rho_50 / {S_MEAN} "
          f"mean_solve, free {FREE}, plateau {PLATEAU}, LOO {LOO_BAR}")
    print("  stage M's N=96 bars are a different size and appear nowhere below.")

    cells = load_pooled(args.scan_dir, args.boot)
    pooled = rung(cells, KB) or []
    fresh = subset(pooled, N_SEEDS)
    old = subset(pooled, K_SEEDS)
    print(f"\n  on disk: {len(old)} of stage K's 0/1/2, {len(fresh)} of stage N's "
          f"3/4/5")

    score_accuracy_first(pooled)
    score_n1(fresh)
    score_n0(fresh)
    score_n2(pooled, fresh)
    score_n3(pooled, fresh)
    score_n4()
    show_cells(cells)

    print("\n" + "=" * 70)
    print("Read in this order: N4a (accuracy, never censors) -> N1 (did it cross)")
    print("-> N0 (did the level hold) -> N2 (the pooled call) -> N3 (plateau).")
    print("N1 or N3 falsified outranks N2: a level from a censoring rung is a")
    print("bound, and an optimization-limited rung changes what to try next.")


if __name__ == "__main__":
    main()
