"""Score stage T: is the shared block an ITERABLE operator, or a 16-step schedule?

REGISTERED BEFORE THE FIRST CELL LANDS. This docstring IS the registration --
committed by scripts/slurm/staget_register_2026-11-11.sbatch, which the eval
array scripts/slurm/sudoku2_n48_tiedT_T_2026-11-11.sbatch depends on with
--dependency=afterok. No cell can start ahead of it.

WHAT STAGE S RETURNED AND WHY THIS IS THE FOLLOW-UP IT NAMED

    P1  tied_no_mlp T=16 d192   270,010  rho_50 CENSORED   escapes 0/3  acc 0.4567
    S1  tied_no_mlp T=16 d320   613,818  rho_50   0.757    escapes 3/3  acc 0.4830
    J2  no_mlp      L=16 d192 2,504,890  rho_50   0.580    escapes 3/3  acc 0.4940

  Stage P's plateau was a CAPACITY floor, not an obstruction created by sharing:
  at d320 all three tied seeds escape 1.64 and all three CROSS 0.5, which stage
  P's d192 never did. So one block applied 16 times does deduce -- it lands
  0.017 past the free line 0.740 rather than clearing it. Stage S's own S5
  registered the T-sweep as the follow-up if the tied cell trained, and this is
  that sweep, moved to d320 because d320 is the cell that trained 3 of 3 (the
  d448 arm lost 2 seeds to an epoch-0 divergence that diagnostic 51860479 traced
  to warmup, not to the cell).

WHAT THIS STAGE MEASURES, AND WHY IT COSTS NO TRAINING

  A tied cell has a dial no untied cell has: apply the ONE shared block T times
  instead of 16, on FIXED weights. If the block is a deduction step, T is the
  number of rounds and the trained weights should tolerate more of them; if the
  network instead learned a particular 16-step trajectory, T=16 is a peak and
  both directions fall away. Same checkpoints, same 14-point grid, same 400
  instances per bin (seed_start fixed, so every T sees IDENTICAL instances and
  the per-bin differences are paired).

    T in {4, 8, 12, 16, 20, 24} x seeds {0, 1, 2}   = 18 grids, no training

T0  THE HEADLINE. rho_50 does not get WORSE above the trained count: T=24 is
    within 0.01 of T=16, and no T >= 16 is more than 0.01 worse than the best.
    Then the block is an iterable operator, "one application = one deduction
    round" is a property of the ARCHITECTURE rather than a reading of a trained
    network, and the object a theory has to analyse is a single map F with
    h_{t+1} = F(h_t) plus a stopping rule.
    THE COMPETING HYPOTHESIS PREDICTS A PEAK AT EXACTLY 16, falling away on
    both sides: the weights encode a 16-step schedule and the recurrence is
    cosmetic. That is a clean negative, it is equally publishable, and it is
    the result if it is what the numbers say.

T1  READ FIRST, BEFORE ANY LEVEL, because it cannot censor. At the easy end
    (realized rho >= 0.85, where dd_mean_T is near 1 and every free cell is at
    ceiling) solve_rate stays >= 0.90 for every T >= 12. If the easy bins
    themselves collapse at T=20 or T=24, the extra applications are destroying
    the representation -- a statement about the recurrence's stability, not
    about deduction -- and T0 is then read as that and nothing more.

T2  HOW MANY OF THE 16 APPLICATIONS ARE ACTUALLY USED. Registered: solve_rate
    falls off below some T* <= 16. T* is reported as the smallest T whose
    mean_solve is within 0.01 of T=16's. T* < 16 means the trained object is
    really a T*-round machine and the last 16 - T* applications are identity-ish
    padding, which would make the analysable object smaller still.

T3  WHERE MORE ROUNDS SHOULD HELP, IF THEY HELP AT ALL. dd_mean_T is 13.8 at
    the grid floor (rho 0.33) against 1.1 at rho 0.94, so extra applications
    should pay at the DEEP end specifically. Registered: T=20 and T=24 show a
    positive paired delta against T=16 at the two lowest-rho bins for at least
    2 of 3 seeds. A gain that is instead spread evenly over all 14 bins is a
    generic benefit and is reported as such, not as evidence about rounds.

T4  GUARDS, each one a mistake this programme has already made once.
    1. FIXED WEIGHTS. Nothing is trained here, so the only spread is the three
       trained seeds; no bootstrap over T is reported as if it were a seed.
    2. PAIRED. seed_start is fixed, so each T sees the same 400 instances per
       bin. Differences are per-bin paired differences and are not re-derived
       from the pooled curve.
    3. SCAN_T STAYS OUT OF SCAN_DIRS, as E/G/H/I/J/K/N/O/P/S all do, so a
       re-evaluated checkpoint cannot enter a pooled yardstick. The T rows also
       carry n_iters_eval, which rung_key appends when it differs from
       num_layers -- without that, six T curves would share one cell key and be
       averaged into a single number with nothing saying so.
    4. NO CONTROL ARM EXISTS, and none is invented. An untied stack's depth is
       its weights; `--n-iters` refuses there rather than quietly scoring 16.
    5. rho_50 CENSORS. A censored T is a result, and the bound-aware median is
       what gets quoted.
    6. N=48 only (finding 55). No N=96 number appears here.

T5  WHAT RUNS NEXT, REGISTERED NOW.
      T0 holds      -> the fixed-point probe on the same checkpoints: does
                       ||h_{t+1} - h_t|| fall toward 0 with t, and does the
                       stopping round line up with each instance's DD depth?
                       That is ResidualCapture's job and it is only trustworthy
                       since the 2026-11-10 application-index fix.
      T0 falsified  -> the recurrence is a 16-step schedule. Report it, and the
                       analyzability route goes back to the shallow untied hub
                       (`full L=6 d256`, 6 composed layers, rho_50 0.592), which
                       is already free.
      T1 collapses  -> stability first: the next question is norm growth across
                       applications, not deduction.
"""
import argparse
import os
import sys
from collections import defaultdict
from statistics import median

from scripts.minarch_analysis import cell_metrics, load_all
from scripts.stagec_score import PLATEAU, S_MEAN, S_RHO50
from scripts.stagek_score import FREE
from scripts.stagep_score import CENSOR_FLOOR

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_T = os.path.join(_SCR, "spin_gnn", "sudoku_n48_tiedT_T_2026-11-11")

TRAINED_T = 16          # what the d320 checkpoints were trained with
T_GRID = (4, 8, 12, 16, 20, 24)
EASY_RHO = 0.85         # T1's easy end
EASY_BAR = 0.90         # T1's floor there
NEAR = 0.01             # "within" for T0 and T2
# Stage S's d320 numbers, frozen here so a drifted input cannot pass unnoticed.
S1_RHO50 = 0.757
S1_TRAINABLE = 613818


def by_seed_t(rows):
    """{(seed, n_iters_eval): [row, ...]} -- the grouping rung_key also makes.

    Grouped here explicitly as well, because this file's whole subject is the
    difference between T values and a scorer that relied on someone else's key
    for that distinction would be one rename away from averaging them.
    """
    out = defaultdict(list)
    for r in rows:
        t = int(r.get("n_iters_eval") or r["num_layers"])
        out[(int(r["seed"]), t)].append(r)
    return out


def paired_delta(a, b):
    """Per-bin solve_rate differences a - b, keyed by requested rho.

    Both sides are the same instances by construction (T4.2); this asserts it
    rather than trusting it, by checking the realized densities line up.
    """
    bb = {round(float(r["rho"]), 6): r for r in b}
    out = []
    for r in a:
        k = round(float(r["rho"]), 6)
        if k not in bb:
            continue
        assert abs(float(r["realized_rho"]) - float(bb[k]["realized_rho"])) < 1e-6, (
            "same rho, different realized density -- the instances are not paired", k)
        out.append((k, float(r["realized_rho"]), r["solve_rate"] - bb[k]["solve_rate"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+", default=[SCAN_T])
    args = ap.parse_args()

    from scripts.minarch_analysis import SCAN_DIRS
    assert SCAN_T not in SCAN_DIRS, "SCAN_T leaked into the pooled yardstick dirs"

    print("STAGE T -- one tied block applied T times, FIXED WEIGHTS")
    print("=" * 70)
    print(f"  checkpoints: stage S's tied_no_mlp T=16 d320, {S1_TRAINABLE:,} trainable,")
    print(f"               rho_50 {S1_RHO50:.3f}, the cell that trained 3 of 3")
    print(f"  bars (N=48): free {FREE}, s* {S_RHO50} rho_50 / {S_MEAN} mean_solve, "
          f"plateau {PLATEAU}, censoring floor {CENSOR_FLOOR:,}")
    print(f"  T grid {T_GRID}, trained at T={TRAINED_T}; no training in this stage")

    present = [d for d in args.scan_dir if os.path.isdir(d)]
    for d in args.scan_dir:
        if d not in present:
            print(f"  (no such run dir yet: {d})")
    rows = load_all(present) if present else []
    if not rows:
        print("\nnothing on disk yet -- nothing scored.")
        return
    groups = by_seed_t(rows)

    seeds = sorted({s for s, _ in groups})
    found_t = sorted({t for _, t in groups})
    print(f"  on disk: seeds {seeds}, T {found_t}")
    for r in rows[:1]:
        assert int(r.get("n_trainable_params") or 0) == S1_TRAINABLE, (
            "these are not stage S's d320 checkpoints", r.get("n_trainable_params"))
        assert int(r.get("n_iters_trained") or TRAINED_T) == TRAINED_T, (
            "checkpoint was not trained at T=16", r.get("n_iters_trained"))

    cells = {k: cell_metrics(v, args.boot) for k, v in groups.items()}

    # ---- T1 first: the easy end, which cannot censor -----------------------
    print("\n=== T1 -- THE EASY END, READ FIRST (it never censors) ============")
    print(f"  registered: solve_rate >= {EASY_BAR} at realized rho >= {EASY_RHO}")
    print("  for every T >= 12. A collapse here is about the recurrence's")
    print("  stability, not about deduction.")
    t1_ok = True
    for t in found_t:
        worst = []
        for s in seeds:
            rs = [r for r in groups.get((s, t), []) if r["realized_rho"] >= EASY_RHO]
            if rs:
                worst.append(min(r["solve_rate"] for r in rs))
        if not worst:
            continue
        bad = t >= 12 and min(worst) < EASY_BAR
        t1_ok = t1_ok and not bad
        print(f"    T={t:2d}  worst easy-bin solve_rate {min(worst):.4f} "
              f"(per seed {' '.join(f'{w:.3f}' for w in worst)})"
              + ("   <-- COLLAPSED" if bad else ""))
    print(f"  T1 {'HOLDS' if t1_ok else 'FALSIFIED'}")

    # ---- the level, per T ---------------------------------------------------
    print("\n=== THE LEVEL, per T (median over seeds, bound-aware) ============")
    print(f"  {'T':>3} {'rho_50':>9} {'crossed':>8} {'mean_solve':>11} {'vs T=16':>8}")
    level = {}
    for t in found_t:
        ms = [cells[(s, t)] for s in seeds if (s, t) in cells]
        ok = [m for m in ms if m["flag"] == "ok"]
        lvl = median([m["rho50"] for m in ok]) if len(ok) * 2 > len(ms) else float("nan")
        msolve = median([m["mean_solve"] for m in ms])
        level[t] = {"rho50": lvl, "crossed": len(ok), "n": len(ms), "mean_solve": msolve}
    base = level.get(TRAINED_T)
    for t in found_t:
        v = level[t]
        txt = f"{v['rho50']:.3f}" if v["rho50"] == v["rho50"] else "CENSORED"
        delta = (f"{v['mean_solve'] - base['mean_solve']:+.4f}"
                 if base and t != TRAINED_T else "--")
        print(f"  {t:3d} {txt:>9} {v['crossed']}/{v['n']:>6} {v['mean_solve']:11.4f} {delta:>8}")

    # ---- T0 ------------------------------------------------------------------
    print("\n=== T0 -- ITERABLE OPERATOR, OR A 16-STEP SCHEDULE? =============")
    above = [t for t in found_t if t >= TRAINED_T]
    readable = [t for t in above if level[t]["rho50"] == level[t]["rho50"]]
    if base is None or TRAINED_T not in found_t:
        print("  PENDING: the T=16 reference is not on disk.")
    elif not readable:
        print("  every T >= 16 CENSORED. Read as: the operator does not survive")
        print("  being iterated past its trained count. T0 falsified.")
    else:
        best = min(level[t]["rho50"] for t in readable)
        worst_above = max(level[t]["rho50"] for t in readable)
        t24 = level.get(24, {}).get("rho50", float("nan"))
        gap = abs(t24 - level[TRAINED_T]["rho50"]) if t24 == t24 else float("nan")
        print(f"  best rho_50 over T >= 16: {best:.3f};  worst: {worst_above:.3f}")
        print(f"  |T=24 - T=16| = {gap:.3f}" if gap == gap else "  T=24 CENSORED")
        holds = (gap == gap and gap <= NEAR and worst_above - best <= NEAR)
        print(f"  T0 {'HOLDS -- one map F, applied T times' if holds else 'FALSIFIED'}")
        if not holds:
            peak = min(found_t, key=lambda t: level[t]["mean_solve"] * -1)
            print(f"  mean_solve peaks at T={peak}; a peak AT the trained count is the")
            print("  competing hypothesis, and it is the result rather than a failure.")

    # ---- T2 ------------------------------------------------------------------
    print("\n=== T2 -- HOW MANY APPLICATIONS ARE ACTUALLY USED ================")
    if base:
        within = [t for t in found_t
                  if t <= TRAINED_T and abs(level[t]["mean_solve"] - base["mean_solve"]) <= NEAR]
        if within:
            print(f"  T* = {min(within)}: the smallest T within {NEAR} of T=16's mean_solve.")
            if min(within) < TRAINED_T:
                print(f"  {TRAINED_T - min(within)} of the 16 applications change nothing "
                      "measurable; the analysable object is smaller than it looks.")
        else:
            print("  no T below 16 comes within 0.01: every application is used.")
    else:
        print("  PENDING.")

    # ---- T3 ------------------------------------------------------------------
    print("\n=== T3 -- DOES DEPTH PAY AT THE DEEP END? (paired, per bin) ======")
    print("  registered: positive paired delta at the two lowest-rho bins for")
    print("  >= 2 of 3 seeds, at T=20 and T=24.")
    for t in [x for x in (20, 24) if x in found_t]:
        wins = 0
        for s in seeds:
            a, b = groups.get((s, t)), groups.get((s, TRAINED_T))
            if not a or not b:
                continue
            d = sorted(paired_delta(a, b), key=lambda x: x[1])[:2]
            gain = sum(x[2] for x in d) / len(d) if d else 0.0
            wins += gain > 0
            print(f"    T={t:2d} seed {s}: deep-bin delta "
                  + " ".join(f"rho{x[1]:.3f} {x[2]:+.4f}" for x in d))
        print(f"    T={t:2d}: {wins}/{len(seeds)} seeds gained at the deep end")

    print("\n" + "=" * 70)
    print("Read in this order: T1 (easy end, never censors) -> the level table ->")
    print("T0 (iterable or scheduled) -> T2 (how many rounds are used) -> T3")
    print("(does depth pay where depth is needed). T1 outranks T0: a representation")
    print("that falls apart at T=24 says nothing about deduction at T=24.")


if __name__ == "__main__":
    main()
