"""Score stage V: TRAIN the shared block at a longer schedule. Is the remaining
0.017 a schedule-LENGTH limit?

REGISTERED BEFORE THE FIRST CELL LANDS. This docstring IS the registration --
committed by scripts/slurm/stagev_register_2026-11-11.sbatch, which the array
scripts/slurm/sudoku2_n48_sched_V_2026-11-11.sbatch depends on with
--dependency=afterok. No cell can start ahead of it.

THE HYPOTHESIS, STATED FIRST

  One shared d320 block does direct deduction (stage S) but lands 0.017 past the
  free line: rho_50 0.757 (0.754/0.757/0.765), 3/3 crossing, val token accuracy
  0.4830 against the untied twin J2's 0.4940. S4 put all three seeds ON the N=48
  accuracy-to-solve curve, so the 0.017 is worth about 0.0011 of token accuracy.

  HYPOTHESIS V: the block is not short of capacity, roles or steps of training --
  it is short of APPLICATIONS. Stage T applied the trained block T = 4..24 times
  on frozen weights and found rho_50 bottoms exactly at the trained count (0.757)
  and rises to 0.819 at T=24, with the easy end collapsing in 2/3 seeds. The
  registered reading was "one weight set tuned to a 16-step schedule, not an
  iterable operator". But stage T could only ever move T at EVAL. Its own page
  said what it does not speak for: NO TIED CELL WAS EVER TRAINED AT ANOTHER T.

  And the task has chains longer than sixteen. The generator's own draw (40
  instances from this exact config) puts reasoning depth at median 5 rounds with
  range 1-19. Sixteen applications cannot serially propagate a 19-round chain,
  and the deepest chains are the SPARSE instances -- exactly where rho_50 is read.
  If that is the binding constraint, training at T=24 buys the 0.0011 and the
  gate opens; if it is not, the schedule-length explanation is dead.

WHY THIS IS THE CHEAPEST KNOB ON THE BOARD (the 11-10 objective)

  (630d + 58) + (4d^2 + 8d)k has no T in it. A tied cell at T=32 holds 613,818
  trainable parameters -- byte for byte the control's count. The object stays ONE
  block and ONE map iterated, h_{t+1} = F(h_t); only the number of iterations
  moves. So if a longer schedule clears the gate, the gate is cleared by the MOST
  tractable cell in the entire ladder, not by a bigger one. That is the opposite
  of stage U, where k buys accuracy at 1.7x and 3.0x the weights.

WHAT CHANGES: EXACTLY ONE KNOB, THE APPLICATION COUNT

    S1  tied_no_mlp T=16 d320 k=1   613,818   rho_50 0.757, 3/3   (stage S, disk)
    V1  tied_no_mlp T=24 d320 k=1   613,818   this stage, tasks 0-2
    V2  tied_no_mlp T=32 d320 k=1   613,818   this stage, tasks 3-5

  Everything else is S1's: d320, nhead 4, ff 1024, k=1, warmup 0, lr 1e-3
  constant, 240k x 50ep, `--shuffle-adj --fall`, the 14-point grid to 0.33 at 400
  per bin, seeds 0/1/2. k=1 is stage S's cell, RE-READ FROM DISK as the control,
  never re-run and never retyped.

  The optimizer is untouched on purpose: finding 47 FALSIFIED cosine decay on
  this task (rho_50 0.679 -> 0.830 and 0.564 -> 0.726), because finding 45's
  cells are step-bound rather than converged and decaying through the bulk
  descent starves them. Data size is untouched because n_train_instances is NOT
  in rung_key, so a 480k cell would pool into S1's rung and contaminate its own
  control. num_layers IS in rung_key, which is why this stage keys cleanly.

V-a  ACCURACY FIRST, AS EVERY STAGE, because it never censors. AND IT IS THE
     ADJUDICATOR HERE. Registered: final val token accuracy at T=24 is >= 0.4841
     -- the control's 0.4830 plus the 0.0011 the gate needs -- in at least 2 of 3
     seeds. The control's own seed spread is 0.0010, so the bar sits one full
     spread above it. Unlike stage U, where the required gain was INSIDE the seed
     spread and accuracy could not adjudicate, here the bar and the spread are
     stated against each other in advance and the scorer prints both.

V-b  THE GATE. rho_50 at T=24 below 0.740 in ALL THREE seeds. Lower is better.
     A CENSOR here cannot be a size effect: the count is identical to a cell that
     crosses 3/3, which is a cleaner version of the argument stage P had to make
     about H1's floor and stage U had to make about its own rungs.

V-c  THE EASY END, the one stage T failed outright. Registered: solve_rate >=
     0.90 at realized rho >= 0.85 in 3/3 seeds. NO T cleared this on frozen
     weights, the trained count included (0.895/0.887/0.900 -- one seed exactly
     ON the bar). If training at the longer schedule does not fix the easy end,
     then nothing about the collapse was a schedule and V's whole premise fails
     even if V-b opens.

V-d  THE TREND, which is why there are two rungs. Registered: rho_50(T=32) <=
     rho_50(T=24) <= 0.757. Stage T's own lesson is that one T cannot separate
     "schedule" from "operator", so a single trained point could not either. If
     T=32 comes back WORSE than T=24, the honest reading is a tuned schedule
     again, at a longer length, and the knob is exhausted. The two rungs may land
     in separate submissions; nothing is called until both are on disk (the stage
     O mistake, pre-empted as stage U pre-empted it).

V-e  THE MATCHED PAIR STAGE T SET UP WITHOUT BEING ABLE TO CLOSE. Stage T's
     frozen weights applied 24 times and stage V's weights TRAINED at 24
     applications are the same architecture, the same count, the same 24
     applications and the same eval grid. They differ in one thing: whether the
     schedule the weights were trained for is the schedule they are run at.
     Registered: trained T=24 beats frozen T=24 (0.819, 2/3 crossing, mean_solve
     0.1873, worst easy bin 0.0000) on all three of rho_50, crossings and the
     easy end. This is the comparison that says whether "tuned to a schedule" was
     a statement about THESE weights or about the recipe at any length. It cannot
     censor away: the frozen side is already measured.

V-f  WHAT RUNS NEXT, REGISTERED NOW RATHER THAN CHOSEN AFTERWARDS.
       V-a and V-b hold at T=24 -> the gate is cleared by a 613,818-parameter
                         single iterated map. The follow-up is the T-sweep on the
                         T=24 checkpoint (stage T's method, new weights): does a
                         block trained at 24 tolerate 32, or does the peak simply
                         move with the trained count? That is the operator
                         question, and only a trained-at-another-T cell can ask it.
       V-b holds only at T=32 -> report T=24 as the negative it is and say
                         plainly that the schedule is 32 applications long.
       V-a misses in 2/3 -> the schedule-length explanation is DEAD. Combined
                         with stage U's reading, the remaining candidate for the
                         0.017 is the training budget, and the registered arm is
                         100 epochs at constant 1e-3, k=1, T=16, one knob (stage
                         S's tail slope was about -0.00013 nll/epoch). Explicitly
                         NOT more width (saturated 0.4567/0.4830/0.4837), NOT
                         cosine (finding 47), NOT larger k if stage U is null.
       BOTH stages null -> the tied family does not clear the free line at N=48
                         under this recipe. That is a boundary worth publishing
                         rather than a failure to report, and it is said here
                         before either result is in.

GUARDS, each one a mistake this programme has already made once
  1. N=48 only (finding 55). No N=96 number appears here.
  2. rho_50 CENSORS. A censored seed is a result, not a missing number, and the
     bound-aware median is what gets quoted.
  3. SCAN_V stays OUT of SCAN_DIRS, as E/G/H/I/J/K/N/O/P/S/T/U all do, so a new
     rung cannot enter a pooled yardstick by accident.
  4. The k=1 T=16 control is re-read from disk and checked against stage S's
     frozen literals. The whole stage is a difference against it.
  5. Every count must equal (630d + 58) + (4d^2 + 8d) EXACTLY -- the SAME number
     at every T. A V cell whose count differs from the control's means the
     schedule knob was not pure and the tractability case collapses; this file
     exits rather than reporting it as a win.
  6. V1 and V2 key on num_layers, so they cannot pool into S1. They also cannot
     collide with stage T's frozen re-reads, which carry a STRING "T24" tail at
     num_layers=16. Both inequalities are asserted, not assumed.
  7. Both rungs are ABOVE H1's 879,418 censoring floor -- no, they are BELOW it,
     at 613,818, exactly as S1 is. That is stated here rather than quietly
     omitted: the censoring-floor argument does NOT protect this stage. What
     protects it is guard 5 -- an identical count to a cell that crosses 3/3, so
     a censor cannot be a size effect even though the count is under the floor.
  8. Stage T's grid stops at T=24, so V2 (T=32) has NO frozen counterpart and
     V-e is scored at T=24 only. Said in advance so the missing pair does not
     look like a gap in the data.
"""

import argparse
import os
import sys
from statistics import median

from scripts.minarch_analysis import (SCAN_DIRS, SCAN_H, SCAN_I, SCAN_J,
                                      cell_metrics, load_all, rung_key)
from scripts.stagec_score import (PLATEAU, S_MEAN, S_RHO50, escapes, rho_txt,
                                  rung, summarize)
from scripts.stagej_score import LOO_BAR, SUFFICIENCY_DIRS
from scripts.stagek_score import FREE, SCAN_K, STANDING
from scripts.stagen_score import SCAN_N, acc_of, load_pooled
from scripts.stageo_score import SCAN_O
from scripts.stagep_score import CENSOR_FLOOR, SCAN_P
from scripts.stages_score import SCAN_S
from scripts.staget_score import (EASY_BAR, EASY_RHO, SCAN_T, TRAINED_T,
                                  by_seed_t)
from scripts.stageu_score import SCAN_U

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_V = os.path.join(_SCR, "spin_gnn", "sudoku_n48_sched_V_2026-11-11")

# Rung keys are (L, nhead, arch, d_model, ff, warmup), L being the number of
# APPLICATIONS -- which is the knob, so the keys separate on their first element.
S1 = (TRAINED_T, 4, "tied_no_mlp", 320, 1024, 0)     # stage S, the control
V1 = (24, 4, "tied_no_mlp", 320, 1024, 0)            # this stage
V2 = (32, 4, "tied_no_mlp", 320, 1024, 0)            # this stage

SCHED_T = (24, 32)


def sched_count(d, k=1):
    """Embeddings and readout once, the block once -- and NO T anywhere in it.

    This is the whole tractability case for stage V in one function: the number
    it returns does not take the application count as an argument, because the
    application count does not cost parameters.
    """
    return (630 * d + 58) + (4 * d * d + 8 * d) * k


# The ladder, applications ascending. (key, label, T, predicted trainable)
LADDER = [
    (S1, "S1  T=16 (trained), the control", 16, sched_count(320)),
    (V1, "V1  T=24, trained at 24", 24, sched_count(320)),
    (V2, "V2  T=32, trained at 32", 32, sched_count(320)),
]

# Stage S's own result, frozen here and re-read from disk by control().
S1_TRAINABLE = 613818
S1_RHO50 = "0.757"
S1_RHO50_SEEDS = (0.754, 0.757, 0.765)
S1_ESCAPES = 3
S1_ACC = 0.4830
S1_ACC_SPREAD = 0.0010
S1_NLL_MEDIAN = 1.5864
S1_MEAN_SOLVE = 0.3148
S1_EASY_WORST = 0.8875                   # stage T, the trained count, 3 seeds

# Stage T's frozen T=24 re-read, frozen here for V-e and checked against disk.
T24_RHO50 = 0.819
T24_CROSS = 2
T24_MEAN_SOLVE = 0.1873
T24_EASY_WORST = 0.0000

# V-a's bar, computed from S4's steepness rather than asserted.
J2_ACC = 0.4940                  # d192 UNTIED -- a reference line, not a control
ACC_GAP = J2_ACC - S1_ACC        # 0.0110
RHO_GAP = 0.177                  # S1 0.757 - J2 0.580, the same pair
NEED = ACC_GAP * (float(S1_RHO50) - FREE) / RHO_GAP     # about 0.0011
ACC_BAR = round(S1_ACC + NEED, 4)                       # 0.4841
NEAR = 0.01                      # "within" for V-d

BASELINE_DIRS = [SCAN_S, SCAN_U, SCAN_P, SCAN_H, SCAN_I, SCAN_J, SCAN_K]


def control(cells):
    """Guard 4: this whole stage is a difference against stage S's T=16 cell."""
    print("\n=== CONTROL -- stage S's T=16 cell, re-read from disk ============")
    ms = rung(cells, S1)
    if not ms:
        print("  S1 (tied_no_mlp T=16 d320 k=1): NOT ON DISK in the dirs scanned.")
        print("  STOP: without the control there is no ladder. Every number this")
        print("  stage reports is a difference against it.")
        return False
    s = summarize(ms)
    n_esc, nll = escapes(ms)
    accs = sorted(a for _, a in acc_of(ms))
    got = (rho_txt(s), s["trainable"], n_esc)
    want = (S1_RHO50, S1_TRAINABLE, S1_ESCAPES)
    same = got == want
    print(f"  S1  rho_50 {got[0]:>9}  trainable {got[1]:>10,}  escapes {got[2]}/{s['n']}"
          f"  val acc median {median(accs):.4f}  median train NLL {median(nll):.4f}")
    print(f"      stage S froze: rho_50 {want[0]}, trainable {want[1]:,}, escapes "
          f"{want[2]}/3, acc {S1_ACC:.4f}, NLL {S1_NLL_MEDIAN:.4f}")
    print(f"      per-seed accuracy {' '.join(f'{a:.4f}' for a in accs)}  "
          f"spread {max(accs) - min(accs):.4f} (froze {S1_ACC_SPREAD:.4f})")
    print("      " + ("matches the frozen literals" if same else "<-- DISAGREES"))
    if not same:
        print("  STOP: the control drifted. Fix that before reading any T.")
    return same


def score_va(cells):
    """V-a. Read first, as every stage -- and here it is the adjudicator."""
    print("\n=== V-a -- VALIDATION TOKEN ACCURACY, READ FIRST AND DECISIVE ====")
    print(f"  registered: at T=24, final val token accuracy >= {ACC_BAR:.4f} in at")
    print(f"  least 2 of 3 seeds. That bar is the control's {S1_ACC:.4f} plus the")
    print(f"  {NEED:.4f} the gate needs, computed from S4's own steepness ({RHO_GAP:.3f}")
    print(f"  of rho_50 per {ACC_GAP:.4f} of accuracy), and it sits one full control")
    print(f"  seed spread ({S1_ACC_SPREAD:.4f}) above the control.")
    out = {}
    for key, label, t, _ in LADDER:
        ms = rung(cells, key)
        if not ms:
            print(f"  {label}: nothing on disk yet")
            out[t] = None
            continue
        rows = sorted(acc_of(ms))
        for seed, a in rows:
            mark = "CLEARS" if a >= ACC_BAR else "misses"
            print(f"    T={t:2d}  seed {seed}  val token acc {a:.4f}  {a - S1_ACC:+.4f}"
                  f" vs control  {mark}")
        vals = [a for _, a in rows]
        n_clear = sum(1 for a in vals if a >= ACC_BAR)
        print(f"    T={t:2d}  median {median(vals):.4f}  spread {max(vals) - min(vals):.4f}"
              f"  {n_clear}/{len(vals)} seeds at or above {ACC_BAR:.4f}")
        out[t] = {"vals": vals, "median": median(vals), "n_clear": n_clear,
                  "spread": max(vals) - min(vals)}
    v = out.get(24)
    if v is None:
        print("  V-a: PENDING at T=24, which is the rung the prediction names.")
        return out
    if v["n_clear"] >= 2:
        print(f"  V-a HOLDS at T=24: {v['n_clear']}/3 seeds clear {ACC_BAR:.4f}. The"
              " schedule")
        print("  length bought accuracy at ZERO parameter cost. Read V-b next.")
    else:
        print(f"  V-a FALSIFIED at T=24: {v['n_clear']}/3 seeds clear {ACC_BAR:.4f}.")
        print("  The schedule-length explanation for the remaining 0.017 is DEAD --")
        print("  and it is dead on the statistic that CANNOT censor, so no reading")
        print("  of V-b can rescue it. See V-f: the budget arm is next, not more T.")
    if v["spread"] > NEED:
        print(f"  NOTE, as registered for stage U and true again here: the seed")
        print(f"  spread {v['spread']:.4f} exceeds the required gain {NEED:.4f}, so a single")
        print("  seed clearing the bar is not evidence. The 2-of-3 form of V-a is")
        print("  what carries it.")
    return out


def report(cells):
    """Each new rung on its own, before anything is called."""
    out = {}
    for key, label, t, pred in LADDER[1:]:
        print(f"\n=== {label} -- {pred:,} trainable ==================")
        ms = rung(cells, key)
        if not ms:
            print("  nothing on disk yet.")
            out[t] = None
            continue
        s = summarize(ms)
        n_esc, nll = escapes(ms)
        print(f"  {s['n']}/3 seeds on disk, {s['n_cross']}/{s['n']} crossed 0.5")
        print(f"  rho_50 median {rho_txt(s):>9}   (gate at or below {FREE}; T=16 "
              f"{S1_RHO50}; untied twin J2 {STANDING[3]:.3f} at d192)")
        print(f"  mean_solve    {s['mean_solve']:>9.4f}   "
              f"[spread {s['spread_mean']:.4f}; T=16 {S1_MEAN_SOLVE:.4f}]")
        print(f"  trainable {s['trainable']:,}   (registration predicted {pred:,}"
              + ("" if s["trainable"] == pred else "  <-- DISAGREES, quote the disk")
              + ")")
        print(f"  plateau escapes {n_esc}/{s['n']}   final train NLL "
              + "/".join(f"{v:.4f}" for v in nll))
        for m in sorted(ms, key=lambda r: r["seed"]):
            flag = "ok" if m["flag"] == "ok" else m["flag"].upper()
            print(f"    seed {m['seed']}  rho_50 {m['rho50']:.3f} {flag:>9}  "
                  f"mean_solve {m['mean_solve']:.4f}")
        if s["trainable"] != pred:                              # guard 5
            print("  STOP: this is not the cell the registration describes.")
        if s["trainable"] != S1_TRAINABLE:                      # guard 5, the point
            print("  STOP: the count does NOT equal the control's. The application")
            print("  count was supposed to be free; if it is not, the tractability")
            print("  case for this stage is void and the cell is not comparable.")
        out[t] = s
    return out


def score_vb(seen):
    """The gate."""
    print("\n=== V-b -- DOES A LONGER TRAINED SCHEDULE CLEAR THE 0.740 GATE? ==")
    print(f"  registered: at T=24, 3 of 3 seeds cross and rho_50 < {FREE}.")
    print(f"  A CENSOR here is not a size effect: {S1_TRAINABLE:,} trainable is the")
    print(f"  SAME count as the control, which crosses 3/3. (It is below H1's")
    print(f"  {CENSOR_FLOOR:,} floor, so guard 7's usual argument does not apply --")
    print("  the identical count is what does the work instead.)")
    cleared = []
    for key, label, t, _ in LADDER[1:]:
        s = seen.get(t)
        if not s:
            print(f"  {label}: PENDING")
            continue
        if s["n"] < 3:
            print(f"  {label}: INCOMPLETE ({s['n']}/3 seeds). Reported, not called.")
            continue
        if not s["rho50_readable"]:
            print(f"  {label}: CENSORED -- no density bin cleared 0.5 in the median")
            print("    seed, at a parameter count that crosses 3/3 when trained at 16.")
            print("    Training at this length made the cell WORSE, not merely flat.")
            continue
        d_ctrl = s["rho50"] - float(S1_RHO50)
        free = s["rho50"] <= FREE
        print(f"  {label}: rho_50 {s['rho50']:.3f} ({'CLEARS' if free else 'past'} {FREE}), "
              f"{s['n_cross']}/3 crossing")
        print(f"    against T=16 {S1_RHO50}: {d_ctrl:+.3f} = {d_ctrl / S_RHO50:+.2f} s*   "
              f"(mean_solve {s['mean_solve']:.4f} vs {S1_MEAN_SOLVE:.4f}, "
              f"{(s['mean_solve'] - S1_MEAN_SOLVE) / S_MEAN:+.2f} s*)")
        if free and s["n_cross"] == 3:
            cleared.append((t, s))
            print("    GATE CLEARED at this application count, at ZERO extra cost.")
        elif abs(d_ctrl) <= NEAR:
            print(f"    WITHIN {NEAR} OF THE CONTROL: the application count is not")
            print("    what held the cell at 0.757.")
    if not cleared:
        print(f"  V-b FALSIFIED so far: no T in {SCHED_T} clears the gate with 3/3.")
    return cleared


def score_vc(rows_v):
    """V-c. The easy end, which stage T failed at every T."""
    print("\n=== V-c -- THE EASY END, the one stage T failed outright =========")
    print(f"  registered: solve_rate >= {EASY_BAR} at realized rho >= {EASY_RHO} in 3/3")
    print(f"  seeds. On FROZEN weights no T cleared this, the trained count")
    print(f"  included ({S1_EASY_WORST:.4f} worst of 0.895/0.887/0.900). If a longer")
    print("  trained schedule does not fix it, the collapse was never a schedule.")
    if not rows_v:
        print("  nothing on disk yet.")
        return None
    groups = by_seed_t(rows_v)
    out = {}
    for t in SCHED_T:
        worst = []
        for s in sorted({sd for sd, tt in groups if tt == t}):
            rs = [r for r in groups.get((s, t), []) if r["realized_rho"] >= EASY_RHO]
            if rs:
                worst.append(min(r["solve_rate"] for r in rs))
        if not worst:
            print(f"    T={t:2d}  PENDING")
            continue
        ok = len(worst) == 3 and min(worst) >= EASY_BAR
        print(f"    T={t:2d}  worst easy-bin solve_rate {min(worst):.4f} "
              f"(per seed {' '.join(f'{w:.3f}' for w in worst)})"
              f"   {'CLEARS 3/3' if ok else 'below the bar'}")
        out[t] = {"worst": min(worst), "per_seed": worst, "ok": ok}
    v = out.get(24)
    if v is not None:
        if v["ok"]:
            print("  V-c HOLDS at T=24: training at the longer schedule fixed the easy")
            print("  end that frozen weights could not hold at ANY T.")
        else:
            print(f"  V-c FALSIFIED at T=24: worst {v['worst']:.4f} < {EASY_BAR}. The easy-end")
            print("  failure survives training at the length it fails at, so it is a")
            print("  property of the recipe and not of the schedule.")
    return out


def score_vd(seen):
    """V-d. Two rungs, because one trained point cannot read a trend either."""
    print("\n=== V-d -- THE TREND ACROSS TRAINED SCHEDULE LENGTHS =============")
    print(f"  registered: rho_50(T=32) <= rho_50(T=24) <= {S1_RHO50}. A longer trained")
    print("  schedule should not hurt if the limit was serial depth.")
    have = [t for t in SCHED_T if seen.get(t)]
    if len(have) < 2:
        print(f"  {len(have)} of 2 rungs on disk. NOTHING IS CALLED until both land")
        print("  (the stage O mistake, pre-empted). The rungs may arrive in separate")
        print("  submissions; that is expected, not a gap.")
        return None
    lvl = []
    for t in (16,) + SCHED_T:
        s = seen.get(t)
        if t == 16:
            lvl.append((t, float(S1_RHO50), True))
            continue
        lvl.append((t, s["rho50"], s["rho50_readable"]))
    print("  rho_50 by trained applications: " + "  ".join(
        f"T={t}:{'CENSORED' if not ok else f'{v:.3f}'}" for t, v, ok in lvl))
    if not all(ok for _, _, ok in lvl):
        print("  A censored rung ends the trend reading: there is no level to order.")
        return None
    vals = [v for _, v, _ in lvl]
    mono = vals[2] <= vals[1] <= vals[0]
    if mono:
        print("  V-d HOLDS: rho_50 falls monotonically with the trained application")
        print("  count. Serial depth was a real constraint and more of it is better.")
    elif vals[1] < vals[0] and vals[2] > vals[1]:
        print("  V-d PARTIAL: T=24 improves on T=16 but T=32 gives it back. The")
        print("  schedule has an optimum that MOVES WITH THE TRAINED COUNT -- which")
        print("  is stage T's reading restated at a different length. Report the")
        print("  optimum, and claim no operator.")
    else:
        print("  V-d FALSIFIED: a longer trained schedule does not lower rho_50.")
    return {"levels": lvl, "monotone": mono}


def score_ve(seen, rows_t, vc):
    """V-e. Frozen-at-24 against trained-at-24: the pair stage T could not close."""
    print("\n=== V-e -- TRAINED AT 24 vs STAGE T's FROZEN AT 24 ===============")
    print("  Same architecture, same 613,818 parameters, same 24 applications, same")
    print("  grid. One difference: whether the weights were trained for the schedule")
    print(f"  they run at. Frozen (stage T): rho_50 {T24_RHO50:.3f}, {T24_CROSS}/3 crossing,")
    print(f"  mean_solve {T24_MEAN_SOLVE:.4f}, worst easy bin {T24_EASY_WORST:.4f}.")
    if not rows_t:
        print("  stage T's curves are not on disk here -- comparing against the frozen")
        print("  literals above, which its scorer (51919538) registered.")
    else:
        gt = by_seed_t(rows_t)
        ms = [cell_metrics(gt[(s, 24)], 2000) for s in sorted({sd for sd, tt in gt
                                                               if tt == 24})]
        if ms:
            ok = [m for m in ms if m["flag"] == "ok"]
            disk = median([m["rho50"] for m in ok]) if len(ok) * 2 > len(ms) else None
            print(f"  re-read from disk: rho_50 "
                  + (f"{disk:.3f}" if disk is not None else "CENSORED")
                  + f", {len(ok)}/{len(ms)} crossing, mean_solve "
                  f"{median([m['mean_solve'] for m in ms]):.4f}")
            if disk is not None and abs(disk - T24_RHO50) > 5e-4:
                print("  <-- DISAGREES with the frozen literal; quote the disk.")
    s = seen.get(24)
    if not s:
        print("  the trained side is PENDING. Nothing compared.")
        return None
    if not s["rho50_readable"]:
        print("  trained T=24 is CENSORED, which is WORSE than the frozen weights it")
        print("  was supposed to beat. Training at the length did not help at all.")
        return {"better": False}
    d = s["rho50"] - T24_RHO50
    better = (s["rho50"] < T24_RHO50 and s["n_cross"] >= T24_CROSS)
    print(f"  trained T=24: rho_50 {s['rho50']:.3f}, {s['n_cross']}/3 crossing, "
          f"mean_solve {s['mean_solve']:.4f}")
    print(f"  difference against frozen: {d:+.3f} rho_50 = {d / S_RHO50:+.2f} s*, "
          f"{s['n_cross'] - T24_CROSS:+d} crossings, "
          f"{s['mean_solve'] - T24_MEAN_SOLVE:+.4f} mean_solve")
    if vc and vc.get(24):
        print(f"  easy end: {vc[24]['worst']:.4f} trained vs {T24_EASY_WORST:.4f} frozen")
    if better:
        print("  V-e HOLDS: the 24-application schedule is learnable; stage T's 0.819")
        print("  measured weights run OFF their trained schedule, not a depth limit.")
        print("  'Tuned to a schedule' is a statement about THOSE weights.")
    else:
        print("  V-e FALSIFIED: training at 24 does not beat running 16-trained")
        print("  weights at 24. The degradation past 16 is a property of the RECIPE")
        print("  at any trained length, which is a stronger claim than stage T made.")
    return {"better": better, "delta": d}


def score_vf(va, cleared, vc):
    print("\n=== V-f -- WHAT RUNS NEXT, registered in advance ==================")
    v = va.get(24) if va else None
    if cleared:
        t_min = min(t for t, _ in cleared)
        print(f"  REPORTED MINIMUM: T={t_min} applications, {sched_count(320):,}")
        print("  trainable -- the SAME count as the control. The gate, if this holds,")
        print("  is cleared by the most tractable cell in the ladder: one block, one")
        print("  map, iterated longer.")
        print(f"  NEXT: stage T's sweep on the T={t_min} checkpoint. Does a block")
        print("  trained at that length tolerate others, or does the peak simply move")
        print("  with the trained count? Only a trained-at-another-T cell can ask it.")
        if vc and vc.get(t_min) and not vc[t_min]["ok"]:
            print("  CARRY THIS CAVEAT: the gate opened while V-c still failed, so the")
            print("  cell solves sparse instances it could not solve before and STILL")
            print("  drops easy ones. Report both; the gate is not the whole curve.")
        return
    if v is not None and v["n_clear"] < 2:
        print("  The schedule-length explanation is dead on the statistic that cannot")
        print("  censor. The registered arm is the TRAINING BUDGET: 100 epochs at")
        print("  constant 1e-3, k=1, T=16, one knob. Stage S's tail slope was about")
        print("  -0.00013 nll/epoch and the gate needs 0.0011 of accuracy.")
        print("  Explicitly NOT: more width (saturated), cosine (finding 47), or more")
        print("  T (this stage's own negative).")
        print("  If stage U is also null, the honest statement is that the tied family")
        print("  does not clear the free line at N=48 under this recipe -- a boundary,")
        print("  published as one, not a failure.")
        return
    print("  Nothing called yet: V-a at T=24 is what selects the branch, and it is")
    print("  either pending or holding without the gate. See V-b.")


def score_sufficiency():
    print("\n=== OUT-OF-SAMPLE SUFFICIENCY (the N=48 curve) ===================")
    print(f"  registered: both rungs land on the fitted accuracy-to-solve curve")
    print(f"  within the LOO bar {LOO_BAR}, as S1's seeds did (+0.0165/-0.0034/")
    print("  +0.0065). A cell OFF it is the headline: iterating longer would then")
    print("  change the accuracy-to-solve RELATION, not the level. N=48 only.")
    print("  Run:")
    print("    python3 -m scripts.acc_vs_solve --scan-dir \\\n      " +
          " \\\n      ".join(SUFFICIENCY_DIRS
                             + [SCAN_K, SCAN_N, SCAN_O, SCAN_P, SCAN_S, SCAN_U,
                                SCAN_V]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+",
                    default=[SCAN_V] + BASELINE_DIRS + SCAN_DIRS)
    args = ap.parse_args()

    # Guard 3: a new rung must not enter a pooled yardstick.
    for name, d in (("SCAN_V", SCAN_V), ("SCAN_U", SCAN_U), ("SCAN_T", SCAN_T),
                    ("SCAN_S", SCAN_S), ("SCAN_P", SCAN_P), ("SCAN_O", SCAN_O),
                    ("SCAN_K", SCAN_K), ("SCAN_N", SCAN_N)):
        assert d not in SCAN_DIRS, f"{name} leaked into the pooled yardstick dirs"

    # Guard 5, as assertions on the registration's own arithmetic. The headline
    # of this stage is that the count does not move, so it is asserted first.
    assert sched_count(320) == S1_TRAINABLE == 613818, sched_count(320)
    # Guard 6: the keys separate from the control and from stage T's re-reads.
    for key, _, t, _ in LADDER[1:]:
        assert key != S1, ("a V rung keys identically to the control", key)
        assert key[0] == t, (key, t)
        assert key != S1 + ("T%d" % t,), (
            "a trained-at-T rung must not key like stage T's frozen re-read", key)
    assert V1 != V2
    # And the key the loader will actually build for a V cell, from a synthetic
    # row: asserted rather than assumed, because the whole stage rests on these
    # cells not pooling into S1's rung.
    for t in SCHED_T:
        row = {"num_layers": t, "nhead": 4, "arch_mode": "tied_no_mlp",
               "d_model": 320, "dim_feedforward": 1024, "warmup_epochs": 0,
               "epochs": 50, "n_blocks": 1, "n_iters_eval": t}
        assert rung_key(row) == (t, 4, "tied_no_mlp", 320, 1024, 0), rung_key(row)
        assert rung_key(row) != S1, "a V cell would pool into stage S's rung"
    # Guard 7, stated rather than omitted: these rungs are BELOW the censoring
    # floor, so that argument does not protect them. The identical count does.
    assert sched_count(320) < CENSOR_FLOOR, sched_count(320)
    # V-a's bar is derived, not typed. If S4's steepness is ever refit this
    # assertion is what will catch the bar having silently moved.
    assert abs(NEED - 0.0011) < 0.0002, NEED
    assert ACC_BAR == 0.4841, ACC_BAR

    print("STAGE V -- one block trained at a LONGER schedule: depth or recipe?")
    print("=" * 70)
    print(f"  scanning: {' '.join(args.scan_dir)}")
    print(f"  bars (N=48, reused not refrozen): s* {S_RHO50} rho_50 / {S_MEAN} "
          f"mean_solve, gate {FREE}, plateau {PLATEAU}, LOO {LOO_BAR}")
    print("  stage M's N=96 bars are a different size and appear nowhere below.")
    print(f"  the ladder at FIXED width d320 and FIXED k=1: T=16 -> 24 -> 32, and")
    print(f"  {sched_count(320):,} trainable AT EVERY RUNG -- the application count")
    print("  is the one knob in this programme that costs nothing.")
    print("  h_{t+1} = F(h_t), iterated T times. Still one map, still one block.")
    print(f"  the instance distribution runs to 19 rounds of deduction; T=16 cannot")
    print("  serially propagate the deepest chains, which is why this stage exists.")

    cells = load_pooled(args.scan_dir, args.boot)
    base_ok = control(cells)

    rows_v = load_all([SCAN_V]) if os.path.isdir(SCAN_V) else []
    rows_t = load_all([SCAN_T]) if os.path.isdir(SCAN_T) else []

    va = score_va(cells)                 # accuracy first, and decisive here
    seen = report(cells)
    cleared = score_vb(seen)
    vc = score_vc(rows_v)
    score_vd(seen)
    score_ve(seen, rows_t, vc)
    score_vf(va, cleared, vc)
    score_sufficiency()

    print("\n" + "=" * 70)
    print("Read in this order: V-a (accuracy, never censors, and the adjudicator)")
    print("-> V-b (the gate) -> V-c (the easy end stage T failed) -> V-d (the trend")
    print("across trained lengths) -> V-e (trained-at-24 against frozen-at-24).")
    print("V-a outranks V-b by registration: if the accuracy the gate needs is not")
    print("there, no reading of rho_50 can supply it. V-f's follow-ups are in the")
    print("docstring and are scored by the next stage, not by this file.")

    if not base_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
