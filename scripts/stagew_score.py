"""Score stage W: give the shared block TWICE THE TRAINING BUDGET. Is the
remaining 0.017 a budget limit?

REGISTERED BEFORE THE FIRST CELL LANDS. This docstring IS the registration --
committed by scripts/slurm/stagew_register_2026-11-12.sbatch, which the array
scripts/slurm/sudoku2_n48_budget_W_2026-11-12.sbatch depends on with
--dependency=afterok. No cell can start ahead of it.

THIS IS THE ARM STAGE V REGISTERED, NOT ONE CHOSEN AFTERWARDS

  Stage V's V-f, written before any V cell landed, said: "V-a misses in 2/3 ->
  the schedule-length explanation is DEAD. Combined with stage U's reading, the
  remaining candidate for the 0.017 is the training budget, and the registered
  arm is 100 epochs at constant 1e-3, k=1, T=16, one knob. Explicitly NOT more
  width (saturated), NOT cosine (finding 47), NOT larger k if stage U is null."

  V-a came back 0/3 at T=24 (0.4833/0.4612/0.4834 against the 0.4841 bar; the
  control's three seeds are 0.4830 within 0.0010). So this file runs the arm
  that branch names, at the k it names. The k=2-at-100-epochs question -- stage
  U's two basin seeds were still descending when the 50-epoch budget cut them
  off -- is a DIFFERENT arm and is registered in W-f, not smuggled in here as a
  second rung. Two knobs at once is how a stage stops being able to answer.

THE HYPOTHESIS, STATED FIRST

  HYPOTHESIS W: one shared d320 block does direct deduction (stage S) but lands
  0.017 past the free line, and it is not short of capacity (width saturated),
  roles (stage U), or serial applications (stage V). It is short of STEPS OF
  OPTIMIZATION. Stage S's cells are step-bound rather than converged: the tail
  slope at epoch 50 was about -0.00013 nll/epoch, finding 45 called the family
  step-bound, and finding 47 falsified cosine decay precisely because decaying
  through the bulk descent starves a step-bound cell. Fifty more epochs at the
  same constant 1e-3 is the knob that finding's own reasoning points at.

  And the gate needs very little: S4 put all three S1 seeds ON the N=48
  accuracy-to-solve curve, so 0.017 of rho_50 is worth about 0.0011 of token
  accuracy -- one eighth of the 0.0110 that separates the tied cell from its
  untied twin.

WHY THE BUDGET IS THE OTHER FREE KNOB

  (630d + 58) + (4d^2 + 8d)k has no epoch count in it either. W1 holds 613,818
  trainable parameters -- byte for byte the control's count, and byte for byte
  stage V's at every T. The object does not change at all: one block, one map,
  h_{t+1} = F(h_t), sixteen applications. Only the number of gradient steps
  moves. So if the gate opens here it is opened by the most tractable cell in
  the ladder, exactly as stage V would have opened it, and the cost is GPU hours
  rather than structure. That is the whole reason this arm outranks width, k and
  data: every one of those buys the gate with a less analyzable object.

WHAT CHANGES: EXACTLY ONE KNOB, THE EPOCH COUNT

    S1  tied_no_mlp T=16 d320 k=1   50 ep   613,818   rho_50 0.757, 3/3  (disk)
    W1  tied_no_mlp T=16 d320 k=1  100 ep   613,818   this stage, tasks 0-2

  Everything else is S1's: d320, nhead 4, ff 1024, k=1, T=16, warmup 0, lr 1e-3
  CONSTANT, 240k instances, `--shuffle-adj --fall`, the 14-point grid to 0.33 at
  400 per bin, seeds 0/1/2. S1 is RE-READ FROM DISK as the control, never re-run
  and never retyped.

  The learning rate stays constant for a second reason beyond finding 47: a
  constant schedule is what makes epoch 50 of a 100-epoch run the same point in
  optimization as epoch 50 of a 50-epoch run. Under cosine or any epoch-indexed
  decay the two runs would differ from step one and W-c below could not be read
  at all. Data size stays at 240k because n_train_instances is NOT in rung_key,
  so a 480k cell pools into stage S's rung and destroys its own control; epochs
  IS in rung_key, which is why this stage keys cleanly (guard 6).

W-a  ACCURACY FIRST, AS EVERY STAGE, because it never censors. AND IT IS THE
     ADJUDICATOR, as in stage V. Registered: final val token accuracy at 100
     epochs is >= 0.4841 -- the control's 0.4830 plus the 0.0011 the gate needs
     -- in at least 2 of 3 seeds. The bar is IMPORTED from stagev_score, not
     retyped, so the two stages are adjudicated against one number; it sits one
     full control seed spread (0.0010) above the control.

W-b  THE GATE. rho_50 at 100 epochs below 0.740 in ALL THREE seeds. Lower is
     better. A CENSOR here cannot be a size effect: the count is identical to a
     cell that crosses 3/3 at half the budget.

W-c  DID THE BUDGET BUY ANYTHING AT ALL -- the question that decides whether
     this stage can speak. Registered: within each run, train NLL at epoch 100
     is below train NLL at epoch 50 by at least 0.0065 (stage S's tail slope,
     -0.00013/epoch, times the 50 added epochs) in at least 2 of 3 seeds. This
     is a WITHIN-RUN paired comparison -- same seed, same data order, same
     weights up to epoch 50 -- and it is the only paired number this stage has.
     If it FAILS, the cell was already converged at 50 epochs, the budget
     hypothesis never had a chance, and W is a null about CONVERGENCE rather
     than about budget. That reading is registered here so it cannot be
     presented afterwards as if the stage had tested something else.

W-d  DID THE LOSS BECOME ACCURACY -- with its confound stated in advance. Stage
     U's two basin seeds (k=2 s2 and k=4 s1) reached train NLL 1.5371/1.5306 and
     val accuracy 0.4929/0.4948 while every k=1 seed sat at about 1.585/0.4830.
     The tempting reading is "NLL below ~1.56 means the better basin". IT DOES
     NOT MEAN THAT AT 100 EPOCHS: twice the budget lowers train NLL
     mechanically, so the threshold alone is confounded with the knob. The basin
     claim therefore requires BOTH train NLL below the derived 1.561 line AND
     val accuracy at or above 0.4841, IN THE SAME SEED. Registered: if NLL falls
     below the line while W-a misses, the extra budget bought TRAINING loss
     without VALIDATION accuracy, the 0.017 is a generalization gap rather than
     a budget one, and that is a different and more informative negative than
     stage V's.

W-e  THE MATCHED PAIR, and its one honest limitation. W1 at 100 epochs against
     S1 at 50 is the same architecture, the same 613,818 parameters, the same 16
     applications and the same eval grid, differing in one knob. But the
     100-epoch run does NOT evaluate rho_50 at epoch 50, so the rho_50 and
     accuracy comparisons are necessarily CROSS-RUN (seeds diverge from step
     one), while W-c's NLL comparison is within-run. Said in advance so the
     asymmetry is not discovered as a gap later.

W-f  WHAT RUNS NEXT, REGISTERED NOW RATHER THAN CHOSEN AFTERWARDS.
       W-a and W-b hold -> the gate is cleared by ONE SHARED BLOCK applied
                         sixteen times, 613,818 parameters, at no structural
                         cost. That answers the 11-10 objective: the simplest
                         object in the ladder learns DD to the gate. Follow-up
                         is stage T's frozen T-sweep on the 100-epoch
                         checkpoint -- a cell that needed the budget may or may
                         not be an iterable operator -- and a re-read of the
                         reported minimum.
       W-c holds, W-a misses -> the budget bought loss and not accuracy. The
                         0.017 is a GENERALIZATION gap. The registered arm is
                         then DATA, 480k instances, which must be run as a
                         MATCHED PAIR at 240k and 480k inside one stage because
                         n_train_instances is not in rung_key and cannot be
                         differenced against stage S.
       W-c fails -> the cell was converged at 50 epochs. Budget was never the
                         constraint, and no reading of W-b speaks to it. The
                         next arm is the same data pair, for the same reason.
       W-d shows NLL below the line in a k=1 seed -> stage U's basin is
                         reachable without extra roles, and the arm after that
                         is 100 epochs at k=2: do roles and budget compound, or
                         is there one basin that either run finds?
       ALL NULL -> with width, cosine, best-val selection, role count (U),
                         schedule length (V) and budget (W) closed on evidence,
                         the honest statement is that the tied family does not
                         clear the free line at N=48 under this recipe. That is
                         a boundary worth publishing as one, and it is said here
                         before the result is in. The standing minimum stays
                         J2's attention-only L=16 d192 at 0.580.

GUARDS, each one a mistake this programme has already made once
  1. N=48 only (finding 55). No N=96 number appears here.
  2. rho_50 CENSORS. A censored seed is a result, not a missing number, and the
     bound-aware median is what gets quoted.
  3. SCAN_W stays OUT of SCAN_DIRS, as E/G/H/I/J/K/N/O/P/S/T/U/V all do, so a
     new rung cannot enter a pooled yardstick by accident.
  4. The 50-epoch control is re-read from disk and checked against stage S's
     frozen literals, which are IMPORTED from stagev_score rather than retyped.
     The whole stage is a difference against it.
  5. The count must equal (630d + 58) + (4d^2 + 8d) EXACTLY -- the same number
     at 50 epochs and at 100. A W cell whose count differs means the budget knob
     was not pure and the tractability case collapses; this file exits rather
     than reporting it as a win.
  6. W1 keys on the EPOCHS tail, which exists only because epochs != EP_DEFAULT
     (50). rung_key appends it as a bare int, distinguishable by type from
     stage U's "k2" and stage T's "T24" strings. The key the loader will
     actually build is asserted from a synthetic row, not assumed.
  7. At 613,818 this rung sits BELOW H1's 879,418 censoring floor, so guard 7's
     usual argument does NOT protect this stage, exactly as it did not protect
     stage V. What protects it is guard 5 -- an identical count to a cell that
     crosses 3/3 at half the budget.
  8. W-c is within-run; W-a, W-b and W-e are cross-run. Stated in W-e and
     printed next to every cross-run number rather than left implicit.
"""

import argparse
import json
import os
import sys
from statistics import median

from scripts.minarch_analysis import (EP_DEFAULT, SCAN_DIRS, SCAN_H, SCAN_I,
                                      SCAN_J, history, rung_key)
from scripts.stagec_score import (PLATEAU, S_MEAN, S_RHO50, escapes, rho_txt,
                                  rung, summarize)
from scripts.stagej_score import LOO_BAR, SUFFICIENCY_DIRS
from scripts.stagek_score import FREE, SCAN_K, STANDING
from scripts.stagen_score import SCAN_N, acc_of, load_pooled
from scripts.stageo_score import SCAN_O
from scripts.stagep_score import CENSOR_FLOOR, SCAN_P
from scripts.stages_score import SCAN_S
from scripts.staget_score import SCAN_T, TRAINED_T
from scripts.stageu_score import SCAN_U
# Guard 4 and W-a's bar: imported, never retyped, so stage V and stage W are
# adjudicated against one number and one control.
from scripts.stagev_score import (ACC_BAR, NEED, S1, S1_ACC, S1_ACC_SPREAD,
                                  S1_ESCAPES, S1_MEAN_SOLVE, S1_NLL_MEDIAN,
                                  S1_RHO50, S1_TRAINABLE, SCAN_V, sched_count)

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_W = os.path.join(_SCR, "spin_gnn", "sudoku_n48_budget_W_2026-11-12")

EPOCHS = 100
W1 = (TRAINED_T, 4, "tied_no_mlp", 320, 1024, 0, EPOCHS)

# W-c's bar, computed from stage S's measured tail slope rather than typed.
TAIL_SLOPE = -0.00013                      # nll/epoch at epoch 50, stage S
ADDED_EPOCHS = EPOCHS - EP_DEFAULT         # 50
NLL_BAR = round(-TAIL_SLOPE * ADDED_EPOCHS, 4)          # 0.0065

# W-d's line, placed BETWEEN the two populations stage U separated rather than
# chosen: the worst basin seed and the best plateau seed on record in this
# family. Asserted in main() to lie strictly between them.
U_BASIN_NLL = (1.5371, 1.5306)             # stage U, k=2 s2 and k=4 s1
U_BASIN_ACC = (0.4929, 0.4948)
FAMILY_PLATEAU_BEST = 1.5845               # stage V, T=24 seed 2 -- the lowest
BASIN_NLL = round((max(U_BASIN_NLL) + FAMILY_PLATEAU_BEST) / 2, 3)   # 1.561

NEAR = 0.01                                # "within" for W-b, as stage V

BASELINE_DIRS = [SCAN_S, SCAN_U, SCAN_V, SCAN_P, SCAN_H, SCAN_I, SCAN_J, SCAN_K]


def nll_at(path, epoch):
    """Train NLL at a 1-indexed epoch, or None. W-c's within-run pair.

    history() in minarch_analysis returns only the tail, which is the right
    thing for every other stage and useless here: the whole point of W-c is a
    number from the MIDDLE of the run.
    """
    try:
        with open(os.path.join(path, "history.json")) as fh:
            h = json.load(fh)
    except (OSError, ValueError):
        return None
    if len(h) < epoch:
        return None
    return float(h[epoch - 1]["train_nll"])


def control(cells):
    """Guard 4: this whole stage is a difference against stage S's 50-epoch cell."""
    print("\n=== CONTROL -- stage S's 50-epoch cell, re-read from disk ========")
    ms = rung(cells, S1)
    if not ms:
        print("  S1 (tied_no_mlp T=16 d320 k=1, 50 ep): NOT ON DISK in the dirs")
        print("  scanned. STOP: without the control there is no difference to read.")
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
        print("  STOP: the control drifted. Fix that before reading the budget.")
    return same


def score_wa(cells):
    """W-a. Read first, as every stage -- and the adjudicator again."""
    print("\n=== W-a -- VALIDATION TOKEN ACCURACY, READ FIRST AND DECISIVE ====")
    print(f"  registered: at {EPOCHS} epochs, final val token accuracy >= {ACC_BAR:.4f}")
    print(f"  in at least 2 of 3 seeds. The bar is the control's {S1_ACC:.4f} plus the")
    print(f"  {NEED:.4f} the gate needs, IMPORTED from stage V so the two stages are")
    print(f"  judged against one number, and it sits one full control seed spread")
    print(f"  ({S1_ACC_SPREAD:.4f}) above the control. CROSS-RUN (guard 8).")
    ms = rung(cells, W1)
    if not ms:
        print(f"  {EPOCHS} ep: nothing on disk yet. W-a is what selects W-f's branch.")
        return None
    rows = sorted(acc_of(ms))
    for seed, a in rows:
        print(f"    seed {seed}  val token acc {a:.4f}  {a - S1_ACC:+.4f} vs control"
              f"  {'CLEARS' if a >= ACC_BAR else 'misses'}")
    vals = [a for _, a in rows]
    n_clear = sum(1 for a in vals if a >= ACC_BAR)
    spread = max(vals) - min(vals)
    print(f"    median {median(vals):.4f}  spread {spread:.4f}  {n_clear}/{len(vals)}"
          f" seeds at or above {ACC_BAR:.4f}")
    if n_clear >= 2:
        print(f"  W-a HOLDS: {n_clear}/3 seeds clear {ACC_BAR:.4f}. Twice the budget "
              "bought the")
        print("  accuracy the gate needs, at ZERO parameter cost. Read W-b next.")
    else:
        print(f"  W-a FALSIFIED: {n_clear}/3 seeds clear {ACC_BAR:.4f}, on the statistic")
        print("  that CANNOT censor. Read W-c before calling anything: whether this")
        print("  is a budget negative or a convergence null depends on it.")
    if spread > NEED:
        print(f"  NOTE, as in stages U and V: the seed spread {spread:.4f} exceeds the")
        print(f"  required gain {NEED:.4f}, so one seed over the bar is not evidence.")
        print("  The 2-of-3 form is what carries W-a.")
    return {"vals": vals, "median": median(vals), "n_clear": n_clear,
            "spread": spread, "by_seed": dict(rows)}


def report(cells):
    """The rung on its own, before anything is called."""
    pred = sched_count(320)
    print(f"\n=== W1  T=16, {EPOCHS} epochs -- {pred:,} trainable ==============")
    ms = rung(cells, W1)
    if not ms:
        print("  nothing on disk yet.")
        return None
    s = summarize(ms)
    n_esc, nll = escapes(ms)
    print(f"  {s['n']}/3 seeds on disk, {s['n_cross']}/{s['n']} crossed 0.5")
    print(f"  rho_50 median {rho_txt(s):>9}   (gate at or below {FREE}; 50 ep "
          f"{S1_RHO50}; untied twin J2 {STANDING[3]:.3f} at d192)")
    print(f"  mean_solve    {s['mean_solve']:>9.4f}   "
          f"[spread {s['spread_mean']:.4f}; 50 ep {S1_MEAN_SOLVE:.4f}]")
    print(f"  trainable {s['trainable']:,}   (registration predicted {pred:,}"
          + ("" if s["trainable"] == pred else "  <-- DISAGREES, quote the disk")
          + ")")
    print(f"  plateau escapes {n_esc}/{s['n']} (below {PLATEAU})   final train NLL "
          + "/".join(f"{v:.4f}" for v in nll))
    for m in sorted(ms, key=lambda r: r["seed"]):
        flag = "ok" if m["flag"] == "ok" else m["flag"].upper()
        n_ep = history(m["path"])[0]
        print(f"    seed {m['seed']}  rho_50 {m['rho50']:.3f} {flag:>9}  "
              f"mean_solve {m['mean_solve']:.4f}  {n_ep} epochs on disk")
        if n_ep != EPOCHS:
            print(f"      STOP: {n_ep} epochs, not {EPOCHS}. This is not the cell the")
            print("      registration describes and the one knob did not move.")
    if s["trainable"] != pred:                                   # guard 5
        print("  STOP: this is not the cell the registration describes.")
    if s["trainable"] != S1_TRAINABLE:                           # guard 5, the point
        print("  STOP: the count does NOT equal the control's. The epoch count was")
        print("  supposed to be free; if it is not, the tractability case is void.")
    return s


def score_wb(s):
    """W-b. The gate."""
    print("\n=== W-b -- DOES TWICE THE BUDGET CLEAR THE 0.740 GATE? ===========")
    print(f"  registered: 3 of 3 seeds cross and rho_50 < {FREE}.")
    print(f"  A CENSOR here is not a size effect: {S1_TRAINABLE:,} trainable is the")
    print(f"  SAME count as a cell that crosses 3/3 at half the budget. (It is below")
    print(f"  H1's {CENSOR_FLOOR:,} floor, so guard 7's usual argument does not apply;")
    print("  the identical count does the work instead.)  CROSS-RUN (guard 8).")
    if not s:
        print("  PENDING.")
        return None
    if s["n"] < 3:
        print(f"  INCOMPLETE ({s['n']}/3 seeds). Reported, not called.")
        return None
    if not s["rho50_readable"]:
        print("  CENSORED -- no density bin cleared 0.5 in the median seed, at a")
        print("  count that crosses 3/3 at 50 epochs. Twice the budget made the cell")
        print("  WORSE, not merely flat.")
        return {"cleared": False, "censored": True}
    d_ctrl = s["rho50"] - float(S1_RHO50)
    free = s["rho50"] <= FREE
    print(f"  rho_50 {s['rho50']:.3f} ({'CLEARS' if free else 'past'} {FREE}), "
          f"{s['n_cross']}/3 crossing")
    print(f"  against 50 ep {S1_RHO50}: {d_ctrl:+.3f} = {d_ctrl / S_RHO50:+.2f} s*   "
          f"(mean_solve {s['mean_solve']:.4f} vs {S1_MEAN_SOLVE:.4f}, "
          f"{(s['mean_solve'] - S1_MEAN_SOLVE) / S_MEAN:+.2f} s*)")
    cleared = free and s["n_cross"] == 3
    if cleared:
        print("  GATE CLEARED by one shared block at 613,818 parameters, bought with")
        print("  GPU hours rather than with structure.")
    elif abs(d_ctrl) <= NEAR:
        print(f"  WITHIN {NEAR} OF THE CONTROL: the epoch count is not what held the")
        print("  cell at 0.757 either.")
    else:
        print("  W-b FALSIFIED: twice the budget does not put rho_50 under the gate.")
    return {"cleared": cleared, "censored": False, "rho50": s["rho50"]}


def score_wc(cells):
    """W-c. The only within-run paired number this stage has."""
    print("\n=== W-c -- DID THE EXTRA 50 EPOCHS BUY ANY LOSS AT ALL? ==========")
    print(f"  registered: within each run, train NLL at epoch {EPOCHS} is at least")
    print(f"  {NLL_BAR:.4f} below epoch {EP_DEFAULT} -- stage S's measured tail slope")
    print(f"  ({TAIL_SLOPE} nll/epoch) over the {ADDED_EPOCHS} added epochs -- in >= 2 of 3 seeds.")
    print("  WITHIN-RUN: same seed, same data order, same weights to epoch 50, and")
    print("  the constant lr is what makes the two points comparable at all.")
    print("  If this FAILS the cell was converged at 50 and W is a null about")
    print("  CONVERGENCE, not about budget. Registered before the result.")
    ms = rung(cells, W1)
    if not ms:
        print("  PENDING.")
        return None
    rows = []
    for m in sorted(ms, key=lambda r: r["seed"]):
        a = nll_at(m["path"], EP_DEFAULT)
        b = nll_at(m["path"], EPOCHS)
        if a is None or b is None:
            print(f"    seed {m['seed']}  history.json does not reach both epochs")
            continue
        d = a - b
        print(f"    seed {m['seed']}  epoch {EP_DEFAULT} {a:.4f} -> epoch {EPOCHS} "
              f"{b:.4f}   fell {d:+.4f}  {'CLEARS' if d >= NLL_BAR else 'below bar'}")
        rows.append((m["seed"], a, b, d))
    if not rows:
        print("  nothing readable.")
        return None
    n_clear = sum(1 for _, _, _, d in rows if d >= NLL_BAR)
    print(f"    median fall {median([d for _, _, _, d in rows]):.4f}   "
          f"{n_clear}/{len(rows)} seeds at or above {NLL_BAR:.4f}")
    if n_clear >= 2:
        print("  W-c HOLDS: the budget bought loss. Whatever W-a says is therefore a")
        print("  statement about what that loss was WORTH, not about convergence.")
    else:
        print("  W-c FALSIFIED: the extra epochs bought no loss. The cell was already")
        print("  converged at 50 epochs, the budget was never the binding constraint,")
        print("  and no reading of W-b speaks to it. See W-f: the data pair is next.")
    return {"rows": rows, "n_clear": n_clear}


def score_wd(wa, wc, cells):
    """W-d. Loss into accuracy, with the confound stated in advance."""
    print("\n=== W-d -- DID THE LOSS BECOME ACCURACY (stage U's basin)? =======")
    print(f"  Stage U's two basin seeds reached train NLL "
          f"{'/'.join(f'{v:.4f}' for v in U_BASIN_NLL)} and val accuracy")
    print(f"  {'/'.join(f'{v:.4f}' for v in U_BASIN_ACC)} while every k=1 seed sat near "
          f"{S1_NLL_MEDIAN:.4f}/{S1_ACC:.4f}.")
    print(f"  The line {BASIN_NLL:.3f} is placed BETWEEN those populations (worst basin")
    print(f"  {max(U_BASIN_NLL):.4f}, best plateau {FAMILY_PLATEAU_BEST:.4f}), not chosen.")
    print("  REGISTERED CONFOUND: at twice the budget a low train NLL is NOT by")
    print("  itself a basin -- more steps lower it mechanically. The basin claim")
    print(f"  needs NLL < {BASIN_NLL:.3f} AND accuracy >= {ACC_BAR:.4f} in the SAME seed.")
    ms = rung(cells, W1)
    if not ms or not wa:
        print("  PENDING.")
        return None
    out = []
    for m in sorted(ms, key=lambda r: r["seed"]):
        nll = history(m["path"])[1]
        acc = wa["by_seed"].get(m["seed"])
        low = nll < BASIN_NLL
        hi = acc is not None and acc >= ACC_BAR
        verdict = ("BASIN (both)" if low and hi else
                   "loss only, no accuracy" if low else
                   "accuracy only" if hi else "neither")
        print(f"    seed {m['seed']}  train NLL {nll:.4f}  val acc "
              + ("n/a" if acc is None else f"{acc:.4f}")
              + f"   {verdict}")
        out.append({"seed": m["seed"], "nll": nll, "acc": acc,
                    "basin": low and hi, "loss_only": low and not hi})
    n_basin = sum(1 for r in out if r["basin"])
    n_loss_only = sum(1 for r in out if r["loss_only"])
    if n_basin:
        print(f"  W-d HOLDS in {n_basin}/{len(out)} seeds: the budget alone reaches the")
        print("  basin stage U only found with extra roles. See W-f -- the next arm")
        print("  asks whether roles and budget compound.")
    elif n_loss_only:
        print(f"  W-d FALSIFIED in the informative direction: {n_loss_only} seed(s) went")
        print("  below the loss line WITHOUT the accuracy. The extra budget bought")
        print("  TRAINING loss and not VALIDATION accuracy, so the remaining 0.017 is")
        print("  a GENERALIZATION gap, not a budget one. That is a different negative")
        print("  from stage V's and it points at data, not at optimization.")
    else:
        print("  W-d FALSIFIED: no seed reached the loss line. Whatever the budget")
        print("  moved, it did not move this cell toward stage U's basin.")
    if wc and wc["n_clear"] < 2:
        print("  CARRY W-c's CAVEAT: the budget bought no loss, so W-d is reading a")
        print("  knob that did not turn.")
    return {"per_seed": out, "n_basin": n_basin, "n_loss_only": n_loss_only}


def score_we(s, wc):
    """W-e. The matched pair and its asymmetry."""
    print("\n=== W-e -- 100 EPOCHS vs STAGE S's 50, THE MATCHED PAIR ==========")
    print("  Same architecture, same 613,818 parameters, same 16 applications, same")
    print("  grid, same data, same constant lr. One knob: the epoch count.")
    print(f"  50 ep (stage S, disk): rho_50 {S1_RHO50}, 3/3 crossing, mean_solve "
          f"{S1_MEAN_SOLVE:.4f}, acc {S1_ACC:.4f}.")
    print("  ASYMMETRY, registered: the 100-epoch run never evaluates rho_50 at")
    print("  epoch 50, so rho_50 and accuracy are CROSS-RUN comparisons here and")
    print("  only W-c's NLL is paired. Seeds diverge from step one.")
    if not s:
        print("  PENDING.")
        return None
    if not s["rho50_readable"]:
        print("  the 100-epoch side is CENSORED, which is worse than the 50-epoch")
        print("  control it was meant to beat.")
        return {"better": False}
    d = s["rho50"] - float(S1_RHO50)
    better = s["rho50"] < float(S1_RHO50) and s["n_cross"] >= 3
    print(f"  100 ep: rho_50 {s['rho50']:.3f}, {s['n_cross']}/3 crossing, mean_solve "
          f"{s['mean_solve']:.4f}")
    print(f"  difference: {d:+.3f} rho_50 = {d / S_RHO50:+.2f} s*, "
          f"{s['mean_solve'] - S1_MEAN_SOLVE:+.4f} mean_solve")
    if better:
        print("  W-e HOLDS: the budget improves the cell. Whether it improves it")
        print("  ENOUGH is W-b's question, not this one.")
    else:
        print("  W-e FALSIFIED: twice the budget does not improve rho_50 against the")
        print("  50-epoch control, and the difference is within or against it.")
    if wc and wc["n_clear"] >= 2 and not better:
        print("  READ THESE TOGETHER: train loss fell and rho_50 did not. That is the")
        print("  generalization reading, arrived at from two directions.")
    return {"better": better, "delta": d}


def score_wf(wa, wb, wc, wd):
    print("\n=== W-f -- WHAT RUNS NEXT, registered in advance ==================")
    if wc is not None and wc["n_clear"] < 2:
        print("  CONVERGENCE NULL. The extra epochs bought no loss, so this stage did")
        print("  not test budget -- it found that 50 epochs already converged this")
        print("  cell. Nothing about W-b or W-a adjudicates the 0.017.")
        print("  NEXT: the DATA pair, 240k and 480k instances TOGETHER in one stage,")
        print("  because n_train_instances is not in rung_key and a 480k cell cannot")
        print("  be differenced against stage S's rung without pooling into it.")
        return
    if wa is None or wb is None:
        print("  Nothing called yet: W-a selects the branch and it is pending.")
        return
    if wa["n_clear"] >= 2 and wb.get("cleared"):
        print("  REPORTED MINIMUM: one shared attention-only block, applied sixteen")
        print(f"  times, {sched_count(320):,} trainable, {EPOCHS} epochs. The gate is")
        print("  cleared by the most tractable object in the ladder, and the price was")
        print("  GPU hours rather than structure -- which is the 11-10 objective met.")
        print("  NEXT: stage T's frozen T-sweep on the 100-epoch checkpoint (is a cell")
        print("  that needed the budget an iterable operator?), then a re-read of the")
        print("  reported minimum against J2's 0.580.")
        return
    if wd and wd["n_basin"]:
        print("  The basin is reachable with budget alone, at k=1. NEXT: 100 epochs at")
        print("  k=2 -- do roles and budget compound, or is there one basin that")
        print("  either route finds?")
        return
    if wd and wd["n_loss_only"]:
        print("  GENERALIZATION, not budget: loss fell, accuracy did not follow.")
        print("  NEXT: the DATA pair, 240k and 480k in one stage (same reason as")
        print("  above). This is the last knob that is not structure.")
        return
    print("  W-a missed and the loss line was not reached. With width, cosine,")
    print("  best-val selection, role count (U), schedule length (V) and budget (W)")
    print("  all closed on evidence, the honest statement is that the tied family")
    print("  does not clear the free line at N=48 under this recipe -- a BOUNDARY,")
    print("  published as one. The standing minimum stays J2's attention-only")
    print(f"  L=16 d192 at {STANDING[3]:.3f}, and no improvement is claimed.")


def score_sufficiency():
    print("\n=== OUT-OF-SAMPLE SUFFICIENCY (the N=48 curve) ===================")
    print(f"  registered: the rung lands on the fitted accuracy-to-solve curve within")
    print(f"  the LOO bar {LOO_BAR}, as S1's seeds did (+0.0165/-0.0034/+0.0065) and as")
    print("  stage U's two basin seeds did (+0.0017/-0.0025). A cell OFF it is the")
    print("  headline: budget would then change the accuracy-to-solve RELATION rather")
    print("  than the level, which nothing in this programme has yet done. N=48 only.")
    print("  Run:")
    print("    python3 -m scripts.acc_vs_solve --scan-dir \\\n      " +
          " \\\n      ".join(SUFFICIENCY_DIRS
                             + [SCAN_K, SCAN_N, SCAN_O, SCAN_P, SCAN_S, SCAN_U,
                                SCAN_V, SCAN_W]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+",
                    default=[SCAN_W] + BASELINE_DIRS + SCAN_DIRS)
    args = ap.parse_args()

    # Guard 3: a new rung must not enter a pooled yardstick.
    for name, d in (("SCAN_W", SCAN_W), ("SCAN_V", SCAN_V), ("SCAN_U", SCAN_U),
                    ("SCAN_T", SCAN_T), ("SCAN_S", SCAN_S), ("SCAN_P", SCAN_P),
                    ("SCAN_K", SCAN_K), ("SCAN_N", SCAN_N), ("SCAN_O", SCAN_O)):
        assert d not in SCAN_DIRS, f"{name} leaked into the pooled yardstick dirs"

    # Guard 5, first, because the headline of this stage is that the count does
    # not move: epochs are as free as applications were.
    assert sched_count(320) == S1_TRAINABLE == 613818, sched_count(320)
    # Guard 6: the key separates from the control on the EPOCHS tail, and from
    # stage U's "k2" and stage T's "T24" tails by type.
    assert W1 != S1, "the budget rung keys identically to the control"
    assert W1 == S1 + (EPOCHS,), (W1, S1)
    assert EPOCHS != EP_DEFAULT, EPOCHS
    row = {"num_layers": TRAINED_T, "nhead": 4, "arch_mode": "tied_no_mlp",
           "d_model": 320, "dim_feedforward": 1024, "warmup_epochs": 0,
           "epochs": EPOCHS, "n_blocks": 1, "n_iters_eval": TRAINED_T}
    assert rung_key(row) == W1, rung_key(row)
    assert rung_key(row) != S1, "a W cell would pool into stage S's rung"
    assert rung_key(dict(row, epochs=EP_DEFAULT)) == S1, "the control key moved"
    # The three tails are distinguishable by TYPE, which is the whole reason
    # epochs kept its bare-integer form. A k=2 cell and a frozen-T re-read must
    # not land on this rung, so both are built and compared rather than assumed.
    assert rung_key(dict(row, epochs=EP_DEFAULT, n_blocks=2)) == S1 + ("k2",)
    assert rung_key(dict(row, epochs=EP_DEFAULT, n_iters_eval=24)) == S1 + ("T24",)
    for other in (S1 + ("k2",), S1 + ("T24",)):
        assert other != W1 and type(other[-1]) is not type(W1[-1]), (other, W1)
    # Guard 7, stated rather than omitted: this rung is BELOW the censoring
    # floor, so that argument does not protect it. The identical count does.
    assert sched_count(320) < CENSOR_FLOOR, sched_count(320)
    # W-a's bar is stage V's, imported. W-c's and W-d's are derived here.
    assert ACC_BAR == 0.4841, ACC_BAR
    assert abs(NEED - 0.0011) < 0.0002, NEED
    assert NLL_BAR == 0.0065, NLL_BAR
    assert max(U_BASIN_NLL) < BASIN_NLL < FAMILY_PLATEAU_BEST, BASIN_NLL
    assert BASIN_NLL < PLATEAU, (BASIN_NLL, PLATEAU)

    print("STAGE W -- one block, twice the training budget: steps or recipe?")
    print("=" * 70)
    print(f"  scanning: {' '.join(args.scan_dir)}")
    print(f"  bars (N=48, reused not refrozen): s* {S_RHO50} rho_50 / {S_MEAN} "
          f"mean_solve, gate {FREE}, plateau {PLATEAU}, LOO {LOO_BAR}")
    print("  stage M's N=96 bars are a different size and appear nowhere below.")
    print(f"  ONE KNOB: {EP_DEFAULT} -> {EPOCHS} epochs at constant 1e-3, k=1, T=16,")
    print(f"  d320, and {sched_count(320):,} trainable AT BOTH BUDGETS -- epochs are")
    print("  the second free knob in this programme, after the application count.")
    print("  h_{t+1} = F(h_t), sixteen times. Still one map, still one block.")
    print("  This is the arm stage V's V-f registered before any V cell landed,")
    print("  taken because V-a came back 0/3 at T=24 on the statistic that cannot")
    print("  censor.")

    cells = load_pooled(args.scan_dir, args.boot)
    base_ok = control(cells)

    wa = score_wa(cells)                 # accuracy first, and decisive here
    s = report(cells)
    wb = score_wb(s)
    wc = score_wc(cells)
    wd = score_wd(wa, wc, cells)
    score_we(s, wc)
    score_wf(wa, wb, wc, wd)
    score_sufficiency()

    print("\n" + "=" * 70)
    print("Read in this order: W-a (accuracy, never censors, the adjudicator) ->")
    print("W-b (the gate) -> W-c (did the budget buy any loss -- WITHIN-RUN, and it")
    print("decides whether this stage tested budget at all) -> W-d (did the loss")
    print("become accuracy, confound registered) -> W-e (the matched pair).")
    print("W-a outranks W-b by registration; W-c outranks both for a NULL, because")
    print("a knob that did not turn cannot be reported as a knob that did not work.")

    if not base_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
