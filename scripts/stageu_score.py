"""Score stage U: k DISTINCT blocks cycled T/k times -- is the remaining gap a
ROLE-COUNT limit, or is it the recipe?

REGISTERED BEFORE THE FIRST CELL LANDS. This docstring IS the registration --
committed by scripts/slurm/stageu_register_2026-11-11.sbatch, which the array
scripts/slurm/sudoku2_n48_cycled_U_2026-11-11.sbatch depends on with
--dependency=afterok. No cell can start ahead of it.

THE HYPOTHESIS, STATED FIRST

  One shared d320 block does direct deduction (stage S) but lands 0.017 past the
  free line: rho_50 0.757 (0.754/0.757/0.765), 3/3 crossing, val token accuracy
  0.4830 against the untied twin J2's 0.4940. Stage S's S4 showed the cell sits
  ON the N=48 accuracy-to-solve curve (residuals +0.0165/-0.0034/+0.0065 against
  the 0.0423 LOO bar), so the 0.177 rho_50 gap to J2 is exactly what J2's extra
  0.011 of token accuracy buys at that steepness -- and 0.017 of rho_50 is
  therefore worth about 0.001 of accuracy. Width does not supply it: accuracy
  saturated 0.4567 -> 0.4830 -> 0.4837.

  HYPOTHESIS U: the missing 0.001 is bought by ROLE COUNT, not by capacity. One
  weight set at d320 is being fit to 16 different jobs; k distinct blocks cycled
  T/k times each is fit to 16/k. If that is the binding constraint, train NLL
  and accuracy both improve with k and rho_50 crosses 0.740 at some k in {2, 4}.

  WHAT STAGE T CONTRIBUTES, AND WHY IT POINTS HERE. Stage T applied the one
  trained block T in {4..24} on FIXED weights, 3 seeds, paired instances:
    T1 FALSIFIED  the easy end does not keep solving -- worst-bin solve_rate
                  0.0000/0.0525/0.7525/0.8875/0.7300/0.0000 for T=4..24. The
                  representation COLLAPSES past the trained count in 2/3 seeds.
    level table   rho_50 CENSORED/CENSORED/0.769/0.757/0.763/0.819, crossings
                  0/0/3/3/3/2, mean_solve 0.0000/0.0159/0.2963/0.3148/0.3020/0.1873
    T0 FALSIFIED  |T=24 - T=16| = 0.062 and mean_solve PEAKS at the trained
                  count. The block is not an iterable operator that can be run
                  longer; it is stable for about as many applications as it was
                  trained for.
    T2            T* = 16: no smaller T comes within 0.01, so all 16
                  applications are used. There is no smaller object inside it.
  T1 outranks T0 by registration, so the honest reading is "one weight set tuned
  to a 16-step schedule", which is precisely the thing k > 1 relieves.

WHY THIS DOES NOT ABANDON THE TRACTABILITY BRIEF (the 11-10 objective)

  k distinct blocks cycled is STILL a fixed-point iteration of ONE map:
      G = F_{k-1} o ... o F_0      h_{t+1} = G(h_t),  applied 16/k times.
  Every rung of this ladder is an iterated map; what grows is how much there is
  to read inside G. So k is the axis along which accuracy can be bought WITHOUT
  leaving the analyzable class, and the figure of merit is the SMALLEST k that
  clears the gate -- not the parameter count, which rises with k (U3).

WHAT CHANGES: EXACTLY ONE KNOB, k

    S1  tied_no_mlp T=16 d320 k=1     613,818   rho_50 0.757, 3/3   (stage S, on disk)
    U1  tied_no_mlp T=16 d320 k=2   1,025,978   ABOVE H1's floor    this stage
    U2  tied_no_mlp T=16 d320 k=4   1,850,298   ABOVE H1's floor    this stage

  Everything else is S1's: d320, nhead 4, ff 1024 (the feedforward writes
  identically zero in EVERY block), warmup 0, lr 1e-3 constant, 240k x 50ep,
  `--shuffle-adj --fall`, the 14-point grid to 0.33 at 400 per bin, seeds 0/1/2,
  16 APPLICATIONS throughout. k=1 is stage S's cell and is RE-READ FROM DISK as
  the control, never re-run and never retyped.

  Warmup stays 0 although stage S's d448 arm diverged at warmup 0 in 2/3 seeds:
  that was a WIDTH/init phenomenon (diagnostic 51860479), and k holds width
  fixed at 320, where all three seeds trained. Changing warmup here would be a
  second knob and would make a null result unreadable.

U0  THE HEADLINE, AND THE GATE. At some k in {2, 4}: 3 of 3 seeds cross and
    median rho_50 <= 0.740. If it holds, an iterated composition of k blocks
    does direct deduction at N=48 and the role count was the binding constraint.
    THE COMPETING HYPOTHESIS PREDICTS THE OPPOSITE: both k land within 0.01 of
    k=1's 0.757 and the gate stays shut, which says the remaining 0.017 is NOT
    about one block serving 16 roles. That is a clean negative and IS the
    result -- see U5.

U1  PLATEAU AND TRAIN NLL, READ BEFORE ANY LEVEL (the P2/S1 precedent).
    Registered: final train NLL falls MONOTONICALLY across k=1 -> k=2 -> k=4
    from k=1's median 1.5864.
      falls           -> capacity PER ROLE was binding. U0 is readable as a
                         claim about role count.
      flat (all three inside 0.01) -> the floor belongs to the RECIPE, not to
                         the role count. No k moves it, and U0 must then be read
                         as "this recipe cannot train it", not as a claim about
                         k. The registered next move is U5's budget arm.
      falls but no gate -> partial. Report the slope and the gap, extrapolate no
                         k from it.

U2  ACCURACY FIRST, AS EVERY STAGE, because it never censors. Registered:
    val token accuracy rises monotonically with k from k=1's 0.4830 toward
    J2's 0.4940 (a d192 UNTIED cell -- a reference line at a different width,
    never differenced as if matched). The required gain is computed from S4's
    own steepness rather than asserted: 0.017 of rho_50 at 0.177 per 0.011 of
    accuracy is about 0.0011. IF THAT IS SMALLER THAN THE SEED SPREAD IN
    ACCURACY, accuracy cannot adjudicate on its own and the call falls to U0's
    3/3 crossing -- stated here in advance, not discovered afterwards.

U3  THE TRACTABILITY ORDERING, SAID IN ADVANCE. The 11-10 brief ranks by what
    the OBJECT is, so among cells that clear the gate the winner is the SMALLEST
    k, and parameter count is neither the score nor the tie-break. If both clear,
    the reported result is k=2. If only k=4 clears, the honest statement is that
    the iterated-map structure survives but the single-operator claim does not:
    four blocks to read, each applied four times. If k=1 is the only cell that
    ever trained cleanly, no k is called (the stage O mistake -- one arm called
    after the other was lost -- is pre-empted by saying this now).

U4  OUT-OF-SAMPLE SUFFICIENCY. Both new rungs land on scripts/acc_vs_solve.py's
    N=48 fitted accuracy-to-solve curve within the LOO bar 0.0423, as S1's three
    seeds did. A k cell OFF it is the headline over U0: it would mean cycling k
    blocks changes the accuracy-to-solve RELATION and not merely the level. That
    curve is an N=48 object and is never applied to N=96 (finding 55).

U5  WHAT RUNS NEXT, REGISTERED NOW RATHER THAN CHOSEN AFTERWARDS.
      U0 holds at k=2  -> the T-sweep on the k=2 checkpoint, T in multiples of 2:
                          does G = F1 o F0 iterate when the single block did not
                          (stage T's T0/T1 both failed at k=1)? If it fails
                          there too, "tuned to a schedule" is a property of the
                          TRAINED SOLUTION at every k, which is a finding about
                          this recipe and not about sharing.
      U0 holds only at k=4 -> same sweep at k=4, plus the k=2 cell reported as
                          the negative it is. No interpolation to k=3: 16 is not
                          divisible by 3 and the constructor refuses it.
      U0 falsified, plateau escaped -> CLEAN NEGATIVE: the 0.017 is not about
                          role count. The registered follow-up is the BUDGET arm
                          -- 100 epochs at constant 1e-3, one knob, k=1 -- since
                          stage S's tail slope was about -0.00013 nll/epoch and
                          0.001 of accuracy is plausibly inside another 50
                          epochs. Explicitly NOT more width (saturated) and NOT
                          larger k (this stage's own negative).
      U1 flat          -> the same budget arm, and k is recorded as not the
                          binding axis.

GUARDS, each one a mistake this programme has already made once
  1. N=48 only (finding 55). No N=96 number appears here.
  2. rho_50 CENSORS. A censored seed is a result, not a missing number, and the
     bound-aware median is what gets quoted.
  3. SCAN_U stays OUT of SCAN_DIRS, as E/G/H/I/J/K/N/O/P/S all do, so a new rung
     cannot enter a pooled yardstick by accident.
  4. The k=1 control is re-read from disk and checked against stage S's frozen
     literals. The whole stage is a difference against it; a drifted control
     invalidates that silently, so this exits non-zero instead.
  5. Every count must match (630d + 58) + (4d^2 + 8d)k EXACTLY and must differ
     from the untied stack's count at the same width. If a k cell ever counts
     what k=1 counts, the extra blocks were not built and the comparison is void.
  6. Both new rungs are ABOVE H1's 879,418 censoring floor, so a censor at
     either is about the architecture and not about size (stage P's confound,
     named in advance here as stage S named it).
  7. The ladder's far end is NOT on disk: untied no_mlp at d320 was never
     trained, so nothing here extrapolates toward k=16, and J2 (d192, untied) is
     a reference line at a different width, never a matched control.
"""

import argparse
import os
import sys
from statistics import median

from scripts.minarch_analysis import SCAN_DIRS, SCAN_H, SCAN_I, SCAN_J
from scripts.stagec_score import (PLATEAU, S_MEAN, S_RHO50, escapes, rho_txt,
                                  rung, summarize)
from scripts.stagej_score import LOO_BAR, SUFFICIENCY_DIRS
from scripts.stagek_score import FREE, SCAN_K, STANDING
from scripts.stagen_score import SCAN_N, acc_of, load_pooled
from scripts.stageo_score import SCAN_O
from scripts.stagep_score import CENSOR_FLOOR, SCAN_P
from scripts.stages_score import SCAN_S

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_U = os.path.join(_SCR, "spin_gnn", "sudoku_n48_cycled_U_2026-11-11")

# Rung keys are (L, nhead, arch, d_model, ff, warmup) + ("kK",) when k != 1.
# L is the number of APPLICATIONS and is 16 on every rung of this ladder; the
# "kK" tail is a STRING so it cannot collide with an eval-T tail (rung_key).
S1 = (16, 4, "tied_no_mlp", 320, 1024, 0)                # stage S, the control
U1 = (16, 4, "tied_no_mlp", 320, 1024, 0, "k2")          # this stage
U2 = (16, 4, "tied_no_mlp", 320, 1024, 0, "k4")          # this stage


def cycled_count(d, k):
    """Embeddings and readout once, the block k times -- k blocks, 16/k
    applications each, the per-block term paid per BLOCK and not per use."""
    return (630 * d + 58) + (4 * d * d + 8 * d) * k


# The ladder, k ascending. (key, label, k, predicted trainable)
LADDER = [
    (S1, "S1  k=1, 16 applications", 1, cycled_count(320, 1)),
    (U1, "U1  k=2, 8 applications each", 2, cycled_count(320, 2)),
    (U2, "U2  k=4, 4 applications each", 4, cycled_count(320, 4)),
]

# Stage S's own result (scorer 51904146 / the partial read 51853969), frozen
# here and re-read from disk by control().
S1_TRAINABLE = 613818
S1_RHO50 = "0.757"
S1_RHO50_SEEDS = (0.754, 0.757, 0.765)
S1_ESCAPES = 3
S1_ACC = 0.4830
S1_NLL_MEDIAN = 1.5864
S1_MEAN_SOLVE = 0.3148

# S4's steepness, from stage S: J2's extra accuracy against the rho_50 gap it
# bought. Used ONLY to state in advance how much accuracy U0 needs.
J2_ACC = 0.4940                  # d192 UNTIED -- a reference line, not a control
ACC_GAP = J2_ACC - S1_ACC        # 0.0110
RHO_GAP = 0.177                  # S1 0.757 - J2 0.580, the same pair
FLAT_BAND = 0.01                 # U1: "all three inside 0.01" = no movement
BUDGET_SLOPE = -0.00013          # stage S's tail slope, nll/epoch (U5)

BASELINE_DIRS = [SCAN_S, SCAN_P, SCAN_H, SCAN_I, SCAN_J, SCAN_K]


def control(cells):
    """Guard 4: this whole stage is a difference against stage S's k=1 cell."""
    print("\n=== CONTROL -- stage S's k=1 cell, re-read from disk =============")
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
    print("      " + ("matches the frozen literals" if same else "<-- DISAGREES"))
    if not same:
        print("  STOP: the control drifted. Fix that before reading any k.")
    return same


def score_accuracy_first(cells):
    """U2. Read before rho_50 at every stage, because accuracy never censors."""
    print("\n=== U2 -- VALIDATION TOKEN ACCURACY, READ FIRST ==================")
    need = ACC_GAP * (FREE - float(S1_RHO50)) / -RHO_GAP if RHO_GAP else 0.0
    print(f"  registered: accuracy rises monotonically with k from k=1's {S1_ACC:.4f}")
    print(f"  toward J2's {J2_ACC:.4f} (d192 UNTIED -- a reference line at another")
    print("  width, never differenced as if matched).")
    print(f"  what U0 needs, from S4's own steepness: {RHO_GAP:.3f} of rho_50 per")
    print(f"  {ACC_GAP:.4f} of accuracy, so the {float(S1_RHO50) - FREE:+.3f} to the free")
    print(f"  line is worth about {need:.4f} of token accuracy.")
    meds, spreads = [], []
    for key, label, k, _ in LADDER:
        ms = rung(cells, key)
        if not ms:
            print(f"  {label}: nothing on disk yet")
            meds.append(None)
            spreads.append(None)
            continue
        rows = sorted(acc_of(ms))
        for seed, a in rows:
            print(f"    {label}  seed {seed}  val token acc {a:.4f}")
        vals = [a for _, a in rows]
        m, sp = median(vals), max(vals) - min(vals)
        print(f"    {label}  median {m:.4f}  spread {sp:.4f}")
        meds.append(m)
        spreads.append(sp)
    got = [m for m in meds if m is not None]
    if len(got) == len(LADDER):
        mono = all(b >= a for a, b in zip(got, got[1:]))
        print("  accuracy by k: " + " -> ".join(f"{m:.4f}" for m in got))
        print("  MONOTONE in k" if mono else
              "  NOT monotone in k -- k is not simply buying accuracy")
        worst = max(sp for sp in spreads if sp is not None)
        if need < worst:
            print(f"  AS REGISTERED: the required gain {need:.4f} is INSIDE the worst")
            print(f"  seed spread {worst:.4f}, so accuracy cannot adjudicate on its own.")
            print("  The call falls to U0's 3/3 crossing, which is what was registered.")
    return meds


def report(cells):
    """Each new rung on its own, before anything is called."""
    out = {}
    for key, label, k, pred in LADDER[1:]:
        print(f"\n=== {label} -- {pred:,} trainable =================")
        ms = rung(cells, key)
        if not ms:
            print("  nothing on disk yet.")
            out[key] = None
            continue
        s = summarize(ms)
        n_esc, nll = escapes(ms)
        print(f"  {s['n']}/3 seeds on disk, {s['n_cross']}/{s['n']} crossed 0.5")
        print(f"  rho_50 median {rho_txt(s):>9}   (gate at or below {FREE}; k=1 "
              f"{S1_RHO50}; untied twin J2 {STANDING[3]:.3f} at d192)")
        print(f"  mean_solve    {s['mean_solve']:>9.4f}   "
              f"[spread {s['spread_mean']:.4f}; k=1 {S1_MEAN_SOLVE:.4f}]")
        print(f"  trainable {s['trainable']:,}   (registration predicted {pred:,}"
              + ("" if s["trainable"] == pred else "  <-- DISAGREES, quote the disk")
              + ")")
        over = "ABOVE" if s["trainable"] > CENSOR_FLOOR else "below"
        print(f"  {over} H1's censoring floor {CENSOR_FLOOR:,} -- "
              + ("a censor here is about the ARCHITECTURE, not about size (guard 6)"
                 if over == "ABOVE" else "a censor here cannot separate k from size"))
        print(f"  plateau escapes {n_esc}/{s['n']}   final train NLL "
              + "/".join(f"{v:.4f}" for v in nll))
        for m in sorted(ms, key=lambda r: r["seed"]):
            flag = "ok" if m["flag"] == "ok" else m["flag"].upper()
            print(f"    seed {m['seed']}  rho_50 {m['rho50']:.3f} {flag:>9}  "
                  f"mean_solve {m['mean_solve']:.4f}")
        if s["trainable"] != pred:
            print("  STOP: this is not the cell the registration describes.")
        if s["trainable"] == S1_TRAINABLE:                      # guard 5
            print("  STOP: the count equals the k=1 cell's. The extra blocks were")
            print("  not built and the comparison is void.")
        out[key] = s
    return out


def score_u1(cells):
    """The plateau and the train NLL, before any level."""
    print("\n=== U1 -- IS CAPACITY PER ROLE BINDING? READ FIRST ===============")
    print(f"  registered: final train NLL falls MONOTONICALLY across k=1 -> k=2 ->")
    print(f"  k=4 from k=1's median {S1_NLL_MEDIAN:.4f}. Flat means the floor belongs to")
    print("  the RECIPE and no k moves it. This outranks U0, as S1 outranked S0.")
    nlls, escs = [], []
    for key, label, k, _ in LADDER:
        ms = rung(cells, key)
        if not ms:
            print(f"  {label}: PENDING")
            nlls.append(None)
            escs.append(None)
            continue
        n_esc, nll = escapes(ms)
        print(f"  {label}  escaped {n_esc}/{len(ms)}  median train NLL "
              f"{median(nll):.4f}   " + "/".join(f"{v:.4f}" for v in nll))
        nlls.append(median(nll))
        escs.append(n_esc)
    if any(v is None for v in nlls):
        print("  INCOMPLETE: the trend needs all three k. Not called.")
        return None
    mono = all(b <= a for a, b in zip(nlls, nlls[1:]))
    span = max(nlls) - min(nlls)
    print("  train NLL by k: " + " -> ".join(f"{v:.4f}" for v in nlls)
          + f"   (span {span:.4f}, flat band {FLAT_BAND})")
    if mono and span > FLAT_BAND:
        print("  U1 READS: CAPACITY PER ROLE WAS BINDING. The floor moves when the")
        print("  same weights serve fewer applications, so U0 below is readable as")
        print("  a claim about ROLE COUNT.")
    elif span <= FLAT_BAND:
        print(f"  U1 READS: THE RECIPE, NOT THE ROLES. The floor sits in the same")
        print(f"  place at every k (span {span:.4f} <= {FLAT_BAND}). One weight set serving")
        print("  16 roles was NOT what held the cell at 1.586, and U0 must then be")
        print("  read as a statement about this training recipe. The registered")
        print(f"  next move is U5's budget arm (tail slope was {BUDGET_SLOPE:+.5f}/epoch).")
    else:
        print(f"  U1 READS: PARTIAL. NOT monotone, span {span:.4f}. Report the levels")
        print("  and extrapolate no k from them.")
    return {"nlls": nlls, "escapes": escs, "monotone": mono, "span": span}


def score_u0(seen, u1):
    """The gate. The whole point of the day."""
    print("\n=== U0 -- DOES CYCLING k BLOCKS CLEAR THE 0.740 GATE? ============")
    print(f"  registered: at some k in (2, 4), 3 of 3 seeds cross and median rho_50")
    print(f"  <= {FREE}. The competing hypothesis -- the 0.017 is not about role")
    print(f"  count -- predicts both k within {FLAT_BAND} of k=1's {S1_RHO50}.")
    optlim = u1 is not None and u1["span"] <= FLAT_BAND
    cleared = []
    for key, label, k, _ in LADDER[1:]:
        s = seen.get(key)
        if not s:
            print(f"  {label}: PENDING")
            continue
        if s["n"] < 3:
            print(f"  {label}: INCOMPLETE ({s['n']}/3 seeds). Reported, not called.")
            continue
        if not s["rho50_readable"]:
            print(f"  {label}: CENSORED at {s['trainable']:,} trainable, which is ABOVE")
            print(f"    H1's {CENSOR_FLOOR:,} -- so this censor is NOT explained by size.")
            if optlim:
                print("    BUT U1 says the floor never moved, so read it as 'not trained'")
                print("    rather than as a claim about k.")
            continue
        d_ctrl = s["rho50"] - float(S1_RHO50)
        free = s["rho50"] <= FREE
        print(f"  {label}: rho_50 {s['rho50']:.3f} ({'CLEARS' if free else 'past'} {FREE}), "
              f"{s['n_cross']}/3 crossing")
        print(f"    against k=1 {S1_RHO50}: {d_ctrl:+.3f} = {d_ctrl / S_RHO50:+.2f} s*   "
              f"(mean_solve {s['mean_solve']:.4f} vs {S1_MEAN_SOLVE:.4f}, "
              f"{(s['mean_solve'] - S1_MEAN_SOLVE) / S_MEAN:+.2f} s*)")
        if free and s["n_cross"] == 3:
            cleared.append((k, s))
            print("    GATE CLEARED at this k.")
        elif abs(d_ctrl) <= FLAT_BAND:
            print(f"    WITHIN {FLAT_BAND} OF THE CONTROL: this is the competing")
            print("    hypothesis' prediction, not U0's.")
    if not cleared:
        print("  U0 FALSIFIED so far: no k in (2, 4) clears the gate with 3/3.")
        if u1 is not None and u1["escapes"][-1] >= 2 and not optlim:
            print("  With the plateau escaped, this is the CLEAN NEGATIVE registered")
            print("  under U5: the cells trained, they hold more weights per role, and")
            print("  rho_50 did not cross. The remaining 0.017 is NOT a role-count")
            print("  limit. That is a real constraint on any theory of this model and")
            print("  it is the result; the budget arm is next, not more k.")
    return cleared


def score_u3(cleared, seen):
    print("\n=== U3 -- THE TRACTABILITY ORDERING, said in advance ==============")
    print("  The 11-10 brief ranks by what the OBJECT is: the smallest k that")
    print("  clears the gate wins, and parameter count is neither the score nor")
    print("  the tie-break (k=4 holds 3.0x k=1's weights and is still an iterated")
    print("  map -- G = F_3 o F_2 o F_1 o F_0 applied four times).")
    have = [lbl for (k, lbl, _, _) in LADDER[1:] if seen.get(k)]
    if len(have) < 2:
        print(f"  {len(have)} of 2 new arms on disk. Nothing is called until both")
        print("  land (the stage O mistake, pre-empted).")
        return
    if not cleared:
        print("  No k clears, so there is no ordering to report -- see U0 and U5.")
        return
    k_min = min(k for k, _ in cleared)
    print(f"  REPORTED MINIMUM: k={k_min}, {cycled_count(320, k_min):,} trainable, "
          f"{16 // k_min} applications of each of {k_min} block(s).")
    if k_min == 2:
        print("  Two blocks to read instead of one. The single-operator claim does")
        print("  not survive; the ITERATED-MAP claim does, which is the part the")
        print("  tractability brief actually rests on.")
    else:
        print("  Four blocks to read, each applied four times -- a quarter of the")
        print("  way to the untied stack. Say that plainly rather than calling it")
        print("  'one operator'.")


def score_u4():
    print("\n=== U4 -- OUT-OF-SAMPLE SUFFICIENCY, reported ahead of a failure ==")
    print(f"  registered: both new rungs land on the N=48 accuracy-to-solve curve")
    print(f"  within the LOO bar {LOO_BAR}, as S1's three seeds did (+0.0165 /")
    print("  -0.0034 / +0.0065). A cell OFF it is the headline over U0: cycling k")
    print("  blocks would then change the accuracy-to-solve RELATION, not the level.")
    print("  Run:")
    print("    python3 -m scripts.acc_vs_solve --scan-dir \\\n      " +
          " \\\n      ".join(SUFFICIENCY_DIRS
                             + [SCAN_K, SCAN_N, SCAN_O, SCAN_P, SCAN_S, SCAN_U]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+",
                    default=[SCAN_U] + BASELINE_DIRS + SCAN_DIRS)
    args = ap.parse_args()

    # Guard 3: a new rung must not enter a pooled yardstick.
    for name, d in (("SCAN_U", SCAN_U), ("SCAN_S", SCAN_S), ("SCAN_P", SCAN_P),
                    ("SCAN_O", SCAN_O), ("SCAN_K", SCAN_K), ("SCAN_N", SCAN_N)):
        assert d not in SCAN_DIRS, f"{name} leaked into the pooled yardstick dirs"
    # Guard 5, as assertions on the registration's own arithmetic.
    assert cycled_count(320, 1) == 613818, cycled_count(320, 1)
    assert cycled_count(320, 2) == 1025978, cycled_count(320, 2)
    assert cycled_count(320, 4) == 1850298, cycled_count(320, 4)
    # The far end of the same ladder, for scale only: k=16 IS the untied stack
    # at this width, 6,796,218 trainable. It is not on disk and guard 7 forbids
    # extrapolating to it; it is asserted here so the arithmetic that makes k=2
    # and k=4 "intermediate" is written down rather than assumed.
    assert cycled_count(320, 16) == 6796218, cycled_count(320, 16)
    for k in (2, 4):
        assert 16 % k == 0, ("k must divide the 16 applications", k)
        assert cycled_count(320, k) != cycled_count(320, 1)
        assert cycled_count(320, k) > CENSOR_FLOOR, (
            "guard 6: a rung below the censoring floor cannot carry a censor", k)
    # Guard 7: the far end is not on disk, so nothing extrapolates to it.
    assert 16 % 3, "16 is not divisible by 3; no k=3 rung exists to interpolate"

    print("STAGE U -- k distinct blocks cycled T/k times: role count or recipe?")
    print("=" * 70)
    print(f"  scanning: {' '.join(args.scan_dir)}")
    print(f"  bars (N=48, reused not refrozen): s* {S_RHO50} rho_50 / {S_MEAN} "
          f"mean_solve, gate {FREE}, plateau {PLATEAU}, LOO {LOO_BAR}")
    print("  stage M's N=96 bars are a different size and appear nowhere below.")
    print(f"  the ladder at FIXED width d320 and FIXED 16 applications: "
          f"{cycled_count(320, 1):,} -> {cycled_count(320, 2):,} -> "
          f"{cycled_count(320, 4):,} trainable")
    print(f"  both new rungs are ABOVE H1's censoring floor {CENSOR_FLOOR:,}, so a "
          "censor at either is about the architecture")
    print("  G = F_{k-1} o ... o F_0 applied 16/k times: EVERY rung is still one")
    print("  map iterated, which is what the tractability brief asks for.")

    cells = load_pooled(args.scan_dir, args.boot)
    base_ok = control(cells)

    score_accuracy_first(cells)
    seen = report(cells)
    u1 = score_u1(cells)                 # before any level, as registered
    cleared = score_u0(seen, u1)
    score_u3(cleared, seen)
    score_u4()

    print("\n" + "=" * 70)
    print("Read in this order: U2 (accuracy, never censors) -> U1 (does the floor")
    print("move with k) -> U0 (does any k clear the gate) -> U3 (which k may be")
    print("called the result) -> U4 (the curve).")
    print("U1 outranks U0 by registration: a cell that never left the plateau says")
    print("nothing about role count, only about training it. U5's follow-ups are in")
    print("the docstring and are scored by the next stage, not by this file.")

    if not base_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
