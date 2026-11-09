"""Score stage J (51280737): at a matched parameter count, does SHAPE matter?

    python3 -m scripts.stagej_score [--boot 2000]

THE QUESTION, AND IT IS A CONTROLLED CONTRAST, NOT ANOTHER RUNG. Depth bought
the MLP back at d256 (rho_50 0.763 at L=6 -> 0.605 at L=12) and width takes it
away at L=12 (0.605 at d256 -> 0.754 at d192). Both axes move the same metric
in the same direction and the grid has never held one FIXED against the other.
Stage J runs two rungs at the SAME parameter count and different shapes:

    rung   shape                 trainable (header arithmetic)   vs E2   vs hub
    J1     no_mlp L=12 d224      2,571,130                       1.30x   1.91x
    J2     no_mlp L=16 d192      2,504,890                       1.33x   1.96x

2,571,130 against 2,504,890 is 2.6% apart, as close to a matched budget as
integer widths and depths allow, and it turns "is depth or width the currency"
from two ladders that were never priced against each other into one contrast.

THIS FILE IS THE REGISTRATION, TRANSCRIBED from the sbatch header
(scripts/slurm/sudoku2_n48_minarch_J_2026-11-05.sbatch), and it was written
while 51280737 was still running its first wave. Every literal below is either
copied from that header or measured off cells that were already final and
scored when stage J was submitted. Nothing here is chosen after seeing a stage
J number; git log is the timestamp.

THE WIDTH LADDER THIS JOB SITS ON, all attention-only at L=12, all final:

    H1  d128    879,418   CENSORED   mean 0.0411  spread 0.2729  1/3  esc 1/3
    I1  d160  1,345,018   0.773      mean 0.3000  spread 0.0164  3/3  esc 3/3
    H2  d192  1,908,922   0.754      mean 0.3220  spread 0.1450  3/3  esc 3/3
    ---------------------------------------- hub + s* = 0.740 -------------
    E2  d256  3,331,642   0.605      mean 0.5214  spread 0.0480  3/3  esc 3/3

160 -> 192 moves rho_50 by 0.019 (0.13 s*, NOT separable); 192 -> 256 moves it
by 0.149 (1.00 s*, just separable). Every separable unit of movement on the
width axis lives in [192, 256], which is why J1 bisects that segment and why
nothing below d192 is worth another cell.

REGISTERED, and reported in this order:

  J0  THE HEADLINE. |rho_50(J1) - rho_50(J2)| <= s* = 0.1484, i.e. NOT
      SEPARABLE, because the accuracy-to-solve fit says model quality over this
      grid is ONE DIMENSIONAL. If quality is one-dimensional and the two rungs
      cost the same, they land in the same place and SHAPE IS NOT A FREE
      PARAMETER -- only size is.
      FALSIFIER, AND IT IS THE RESULT WORTH HAVING: a separable difference in
      either direction, naming the winner -- (DEPTH) if rho_50(J2) is lower by
      more than s*, (WIDTH) if rho_50(J1) is lower by more than s*. That is the
      grid's first evidence that arrangement matters independently of count,
      and it would make every "smallest architecture" claim conditional.
      TIE-BREAK, fixed in advance: exactly s*, or anything inside it, is NOT
      SEPARABLE. The falsifier is the claim that needs the evidence.

  J1  no_mlp L=12 d224, the bisector of the only separable segment, and a
      number rather than a direction because both ends are measured.
      Interpolating H2 (0.754) and E2 (0.605): linear in d -> 0.680, linear in
      trainable -> 0.685, linear in 1/d -> 0.669.
      PREDICTION: rho_50 median in [0.665, 0.690], on 3 of 3 crossing seeds and
      3 of 3 plateau escapes (both neighbours escape 3/3).
      FALSIFIER: outside that interval. Above 0.740 in particular would put the
      whole 0.149 drop inside [224, 256] -- a factor of 1.14 in width -- which
      is a threshold, not a slope, and the brief says the breaking point is the
      result.

  J2  no_mlp L=16 d192, A DEPTH NEVER RUN IN THIS GRID.
      PREDICTION: rho_50 below H2's 0.754 (more depth at fixed width has
      lowered rho_50 every time on this axis: d256 no_mlp goes 0.763 at L=6,
      0.749 at L=8, 0.605 at L=12), and by J0 within s* of J1.
      TRAINABILITY: >= 2 of 3 escapes. Every d192 cell escaped 3 of 3 and depth
      has monotonically improved escapes on the no_mlp axis, so for once both
      arguments point the same way -- unlike stage H, where the monotone-depth
      expectation and escape_timing.py's Q3 disagreed and Q3 came closer.
      NOT CLAIMED: nothing in this grid sits above L=12, so J2's depth is an
      extrapolation of the ladder, not a point on it.

  J3  THE STANDING-MINIMUM RULE, unchanged from H3 and I3 and NOT softened
      because these are the last candidates. A rung replaces the standing
      minimum ONLY IF 3 of 3 seeds cross AND median rho_50 <= 0.740. Two of
      three is not enough, and finding 51 is why. If J1 lands as predicted it
      FIRES, moving the minimum from E2 (3,331,642 at 0.605) to 2,571,130.
      STATED HONESTLY IN ADVANCE: rho_50 then RISES from 0.605 to about 0.68,
      which is +0.075 or 0.51 s*, BELOW the separability bar. The claim is
      "1.30x smaller at no separable cost in rho_50", never "better".

  J4  OUT-OF-SAMPLE SUFFICIENCY, free, no extra cells. At registration the
      depth-ladder fit held on 19 of 19 well-determined held-out rungs over 41
      cells, worst discriminating miss 0.0222 against a leave-one-out bar of
      0.0423. PREDICTION: both stage J rungs land on the curve within 0.0423.
      FALSIFIER: either falls off. This is the same statistic J0 leans on, so
      if J0's falsifier fires, look here first -- a real shape effect should
      show up as a rung sitting off the curve.

  W   WATCH ITEMS. Spread is quoted for every rung always (findings 41/42/47)
      and J0's comparison is median-vs-median. Every cell trains and evaluates
      with --shuffle-adj.

SCORING RULE, fixed in advance: every comparison is median-vs-median against
the frozen bar s* = 0.1484 / 0.2918, NOT recomputed. Smaller than s* is "not
separable at this budget", never a result.
"""
import argparse
from collections import defaultdict

from scripts.minarch_analysis import (SCAN_DIRS, SCAN_E, SCAN_G, SCAN_H, SCAN_I,
                                      SCAN_J, cell_key, cell_metrics, load_all)
from scripts.stagec_score import (S_MEAN, S_RHO50, escapes, rho_txt, rung,
                                  summarize, verdict)

HUB_RHO50 = 0.592        # full L=6 d256, the reference architecture
HUB_TRAINABLE = 4899898
FREE = 0.740             # hub + s*, and the standing-minimum rule's ceiling
BAND_LO, BAND_HI = 0.72, 0.80
LOO_BAR = 0.0423         # acc_vs_solve's leave-one-out residual bar
PLATEAU = 1.64           # finding 44

J1 = (12, 4, "no_mlp", 224, 1024, 0)
J2 = (16, 4, "no_mlp", 192, 1024, 0)
H1 = (12, 4, "no_mlp", 128, 1024, 0)
I1 = (12, 4, "no_mlp", 160, 1024, 0)
H2 = (12, 4, "no_mlp", 192, 1024, 0)
E2 = (12, 4, "no_mlp")                      # full-width partner, same depth

PREDICTED_TRAINABLE = {"J1": 2571130, "J2": 2504890}
# The matched-budget assertion. If the two rungs are further apart than this on
# disk, J0 is not a matched contrast and must not be read as one.
MATCH_TOL = 0.05
MATCH_AT_REGISTRATION = 0.026

PRED_J1_LO, PRED_J1_HI = 0.665, 0.690
PRED_J1_INTERP = {"linear in d": 0.680, "linear in trainable": 0.685,
                  "linear in 1/d": 0.669}
PRED_J1_ESCAPES, PRED_J1_CROSS = 3, 3
PRED_J2_BELOW = 0.754                       # H2, the same width at L=12
PRED_J2_ESCAPES = 2
# The depth evidence J2's direction rests on: no_mlp at d256, rho_50 by depth.
DEPTH_EVIDENCE = {6: 0.763, 8: 0.749, 12: 0.605}

# Frozen off disk before stage J was submitted, all final and all scored:
#   (n, n_cross, rho_50 text, mean_solve, spread, trainable, escapes)
PARTNER = {
    "H1": (3, 1, "CENSORED", 0.0411, 0.2729, 879418, 1),
    "I1": (3, 3, "0.773", 0.3000, 0.0164, 1345018, 3),
    "H2": (3, 3, "0.754", 0.3220, 0.1450, 1908922, 3),
    "E2": (3, 3, "0.605", 0.5214, 0.0480, 3331642, 3),
}
PARTNER_KEY = {"H1": H1, "I1": I1, "H2": H2, "E2": E2}

# The literal the sbatch header registered. The H+I write-up (commit 4c9dc7d)
# refit the same curve over all eight scan dirs an hour later and got 21 of 21
# well-determined held-out rungs over 46 cells, 20 discriminating, SAME worst
# miss 0.0222 against the same bar. The registered literal is kept as written;
# the refit is noted so the two numbers are not read as a disagreement.
SUFFICIENCY_AT_REGISTRATION = dict(rungs=19, cells=41, worst=0.0222)
SUFFICIENCY_AT_DELIVERY = dict(rungs=21, cells=46, disc=20, worst=0.0222)
# Stage J's own refit, over the nine dirs below -- the number of record for J4.
SUFFICIENCY_AT_STAGEJ = dict(rungs=22, cells=47, disc=21, worst=0.0222)

# The sufficiency refit reads ONE MORE DIR than the paired scoring above. That
# is deliberate in both directions. SCAN_G is 100 epochs where everything else
# is 50, so it stays out of --scan-dir: nothing at a different budget belongs in
# a pooled bar or a paired median. But acc_vs_solve fits token accuracy against
# solve rate ACROSS rungs, and a rung at another budget is a held-out test of
# that curve rather than a contaminant -- the H+I write-up's "eight scan dirs"
# and stage J's 22/47 both include it. Printing --scan-dir here instead would
# reproduce 20 of 20 over 42 and quietly contradict the deck.
SUFFICIENCY_DIRS = SCAN_DIRS + [SCAN_E, SCAN_G, SCAN_H, SCAN_I, SCAN_J]

STANDING = ("E2", "attention-only L=12 d256", 3331642, 0.605)


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
          f"{PARTNER['E2'][2]}, H2 {PARTNER['H2'][2]}, free at or below {FREE})")
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


def score_match(j1, j2):
    """The assertion J0 depends on: are the two rungs actually the same size?"""
    t1, t2 = j1["s"]["trainable"], j2["s"]["trainable"]
    gap = abs(t1 - t2) / ((t1 + t2) / 2)
    print("\nMATCHED-BUDGET ASSERTION (read off config.json, not the header):")
    print(f"  J1 {t1:,} vs J2 {t2:,} -- {gap * 100:.1f}% apart "
          f"(registered {MATCH_AT_REGISTRATION * 100:.1f}%, tolerance "
          f"{MATCH_TOL * 100:.0f}%)")
    if gap <= MATCH_TOL:
        print("  MATCHED. J0 below is a controlled contrast.")
        return True
    verdict(f"""
NOT MATCHED. The two rungs are {gap * 100:.1f}% apart in trainable parameters, past the
{MATCH_TOL * 100:.0f}% tolerance fixed in advance, so J0 is VOID: any difference between them
could be size rather than shape. Report the two rungs separately against their
own one-knob neighbours (J1 against H2 and E2, J2 against H2) and do not call
anything below a shape effect.""")
    return False


def score_shape(j1, j2, matched):
    """J0, the headline: shape at a fixed price."""
    s1, s2 = j1["s"], j2["s"]
    print("\nJ0 -- THE HEADLINE: is SHAPE a free parameter at a matched size?")
    print(f"  registered: |rho_50(J1) - rho_50(J2)| <= s* = {S_RHO50} "
          f"(NOT separable)")
    if not (s1["rho50_readable"] and s2["rho50_readable"]):
        which = " and ".join(t for t, s in (("J1", s1), ("J2", s2))
                             if not s["rho50_readable"])
        print(f"  {which} CENSORED, so the median-vs-median contrast is "
              f"UNREADABLE, not passed.")
        print(f"  mean_solve (never censors): J1 {s1['mean_solve']:.4f} vs "
              f"J2 {s2['mean_solve']:.4f}, |diff| "
              f"{abs(s1['mean_solve'] - s2['mean_solve']):.4f} against "
              f"s*_mean {S_MEAN}.")
        return "unreadable"
    d = s1["rho50"] - s2["rho50"]
    print(f"  observed: J1 (L=12 d224) {s1['rho50']:.3f} vs J2 (L=16 d192) "
          f"{s2['rho50']:.3f}")
    print(f"  difference {d:+.3f} = {abs(d) / S_RHO50:.2f} s*  "
          f"[spreads {s1['spread_mean']:.4f} / {s2['spread_mean']:.4f}]")
    if not matched:
        verdict(f"""
J0 IS VOID AND NO SHAPE VERDICT IS GIVEN. The budgets did not match (see above),
so the {abs(d):.3f} = {abs(d) / S_RHO50:.2f} s* between the two rungs confounds shape with size and
cannot be read either way. The difference is printed above so it is on the
record, not so it can be quoted.""")
        return "void"
    if abs(d) <= S_RHO50:
        verdict(f"""
SHAPE IS NOT A FREE PARAMETER: ONLY SIZE IS. At a matched budget the deep-narrow
and shallow-wide rungs land {abs(d):.3f} apart, {abs(d) / S_RHO50:.2f} s*, inside the frozen bar, so this
grid cannot tell them apart. That is the prediction holding, and it says the
accuracy-to-solve result generalises past the depth ladder it was fit on: model
quality over this grid is ONE DIMENSIONAL, and both ladders are reading the same
axis at different prices. Every "smallest architecture" claim in this programme
stays unconditional on shape.""")
        return "not_separable"
    winner = "DEPTH" if d > 0 else "WIDTH"
    lower = ("J2, L=16 d192" if d > 0 else "J1, L=12 d224")
    verdict(f"""
({winner}) SHAPE MATTERS AT A FIXED PRICE, and this is the grid's first evidence of
it. {lower} is lower by {abs(d):.3f} = {abs(d) / S_RHO50:.2f} s*, past the frozen bar, at a parameter
count the two rungs share to within {MATCH_AT_REGISTRATION * 100:.0f}%. Arrangement therefore does
something that count does not, and every "smallest architecture" claim in this
programme becomes CONDITIONAL ON SHAPE: the minimum must be quoted as the
smallest rung of a given shape, not the smallest rung.
Look at J4 next. The sufficiency fit is what predicted no difference here, so a
real shape effect should also show up as one of these rungs sitting off the
accuracy-to-solve curve. If both rungs are ON the curve while their rho_50s are
separable, the two results are in tension and neither should be quoted alone.""")
    return winner.lower()


def score_j1(j1):
    """The registered number."""
    s, n_esc = j1["s"], j1["n_esc"]
    print(f"\nJ1 -- THE REGISTERED NUMBER: rho_50 median in "
          f"[{PRED_J1_LO:.3f}, {PRED_J1_HI:.3f}] on {PRED_J1_CROSS} of 3 "
          f"crossing and {PRED_J1_ESCAPES} of 3 escapes")
    print("  interpolations it came from: "
          + ", ".join(f"{k} -> {v:.3f}" for k, v in PRED_J1_INTERP.items()))
    print(f"  observed rho_50 median {rho_txt(s)}, {s['n_cross']} of {s['n']} "
          f"crossing, {n_esc} of {s['n']} escapes")
    if not s["rho50_readable"]:
        verdict(f"""
FALSIFIED, AND IN THE DIRECTION THAT IS A RESULT. J1 CENSORS at d224 while E2 at
d256 solves at 0.605, so the whole 0.149 drop on [192, 256] happens inside
[224, 256] -- a factor of 1.14 in width. That is a THRESHOLD, not a slope, and
by the brief's own instruction the breaking point is the result -- BUT ONLY IF
THE RUNG TRAINED, and {n_esc} of {s['n']} seeds got below the {PLATEAU} plateau.
{'That is enough: it learned and still cannot solve, so the censoring is an ARCHITECTURE statement and the threshold reading stands.' if n_esc >= 2 else 'That is NOT enough (registered ' + str(PRED_J1_ESCAPES) + ' of 3), so this is a TRAINABILITY statement and the threshold reading is NOT available: d224 says nothing about architecture, exactly as H1 did not at d128.'}""")
        return "censored"
    inside = PRED_J1_LO <= s["rho50"] <= PRED_J1_HI
    if inside:
        print(f"  interval: INSIDE, as predicted. Width is a SLOPE on "
              f"[192, 256] and the bisector sits on it.")
        return "hit"
    if s["rho50"] < PRED_J1_LO:
        verdict(f"""
MISSED LOW at {s['rho50']:.3f} against [{PRED_J1_LO:.3f}, {PRED_J1_HI:.3f}]. The direction was right and the
size was not: d224 does better than any interpolation of its neighbours, so the
width curve is CONVEX on [192, 256] and most of the 0.149 is bought in the first
32 units of width rather than the last 32. Missing low is the cheap direction --
it makes J1 a better standing-minimum candidate, not a worse one -- but the
three interpolations are falsified and nothing should be interpolated on this
axis again without a fourth point.""")
        return "low"
    above_free = s["rho50"] > FREE
    verdict(f"""
MISSED HIGH at {s['rho50']:.3f} against [{PRED_J1_LO:.3f}, {PRED_J1_HI:.3f}]. {f'AND IT IS ABOVE THE FREE LINE {FREE:.3f}, which is' if above_free else 'That is'}
the falsifier the header named{'' if above_free else ' (though it stays under the free line)'}: the width curve on [192, 256] is CONCAVE,
{'the whole 0.149 drop is packed into [224, 256] -- a factor of 1.14 in width -- so this is a threshold, not a slope, and the brief says that breaking point is the result.' if above_free else 'so the drop is back-loaded toward d256 and the bisector buys less than linear.'}
All three interpolations are falsified by the same amount and in the same
direction, so the failure is in the curve's shape, not in the choice between
them.""")
    return "high"


def score_j2(j2):
    """A depth never run in this grid."""
    s, n_esc = j2["s"], j2["n_esc"]
    print(f"\nJ2 -- L=16 AT d192, A DEPTH NEVER RUN: registered below H2's "
          f"{PRED_J2_BELOW} on >= {PRED_J2_ESCAPES} of 3 escapes")
    print("  the depth evidence it rests on (no_mlp d256): "
          + ", ".join(f"L={k} -> {v:.3f}" for k, v in sorted(DEPTH_EVIDENCE.items())))
    print(f"  observed rho_50 median {rho_txt(s)}, {s['n_cross']} of {s['n']} "
          f"crossing, {n_esc} of {s['n']} escapes "
          f"[spread {s['spread_mean']:.4f}]")
    if n_esc < PRED_J2_ESCAPES:
        verdict(f"""
TRAINABILITY FALSIFIED FIRST, so its rho_50 is NOT a statement about L=16. Only
{n_esc} of {s['n']} seeds got below the {PLATEAU} plateau, against a registered >= {PRED_J2_ESCAPES}, and both
arguments that pointed here -- every d192 cell escaping 3 of 3, and depth
monotonically improving escapes on the no_mlp axis -- are wrong together. The
honest reading is that depth past L=12 is HARDER to train at this budget, which
is itself a breaking point and is the result for this rung. J0 above must not be
read as a shape effect if this is the arm that failed.""")
        return "untrained"
    if not s["rho50_readable"]:
        verdict(f"""
TRAINED AND STILL CENSORED. {n_esc} of {s['n']} seeds escaped the plateau, so this rung
learned something and still cannot solve at any hint density on the grid --
which by the standing distinction (findings on H1 vs I1/H2) IS an architecture
statement: at a matched budget, spending it on depth past L=12 at d192 buys
less than H2 already had at L=12.""")
        return "censored"
    better = s["rho50"] < PRED_J2_BELOW
    print(f"  against H2 (same width, L=12): {s['rho50'] - PRED_J2_BELOW:+.3f} "
          f"= {abs(s['rho50'] - PRED_J2_BELOW) / S_RHO50:.2f} s* "
          f"-- {'lower' if better else 'higher'}, "
          f"{'separable' if abs(s['rho50'] - PRED_J2_BELOW) > S_RHO50 else 'NOT separable'}")
    if better:
        print(f"  PREDICTION HELD: more depth at fixed width lowered rho_50 again, "
              f"now for the first time ABOVE L=12, where the ladder had no point.")
        return "hit"
    verdict(f"""
FALSIFIED, AND IT IS THE FIRST TIME DEPTH HAS NOT HELPED ON THIS AXIS. L=16 at
d192 lands at {s['rho50']:.3f}, ABOVE H2's {PRED_J2_BELOW} at the same width and L=12, after
0.763 -> 0.749 -> 0.605 going 6 -> 8 -> 12 at d256. The depth ladder therefore
TURNS OVER somewhere in (12, 16], and the brief says that breaking point is the
result. Note what was registered as not claimed: nothing in this grid sat above
L=12, so this was always an extrapolation, and it is the extrapolation that
broke rather than a measured trend.""")
    return "miss"


def score_minimum(j1, j1_call, j2, j2_call, shape):
    print("\nJ3 -- THE STANDING-MINIMUM RULE, unchanged from H3 and I3:")
    print(f"  a rung replaces the standing minimum ONLY IF 3 of 3 seeds cross")
    print(f"  AND the median rho_50 is at or below {FREE:.3f}. Two of three is not")
    print(f"  enough (finding 51: a 2-of-3 at 50 epochs was ONE UNFINISHED SEED).")
    tag0, desc0, tr0, rho0 = STANDING
    print(f"  standing: {tag0}, {desc0}, {tr0:,} trainable at {rho0:.3f}")
    fired = []
    for tag, r, call, desc in (("J1", j1, j1_call, "attention-only L=12 d224"),
                               ("J2", j2, j2_call, "attention-only L=16 d192")):
        if r is None:
            print(f"  {tag}: nothing on disk.")
            continue
        s = r["s"]
        if call == "untrained":
            print(f"  {tag} DOES NOT FIRE: the rung did not train, so its rho_50 "
                  f"is not a measurement of its architecture.")
        elif not s["rho50_readable"]:
            print(f"  {tag} DOES NOT FIRE: CENSORED.")
        elif not (s["n"] == 3 and s["n_cross"] == 3):
            print(f"  {tag} DOES NOT FIRE: only {s['n_cross']} of {s['n']} seeds "
                  f"cross. Both conditions or neither.")
        elif s["rho50"] > FREE:
            print(f"  {tag} DOES NOT FIRE: median {s['rho50']:.3f} did not clear "
                  f"{FREE:.3f} ({(s['rho50'] - FREE) / S_RHO50:+.2f} s* past it).")
        else:
            fired.append((tag, desc, s))
            print(f"  {tag} QUALIFIES: 3 of 3 cross at median {s['rho50']:.3f} "
                  f"<= {FREE:.3f}.")
    if not fired:
        print(f"  J3 DOES NOT FIRE. The standing minimum stays {tag0}, {tr0:,} "
              f"trainable at {rho0:.3f}.")
        return
    tag, desc, s = min(fired, key=lambda f: f[2]["trainable"])
    if len(fired) == 2:
        big = max(f[2]["trainable"] for f in fired)
        print(f"  BOTH qualify. The registration did not fix a tie-break, so this")
        print(f"  one is named here and not chosen afterwards on a metric: take the")
        print(f"  SMALLER parameter count, which is {tag}. BUT the two differ by")
        print(f"  {(big - s['trainable']) / ((big + s['trainable']) / 2) * 100:.1f}%, "
              f"inside this job's own {MATCH_TOL * 100:.0f}% matched-budget")
        print(f"  tolerance, so the honest statement is that the minimum is "
              f"~{round(s['trainable'], -4):,} AT EITHER SHAPE and the choice "
              f"between them is not a result.")
    print(f"  J3 FIRES. The standing minimum moves from {tag0} ({tr0:,} at "
          f"{rho0:.3f}) to {desc}, {s['trainable']:,} at {s['rho50']:.3f}:")
    print(f"    size DOWN {tr0 / s['trainable']:.2f}x, and {HUB_TRAINABLE / s['trainable']:.2f}x "
          f"below the hub ({HUB_TRAINABLE:,} at {HUB_RHO50:.3f}).")
    drho = s["rho50"] - rho0
    print(f"    rho_50 moves {drho:+.3f} = {abs(drho) / S_RHO50:.2f} s*, which is "
          f"{'BELOW' if abs(drho) <= S_RHO50 else 'ABOVE'} the separability bar.")
    if abs(drho) <= S_RHO50:
        print(f"    THE CLAIM IS THEREFORE \"{tr0 / s['trainable']:.2f}x smaller at no "
              f"separable cost in rho_50\", never \"better\".")
    else:
        print(f"    THE COST IS SEPARABLE, so the claim is \"{tr0 / s['trainable']:.2f}x smaller "
              f"and {abs(drho):.3f} WORSE in rho_50\" -- both halves, always together.")
    if shape not in ("not_separable", "unreadable"):
        print(f"    AND J0 fired ({shape}), so quote this minimum as the smallest "
              f"rung OF ITS SHAPE, not the smallest rung.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+",
                    default=SCAN_DIRS + [SCAN_E, SCAN_H, SCAN_I, SCAN_J],
                    help="SCAN_E/H/I/J are out of SCAN_DIRS on purpose; they "
                         "are added here only.")
    args = ap.parse_args()

    by = defaultdict(list)
    for r in load_all(args.scan_dir):
        by[cell_key(r)].append(r)
    cells = {k: dict(cell_metrics(v, args.boot), path=v[0]["path"], seed=k[-1])
             for k, v in by.items()}

    print("STAGE J: at a matched parameter count, does SHAPE matter?")
    print(f"  {len(cells)} cells read; TWO rungs 2.6% apart in trainable params")
    print(f"  frozen bar s* = {S_RHO50} rho_50 / {S_MEAN} mean_solve, NOT recomputed")
    print(f"  registered: J0 |J1 - J2| <= s*; J1 in "
          f"[{PRED_J1_LO:.3f}, {PRED_J1_HI:.3f}]; J2 below {PRED_J2_BELOW}")

    print("\nFROZEN-LITERAL CONTROL (the L=12 width ladder, re-read from disk):")
    for tag in ("H1", "I1", "H2", "E2"):
        ms = rung(cells, PARTNER_KEY[tag])
        if not ms:
            print(f"  {tag}: NOT ON DISK -- cannot verify the literals")
            continue
        s, (ne, _nll) = summarize(ms), escapes(ms)
        got = (s["n"], s["n_cross"], rho_txt(s), round(s["mean_solve"], 4),
               round(s["spread_mean"], 4), s["trainable"], ne)
        want = PARTNER[tag]
        print(f"  {tag} {'matches' if got == want else 'DISAGREES WITH'} "
              f"the frozen literals")
        if got != want:
            print(f"    frozen {want}")
            print(f"    disk   {got}")
            print("    the registered grid changed; fix that before reading "
                  "anything below.")

    j1 = report("J1", cells, J1, "no_mlp L=12 d224, the wide-shallow arm",
                PREDICTED_TRAINABLE["J1"])
    j2 = report("J2", cells, J2, "no_mlp L=16 d192, the deep-narrow arm",
                PREDICTED_TRAINABLE["J2"])

    print("\n" + "=" * 72)
    if not (j1 and j2):
        missing = [t for t, r in (("J1", j1), ("J2", j2)) if not r]
        print(f"VERDICT: NOT YET -- {' and '.join(missing)} "
              f"{'has' if len(missing) == 1 else 'have'} no cell on disk, and "
              f"J0 is a paired contrast.")
        if j1:
            score_j1(j1)
        if j2:
            score_j2(j2)
        print("=" * 72)
        return

    matched = score_match(j1, j2)
    shape = score_shape(j1, j2, matched)
    j1_call = score_j1(j1)
    j2_call = score_j2(j2)
    score_minimum(j1, j1_call, j2, j2_call, shape)
    print("=" * 72)

    sr = SUFFICIENCY_AT_REGISTRATION
    print("\nJ4 -- OUT-OF-SAMPLE SUFFICIENCY, free, and reported ahead of")
    print(f"  everything if it fails. At registration the depth-ladder fit held")
    print(f"  on {sr['rungs']} of {sr['rungs']} well-determined held-out rungs over "
          f"{sr['cells']} cells, worst")
    print(f"  discriminating miss {sr['worst']} against the LOO bar {LOO_BAR}.")
    sd = SUFFICIENCY_AT_DELIVERY
    print(f"  (the H+I write-up refit the same curve over all eight scan dirs "
          f"and got\n   {sd['rungs']} of {sd['rungs']} over {sd['cells']} cells, "
          f"{sd['disc']} discriminating, same worst miss {sd['worst']}.)")
    sj = SUFFICIENCY_AT_STAGEJ
    print(f"  Delivered, with stage J's dir added: {sj['rungs']} of {sj['rungs']} "
          f"over {sj['cells']} cells,\n   {sj['disc']} discriminating, same worst "
          f"miss {sj['worst']}. That is the number of record.")
    print(f"  Prediction: BOTH stage J rungs land on the curve within {LOO_BAR}.")
    print("  Run (NINE dirs -- SCAN_G is held out of the paired scoring above as")
    print("  a different budget, but it is a fair held-out test of this curve):")
    print("    python3 -m scripts.acc_vs_solve --scan-dir \\\n      " +
          " \\\n      ".join(SUFFICIENCY_DIRS))
    print("  A rung falling OFF the curve is where a real shape effect would")
    print("  show up, so if J0's falsifier fired, read that output before")
    print("  quoting J0 anywhere.")


if __name__ == "__main__":
    main()
