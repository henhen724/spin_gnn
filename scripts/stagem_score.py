"""Score stage M: which half of the stage-F cut is load-bearing at N=96?

    python3 -m scripts.stagem_score [--boot 2000]

THE QUESTION. Stage F (51656823, finding 55) falsified the transfer of the N=48
standing minimum to N=96: stripped `no_mlp L=16 d192` scored rho_50 0.850 against
the paired control `full L=12 d256` at 0.676, +0.175 = 1.47 s*_96 against a bar
of 0.119. But that contrast moves TWO things at once -- the MLP and the SHAPE --
so it cannot say which member of the pair did it, and the two answers point
opposite ways. M completes the 2x2 square whose other two corners stage F already
measured at three seeds each:

                   L=12 d256                 L=16 d192
   full      A  control, ON DISK           B  stage M, 3 seeds
   no_mlp    C  stage M, 3 seeds           D  stripped, ON DISK

   dMLP   = C - A        the MLP, at the control's shape
   dSHAPE = B - A        the shape, with the MLP intact
   inter  = (D - A) - (dMLP + dSHAPE)

WHAT M DOES NOT SEPARATE: the SHAPE factor bundles depth (12->16) and width
(256->192). M splits ARCH from SHAPE, never depth from width. Separating those is
the next stage, and only if the shape half turns out load-bearing.

THIS FILE IS THE REGISTRATION, TRANSCRIBED from the sbatch header
(scripts/slurm/sudoku2_n96_cuthalf_M_2026-11-09.sbatch), committed in the same
commit and before any cell runs -- the array is submitted with --dependency on
the commit job, so no cell can start ahead of its own registration. Every
literal below is either copied from that header or measured off cells that were
already final when M was submitted (jobs 51743995 and 51744471, read-only).
Nothing here is chosen after seeing a stage M number; git log is the timestamp.

FIVE THINGS THAT WOULD SILENTLY CORRUPT THIS SCORE, and are guarded below:

  1. s* = 0.1484 AND mean_solve 0.2918 ARE N=48 NUMBERS AND ARE NOT REUSED. All
     three bars here are the mean |max-min| across the four multi-seed leak-free
     N=96 rungs that existed at registration. s*_M = 0.117 is the re-estimate
     stage F printed and explicitly said a later stage should freeze from; F's
     own verdicts used 0.119 and are untouched.
  2. RHO_50 CENSORS. F's worst stripped seed landed at 0.937 with the grid's top
     bin at 0.95 -- one bin of headroom -- so a corner worse than the stripped
     arm can censor and have no quotable level at all. Hence two never-censoring
     metrics with their own frozen bars, and M2 gates the levels.
  3. FINDING 44's 1.64 PLATEAU IS AN N=48 NUMBER. Every leak-free N=96 cell sits
     at 1.7456-1.8320 and N=96's own plateau has never been measured, so NO
     ESCAPE COUNT IS QUOTED ANYWHERE IN M.
  4. THE N=96 GRID ENDS AT 0.32, the N=48 grid at 0.33. Never pooled, and no
     N=96 dir may enter minarch_analysis.SCAN_DIRS -- an N=96 `full` L=12 cell
     reduces to the same rung_key as the N=48 `full` L=12 rung and would move a
     frozen bar. main() asserts this before reading anything.
  5. acc_vs_solve.py's FITTED accuracy-to-solve curve is an N=48 object. Only its
     val_accuracy() reader is used here. Evaluating the N=48 fit against an N=96
     cell would be the same error as reusing s* = 0.1484.

REGISTERED, and reported in this order:

  M2  EVERY CORNER TRAINS. 3 of 3 seeds in each new arm produce a MEASURED
      rho_50. FALSIFIER: any arm censors -- then that corner has no quotable
      level, M0/M1 are read on mean_solve and token accuracy only, and the censor
      IS that corner's result ("this corner does not train at this budget"),
      reported as the finding and not as a failed run. Three of three censoring
      is an architecture statement; one of three is finding 51's seed story.

  M0  THE HEADLINE: THE DECOMPOSITION NAMES ONE HALF. Exactly one of |dMLP| and
      |dSHAPE| exceeds s*_M = 0.117 on rho_50, and that half carries F's +0.175.
      (a) MLP ONLY: the attention-only cut is what fails to transfer; a future
          N=96 minimum keeps the MLP and looks for its savings in the shape.
      (b) SHAPE ONLY: the attention-only cut DOES transfer and L=16 d192 is what
          fails; the next stage separates depth from width and J1's convexity
          question reopens on the condition under which it was dropped.
      (c) BOTH: the gap is shared, there is no single load-bearing half, and the
          honest report is both main effects with their sum.
      (d) NEITHER: THE INTERESTING ONE -- each single step is free, the pair is
          not, so the cost is an INTERACTION, the F gap does not decompose into
          main effects, and the "smallest architecture" question cannot be
          answered one axis at a time. That would indict the ladder method
          itself, which is why it is written down in advance.
      TIE-BREAK, fixed now: exactly s*_M, or anything inside it, is NOT
      separable. The falsifier is the claim that needs the evidence.

  M1  ADDITIVITY. |(dMLP + dSHAPE) - 0.175| <= s*_M = 0.117.
      FALSIFIER: outside. Then the interaction exceeds a seed spread and NO
      single-factor statement generalizes to the other corner. The SIGN is
      reported explicitly: super-additive (both cuts together cost MORE than the
      sum of the parts) is the reading that most threatens the ladder method.

  M3  THE THREE METRICS AGREE ON THE SIGN. On stage F they did, at nearly the
      same strength: stripped worse by 1.47 bars on rho_50, 1.52 on mean_solve,
      1.43 on token accuracy -- which is why F's verdict is not a rho_50
      artifact. Each metric is called WORSE / NO CALL / BETTER against its OWN
      frozen bar; a metric inside its bar is noise and makes no call.
      HELD iff no metric says WORSE while another says BETTER, and, where rho_50
      does make a call, at least one never-censoring metric makes the same one.
      FALSIFIER: a corner worse on rho_50 but not on accuracy, or the reverse.
      That corner is then THE result of the stage -- the standing non-negotiable
      says a rung that falls off the accuracy-to-solve relation is the
      interesting one -- and it is reported, not averaged away.

  M4  THE N=48 STANDING MINIMUM DOES NOT MOVE ON N=96 EVIDENCE. The standing
      minimum is attention-only L=16 d192, 2,504,890 trainable at rho_50 0.580
      (finding 53, rule H3 = I3 = J3 = K2), an N=48 ladder result on the N=48
      grid. M can corroborate or undermine it; it can never replace it. Nor can M
      restore the cross-size reading F removed: after finding 55 every "smallest
      architecture" claim in this programme is an N=48 claim, whichever half of
      the cut turns out to be load-bearing.
"""
import argparse
import os
from collections import defaultdict

from scripts.acc_vs_solve import val_accuracy
from scripts.minarch_analysis import (SCAN_DIRS, _bound_median, cell_metrics, history,
                                      load_all, median, spread)
from scripts.n96_control import SCAN_N96_COSINE, SCAN_N96_FALL, label, n96_key
from scripts.stagef_score import SCAN_F

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_M = os.path.join(_SCR, "spin_gnn", "sudoku_n96_cuthalf_M_2026-11-09")
SCAN_M_DIRS = [SCAN_M, SCAN_F, SCAN_N96_FALL, SCAN_N96_COSINE]

# The four corners of the square, as n96_key tuples:
#   (n_nodes, num_layers, nhead, arch_mode, d_model, ff, warmup, epochs, n_inst, sched)
A_CONTROL = (96, 12, 4, "full", 256, 1024, 0, 50, 240000, "constant")
B_SHAPE = (96, 16, 4, "full", 192, 1024, 0, 50, 240000, "constant")
C_MLP = (96, 12, 4, "no_mlp", 256, 1024, 0, 50, 240000, "constant")
D_STRIPPED = (96, 16, 4, "no_mlp", 192, 1024, 0, 50, 240000, "constant")

CORNERS = [("A", "control  full L=12 d256", A_CONTROL),
           ("B", "shape    full L=16 d192", B_SHAPE),
           ("C", "MLP      no_mlp L=12 d256", C_MLP),
           ("D", "stripped no_mlp L=16 d192", D_STRIPPED)]

# --- frozen at registration, never recomputed into a verdict (guard 1) --------
# Measured read-only by 51743995 (rho_50) and 51744471 (the other two) over the
# four multi-seed leak-free N=96 rungs that existed before M was submitted.
S_RHO = 0.117
S_RHO_PARTS = {"L=8 full": 0.138, "L=12 full": 0.115, "L=12 cosine": 0.105,
               "L=16 no_mlp": 0.111}
S_MS = 0.1586
S_MS_PARTS = {"L=8 full": 0.1766, "L=12 full": 0.1623, "L=12 cosine": 0.1355,
              "L=16 no_mlp": 0.1598}
S_ACC = 0.0079
S_ACC_PARTS = {"L=8 full": 0.0094, "L=12 full": 0.0045, "L=12 cosine": 0.0070,
               "L=16 no_mlp": 0.0107}

# Stage F's measured gap, D - A, in all three metrics. M1 checks the square's
# main effects against the rho_50 figure; the other two are the cross-checks.
F_GAP_RHO = 0.175        # 0.850 - 0.676, as stagef_score printed it
F_GAP_MS = -0.2418       # 0.1811 - 0.4229
F_GAP_ACC = -0.0113      # 0.4709 - 0.4822

CONTROL_NLL_BAND = (1.7456, 1.8320)
GRID_FLOOR_96, GRID_TOP_96 = 0.32, 0.95
S48, S48_MS = 0.1484, 0.2918        # quoted ONLY to say they are not used here
S96_F = 0.119                       # stage F's own frozen bar, also not reused

N48_MINIMUM = ("attention-only L=16 d192", 2504890, 0.580)
A_TOTAL, D_TOTAL = 9790058, 8929514


def collect(scan_dirs, boot):
    """Every leak-free N=96 cell, keyed so it cannot collide with an N=48 rung."""
    by = defaultdict(list)
    for r in load_all(scan_dirs):
        if int(r["n_nodes"]) != 96:
            continue
        by[n96_key(r) + (int(r["seed"]),)].append(r)
    return {k: dict(cell_metrics(v, boot), seed=k[-1]) for k, v in by.items()}


def rung(cells, key):
    seeds = sorted(k[-1] for k in cells if k[:-1] == key)
    return [cells[key + (s,)] for s in seeds] or None


def summarize(ms):
    bm = _bound_median([(m["rho50"], m["flag"] == "ok") for m in ms])
    accs = [val_accuracy(m["path"]) for m in ms]
    accs = [a for a in accs if a == a]
    return {
        "n": len(ms), "n_cross": sum(1 for m in ms if m["flag"] == "ok"),
        "rho50": bm["rho50_bm"], "rho50_readable": bm["bm_readable"],
        "mean_solve": median([m["mean_solve"] for m in ms]),
        "spread_ms": spread([m["mean_solve"] for m in ms]),
        "acc": median(accs) if accs else float("nan"),
        "spread_acc": spread(accs) if len(accs) > 1 else float("nan"),
        "seeds": [m["seed"] for m in ms],
        "per_seed": [(m["seed"], m["rho50"], m["flag"] == "ok") for m in ms],
        "per_seed_acc": list(zip([m["seed"] for m in ms], accs)),
        "n_params": ms[0]["n_params"], "n_trainable": ms[0]["n_trainable"],
        "nll": [history(m["path"])[1] for m in ms],
    }


def show(tag, name, s):
    print(f"\n=== CORNER {tag} -- {name} ===")
    if s is None:
        print("  nothing on disk yet.")
        return
    rho = f"{s['rho50']:.3f}" if s["rho50_readable"] else "CENSORED"
    print(f"  {s['n']} seed(s) {s['seeds']}, {s['n_cross']} crossing")
    print(f"  rho_50 median {rho}   mean_solve {s['mean_solve']:.4f}   "
          f"val_token_acc {s['acc']:.4f}")
    print("  per seed rho_50: " + ", ".join(
        f"s{sd} " + (f"{v:.3f}" if ok else "CENSORED") for sd, v, ok in s["per_seed"]))
    print("  per seed acc:    " + ", ".join(f"s{sd} {a:.4f}" for sd, a in s["per_seed_acc"]))
    if s["n"] > 1:
        print(f"  spreads: mean_solve {s['spread_ms']:.4f}   acc {s['spread_acc']:.4f}")
    # No escape count: finding 44's plateau is an N=48 number (guard 3).
    print("  final train NLL: " + " / ".join(f"{v:.4f}" for v in s["nll"])
          + f"   [N=96 band to date {CONTROL_NLL_BAND[0]:.4f}-{CONTROL_NLL_BAND[1]:.4f};")
    print("   no escape count is quoted at N=96 -- its plateau has never been measured]")
    print(f"  n_params TOTAL {s['n_params']:,}   trainable {s['n_trainable']:,}"
          "   (cross-corner sizes use TOTAL on all four sides, guard 4 of the sbatch)")


def call(delta, bar, higher_is_better):
    """WORSE / NO CALL / BETTER against this metric's OWN frozen bar."""
    if delta != delta:
        return "no data"
    if abs(delta) <= bar:
        return "no call"
    improved = delta > 0 if higher_is_better else delta < 0
    return "better" if improved else "worse"


METRICS = [("rho_50", "rho50", S_RHO, False),
           ("mean_solve", "mean_solve", S_MS, True),
           ("val_token_acc", "acc", S_ACC, True)]


def deltas(x, a):
    """Corner x minus corner A in all three metrics, with each metric's call."""
    out = {}
    for name, key, bar, hib in METRICS:
        if x is None or a is None:
            out[name] = (float("nan"), bar, "no data")
            continue
        if key == "rho50" and not (x["rho50_readable"] and a["rho50_readable"]):
            out[name] = (float("nan"), bar, "censored")
            continue
        d = x[key] - a[key]
        out[name] = (d, bar, call(d, bar, hib))
    return out


def score_m2(b, c):
    """M2 gates the levels: an unmeasured rho_50 cannot be compared to one."""
    print(f"\nM2 -- EVERY CORNER TRAINS: registered 3 of 3 seeds per new arm "
          f"produce a MEASURED rho_50")
    verdict = "held"
    for tag, s in (("B shape", b), ("C MLP", c)):
        if s is None:
            print(f"  {tag}: nothing on disk. Not scored.")
            verdict = "absent"
            continue
        n_cens = s["n"] - s["n_cross"]
        print(f"  {tag}: {s['n_cross']} of {s['n']} crossing, {n_cens} censored "
              f"(grid {GRID_FLOOR_96}-{GRID_TOP_96})")
        if s["n"] < 3:
            print(f"    INCOMPLETE: {s['n']} of 3 seeds. Not scored -- the registered "
                  f"rule is median-vs-median at three seeds (finding 51).")
            verdict = "incomplete" if verdict != "absent" else verdict
        elif n_cens:
            print(f"    CENSORED, AND THAT IS THIS CORNER'S RESULT: it does not train "
                  f"to a measurable rho_50 at this budget.")
            print(f"    Read M0/M1 for it on mean_solve {s['mean_solve']:.4f} and "
                  f"accuracy {s['acc']:.4f} only; its LEVEL is not quotable.")
            print("    " + ("all three censoring is an ARCHITECTURE statement"
                            if n_cens == 3 else
                            f"{n_cens} of 3 is finding 51's SEED story, not yet an "
                            f"architecture statement"))
            verdict = "censored"
    if verdict == "held":
        print("  M2 HELD. Both new corners have quotable rho_50 levels.")
    return verdict


def score_m0(a, b, c, d, m2):
    print(f"\nM0 -- THE HEADLINE: registered that EXACTLY ONE of |dMLP|, |dSHAPE| "
          f"exceeds s*_M = {S_RHO:.3f}")
    print(f"  s*_M is the mean |max-min| rho_50 over the four multi-seed N=96 rungs "
          f"at registration")
    print("  (" + ", ".join(f"{k} {v:.3f}" for k, v in S_RHO_PARTS.items())
          + f"). s* = {S48} is an N=48 number; F's own {S96_F} is F's.")
    if a is None or b is None or c is None:
        print("  a corner is missing. Not scored.")
        return None
    dm = deltas(c, a)["rho_50"]
    ds = deltas(b, a)["rho_50"]
    if dm[2] in ("censored", "no data") or ds[2] in ("censored", "no data"):
        print(f"  NOT SCORED AS LEVELS: dMLP is '{dm[2]}', dSHAPE is '{ds[2]}'. "
              f"See M2; read the never-censoring metrics below.")
        return None
    print(f"  dMLP   = C - A = {c['rho50']:.3f} - {a['rho50']:.3f} = {dm[0]:+.3f}"
          f" = {abs(dm[0]) / S_RHO:.2f} s*_M   [{dm[2].upper()}]")
    print(f"  dSHAPE = B - A = {b['rho50']:.3f} - {a['rho50']:.3f} = {ds[0]:+.3f}"
          f" = {abs(ds[0]) / S_RHO:.2f} s*_M   [{ds[2].upper()}]")
    print(f"  for reference, stage F's two-factor gap D - A = {F_GAP_RHO:+.3f} "
          f"= {F_GAP_RHO / S_RHO:.2f} s*_M")
    big_m, big_s = abs(dm[0]) > S_RHO, abs(ds[0]) > S_RHO
    if big_m and not big_s:
        out = "mlp"
        print("  M0 HELD, OUTCOME (a): THE MLP IS LOAD-BEARING AT N=96.")
        print("  The attention-only cut is what fails to transfer; changing the shape")
        print("  to L=16 d192 with the MLP intact is free. A future N=96 minimum keeps")
        print("  the MLP and looks for its savings in the shape, not in the MLP.")
    elif big_s and not big_m:
        out = "shape"
        print("  M0 HELD, OUTCOME (b): THE SHAPE IS LOAD-BEARING AT N=96.")
        print("  The attention-only cut DOES transfer -- removing the MLP at the")
        print("  control's own shape is free -- and L=16 d192 is what fails. The next")
        print("  stage separates DEPTH from WIDTH, and J1's convexity question reopens")
        print("  on exactly the condition under which it was dropped after F.")
    elif big_m and big_s:
        out = "both"
        print("  M0 FALSIFIED, OUTCOME (c): THE GAP IS SHARED. Both halves cost more")
        print("  than a seed spread, so there is no single load-bearing half and no")
        print(f"  headline beyond the pair: dMLP {dm[0]:+.3f} and dSHAPE {ds[0]:+.3f},")
        print("  reported together with their sum, never as one cause.")
    else:
        out = "interaction"
        print("  M0 FALSIFIED, OUTCOME (d): NEITHER HALF IS SEPARABLE ON ITS OWN, SO")
        print("  THE COST IS AN INTERACTION. Each single step is inside the bar and")
        print(f"  the pair costs {F_GAP_RHO:+.3f}. The F gap does NOT decompose into")
        print("  main effects, and the 'smallest architecture' question cannot be")
        print("  answered one axis at a time -- which is an indictment of the ladder")
        print("  method this programme has used since stage A, and was registered in")
        print("  advance as the outcome that would say so.")
    if m2 in ("censored", "incomplete", "absent"):
        print(f"  CAVEAT FROM M2 ('{m2}'): at least one corner has no quotable level,")
        print("  so this verdict survives only as far as M2 allows. See above.")
    return out


def score_m1(a, b, c, d):
    print(f"\nM1 -- ADDITIVITY: registered |(dMLP + dSHAPE) - {F_GAP_RHO:+.3f}| "
          f"<= s*_M = {S_RHO:.3f}")
    if None in (a, b, c, d):
        print("  a corner is missing. Not scored.")
        return
    for name, key, bar, hib, fgap in (("rho_50", "rho50", S_RHO, False, F_GAP_RHO),
                                      ("mean_solve", "mean_solve", S_MS, True, F_GAP_MS),
                                      ("val_token_acc", "acc", S_ACC, True, F_GAP_ACC)):
        if key == "rho50" and not all(s["rho50_readable"] for s in (a, b, c, d)):
            print("  rho_50: a corner is censored; additivity is not readable on it.")
            continue
        dm, ds = c[key] - a[key], b[key] - a[key]
        obs = d[key] - a[key]
        inter = obs - (dm + ds)
        tag = "PRIMARY" if key == "rho50" else "cross-check"
        print(f"  {name} ({tag}): dMLP {dm:+.4f} + dSHAPE {ds:+.4f} = {dm + ds:+.4f}"
              f"  vs D - A {obs:+.4f}")
        print(f"    interaction {inter:+.4f} = {abs(inter) / bar:.2f} bars "
              f"(bar {bar}); F's registered gap was {fgap:+.4f}")
        if abs(inter) <= bar:
            print("    ADDITIVE on this metric: the main effects are the whole story.")
        else:
            # Super-additive = the two cuts together cost more than the sum of the
            # parts. "More costly" is +inter for rho_50 and -inter for the two
            # higher-is-better metrics.
            worse_together = inter > 0 if not hib else inter < 0
            print(f"    NOT ADDITIVE on this metric. The interaction exceeds the seed")
            print(f"    spread, so no single-factor statement here generalizes to the")
            print(f"    other corner. Sign: {'SUPER-ADDITIVE -- the two cuts together cost MORE than the sum of the parts' if worse_together else 'sub-additive -- the pair costs LESS than the sum of the parts'}.")


def score_m3(a, b, c):
    print(f"\nM3 -- THE THREE METRICS AGREE ON THE SIGN (bars: rho_50 {S_RHO}, "
          f"mean_solve {S_MS}, acc {S_ACC})")
    print(f"  on stage F they did, at nearly equal strength: {F_GAP_RHO / S_RHO:.2f} / "
          f"{abs(F_GAP_MS) / S_MS:.2f} / {abs(F_GAP_ACC) / S_ACC:.2f} bars, which is why")
    print("  F's verdict is not a rho_50 artifact.")
    print("  GUARD: acc_vs_solve's FITTED curve is an N=48 object and is not applied")
    print("  to any N=96 cell; this comparison is N=96 to N=96 (guard 5).")
    if a is None:
        print("  no control. Not scored.")
        return
    verdict = "held"
    for tag, s in (("B shape  (full L=16 d192)", b), ("C MLP    (no_mlp L=12 d256)", c)):
        if s is None:
            print(f"  {tag}: nothing on disk.")
            continue
        ds = deltas(s, a)
        print(f"  {tag} vs control A:")
        calls = []
        for name, _, _, _ in METRICS:
            dv, bar, cl = ds[name]
            calls.append(cl)
            bars = f"{abs(dv) / bar:.2f} bars" if dv == dv else "--"
            print(f"    {name:<14} {dv:+.4f}  {bars:<10} {cl.upper()}")
        if "worse" in calls and "better" in calls:
            print("    M3 FALSIFIED FOR THIS CORNER, AND IT IS THE RESULT OF THE STAGE:")
            print("    it is worse on one metric and better on another beyond both")
            print("    bars. The standing non-negotiable says a rung that falls off the")
            print("    accuracy-to-solve relation is the interesting one -- report it,")
            print("    do not average it away.")
            verdict = "falsified"
        elif calls[0] in ("worse", "better") and calls[0] not in calls[1:]:
            print(f"    M3 FALSIFIED FOR THIS CORNER: rho_50 calls '{calls[0]}' and")
            print("    neither never-censoring metric corroborates it. A level that")
            print("    only one metric sees is a grid artifact until shown otherwise.")
            verdict = "falsified"
        else:
            print("    consistent.")
    if verdict == "held":
        print("  M3 HELD. No corner contradicts itself across the three metrics.")
    return verdict


def score_m4(a, b, c, d):
    desc, tr, rho = N48_MINIMUM
    print("\nM4 -- THE N=48 STANDING MINIMUM DOES NOT MOVE ON N=96 EVIDENCE")
    print(f"  standing (finding 53, rule H3 = I3 = J3 = K2): {desc}, {tr:,} trainable")
    print(f"  at rho_50 {rho:.3f} on the N=48 grid (floor 0.33). Not comparable to")
    print(f"  anything above: the N=96 grid floor is {GRID_FLOOR_96} and the yardsticks")
    print(f"  differ ({S48} / {S48_MS} at N=48 against {S_RHO} / {S_MS} here).")
    print("  M4 is a REGISTRATION, not a measurement. M can corroborate or undermine")
    print("  the N=48 minimum; it can never replace it. Nor can any outcome here")
    print("  restore the cross-size reading finding 55 removed -- every 'smallest")
    print("  architecture' claim in this programme stays an N=48 claim.")
    if all(s is not None for s in (a, b, c, d)):
        print("  SIZES, TOTAL n_params on all four sides (guard 4):")
        for tag, name, s in (("A", "full   L=12 d256", a), ("B", "full   L=16 d192", b),
                             ("C", "no_mlp L=12 d256", c), ("D", "no_mlp L=16 d192", d)):
            print(f"    {tag}  {name}  {s['n_params']:>12,}")
        print(f"    registered: A {A_TOTAL:,}, D {D_TOTAL:,} -- "
              f"{'ok' if (a['n_params'], d['n_params']) == (A_TOTAL, D_TOTAL) else 'MISMATCH'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+", default=SCAN_M_DIRS)
    args = ap.parse_args()

    for dd in args.scan_dir:
        assert dd not in SCAN_DIRS, (
            f"{os.path.basename(dd)} is in SCAN_DIRS -- an N=96 dir in the pooled "
            f"list merges N=96 cells into N=48 rungs and moves a frozen bar "
            f"(guard 4 in this file's docstring). Take it out before running "
            f"anything.")

    print("STAGE M -- which half of the stage-F cut is load-bearing at N=96?")
    print("  registration: scripts/slurm/sudoku2_n96_cuthalf_M_2026-11-09.sbatch")
    print(f"  bars FROZEN at registration: rho_50 {S_RHO}, mean_solve {S_MS}, "
          f"acc {S_ACC},")
    print("  each the mean |max-min| over the four multi-seed N=96 rungs that existed")
    print("  then (jobs 51743995 / 51744471). N=48 numbers are not reused; F's 0.119")
    print("  is F's and is not reused either.")
    print("  no escape count is quoted at N=96; the grid floor is 0.32, never pooled")
    print("  with the N=48 grid's 0.33.")

    cells = collect(args.scan_dir, args.boot)
    print(f"\n{len(cells)} leak-free N=96 cell(s) read from {len(args.scan_dir)} dirs")

    got = {}
    for tag, name, key in CORNERS:
        ms = rung(cells, key)
        got[tag] = summarize(ms) if ms else None
        show(tag, name, got[tag])
    a, b, c, d = got["A"], got["B"], got["C"], got["D"]

    m2 = score_m2(b, c)
    score_m0(a, b, c, d, m2)
    score_m1(a, b, c, d)
    score_m3(a, b, c)
    score_m4(a, b, c, d)

    print("\nRUNGS ON DISK (orientation; the square's four corners are above)")
    rungs = defaultdict(list)
    for k, m in cells.items():
        rungs[k[:-1]].append(m)
    for key, ms in sorted(rungs.items()):
        ok = [m["rho50"] for m in ms if m["flag"] == "ok"]
        print(f"  {label(key):<52} {len(ms)} seed(s), {len(ok)} crossing"
              + (f", spread {max(ok) - min(ok):.3f}" if len(ok) > 1 else ""))


if __name__ == "__main__":
    main()
