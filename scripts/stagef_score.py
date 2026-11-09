"""Score stage F (51656823): does the minimal architecture replicate at N=96?

    python3 -m scripts.stagef_score [--boot 2000]

THE QUESTION. Stages A-J cut machinery out of the N=48 model and left a
standing minimum: attention-only, L=16, d192, 2,504,890 trainable at rho_50
0.580 against the N=48 hub's 4,899,898 at 0.592 -- 1.96x smaller at no
separable cost. EVERY ONE OF THOSE CUTS WAS MEASURED AT ONE PROBLEM SIZE. F
asks whether the surviving architecture is a fact about direct deduction or a
fact about N=48, by running it at N=96 against a PAIRED complete-model control.
It is a REPLICATION question, not another cut: F does not isolate depth, width
and the MLP at N=96, which would be another ladder and another 200 GPU-h.

THIS FILE IS THE REGISTRATION, TRANSCRIBED from the sbatch header
(scripts/slurm/sudoku2_n96_minarch_F_2026-11-09.sbatch, committed 217d052
before the array was submitted), and it is committed before the first cell
lands. Every literal below is either copied from that header or measured off
cells that were already final when F was submitted. Nothing here is chosen
after seeing a stage F number; git log is the timestamp.

WHAT ALREADY EXISTED, and why F is only 4 cells (scripts/n96_control.py):

    N=96 full L=12  240k x 50ep   rho_50 0.621   seeds 0.679 / 0.564   9,790,058
    N=96 full L=8   240k x 50ep   rho_50 0.760   seeds 0.691 / 0.829   6,631,018
    N=96 full L=16  240k x 50ep   rho_50 0.613   one seed             12,949,098
    N=96 full L=12  240k x 50 cos rho_50 0.778   seeds 0.830 / 0.726   (finding 47)

All leak-free (`shuffle_adj` AND `fall`), all at the same 240k x 50ep budget
the N=48 minarch grid uses, all on the IDENTICAL 14 x 400 rho grid. So F buys
only what is missing: the stripped arm at three seeds, and the THIRD CONTROL
SEED -- no N=96 rung anywhere has three, and at two the L=12 control median is
a coin flip between 0.564 and 0.679.

FOUR THINGS THAT WOULD SILENTLY CORRUPT THIS SCORE, and are guarded below:

  1. s* = 0.1484 IS AN N=48 NUMBER AND IS NOT REUSED. The N=96 yardstick is
     s*_96 = 0.119, the mean |max-min| rho_50 over the three two-seed N=96
     rungs that existed at registration (0.115 / 0.138 / 0.104). It is a
     two-seed estimate of a spread and therefore weak; task 3 improves it, and
     the re-estimate is PRINTED alongside but never substituted into a verdict.
  2. FINDING 44's 1.64 PLATEAU IS AN N=48 NUMBER TOO. Every leak-free N=96 cell
     sits at 1.7456-1.8320, which is not eight failures to train -- N=96's own
     plateau level has never been measured. So NO ESCAPE COUNT IS QUOTED
     ANYWHERE IN F. Final train NLL is printed and compared N=96 to N=96 only.
  3. THE N=96 RHO GRID ENDS AT 0.32, the N=48 grid at 0.33. The two are never
     pooled, and no N=96 dir may enter SCAN_DIRS -- the N=96 `full` L=12 cells
     reduce to the same rung_key as the N=48 `full` L=12 rung and would move a
     frozen bar. main() asserts this before reading anything.
  4. n_trainable_params DOES NOT EXIST in the old control artifacts (the field
     was added 11-04). The new cells have it. Sizes are therefore quoted as
     TOTAL n_params on both sides, and both numbers are printed so neither is
     taken for the other.

REGISTERED, and reported in this order:

  F0  THE HEADLINE. The stripped arm reaches the complete model's rho_50 at
      N=96: |rho_50(no_mlp L=16 d192) - rho_50(full L=12 d256)| <= s*_96 =
      0.119, with the control median over its THREE seeds once task 3 lands.
      Then the minimal architecture is a fact about the task and the headline
      is "~4x fewer parameters at no separable cost, at two problem sizes".
      FALSIFIER, AND IT NAMES THE DIRECTION. HIGHER by more than 0.119: the
      N=48 minimum does NOT transfer, the MLP or the width is load-bearing at
      N=96, and every "smallest architecture" claim in this programme is an
      N=48 claim and must be relabelled as one. LOWER by more than 0.119: do
      NOT celebrate -- that is evidence the N=96 CONTROL is undertrained at this
      budget, not that stripping helps, and the next move is a step-budget check
      on the control rather than a victory lap.
      TIE-BREAK, fixed in advance: exactly s*_96, or anything inside it, is NOT
      separable. The falsifier is the claim that needs the evidence.

  F1  IT TRAINS AT ALL. 3 of 3 stripped seeds produce a MEASURED rho_50 -- the
      solve-rate curve crosses 0.5 somewhere on the 14-point grid.
      FALSIFIER: any seed censors. Then F0 is unreadable as a level comparison
      and THE RESULT IS THE BREAKING POINT -- "the attention-only stack that
      works at N=48 does not train at N=96 at this budget" -- reported as the
      headline, not as a failed run. All three censoring is an architecture
      statement; one of three is finding 51's seed story and needs the other
      two read on mean_solve.

  F2  THE CONTROL IS NOT ONE DRAW. The third control seed lands inside the
      existing [0.564, 0.679] bracket, so the 3-seed control median stays
      within s*_96 of the 2-seed median 0.621.
      FALSIFIER: outside. Then the N=96 control is seed-dominated at two seeds,
      F0 survives only as an ORDERING ("the stripped arm is / is not worse"),
      and the LEVELS ARE NOT QUOTABLE. This is the single most likely way F
      ends up weaker than it looks, which is why task 3 is in the job.

  F3  THE N=48 STANDING MINIMUM DOES NOT MOVE ON N=96 EVIDENCE. Stated as a
      registration so that no verdict here is later read as one: the standing
      minimum is an N=48 ladder result (rule H3 = I3 = J3) and N=96 is a
      different problem with a different grid floor. F can CORROBORATE or
      UNDERMINE the N=48 minimum; it can never replace it.
"""
import argparse
import os
from collections import defaultdict

from scripts.minarch_analysis import (SCAN_DIRS, _bound_median, _config, cell_metrics,
                                      history, load_all, median, spread)
from scripts.n96_control import SCAN_N96_COSINE, SCAN_N96_FALL, label, n96_key

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_F = os.path.join(_SCR, "spin_gnn", "sudoku_n96_minarch_F_2026-11-09")
SCAN_F_DIRS = [SCAN_F, SCAN_N96_FALL, SCAN_N96_COSINE]

# The two rungs F scores, as n96_key tuples:
#   (n_nodes, num_layers, nhead, arch_mode, d_model, ff, warmup, epochs, n_inst, sched)
STRIPPED = (96, 16, 4, "no_mlp", 192, 1024, 0, 50, 240000, "constant")
CONTROL = (96, 12, 4, "full", 256, 1024, 0, 50, 240000, "constant")

# --- frozen at registration, never recomputed into a verdict -----------------
S96 = 0.119                     # mean |max-min| over the three two-seed N=96 rungs
# The header derived these by subtracting the ROUNDED per-seed rho_50s it
# printed, so the cosine part reads 0.104 there and 0.105 off disk. The mean is
# 0.119 either way; the registered value is what every verdict uses.
S96_PARTS = {"L=12 fall": 0.115, "L=8 fall": 0.138, "L=12 cosine": 0.104}
CONTROL_2SEED = 0.621           # the control median as it stood at registration
CONTROL_BRACKET = (0.564, 0.679)
CONTROL_NLL_BAND = (1.7456, 1.8320)   # every leak-free N=96 cell to date
GRID_FLOOR_96 = 0.32            # the N=96 rho grid's bottom bin; N=48's is 0.33
S48 = 0.1484                    # quoted ONLY to say it is not used here

# The N=48 result F replicates, frozen (finding 53, rule H3 = I3 = J3).
N48_MINIMUM = ("attention-only L=16 d192", 2504890, 0.580)
N48_HUB = ("full L=6 d256", 4899898, 0.592)

PREDICTED_TRAINABLE_96 = 2618602   # no_mlp L=16 d192 at N=96, from the sbatch
PREDICTED_TOTAL_96 = 8929514
CONTROL_TOTAL = 9790058


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
    return {
        "n": len(ms), "n_cross": sum(1 for m in ms if m["flag"] == "ok"),
        "rho50": bm["rho50_bm"], "rho50_readable": bm["bm_readable"],
        "mean_solve": median([m["mean_solve"] for m in ms]),
        "spread_mean": spread([m["mean_solve"] for m in ms]),
        "seeds": [m["seed"] for m in ms],
        "per_seed": [(m["seed"], m["rho50"], m["flag"] == "ok") for m in ms],
        "n_params": ms[0]["n_params"],
        "nll": [history(m["path"])[1] for m in ms],
    }


def rho_txt(s):
    return f"{s['rho50']:.3f}" if s["rho50_readable"] else "CENSORED"


def show(tag, name, s):
    print(f"\n=== {tag} -- {name} ===")
    if s is None:
        print("  nothing on disk yet.")
        return
    print(f"  {s['n']} seed(s) {s['seeds']}, {s['n_cross']} crossing")
    print(f"  rho_50 median {rho_txt(s)}   mean_solve {s['mean_solve']:.4f}"
          + (f"   spread {s['spread_mean']:.4f}" if s["n"] > 1 else ""))
    print("  per seed: " + ", ".join(
        f"s{sd} " + (f"{v:.3f}" if ok else "CENSORED") for sd, v, ok in s["per_seed"]))
    # No escape count: finding 44's plateau is an N=48 number (guard 2).
    print(f"  final train NLL: " + " / ".join(f"{v:.4f}" for v in s["nll"])
          + f"   [the N=96 control band to date is "
            f"{CONTROL_NLL_BAND[0]:.4f}-{CONTROL_NLL_BAND[1]:.4f}; no escape count is")
    print(f"   quoted at N=96 -- its plateau level has never been measured]")
    print(f"  n_params (TOTAL, incl. embeddings) {s['n_params']:,}")


def score_f1(st):
    """F1 gates everything: an unmeasured rho_50 cannot be compared to a level."""
    print("\nF1 -- IT TRAINS AT ALL: registered 3 of 3 stripped seeds produce a "
          "MEASURED rho_50")
    if st is None:
        print("  no stripped cell on disk. Nothing to score.")
        return "absent"
    n_cens = st["n"] - st["n_cross"]
    print(f"  observed: {st['n_cross']} of {st['n']} crossing, {n_cens} censored "
          f"at the top of a grid whose floor is {GRID_FLOOR_96}")
    if st["n"] < 3:
        print(f"  INCOMPLETE: {st['n']} of 3 seeds. Not scored -- the registered "
              f"rule is median-vs-median at three seeds (finding 51: a 2-of-3 at "
              f"50 epochs was ONE UNFINISHED SEED).")
        return "incomplete"
    if n_cens == 0:
        print("  F1 HELD. F0 is readable as a level comparison.")
        return "held"
    print(f"  F1 FALSIFIED -- AND THE BREAKING POINT IS THE RESULT, not a failed run.")
    if n_cens == 3:
        print("  All three censored: this is an ARCHITECTURE statement. The headline")
        print("  becomes 'the attention-only stack that works at N=48 does not train")
        print("  at N=96 at this budget', and F0 is not quoted as a level.")
    else:
        print(f"  {n_cens} of 3 censored: this is finding 51's seed story, not yet an")
        print(f"  architecture statement. Read the other seeds on mean_solve "
              f"({st['mean_solve']:.4f}) and do NOT quote a median level.")
    return "censored"


def score_f0(st, ct, f1_call, f2_call):
    print(f"\nF0 -- THE HEADLINE: registered |stripped - control| <= s*_96 = {S96:.3f}")
    print(f"  s*_96 is frozen from the three two-seed N=96 rungs that existed at")
    print(f"  registration (" + ", ".join(f"{k} {v:.3f}" for k, v in S96_PARTS.items())
          + f"). s* = {S48} is an N=48 number and is NOT used here.")
    if st is None or ct is None:
        print("  one arm is missing. Not scored.")
        return
    if f1_call in ("censored", "incomplete", "absent"):
        print(f"  NOT SCORED AS A LEVEL: F1 came back '{f1_call}'. A censored or")
        print(f"  unfinished stripped arm has no median to compare. See F1.")
        return
    if not ct["rho50_readable"]:
        print("  NOT SCORED: the control itself is censored.")
        return
    d = st["rho50"] - ct["rho50"]
    print(f"  observed: stripped {st['rho50']:.3f} vs control {ct['rho50']:.3f} "
          f"over {ct['n']} control seed(s)")
    print(f"  difference {d:+.3f} = {abs(d) / S96:.2f} s*_96"
          f"   [spreads {st['spread_mean']:.4f} / {ct['spread_mean']:.4f}]")
    if abs(d) <= S96:
        print(f"  F0 HELD -- NOT SEPARABLE. The stripped arm reaches the complete")
        print(f"  model's rho_50 at a second problem size. The claim is "
              f"{ct['n_params'] / st['n_params']:.1f}x fewer parameters")
        print(f"  (total, {ct['n_params']:,} -> {st['n_params']:,}) at no separable "
              f"cost, AT TWO PROBLEM SIZES -- never 'better'.")
    elif d > 0:
        print(f"  F0 FALSIFIED, DIRECTION (HIGHER): THE N=48 MINIMUM DOES NOT TRANSFER.")
        print(f"  The MLP, the width, or both are load-bearing at N=96. Every")
        print(f"  'smallest architecture' claim in this programme is an N=48 claim")
        print(f"  and must be RELABELLED as one. This is the registered outcome for")
        print(f"  this direction and is reported as a result, not a disappointment.")
    else:
        print(f"  F0 FALSIFIED, DIRECTION (LOWER): DO NOT CELEBRATE. The registration")
        print(f"  reads this as evidence the N=96 CONTROL IS UNDERTRAINED at 375k")
        print(f"  steps, not that stripping helps. The next move is a step-budget")
        print(f"  check on the control, not a victory lap.")
    if f2_call == "falsified":
        print(f"  CAVEAT FROM F2: the control is seed-dominated, so this verdict")
        print(f"  survives only as an ORDERING. The LEVELS ARE NOT QUOTABLE.")


def score_f2(ct):
    lo, hi = CONTROL_BRACKET
    print(f"\nF2 -- THE CONTROL IS NOT ONE DRAW: registered third seed inside "
          f"[{lo:.3f}, {hi:.3f}]")
    if ct is None:
        print("  no control cell on disk.")
        return "absent"
    if ct["n"] < 3:
        print(f"  {ct['n']} of 3 control seeds on disk. The third has not landed; "
              f"not scored.")
        return "incomplete"
    new = [(sd, v, ok) for sd, v, ok in ct["per_seed"] if sd == 2]
    if not new:
        print("  the new seed (2) is not among the control cells found. Check SCAN_F.")
        return "absent"
    sd, v, ok = new[0]
    if not ok:
        print(f"  seed {sd} CENSORED -- outside the bracket by construction. F2 "
              f"FALSIFIED.")
        return "falsified"
    inside = lo <= v <= hi
    print(f"  observed: seed {sd} at {v:.3f}, bracket [{lo:.3f}, {hi:.3f}] -- "
          f"{'INSIDE' if inside else 'OUTSIDE'}")
    shift = ct["rho50"] - CONTROL_2SEED
    print(f"  3-seed control median {ct['rho50']:.3f} vs the 2-seed {CONTROL_2SEED:.3f}"
          f": {shift:+.3f} = {abs(shift) / S96:.2f} s*_96")
    if inside and abs(shift) <= S96:
        print("  F2 HELD. The control's level is quotable at three seeds.")
        return "held"
    print("  F2 FALSIFIED. The N=96 control is SEED-DOMINATED at two seeds, so F0")
    print("  survives only as an ORDERING and the levels are not quotable. This was")
    print("  named at registration as the most likely way F ends up weaker than it")
    print("  looks, which is exactly why task 3 was in the job.")
    return "falsified"


def score_f3(st, ct):
    desc, tr, rho = N48_MINIMUM
    hdesc, htr, hrho = N48_HUB
    print(f"\nF3 -- THE N=48 STANDING MINIMUM DOES NOT MOVE ON N=96 EVIDENCE")
    print(f"  standing (finding 53, rule H3 = I3 = J3): {desc}, {tr:,} trainable "
          f"at rho_50 {rho:.3f},")
    print(f"  against the N=48 hub {hdesc}, {htr:,} at {hrho:.3f}. Both are N=48")
    print(f"  numbers on the N=48 grid (floor 0.33) and are NOT comparable to "
          f"anything above.")
    print(f"  F3 is a REGISTRATION, not a measurement: F can CORROBORATE or")
    print(f"  UNDERMINE the N=48 minimum; it can never replace it. No N=96 number")
    print(f"  in this file changes the standing minimum.")
    if st is not None and ct is not None and st["rho50_readable"] and ct["rho50_readable"]:
        n48 = abs(rho - hrho)
        n96 = abs(st["rho50"] - ct["rho50"])
        print(f"  FOR ORIENTATION ONLY, each against ITS OWN control and yardstick:")
        print(f"    N=48  minimum - hub      {rho - hrho:+.3f} = "
              f"{n48 / S48:.2f} s*  (s* = {S48})")
        print(f"    N=96  stripped - control {st['rho50'] - ct['rho50']:+.3f} = "
              f"{n96 / S96:.2f} s*_96  (s*_96 = {S96})")
        print(f"  The two gaps are in different units on different grids. Read them")
        print(f"  as 'separable or not' on each side, never as a ratio.")


def reestimate_s96(cells):
    """Printed alongside, NEVER substituted into a verdict (guard 1)."""
    rungs = defaultdict(list)
    for k, m in cells.items():
        rungs[k[:-1]].append(m)
    parts = []
    for key, ms in sorted(rungs.items()):
        ok = [m["rho50"] for m in ms if m["flag"] == "ok"]
        if len(ok) > 1:
            parts.append((label(key), max(ok) - min(ok), len(ok)))
    print(f"\nTHE YARDSTICK, RE-ESTIMATED (reported, never substituted)")
    print(f"  registered and used in every verdict above: s*_96 = {S96:.3f}, the mean")
    print(f"  of " + ", ".join(f"{k} {v:.3f}" for k, v in S96_PARTS.items()))
    if not parts:
        print("  nothing multi-seed on disk to re-estimate from.")
        return
    for name, v, n in parts:
        print(f"    {name:<52} {v:.3f}  ({n} seeds)")
    print(f"  mean over {len(parts)} multi-seed rung(s): "
          f"{sum(v for _, v, _ in parts) / len(parts):.3f}")
    print(f"  This is the better estimate and it is what a LATER stage should freeze")
    print(f"  its own bar from. It does not retroactively change F0, F1 or F2.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+", default=SCAN_F_DIRS)
    args = ap.parse_args()

    for d in args.scan_dir:
        assert d not in SCAN_DIRS, (
            f"{os.path.basename(d)} is in SCAN_DIRS -- an N=96 dir in the pooled "
            f"list merges N=96 cells into N=48 rungs and moves a frozen bar "
            f"(guard 3 in this file's docstring). Take it out before running "
            f"anything.")

    print("STAGE F (51656823) -- does the minimal architecture replicate at N=96?")
    print(f"  registration: scripts/slurm/sudoku2_n96_minarch_F_2026-11-09.sbatch "
          f"(217d052)")
    print(f"  yardstick s*_96 = {S96:.3f}, FROZEN. s* = {S48} is an N=48 number and "
          f"is not used.")
    print(f"  no escape count is quoted: finding 44's 1.64 plateau is an N=48 number")
    print(f"  and N=96's own plateau level has never been measured.")
    print(f"  sizes are TOTAL n_params on both sides -- n_trainable_params does not")
    print(f"  exist in the old control artifacts (guard 4).")

    cells = collect(args.scan_dir, args.boot)
    print(f"\n{len(cells)} leak-free N=96 cell(s) read from {len(args.scan_dir)} dirs")

    st_ms, ct_ms = rung(cells, STRIPPED), rung(cells, CONTROL)
    st = summarize(st_ms) if st_ms else None
    ct = summarize(ct_ms) if ct_ms else None

    show("THE REPLICATE", "no_mlp L=16 d192 at N=96", st)
    if st is not None:
        print(f"  registered at {PREDICTED_TRAINABLE_96:,} trainable / "
              f"{PREDICTED_TOTAL_96:,} total -- "
              f"{'ok' if st['n_params'] == PREDICTED_TOTAL_96 else 'MISMATCH'}")
    show("THE CONTROL", "full L=12 d256 at N=96", ct)
    if ct is not None:
        print(f"  registered at {CONTROL_TOTAL:,} total -- "
              f"{'ok' if ct['n_params'] == CONTROL_TOTAL else 'MISMATCH'}")

    f1 = score_f1(st)
    f2 = score_f2(ct)
    score_f0(st, ct, f1, f2)
    score_f3(st, ct)
    reestimate_s96(cells)


if __name__ == "__main__":
    main()
