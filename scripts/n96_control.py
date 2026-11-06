"""What N=96 evidence already exists, read off disk, for stage F to be paired to.

    python3 -m scripts.n96_control [--boot 2000]

WHY THIS FILE EXISTS. Stage F replicates the surviving minimal architecture at
N=96 and needs a PAIRED complete-model control at the same budget and the same
eval -- nothing at N=96 is comparable to the N=48 ladder. It turns out most of
that control is already on disk and does not need re-running: eight cells in
two dirs were trained leak-free (`shuffle_adj` true AND `fall` true) at the
SAME budget the N=48 minarch grid uses (240k instances x 50 epochs, batch 32,
no warmup) and evaluated on the IDENTICAL 14-point x 400-instance hint-density
grid, realized_rho matching the N=48 files point for point.

THIS IS A READER, NOT A SCORER. Stage F is not registered yet and its subject
depends on stage J, so nothing here predicts or scores anything. It prints what
exists so that F's header can be written against measured numbers.

THREE TRAPS, and they are why this is a separate module rather than another
scan dir:

  1. KEY COLLISION. The N=96 fall cells carry d256 / ff1024 / warmup 0 / 50
     epochs, every one of them a DEFAULT, so `rung_key` reduces them to plain
     (12, 4, 'full') -- THE SAME KEY AS THE N=48 FULL L=12 RUNG. Adding an N=96
     dir to SCAN_DIRS would silently merge two different problems into one rung
     and move a bar that finished verdicts were measured against. The key below
     carries `n_nodes` and `lr_schedule` for exactly that reason, and the
     assertion in main() refuses to run if an N=96 dir ever reaches SCAN_DIRS.

  2. `n_trainable_params` IS ABSENT from every N=96 artifact -- the field was
     added at N=48 on 11-04. `cell_metrics` falls back to `n_params`, which is
     the TOTAL count including embeddings, so the sizes printed below are NOT
     the same quantity as the trainable counts quoted throughout the N=48 work.
     Stage F must record both or compare N=96 against N=96 only.

  3. THE 1.64 PLATEAU IS AN N=48 NUMBER (finding 44) AND IS NOT REUSED HERE.
     Every leak-free N=96 cell lands between 1.7456 and 1.8320 final train NLL,
     above that level, and that does not mean eight cells failed to train -- it
     means the task is harder and the plateau level at N=96 has never been
     measured. Final train NLL is printed; no escape count is.
"""
import argparse
import os
from collections import defaultdict

from scripts.minarch_analysis import (SCAN_DIRS, cell_metrics, history, load_all,
                                      median, spread, _bound_median, _config)

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
# Leak-free and task-compatible with the N=48 grid: shuffle_adj AND fall, the
# same 240k x 50 budget, the same 14 x 400 eval.
SCAN_N96_FALL = os.path.join(_SCR, "spin_gnn", "sudoku_n96_fall_scan_2026-11-02")
SCAN_N96_COSINE = os.path.join(_SCR, "spin_gnn", "sudoku_n96_cosine_2026-11-03")
SCAN_N96 = [SCAN_N96_FALL, SCAN_N96_COSINE]
# The other nine N=96 dirs are NOT usable as a control: seven cells have
# shuffle_adj false and five have it absent entirely, so their solve rates are
# decoding scores rather than deduction scores (findings 16/17/23), and all of
# them ran the old 80k or 20k x 150-epoch budget with no `fall` ensemble.
N96_UNUSABLE = ("baseline_2026-10-25", "layer_scan_2026-10-26",
                "plateau_2026-10-26", "stack_2026-10-26", "ff256_2026-10-27",
                "L07_2026-10-28", "warmup_2026-10-28")

HUB48_RHO50, HUB48_MEAN = 0.592, 0.5389   # full L=6 d256 at N=48, for orientation only
BEST48_RHO50 = 0.385                      # full L=12 at N=48, the best solver


def n96_key(r):
    """Rung key that cannot collide with an N=48 rung: n_nodes and schedule in."""
    cfg = _config(r["path"])
    return (int(r["n_nodes"]), int(r["num_layers"]), int(r["nhead"]),
            r["arch_mode"], int(r["d_model"]), int(r["dim_feedforward"]),
            int(r.get("warmup_epochs") or 0), int(r.get("epochs") or 0),
            int(cfg.get("n_train_instances") or 0),
            cfg.get("lr_schedule") or "constant")


def label(key):
    (n, L, h, arch, d, ff, w, ep, ninst, sched) = key
    extra = "".join((f" w{w:02d}" if w else "", f" {sched}" if sched != "constant" else ""))
    return (f"N={n} L={L:<2d} h={h} {arch} d{d} ff{ff} "
            f"{ninst // 1000}k x {ep}ep{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan-dir", nargs="+", default=SCAN_N96)
    args = ap.parse_args()

    for d in args.scan_dir:
        assert d not in SCAN_DIRS, (
            f"{os.path.basename(d)} is in SCAN_DIRS -- an N=96 dir in the pooled "
            f"list merges N=96 cells into N=48 rungs (trap 1 in this file's "
            f"docstring). Take it out before running anything.")

    rows = load_all(args.scan_dir)
    by = defaultdict(list)
    for r in rows:
        by[n96_key(r) + (int(r["seed"]),)].append(r)
    cells = {k: dict(cell_metrics(v, args.boot), path=v[0]["path"], seed=k[-1])
             for k, v in by.items()}

    print("N=96 CONTROL INVENTORY -- what stage F can be paired to without "
          "re-running it")
    print(f"  {len(cells)} leak-free cells read from "
          f"{len(args.scan_dir)} dir(s); every one is the COMPLETE model")
    print(f"  (`arch_mode` is absent at N=96, so `load()` defaults it to 'full', "
          f"which is\n   correct here: the arch axis is an N=48-only cut.)")
    print(f"  sizes below are n_params (TOTAL, incl. embeddings), NOT the "
          f"trainable counts\n   the N=48 work quotes -- `n_trainable_params` "
          f"does not exist at N=96 (trap 2).")

    rungs = defaultdict(list)
    for k, m in cells.items():
        rungs[k[:-1]].append(m)

    print(f"\n{'rung':<52} {'n':>2} {'cross':>5} {'rho_50':>8} {'mean':>7} "
          f"{'spread':>7} {'floor':>6} {'train NLL':>18} {'n_params':>10}")
    print("-" * 128)
    for key in sorted(rungs, key=lambda k: (k[1], k[9], k[8])):
        ms = sorted(rungs[key], key=lambda m: m["seed"])
        ok = [m for m in ms if m["flag"] == "ok"]
        bm = _bound_median([(m["rho50"], m["flag"] == "ok") for m in ms])
        rho = f"{bm['rho50_bm']:.3f}" if bm["bm_readable"] else "CENSORED"
        nlls = [history(m["path"])[1] for m in ms]
        print(f"{label(key):<52} {len(ms):>2} {len(ok):>5} {rho:>8} "
              f"{median([m['mean_solve'] for m in ms]):>7.4f} "
              f"{spread([m['mean_solve'] for m in ms]):>7.4f} "
              f"{median([m['solve_at_floor'] for m in ms]):>6.3f} "
              + "/".join(f"{v:.4f}" for v in nlls).rjust(18)
              + f" {ms[0]['n_params']:>10,}")
        print(f"{'':<52} seeds {[m['seed'] for m in ms]}, "
              f"per-seed rho_50 "
              + ", ".join(f"{m['rho50']:.3f}" if m["flag"] == "ok" else "CENSORED"
                          for m in ms))

    print("\nWHAT THIS MEANS FOR STAGE F, stated as facts and not as a plan:")
    print("  - The complete-model control at N=96 EXISTS at the N=48 budget and")
    print("    eval. F does not need to re-run it from scratch; it needs seeds.")
    print("  - NO N=96 RUNG ANYWHERE HAS 3 SEEDS (2 at L8 and L12, 1 at L16), and")
    print("    the brief requires 3 with the spread quoted (findings 41/42/47).")
    print("    The cheapest honest F therefore adds the missing control seed(s)")
    print("    alongside the stripped arm, in the SAME job, at the same budget.")
    print("  - Measured per-cell cost at this budget on one H100 (from the")
    print("    original job logs): L8 7h42m, L12 11h24m, L16 15h03m training,")
    print("    plus 12-24 min for the 14x400 eval. A 3-seed pair at L=12 is")
    print("    ~68 GPU-h if the stripped arm costs what the complete one does,")
    print("    and less in proportion to the MLP and width it drops.")
    print("  - The 1.64 plateau is an N=48 number and is NOT applied here "
          "(trap 3);")
    print("    N=96's own plateau level has never been measured.")
    print(f"  - FOR ORIENTATION ONLY, never as a paired comparison: at N=48 the")
    print(f"    hub is rho_50 {HUB48_RHO50} / mean {HUB48_MEAN} and the best solver is")
    print(f"    {BEST48_RHO50}. Same grid, harder problem; the N and the architecture move")
    print(f"    the metric together, which is the whole reason F needs its own control.")


if __name__ == "__main__":
    main()
