"""Score 51013090 against its pre-registered predictions.

Reads the per-cell JSONs written by scripts.direction_ablation_rho and answers
P1-P4 from sudoku2_n96_dirablate_2026-11-03.sbatch:

  P1 the knife is sharp      ablate_next_node beats ablate_random by a margin
  P2 THE TEST                ablate_frontier_mask vs ablate_random
  P3 not just logit proximity ablate_next_color vs ablate_next_node
  P4 generality              same ordering on t04 (best cell) and t05 (worst)

Everything is read as an EXCESS OVER ablate_random, never over clean: removing
any k dimensions from a 256-dimensional stream costs something, and only the
part above the count-matched random subspace is attributable to the direction.

Two gates run before any of that, and a failure on either makes the rest
unreadable rather than negative:

  harness  clean must reproduce the 11-02 fall scan's rho_50 for the cell
           (t04 0.5635, t05 0.8290) -- same grid, same n_per_bin, same
           --seed-start, so the only honest outcome is a match.
  bite     each ablation's read-back must show its OWN probe's score collapse.
           An ablation that leaves its probe intact removed nothing, and its
           rho_50 is about whatever else those k dimensions carried.

Usage:
    python3 -m scripts.dirablate_analysis --dir $SCRATCH/spin_gnn/sudoku_n96_dirablate_2026-11-03
"""
import argparse
import glob
import json
import os

# rho_50 from the 11-02 fall scan, the number clean has to reproduce.
SCAN_RHO50 = {"t04": 0.5635, "t05": 0.8290}
# next_node top-1 at the final layer from 50979059, for the dose-response read.
SCAN_NEXT_NODE = {"t04": 0.8704, "t05": 0.6034}
ORDER = ["clean", "ablate_next_node", "ablate_frontier_mask", "ablate_next_color",
         "ablate_random"]
# Which read-back target each condition is supposed to destroy.
AIMED_AT = {"ablate_next_node": ("next_node", "top1"),
            "ablate_frontier_mask": ("frontier_mask", "auroc"),
            "ablate_next_color": ("next_color", "top1")}


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*_dirablate.json"))):
        out[os.path.basename(p).split("_")[0]] = json.load(open(p))
    return out


def rb(cell, cond, target, field):
    return cell["conditions"][cond]["readback"][target][field]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--harness-tol", type=float, default=0.03,
                    help="how far clean may sit from the fall scan's rho_50 and still "
                         "count as a reproduction; 400 instances per bin puts binomial "
                         "noise on rho_50 well inside this")
    args = ap.parse_args()

    cells = load(args.dir)
    if not cells:
        raise SystemExit(f"no *_dirablate.json under {args.dir}")
    verdicts = {}

    for tag, cell in cells.items():
        conds = cell["conditions"]
        present = [c for c in ORDER if c in conds]
        k = cell["k_used"]
        print(f"\n{'=' * 78}\n{tag}: L={cell['config']['num_layers']}, k={k}, "
              f"ablating layers {cell['args']['ablate_from']}..{cell['n_layers']}, "
              f"{len(present)}/{len(ORDER)} conditions done\n{'=' * 78}")

        print("\n-- gate 1: does clean reproduce the 11-02 fall scan? --")
        if "clean" in conds and tag in SCAN_RHO50:
            got, want = conds["clean"]["rho_50"], SCAN_RHO50[tag]
            ok = abs(got - want) <= args.harness_tol
            print(f"   clean rho_50 {got:.4f} vs scan {want:.4f}  "
                  f"({'MATCH' if ok else 'MISMATCH -- stop, the harness differs'})")
            verdicts[f"{tag}/harness"] = ok
        else:
            print("   clean not run; nothing to check the harness against")

        print("\n-- gate 2: did each ablation remove what it was aimed at? --")
        for cond in present:
            if cond not in AIMED_AT:
                continue
            target, field = AIMED_AT[cond]
            base = rb(cell, "clean", target, field) if "clean" in conds else float("nan")
            got = rb(cell, cond, target, field)
            floor = rb(cell, "ablate_random", target, field) if "ablate_random" in conds \
                else float("nan")
            print(f"   {cond:22s} {target}.{field}: clean {base:.4f} -> {got:.4f}   "
                  f"(random-subspace control leaves it at {floor:.4f})")

        print("\n-- the table. Read the LAST column: excess over the random subspace. --")
        print(f"   {'condition':22s} {'rho_50':>8} {'mean solve':>11} {'wellform':>9} "
              f"{'trunc':>7} {'excess vs random':>17}")
        rnd = conds.get("ablate_random")
        for cond in present:
            c = conds[cond]
            exc = "" if rnd is None else f"{rnd['mean_solve_rate'] - c['mean_solve_rate']:+.4f}"
            print(f"   {cond:22s} {c['rho_50']:8.4f} {c['mean_solve_rate']:11.4f} "
                  f"{c['mean_well_formed_rate']:9.3f} {c['mean_truncated_rate']:7.3f} "
                  f"{exc:>17}")
        print("   (excess > 0 = this direction costs MORE than deleting k random "
              "dimensions,\n    i.e. the model was using it. excess ~ 0 = it was not.)")

        if rnd is None:
            print("\n   ablate_random not done yet; P1-P3 need it.")
            continue
        base_sr = rnd["mean_solve_rate"]
        exc = {c: base_sr - conds[c]["mean_solve_rate"] for c in present if c in AIMED_AT}

        print("\n-- P1: is the knife sharp? --")
        if "ablate_next_node" in exc:
            v = exc["ablate_next_node"]
            verdicts[f"{tag}/P1"] = v > 0
            print(f"   ablate_next_node costs {v:+.4f} over the random subspace: "
                  f"{'PASS' if v > 0 else 'FAIL -- nothing below is readable'}")

        print("\n-- P2: THE TEST. Is the DD predicate load-bearing? --")
        if "ablate_frontier_mask" in exc:
            v = exc["ablate_frontier_mask"]
            nn = exc.get("ablate_next_node", float("nan"))
            verdicts[f"{tag}/P2"] = v > 0
            print(f"   ablate_frontier_mask costs {v:+.4f} over random, against "
                  f"{nn:+.4f} for next_node")
            if v <= 0:
                print("   -> the model REPRESENTS DD's predicate and does not USE it. The")
                print("      decodability YES and the causal NO are both real and they are")
                print("      the finding; finding 46's -0.086 predicted exactly this.")
            elif nn == nn and v < nn:
                print("   -> the predicate is used, but selection is the bigger lever "
                      f"({nn / max(1e-9, v):.1f}x).")
            else:
                print("   -> the predicate is load-bearing and at least as big a lever as "
                      "selection.")

        print("\n-- P3: is a next_node effect just logit proximity? --")
        if "ablate_next_color" in exc and "ablate_next_node" in exc:
            nc, nn = exc["ablate_next_color"], exc["ablate_next_node"]
            verdicts[f"{tag}/P3"] = nn > nc
            print(f"   next_color costs {nc:+.4f}, next_node {nn:+.4f}: "
                  f"{'PASS -- selection is specific' if nn > nc else 'FAIL -- both output-adjacent targets cost the same, so proximity explains it'}")

        if tag in SCAN_NEXT_NODE:
            print(f"\n-- dose-response: this cell's clean next_node probe was "
                  f"{SCAN_NEXT_NODE[tag]:.4f} in 50979059 --")

    tags = [t for t in ("t04", "t05") if t in cells]
    if len(tags) == 2:
        print(f"\n{'=' * 78}\n-- P4: does the ordering hold on both cells? --")
        for p in ("P1", "P2", "P3"):
            got = [verdicts.get(f"{t}/{p}") for t in tags]
            same = got[0] is not None and got[0] == got[1]
            print(f"   {p}: t04 {got[0]}  t05 {got[1]}  -> "
                  f"{'same' if same else 'DIFFERENT, so it is cell-specific'}")

    print("\n-- verdict summary --")
    for key in sorted(verdicts):
        print(f"   {key:16s} {'PASS' if verdicts[key] else 'FAIL'}")


if __name__ == "__main__":
    main()
