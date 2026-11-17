"""The runs stage POS measures: every SUCCESSFUL (cell, seed) in the minarch
scan, plus the one extra cell hshunt named by hand.

SUCCESSFUL is hshunt's definition, not a new one: take every row under the
fifteen scan directories in `notebooks/make_minarch_web.py:ALL_DIRS`, group
them into cells with `minarch_analysis.cell_key` (a rung plus its seed, seed
last), and keep the cell if its rho = 0.50 bin has a NONZERO solve rate. That
is the same filter the rho/accuracy power-law fit used, so a hazard curve
measured here can be laid against that fit without a second population.

The count is 79 and `successful_runs()` RAISES if it is not. A silent 78 or 80
would mean a scan directory moved, a curve was rescored, or ALL_DIRS changed,
and every exponent downstream would be fitted on a different population than
the one it is compared against. Checked, not assumed.

What the 79 are: 20 `full`, 40 `no_mlp`, 11 `tied_no_mlp`, 5 `linear`,
3 `rf_mlp`; depths 3, 4, 6, 8, 12, 16. No depth-1 or depth-2 cell survives the
filter -- at rho = 0.50 they solve nothing at all -- which is why the set is
79 and not 82.

THE EIGHTIETH RUN. hshunt's first message named seven cells by hand, and one
of them, `full L=2 d256 seed 0`, has solve rate 0.000 at rho = 0.50 and so is
NOT in the 79. It is measured anyway, flagged `in79 = False`, and excluded
from anything fitted against the 79-cell population. It is worth the one extra
run: it is the only cell here that fails ABOVE rho = 0.50 (0.580 at rho = 0.95,
0.000 by rho = 0.75), so it is where a hazard curve should be steepest and
earliest, and it was explicitly asked for. Every consumer either filters on
`in79` or says it did not.

All 80 share one instance distribution -- N=48, d=5, q=4, hint_mode=removal,
fall, shuffle_adj, no shuffle_events, no scratchpad -- and one `max_len` of
512, checked here rather than assumed, because the whole design rests on
building the eval instances ONCE and reusing them for every run.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.minarch_analysis import (SCAN_C, SCAN_D, SCAN_DEFAULT, SCAN_E, SCAN_G,
                                      SCAN_H, SCAN_I, SCAN_J, SCAN_L6, cell_key,
                                      load_all, rung_label, sort_key)
from scripts.stagek_score import SCAN_K
from scripts.stagen_score import SCAN_N
from scripts.stageo_score import SCAN_O
from scripts.stagep_score import SCAN_P
from scripts.stages_score import SCAN_S
from scripts.stageu_score import SCAN_U

# Verbatim from notebooks/make_minarch_web.py. Imported from the stage scorers
# rather than hard-coded as strings so a directory that moves breaks loudly.
ALL_DIRS = [SCAN_DEFAULT, SCAN_L6, SCAN_C, SCAN_D, SCAN_E, SCAN_G, SCAN_H,
            SCAN_I, SCAN_J, SCAN_K, SCAN_N, SCAN_O, SCAN_P, SCAN_S, SCAN_U]

N_SUCCESSFUL = 79

# The eval grid every rho curve in the scan was measured on, and therefore the
# grid stage POS must use for its hazard to be comparable point by point.
RHO_GRID = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.33]
N_PER_BIN = 400
SEED_START = 60_000_000_000  # rho_transition_eval's default --seed-start

# The seven hshunt named, by checkpoint-directory suffix. Order is hshunt's.
SEVEN_SUFFIXES = [
    "sudoku_n48_minarch_2026-11-04/t00_L12_h04_full_s0",
    "sudoku_n48_minarch_2026-11-04/t06_L06_h04_full_s0",
    "sudoku_n48_minarch_2026-11-04/t27_L03_h04_full_s0",
    "sudoku_n48_minarch_2026-11-04/t30_L02_h04_full_s0",
    "sudoku_n48_minarch_J_2026-11-05/t01_d0192_ff01024_h04_no_mlp_L16_s0",
    "sudoku_n48_tiedwidth_S_2026-11-10/t00_d0320_ff01024_h04_tied_no_mlp_L16_s0",
    # "one mid-accuracy no_mlp cell": the no_mlp seed-0 family spans val token
    # accuracy 0.4558 to 0.4948 and this sits at 0.4795, nearest the midpoint,
    # and comes from the SAME scan and epoch budget as the four `full` cells
    # above, so the architecture is the only thing that differs.
    "sudoku_n48_minarch_2026-11-04/t21_L04_h04_no_mlp_s0",
]

INSTANCE_FIELDS = ["n_nodes", "degree", "n_colors", "hint_mode", "scratchpad",
                   "shuffle_adj", "shuffle_events", "fall"]


def _config(path):
    with open(os.path.join(path, "config.json")) as fh:
        return json.load(fh)


def _val_accuracy(path):
    with open(os.path.join(path, "history.json")) as fh:
        return float(json.load(fh)[-1]["accuracy"])


def successful_runs(verbose=True):
    """The 79 successful cells plus the flagged eighth-of-seven, as dicts.

    Each dict: run_id (stable, sorted by minarch_analysis.sort_key), cell
    (rung_label), key (the cell key), seed, path, arch_mode, num_layers,
    d_model, n_trainable_params, val_acc, solve50, in79, in_seven.
    """
    rows = load_all(ALL_DIRS)
    cells = collections.defaultdict(list)
    for r in rows:
        cells[cell_key(r)].append(r)

    keep = {}
    for key, rs in cells.items():
        at50 = [r for r in rs if abs(float(r["rho"]) - 0.50) < 1e-9]
        if len(at50) == 1 and float(at50[0]["solve_rate"]) > 0:
            keep[key] = (rs, float(at50[0]["solve_rate"]))

    if verbose:
        print("scan dirs            %d" % len(ALL_DIRS))
        print("curve rows           %d" % len(rows))
        print("cells                %d" % len(cells))
        print("SUCCESSFUL (solve rate at rho=0.50 > 0)   %d" % len(keep))
    if len(keep) != N_SUCCESSFUL:
        raise SystemExit(
            "expected %d successful runs, found %d. The population the "
            "rho/accuracy power law was fitted on has changed; stopping rather "
            "than measuring a different set under the same name."
            % (N_SUCCESSFUL, len(keep)))

    out = []
    for key in sorted(keep, key=sort_key):
        rs, solve50 = keep[key]
        path = rs[0]["path"]
        cfg = _config(path)
        out.append({
            "key": key, "cell": rung_label(*key[:-1]).strip(), "seed": int(key[-1]),
            "path": path, "arch_mode": cfg.get("arch_mode", "full"),
            "num_layers": int(cfg["num_layers"]), "d_model": int(cfg["d_model"]),
            "n_trainable_params": int(cfg.get("n_trainable_params") or 0),
            "val_acc": _val_accuracy(path), "solve50": solve50,
            "in79": True, "in_seven": False, "max_len": int(cfg["max_len"]),
        })

    # A checkpoint path is <scan dir>/<task dir>/<tag dir>; the suffixes name
    # the task dir, so this is containment, not endswith.
    def _named(path, suf):
        return (os.path.normpath(suf) + os.sep) in (os.path.normpath(path) + os.sep)

    by_suffix = {}
    for rec in out:
        for suf in SEVEN_SUFFIXES:
            if _named(rec["path"], suf):
                rec["in_seven"] = True
                by_suffix[suf] = rec

    # The named cells that the success filter drops -- appended, flagged, and
    # never pooled with the 79. Resolved against the scan rows so the path,
    # the cell key and the seed come from the same place as every other run.
    for suf in SEVEN_SUFFIXES:
        if suf in by_suffix:
            continue
        hit = [(k, rs) for k, rs in cells.items() if _named(rs[0]["path"], suf)]
        if len(hit) != 1:
            raise SystemExit("cannot resolve named cell %s (%d matches)" % (suf, len(hit)))
        key, rs = hit[0]
        path = rs[0]["path"]
        cfg = _config(path)
        at50 = [r for r in rs if abs(float(r["rho"]) - 0.50) < 1e-9]
        out.append({
            "key": key, "cell": rung_label(*key[:-1]).strip(), "seed": int(key[-1]),
            "path": path, "arch_mode": cfg.get("arch_mode", "full"),
            "num_layers": int(cfg["num_layers"]), "d_model": int(cfg["d_model"]),
            "n_trainable_params": int(cfg.get("n_trainable_params") or 0),
            "val_acc": _val_accuracy(path),
            "solve50": float(at50[0]["solve_rate"]) if at50 else float("nan"),
            "in79": False, "in_seven": True, "max_len": int(cfg["max_len"]),
        })
        if verbose:
            print("NAMED BUT NOT SUCCESSFUL, measured anyway and flagged in79=False: "
                  "%s  solve@0.50=%.3f" % (out[-1]["cell"], out[-1]["solve50"]))

    for i, rec in enumerate(out):
        rec["run_id"] = "r%02d" % i

    # The design builds the eval instances once and reuses them for every run,
    # which is only sound if every run was trained on the same distribution and
    # decodes with the same budget.
    sigs = {tuple(str(_config(r["path"]).get(f)) for f in INSTANCE_FIELDS) for r in out}
    if len(sigs) != 1:
        raise SystemExit("runs disagree on the instance distribution: %r" % (sigs,))
    mls = {r["max_len"] for r in out}
    if mls != {512}:
        raise SystemExit("runs disagree on max_len: %r -- the decode budget is "
                         "per-run and cannot be shared" % (mls,))
    missing = [r["path"] for r in out if not os.path.exists(os.path.join(r["path"], "model.pt"))]
    if missing:
        raise SystemExit("missing model.pt for %d runs, first: %s" % (len(missing), missing[0]))

    if verbose:
        n7 = sum(r["in_seven"] for r in out)
        print("TOTAL RUNS TO MEASURE %d  (%d successful + %d named-but-not)"
              % (len(out), sum(r["in79"] for r in out), sum(not r["in79"] for r in out)))
        print("of which hshunt named by hand: %d" % n7)
        arch = collections.Counter(r["arch_mode"] for r in out if r["in79"])
        print("the 79 by arch: %s" % dict(sorted(arch.items())))
    return out


if __name__ == "__main__":
    runs = successful_runs()
    print()
    print("%-5s %-34s %-5s %-6s %-9s %-7s %-7s %-5s %s"
          % ("id", "cell", "seed", "arch", "ntp", "val_acc", "s@0.50", "in79", "named"))
    for r in runs:
        print("%-5s %-34s %-5d %-6s %-9d %-7.4f %-7.3f %-5s %s"
              % (r["run_id"], r["cell"], r["seed"], r["arch_mode"][:6],
                 r["n_trainable_params"], r["val_acc"], r["solve50"],
                 r["in79"], "*" if r["in_seven"] else ""))
