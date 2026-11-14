"""Every number the stage-X deck quotes, printed as markdown, from the
artifacts the jobs wrote.

This project's largest defect class is a number typed by hand into a page.
Fifty-six read-back defects across six decks on 2026-11-13 included two claim
pages comparing different denominators and four titles their own tables
refuted. So every table in the stage-X deck is generated here and pasted
whole, and nothing is retyped:

  rho_50       from scripts.n96_fallscan_analysis.rho_50, the project's one
               implementation, on each cell's own 14-point grid. Censoring is
               reported with its flag and never silently dropped.
  guard        from scripts.ddguard_score.ratio, CONDITIONAL on clean_emits_v
               and labelled in words with its denominator beside it. The
               unconditional rate is printed too, so the pair can never be
               confused by column position -- this mistake has been made
               twice by two different scripts.
  AUROC        from the one fixed held-out set both arms were scored on, in
               one process, not from either cell's own validation draw.
  accuracy     from the last epoch of each cell's history.json.
  verdict      from scripts.stagex_score.score, the registered scorer, which
               is the only thing allowed to say whether a bar was met.

The mid-run prediction committed at f95e2b7 is printed beside the measured
rho_50 for the same reason: a prediction that is only quoted after the fact
is not a prediction.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.stagex_score as sx
from scripts.ddguard_score import ratio
from scripts.n96_fallscan_analysis import rho_50
from scripts.plot_stagex_deck import (GUARD_RUN, SCAN_S, SCAN_X, S_CELLS,
                                      X_CELLS, X_LOGS, load_arm, load_rho)


def md(rows, head):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt_rho(v, flag):
    if flag == "ok":
        return "%.3f" % v
    return ">0.95" if flag == ">top" else "<%.2f" % v


def table_rho(xr, sr, pred):
    rows = []
    for i in range(3):
        s = fmt_rho(*rho_50(sr[i])) if sr[i] else "--"
        x = fmt_rho(*rho_50(xr[i])) if xr[i] else "not yet measured"
        rows.append(["seed %d" % i, s, x])
    rows.append(["**predicted at epoch 30**", "--",
                 "**%.3f**" % pred["predicted_rho50"]])
    return md(rows, ["", "without the term", "with the term"])


def table_acc(xs, ss):
    rows = []
    for i in range(3):
        if not xs[i] or not ss[i]:
            continue
        a, b = xs[i][-1]["accuracy"], ss[i][-1]["accuracy"]
        rows.append(["seed %d" % i, "%.4f" % b, "%.4f" % a, "%+.4f" % (a - b),
                     "%.4f" % ss[i][-1]["train_nll"],
                     "%.4f" % xs[i][-1]["train_nll"]])
    return md(rows, ["", "accuracy, no term", "accuracy, with term",
                     "difference", "train NLL, no term", "train NLL, with"])


def _sweep_rows(scan, cells):
    """The one-row-per-cell summary each task writes beside its checkpoint."""
    out = []
    for c in cells:
        p = os.path.join(scan, c.split("/")[0], "sweep.csv")
        if not os.path.exists(p):
            out.append(None)
            continue
        with open(p) as f:
            rs = list(csv.DictReader(f))
        out.append(rs[0] if rs else None)
    return out


def table_easy_end(xw, sw):
    """k = number of nodes left uncoloured, 0 to 4: the end of the density
    range where stopping IS the task. The control's non-monotonicity here
    (0.070 at one node missing, 0.960 at three) sat unexplained in the
    stage-S scan and is the missing guard of finding 68."""
    rows = []
    for i in range(3):
        for tag, w in (("without the term", sw[i]), ("with the term", xw[i])):
            if not w:
                continue
            rows.append(["seed %d, %s" % (i, tag)]
                        + ["%.3f" % float(w["solve_rate_k%d" % k])
                           for k in range(5)])
    return md(rows, ["", "k=0", "k=1", "k=2", "k=3", "k=4"])


def table_whole_run(xw, sw):
    """Aggregate solve rate and the SOLVED/UNSOLVED call, which is the one
    decision that asks directly whether anything is left to derive."""
    rows = []
    for i in range(3):
        if not xw[i] or not sw[i]:
            continue
        for k, name in (("agg_solve_rate", "solve rate, whole grid"),
                        ("agg_outcome_accuracy", "solved/unsolved call"),
                        ("mean_o1_solve_rate", "solve rate, k=0..4 only")):
            rows.append(["seed %d, %s" % (i, name), "%.3f" % float(sw[i][k]),
                         "%.3f" % float(xw[i][k])])
    return md(rows, ["", "without the term", "with the term"])


def table_guard(run):
    """CONDITIONAL on clean_emits_v, which is the published number, with the
    unconditional rate printed beside it and both named in words."""
    rows = []
    for i in range(3):
        p = os.path.join(run, "guard", "stagex_s%d" % i, "skip_behavior.csv")
        if not os.path.exists(p):
            rows.append(["seed %d" % i, "not yet measured", "--", "--"])
            continue
        with open(p) as f:
            cs = list(csv.DictReader(f))
        cond, den = ratio(cs, "clean_emits_v", "corrupt_emits_v")
        uncond = sum(1 for r in cs if r["corrupt_emits_v"] in ("True", "1")) \
            / max(len(cs), 1)
        rows.append(["seed %d" % i, "%.4f" % cond, "%d" % den,
                     "%.4f" % uncond])
    # The control's denominator is NOT 350 and was hardcoded as 350 in the
    # first version of this table. 350 is the number of counterfactual cases
    # offered; the denominator is how many of them the model named cleanly,
    # which differs per cell and per arm (control 232/237/233, stage X
    # 194/175/185). Read it, never assume it.
    ctrl = os.path.join(os.path.dirname(run), "2026-11-13_ddguard", "guard")
    dens = []
    for i in range(3):
        p = os.path.join(ctrl, "S1_s%d" % i, "skip_behavior.csv")
        if not os.path.exists(p):
            dens.append("?")
            continue
        with open(p) as f:
            dens.append(str(ratio(list(csv.DictReader(f)), "clean_emits_v",
                                  "corrupt_emits_v")[1]))
    rows.append(["**control (no term)**",
                 "/".join("%.4f" % v for v in sx.S1_GUARD),
                 "/".join(dens), "--"])
    return md(rows, ["", "guard rate, conditional on naming it cleanly",
                     "cases in the denominator", "unconditional, for contrast"])


def table_auroc(both_csv, ctrl_csv):
    rows, src = [], both_csv if os.path.exists(both_csv) else ctrl_csv
    if not os.path.exists(src):
        return "the fixed-set AUROC has not been measured yet"
    with open(src) as f:
        for r in csv.DictReader(f):
            lam = float(r["determinacy_weight"] or 0.0)
            rows.append(["seed %s" % r["seed"],
                         "with the term" if lam else "without it",
                         "%.4f" % float(r["determinacy_auroc"]),
                         "%.4f" % float(r["determinacy_bce"])])
    rows.sort(key=lambda r: (r[1], r[0]))
    return md(rows, ["", "arm", "determinacy AUROC", "determinacy BCE"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-logs", action="store_true")
    a = ap.parse_args()

    xs = load_arm(SCAN_X, X_CELLS, X_LOGS, from_logs=a.from_logs)
    ss = load_arm(SCAN_S, S_CELLS)
    xr = load_rho(SCAN_X, "52285051") if os.path.isdir(SCAN_X) else [[]] * 3
    sr = load_rho(SCAN_S, "51853969")
    pred = sx.predict_rho50()

    print("## rho_50, lower is better\n")
    print(table_rho(xr, sr, pred))
    print("\n## validation token accuracy at the last epoch\n")
    print(table_acc(xs, ss))
    xw, sw = _sweep_rows(SCAN_X, X_CELLS), _sweep_rows(SCAN_S, S_CELLS)
    print("\n## solve rate with k nodes left to colour, the easy end\n")
    print(table_easy_end(xw, sw))
    print("\n## the whole run, not just the easy end\n")
    print(table_whole_run(xw, sw))

    print("\n## the guard probe, unchanged from finding 68\n")
    print(table_guard(GUARD_RUN))
    print("\n## determinacy AUROC on the one fixed held-out set\n")
    print(table_auroc(os.path.join(GUARD_RUN, "determinacy_both_arms.csv"),
                      os.path.join(SCAN_X, "determinacy_control_stageS.csv")))

    print("\n## the registered scorer's verdict\n")
    ok = all(xr) and all(xs)
    if not ok:
        print("not scoreable yet: "
              + ("no stage-X rho grid; " if not all(xr) else "")
              + ("no stage-X history" if not all(xs) else ""))
        return
    guard = []
    for i in range(3):
        p = os.path.join(GUARD_RUN, "guard", "stagex_s%d" % i,
                         "skip_behavior.csv")
        with open(p) as f:
            guard.append(ratio(list(csv.DictReader(f)),
                               "clean_emits_v", "corrupt_emits_v")[0])
    au = {}
    with open(os.path.join(GUARD_RUN, "determinacy_both_arms.csv")) as f:
        for r in csv.DictReader(f):
            au.setdefault(bool(float(r["determinacy_weight"] or 0.0)),
                          []).append(float(r["determinacy_auroc"]))
    out = sx.score([h[-1]["accuracy"] for h in xs],
                   [rho_50(c)[0] for c in xr], guard, au[True], au[False])
    print("```\n" + json.dumps(out, indent=2) + "\n```")


if __name__ == "__main__":
    main()
