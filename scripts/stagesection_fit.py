"""Stage SECTION's scorer.  Stdlib only -- py-matplotlib and py-pytorch unload
each other on this cluster, so this must import neither.

Scores S1-S5 and E1-E3 exactly as registered in
scripts/slurm/sudoku2_n48_section_2026-11-20.sbatch, which was committed before
the probe ran.  Read that header for the predictions, their falsifiers, and the
disclosure of what the one-run smoke had already shown.

Usage:  python3 scripts/stagesection_fit.py [--scan DIR]
"""
import argparse
import csv
import glob
import math
import os
from collections import defaultdict

SCRATCH = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN = os.path.join(SCRATCH, "spin_gnn/runs/stagesection_2026-11-20")
GATE = 0.740
N_NODES = 48
DEDUCE = ("reason_node", "reason_color")
STRUCTURAL = ("graph_lead", "round_mark", "term", "begin")
LOGN = math.log(N_NODES)

# Registered bands.  S1: deduction share of the loss.  S2: count of runs with
# an exact colour step.  S3: the ordering comparison.  S4: gate separation.
# S5: how close the dominant section sits to its information-theoretic ceiling.
S1_HI_RHO, S1_LO_RHO = 0.02, 0.10
S1_FALSIFIER = 0.25
S2_BAR, S2_FALSIFIER = 70, 50
S2_ACC = 0.99
S3_SLACK = 0.05
S4_SEC_BAR, S4_TOT_BAR = 0.05, 0.02
S5_WINDOW, S5_FALSIFIER = 0.5, 2.5


def load(scan):
    rows = []
    for path in sorted(glob.glob(os.path.join(scan, "section_*of*.csv"))):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                for k in ("rho50_pub", "rho", "nll_sum", "nll", "accuracy",
                          "total_nll", "total_accuracy"):
                    r[k] = float(r[k]) if r[k] != "" else None
                for k in ("n_params", "gate_pass", "n_instances", "n_tokens",
                          "n_correct", "total_tokens", "e1_bad"):
                    r[k] = int(r[k])
                rows.append(r)
    return rows


def by_run_rho(rows):
    """(run_id, rho) -> {section: row}, plus the per-cell totals carried on
    every row of that group."""
    out = defaultdict(dict)
    for r in rows:
        out[(r["run_id"], r["rho"])][r["section"]] = r
    return out


def spearman(xs, ys):
    """Rank correlation with average ranks for ties; stdlib only."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def deduce_share(g):
    """Fraction of the instance's total NLL spent on the reasoning section's
    NODE and COLOR slots.  Uses nll_sum, not nll, because the sections have
    wildly different token counts and a mean of means would be meaningless."""
    tot = sum(g[s]["nll_sum"] for s in g)
    ded = sum(g[s]["nll_sum"] for s in DEDUCE if s in g)
    return ded / tot if tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=SCAN)
    args = ap.parse_args()

    rows = load(args.scan)
    if not rows:
        print("no rows under %s" % args.scan)
        return
    g = by_run_rho(rows)
    rhos = sorted({r["rho"] for r in rows})
    runs = sorted({r["run_id"] for r in rows})
    print("stage SECTION: %d rows, %d runs, rhos %s"
          % (len(rows), len(runs), ", ".join("%.2f" % x for x in rhos)))

    # ---- E1/E2/E3, first, because every S below is void if the labels are wrong.
    e1 = sum(r["e1_bad"] for r in rows if r["section"] == "begin")
    e2_bad, e2_max = 0, 0.0
    for (run, rho), sec in g.items():
        tot_nll = sum(v["nll_sum"] for v in sec.values())
        tot_tok = sum(v["n_tokens"] for v in sec.values())
        want = list(sec.values())[0]["total_nll"]
        d = abs(tot_nll / tot_tok - want)
        e2_max = max(e2_max, d)
        if d > 1e-6:
            e2_bad += 1
    e3_bad = [(run, rho, s, sec[s]["accuracy"])
              for (run, rho), sec in g.items() for s in STRUCTURAL
              if s in sec and sec[s]["n_tokens"] and sec[s]["accuracy"] < 1.0]
    print("\nE1 label-walk vs encoder: %d bad %s" % (e1, "PASS" if e1 == 0 else "FAIL"))
    print("E2 section NLLs reconstruct the total: %d of %d cells off, max %.2e  %s"
          % (e2_bad, len(g), e2_max, "PASS" if e2_bad == 0 else "FAIL"))
    print("E3 structural positions free: %d violations  %s"
          % (len(e3_bad), "PASS" if not e3_bad else "FAIL"))
    for v in e3_bad[:5]:
        print("     %s rho %.2f %s acc %.4f" % v)

    # ---- the per-section table, medians over runs, one block per density.
    sections = [s for s in ("hint_node", "hint_color", "graph_lead",
                            "graph_nbr_novel", "graph_nbr_copy", "round_mark",
                            "reason_node", "reason_color", "term", "begin")
                if any(r["section"] == s for r in rows)]
    for rho in rhos:
        print("\n-- rho %.2f, median over %d runs" % (rho, len(runs)))
        print("   %-17s %7s %9s %9s %9s" % ("section", "ntok", "nll", "acc", "share"))
        tot = median([sum(v["nll_sum"] for v in g[(r, rho)].values())
                      for r in runs if (r, rho) in g])
        for s in sections:
            cells = [g[(r, rho)][s] for r in runs
                     if (r, rho) in g and s in g[(r, rho)] and g[(r, rho)][s]["n_tokens"]]
            if not cells:
                continue
            share = median([c["nll_sum"] for c in cells]) / tot if tot else float("nan")
            print("   %-17s %7d %9.4f %9.4f %9.4f"
                  % (s, median([c["n_tokens"] for c in cells]),
                     median([c["nll"] for c in cells]),
                     median([c["accuracy"] for c in cells]), share))

    # ---- S1: the deduction share of the loss.
    print("\nS1 deduction share of total NLL (reason_node + reason_color)")
    for rho in rhos:
        vals = [deduce_share(g[(r, rho)]) for r in runs if (r, rho) in g]
        m = median(vals)
        bar = S1_HI_RHO if rho >= 0.85 else S1_LO_RHO
        fal = m > S1_FALSIFIER
        print("   rho %.2f: median %.4f  max %.4f  (bar < %.2f) %s%s"
              % (rho, m, max(vals), bar, "HIT" if m < bar else "MISS",
                 "  FALSIFIED" if fal else ""))

    # ---- S2: exactness of the colour step, run count.
    print("\nS2 reason_color accuracy >= %.2f" % S2_ACC)
    for rho in rhos:
        ok = [r for r in runs if (r, rho) in g
              and "reason_color" in g[(r, rho)]
              and g[(r, rho)]["reason_color"]["n_tokens"]
              and g[(r, rho)]["reason_color"]["accuracy"] >= S2_ACC]
        n = len([r for r in runs if (r, rho) in g])
        v = "HIT" if len(ok) >= S2_BAR else "MISS"
        print("   rho %.2f: %d of %d runs (bar %d, falsifier < %d) %s%s"
              % (rho, len(ok), n, S2_BAR, S2_FALSIFIER, v,
                 "  FALSIFIED" if len(ok) < S2_FALSIFIER else ""))

    # ---- S3: does the section metric order architectures better?
    print("\nS3 |Spearman(acc, rho50)| -- section metric vs the published total")
    for rho in rhos:
        keep = [r for r in runs if (r, rho) in g
                and "reason_node" in g[(r, rho)] and g[(r, rho)]["reason_node"]["n_tokens"]]
        if len(keep) < 10:
            print("   rho %.2f: only %d runs, UNSCOREABLE" % (rho, len(keep)))
            continue
        y = [g[(r, rho)]["reason_node"]["rho50_pub"] for r in keep]
        sec = abs(spearman([g[(r, rho)]["reason_node"]["accuracy"] for r in keep], y))
        tot = abs(spearman([g[(r, rho)]["reason_node"]["total_accuracy"] for r in keep], y))
        v = "HIT" if sec >= tot - S3_SLACK else "MISS"
        print("   rho %.2f: reason_node %.4f vs total %.4f  (n=%d) %s"
              % (rho, sec, tot, len(keep), v))

    # ---- S4: gate separation on the section metric but not on the total.
    print("\nS4 gate separation, median(pass) - median(fail)")
    for rho in rhos:
        keep = [r for r in runs if (r, rho) in g
                and "reason_node" in g[(r, rho)] and g[(r, rho)]["reason_node"]["n_tokens"]]
        if len(keep) < 10:
            continue
        p = [r for r in keep if g[(r, rho)]["reason_node"]["gate_pass"]]
        f = [r for r in keep if not g[(r, rho)]["reason_node"]["gate_pass"]]
        if not p or not f:
            print("   rho %.2f: %d pass / %d fail, UNSCOREABLE" % (rho, len(p), len(f)))
            continue
        d_sec = (median([g[(r, rho)]["reason_node"]["accuracy"] for r in p])
                 - median([g[(r, rho)]["reason_node"]["accuracy"] for r in f]))
        d_tot = (median([g[(r, rho)]["reason_node"]["total_accuracy"] for r in p])
                 - median([g[(r, rho)]["reason_node"]["total_accuracy"] for r in f]))
        v = "HIT" if (d_sec >= S4_SEC_BAR and abs(d_tot) <= S4_TOT_BAR) else "MISS"
        print("   rho %.2f: reason_node %+.4f (bar >= %.2f), total %+.4f (bar <= %.2f), "
              "%d pass / %d fail %s"
              % (rho, d_sec, S4_SEC_BAR, d_tot, S4_TOT_BAR, len(p), len(f), v))

    # ---- S5: distance from the novel-slot ceiling, which is 109's whole premise.
    print("\nS5 graph_nbr_novel NLL vs log N = %.4f (window %.2f, falsifier < %.2f)"
          % (LOGN, S5_WINDOW, S5_FALSIFIER))
    for rho in rhos:
        vals = [g[(r, rho)]["graph_nbr_novel"]["nll"] for r in runs
                if (r, rho) in g and "graph_nbr_novel" in g[(r, rho)]]
        m = median(vals)
        v = "HIT" if abs(m - LOGN) <= S5_WINDOW else "MISS"
        print("   rho %.2f: median %.4f  min %.4f  gap %+.4f  %s%s"
              % (rho, m, min(vals), m - LOGN, v,
                 "  FALSIFIED" if m < S5_FALSIFIER else ""))

    # ---- the number the brief actually wants: cheapest architecture per
    # reason_node accuracy, ranked on the section metric rather than the total.
    rho_rank = min(rhos)
    keep = [r for r in runs if (r, rho_rank) in g
            and "reason_node" in g[(r, rho_rank)] and g[(r, rho_rank)]["reason_node"]["n_tokens"]]
    keep.sort(key=lambda r: -g[(r, rho_rank)]["reason_node"]["accuracy"])
    print("\nTop 12 by reason_node accuracy at rho %.2f (the minimality ranking "
          "this stage exists to produce)" % rho_rank)
    print("   %-6s %-26s %10s %9s %9s %9s"
          % ("run", "cell", "params", "reas_node", "total_acc", "rho50"))
    for r in keep[:12]:
        c = g[(r, rho_rank)]["reason_node"]
        print("   %-6s %-26s %10d %9.4f %9.4f %9.4f"
              % (r, c["cell"][:26], c["n_params"], c["accuracy"],
                 c["total_accuracy"], c["rho50_pub"]))


if __name__ == "__main__":
    main()
