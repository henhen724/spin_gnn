"""Stage CAND -- does the model compute the candidate set, or a proxy that agrees?

STATUS next-step 1 part (iii): naming only forced cells is NECESSARY for direct
deduction and not SUFFICIENT.  The models' output is consistent with having
deduced, and 98/100/101 do not distinguish that from computing something else
that agrees.  STATUS says this needs a new instrument.  Maxim 100 says a
"needs a job" claim in STATUS is a hypothesis, and that claim has been made
twice and been wrong twice.

The instrument is already on disk.  `walk_forced` records, for every rollout's
first error, the node named (`err_node`) and the cache index of the instance
(`inst`), and `instance_cache` is deterministic.  At j = 0 -- and only at
j = 0 -- the pre-state is exactly the hints, so for every recorded first error
the exact candidate set of the named cell can be rebuilt with no model:

    c(v) = q - |distinct colours on v's assigned neighbours|,   v uncoloured

which is `forced_mask`'s own arithmetic one step earlier: that function returns
(asg < 0) & (cnt == q - 1), i.e. exactly c(v) == 1.  D1 checks the
reconstruction against the scorer's own `err_in_F` on every j = 0 row, so the
two cannot drift apart.

Predictions C1-C5 + D1-D2 are in REGISTERED.md, committed at b5a1da9 before
this file existed.  No training, no GPU, no new instances, no model loaded:
a refit of stage LOGIC's CSVs plus stage POS's cached instances.  Seconds on a
login node.
"""

import collections
import csv
import glob
import os
import pickle
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line  # noqa: E402
from scripts.stageexp_fit import fnum, tobool, med  # noqa: E402
from scripts.stagegate_fit import GRID, FIT_RHOS, spearman  # noqa: E402

OUT = os.environ.get("CAND_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stagecand_2026-11-17"
LOGIC = os.environ.get("LOGIC_OUT") or \
    "/scratch/users/hshunt/spin_gnn/runs/stagelogic_2026-11-17"
INST = os.environ.get("CAND_INST") or \
    "/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17/instances"
FIGS = os.path.join(OUT, "figures")
DPI = 150
Q = 4                      # colours; c(v) in 1..Q, forced is c == 1
N_PER_BIN = 400
SEED_START_TAG = 0         # the pickles are named ..._s0.pkl
CVALS = [2, 3, 4]          # unforced candidate counts


# --------------------------------------------------------------------------
# the instance side: candidate counts, with no model and no planted colouring
# --------------------------------------------------------------------------

def hint_asg(n_nodes, hints):
    asg = np.full(n_nodes, -1, dtype=np.int8)
    for node, col in hints.items():
        asg[int(node)] = int(col)
    return asg


def cand_and_degree(adj, asg, q=Q):
    """(c, k) per node.  `c` is the number of colours still locally consistent
    and is meaningful only where asg < 0; `k` is the number of ASSIGNED
    neighbours, which is the nearest non-deductive proxy for forcedness and is
    what C5 holds fixed.  The bit arithmetic is copied from
    stagelogic_score.forced_mask so that c == 1 is exactly that function's
    True, which D1 then verifies against the scorer's own output."""
    nbr = asg[adj]
    seen = np.zeros(adj.shape[0], dtype=np.int64)
    for col in range(q):
        seen |= np.any(nbr == col, axis=1).astype(np.int64) << col
    cnt = np.zeros(adj.shape[0], dtype=np.int64)
    for col in range(q):
        cnt += (seen >> col) & 1
    return q - cnt, (nbr >= 0).sum(axis=1)


def instance_table(rho):
    """Per instance at this density: the candidate count, the assigned-neighbour
    count, and the uncoloured mask, under the HINTS alone -- which is the j = 0
    pre-state exactly."""
    path = os.path.join(INST, "inst_rho%05d_n%d_s%d.pkl"
                        % (round(rho * 10000), N_PER_BIN, SEED_START_TAG))
    with open(path, "rb") as fh:
        inst = pickle.load(fh)
    out = []
    for adj, colors, hints, trace in inst:
        asg = hint_asg(adj.shape[0], hints)
        c, k = cand_and_degree(adj, asg)
        free = asg < 0
        if np.any(c[free] < 1):
            raise SystemExit("c < 1 on an uncoloured cell at rho=%.2f -- the "
                             "planted colouring would not be proper" % rho)
        out.append((c, k, free))
    return out


# --------------------------------------------------------------------------
# the measurement side
# --------------------------------------------------------------------------

def load_hazard0():
    """j = 0 aggregates per (run, rho) from F_hazard: N, n_in_F, n_err,
    n_err_in_F, n_instances."""
    keep = ["run_id", "cell", "in79", "rho", "n_instances", "j",
            "n_at_risk", "n_in_F", "n_err", "n_err_in_F"]
    out = {}
    for path in sorted(glob.glob(os.path.join(LOGIC, "F_hazard_*.csv"))):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["j"] != "0" or not tobool(r, "in79"):
                    continue
                out[(r["run_id"], round(float(r["rho"]), 4))] = dict(
                    cell=r["cell"],
                    n_inst=int(float(r["n_instances"])),
                    N=int(float(r["n_at_risk"])),
                    n_in_F=int(float(r["n_in_F"])),
                    n_err=int(float(r["n_err"])),
                    n_err_in_F=int(float(r["n_err_in_F"])))
    return out


def load_errors0():
    """One record per rollout whose FIRST error is at j = 0."""
    out = []
    for path in sorted(glob.glob(os.path.join(LOGIC, "F_errors_*.csv"))):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["first_error_j"] != "0" or not tobool(r, "in79"):
                    continue
                out.append((r["run_id"], round(float(r["rho"]), 4),
                            int(r["inst"]), int(r["err_node"]),
                            int(r["err_in_F"]), r["first_kind"]))
    return out


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def share(counts, keys):
    tot = sum(counts.get(k, 0) for k in keys)
    return {k: (counts.get(k, 0) / tot if tot else float("nan")) for k in keys}, tot


def model_a_phat(w):
    """pi(c) ~ w(c) * c/(c-1), then P_hat = sum pi(c)/c.  Model A: the cell is
    chosen by an unknown rule, the candidate set is computed correctly, the
    colour is uniform on it.  w comes from F_errors; the M it is compared
    against comes from F_hazard.  Nothing links them but the hypothesis."""
    pi = {c: w[c] * c / (c - 1.0) for c in CVALS if w[c] == w[c]}
    z = sum(pi.values())
    if not z:
        return float("nan"), {c: float("nan") for c in CVALS}
    pi = {c: v / z for c, v in pi.items()}
    return sum(v / c for c, v in pi.items()), pi


def main():
    os.makedirs(FIGS, exist_ok=True)
    haz = load_hazard0()
    errs = load_errors0()
    runs = sorted({k[0] for k in haz})
    print("j=0 hazard cells %d  runs %d   j=0 error records %d"
          % (len(haz), len(runs), len(errs)))

    by_rho = collections.defaultdict(list)
    for e in errs:
        by_rho[e[1]].append(e)

    d1_bad, d1_tot, d1_dom, d1_bad_dom = [], 0, 0, []
    rows, strat_rows = [], []
    pooled_kwk = collections.Counter()    # k -> named-and-wrong count
    pooled_kbk = collections.Counter()    # k -> summed matched share
    pooled_kbn = 0
    pooled_w = collections.Counter()
    pooled_b = collections.Counter()      # summed matched shares, not counts
    pooled_bn = 0
    pooled_kw = collections.Counter()     # (k, c) -> count
    pooled_kb = collections.Counter()     # (k, c) -> summed matched share
    pooled_kn = collections.Counter()     # k -> matched instances contributing

    for rho in GRID:
        tab = instance_table(rho)
        cells = [v for k, v in haz.items() if k[1] == rho]
        N = sum(c["N"] for c in cells)
        n_in_F = sum(c["n_in_F"] for c in cells)
        n_err = sum(c["n_err"] for c in cells)
        n_eif = sum(c["n_err_in_F"] for c in cells)
        n_inst_mismatch = sum(1 for c in cells if c["N"] != c["n_inst"])
        U = N - n_in_F
        E_u = n_err - n_eif
        M = 1.0 - E_u / U if U else float("nan")

        wc = collections.Counter()
        bsum = collections.Counter()
        bn = 0
        kw = collections.Counter()
        kb = collections.Counter()
        kn = collections.Counter()
        kwk = collections.Counter()
        kbk = collections.Counter()
        kbn = 0
        n_invalid = n_on_forced = 0
        for run_id, _, inst, node, err_in_F, kind in by_rho.get(rho, []):
            c, k, free = tab[inst]
            d1_tot += 1
            if (int(c[node]) == 1) != bool(err_in_F):
                d1_bad.append((run_id, rho, inst, node, int(c[node]), err_in_F,
                               int(free[node])))
            if free[node]:
                # c(v) is DEFINED only for an uncoloured cell -- that is how the
                # registration defines it and how forced_mask reads it.  The
                # restricted form is the one that tests the reconstruction.
                d1_dom += 1
                if (int(c[node]) == 1) != bool(err_in_F):
                    d1_bad_dom.append((run_id, rho, inst, node, int(c[node]),
                                       err_in_F, 1))
            if not free[node]:
                n_invalid += 1          # named an already-coloured cell
                continue
            if err_in_F:
                n_on_forced += 1
                continue
            cv = int(c[node])
            wc[cv] += 1
            # matched null: this instance's own share among its uncoloured
            # unforced cells, so instance composition cannot confound it
            m = free & (c >= 2)
            tot = int(m.sum())
            if tot:
                bn += 1
                for cc in CVALS:
                    bsum[cc] += float(((c == cc) & m).sum()) / tot
            kv = int(k[node])
            kw[(kv, cv)] += 1
            # POST-HOC companion to C5: is the CELL CHOICE itself selected on k?
            kwk[kv] += 1
            if tot:
                kbn += 1
                for kk in range(6):
                    kbk[kk] += float(((k == kk) & m).sum()) / tot
            mk = m & (k == kv)
            totk = int(mk.sum())
            if totk:
                kn[kv] += 1
                for cc in CVALS:
                    kb[(kv, cc)] += float(((c == cc) & mk).sum()) / totk

        w, w_n = share(wc, CVALS)
        b = {c: (bsum[c] / bn if bn else float("nan")) for c in CVALS}
        phat, pi = model_a_phat(w)
        rows.append(dict(
            rho=rho, held_out=int(rho not in FIT_RHOS), n_cells=len(cells),
            N=N, n_inst_mismatch=n_inst_mismatch, n_in_F=n_in_F, U=U,
            n_err=n_err, n_err_in_F=n_eif, E_u=E_u, M=M,
            n_err0=len(by_rho.get(rho, [])), n_invalid=n_invalid,
            n_on_forced=n_on_forced, n_scored=w_n,
            M_lo=wilson(U - E_u, U)[0], M_hi=wilson(U - E_u, U)[1],
            w2=w[2], w3=w[3], w4=w[4], b2=b[2], b3=b[3], b4=b[4],
            gap2=w[2] - b[2], phat=phat,
            pi2=pi.get(2, float("nan")), pi3=pi.get(3, float("nan")),
            pi4=pi.get(4, float("nan")), absdiff=abs(phat - M)))
        if rho in FIT_RHOS:
            for c in CVALS:
                pooled_w[c] += wc[c]
                pooled_b[c] += bsum[c]
            pooled_bn += bn
            for key, v in kw.items():
                pooled_kw[key] += v
            for key, v in kb.items():
                pooled_kb[key] += v
            for key, v in kn.items():
                pooled_kn[key] += v
            for key, v in kwk.items():
                pooled_kwk[key] += v
            for key, v in kbk.items():
                pooled_kbk[key] += v
            pooled_kbn += kbn
        for kv in sorted(kn):
            tot = sum(kw[(kv, c)] for c in CVALS)
            strat_rows.append(dict(
                rho=rho, k=kv, n=tot,
                **{("w%d" % c): (kw[(kv, c)] / tot if tot else float("nan"))
                   for c in CVALS},
                **{("b%d" % c): (kb[(kv, c)] / kn[kv] if kn[kv] else float("nan"))
                   for c in CVALS}))
        print("  rho=%.2f  U=%6d  M=%.4f  err0=%5d  scored=%5d  w2=%.4f b2=%.4f"
              % (rho, U, M, len(by_rho.get(rho, [])), w_n, w[2], b[2]))

    pw, pw_n = share(pooled_w, CVALS)
    pb = {c: (pooled_b[c] / pooled_bn if pooled_bn else float("nan"))
          for c in CVALS}
    pphat, ppi = model_a_phat(pw)

    fit = [r for r in rows if not r["held_out"]]
    # ---- C1
    c1_n = sum(1 for r in fit if r["gap2"] > 0)
    c1_pooled = pw[2] - pb[2]
    c1 = "HIT" if (c1_pooled >= 0.10 and c1_n >= 8) else "MISS"
    c1_fals = (c1_n <= 5) or (c1_pooled < 0.02)
    # a density where EVERY unforced cell has exactly two candidates cannot
    # produce a positive gap, whatever the model does
    n_degen = sum(1 for r in fit if r["b2"] >= 0.99999)
    c1_nd = sum(1 for r in fit if r["b2"] < 0.99999 and r["gap2"] > 0)
    # ---- C2
    c2_n = sum(1 for r in fit if r["absdiff"] <= 0.08)
    c2_med = med([r["absdiff"] for r in fit])
    c2 = "HIT" if (c2_n >= 7 and c2_med <= 0.05) else "MISS"
    # ---- C3
    over = [r for r in fit if r["M"] > 0.50]
    c3_max = max([r["M"] - 0.50 for r in over], default=0.0)
    c3 = "HIT" if (1 <= len(over) <= 3 and c3_max <= 0.03) else "MISS"
    # ---- C4
    c4 = spearman([r["rho"] for r in fit], [r["gap2"] for r in fit])
    c4v = "HIT" if c4 <= -0.60 else "MISS"
    # ---- C5
    modal_k = max(set(k for k, _ in pooled_kw),
                  key=lambda kv: sum(pooled_kw[(kv, c)] for c in CVALS))
    tot_k = sum(pooled_kw[(modal_k, c)] for c in CVALS)
    w2k = pooled_kw[(modal_k, 2)] / tot_k if tot_k else float("nan")
    b2k = pooled_kb[(modal_k, 2)] / pooled_kn[modal_k] if pooled_kn[modal_k] \
        else float("nan")
    c5gap = w2k - b2k
    c5 = "HIT" if c5gap >= 0.05 else "MISS"

    for name, rs, keys in (("cand_density.csv", rows, list(rows[0])),
                           ("cand_strata.csv", strat_rows, list(strat_rows[0]))):
        with open(os.path.join(OUT, name), "w", newline="") as fh:
            w_ = csv.DictWriter(fh, fieldnames=keys)
            w_.writeheader()
            w_.writerows(rs)

    # ----------------------------------------------------------------- verdict
    L = []
    A = L.append
    A("STAGE CAND -- the candidate set at j = 0, from disk, with no model")
    A("=" * 78)
    A("j=0 hazard cells %d over %d runs; j=0 first-error records %d; q = %d"
      % (len(haz), len(runs), len(errs), Q))
    A("")
    A("c(v) = q - |distinct colours on v's assigned neighbours|, v uncoloured.")
    A("Forced is c = 1.  At j = 0 the pre-state is exactly the hints, so this")
    A("is reconstructible with no model and no rollout.")
    A("")
    A("-" * 78)
    A("D1 -- NO RECONSTRUCTION DRIFT, AND A DIAGNOSTIC THAT OUTRAN ITS DOMAIN")
    A("  AS LITERALLY REGISTERED -- (c == 1) vs the scorer's own err_in_F on")
    A("  ALL %d j=0 rows at 14 densities: %d mismatches  ->  FAILS THE WORDING"
      % (d1_tot, len(d1_bad)))
    for bad in d1_bad[:8]:
        A("    %s rho %.2f inst %d node %d  c=%d  err_in_F=%d  uncoloured=%d"
          % bad)
    A("  THE REGISTERED CONSEQUENCE DOES NOT FOLLOW.  It said a mismatch means")
    A("  the instance index or the pre-state reconstruction is wrong.  It is")
    A("  neither.  Every mismatching row has uncoloured=0: the model named an")
    A("  ALREADY-COLOURED cell.  c(v) is defined in the registration only for")
    A("  an uncoloured v, and forced_mask is False there for the separate")
    A("  reason that (asg < 0) fails, not because the cell has two candidates.")
    A("  The %d mismatches are EXACTLY the %d node_invalid emissions already"
      % (len(d1_bad), sum(r["n_invalid"] for r in rows)))
    A("  counted and excluded in D2's ledger, one each at rho 0.50/0.45/0.40.")
    A("  RESTRICTED TO ITS OWN DOMAIN -- the %d rows naming an uncoloured"
      % d1_dom)
    A("  cell -- mismatches: %d  ->  %s"
      % (len(d1_bad_dom), "PASS" if not d1_bad_dom else "FAIL"))
    A("  THE LESSON, WHICH IS NEW: a diagnostic must carry the DOMAIN of the")
    A("  quantity it checks.  I defined c(v) for uncoloured v in one paragraph")
    A("  and then registered a check over every row in the next.  Scored as")
    A("  failing the wording and passing on its domain, both reported.")
    A("")
    A("-" * 78)
    A("D2 -- THE POPULATION LEDGER, j = 0")
    A("   rho      N  N!=n_inst   n_in_F       U    err0  invalid  on_forced"
      "  scored        M")
    for r in rows:
        A("  %.2f %6d %10d %8d %7d %7d %8d %10d %7d  %7.4f%s"
          % (r["rho"], r["N"], r["n_inst_mismatch"], r["n_in_F"], r["U"],
             r["n_err0"], r["n_invalid"], r["n_on_forced"], r["n_scored"],
             r["M"], "  <- extension" if r["held_out"] else ""))
    A("  N == n_instances at every density means no rollout died before its")
    A("  first pair, which is 98's anchor restated.  j = 0 has no S=0 / S=1")
    A("  censoring: every rollout makes a first emission.")
    A("")
    A("-" * 78)
    A("C1 -- DOES THE MODEL SELECT NEAR-FORCED CELLS")
    A("  w(c): candidate counts of unforced cells NAMED AND COLOURED WRONG")
    A("  b(c): the same instances' own share among uncoloured unforced cells")
    A("   rho     w(2)    b(2)     gap     w(3)    b(3)     w(4)    b(4)")
    for r in rows:
        A("  %.2f  %7.4f %7.4f %+7.4f  %7.4f %7.4f  %7.4f %7.4f%s"
          % (r["rho"], r["w2"], r["b2"], r["gap2"], r["w3"], r["b3"],
             r["w4"], r["b4"], "  <- extension" if r["held_out"] else ""))
    A("  pooled over the ten fit densities (%d cells): w(2) %.4f  b(2) %.4f"
      "  gap %+.4f" % (pw_n, pw[2], pb[2], c1_pooled))
    A("  densities with gap > 0: %d of 10" % c1_n)
    A("  NOTE the null pushes the other way: blind choice with a")
    A("  uniform-over-candidates colour rule gives w(c) ~ b(c)(1-1/c), which")
    A("  DOWN-weights c=2 by 0.50 against 0.67 and 0.75.")
    A("  registered: pooled gap >= +0.10 AND > 0 at >= 8 of 10  ->  %s" % c1)
    A("  falsifier (gap <= 0 at >= 5, or pooled < +0.02): %s"
      % ("TRIPPED" if c1_fals else "not tripped"))
    A("  THE MAGNITUDE BAR WAS MET (%+.4f >= +0.10) AND THE COUNT BAR WAS NOT,"
      % c1_pooled)
    A("  AND THE COUNT BAR WAS PARTLY DEAD ON ARRIVAL: at rho 0.95 and 0.90")
    A("  w(2) = b(2) = 1.0000 EXACTLY -- every unforced cell at those densities")
    A("  has precisely two candidates, so the gap is identically zero by")
    A("  construction and cannot be positive.  %d of the 10 densities are"
      % n_degen)
    A("  degenerate that way, capping the achievable count at %d.  On the %d"
      % (10 - n_degen, 10 - n_degen))
    A("  non-degenerate densities the gap is positive at %d.  A MAGNITUDE" % c1_nd)
    A("  CHECKED AT A DEGENERATE POINT IS NOT CHECKED (76, 79) -- and neither")
    A("  is a COUNT.  Scored MISS as registered; the bar was mine to set.")
    A("")
    A("-" * 78)
    A("C2 -- THE OVER-IDENTIFYING TEST: P_hat FROM F_errors vs M FROM F_hazard")
    A("   rho    P_hat        M     |diff|      pi(2)   pi(3)   pi(4)")
    for r in rows:
        A("  %.2f  %7.4f  %7.4f   %7.4f    %7.4f %7.4f %7.4f%s"
          % (r["rho"], r["phat"], r["M"], r["absdiff"], r["pi2"], r["pi3"],
             r["pi4"], "  <- extension" if r["held_out"] else ""))
    A("  pooled over the ten fit densities: P_hat %.4f" % pphat)
    A("  |diff| <= 0.08 at %d of 10;  median |diff| %.4f" % (c2_n, c2_med))
    A("  registered: >= 7 of 10 within 0.08 AND median <= 0.05  ->  %s" % c2)
    A("  AND THE HIT IS WEAKER THAN IT LOOKS, WHICH IS MY REGISTRATION'S")
    A("  FAULT AND NOT THE DATA'S.  P_hat spans only %.4f over all fourteen"
      % (max(r["phat"] for r in rows) - min(r["phat"] for r in rows)))
    A("  densities, because pi(2) never leaves [%.3f, %.3f]: Model A's"
      % (min(r["pi2"] for r in rows), max(r["pi2"] for r in rows)))
    A("  prediction is effectively the CONSTANT %.4f.  A +-0.08 band around a"
      % med([r["phat"] for r in rows]))
    A("  constant is not a demanding test.  Set a band relative to the SPREAD")
    A("  of the predicted quantity, not absolutely.")
    A("  POST-HOC, NOT REGISTERED: M spans %.4f across the ten fit densities,"
      % (max(r["M"] for r in fit) - min(r["M"] for r in fit)))
    A("  %.1f times P_hat's range there, and Spearman(P_hat, M) over them is"
      % ((max(r["M"] for r in fit) - min(r["M"] for r in fit))
         / max(max(r["phat"] for r in fit) - min(r["phat"] for r in fit), 1e-9)))
    A("  %+.4f." % spearman([r["phat"] for r in fit], [r["M"] for r in fit]))
    A("  So Model A gets the LEVEL right and carries no information about the")
    A("  DENSITY DEPENDENCE, which is the part that would have been evidence.")
    A("")
    A("-" * 78)
    A("C3 -- WHERE MODEL A's CEILING BREAKS")
    A("  any rule drawing the colour from the named cell's TRUE candidate set")
    A("  has Pr(correct | unforced) <= 1/2, with equality only if every named")
    A("  unforced cell has exactly two candidates.")
    A("  fit densities with M > 0.50: %d" % len(over))
    A("     rho        M      excess   95% Wilson on M      U   strictly above?")
    for r in over:
        A("    %.2f   %7.4f    +%.4f   [%.4f, %.4f] %7d   %s"
          % (r["rho"], r["M"], r["M"] - 0.50, r["M_lo"], r["M_hi"], r["U"],
             "yes" if r["M_lo"] > 0.50 else "no -- interval covers 0.50"))
    A("  largest excess %.4f" % c3_max)
    A("  registered: 1 to 3 densities AND every excess <= 0.03  ->  %s" % c3)
    A("  the falsifier fired on the COUNT (>= 5), not the size: every excess")
    A("  is under 0.02, a fifth of the 0.06 bar.  The intervals are Wilson on")
    A("  U emissions and are NOT independent across runs -- 79 runs see the")
    A("  same 400 instances -- so read them as a floor on the uncertainty.")
    A("  DISCLOSED IN ADVANCE: STATUS already recorded M up to 0.5159, so the")
    A("  existence of an excess was not a live prediction; the count and the")
    A("  size were.")
    A("")
    A("-" * 78)
    A("C4 -- DOES THE SELECTION STRENGTHEN WITH DIFFICULTY")
    A("  Spearman(w(2) - b(2), rho) over the ten fit densities = %+.4f" % c4)
    A("  registered: <= -0.60, falsified at >= +0.30  ->  %s" % c4v)
    A("  READ IT WITH C5 AND THE HIT LOSES ITS CONTENT.  b(2) falls from")
    A("  1.0000 to %.4f across the grid while w(2) stays near %.2f, so the gap"
      % (min(r["b2"] for r in fit), med([r["w2"] for r in fit])))
    A("  widens whether the model computes candidate sets or only names cells")
    A("  with many coloured neighbours -- and C5 says it is the latter.  A")
    A("  CORRELATION WHOSE TREND IS CARRIED BY THE BASELINE IS NOT EVIDENCE")
    A("  ABOUT THE MODEL.")
    A("")
    A("-" * 78)
    A("C5 -- THE HEADLINE: HOLDING COLOURED-NEIGHBOUR COUNT FIXED")
    A("  k = number of ASSIGNED neighbours of the named cell.  The rival")
    A("  account is 'name the cell with the most coloured neighbours', which")
    A("  reproduces forcedness most of the time without propagating anything.")
    A("    k      n   w(2|k)   b(2|k)      gap")
    for kv in sorted(pooled_kn):
        tot = sum(pooled_kw[(kv, c)] for c in CVALS)
        if not tot:
            continue
        A("  %3d %6d  %7.4f  %7.4f  %+7.4f"
          % (kv, tot, pooled_kw[(kv, 2)] / tot,
             pooled_kb[(kv, 2)] / pooled_kn[kv],
             pooled_kw[(kv, 2)] / tot - pooled_kb[(kv, 2)] / pooled_kn[kv]))
    A("  modal stratum k = %d on %d cells: w(2|k) %.4f  b(2|k) %.4f  gap %+.4f"
      % (modal_k, tot_k, w2k, b2k, c5gap))
    A("  registered: modal-stratum gap >= +0.05  ->  %s" % c5)
    A("  if C1 hits and C5 misses the finding is the RIVAL, not the rule.")
    A("")
    A("  POST-HOC, NOT REGISTERED, AND IT IS WHAT MAKES THE MISS CONCLUSIVE:")
    A("  IS THE CELL CHOICE ITSELF SELECTED ON k?")
    A("    k    named w(k)   available b(k)        gap")
    for kv in sorted(pooled_kwk):
        A("  %3d      %7.4f          %7.4f   %+8.4f"
          % (kv, pooled_kwk[kv] / max(sum(pooled_kwk.values()), 1),
             pooled_kbk[kv] / max(pooled_kbn, 1),
             pooled_kwk[kv] / max(sum(pooled_kwk.values()), 1)
             - pooled_kbk[kv] / max(pooled_kbn, 1)))
    A("  The model names high-k cells far more often than blind choice would,")
    A("  and a high-k cell almost always has exactly two candidates -- at")
    A("  k = 4, w(2|k) and b(2|k) are both 1.0000, so there is nothing left")
    A("  for a candidate-set computation to explain.  C1's +0.1064 is")
    A("  ACCOUNTED FOR by selection on the number of coloured neighbours.")
    A("")
    A("=" * 78)
    A("WHAT THIS ANSWERS, AND WHAT IT LEAVES OPEN")
    A("  STATUS next-step 1 part (iii) said distinguishing deduction from")
    A("  'something else that agrees' needs a new instrument and not a refit.")
    A("  HALF OF THAT IS WRONG FOR THE THIRD TIME: the instrument was on disk.")
    A("  err_node plus a deterministic instance cache plus the j = 0 pre-state")
    A("  give the exact candidate set with no model, in six seconds.")
    A("  THE OTHER HALF IS RIGHT, AND NOW FOR A STATED REASON.  The instrument")
    A("  cannot separate the two accounts at this q and d, because the")
    A("  candidate count and the coloured-neighbour count are nearly")
    A("  COLLINEAR on these instances: an unforced cell with k = 4 has exactly")
    A("  two candidates %.2f%% of the time and one with k = 3 does %.2f%% of"
      % (100 * pooled_kb[(4, 2)] / max(pooled_kn[4], 1),
         100 * pooled_kb[(3, 2)] / max(pooled_kn[3], 1)))
    A("  the time, so 'name the cell with the most assigned neighbours'")
    A("  reproduces 'name a near-forced cell' almost exactly.  Every positive")
    A("  signal here (C1, C4) survives the collinearity and dies under the")
    A("  control (C5).")
    A("  WHAT WOULD SEPARATE THEM IS A GENERATOR CHANGE, NOT A SCORER: an")
    A("  ensemble where k and c are decorrelated -- a larger q, or instances")
    A("  selected so that high-k cells often keep three or more candidates.")
    A("  THAT genuinely needs a job, and now the reason is structural and")
    A("  stated instead of 'nothing on disk distinguishes them'.")
    A("")
    A("=" * 78)
    A("SCORECARD")
    for nm, v in (("C1 selects near-forced cells", c1),
                  ("C2 over-identifying test", c2),
                  ("C3 ceiling breaks", c3),
                  ("C4 selection vs difficulty", c4v),
                  ("C5 survives the k control", c5),
                  ("D1 reconstruction", "FAILS THE WORDING (3 of 43598) / "
                   "PASSES ON ITS DOMAIN (0 of 43595)")):
        A("  %-34s %s" % (nm, v))
    txt = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "VERDICT.txt"), "w") as fh:
        fh.write(txt)
    print("\n" + txt)

    figures(rows, fit, pw, pb, pphat, pooled_kw, pooled_kb, pooled_kn,
            modal_k, c4, pooled_kwk, pooled_kbk, pooled_kbn)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def _save(fig, name):
    fig.savefig(os.path.join(FIGS, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figures(rows, fit, pw, pb, pphat, kw, kb, kn, modal_k, c4,
            kwk, kbk, kbn):
    fr = [r["rho"] for r in fit]

    # C1 -- w(c) against b(c), pooled
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(CVALS))
    ax.bar(x - 0.19, [pb[c] for c in CVALS], 0.36, color=MUTED,
           label="_", rasterized=True)
    ax.bar(x + 0.19, [pw[c] for c in CVALS], 0.36, color=SERIES[0],
           label="_", rasterized=True)
    ax.set_xticks(x)
    ax.set_xticklabels(["c = %d" % c for c in CVALS])
    style_axes(ax, xlabel="candidate count of the named cell",
               ylabel="share of unforced cells")
    top = max(max(pb.values()), max(pw.values()))
    ax.set_ylim(0, top * 1.30)
    ax.text(0 - 0.19, pb[2] + top * 0.03, "available", color=MUTED,
            ha="center", fontsize=9)
    ax.text(0 + 0.19, pw[2] + top * 0.03, "named", color=SERIES[0],
            ha="center", fontsize=9)
    _save(fig, "C1_selection.png")

    # C2 -- P_hat and M
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(fr, [r["phat"] for r in fit], "-o", ms=4, color=SERIES[0],
            rasterized=True)
    ax.plot(fr, [r["M"] for r in fit], "-o", ms=4, color=SERIES[1],
            rasterized=True)
    style_axes(ax, xlabel=r"hint density $\rho$",
               ylabel="correct colour on an unforced cell")
    ax.invert_xaxis()
    label_line(ax, 0.70, dict((r["rho"], r["phat"]) for r in fit)[0.70],
               r"predicted from $w(c)$", color=SERIES[0], dy=10)
    label_line(ax, 0.60, dict((r["rho"], r["M"]) for r in fit)[0.60],
               "measured", color=SERIES[1], dy=-14)
    _save(fig, "C2_overid.png")

    # C3 -- M with the ceiling
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot([r["rho"] for r in rows], [r["M"] for r in rows], "-", lw=1.6,
            color=SERIES[0], rasterized=True)
    ax.plot(fr, [r["M"] for r in fit], "o", ms=4.5, color=SERIES[0],
            rasterized=True)
    ho0 = [r for r in rows if r["held_out"]]
    ax.plot([r["rho"] for r in ho0], [r["M"] for r in ho0], "o", ms=5,
            mfc="white", mec=SERIES[0], rasterized=True)
    ax.axhline(0.50, ls="--", lw=1, color=MUTED)
    ax.axhline(0.25, ls=":", lw=1, color=MUTED)
    style_axes(ax, xlabel=r"hint density $\rho$",
               ylabel="correct colour on an unforced cell")
    ax.invert_xaxis()
    ax.set_ylim(0.15, 0.62)
    ax.text(0.93, 0.555, "ceiling if the colour is drawn from the candidate set",
            color=MUTED, fontsize=8.5, ha="left")
    ax.text(0.93, 0.265, r"blind guess, $1/q$", color=MUTED, fontsize=8.5,
            ha="left")
    ax.text(0.40, 0.355, "open: densities no fit\never used", color=MUTED,
            fontsize=8.5, ha="center")
    _save(fig, "C3_ceiling.png")

    # C4 -- the gap against rho
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0.0, lw=1, color=MUTED)
    ax.plot(fr, [r["gap2"] for r in fit], "-o", ms=4, color=SERIES[0],
            rasterized=True)
    ho = [r for r in rows if r["held_out"]]
    ax.plot([r["rho"] for r in ho], [r["gap2"] for r in ho], "o", ms=5,
            mfc="white", mec=SERIES[0], rasterized=True)
    style_axes(ax, xlabel=r"hint density $\rho$",
               ylabel=r"$w(2) - b(2)$")
    ax.invert_xaxis()
    ax.set_ylim(-0.045, max(r["gap2"] for r in rows) * 1.10)
    ax.text(0.90, -0.030, "blind choice", color=MUTED, fontsize=8.5,
            ha="center", va="center")
    ax.text(0.63, max(r["gap2"] for r in rows) * 0.72,
            r"Spearman with $\rho$ = %+.3f" % c4, color=SERIES[0],
            fontsize=9, ha="center")
    ax.text(0.385, max(r["gap2"] for r in rows) * 0.40,
            "open: densities\nno fit ever used", color=MUTED, fontsize=8.5,
            ha="center")
    _save(fig, "C4_trend.png")

    # C5 -- stratified by coloured-neighbour count
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ks = [kv for kv in sorted(kn) if sum(kw[(kv, c)] for c in CVALS) >= 50]
    gaps = [kw[(kv, 2)] / sum(kw[(kv, c)] for c in CVALS)
            - kb[(kv, 2)] / kn[kv] for kv in ks]
    cols = [SERIES[0] if kv == modal_k else MUTED for kv in ks]
    ax.bar(ks, gaps, 0.6, color=cols, rasterized=True)
    ax.axhline(0.0, lw=1, color="black")
    ax.axhline(0.05, ls="--", lw=1, color=SERIES[1])
    style_axes(ax, xlabel="assigned neighbours $k$ of the named cell",
               ylabel=r"$w(2\,|\,k) - b(2\,|\,k)$")
    ax.set_xticks(ks)
    lo = min(gaps + [0.0])
    ax.set_ylim(lo - 0.012, 0.072)
    ax.text(ks[0] - 0.3, 0.055, "registered prediction", color=SERIES[1],
            fontsize=9, ha="left")
    ax.text(modal_k, 0.006, "modal stratum", color=SERIES[0], fontsize=9,
            ha="center")
    # a bar of height zero is invisible; say which zeros are which
    for kv, g in zip(ks, gaps):
        if abs(g) < 1e-9:
            ax.text(kv, 0.006, "both\nshares\nequal", color=MUTED, fontsize=7.5,
                    ha="center", va="bottom")
    _save(fig, "C5_stratified.png")

    # PH -- the cell choice IS selected on k, which is what explains C1
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ks2 = sorted(set(kwk) | {k for k in range(6) if kbk[k] / max(kbn, 1) > 1e-4})
    tw = max(sum(kwk.values()), 1)
    x = np.arange(len(ks2))
    ax.bar(x - 0.19, [kbk[k] / max(kbn, 1) for k in ks2], 0.36, color=MUTED,
           rasterized=True)
    ax.bar(x + 0.19, [kwk[k] / tw for k in ks2], 0.36, color=SERIES[1],
           rasterized=True)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks2])
    style_axes(ax, xlabel="assigned neighbours $k$ of the cell",
               ylabel="share of unforced cells")
    top = max(max(kwk[k] / tw for k in ks2),
              max(kbk[k] / max(kbn, 1) for k in ks2))
    ax.set_ylim(0, top * 1.32)
    # label each series where IT is the taller bar, so neither caption lands
    # on the other colour
    ja = ks2.index(max(ks2, key=lambda k: kbk[k] / max(kbn, 1) - kwk[k] / tw))
    jn = ks2.index(max(ks2, key=lambda k: kwk[k] / tw - kbk[k] / max(kbn, 1)))
    ax.text(ja - 0.19, kbk[ks2[ja]] / max(kbn, 1) + top * 0.035, "available",
            color=MUTED, ha="center", fontsize=9)
    ax.text(jn + 0.19, kwk[ks2[jn]] / tw + top * 0.035, "named",
            color=SERIES[1], ha="center", fontsize=9)
    _save(fig, "PH_kselect.png")
    print("figures ->", FIGS)


if __name__ == "__main__":
    main()
