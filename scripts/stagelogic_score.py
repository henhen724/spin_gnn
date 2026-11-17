"""Stage LOGIC -- teacher-forced LOGICAL accuracy, and whether a free-generation
first error names an unforced node.

NOTHING IS TRAINED HERE. Every model is a checkpoint that already exists, and
every instance is read from stage POS's cache, so the eval is point-for-point
the one the rho curves and stage POS were measured on.

-------------------------------------------------------------------------
THE QUESTION
-------------------------------------------------------------------------
From stage POS's measurement A, hshunt's fit makes the solve rate a clean
power of the teacher-forced reasoning-trace token accuracy p,

    S(rho)  ~  C * p^k ,     C ~ 1 for rho >= 0.65,     k ~ 0.73 n ,

with n = 48(1 - realized_rho) the number of blank cells and a free power fit
giving n^1.07 -- i.e. k is LINEAR in n, not quadratic, once the exponent is
measured against the teacher-forced token accuracy rather than against the
final validation accuracy.

The coefficient is 0.73 and not 1.0, and the reason is a scoring artifact:
MANY TEACHER-FORCED NODE ERRORS ARE HARMLESS. DD deductions inside a round
are order-free, so when the model names a different forced node than the one
the ground-truth trace happens to write next, measurement A scores it wrong
although it is a perfectly legal DD move. If every LOGICAL error were fatal
and only logical errors were counted, the exponent would have to be k = n
exactly -- one chance to fail per deduction, each failure fatal -- with
prefactor 1. The gap between 0.73 and 1 is then a measurement of how much of
A's error budget is bookkeeping rather than reasoning.

This script replaces A's "is the argmax the token the trace wrote" with "is
the argmax a token DD permits here".

-------------------------------------------------------------------------
MEASUREMENT L -- TEACHER-FORCED LOGICAL ACCURACY
-------------------------------------------------------------------------
One forward pass per instance over the full ground-truth sequence, exactly as
stage POS's measurement A, so the two differ ONLY in the scoring rule. At each
position of the reasoning section the argmax is scored against a set of
PERMITTED tokens rather than against the single true token:

  NODE position.   F = the set of nodes DD forces in the current round, read
                   from the hints plus all COMPLETED earlier rounds (the usual
                   synchronous round semantics -- same-round siblings do not
                   count as evidence), minus the nodes already emitted in this
                   round. Logically correct iff the argmax is a node token
                   whose node is in F.
                   F IS READ OFF THE TRACE, NOT RECOMPUTED. sudoku_trace's
                   round r is by construction exactly the set of nodes forced
                   by rounds < r, so F at a node position is the rest of the
                   current round. _stagelogic_dry_check verifies that identity
                   against a from-scratch forced-set computation rather than
                   assuming it.

  COLOUR position. Logically correct iff the argmax colour equals the DD
                   colour of the node in the true prefix -- unchanged from A,
                   because the colour of a named node is not order-free.

  BOUNDARY.        A BEGIN_ROUND position or the final terminator. Correct iff
                   the argmax is that token. At the end of a round F is empty
                   and BEGIN_ROUND (or SOLVED) is the only legal continuation,
                   so this is both A's rule and the logical one. THE ONE
                   EXCEPTION is the very first position of the section, which
                   predicts the first BEGIN_ROUND while F is NOT yet empty; the
                   grammar still forces BEGIN_ROUND there, it is scored that
                   way, and it is reported as scheme "round" index 0 so it can
                   be dropped from any re-binning.

Every argmax at a node position is additionally classified, which is where the
guard failures show up:

    in_F        a node DD forces right now        -- the logical hit
    premature   a node token, not in F, NOT yet coloured. The model has named
                a cell it cannot yet know: a GUARD FAILURE.
    recolor     a node token for a node already coloured (a hint, or emitted
                earlier in the trace)
    nonnode     not a node token at all

Saved per (run, rho, index, token type) as raw correct/total counts under three
indexing schemes -- "j" (deduction index), "round", and "all" -- so anything
downstream can re-bin. `n_strict_correct` carries measurement A's rule in the
same row, so the logical and strict accuracies are always compared on exactly
the same positions and L reproducing A is checkable, not assumed.

-------------------------------------------------------------------------
MEASUREMENT F -- WAS THE FIRST FREE-GENERATION ERROR AN UNFORCED NODE?
-------------------------------------------------------------------------
Stage POS's measurement B found that 99.7% of first errors name an UNCOLOURED
node with the WRONG colour. That leaves two very different readings: the model
picked a node DD had already forced and then got its colour wrong (a colour
failure), or it picked a node DD had not yet forced and so could not have known
the colour of (a guard failure, and the same mechanism as "premature" above).

B did not save WHICH node the model named, so this cannot be answered from its
CSVs. It is answered by REPLAYING the rollout: the same greedy decode, the same
cached instances, the same batch size, bit-identical by construction, and the
agreement is checked offline by requiring the replayed solve rate to equal B's
to the digit. At the first error the model's OWN prefix is still perfect (that
is what "first error" means), so DD can be run on it:

    in_F        the named node is in the forced set of the model's current
                round, minus what it already emitted this round -- the same
                definition as L, applied to the model's own state
    in_F_now    the named node is forced by the model's FULL current state,
                same-round siblings included. Recorded as a second column
                rather than substituted, because the round-start rule is the
                registered one and the two can differ.

`in_F` is also recorded for every pair up to and including the first error, so
the rate of unforced-but-correct moves -- a guard failure the model got away
with -- has a denominator.

-------------------------------------------------------------------------
REGISTERED BEFORE THE JOB RAN
-------------------------------------------------------------------------
Full text, with falsifiers and the figure list, in
runs/stagelogic_2026-11-17/REGISTERED.md. In brief:

L1  FITTING log S on log p_logic per density over the 79 in79 runs with
    0 < S < 1 gives an exponent k_logic close to n and a prefactor close to 1.
    Registered band: the slope of k_logic against n lands in [0.85, 1.20] and
    the median prefactor over rho >= 0.65 lands in [0.5, 2.0].
    FALSIFIED outside [0.7, 1.4] on the slope.

L2  p_logic > p_strict at every (run, rho) -- the logical rule can only forgive
    errors, never create them. This is an arithmetic identity, so a violation
    is a bug, not a finding.

L3  The 0.73 is mostly order bookkeeping: the median ratio k_logic/k_strict
    over rho >= 0.50 lands in [1.1, 1.6], i.e. removing the harmless node
    errors moves the exponent most of the way from 0.73n to n.

L4  A MAJORITY OF FIRST ERRORS NAME AN UNFORCED NODE. Over the in79 runs at
    rho = 0.50..0.75, the median fraction of first errors whose node is not in
    F exceeds 0.5.
    FALSIFIED below 0.5, which would say the model picks the right cell and
    mis-colours it, and would make "guard failure" the wrong name for this.

NOTHING IS APPLIED AND NO VERDICT IS PRINTED HERE; scoring lives in
scripts/stagelogic_fit.py.

-------------------------------------------------------------------------
HOW IT IS DRIVEN
-------------------------------------------------------------------------
    LOGIC_OUT      run directory (required)
    LOGIC_INST     instance cache to REUSE (required; stage POS's)
    LOGIC_RUNS     "all" | "seven" | "rest" | "r03,r07" | "12:24"
    LOGIC_SHARD    "i/n" -- every n-th run of the selection, for an array
    LOGIC_MEASURE  "L" | "F" | "LF"   (default LF)
    LOGIC_BATCH    forward / rollout batch size (default 64)
    LOGIC_SMOKE    1 = 2 densities, 8 instances, 2 runs; fast, not a result
"""
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.rho_transition_eval import cfg_from_checkpoint
from scripts.stagepos_cells import N_PER_BIN, RHO_GRID, SEED_START, successful_runs
from scripts.stagepos_score import (_run_cols, instance_cache, rollout_batch,
                                    select_runs, shard, true_pairs,
                                    walk_continuation)
from spin_gnn.sudoku_tokens import check_solution, decode, encode_instance
from spin_gnn.train_sudoku_transformer import load_checkpoint

CLASSES = ["in_F", "premature", "recolor", "nonnode"]
CNT = ["n_total", "n_logic_correct", "n_strict_correct"] + ["n_" + c for c in CLASSES]


# --------------------------------------------------------------------------
# DD forced sets
# --------------------------------------------------------------------------

def forced_mask(adj, asg, n_colors):
    """Boolean (N,) -- the nodes DD forces given the partial colouring `asg`.

    `asg` is an int8 (N,) array, -1 where unassigned. A node is forced when its
    ASSIGNED neighbours show all q-1 colours other than its own, which is the
    "naked single" rule sudoku_dd/sudoku_trace implement; counting distinct
    assigned neighbour colours is the same test only while the partial colouring
    is PROPER, which holds everywhere this is called (the true prefix, or a
    model prefix before its first error, where every emitted colour is the
    planted one).
    """
    nbr = asg[adj]                                   # (N, d)
    seen = np.zeros(adj.shape[0], dtype=np.int64)
    for c in range(n_colors):
        seen |= np.any(nbr == c, axis=1).astype(np.int64) << c
    cnt = np.zeros(adj.shape[0], dtype=np.int64)
    for c in range(n_colors):
        cnt += (seen >> c) & 1
    return (asg < 0) & (cnt == n_colors - 1)


def hint_asg(n_nodes, hints):
    asg = np.full(n_nodes, -1, dtype=np.int8)
    for k, v in hints.items():
        asg[int(k)] = int(v)
    return asg


# --------------------------------------------------------------------------
# MEASUREMENT L -- teacher forced, logical rule
# --------------------------------------------------------------------------

def logic_schema(trace, vocab):
    """Per reasoning position: (tok_type, j, round, allowed_nodes, true_tok).

    `allowed_nodes` is F at a node position (a frozenset of node indices) and
    None elsewhere. The order is encode_instance's, which
    stagepos_score.trace_schema already pins down and the dry check re-pins
    here, because one token of drift would shift every curve.
    """
    out = []
    j = 0
    for r, ev in enumerate(trace["rounds"]):
        out.append(("round", -1, r, None, vocab.BEGIN_ROUND))
        remaining = {int(nd) for nd, _c in ev}
        for nd, c in ev:
            nd, c = int(nd), int(c)
            out.append(("node", j, r, frozenset(remaining), vocab.node_token(nd)))
            remaining.discard(nd)
            out.append(("color", j, r, None, vocab.color_token(c)))
            j += 1
    out.append(("term", -1, len(trace["rounds"]), None,
                vocab.SOLVED if trace["resolved"] else vocab.UNRESOLVED))
    return out


def measure_L(model, vocab, instances, device, max_len, batch=64):
    """counts[(scheme, index, tok_type)] = {n_total, n_logic_correct, ...}."""
    counts = {}
    n_skipped = 0
    big = 1 << 30

    prepared = []
    for adj, _colors, hints, trace in instances:
        toks = encode_instance(adj, trace, vocab)
        if len(toks) > max_len:
            n_skipped += 1
            continue
        # coloured-before test at deduction j: a hint, or emitted at some k < j
        order = {}
        for k, (nd, _c) in enumerate(true_pairs(trace)):
            order[int(nd)] = k
        prepared.append((toks, logic_schema(trace, vocab),
                         {int(v) for v in trace["hints"]}, order))

    for s in range(0, len(prepared), batch):
        chunk = prepared[s:s + batch]
        width = max(len(t) for t, _, _, _ in chunk)
        buf = torch.zeros(len(chunk), width, dtype=torch.long, device=device)
        for i, (t, _, _, _) in enumerate(chunk):
            buf[i, :len(t)] = torch.from_numpy(t)
        with torch.no_grad():
            logits = model(buf)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = logits.argmax(dim=-1).cpu().numpy()
        for i, (toks, schema, hint_nodes, order) in enumerate(chunk):
            rs = int(np.flatnonzero(toks == vocab.BEGIN_REASONING)[0]) + 1
            if len(toks) - rs != len(schema):
                raise SystemExit(
                    "schema/token mismatch: %d scored positions but %d schema "
                    "entries" % (len(toks) - rs, len(schema)))
            got = pred[i, rs - 1:len(toks) - 1]
            for p, (tok_type, j, rnd, allowed, true_tok) in enumerate(schema):
                t = int(got[p])
                strict = t == true_tok
                cls = None
                if tok_type == "node":
                    if vocab.is_node_token(t):
                        u = vocab.node_index(t)
                        cls = ("in_F" if u in allowed else
                               "recolor" if (u in hint_nodes or order.get(u, big) < j)
                               else "premature")
                    else:
                        cls = "nonnode"
                    logic = cls == "in_F"
                else:
                    logic = strict
                for scheme, idx in (("j", j), ("round", rnd), ("all", -1)):
                    if scheme == "j" and j < 0:
                        continue
                    for tt in ((tok_type, "any") if scheme == "all" else (tok_type,)):
                        c = counts.setdefault((scheme, idx, tt),
                                              dict.fromkeys(CNT, 0))
                        c["n_total"] += 1
                        c["n_logic_correct"] += logic
                        c["n_strict_correct"] += strict
                        if cls is not None:
                            c["n_" + cls] += 1
    return counts, n_skipped


# --------------------------------------------------------------------------
# MEASUREMENT F -- the first free-generation error, against DD
# --------------------------------------------------------------------------

def walk_forced(cont, vocab, adj, n_colors, hints, colors, first_err):
    """Re-walk a generated continuation tracking the model's OWN DD state.

    A second pass, deliberately: `walk_continuation` stays the single authority
    on what counts as an error, and this only answers "was that node forced",
    so the two cannot drift apart. Stops after the first error (the model's
    colouring is only guaranteed proper up to there).

    Returns (per_pair, err) where err is None if the named first error is not a
    node-bearing pair.
    """
    n_nodes = adj.shape[0]
    asg = hint_asg(n_nodes, hints)
    f_round = forced_mask(adj, asg, n_colors)
    emitted = np.zeros(n_nodes, dtype=bool)
    per_pair, err = [], None
    rnd, i, j = -1, 0, 0
    while i < len(cont):
        t = int(cont[i])
        if t == vocab.SOLVED or t == vocab.UNRESOLVED:
            break
        if t == vocab.BEGIN_ROUND:
            rnd += 1
            f_round = forced_mask(adj, asg, n_colors)
            emitted[:] = False
            i += 1
            continue
        if not vocab.is_node_token(t):
            break
        node = vocab.node_index(t)
        f = f_round & ~emitted
        per_pair.append({"j": j, "in_F": int(bool(f[node])), "n_F": int(f.sum()),
                         "round": rnd})
        if j == first_err:
            f_now = forced_mask(adj, asg, n_colors)
            c_ok = -1
            if i + 1 < len(cont) and vocab.is_color_token(int(cont[i + 1])):
                c_ok = int(vocab.color_index(int(cont[i + 1])) == int(colors[node]))
            err = {"err_node": node, "err_round": rnd,
                   "err_in_F": int(bool(f[node])), "err_n_F": int(f.sum()),
                   "err_in_F_now": int(bool(f_now[node])),
                   "err_n_F_now": int(f_now.sum()),
                   "err_uncolored": int(asg[node] < 0),
                   "err_color_correct": c_ok}
            break
        if i + 1 >= len(cont) or not vocab.is_color_token(int(cont[i + 1])):
            break
        if asg[node] < 0:
            asg[node] = vocab.color_index(int(cont[i + 1]))
            emitted[node] = True
        i += 2
        j += 1
    return per_pair, err


def measure_F(model, vocab, instances, device, max_len, n_colors, batch=64):
    """Per-pair forced counts, one record per instance that errs, and the
    replayed solve rate (which must equal measurement B's)."""
    haz, recs = {}, []
    n_solved = 0
    n_first_err = n_err_node = n_unforced = n_unforced_now = 0
    for s in range(0, len(instances), batch):
        chunk = instances[s:s + batch]
        prompts, metas = [], []
        for adj, colors, hints, trace in chunk:
            full = encode_instance(adj, trace, vocab)
            rs = int(np.flatnonzero(full == vocab.BEGIN_REASONING)[0]) + 1
            prompts.append(full[:rs])
            metas.append((adj, colors, hints, trace, rs))
        seqs, _ = rollout_batch(model, prompts, vocab, device, max_len, max_len)
        for k, seq in enumerate(seqs):
            adj, colors, hints, trace, rs = metas[k]
            cont = seq[rs:]
            truth = true_pairs(trace)
            pairs, terminated = walk_continuation(cont, vocab, adj, colors, hints, truth)
            first_err, first_kind = -1, "none"
            for p in pairs:
                if p["kind"] != "ok":
                    first_err, first_kind = p["j"], p["kind"]
                    break
            dec = decode(seq, vocab)
            solved = (dec["outcome"] == ("SOLVED" if trace["resolved"] else "UNRESOLVED")
                      and check_solution(dec, colors, vocab.n_nodes))
            n_solved += bool(solved)

            per_pair, err = walk_forced(cont, vocab, adj, n_colors, hints,
                                        colors, first_err)
            for pp in per_pair:
                h = haz.setdefault(pp["j"], dict(n_at_risk=0, n_in_F=0, n_err=0,
                                                 n_err_in_F=0, n_F_sum=0))
                h["n_at_risk"] += 1
                h["n_in_F"] += pp["in_F"]
                h["n_F_sum"] += pp["n_F"]
            if first_err >= 0:
                n_first_err += 1
                h = haz.setdefault(first_err, dict(n_at_risk=0, n_in_F=0, n_err=0,
                                                   n_err_in_F=0, n_F_sum=0))
                h["n_err"] += 1
                if err is not None:
                    n_err_node += 1
                    n_unforced += 1 - err["err_in_F"]
                    n_unforced_now += 1 - err["err_in_F_now"]
                    h["n_err_in_F"] += err["err_in_F"]
                    recs.append(dict({"inst": s + k, "n_blank": len(truth),
                                      "T_true": len(trace["rounds"]),
                                      "first_error_j": first_err,
                                      "first_kind": first_kind,
                                      "n_pairs": len(pairs),
                                      "terminated": terminated or "none",
                                      "solved": int(bool(solved))}, **err))
    n = len(instances)
    summary = dict(n_first_err=n_first_err, n_err_node=n_err_node,
                   n_err_unforced=n_unforced, n_err_unforced_now=n_unforced_now,
                   frac_unforced=(n_unforced / n_err_node) if n_err_node else float("nan"),
                   solve_rate=(n_solved / n) if n else float("nan"))
    return haz, recs, summary


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main():
    out_dir = os.environ.get("LOGIC_OUT") or ""
    inst_dir = os.environ.get("LOGIC_INST") or ""
    if not out_dir or not inst_dir:
        raise SystemExit("LOGIC_OUT and LOGIC_INST are both required")
    spec = os.environ.get("LOGIC_RUNS", "all")
    measure = os.environ.get("LOGIC_MEASURE", "LF").upper()
    batch = int(os.environ.get("LOGIC_BATCH", "64"))
    smoke = os.environ.get("LOGIC_SMOKE") == "1"
    tag = spec.replace(":", "_").replace(",", "-")
    if os.environ.get("LOGIC_SHARD"):
        tag = "%s_%s" % (tag, os.environ["LOGIC_SHARD"].replace("/", "of"))

    os.makedirs(out_dir, exist_ok=True)
    runs = successful_runs(verbose=True)
    sel = shard(select_runs(runs, spec), os.environ.get("LOGIC_SHARD", ""))
    rho_grid, n_per_bin = list(RHO_GRID), N_PER_BIN
    if smoke:
        sel, rho_grid, n_per_bin, tag = sel[:2], [0.95, 0.60], 8, "smoke"
        inst_dir = os.path.join(out_dir, "instances_smoke")

    if not smoke and not os.path.isdir(inst_dir):
        raise SystemExit("LOGIC_INST=%s does not exist -- stage LOGIC reuses "
                         "stage POS's cached instances and never rebuilds them"
                         % inst_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\ndevice=%s  runs=%d  measure=%s  bins=%d  n_per_bin=%d  batch=%d  tag=%s"
          % (device, len(sel), measure, len(rho_grid), n_per_bin, batch, tag))
    print("out=%s\ninst=%s" % (out_dir, inst_dir))

    l_path = os.path.join(out_dir, "L_position_%s.csv" % tag)
    f_path = os.path.join(out_dir, "F_hazard_%s.csv" % tag)
    e_path = os.path.join(out_dir, "F_errors_%s.csv" % tag)
    s_path = os.path.join(out_dir, "LF_summary_%s.csv" % tag)
    l_h, f_h, e_h = [None, None], [None, None], [None, None]
    s_rows = []

    def _w(holder, path, row, fieldnames):
        if holder[0] is None:
            holder[0] = open(path, "w", newline="")
            holder[1] = csv.DictWriter(holder[0], fieldnames=fieldnames)
            holder[1].writeheader()
        holder[1].writerow(row)

    t_all = time.time()
    for ri, rec in enumerate(sel):
        model, vocab, config = load_checkpoint(rec["path"], device=device)
        sudoku_cfg = cfg_from_checkpoint(config)
        max_len = int(config["max_len"])
        n_colors = int(vocab.n_colors)
        cols = _run_cols(rec)
        print("\n=== [%d/%d] %s  %s seed %d  acc=%.4f  in79=%s ==="
              % (ri + 1, len(sel), rec["run_id"], rec["cell"], rec["seed"],
                 rec["val_acc"], rec["in79"]))
        for rho in rho_grid:
            t0 = time.time()
            inst = instance_cache(inst_dir, sudoku_cfg, rho, n_per_bin, SEED_START)
            realized = sum(len(h) for _, _, h, _ in inst) / (len(inst) * sudoku_cfg.n_nodes)
            dd_T = sum(len(t["rounds"]) for _, _, _, t in inst) / len(inst)
            n_blank = sum(len(true_pairs(t)) for _, _, _, t in inst) / len(inst)
            base = dict(cols, rho=float(rho), realized_rho=realized, dd_mean_T=dd_T,
                        mean_n_blank=n_blank, n_instances=len(inst))
            srow = dict(base)

            if "L" in measure:
                counts, n_skip = measure_L(model, vocab, inst, device, max_len, batch)
                for (scheme, idx, tt), c in sorted(counts.items()):
                    _w(l_h, l_path, dict(base, scheme=scheme, index=idx, tok_type=tt,
                                         n_skipped=n_skip, **c),
                       list(base) + ["scheme", "index", "tok_type", "n_skipped"] + CNT)
                for name, tt in (("", "any"), ("_node", "node"), ("_color", "color")):
                    c = counts.get(("all", -1, tt))
                    srow["p_logic" + name] = (c["n_logic_correct"] / c["n_total"]
                                              if c and c["n_total"] else float("nan"))
                    srow["p_strict" + name] = (c["n_strict_correct"] / c["n_total"]
                                               if c and c["n_total"] else float("nan"))
                c = counts.get(("all", -1, "node")) or dict.fromkeys(CNT, 0)
                for k in CLASSES:
                    srow["node_" + k] = c["n_" + k]

            if "F" in measure:
                haz, errs, fsum = measure_F(model, vocab, inst, device, max_len,
                                            n_colors, batch)
                for j in sorted(haz):
                    _w(f_h, f_path, dict(base, j=j, **haz[j]),
                       list(base) + ["j"] + list(haz[j]))
                for r in errs:
                    _w(e_h, e_path, dict(cols, rho=float(rho), **r),
                       list(cols) + ["rho"] + list(r))
                srow.update(fsum)

            s_rows.append(dict(srow, elapsed_s=round(time.time() - t0, 1)))
            print("  rho=%.2f real=%.4f n=%.1f  p_logic=%s p_strict=%s  "
                  "solve=%s unforced=%s  %.1fs"
                  % (rho, realized, n_blank,
                     "%.4f" % srow.get("p_logic", float("nan")),
                     "%.4f" % srow.get("p_strict", float("nan")),
                     "%.4f" % srow.get("solve_rate", float("nan")),
                     "%.3f" % srow.get("frac_unforced", float("nan")),
                     time.time() - t0))
            for holder in (l_h, f_h, e_h):
                if holder[0] is not None:
                    holder[0].flush()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    for holder in (l_h, f_h, e_h):
        if holder[0] is not None:
            holder[0].close()
    if s_rows:
        keys = list(s_rows[0])
        with open(s_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(s_rows)
    print("\nwrote %s and %s" % (
        ", ".join(os.path.basename(p) for p, h in
                  ((l_path, l_h), (f_path, f_h), (e_path, e_h)) if h[0] is not None),
        os.path.basename(s_path)))
    print("total %.1f min" % ((time.time() - t_all) / 60))


if __name__ == "__main__":
    main()
