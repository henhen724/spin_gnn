"""Stage POS -- where along the reasoning trace the model goes wrong.

NOTHING IS TRAINED HERE. Every number comes from checkpoints that already
exist, on the instances the rho curves were already measured on.

-------------------------------------------------------------------------
THE QUESTION
-------------------------------------------------------------------------
hshunt's background analysis of the 15-directory minarch scan found that at
every fixed hint density the solve rate is close to a power law in final
validation TOKEN accuracy,

    S(rho)  ~  acc^k(rho),        R^2 ~ 0.92 for rho = 0.50 .. 0.75,

with the exponent k growing steeply as the instances get sparser: k = 2.0 at
rho = 0.95, 4.6 at 0.90, 19.9 at 0.80, 38 at 0.70, 69 at 0.60, 92 at 0.50.
Plotted against the number of BLANK cells,

    n = 48 (1 - realized_rho),

that exponent scales as k ~ n^(1.84 +- 0.03) for rho >= 0.50 -- roughly n
squared. Mean DD depth T fits much worse (residuals 7x larger) and the trace
token count 2n + T + 1 gives the same answer as n, which is expected since it
is dominated by the 2n term.

n squared is the signature of an error rate that GROWS ALONG THE TRACE. If
each of the n deductions failed independently with the same probability p,
then log(1/S) would be n * log(1/(1-p)), linear in n, and k would scale as n,
not n^2. If instead the per-step failure probability rises linearly with
position, p_j ~ b*j, then

    log(1/S)  =  -sum_j log(1 - p_j)  ~  sum_j p_j  ~  b n^2 / 2,

quadratic in n. That is the hypothesis. It is a statement about a quantity
nobody here has ever measured -- the per-position error rate -- and this
script measures it.

-------------------------------------------------------------------------
WHAT IS MEASURED, AND ON WHAT
-------------------------------------------------------------------------
The population is `stagepos_cells.successful_runs()`: the 79 (cell, seed)
runs whose solve rate at rho = 0.50 is nonzero, i.e. exactly the cells the
power law above was fitted on, plus `full L=2 d256 s0` measured and flagged
`in79=False` because hshunt named it by hand and it fails the filter.

The eval instances are the rho curves' own: the 14-point grid
0.95 .. 0.33, 400 per bin, `rho_transition_eval`'s default seed start
60,000,000,000, and `fall` / `shuffle_adj` read from each checkpoint's config
by `cfg_from_checkpoint`. All 80 runs share one instance distribution (checked
in stagepos_cells), so the instances are BUILT ONCE and reused, which makes
every run's curve point-by-point comparable and makes the array affordable.

MEASUREMENT A -- TEACHER-FORCED. One forward pass per instance over the full
ground-truth sequence. At every position of the REASONING section, is the
next-token argmax the true next token, given a PERFECT prefix? Indexed three
ways and split by token type, with raw correct/total counts saved per density
so any re-binning is possible later:

    scheme "offset"  token offset from BEGIN_REASONING (BEGIN_REASONING = 0,
                     so the first BEGIN_ROUND is 1)
    scheme "j"       deduction index: the j-th deduced (node, colour) pair in
                     reading order, 0-based, with the NODE token and the
                     COLOUR token scored separately via tok_type
    scheme "round"   the DD round the token belongs to, 0-based

    tok_type         "node" | "color" | "round" (a BEGIN_ROUND token) |
                     "term" (the final SOLVED / UNRESOLVED)

A cannot by itself test the hypothesis: under teacher forcing the prefix is
always correct, so A measures intrinsic per-step difficulty with no error
accumulation at all. That is precisely why it is worth having alongside B --
it separates "later deductions are harder" from "later deductions are harder
BECAUSE the model has already corrupted its own prefix".

MEASUREMENT B -- FREE GENERATION, and the direct test. Greedy rollout from the
prompt through BEGIN_REASONING, exactly as `train_sudoku_transformer.solve_rate`
does it (temperature 0, same stopping rules, same budget), then the emitted
continuation is walked pair by pair. For the j-th emitted (node, colour) pair:

    node_valid     the node slot holds a node token for a node that is not
                   already coloured -- not a hint, not emitted earlier. This is
                   "a legal move", NOT "the move the ground-truth trace made":
                   DD deductions inside a round are order-free, so a different
                   order is not an error.
    color_correct  the colour slot holds a colour token equal to the planted
                   colouring of the node the model actually named.
    matches_true   the node equals the ground-truth trace's j-th node. Recorded
                   separately, never used in the error definition, so the
                   ordering question stays answerable.
    color_proper   the colour conflicts with no already-known neighbour -- a
                   locally legal choice even when globally wrong.

    error at j  :=  not (node_valid and color_correct)

    HAZARD      h(j) = P(first failure at j | no failure before j)

A rollout can also fail WITHOUT a bad pair: it can emit SOLVED before all n
blanks are filled, emit UNRESOLVED when the truth is SOLVED, or run out of
token budget. Those are counted as a separate `n_stop` column at j = the
number of pairs emitted, so they are neither silently dropped (which would
make a model that quits early look error-free) nor mixed into the per-pair
error rate. `hazard` uses both; `hazard_err` uses pair errors alone. Both are
reported.

-------------------------------------------------------------------------
REGISTERED BEFORE THE JOB RAN -- predictions, and what would overturn them
-------------------------------------------------------------------------
P1  THE HYPOTHESIS. Hazard rises with j. Spearman correlation between j and
    h(j), over j with n_at_risk >= 100, is POSITIVE at every density in
    rho = 0.50 .. 0.75 for at least 5 of the 7 hand-named runs, and for a
    majority of the 79.
    FALSIFIED by a flat (|rho_s| < 0.3) or decreasing hazard on the majority.
    A flat hazard would mean the n^2 exponent comes from somewhere other than
    position-dependent error, and the hypothesis is simply wrong.

P2  THE QUANTITATIVE LINK. Summing the measured hazard must reproduce the
    measured solve rate: sum_j -log(1 - h(j)) against -log(S) from the same
    rollouts, within a factor of 2 at rho = 0.50 .. 0.75 (it is an identity up
    to the discreteness of the at-risk sets, so this is an arithmetic check on
    the pipeline, and a failure means a bug, not a finding). The real
    prediction is the EXPONENT: fitting sum_j h(j) against n over
    rho >= 0.50 gives a slope in [1.5, 2.3], consistent with hshunt's
    1.84 +- 0.03.
    FALSIFIED by an exponent outside [1.3, 2.5].

P3  WHAT A ADDS. Under a perfect prefix the per-token error ALSO rises with
    position: teacher-forced colour-token accuracy at the last deduction is
    below that at the first, on >= 5 of the 7 named runs at rho = 0.60.
    THE OUTCOME I ACTUALLY EXPECT IS THE OPPOSITE, and it is the more
    interesting one: A flat while B rises would say the position dependence is
    entirely error ACCUMULATION -- the model's own earlier mistakes corrupting
    the prefix it then reads -- rather than an intrinsic difficulty gradient.
    Reported that way, not as a failed prediction.

P4  THE CONFOUND THAT MUST BE REPORTED EITHER WAY. Later deductions sit in
    later ROUNDS, and a later round depends on more previously deduced
    colours, so "position in the trace" and "depth of the deduction" are
    confounded by construction. Every row therefore carries the round index,
    and hazard is reported both against j and against position WITHIN a round.
    If the within-round trend is flat and only the across-round trend
    survives, the right statement is "error grows with DD depth", not "with
    token position", and that is what will be written.

P5  THE SECOND CONFOUND. At fixed rho, a larger j also means fewer blanks
    left, so the remaining deductions are not drawn from a fixed difficulty
    distribution. Hazard is therefore also reported against j/n.

NOTHING IS APPLIED AND NO VERDICT IS PRINTED. The exponent fit, the choice of
error definition and the question of whether to re-run anything stay hshunt's.

-------------------------------------------------------------------------
HOW IT IS DRIVEN
-------------------------------------------------------------------------
    POS_OUT       run directory (required)
    POS_RUNS      "seven" | "rest" | "all" | "r03,r07,..." | "12:24" (slice)
    POS_SHARD     "i/n" -- every n-th run of the selection, for an array
    POS_MEASURE   "A" | "B" | "AB"   (default AB)
    POS_TAG       shard tag for the output filenames (default from POS_RUNS)
    POS_BATCH     rollout / forward batch size (default 64)
    POS_SMOKE     1 = 2 densities, 8 instances, 2 runs; wrong on purpose, fast
"""
import csv
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.rho_transition_eval import cfg_from_checkpoint
from scripts.stagepos_cells import N_PER_BIN, RHO_GRID, SEED_START, successful_runs
from spin_gnn.sudoku_tokens import check_solution, decode, encode_instance
from spin_gnn.train_sudoku_transformer import (build_instances, generate,
                                               load_checkpoint)

from dataclasses import replace

RUN_FIELDS = ["run_id", "cell", "key", "seed", "arch_mode", "num_layers", "d_model",
              "n_trainable_params", "val_acc", "in79", "in_seven", "solve50"]


def _run_cols(rec):
    d = {k: rec[k] for k in RUN_FIELDS if k != "key"}
    d["key"] = "|".join(str(x) for x in rec["key"])
    return d


# --------------------------------------------------------------------------
# instances: built once, reused by every run and every array task
# --------------------------------------------------------------------------

def instance_cache(cache_dir, sudoku_cfg, rho, n_per_bin, seed_start):
    """The bin's instances, built on first call and pickled.

    Identical for every run by construction (stagepos_cells checks that all 80
    share the instance distribution), so building them 80 times would be 80x
    the same CPU-bound removal_path work for byte-identical output.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "inst_rho%05d_n%d_s%d.pkl"
                        % (round(rho * 10000), n_per_bin, seed_start % 1000000))
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    bin_cfg = replace(sudoku_cfg, rho=float(rho), rho_max=float(rho))
    inst = build_instances(bin_cfg, n_per_bin, seed_start)
    tmp = path + ".%d.tmp" % os.getpid()
    with open(tmp, "wb") as fh:
        pickle.dump(inst, fh, protocol=4)
    os.replace(tmp, path)  # atomic, so two array tasks racing cannot tear a file
    return inst


def trace_schema(trace, vocab):
    """Per-position (tok_type, j, round) for the reasoning section, in order.

    Mirrors sudoku_tokens.encode_instance's non-scratchpad branch exactly; the
    dry check asserts the two agree token for token on real instances, because
    a one-token offset here would shift every curve this script produces.
    """
    out = []
    j = 0
    for r, round_events in enumerate(trace["rounds"]):
        out.append(("round", -1, r))
        for _node, _c in round_events:
            out.append(("node", j, r))
            out.append(("color", j, r))
            j += 1
    out.append(("term", -1, len(trace["rounds"])))
    return out


def true_pairs(trace):
    return [(int(n), int(c)) for rd in trace["rounds"] for n, c in rd]


# --------------------------------------------------------------------------
# MEASUREMENT A -- teacher forced
# --------------------------------------------------------------------------

def measure_A(model, vocab, instances, device, max_len, batch=64):
    """counts[(scheme, index, tok_type)] = [n_correct, n_total].

    Right padding with no key-padding mask: the model's attention is causal
    (sat_transformer2.forward builds a triu mask every call), so a position can
    never see a later pad, and the logits at every real position are bitwise
    what an unpadded forward would give.
    """
    counts = {}
    n_skipped = 0
    enc = []
    for adj, _colors, _hints, trace in instances:
        toks = encode_instance(adj, trace, vocab)
        if len(toks) > max_len:
            n_skipped += 1
            continue
        enc.append((toks, trace_schema(trace, vocab)))

    for s in range(0, len(enc), batch):
        chunk = enc[s:s + batch]
        lens = [len(t) for t, _ in chunk]
        width = max(lens)
        buf = torch.zeros(len(chunk), width, dtype=torch.long, device=device)
        for i, (t, _) in enumerate(chunk):
            buf[i, :len(t)] = torch.from_numpy(t)
        with torch.no_grad():
            logits = model(buf)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = logits.argmax(dim=-1).cpu().numpy()
        for i, (toks, schema) in enumerate(chunk):
            rs = int(np.flatnonzero(toks == vocab.BEGIN_REASONING)[0]) + 1
            n = len(toks) - rs
            if n != len(schema):
                raise SystemExit(
                    "schema/token mismatch: %d scored positions but %d schema "
                    "entries -- encode_instance and trace_schema disagree"
                    % (n, len(schema)))
            # position t is predicted from the logits at t-1
            ok = pred[i, rs - 1:len(toks) - 1] == toks[rs:]
            for p in range(n):
                tok_type, j, rnd = schema[p]
                hit = bool(ok[p])
                for scheme, idx in (("offset", p + 1), ("j", j), ("round", rnd)):
                    if scheme == "j" and j < 0:
                        continue
                    c = counts.setdefault((scheme, idx, tok_type), [0, 0])
                    c[0] += hit
                    c[1] += 1
    return counts, n_skipped


# --------------------------------------------------------------------------
# MEASUREMENT B -- free generation
# --------------------------------------------------------------------------

def rollout_batch(model, prompts, vocab, device, max_len, max_new_tokens):
    """Greedy continuations, batched. Same decisions as
    train_sudoku_transformer.generate, one row at a time, verified by
    _stagepos_dry_check.

    Right padding is sound for the same causal-mask reason as measure_A, and
    the unfinished rows are compacted out every step so the batch shrinks as
    sequences terminate instead of forwarding dead rows to the budget.
    """
    b = len(prompts)
    buf = torch.zeros(b, max_len, dtype=torch.long, device=device)
    lens = np.zeros(b, dtype=np.int64)
    for i, p in enumerate(prompts):
        lens[i] = len(p)
        buf[i, :len(p)] = torch.from_numpy(p.astype(np.int64))
    start = lens.copy()
    unresolved_at = np.full(b, -1, dtype=np.int64)
    done = np.zeros(b, dtype=bool)

    with torch.no_grad():
        for _step in range(max_new_tokens):
            active = np.flatnonzero(~done & (lens < max_len))
            if active.size == 0:
                break
            idx = torch.from_numpy(active).to(device)
            width = int(lens[active].max())
            logits = model(buf[idx, :width])
            if isinstance(logits, tuple):
                logits = logits[0]
            at = torch.from_numpy(lens[active] - 1).to(device)
            nxt = logits[torch.arange(len(active), device=device), at].argmax(dim=-1)
            buf[idx, torch.from_numpy(lens[active]).to(device)] = nxt
            nxt_np = nxt.cpu().numpy()
            lens[active] += 1
            done[active[nxt_np == vocab.SOLVED]] = True
            fresh = active[(nxt_np == vocab.UNRESOLVED) & (unresolved_at[active] < 0)]
            unresolved_at[fresh] = lens[fresh]
            seen = active[unresolved_at[active] >= 0]
            done[seen[lens[seen] - unresolved_at[seen] >= vocab.n_nodes]] = True

    out = buf.cpu().numpy()
    return [out[i, :lens[i]] for i in range(b)], start


def walk_continuation(cont, vocab, adj, colors, hints, truth):
    """Pair-by-pair record of one generated continuation.

    Returns (pairs, terminated, n_pairs). `pairs[j]` has node_valid,
    color_correct, matches_true, color_proper, round, and `err`.
    """
    assigned = {int(k): int(v) for k, v in hints.items()}
    pairs = []
    rnd = -1
    i = 0
    terminated = None
    while i < len(cont):
        t = int(cont[i])
        if t == vocab.SOLVED or t == vocab.UNRESOLVED:
            terminated = "SOLVED" if t == vocab.SOLVED else "UNRESOLVED"
            break
        if t == vocab.BEGIN_ROUND:
            rnd += 1
            i += 1
            continue
        if not vocab.is_node_token(t):
            # a special token where a node must stand: the grammar is broken
            # and nothing after it can be parsed as a pair
            pairs.append({"j": len(pairs), "round": rnd, "node_valid": False,
                          "color_correct": False, "matches_true": False,
                          "color_proper": False, "kind": "grammar"})
            break
        node = vocab.node_index(t)
        node_valid = node not in assigned
        j = len(pairs)
        matches_true = j < len(truth) and node == truth[j][0]
        if i + 1 >= len(cont):
            pairs.append({"j": j, "round": rnd, "node_valid": node_valid,
                          "color_correct": False, "matches_true": matches_true,
                          "color_proper": False, "kind": "cut"})
            break
        t2 = int(cont[i + 1])
        color_is = vocab.is_color_token(t2)
        c = vocab.color_index(t2) if color_is else None
        color_correct = bool(color_is and c == int(colors[node]))
        color_proper = bool(color_is and all(assigned.get(int(u)) != c for u in adj[node]))
        kind = ("ok" if (node_valid and color_correct) else
                "grammar" if not color_is else
                "node_invalid" if not node_valid else "color_wrong")
        pairs.append({"j": j, "round": rnd, "node_valid": node_valid,
                      "color_correct": color_correct, "matches_true": matches_true,
                      "color_proper": color_proper, "kind": kind})
        if node_valid and color_is:
            assigned[node] = c
        if not color_is:
            break
        i += 2
    return pairs, terminated


def measure_B(model, vocab, instances, device, max_len, max_new_tokens, batch=64):
    """Per-pair hazard counts and one record per instance."""
    haz = {}
    recs = []
    n_solved = 0
    for s in range(0, len(instances), batch):
        chunk = instances[s:s + batch]
        prompts, metas = [], []
        for adj, colors, hints, trace in chunk:
            full = encode_instance(adj, trace, vocab)
            rs = int(np.flatnonzero(full == vocab.BEGIN_REASONING)[0]) + 1
            prompts.append(full[:rs])
            metas.append((adj, colors, hints, trace, rs))
        seqs, _start = rollout_batch(model, prompts, vocab, device, max_len, max_new_tokens)
        for k, seq in enumerate(seqs):
            adj, colors, hints, trace, rs = metas[k]
            cont = seq[rs:]
            truth = true_pairs(trace)
            n_blank = len(truth)
            pairs, terminated = walk_continuation(cont, vocab, adj, colors, hints, truth)

            first_err, first_kind = -1, "none"
            for p in pairs:
                if p["kind"] != "ok":
                    first_err, first_kind = p["j"], p["kind"]
                    break
            # a rollout that never emits a bad pair can still fail: it can stop
            # early, call UNRESOLVED on a resolvable instance, or run out of
            # budget. that is a STOP event at j = the number of pairs emitted.
            stop_at = -1
            if first_err < 0:
                if terminated == "UNRESOLVED":
                    stop_at, first_kind = len(pairs), "unresolved"
                elif terminated is None:
                    stop_at, first_kind = len(pairs), "truncated"
                elif len(pairs) < n_blank:
                    stop_at, first_kind = len(pairs), "early_term"

            dec = decode(seq, vocab)
            solved = (dec["outcome"] == ("SOLVED" if trace["resolved"] else "UNRESOLVED")
                      and check_solution(dec, colors, vocab.n_nodes))
            n_solved += bool(solved)

            stop_j = first_err if first_err >= 0 else (stop_at if stop_at >= 0 else len(pairs))
            for p in pairs:
                h = haz.setdefault(p["j"], dict(n_at_risk=0, n_err=0, n_stop=0,
                                                n_node_valid=0, n_color_correct=0,
                                                n_matches_true=0, n_emitted=0,
                                                n_node_valid_u=0, n_color_correct_u=0,
                                                n_color_proper_u=0, n_round_sum=0))
                h["n_emitted"] += 1
                h["n_node_valid_u"] += p["node_valid"]
                h["n_color_correct_u"] += p["color_correct"]
                h["n_color_proper_u"] += p["color_proper"]
                h["n_round_sum"] += max(p["round"], 0)
                if p["j"] <= stop_j:
                    h["n_at_risk"] += 1
                    h["n_node_valid"] += p["node_valid"]
                    h["n_color_correct"] += p["color_correct"]
                    h["n_matches_true"] += p["matches_true"]
                    if p["j"] == first_err:
                        h["n_err"] += 1
            if stop_at >= 0:
                h = haz.setdefault(stop_at, dict(n_at_risk=0, n_err=0, n_stop=0,
                                                 n_node_valid=0, n_color_correct=0,
                                                 n_matches_true=0, n_emitted=0,
                                                 n_node_valid_u=0, n_color_correct_u=0,
                                                 n_color_proper_u=0, n_round_sum=0))
                if stop_at >= len(pairs):
                    h["n_at_risk"] += 1
                h["n_stop"] += 1

            recs.append({
                "inst": s + k, "n_blank": n_blank, "T_true": len(trace["rounds"]),
                "n_hints": len(hints), "prompt_len": rs, "gen_len": len(seq),
                "n_pairs": len(pairs), "first_error_j": first_err,
                "stop_j": stop_at, "first_kind": first_kind,
                "terminated": terminated or "none", "solved": int(bool(solved)),
                "well_formed": int(bool(dec["well_formed"])),
            })
    return haz, recs, n_solved / len(instances) if instances else float("nan")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def select_runs(runs, spec):
    if spec in ("", "all"):
        return runs
    if spec == "seven":
        return [r for r in runs if r["in_seven"]]
    if spec == "rest":
        return [r for r in runs if not r["in_seven"]]
    if ":" in spec:
        a, b = spec.split(":")
        return runs[int(a):int(b)]
    ids = {x.strip() for x in spec.split(",") if x.strip()}
    sel = [r for r in runs if r["run_id"] in ids]
    if len(sel) != len(ids):
        raise SystemExit("POS_RUNS named %d ids, matched %d" % (len(ids), len(sel)))
    return sel


def shard(sel, spec):
    """`POS_SHARD="i/n"` -- every n-th run starting at i.

    ROUND ROBIN, not a contiguous slice. The run list is sorted by depth and
    then width, so contiguous task k would get eight L=16 cells while task 0
    got eight L=3 cells, and the array's wall clock would be set by whichever
    task drew the big ones. Striding mixes the sizes into every task.
    """
    if not spec:
        return sel
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise SystemExit("POS_SHARD=%r out of range" % spec)
    return sel[i::n]


def main():
    out_dir = os.environ.get("POS_OUT") or ""
    if not out_dir:
        raise SystemExit("POS_OUT is required")
    spec = os.environ.get("POS_RUNS", "all")
    measure = os.environ.get("POS_MEASURE", "AB").upper()
    tag = os.environ.get("POS_TAG") or spec.replace(":", "_").replace(",", "-")
    if os.environ.get("POS_SHARD"):
        tag = "%s_%s" % (tag, os.environ["POS_SHARD"].replace("/", "of"))
    batch = int(os.environ.get("POS_BATCH", "64"))
    smoke = os.environ.get("POS_SMOKE") == "1"

    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "instances")

    runs = successful_runs(verbose=True)
    sel = shard(select_runs(runs, spec), os.environ.get("POS_SHARD", ""))
    rho_grid = list(RHO_GRID)
    n_per_bin = N_PER_BIN
    if smoke:
        sel = sel[:2]
        rho_grid = [0.95, 0.60]
        n_per_bin = 8
        tag = "smoke"
        cache_dir = os.path.join(out_dir, "instances_smoke")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\ndevice=%s  runs=%d  measure=%s  rho bins=%d  n_per_bin=%d  batch=%d  tag=%s"
          % (device, len(sel), measure, len(rho_grid), n_per_bin, batch, tag))
    print("out=%s" % out_dir)

    a_path = os.path.join(out_dir, "A_position_%s.csv" % tag)
    h_path = os.path.join(out_dir, "B_hazard_%s.csv" % tag)
    i_path = os.path.join(out_dir, "B_instances_%s.csv" % tag)
    s_path = os.path.join(out_dir, "B_summary_%s.csv" % tag)
    a_fh = h_fh = i_fh = None
    s_rows = []

    def _w(fh_holder, path, row, fieldnames):
        if fh_holder[0] is None:
            fh_holder[0] = open(path, "w", newline="")
            fh_holder[1] = csv.DictWriter(fh_holder[0], fieldnames=fieldnames)
            fh_holder[1].writeheader()
        fh_holder[1].writerow(row)

    a_h, h_h, i_h = [None, None], [None, None], [None, None]
    t_all = time.time()

    for ri, rec in enumerate(sel):
        model, vocab, config = load_checkpoint(rec["path"], device=device)
        sudoku_cfg = cfg_from_checkpoint(config)
        max_len = int(config["max_len"])
        cols = _run_cols(rec)
        print("\n=== [%d/%d] %s  %s seed %d  ntp=%d  acc=%.4f  in79=%s ==="
              % (ri + 1, len(sel), rec["run_id"], rec["cell"], rec["seed"],
                 rec["n_trainable_params"], rec["val_acc"], rec["in79"]))
        for rho in rho_grid:
            t0 = time.time()
            inst = instance_cache(cache_dir, sudoku_cfg, rho, n_per_bin, SEED_START)
            realized = sum(len(h) for _, _, h, _ in inst) / (len(inst) * sudoku_cfg.n_nodes)
            dd_T = sum(len(t["rounds"]) for _, _, _, t in inst) / len(inst)
            base = dict(cols, rho=float(rho), realized_rho=realized, dd_mean_T=dd_T,
                        n_instances=len(inst))

            if "A" in measure:
                counts, n_skip = measure_A(model, vocab, inst, device, max_len, batch)
                for (scheme, idx, tt), (nc, nt) in sorted(counts.items()):
                    _w(a_h, a_path, dict(base, scheme=scheme, index=idx, tok_type=tt,
                                         n_correct=nc, n_total=nt, n_skipped=n_skip),
                       list(base) + ["scheme", "index", "tok_type", "n_correct",
                                     "n_total", "n_skipped"])

            srate = float("nan")
            if "B" in measure:
                haz, recs, srate = measure_B(model, vocab, inst, device, max_len,
                                             max_len, batch)
                for j in sorted(haz):
                    d = haz[j]
                    _w(h_h, h_path, dict(base, j=j, **d),
                       list(base) + ["j"] + list(d))
                for r in recs:
                    _w(i_h, i_path, dict(cols, rho=float(rho), **r),
                       list(cols) + ["rho"] + list(r))

            s_rows.append(dict(base, solve_rate=srate, elapsed_s=round(time.time() - t0, 1)))
            print("  rho=%.2f real=%.4f T=%.2f n=%d  solve=%s  %.1fs"
                  % (rho, realized, dd_T, len(inst),
                     "%.4f" % srate if srate == srate else "-", time.time() - t0))
            for holder in (a_h, h_h, i_h):
                if holder[0] is not None:
                    holder[0].flush()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    for holder in (a_h, h_h, i_h):
        if holder[0] is not None:
            holder[0].close()
    if s_rows:
        with open(s_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(s_rows[0]))
            w.writeheader()
            w.writerows(s_rows)
    print("\nwrote %s" % ", ".join(
        os.path.basename(p) for p, holder in
        ((a_path, a_h), (h_path, h_h), (i_path, i_h)) if holder[0] is not None))
    print("and %s" % os.path.basename(s_path))
    print("total %.1f min" % ((time.time() - t_all) / 60))


if __name__ == "__main__":
    main()
