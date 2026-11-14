"""WHERE IN THE ITERATION IS THE DECISION MADE? A lens and a patch, per application.

Findings 68-71 are all decision-level: they read the model's output and say
what it does, never where in the computation it does it. STATUS has carried
"no localisation" as a caveat on four decks, with the reason that weight-space
ablation is undefined on a tied block -- one set of weights shared by all 16
applications, so zeroing a head zeroes it in every round at once and
`sudoku_circuits._require_untied` refuses to pretend otherwise.

That reason does not apply to ACTIVATION interventions. The tied model's
forward is `for t in range(n_iters): h = blk(h)`, so the state after the t-th
application is a well-defined object in both arms, and `sudoku_circuits`
already hooks it by counting calls rather than by module index. The axis is the
APPLICATION INDEX, not the layer, and it is the same axis in both arms because
both have depth 16. This module reads along it two ways:

  THE LENS, one forward per variant. Capture h_t for every t and apply the
  model's own final norm and readout to the two positions the earlier findings
  read -- the node position (who gets named) and the colour position (what
  colour v gets). This gives emits_v(t), p_freed(t), p_cand(t) as trajectories.
  The lens is exact at t = depth-1 by construction, which is the gate: if the
  last-application lens does not reproduce the model's own output bit for bit,
  the readout is being applied wrongly and nothing else here may be read.

  THE PATCH, 16 forwards per site per case. Run the BROKEN prompt and splice
  the CLEAN activation back in at one (application, position). Recovery is the
  fraction of cases where the clean behaviour returns. Reading recovery against
  t gives a deadline: the last application at which repairing the evidence
  still changes the answer is the application by which the decision has been
  taken.

Two sites, because the question has two ends:
  edit -- the hint colour token the edit rewrote. Repairing it asks WHEN the
          broken evidence has to be correct.
  read -- the node position, where the guard would act. Repairing it asks when
          the DECISION itself is still movable.

The case list is finding 68's, byte for byte: the same generator stream
(seed_start 600000000), the same finder, the same cap, and `build_quad` from
sudoku_dissoc, so clean and broken are the same two token arrays findings 68,
69 and (for the round-0 rows) 71 measured. Nothing is trained and no published
number can move.

The broken edit frees a second colour for v, so DD must not name v at all.
Finding 68: the untied model stops naming it (0.0000 / 0.0000 / 0.8444) and the
tied model never does (1.0000 x3). So the two arms are asking different
questions of the same measurement, and the registration says so:

  the UNTIED arm has a guard to localise, and the patch asks where it is;
  the TIED arm has no guard, but finding 69 showed it computes the ambiguity
  anyway (p_cand exactly 1.0000, p_freed ~0.52) -- so the patch asks where
  THAT appears instead. The shared block is the object the project wants a
  theory of, and "the candidate set is resolved by application k" is a
  statement about the iterated map rather than about a stack.

Usage:
    python3 -m spin_gnn.sudoku_lens --checkpoint CKPT --out-dir DIR
    python3 -m spin_gnn.sudoku_lens --checkpoint CKPT --cases-only   # census
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from .config import SudokuReasoningConfig
from .sat_transformer2 import is_tied as _is_tied, model_depth as _depth
from .sudoku_dissoc import build_quad, rule_edit_valid
from .sudoku_skip import find_unforceable_deductions
from .train_sudoku_transformer import build_instances, load_checkpoint

# Only the two variants this module needs. The placebo and rule variants are
# finding 69's controls and are not re-measured here: that run already passed
# them on this exact case list, and re-running a passed control on the same
# cases is not a second piece of evidence.
VARIANTS = ("clean", "broken")

# The two sites, in the order every table prints them.
SITES = ("edit", "read")


def _case_key(case):
    """The four fields that identify a case, as ddguard_score.CASE_KEY has them.

    `node` ALONE DOES NOT IDENTIFY A CASE: it is a node index in 0..47 and the
    stream yields hundreds of cases across 400 instances, so node 7 recurs in
    dozens of them. Every row therefore carries a sequential `case_id`, which
    is what the scorer groups on, plus these four so a row joins to finding
    68's and finding 71's CSVs, which are keyed this way.
    """
    return {"node": case["node"], "binding_nbr": case["recolored_nbr"],
            "n_known": case["n_known"], "round0_rank": case["round0_rank"]}


def _norm_of(model):
    """The final norm the model's own forward applies before the readout.

    Tied path: the loop calls the block directly and then `encoder.norm`.
    Untied path: `nn.TransformerEncoder` applies the same `norm` after its last
    layer. Either way the hook on a block captures the PRE-norm state, so the
    lens has to apply this to be the model's own readout and not a different
    one that happens to share weights.
    """
    norm = getattr(model.encoder, "norm", None)
    return (lambda h: h) if norm is None else norm


def _capture(model, tokens, device, sites=None):
    """(activations after every application, the model's real logits).

    sudoku_circuits._capture_clean with the logits kept instead of discarded,
    so the lens trajectory and the behaviour it must agree with at t = depth-1
    come from ONE forward and cannot drift apart. Only the columns in `sites`
    are kept when given: the full stream is 16 x L x d per case and only two
    positions are ever read.
    """
    acts = {}
    keep = None if sites is None else sorted(set(sites))

    if _is_tied(model):
        calls = [0]

        def _hook(_m, _i, o):
            acts[calls[0]] = (o if keep is None else o[:, keep]).detach().clone()
            calls[0] += 1

        handles = [model.encoder.layers[0].register_forward_hook(_hook)]
    else:
        handles = [
            model.encoder.layers[i].register_forward_hook(
                lambda _m, _i, o, i=i: acts.__setitem__(
                    i, (o if keep is None else o[:, keep]).detach().clone())
            )
            for i in range(len(model.encoder.layers))
        ]
    try:
        with torch.no_grad():
            logits = model(torch.from_numpy(tokens).unsqueeze(0).to(device))
    finally:
        for h in handles:
            h.remove()
    assert len(acts) == _depth(model), (
        "captured %d activations for a depth-%d model" % (len(acts), _depth(model)))
    return acts, logits


def _read_at(logits, pos, v_tok, col_toks, c, a):
    """The two readouts, identical in definition to sudoku_dissoc._read.

    Kept to the same four quantities on purpose: emits_v is finding 68's guard
    statistic and p_freed/p_cand are finding 69's, so a number here is
    comparable to the number those findings published without a conversion.
    """
    node_row = logits[0, pos]
    probs = torch.softmax(logits[0, pos + 1].float(), dim=-1)
    col_probs = [float(probs[t]) for t in col_toks]
    return {
        "emits_v": int(int(node_row.argmax()) == v_tok),
        "v_logit": float(node_row[v_tok]),
        "p_true": col_probs[c],
        "p_freed": col_probs[a],
        "p_cand": col_probs[c] + col_probs[a],
    }


def run_lens(model, vocab, device, cases):
    """Per-application trajectories, one forward per (case, variant).

    Row t carries the relative movement the t-th application PRODUCED at each
    read position, ||h_t - h_{t-1}|| / ||h_t-1||, and so is nan at t = 0: the
    hook fires on block outputs, and the embedding that application 0 consumed
    is not one of them. The sum over t is therefore the movement after the
    first application, which is the quantity a fixed-point claim is about.
    """
    norm = _norm_of(model)
    depth = _depth(model)
    rows = []
    for cid, (adj, colors, hints, trace, case) in enumerate(cases):
        quad, pos = build_quad(adj, trace, hints, vocab, case)
        v_tok = vocab.node_token(case["node"])
        col_toks = [vocab.color_token(i) for i in range(vocab.n_colors)]
        c, a = case["true_color"], case["freed_color"]
        for name in VARIANTS:
            # Keep both read positions AND the two columns the lens needs; the
            # capture is sliced to [pos, pos+1] so a 16 x L x d stream never
            # exists for a case.
            acts, real = _capture(model, quad[name], device, sites=[pos, pos + 1])
            truth = _read_at(real, pos, v_tok, col_toks, c, a)
            prev = None
            for t in range(depth):
                h = acts[t]                       # [1, 2, d]: pos, pos+1
                with torch.no_grad():
                    lens = model.readout(norm(h))
                # lens columns 0,1 are the model's pos, pos+1, so _read_at's
                # (pos, pos+1) indexing becomes (0, 1) on the sliced tensor.
                r = _read_at(lens, 0, v_tok, col_toks, c, a)
                d_node = d_col = float("nan")
                if prev is not None:
                    for j, key in ((0, "node"), (1, "col")):
                        num = float(torch.linalg.vector_norm(h[0, j] - prev[0, j]))
                        den = float(torch.linalg.vector_norm(prev[0, j])) or float("nan")
                        if key == "node":
                            d_node = num / den
                        else:
                            d_col = num / den
                prev = h
                rows.append({
                    "case_id": cid, "variant": name, "t": t,
                    **_case_key(case),
                    "emits_v": r["emits_v"], "v_logit": r["v_logit"],
                    "p_true": r["p_true"], "p_freed": r["p_freed"],
                    "p_cand": r["p_cand"],
                    "delta_node": d_node, "delta_col": d_col,
                    # the gate: at t = depth-1 these must equal the lens row
                    "final_emits_v": truth["emits_v"],
                    "final_p_freed": truth["p_freed"],
                })
    return rows


def run_patch(model, vocab, device, cases):
    """Splice the CLEAN activation into the BROKEN run at one (application, site).

    One row per (case, site, t). The per-case clean and broken readouts are
    written on every row so the scorer can form recovery without joining
    against the trajectory file, and so a row is self-contained if the files
    are ever read apart.
    """
    depth = _depth(model)
    tied = _is_tied(model)
    rows = []
    for cid, (adj, colors, hints, trace, case) in enumerate(cases):
        quad, pos = build_quad(adj, trace, hints, vocab, case)
        v_tok = vocab.node_token(case["node"])
        col_toks = [vocab.color_token(i) for i in range(vocab.n_colors)]
        c, a = case["true_color"], case["freed_color"]
        site_pos = {"edit": int(case["hint_color_pos"]), "read": int(pos)}
        assert site_pos["edit"] != site_pos["read"], "sites must be distinct"

        clean_acts, clean_lg = _capture(model, quad["clean"], device)
        base_clean = _read_at(clean_lg, pos, v_tok, col_toks, c, a)
        broken_t = torch.from_numpy(quad["broken"]).unsqueeze(0).to(device)
        with torch.no_grad():
            base_broken = _read_at(model(broken_t), pos, v_tok, col_toks, c, a)

        for site in SITES:
            p = site_pos[site]
            for t in range(depth):
                if tied:
                    # The hook is on the ONE shared block and fires every
                    # application, so splice on the t-th call only. Without the
                    # call counter the patch would land at all 16 depths at
                    # once and the row would not mean what the untied row means.
                    calls = [0]

                    def hook(_m, _i, o, t=t, p=p, calls=calls):
                        i, calls[0] = calls[0], calls[0] + 1
                        if i != t:
                            return None
                        o = o.clone()
                        o[:, p] = clean_acts[t][:, p]
                        return o
                else:
                    def hook(_m, _i, o, t=t, p=p):
                        o = o.clone()
                        o[:, p] = clean_acts[t][:, p]
                        return o

                h = model.encoder.layers[0 if tied else t].register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        r = _read_at(model(broken_t), pos, v_tok, col_toks, c, a)
                finally:
                    h.remove()
                rows.append({
                    "case_id": cid, "site": site, "t": t,
                    **_case_key(case),
                    "emits_v": r["emits_v"], "p_freed": r["p_freed"],
                    "p_cand": r["p_cand"], "v_logit": r["v_logit"],
                    "clean_emits_v": base_clean["emits_v"],
                    "broken_emits_v": base_broken["emits_v"],
                    "clean_p_freed": base_clean["p_freed"],
                    "broken_p_freed": base_broken["p_freed"],
                })
    return rows


def run_noop(model, vocab, device, cases):
    """GUARD A2: the same splice, clean into CLEAN, must change nothing.

    The patch is only readable if the hook writes exactly what the forward
    would have written anyway when the source and target runs are the same. A
    hook that silently shifted a position, or cloned the wrong tensor, would
    produce a clean-looking recovery curve out of its own bug. Run on a small
    subset because it is a code check, not a measurement.
    """
    depth = _depth(model)
    tied = _is_tied(model)
    rows = []
    for cid, (adj, colors, hints, trace, case) in enumerate(cases):
        quad, pos = build_quad(adj, trace, hints, vocab, case)
        v_tok = vocab.node_token(case["node"])
        col_toks = [vocab.color_token(i) for i in range(vocab.n_colors)]
        c, a = case["true_color"], case["freed_color"]
        clean_acts, clean_lg = _capture(model, quad["clean"], device)
        base = _read_at(clean_lg, pos, v_tok, col_toks, c, a)
        clean_t = torch.from_numpy(quad["clean"]).unsqueeze(0).to(device)
        for site, p in (("edit", int(case["hint_color_pos"])), ("read", int(pos))):
            for t in range(depth):
                if tied:
                    calls = [0]

                    def hook(_m, _i, o, t=t, p=p, calls=calls):
                        i, calls[0] = calls[0], calls[0] + 1
                        if i != t:
                            return None
                        o = o.clone()
                        o[:, p] = clean_acts[t][:, p]
                        return o
                else:
                    def hook(_m, _i, o, t=t, p=p):
                        o = o.clone()
                        o[:, p] = clean_acts[t][:, p]
                        return o

                h = model.encoder.layers[0 if tied else t].register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        r = _read_at(model(clean_t), pos, v_tok, col_toks, c, a)
                finally:
                    h.remove()
                rows.append({
                    "case_id": cid, "site": site, "t": t,
                    **_case_key(case),
                    "same_emits_v": int(r["emits_v"] == base["emits_v"]),
                    "abs_freed_drift": abs(r["p_freed"] - base["p_freed"]),
                })
    return rows


def collect_cases(sudoku_cfg, vocab, n_instances, seed_start):
    """Finding 68's case list: same rng, same stream, same finder, same cap."""
    rng = np.random.default_rng(0)
    cases = []
    for adj, colors, hints, trace in build_instances(sudoku_cfg, n_instances, seed_start):
        for case in find_unforceable_deductions(adj, colors, hints, trace, vocab, rng):
            case["n_colors"] = vocab.n_colors
            case["rule_valid"] = rule_edit_valid(adj, trace, hints, case)
            cases.append((adj, colors, hints, trace, case))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n-instances", type=int, default=400)
    ap.add_argument("--n-cases", type=int, default=350,
                    help="finding 68's cap, for the cheap lens pass")
    ap.add_argument("--n-patch", type=int, default=150,
                    help="subset for the causal pass: 2 sites x depth forwards each")
    ap.add_argument("--n-noop", type=int, default=10,
                    help="subset for guard A2, a code check and not a measurement")
    ap.add_argument("--seed-start", type=int, default=600_000_000,
                    help="finding 68's stream, unchanged, so the case list is identical")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--cases-only", action="store_true",
                    help="census: count cases and stop, before any cell pays for them")
    args = ap.parse_args()

    model, vocab, config = load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    assert bool(config.get("shuffle_adj", False)), (
        "checkpoint was not trained with shuffle_adj; the eval grid is not comparable")
    sudoku_cfg = SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        rho=config["rho"], rho_max=config["rho_max"],
        scratchpad=config.get("scratchpad", False),
        hint_mode=config.get("hint_mode", "removal"),
        shuffle_adj=bool(config.get("shuffle_adj", False)),
        shuffle_events=bool(config.get("shuffle_events", False)),
        fall=bool(config.get("fall", False)),
    )

    depth = _depth(model)
    cases = collect_cases(sudoku_cfg, vocab, args.n_instances, args.seed_start)
    print("%d breakable round-0 cases from %d instances" % (len(cases), args.n_instances))
    print("depth %d applications, tied=%s" % (depth, _is_tied(model)))
    if args.cases_only:
        return
    if not cases:
        raise SystemExit("no breakable cases found")
    if args.out_dir is None:
        raise SystemExit("--out-dir is required unless --cases-only")
    os.makedirs(args.out_dir, exist_ok=True)

    noop = run_noop(model, vocab, args.device, cases[:args.n_noop])
    bad = [r for r in noop if not r["same_emits_v"]]
    drift = max(r["abs_freed_drift"] for r in noop)
    print("\nA2 clean-into-clean no-op: %d/%d rows unchanged, max |p_freed| drift %.2e"
          % (len(noop) - len(bad), len(noop), drift))
    if bad or drift > 1e-4:
        raise SystemExit("A2 FAILED: the splice is not a no-op on its own source run")

    lens = run_lens(model, vocab, args.device, cases[:args.n_cases])
    last = [r for r in lens if r["t"] == depth - 1]
    agree = float(np.mean([r["emits_v"] == r["final_emits_v"] for r in last]))
    maxdiff = max(abs(r["p_freed"] - r["final_p_freed"]) for r in last)
    print("L1 lens fidelity at t=%d: emits_v agreement %.4f over %d rows, "
          "max |p_freed| gap %.2e" % (depth - 1, agree, len(last), maxdiff))
    if agree < 1.0 or maxdiff > 1e-4:
        raise SystemExit("L1 FAILED: the lens is not the model's own readout")

    patch = run_patch(model, vocab, args.device, cases[:args.n_patch])
    disc = {r["case_id"] for r in patch
            if r["clean_emits_v"] and not r["broken_emits_v"]}
    print("L3 discordant cases (clean names v, broken does not): %d of %d"
          % (len(disc), min(args.n_patch, len(cases))))

    for name, rows in (("lens_traj", lens), ("lens_patch", patch), ("lens_noop", noop)):
        path = os.path.join(args.out_dir, "%s.csv" % name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote %s (%d rows)" % (path, len(rows)))

    with open(os.path.join(args.out_dir, "lens_meta.json"), "w") as f:
        json.dump({"checkpoint": args.checkpoint, "depth": depth,
                   "tied": bool(_is_tied(model)), "seed_start": args.seed_start,
                   "n_cases": min(args.n_cases, len(cases)),
                   "n_patch": min(args.n_patch, len(cases)),
                   "n_discordant": len(disc),
                   "lens_agree": agree, "noop_max_drift": drift}, f, indent=2)


if __name__ == "__main__":
    main()
