"""What fraction of the training loss at N=192 is DIRECT DEDUCTION, and what
fraction is copying an adjacency table? Light CPU inspection, no GPU, no model.

WHY THIS EXISTS. The N=192 ladder (job 53062385) reports `val_nll` and
`val_accuracy` over EVERY token, and at epoch 9 both cells sit at ~2.19 nll /
0.416 accuracy and are moving ~0.006 nll per epoch. Read against the N=48
plateau of 1.64 that looks like a stall. But trap 3 of the N=96 control says
the 1.64 plateau is an N=48 number, and this probe asks the sharper version of
that question: the two sizes' losses are not even measuring the same mixture of
tasks, because the per-token entropy of the GRAPH section grows like log N
while the reasoning section's share of the sequence does not.

THE DECOMPOSITION, which is exact given `encode_instance`'s layout:

  BEGIN_HINTS   (NODE COLOR)*         1 + 2*rho*N tokens
  BEGIN_GRAPH   (NODE (NODE){d})*     1 + N*(1+d)
  BEGIN_REASONING (BEGIN_ROUND (NODE COLOR)+)*   1 + 2*n + T
  SOLVED                              1

`encode_instance` walks `for i in range(vocab.n_nodes)`, so inside the graph
section the ROW-LEADING node token is perfectly determined by its position
(it is the row index), and each undirected edge is written TWICE -- once in
each endpoint's row. So a neighbour token naming j from row i is:

  j < i : a COPY. Row j already named i, so the prefix already entails that i
          and j are adjacent; the only freedom left is which remaining slot of
          row i it lands in (an ORDERING choice, and `--shuffle-adj` makes that
          order uniformly random rather than sorted).
  j > i : NOVEL. Nothing in the prefix constrains it beyond degree bookkeeping.

Exactly half the d*N neighbour tokens are copies, because every edge has one
lower and one higher endpoint. This is the same redundancy finding 106
measured from the other side (blocking one of the six copies of v's adjacency
costs nothing), expressed as a loss budget instead of an ablation.

WHAT IS ESTIMATED AND WHAT IS EXACT. The token COUNTS per class are exact. The
per-class entropies are an ESTIMATE of an ideal prefix-conditioned predictor:
a novel slot is scored as uniform over the nodes above i that still have
residual degree, an ordering slot as uniform over row i's not-yet-emitted
revealed neighbours, and the reasoning/hints/structural tokens as free. That
over-states the novel term slightly (it ignores the global degree constraint
coupling rows) so the resulting floor is a LOWER BOUND ON ACHIEVABLE NLL that
is itself an over-estimate -- i.e. the true floor is somewhat below what this
prints, and the conclusion it supports only gets stronger.

THE CLOSED FORM, which is the part that is analytically tractable. Of the
(d+3)N tokens, dN are neighbour slots and exactly half of those are novel
(every edge has one lower and one higher endpoint), so dN/2 tokens each carry
roughly log N nats and the graph term is

    f_graph  =  [d / (2(d+3))] * log N  =  (5/16) log N   at d = 5

which predicts 1.210 at N=48 and 1.643 at N=192 against the 1.192 and 1.640
this probe measures -- agreeing to ~1%, the residual being that the candidate
set for a novel slot is the nodes ABOVE i with residual degree, not all N.
So the irreducible part of total `val_nll` grows like log N with a coefficient
fixed by the degree alone, and NOTHING about the model enters it.

MEASURED FLOORS AND WHAT THE TRAINED CELLS ACTUALLY SCORED (history.json):

    N     FLOOR    best val_nll   excess above floor
    48    1.371    1.4840 (t08)   0.113
    96    1.625    1.7455 (t00)   0.121
    96    1.625    1.7283 (t04)   0.103
   192    1.861    2.1884 (ep 9)  0.327   <- still training

The excess is the ONLY part of total `val_nll` that is about the model, and on
the two converged rungs it is 0.103-0.121 nats out of a reported 1.48-1.75 --
about 7%. The N=48 best (1.4840) is BELOW the N=192 FLOOR (1.861), so a
PERFECT solver at N=192 would report a worse total `val_nll` than the actual
N=48 model: comparing raw `val_nll` across sizes is a category error, and the
N=192 cells sitting at 2.19 against the N=48 figure of 1.64 is not evidence of
a stall. This is the quantitative form of what `scripts/acc_vs_solve.py`
already found qualitatively -- train NLL does not order models.

REGISTERED PREDICTION, committed while job 53062385 is at epoch 9 of 50 and
BEFORE any N=192 checkpoint exists, so it cannot be fitted after the fact:

  TB1. Both N=192 cells converge to a total `val_nll` whose excess above the
       1.861 floor lands in [0.09, 0.15], i.e. total val_nll in [1.95, 2.01].
       FALSIFIER: an excess outside [0.05, 0.25] at epoch 50 on either cell.
       Rationale: the excess is 0.103-0.121 on both converged rungs and is the
       only model-dependent term. At epoch 9 it is 0.327 and falling ~0.006 an
       epoch, so the trajectory reaches the band with the budget it has.
  TB2. The per-epoch `val_accuracy` curve is NOT a usable progress signal for
       deduction at N=192: >= 70% of its tokens are graph-section slots.
       FALSIFIER: the measured graph fraction is < 0.60 at N=192.
       (Scored by this probe: 73.2%. Registered anyway because it is the
       premise TB1's reading depends on, and because a premise that is not
       written down is one that gets assumed later.)

WHAT THIS DOES NOT SHOW. The floor is a bound on TOTAL `val_nll`, not a claim
about DD competence: two models with identical excess can differ in solve rate
by a factor of eleven, which is exactly stage C's observation. The excess is a
necessary-not-sufficient statistic, so it says the ladder is on trajectory --
it does NOT say the N=192 cells will deduce.

Run: python3 scripts/token_budget_probe.py [--sizes 48 96 192] [--n 3]
"""
import argparse
import math
import sys

import numpy as np

sys.path.insert(0, "/home/users/hshunt/spin_gnn")

from spin_gnn.sudoku_dd import planted_regular_graph, removal_path
from spin_gnn.sudoku_tokens import Vocab, encode_instance
from spin_gnn.sudoku_trace import solve_with_dd_trace

DEGREE = 5
N_COLORS = 4


def graph_section_entropy(adj, n_nodes, degree):
    """(novel_nats, order_nats, n_novel, n_order) for one adjacency table.

    Walks the rows in the order `encode_instance` emits them and scores each
    neighbour slot against the prefix, exactly as the docstring describes.
    """
    residual = np.full(n_nodes, degree, dtype=int)   # unrevealed degree
    novel_nats = order_nats = 0.0
    n_novel = n_order = 0
    for i in range(n_nodes):
        nbrs = [int(j) for j in adj[i]]
        revealed = [j for j in nbrs if j < i]        # copies
        unrevealed = [j for j in nbrs if j > i]      # novel
        n_left_order = len(revealed)
        n_left_novel = len(unrevealed)
        # Slot order is uniformly random under --shuffle-adj, so score the row
        # as the mixture it is: at each slot the predictor does not know
        # whether the next token is a copy or a novel edge.
        for _ in range(len(nbrs)):
            tot = n_left_order + n_left_novel
            if tot == 0:
                break
            # Candidates for a novel slot: nodes above i with residual degree.
            cand = int(sum(1 for j in range(i + 1, n_nodes) if residual[j] > 0))
            cand = max(cand, 1)
            p_order = n_left_order / tot
            if p_order > 0:
                order_nats += p_order * math.log(max(n_left_order, 1))
                n_order += p_order
            p_novel = n_left_novel / tot
            if p_novel > 0:
                novel_nats += p_novel * math.log(cand)
                n_novel += p_novel
            # Consume one slot in expectation.
            n_left_order -= p_order
            n_left_novel -= p_novel
        for j in unrevealed:
            residual[j] -= 1
        residual[i] = 0
    return novel_nats, order_nats, n_novel, n_order


def probe(n_nodes, n_instances, seed_start):
    rows = []
    for k in range(n_instances):
        rng = np.random.default_rng(seed_start + k)
        adj, colors = planted_regular_graph(
            n_nodes, DEGREE, N_COLORS, rng, fall=True)
        path = removal_path(adj, colors, N_COLORS, rng)
        t_final = path["final_t"]
        hints = {int(v): int(colors[v]) for v in np.flatnonzero(t_final == 0)}
        trace = solve_with_dd_trace(adj, colors, hints, N_COLORS)
        assert trace["resolved"]
        vocab = Vocab(n_nodes, N_COLORS, DEGREE, scratchpad=False)
        toks = encode_instance(adj, trace, vocab)

        n_hint = len(trace["hints"])
        n_blank = n_nodes - n_hint
        T = len(trace["rounds"])
        n_graph = 1 + n_nodes * (1 + DEGREE)
        n_hints_sec = 1 + 2 * n_hint
        n_reason = 1 + 2 * n_blank + T
        total = n_hints_sec + n_graph + n_reason + 1
        assert total == len(toks), f"{total} != {len(toks)}"

        nv, no, cv, co = graph_section_entropy(adj, n_nodes, DEGREE)

        # The two terms the first version of this probe omitted, both of which
        # GROW with N, so leaving them out does not cancel in a cross-N
        # comparison. Added because the floor-relative reading below depends on
        # them, not because they are large.
        #
        # HINTS. The section comes FIRST, so its tokens are predicted from an
        # empty prefix. `encode_instance` emits `sorted(trace["hints"])`, so
        # given the SET the order is free; the set itself is one of C(N, h),
        # and the colours are bounded above by log(q) each (a proper colouring
        # constrains them, so this is an over-estimate of that half).
        hint_nats = math.lgamma(n_nodes + 1) - math.lgamma(n_hint + 1) \
            - math.lgamma(n_nodes - n_hint + 1) + n_hint * math.log(N_COLORS)
        # REASONING. DD fixes each round's SET of deductions, never their
        # within-round order -- finding 94's reason for scoring against the
        # forced set F rather than the trace order. So each round of k events
        # contributes log(k!) nats that no solver, however perfect, can avoid.
        order_reason_nats = float(sum(
            math.lgamma(len(ev) + 1) for ev in trace["rounds"]))

        rows.append({
            "total": total, "n_graph": n_graph, "n_reason": n_reason,
            "n_hints_sec": n_hints_sec, "rho": n_hint / n_nodes, "T": T,
            "novel_nats": nv, "order_nats": no, "n_novel": cv, "n_order": co,
            "hint_nats": hint_nats, "order_reason_nats": order_reason_nats,
        })

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    total, n_graph, n_reason = m("total"), m("n_graph"), m("n_reason")
    nv, no = m("novel_nats"), m("order_nats")
    hn, orn = m("hint_nats"), m("order_reason_nats")
    graph_floor = (nv + no) / total
    return {
        "n_nodes": n_nodes, "total": total,
        "graph_frac": n_graph / total,
        "reason_frac": n_reason / total,
        "hints_frac": m("n_hints_sec") / total,
        "novel_nats": nv, "order_nats": no,
        "n_novel": m("n_novel"), "n_order": m("n_order"),
        "per_novel": nv / max(m("n_novel"), 1e-9),
        "floor_graph": graph_floor,
        "floor_hints": hn / total,
        "floor_reason": orn / total,
        "floor_nll": (nv + no + hn + orn) / total,
        "rho": m("rho"), "T": m("T"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[48, 96, 192])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=91_000_000)
    args = ap.parse_args()

    out = [probe(n, args.n, args.seed_start) for n in args.sizes]

    print(f"{'N':>5} {'tokens':>8} {'graph%':>7} {'reason%':>8} {'hints%':>7} "
          f"{'f_graph':>8} {'f_hints':>8} {'f_order':>8} {'FLOOR':>7} "
          f"{'rho_min':>8} {'T':>6}")
    for r in out:
        print(f"{r['n_nodes']:>5} {r['total']:>8.0f} "
              f"{100 * r['graph_frac']:>6.1f}% {100 * r['reason_frac']:>7.1f}% "
              f"{100 * r['hints_frac']:>6.1f}% {r['floor_graph']:>8.3f} "
              f"{r['floor_hints']:>8.3f} {r['floor_reason']:>8.3f} "
              f"{r['floor_nll']:>7.3f} {r['rho']:>8.3f} {r['T']:>6.1f}")

    print("\nThe FLOOR is the graph section's irreducible content spread over "
          "ALL tokens -- what a perfect DD solver that still has to transmit "
          "a random 5-regular graph would score on total val_nll.")
    print("ln(N) for reference: " + ", ".join(
        f"N={r['n_nodes']} {math.log(r['n_nodes']):.3f}" for r in out))

    base = next((r for r in out if r["n_nodes"] == 48), None)
    if base:
        print("\nfloor relative to N=48:")
        for r in out:
            print(f"  N={r['n_nodes']:>4} floor {r['floor_nll']:.3f} "
                  f"({r['floor_nll'] / base['floor_nll']:.2f}x)")


if __name__ == "__main__":
    main()
