"""Per-node determinacy targets for stage X's determinacy loss.

WHAT THIS IS FOR. Findings 68 and 69 measured that the shared-block model S1
applies DD's elimination rule flawlessly and does not run DD's *scheduler*: it
picks a node to emit without checking that the node is actually determined,
and it does so even though finding 69 showed the determinacy IS computed
(p_freed lands at ~0.52 under an edit that destroys it, i.e. correctly
uncertain, and the model emits anyway). "Present and unused" is the phrase.

The main next-token loss cannot teach the missing half, and the reason is a
gauge argument. Cross-entropy sees the readout's logit vector only through a
softmax, so it constrains the logits only up to an additive constant: it says
"the next node outranks the others" and says nothing whatever about the
absolute height of any logit. "Is this node emittable at all" is precisely a
statement about absolute height, so it lives in the one direction the main
loss never touches. That is where a guard would have to live, and nothing
trains it.

So this module supplies the targets that fix the gauge. For every position
where the model is about to emit a node, it labels every still-uncoloured node
with whether that node is deducible RIGHT NOW from the prefix the causal mask
exposes:

    determined(p, v)  <=>  v is uncoloured at p
                           and v's coloured neighbours show exactly
                           n_colors - 1 distinct colours

which is DD's own deducibility predicate (the scalar form is
sudoku_skip._is_forced, and sudoku_activations.dd_state_labels documents why
n_known_colors and not n_known_nbrs is the quantity DD thresholds).

WHY NOT JUST CALL dd_state_labels. It computes eight (P, N) arrays and
re-encodes the instance to find its own positions; at 240,000 instances that
is work we do not need. This is the lean path, and
tests/test_sudoku_determinacy.py pins it against dd_state_labels as the
oracle on random instances, so the tested function stays the definition.

THE TARGET ENCODING is one int8 (P, N) array:

    -1  v is already coloured at this position -- masked OUT of the loss
     0  v is uncoloured and not deducible yet  -- a negative
     1  v is uncoloured and deducible now      -- a positive

Already-coloured nodes are masked rather than labelled 0 because they are
trivially undeducible (they are done), and at N=48 they are five sixths of
every row -- letting them in would hand the loss most of its targets for
free. What is left is exactly DD's scheduler question: among the nodes still
open, which ones can be named? Measured on 200 instances at N=48 d=5 q=4
fall shuffle_adj, that question has 7.5 open nodes in it at the median
position, of which 4.9 are deducible, so the target is neither constant nor
rare in either class.

THE LOSS ATTACHES TO THE READOUT THE MODEL ALREADY HAS, not to a new head.
The node-token logits at a pre_node position are trained as a multi-label
determinacy indicator at the same time as they are trained, by the main
cross-entropy, to rank the next node first. No parameter is added: a stage-X
cell is S1's architecture exactly, 613,818 trainable, and the state dict
round-trips through load_checkpoint unchanged. What changes is only what the
absolute logit height is asked to mean.

NOTHING HERE DEPENDS ON A MODEL. The target is a deterministic function of
(adj, trace) and draws from no rng stream, so adding it to a training run
leaves the instance stream, the weight init and the batch order
bit-identical; at weight 0 a stage-X cell is its stage-S cell.
"""
import numpy as np
import torch.nn.functional as F

from .sudoku_activations import reasoning_positions

# Values of the int8 target. MASKED is negative so that `target >= 0` is the
# loss mask and `target == 1` the positive class, with no second array.
MASKED = -1


def determinacy_targets(adj, trace, vocab, tokens):
    """Determinacy target at every pre_node position of one instance.

    adj     (N, degree) int   neighbour table, as build_instances returns it
    trace   dict              solve_with_dd_trace output
    vocab   Vocab
    tokens  (T,) int64        encode_instance(adj, trace, vocab), passed in
                              rather than recomputed -- the caller already has
                              it, and encoding twice per instance is the one
                              cost this module exists to avoid

    Returns (pos, target):
      pos     (P,)    int64, token indices. tokens[pos[i] + 1] is a node id,
                      so a model whose input is tokens[:-1] reads position
                      pos[i] of its own input to predict that node -- the
                      same convention sudoku_activations documents, and the
                      reason no index shift is needed anywhere downstream.
      target  (P, N)  int8 in {-1, 0, 1} as described in the module docstring.

    P is the number of deductions in the trace, i.e. one position per node the
    trace ever emits -- NOT per token.
    """
    n_nodes, n_colors = vocab.n_nodes, vocab.n_colors
    adj = np.asarray(adj)
    positions = [p for p in reasoning_positions(tokens, vocab) if p["kind"] == "pre_node"]
    rounds = [[(int(u), int(c)) for u, c in rd] for rd in trace["rounds"]]

    # Known colours at the START of each round, exactly as DD had them.
    pre_round_known = []
    known = {int(u): int(c) for u, c in trace["hints"].items()}
    for rd in rounds:
        pre_round_known.append(dict(known))
        known.update(dict(rd))

    P = len(positions)
    pos = np.zeros(P, dtype=np.int64)
    target = np.zeros((P, n_nodes), dtype=np.int8)
    # Scratch reused across positions; (N, n_colors + 1) with the last column
    # standing for "neighbour's colour unknown", which is never counted.
    seen = np.zeros((n_nodes, n_colors + 1), dtype=bool)
    row_ix = np.repeat(np.arange(n_nodes), adj.shape[1])

    for p, info in enumerate(positions):
        pos[p] = info["pos"]
        # What the model can see here: everything known before this round, plus
        # the deductions of this round already written out. info["k"] of them.
        cur = dict(pre_round_known[info["round"]])
        cur.update(dict(rounds[info["round"]][: info["k"]]))

        node_color = np.full(n_nodes, n_colors, dtype=np.int64)
        if cur:
            idx = np.fromiter(cur.keys(), dtype=np.int64, count=len(cur))
            val = np.fromiter(cur.values(), dtype=np.int64, count=len(cur))
            node_color[idx] = val
        seen[:] = False
        seen[row_ix, node_color[adj].ravel()] = True
        n_known_colors = seen[:, :n_colors].sum(1)

        is_known = node_color < n_colors
        target[p] = (n_known_colors == n_colors - 1).astype(np.int8)
        target[p][is_known] = MASKED
    return pos, target


def node_logits_at(logits, pos, node_offset, n_nodes):
    """Gather the node-token block of the readout at the given positions.

    logits (B, L, V) -> (B, P, n_nodes), where pos is (B, P) long. Node tokens
    occupy one contiguous block of the vocabulary (Vocab.node_offset onwards),
    so the per-node scores are a slice and not a gather.
    """
    sel = pos.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    return logits.gather(1, sel)[..., node_offset : node_offset + n_nodes]


def determinacy_loss(logits, pos, target, node_offset, n_nodes):
    """Multi-label BCE on the readout's node logits against DD determinacy.

    target (B, P, n_nodes) int8 as this module defines it: cells < 0 are
    dropped, so padded positions cost nothing as long as the caller pads with
    MASKED. Returns None when a batch happens to score no cell at all, which
    the training loop reads as "no determinacy term this step" rather than as
    a zero.
    """
    scored = target >= 0
    if not bool(scored.any()):
        return None
    node_logits = node_logits_at(logits, pos, node_offset, n_nodes)
    return F.binary_cross_entropy_with_logits(
        node_logits[scored], (target[scored] == 1).to(node_logits.dtype))


def auroc(scores, labels):
    """Rank AUROC of `scores` against boolean `labels`, ties averaged.

    Threshold-free on purpose: the positive rate is 0.65, so an accuracy at
    0.5 would mostly report the base rate. nan when either class is empty.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # Average the ranks inside each tie group, or a head that outputs one
    # constant would score anything but 0.5.
    s_sorted = scores[order]
    start = 0
    for i in range(1, s_sorted.size + 1):
        if i == s_sorted.size or s_sorted[i] != s_sorted[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + 1 + i) / 2.0
            start = i
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def summarize(targets):
    """Positive / negative / masked counts over a list of (P, N) targets.

    Printed once at dataset build time: if the positive rate is not a few
    percent the label is wrong, and that is cheaper to see before 8 hours of
    training than after.
    """
    pos = neg = msk = 0
    for t in targets:
        pos += int((t == 1).sum())
        neg += int((t == 0).sum())
        msk += int((t == MASKED).sum())
    scored = pos + neg
    return {
        "positives": pos, "negatives": neg, "masked": msk,
        "positive_rate": (pos / scored) if scored else float("nan"),
        "scored_fraction": scored / (scored + msk) if (scored + msk) else float("nan"),
    }
