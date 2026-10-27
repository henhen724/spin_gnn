"""Residual-stream capture + ground-truth DD state labels for DD-Sudoku traces.

The foundation for both the SAE work (sudoku_sae.py) and the linear probes
(sudoku_probes.py). What makes this problem unusually good for mechanistic
work is that the target algorithm is known exactly: at every token position
in a trace we can compute what deduction-by-DD *would* know, so "did the
transformer learn DD" becomes a concrete decodability/causality question
rather than a vibes question about attention maps.

The central distinction this module is built to support:

  known_mask       -- which nodes have a color yet. Pure bookkeeping; a model
                      could track this by copying and position alone.
  frontier_mask    -- which nodes are deducible RIGHT NOW. This is the DD
                      predicate itself: it requires combining the graph
                      structure with the known set (a node is forced exactly
                      when its known neighbors cover n_colors-1 colors). A
                      model that linearly represents this is doing DD; one
                      that only represents known_mask is not.

A useful identity makes the labels exact rather than reconstructed: DD
deduces node u in round r using only colors known strictly before round r,
so the frontier at round r is precisely the set of nodes that round r
emits -- trace["rounds"][r]. No re-derivation, no ambiguity.

Positions are indexed so that residual[pos] is the state from which the model
predicts token[pos + 1], i.e. a "pre_node" position is the one whose next
token is the node id of a deduction.
"""
import numpy as np
import torch

from .sudoku_tokens import encode_instance


def reasoning_positions(tokens, vocab):
    """Walk an encoded instance and locate every point inside the reasoning
    section where the model is about to emit a deduction's node or color.

    Returns a list of dicts with:
      pos    -- index such that tokens[pos + 1] is the token being predicted
      kind   -- "pre_node" (next token is the deduced node id) or "pre_color"
      round  -- 0-based round index
      k      -- index of this deduction within its round
    Works for both the plain and the scratchpad grammar; in scratchpad mode
    the evidence row sits between BEGIN_EVIDENCE and the deduction pair, and
    only the final (node, color) pair of each row is labelled here."""
    tokens = np.asarray(tokens)
    begin = np.flatnonzero(tokens == vocab.BEGIN_REASONING)
    if begin.size == 0:
        return []
    out = []
    r = -1
    k = 0
    i = int(begin[0]) + 1
    n = len(tokens)
    while i < n:
        t = int(tokens[i])
        if t == vocab.BEGIN_ROUND:
            r += 1
            k = 0
            i += 1
            continue
        if t in (vocab.SOLVED, vocab.UNRESOLVED):
            break
        if getattr(vocab, "scratchpad", False) and t == vocab.BEGIN_EVIDENCE:
            # BEGIN_EVIDENCE, node, <degree evidence colors>, then the pair
            i += 1 + 1 + vocab.degree
            continue
        if vocab.is_node_token(t) and r >= 0:
            # tokens[i] is the deduced node, tokens[i+1] its color
            out.append({"pos": i - 1, "kind": "pre_node", "round": r, "k": k})
            if i + 1 < n:
                out.append({"pos": i, "kind": "pre_color", "round": r, "k": k})
            k += 1
            i += 2
            continue
        i += 1
    return out


def dd_state_labels(adj, trace, vocab):
    """Ground-truth DD state at each labelled position of one instance.

    Returns (positions, labels) where labels is a dict of arrays, each with
    leading axis = len(positions):
      next_node        (P,)      node id the model must emit next
      next_color       (P,)      its forced color
      frontier_mask    (P, N)    the full DD frontier of this round
      remaining_mask   (P, N)    frontier members not yet emitted this round
      known_mask       (P, N)    nodes with a color known at this position
      node_color       (P, N)    known color per node, n_colors = "unknown"
      n_known_nbrs     (P, N)    how many of each node's neighbors are known
      n_known_colors   (P, N)    how many DISTINCT colors v's known neighbors
                                 show. This, not n_known_nbrs, is the quantity
                                 DD actually thresholds: v is deducible iff its
                                 known neighbors cover n_colors-1 distinct
                                 colors. Two known neighbors of the same color
                                 advance n_known_nbrs and not the deduction.
      nbr_of_next_mask (P, N)    u is a neighbor of the node about to be emitted
      justifier_mask   (P, N)    u is a KNOWN neighbor of the node about to be
                                 emitted, i.e. one of the edges that justifies
                                 this deduction. The edge-level ground truth:
                                 a feature tracking this is looking at the right
                                 edges at the right time.
      round_index      (P,)      0-based round
      kind_is_node     (P,)      True for pre_node, False for pre_color
    """
    n_nodes, n_colors = vocab.n_nodes, vocab.n_colors
    tokens = encode_instance(adj, trace, vocab)
    positions = reasoning_positions(tokens, vocab)
    rounds = [[(int(u), int(c)) for u, c in rd] for rd in trace["rounds"]]

    # Known set at the START of each round (what DD actually had available).
    pre_round_known = []
    known = {int(u): int(c) for u, c in trace["hints"].items()}
    for rd in rounds:
        pre_round_known.append(dict(known))
        known.update(dict(rd))

    P = len(positions)
    lab = {
        "next_node": np.zeros(P, dtype=np.int64),
        "next_color": np.zeros(P, dtype=np.int64),
        "frontier_mask": np.zeros((P, n_nodes), dtype=np.int8),
        "remaining_mask": np.zeros((P, n_nodes), dtype=np.int8),
        "known_mask": np.zeros((P, n_nodes), dtype=np.int8),
        "node_color": np.full((P, n_nodes), n_colors, dtype=np.int64),
        "n_known_nbrs": np.zeros((P, n_nodes), dtype=np.int64),
        "n_known_colors": np.zeros((P, n_nodes), dtype=np.int64),
        "nbr_of_next_mask": np.zeros((P, n_nodes), dtype=np.int8),
        "justifier_mask": np.zeros((P, n_nodes), dtype=np.int8),
        "round_index": np.zeros(P, dtype=np.int64),
        "kind_is_node": np.zeros(P, dtype=bool),
    }
    for p, info in enumerate(positions):
        r, k = info["round"], info["k"]
        rd = rounds[r]
        # State visible to the model here: everything known before the round,
        # plus the deductions of this round it has already written out. For a
        # "pre_color" position the node itself is already on the page but its
        # color is not, so it is not yet known.
        # Same count for both kinds: at a pre_color position the node id is on
        # the page but its color is not, so no new color is known yet.
        emitted = k
        cur = dict(pre_round_known[r])
        cur.update(dict(rd[:emitted]))

        node, color = rd[min(k, len(rd) - 1)]
        lab["next_node"][p] = node
        lab["next_color"][p] = color
        lab["round_index"][p] = r
        lab["kind_is_node"][p] = info["kind"] == "pre_node"
        for u in (n for n, _ in rd):
            lab["frontier_mask"][p, u] = 1
        for u in (n for n, _ in rd[emitted:]):
            lab["remaining_mask"][p, u] = 1
        for u, c in cur.items():
            lab["known_mask"][p, u] = 1
            lab["node_color"][p, u] = c
        # Neighbor colors as a (N, degree) table, n_colors standing for unknown.
        # Vectorized because the per-node Python loop this replaces was O(P*N*d)
        # and dominated collect_dataset at N=96.
        nc = lab["node_color"][p]
        nbr_col = nc[adj]
        lab["n_known_nbrs"][p] = (nbr_col < n_colors).sum(1)
        seen = np.zeros((n_nodes, n_colors + 1), dtype=bool)
        seen[np.repeat(np.arange(n_nodes), adj.shape[1]), nbr_col.ravel()] = True
        lab["n_known_colors"][p] = seen[:, :n_colors].sum(1)
        # The edges in play at THIS position: the neighborhood of the node about
        # to be emitted, and the known part of it that forces its color.
        nbrs_of_next = adj[node]
        lab["nbr_of_next_mask"][p, nbrs_of_next] = 1
        lab["justifier_mask"][p, nbrs_of_next[nc[nbrs_of_next] < n_colors]] = 1
    return positions, lab


class ResidualCapture:
    """Forward hooks on every encoder block, plus the embedding input.

    Layer indexing: 0 = token+position embedding (no attention or MLP has
    run), i >= 1 = output of block i-1. Index 0 is the control -- a probe
    that already succeeds there is reading the tokenization, not a learned
    computation."""

    def __init__(self, model):
        self.model = model
        self._acts = {}
        self._handles = []

    def __enter__(self):
        def mk(i):
            def hook(_mod, _inp, out):
                self._acts[i] = out.detach()
            return hook

        for i, block in enumerate(self.model.encoder.layers):
            self._handles.append(block.register_forward_hook(mk(i + 1)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def run(self, tokens):
        """tokens: (B, L) long tensor. Returns (n_layers + 1, B, L, d_model)."""
        self._acts = {}
        with torch.no_grad():
            emb = self.model.token_embed(tokens) + self.model.pos_embed[: tokens.shape[1]]
            self.model(tokens)
        self._acts[0] = emb.detach()
        return torch.stack([self._acts[i] for i in sorted(self._acts)], dim=0)


def collect_dataset(model, instances, vocab, device, max_positions_per_instance=None, rng=None):
    """Residual activations at every labelled position, with DD labels.

    Returns (acts, labels) where acts is (n_layers + 1, P_total, d_model) on
    CPU and labels is the per-position label dict concatenated across
    instances. Runs one instance per forward pass (sequences are a single
    instance each and lengths differ), teacher-forced on the TRUE trace --
    this measures what the model represents while reading a correct
    derivation, which is the right setting for asking whether DD state is
    present at all."""
    rng = rng or np.random.default_rng(0)
    per_layer, label_chunks = [], []
    with ResidualCapture(model) as cap:
        for adj, _colors, _hints, trace in instances:
            positions, lab = dd_state_labels(adj, trace, vocab)
            if not positions:
                continue
            idx = np.arange(len(positions))
            if max_positions_per_instance and len(idx) > max_positions_per_instance:
                idx = np.sort(rng.choice(idx, size=max_positions_per_instance, replace=False))
            tokens = torch.as_tensor(
                encode_instance(adj, trace, vocab)[None, :], dtype=torch.long, device=device,
            )
            acts = cap.run(tokens)  # (L+1, 1, T, d)
            pos = np.array([positions[i]["pos"] for i in idx], dtype=np.int64)
            per_layer.append(acts[:, 0, pos, :].cpu())
            label_chunks.append({k: v[idx] for k, v in lab.items()})
    if not per_layer:
        raise ValueError("no labelled positions found across the given instances")
    acts = torch.cat(per_layer, dim=1)
    labels = {k: np.concatenate([c[k] for c in label_chunks], axis=0) for k in label_chunks[0]}
    return acts, labels
