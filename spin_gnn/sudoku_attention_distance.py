"""Compares the token distance DD's own deduction process needs to reference
against the trained transformer's learned attention-weight decay with token
distance.

Motivation (STATUS.md, user hypothesis): sudoku_tokens.encode_instance lays
out BEGIN_GRAPH (every node's neighbor list) and BEGIN_HINTS once, near the
start of the sequence, but every later BEGIN_REASONING round's deductions
depend on that fixed-position information -- so the token distance a
deduction must reference grows as the trace gets deeper (up to O(n_nodes *
degree) for the graph lookup alone). If the model's learned attention decays
with distance faster than these required distances grow, that's a direct,
checkable architectural bottleneck distinct from "needs more
layers/width" -- this module makes the comparison concrete instead of
leaving it as a guess.

Two independent things are computed per validation instance and then
aggregated:

1. `solve_with_dd_provenance` reruns the exact DD frontier algorithm
   (mirroring sudoku_trace.solve_with_dd_trace) while additionally recording,
   for each deduced node, which other node's already-known color newly
   established each of the n_colors-1 required distinct neighbor-colors --
   the minimal evidence set the algorithm actually used. `token_positions`
   then maps hints/graph-rows/round-events to their exact token positions
   (mirroring sudoku_tokens.encode_instance's layout), so each deduction's
   evidence can be converted from "which node" to "how many tokens back".

2. `forward_capture_attention` replicates SATReasoningTransformer.forward
   layer-by-layer by hand (bypassing nn.TransformerEncoderLayer's fused
   eval-mode fast path, which does not expose attention weights), calling
   each layer's self_attn directly with need_weights=True to recover the
   exact (num_heads, L, L) attention matrix per layer.

`attention_weight_by_distance` aggregates attention weight as a function of
relative distance, restricted to the query positions that matter for this
comparison: the NODE token immediately preceding each round's predicted
COLOR token (the position whose hidden state the model must use to look up
that node's neighbors and their colors).
"""
import numpy as np
import torch


def solve_with_dd_provenance(adj, colors, hints, n_colors):
    """Same algorithm/return shape as sudoku_trace.solve_with_dd_trace, plus:
      provenance: {node: [via_node, ...]} -- one entry per distinct
        neighbor-color newly established, in the order DD's synchronous
        frontier sweep encountered them (length n_colors-1 for every
        non-hint node, since that's exactly the threshold for deduction).
      deduced_at: {node: ("hint", None) | ("round", r)} (r is 0-indexed,
        matching trace["rounds"][r])."""
    N, d = adj.shape
    is_deduced = np.zeros(N, dtype=bool)
    deduced_color = {}
    deduced_at = {}
    for node, c in hints.items():
        if c != int(colors[node]):
            raise ValueError(f"hint at node {node} ({c}) disagrees with planted color {colors[node]}")
        is_deduced[node] = True
        deduced_color[node] = int(c)
        deduced_at[node] = ("hint", None)

    seen = np.zeros(N, dtype=np.int64)
    nseen = np.zeros(N, dtype=np.int64)
    provenance = {node: [] for node in range(N)}
    frontier = list(hints.keys())
    rounds = []
    while frontier:
        nxt = []
        for w in frontier:
            bit = 1 << int(colors[w])
            for u in adj[w]:
                u = int(u)
                if is_deduced[u]:
                    continue
                if not (seen[u] & bit):
                    seen[u] |= bit
                    nseen[u] += 1
                    provenance[u].append(w)
                    if nseen[u] == n_colors - 1:
                        is_deduced[u] = True
                        deduced_color[u] = int(colors[u])
                        nxt.append(u)
        if nxt:
            round_idx = len(rounds)
            rounds.append(sorted((node, deduced_color[node]) for node in nxt))
            for node in nxt:
                deduced_at[node] = ("round", round_idx)
        frontier = nxt

    unresolved_nodes = sorted(int(i) for i in np.flatnonzero(~is_deduced))
    return {
        "rounds": rounds, "hints": dict(hints), "resolved": len(unresolved_nodes) == 0,
        "unresolved_nodes": unresolved_nodes, "provenance": provenance, "deduced_at": deduced_at,
    }


def token_positions(adj, trace, vocab):
    """Mirrors sudoku_tokens.encode_instance's token layout exactly, without
    re-encoding, to map (node, section) -> the 0-indexed position of its
    COLOR token (hints/rounds) or its row-header NODE token (graph)."""
    pos = 0
    hint_color_pos = {}
    pos += 1  # BEGIN_HINTS
    for node, c in sorted(trace["hints"].items()):
        pos += 1  # node token
        hint_color_pos[node] = pos
        pos += 1  # color token

    graph_row_pos = {}
    pos += 1  # BEGIN_GRAPH
    for i in range(vocab.n_nodes):
        graph_row_pos[i] = pos
        pos += 1  # node token
        pos += len(adj[i])  # neighbor tokens

    round_color_pos = {}
    pos += 1  # BEGIN_REASONING
    for round_events in trace["rounds"]:
        pos += 1  # BEGIN_ROUND
        for node, c in round_events:
            pos += 1  # node token
            round_color_pos[node] = pos
            pos += 1  # color token
    return {"hint_color_pos": hint_color_pos, "graph_row_pos": graph_row_pos, "round_color_pos": round_color_pos}


def required_distances(adj, trace, provenance, positions):
    """For every non-hint (deduced) node, returns a dict:
      query_pos: the NODE-token position whose hidden state must predict
        this node's color (one before its COLOR token).
      graph_distance: query_pos - (that node's BEGIN_GRAPH row position).
      evidence_distances: list of query_pos - (position of each neighbor's
        color-establishing event), length n_colors-1.
      max_distance: max(graph_distance, *evidence_distances) -- the single
        farthest-back token this deduction needs, in the worst case a model
        with no cross-layer relay would need to jump directly.
    records: one dict per deduced node, in no particular order."""
    hint_color_pos = positions["hint_color_pos"]
    graph_row_pos = positions["graph_row_pos"]
    round_color_pos = positions["round_color_pos"]
    records = []
    for node, via_list in provenance.items():
        if node in trace["hints"]:
            continue
        if node not in round_color_pos:
            continue  # unresolved (shouldn't happen for removal_path-derived instances)
        query_pos = round_color_pos[node] - 1
        graph_distance = query_pos - graph_row_pos[node]
        evidence_distances = []
        for via_node in via_list:
            via_pos = hint_color_pos.get(via_node, round_color_pos.get(via_node))
            evidence_distances.append(query_pos - via_pos)
        records.append({
            "node": node, "query_pos": query_pos, "graph_distance": graph_distance,
            "evidence_distances": evidence_distances,
            "max_distance": max([graph_distance] + evidence_distances),
        })
    return records


def _sa_block(layer, x, attn_mask):
    attn_out, attn_weights = layer.self_attn(
        x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=False,
    )
    return layer.dropout1(attn_out), attn_weights


def _ff_block(layer, x):
    return layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))))


def forward_capture_attention(model, x):
    """x: (1, L) long token ids. Replicates SATReasoningTransformer.forward
    layer by layer (bypassing nn.TransformerEncoderLayer's fused eval-mode
    fast path, which never exposes attention weights), so each layer's
    self_attn is called directly with need_weights=True.

    Returns (logits, attn_maps): attn_maps is a list of length num_layers,
    each a (num_heads, L, L) tensor (batch dim squeezed, since x has batch
    size 1): attn_maps[layer][head, i, j] = how much query position i
    attends to key position j (0 for j > i, causal)."""
    assert x.shape[0] == 1, "one instance at a time (batch dim squeezed in the returned attn maps)"
    l = x.shape[1]
    h = model.token_embed(x) + model.pos_embed[:l]
    causal_mask = torch.triu(torch.ones(l, l, dtype=torch.bool, device=x.device), diagonal=1)
    attn_maps = []
    with torch.no_grad():
        for layer in model.encoder.layers:
            if model.norm_first:
                sa_out, attn_w = _sa_block(layer, layer.norm1(h), causal_mask)
                h = h + sa_out
                h = h + _ff_block(layer, layer.norm2(h))
            else:
                sa_out, attn_w = _sa_block(layer, h, causal_mask)
                h = layer.norm1(h + sa_out)
                h = layer.norm2(h + _ff_block(layer, h))
            attn_maps.append(attn_w[0])
        if model.encoder.norm is not None:
            h = model.encoder.norm(h)
        logits = model.readout(h)
    return logits, attn_maps


def attention_weight_by_distance(attn_maps, query_positions, max_distance):
    """attn_maps: list of (num_heads, L, L) tensors (one per layer).
    query_positions: iterable of query indices to include (e.g. the
    NODE-token positions immediately before each round's predicted COLOR
    token). Averages over heads and the given query positions.

    Returns (sum_by_layer, count_by_distance): sum_by_layer is (num_layers,
    max_distance+1), count_by_distance is (max_distance+1,) -- divide to get
    the mean attention weight at each relative distance, per layer."""
    num_layers = len(attn_maps)
    sum_by_layer = np.zeros((num_layers, max_distance + 1))
    count_by_distance = np.zeros(max_distance + 1)
    for q in query_positions:
        d_max_here = min(q, max_distance)
        count_by_distance[: d_max_here + 1] += 1
        for layer_idx, attn in enumerate(attn_maps):
            # mean over heads, then index key positions q, q-1, ..., q-d_max_here
            head_mean = attn[:, q, :].mean(dim=0).detach().cpu().numpy()
            for d in range(d_max_here + 1):
                sum_by_layer[layer_idx, d] += head_mean[q - d]
    return sum_by_layer, count_by_distance
