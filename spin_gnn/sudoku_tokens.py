"""Vocabulary + encode/decode for the DD-Sudoku event-based reasoning-trace
encoding (spin_gnn.sudoku_trace), directly modeled on
spin_gnn.ksat2_tokens.Vocab: every token is a single categorical id into a
small per-instance vocabulary, meant for nn.Embedding, applying the same
lesson the k-SAT work established -- an event-based trace (what changed and
why, one token stream per reasoning step) tokenizes and learns far better
than a full-board-snapshot-per-step encoding would (see ksat.py/ksat_render.py
vs ksat2.py/ksat2_tokens.py, and notebooks/tokenization_comparison.ipynb on
`master`).

Sequence grammar (see sudoku_trace.py's module docstring for event
semantics):
  BEGIN_HINTS       (NODE COLOR)*
  BEGIN_GRAPH       (NODE (NODE){degree})*                 -- node then its neighbors, any order/subset
  BEGIN_REASONING   (BEGIN_ROUND (NODE COLOR)+)*
  SOLVED  |  UNRESOLVED (NODE)*
"""
import numpy as np

SPECIAL_BASE_NAMES = ["BEGIN_HINTS", "BEGIN_GRAPH", "BEGIN_REASONING", "BEGIN_ROUND", "SOLVED", "UNRESOLVED"]


class Vocab:
    def __init__(self, n_nodes, n_colors, degree):
        self.n_nodes = n_nodes
        self.n_colors = n_colors
        self.degree = degree
        self._special_names = SPECIAL_BASE_NAMES + [f"COLOR_{c}" for c in range(n_colors)]
        self.n_special = len(self._special_names)
        for i, name in enumerate(self._special_names):
            setattr(self, name, i)
        self._color_tokens = [getattr(self, f"COLOR_{c}") for c in range(n_colors)]
        self._color_token_set = set(self._color_tokens)
        self.node_offset = self.n_special
        self.size = self.n_special + n_nodes

    def node_token(self, i):
        return self.node_offset + i

    def is_node_token(self, t):
        return self.node_offset <= t < self.node_offset + self.n_nodes

    def node_index(self, t):
        return t - self.node_offset

    def color_token(self, c):
        return self._color_tokens[c]

    def is_color_token(self, t):
        return t in self._color_token_set

    def color_index(self, t):
        return self._color_tokens.index(t)

    def name(self, t):
        if t < self.n_special:
            return self._special_names[t]
        if self.is_node_token(t):
            return f"N{self.node_index(t)}"
        return f"<oob:{t}>"


def encode_instance(adj, trace, vocab):
    """adj: (N, degree) int neighbor table. trace: sudoku_trace.solve_with_dd_trace's
    return dict. Returns an (T,) int64 array of token ids."""
    tokens = [vocab.BEGIN_HINTS]
    for node, c in sorted(trace["hints"].items()):
        tokens.append(vocab.node_token(node))
        tokens.append(vocab.color_token(c))

    tokens.append(vocab.BEGIN_GRAPH)
    for i in range(vocab.n_nodes):
        tokens.append(vocab.node_token(i))
        for j in adj[i]:
            tokens.append(vocab.node_token(int(j)))

    tokens.append(vocab.BEGIN_REASONING)
    for round_events in trace["rounds"]:
        tokens.append(vocab.BEGIN_ROUND)
        for node, c in round_events:
            tokens.append(vocab.node_token(node))
            tokens.append(vocab.color_token(c))

    if trace["resolved"]:
        tokens.append(vocab.SOLVED)
    else:
        tokens.append(vocab.UNRESOLVED)
        for node in trace["unresolved_nodes"]:
            tokens.append(vocab.node_token(node))

    return np.array(tokens, dtype=np.int64)


class ParseError(Exception):
    """Raised by the strict helpers below (_expect*) on well-formed training
    data, where any mismatch is a real bug. decode() itself never raises --
    it catches this and reports where parsing stopped, since it's also used
    on freely model-generated sequences that may be malformed by
    construction, not by bug."""


def decode(tokens, vocab):
    """Parse back into a structured dict, tolerant of malformed/truncated
    input, mirroring ksat2_tokens.decode's "informative, not a crash"
    philosophy.

    Returns: {"hints": {node: color}, "graph": {node: [neighbors]},
    "rounds": [[(node, color), ...], ...], "outcome": "SOLVED"|"UNRESOLVED"|None,
    "unresolved_nodes": [...], "well_formed": bool, "error": str or None}.
    """
    tokens = [int(t) for t in tokens]
    i = 0
    n = len(tokens)
    result = {
        "hints": {}, "graph": {}, "rounds": [], "outcome": None,
        "unresolved_nodes": [], "well_formed": False, "error": None,
    }

    def peek():
        return tokens[i] if i < n else None

    def expect(tok, what):
        nonlocal i
        if i >= n or tokens[i] != tok:
            raise ParseError(f"expected {what} at position {i}, got {tokens[i] if i < n else '<eof>'}")
        i += 1

    def read_node():
        nonlocal i
        if i >= n or not vocab.is_node_token(tokens[i]):
            raise ParseError(f"expected a node token at position {i}")
        node = vocab.node_index(tokens[i])
        i += 1
        return node

    def read_color():
        nonlocal i
        if i >= n or not vocab.is_color_token(tokens[i]):
            raise ParseError(f"expected a color token at position {i}")
        c = vocab.color_index(tokens[i])
        i += 1
        return c

    try:
        expect(vocab.BEGIN_HINTS, "BEGIN_HINTS")
        while peek() is not None and vocab.is_node_token(peek()):
            node = read_node()
            result["hints"][node] = read_color()

        expect(vocab.BEGIN_GRAPH, "BEGIN_GRAPH")
        while peek() is not None and vocab.is_node_token(peek()):
            node = read_node()
            result["graph"][node] = [read_node() for _ in range(vocab.degree)]

        expect(vocab.BEGIN_REASONING, "BEGIN_REASONING")
        while peek() == vocab.BEGIN_ROUND:
            i += 1
            round_events = []
            while peek() is not None and vocab.is_node_token(peek()):
                node = read_node()
                round_events.append((node, read_color()))
            if not round_events:
                raise ParseError(f"empty BEGIN_ROUND at position {i}")
            result["rounds"].append(round_events)

        if peek() == vocab.SOLVED:
            i += 1
            result["outcome"] = "SOLVED"
        elif peek() == vocab.UNRESOLVED:
            i += 1
            result["outcome"] = "UNRESOLVED"
            while peek() is not None and vocab.is_node_token(peek()):
                result["unresolved_nodes"].append(read_node())
        else:
            raise ParseError(f"expected SOLVED/UNRESOLVED at position {i}, got {tokens[i] if i < n else '<eof>'}")

        result["well_formed"] = True
    except ParseError as exc:
        result["error"] = str(exc)
    return result


def decoded_colors(decoded):
    """Flatten hints + every round's events into one {node: color} dict --
    the full set of colors the trace actually claims, regardless of section."""
    colors = dict(decoded["hints"])
    for round_events in decoded["rounds"]:
        for node, c in round_events:
            colors[node] = c
    return colors


def check_solution(decoded, true_colors, n_nodes):
    """true_colors: (n_nodes,) int array, the planted coloring. True iff
    the decoded trace is a fully correct solve: every node 0..n_nodes-1 has
    a claimed color (via hints or some round) and every claimed color
    matches the truth, and (if outcome == "UNRESOLVED") the claimed
    unresolved set is non-empty and consistent with that -- mirrors
    ksat2_tokens.check_satisfies's "did it actually reach a valid answer"
    semantics, not just "did it emit the right outcome label"."""
    colors = decoded_colors(decoded)
    if decoded["outcome"] == "SOLVED":
        if set(colors) != set(range(n_nodes)):
            return False
        return all(colors[v] == int(true_colors[v]) for v in range(n_nodes))
    if decoded["outcome"] == "UNRESOLVED":
        unresolved = set(decoded["unresolved_nodes"])
        if not unresolved or set(colors) | unresolved != set(range(n_nodes)):
            return False
        return all(colors[v] == int(true_colors[v]) for v in colors)
    return False


def render(tokens, vocab):
    """One line per event/section, human-readable -- for notebook display
    and quick debugging, analogous to ksat2_tokens.render."""
    decoded = decode(tokens, vocab)
    lines = []
    hints_str = " ".join(f"N{v}=C{c}" for v, c in sorted(decoded["hints"].items()))
    lines.append(f"HINTS: {hints_str or '(none)'}")
    for node, neighbors in sorted(decoded["graph"].items()):
        lines.append(f"N{node}: neighbors=" + ",".join(f"N{u}" for u in neighbors))
    for r, round_events in enumerate(decoded["rounds"], start=1):
        ev = " ".join(f"N{v}=C{c}" for v, c in round_events)
        lines.append(f"ROUND {r}: {ev}")
    if decoded["outcome"] == "SOLVED":
        lines.append("SOLVED")
    elif decoded["outcome"] == "UNRESOLVED":
        lines.append("UNRESOLVED: " + ",".join(f"N{v}" for v in decoded["unresolved_nodes"]))
    if not decoded["well_formed"]:
        lines.append(f"(parse stopped early: {decoded['error']})")
    return "\n".join(lines)
