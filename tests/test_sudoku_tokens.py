import numpy as np
import pytest

from spin_gnn.sudoku_dd import planted_regular_graph
from spin_gnn.sudoku_tokens import (
    Vocab, check_evidence, check_solution, decode, encode_instance, evidence_rounds, render,
)
from spin_gnn.sudoku_trace import sample_hints, solve_with_dd_trace


def _instance(seed=0, N=24, d=5, q=4, rho=0.4, n_hints=None):
    rng = np.random.default_rng(seed)
    adj, colors = planted_regular_graph(N, d, q, rng)
    hints = sample_hints(N, n_hints if n_hints is not None else max(1, round(rho * N)), colors, rng)
    trace = solve_with_dd_trace(adj, colors, hints, q)
    vocab = Vocab(N, q, d)
    return adj, colors, hints, trace, vocab


def test_vocab_ranges_dont_overlap():
    vocab = Vocab(n_nodes=20, n_colors=4, degree=5)
    assert vocab.size == vocab.n_special + 20
    for i in range(20):
        t = vocab.node_token(i)
        assert vocab.is_node_token(t) and not vocab.is_color_token(t)
        assert vocab.node_index(t) == i and t >= vocab.n_special
    for c in range(4):
        t = vocab.color_token(c)
        assert vocab.is_color_token(t) and not vocab.is_node_token(t)
        assert vocab.color_index(t) == c


def test_encode_decode_roundtrip_resolved_and_unresolved():
    found_resolved = found_unresolved = False
    for seed in range(40):
        # alternate dense hints (reliably resolves) with a single hint (too sparse
        # to resolve) -- rho=0.5 is too close to d=5's own rho_c~0.46 to reliably
        # resolve at this small N, so lean further into the easy side.
        kwargs = {"rho": 0.9} if seed % 2 == 0 else {"n_hints": 1}
        adj, colors, hints, trace, vocab = _instance(seed=seed, **kwargs)
        tokens = encode_instance(adj, trace, vocab)
        decoded = decode(tokens, vocab)

        assert decoded["well_formed"], decoded["error"]
        assert decoded["hints"] == hints
        for i in range(vocab.n_nodes):
            assert decoded["graph"][i] == [int(j) for j in adj[i]]

        if trace["resolved"]:
            found_resolved = True
            assert decoded["outcome"] == "SOLVED"
        else:
            found_unresolved = True
            assert decoded["outcome"] == "UNRESOLVED"
            assert set(decoded["unresolved_nodes"]) == set(trace["unresolved_nodes"])
        assert check_solution(decoded, colors, vocab.n_nodes), "decoding the true trace must check out as solved"
    assert found_resolved and found_unresolved, "test seed/rho range should hit both outcomes"


def test_check_solution_rejects_wrong_color():
    # Even dense hints can leave a small irreducible unhinted core unresolved at
    # this tiny N (the same phenomenon the branch's hint-removal sweep studies at
    # large N) -- search a few seeds for one that actually resolves rather than
    # assuming any fixed seed/rho does.
    for seed in range(20):
        adj, colors, hints, trace, vocab = _instance(seed=seed, rho=0.9)
        if trace["resolved"]:
            break
    else:
        pytest.fail("no resolved instance found in seed range")
    tokens = encode_instance(adj, trace, vocab)
    decoded = decode(tokens, vocab)
    assert decoded["outcome"] == "SOLVED"
    assert check_solution(decoded, colors, vocab.n_nodes)

    claimed_node = next(iter(hints))
    corrupted = colors.copy()
    corrupted[claimed_node] = (int(corrupted[claimed_node]) + 1) % vocab.n_colors
    assert not check_solution(decoded, corrupted, vocab.n_nodes)


def test_decode_tolerates_truncated_sequence():
    adj, colors, hints, trace, vocab = _instance(seed=2, rho=0.5)
    tokens = encode_instance(adj, trace, vocab)
    truncated = tokens[: len(tokens) // 2]
    decoded = decode(truncated, vocab)
    assert not decoded["well_formed"]
    assert decoded["error"] is not None


def test_render_runs_without_crashing():
    adj, colors, hints, trace, vocab = _instance(seed=4, rho=0.4)
    tokens = encode_instance(adj, trace, vocab)
    text = render(tokens, vocab)
    assert "HINTS" in text
    assert ("SOLVED" in text) or ("UNRESOLVED" in text)


def _scratchpad_instance(seed=0, N=24, d=5, q=4, rho=0.75):
    """A *resolved* instance -- rho here is well above d=5's own rho_c~0.46
    (random hint sampling at rho=0.4 often leaves nodes undeduced), and seeds
    are retried until DD recovers the whole coloring, since every scratchpad
    assertion below is about the structure of real deductions."""
    for attempt in range(40):
        rng = np.random.default_rng(seed + attempt * 1_000)
        adj, colors = planted_regular_graph(N, d, q, rng)
        hints = sample_hints(N, max(1, round(rho * N)), colors, rng)
        trace = solve_with_dd_trace(adj, colors, hints, q)
        if trace["resolved"] and trace["rounds"]:
            return adj, colors, hints, trace, Vocab(N, q, d, scratchpad=True)
    raise AssertionError("no resolved instance found; fixture is broken")


def test_scratchpad_vocab_keeps_base_token_ids():
    """A scratchpad=False vocab must stay byte-identical to pre-flag
    checkpoints; the scratchpad one only adds two specials."""
    base = Vocab(n_nodes=20, n_colors=4, degree=5)
    sp = Vocab(n_nodes=20, n_colors=4, degree=5, scratchpad=True)
    for name in ["BEGIN_HINTS", "BEGIN_GRAPH", "BEGIN_REASONING", "BEGIN_ROUND", "SOLVED", "UNRESOLVED"]:
        assert getattr(base, name) == getattr(sp, name)
    assert sp.size == base.size + 2
    assert not base.scratchpad and sp.scratchpad
    assert sp.is_evidence_color_token(sp.COLOR_UNKNOWN)
    assert not sp.is_color_token(sp.COLOR_UNKNOWN)
    assert sp.evidence_color_index(sp.COLOR_UNKNOWN) is None
    assert sp.evidence_color_token(None) == sp.COLOR_UNKNOWN
    assert sp.evidence_color_token(2) == sp.color_token(2)
    with pytest.raises(ValueError):
        base.evidence_color_token(None)


def test_scratchpad_roundtrip_and_evidence_is_consistent():
    adj, colors, hints, trace, vocab = _scratchpad_instance(seed=1)
    assert trace["resolved"], "need a resolved instance for this test"
    tokens = encode_instance(adj, trace, vocab)
    decoded = decode(tokens, vocab)
    assert decoded["well_formed"], decoded["error"]
    assert decoded["hints"] == {int(k): int(v) for k, v in trace["hints"].items()}
    assert decoded["rounds"] == [[(int(n), int(c)) for n, c in r] for r in trace["rounds"]]
    assert check_solution(decoded, colors, vocab.n_nodes)
    # one evidence row per deduction, each `degree` slots wide
    assert [len(r) for r in decoded["evidence"]] == [len(r) for r in trace["rounds"]]
    for round_rows in decoded["evidence"]:
        for node, slots in round_rows:
            assert len(slots) == vocab.degree
    ev = check_evidence(decoded)
    assert ev["n_rows"] > 0
    assert ev["n_rows_correct"] == ev["n_rows"], "ground-truth encoding must be self-consistent"
    assert ev["n_slots_correct"] == ev["n_slots"]


def test_scratchpad_evidence_leaves_exactly_one_color_free():
    """DD forces a node once its neighbors cover n_colors-1 distinct colors, so
    every restated evidence row must show exactly that many distinct known
    colors, and the deduced color must be the missing one -- i.e. the
    scratchpad really does make each step locally determined."""
    adj, colors, hints, trace, vocab = _scratchpad_instance(seed=2)
    assert trace["resolved"]
    for round_rows, round_events in zip(evidence_rounds(adj, trace), trace["rounds"]):
        for (node, slots, c), (ev_node, ev_c) in zip(round_rows, round_events):
            assert node == ev_node and c == ev_c
            known = {s for s in slots if s is not None}
            assert len(known) == vocab.n_colors - 1
            assert c not in known
            assert known == set(range(vocab.n_colors)) - {c}


def test_scratchpad_evidence_excludes_same_round_siblings():
    """Evidence must reflect the state *before* the round: a neighbor deduced
    in the same round contributed nothing to this deduction, so its slot has
    to read COLOR_UNKNOWN, not its (already known to us) color."""
    adj, colors, hints, trace, vocab = _scratchpad_instance(seed=3)
    assert trace["resolved"]
    rounds = evidence_rounds(adj, trace)
    checked = 0
    for r, round_rows in enumerate(rounds):
        same_round = {int(n) for n, _ in trace["rounds"][r]}
        for node, slots, _ in round_rows:
            for u, slot in zip(adj[node], slots):
                if int(u) in same_round:
                    assert slot is None
                    checked += 1
    assert checked > 0, "no same-round sibling adjacency found; test is vacuous"


def test_check_evidence_flags_a_corrupted_row():
    adj, colors, hints, trace, vocab = _scratchpad_instance(seed=1)
    tokens = encode_instance(adj, trace, vocab)
    decoded = decode(tokens, vocab)
    node, slots = decoded["evidence"][0][0]
    wrong = list(slots)
    wrong[0] = (wrong[0] + 1) % vocab.n_colors if wrong[0] is not None else 0
    decoded["evidence"][0][0] = (node, wrong)
    ev = check_evidence(decoded)
    assert ev["n_rows_correct"] == ev["n_rows"] - 1
    assert ev["n_slots_correct"] == ev["n_slots"] - 1


def test_scratchpad_sequences_are_longer_than_plain():
    adj, colors, hints, trace, _ = _scratchpad_instance(seed=1)
    plain = encode_instance(adj, trace, Vocab(24, 4, 5))
    sp = encode_instance(adj, trace, Vocab(24, 4, 5, scratchpad=True))
    n_deductions = sum(len(r) for r in trace["rounds"])
    # each deduction gains BEGIN_EVIDENCE + node + degree evidence slots
    assert len(sp) == len(plain) + n_deductions * (2 + 5)


def test_scratchpad_decode_rejects_missing_evidence_header():
    """A scratchpad model that emits a bare deduction pair (no BEGIN_EVIDENCE)
    is malformed, and decode must say so rather than silently accepting it."""
    adj, colors, hints, trace, vocab = _scratchpad_instance(seed=1)
    tokens = list(encode_instance(adj, trace, vocab))
    first = tokens.index(vocab.BEGIN_EVIDENCE)
    del tokens[first]
    decoded = decode(np.array(tokens, dtype=np.int64), vocab)
    assert not decoded["well_formed"]
    assert decoded["error"]


def test_scratchpad_render_shows_evidence():
    adj, colors, hints, trace, vocab = _scratchpad_instance(seed=1)
    text = render(encode_instance(adj, trace, vocab), vocab)
    assert "evidence N" in text
