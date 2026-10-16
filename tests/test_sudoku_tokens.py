import numpy as np
import pytest

from spin_gnn.sudoku_dd import planted_regular_graph
from spin_gnn.sudoku_tokens import Vocab, check_solution, decode, encode_instance, render
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
