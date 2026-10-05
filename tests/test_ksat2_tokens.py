import numpy as np
import pytest

from spin_gnn.config import SATConfig
from spin_gnn.ksat import sample_random_ksat
from spin_gnn.ksat2 import sample_hints, solve_with_reasoning_trace
from spin_gnn.ksat2_tokens import Vocab, check_satisfies, decode, encode_instance, render


def _instance(seed=0, n_vars=10, k=3, alpha=4.2, n_hints=0):
    cfg = SATConfig(n_vars=n_vars, k=k, alpha=alpha)
    rng = np.random.default_rng(seed)
    clauses = sample_random_ksat(cfg, rng)
    hints = sample_hints(n_vars, n_hints, np.random.default_rng(seed + 10_000)) if n_hints else {}
    trace = solve_with_reasoning_trace(clauses, n_vars, hints=hints)
    vocab = Vocab(n_vars, clauses.shape[0])
    return clauses, n_vars, hints, trace, vocab


def test_vocab_ranges_dont_overlap():
    vocab = Vocab(n_vars=5, m_clauses=7)
    assert vocab.size == vocab.n_special + 5 + 7
    for v in range(5):
        t = vocab.var_token(v)
        assert vocab.is_var_token(t) and not vocab.is_clause_token(t) and t >= vocab.n_special
        assert vocab.var_index(t) == v
    for c in range(7):
        t = vocab.clause_token(c)
        assert vocab.is_clause_token(t) and not vocab.is_var_token(t)
        assert vocab.clause_index(t) == c


def test_encode_decode_roundtrip_sat_and_unsat():
    found_sat = found_unsat = False
    for seed in range(60):
        clauses, n_vars, hints, trace, vocab = _instance(seed=seed, n_vars=8, alpha=5.0)
        tokens = encode_instance(clauses, n_vars, hints, trace, vocab)
        decoded = decode(tokens, vocab)

        assert decoded["well_formed"], decoded["error"]
        assert decoded["hints"] == hints
        for c in range(clauses.shape[0]):
            expected = {v: int(clauses[c, v]) for v in range(n_vars) if clauses[c, v] != 0}
            assert decoded["clauses"][c] == expected

        if trace["satisfiable"]:
            found_sat = True
            assert decoded["outcome"] == "SAT"
            expected_assignment = {v: int(trace["assignment"][v]) for v in range(n_vars) if trace["assignment"][v] != 0}
            assert decoded["assignment"] == expected_assignment
            assert check_satisfies(decoded["assignment"], clauses, n_vars)
        else:
            found_unsat = True
            assert decoded["outcome"] == "UNSAT"
    assert found_sat and found_unsat, "test instance range should hit both outcomes at alpha=5.0"


def test_encode_decode_roundtrip_with_hints():
    clauses, n_vars, hints, trace, vocab = _instance(seed=3, n_hints=2)
    tokens = encode_instance(clauses, n_vars, hints, trace, vocab)
    decoded = decode(tokens, vocab)
    assert decoded["well_formed"]
    assert decoded["hints"] == hints


def test_decode_truncated_sequence_reports_error_not_crash():
    clauses, n_vars, hints, trace, vocab = _instance(seed=1)
    tokens = encode_instance(clauses, n_vars, hints, trace, vocab)
    truncated = tokens[: len(tokens) // 2]
    decoded = decode(truncated, vocab)
    assert decoded["well_formed"] is False
    assert decoded["error"] is not None


def test_decode_garbage_sequence_does_not_crash():
    vocab = Vocab(n_vars=6, m_clauses=10)
    rng = np.random.default_rng(0)
    garbage = rng.integers(0, vocab.size, size=50)
    decoded = decode(garbage, vocab)
    assert isinstance(decoded["well_formed"], bool)


def test_check_satisfies_on_toy_clauses():
    n_vars = 3
    clauses = np.array([[1, -1, 0], [0, 0, 1]], dtype=np.int8)  # (x1 | ~x2) & (x3)
    assert check_satisfies({0: 1, 1: -1, 2: 1}, clauses, n_vars)
    assert not check_satisfies({0: -1, 1: 1, 2: 1}, clauses, n_vars)  # violates clause 1
    assert not check_satisfies({0: 1, 1: -1, 2: -1}, clauses, n_vars)  # violates clause 2


def test_render_smoke():
    clauses, n_vars, hints, trace, vocab = _instance(seed=2)
    tokens = encode_instance(clauses, n_vars, hints, trace, vocab)
    text = render(tokens, vocab)
    assert "HINTS" in text
    assert "CLAUSE1" in text
    assert ("SAT" in text) or ("UNSAT" in text)
