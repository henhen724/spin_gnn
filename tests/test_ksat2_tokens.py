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


def test_default_vocab_size_unchanged_backward_compat():
    """Every existing checkpoint's embedding/readout shapes were built from
    this exact formula (12 special tokens) -- must never silently change for
    the default (use_clause_status=False) call every prior caller uses."""
    vocab = Vocab(n_vars=16, m_clauses=20)
    assert vocab.use_clause_status is False
    assert vocab.n_special == 12
    assert vocab.size == 12 + 16 + 20 == 48


def test_use_clause_status_adds_exactly_two_special_tokens():
    plain = Vocab(n_vars=5, m_clauses=7)
    status = Vocab(n_vars=5, m_clauses=7, use_clause_status=True)
    assert status.n_special == plain.n_special + 2
    assert status.size == plain.size + 2
    assert status.SATISFIED == plain.n_special
    assert status.REACTIVATED == plain.n_special + 1
    # var/clause offsets shift to make room, but ranges still don't overlap
    assert status.var_token(0) == status.n_special
    assert status.clause_token(0) == status.n_special + 5


def test_encode_decode_roundtrip_with_clause_status_matches_ground_truth():
    found_satisfied = found_reactivated = False
    for seed in range(60):
        cfg_n_vars = 8
        clauses, n_vars, hints, trace, _ = _instance(seed=seed, n_vars=cfg_n_vars, alpha=5.0)
        vocab = Vocab(n_vars, clauses.shape[0], use_clause_status=True)
        tokens = encode_instance(clauses, n_vars, hints, trace, vocab)
        decoded = decode(tokens, vocab)

        assert decoded["well_formed"], decoded["error"]
        assert len(decoded["events"]) == len(trace["events"])
        for decoded_event, true_event in zip(decoded["events"], trace["events"]):
            assert decoded_event["satisfied"] == true_event["satisfied"]
            assert decoded_event["reactivated"] == true_event["reactivated"]
            if true_event["satisfied"]:
                found_satisfied = True
            if true_event["reactivated"]:
                found_reactivated = True
    assert found_satisfied, "test range should hit at least one SATISFIED tag"
    assert found_reactivated, "test range should hit at least one REACTIVATED tag (needs a backtrack)"


def test_decode_without_clause_status_vocab_ignores_satisfied_concept():
    """A plain (use_clause_status=False) vocab never looks for SATISFIED/
    REACTIVATED tokens at all -- encode_instance with it must not emit them,
    and decode must not crash trying to recognize them (they don't exist in
    this vocab's token space)."""
    clauses, n_vars, hints, trace, vocab = _instance(seed=5, n_vars=8, alpha=5.0)
    assert vocab.use_clause_status is False
    tokens = encode_instance(clauses, n_vars, hints, trace, vocab)
    decoded = decode(tokens, vocab)
    assert decoded["well_formed"]
    for event in decoded["events"]:
        assert event["satisfied"] == []
        assert event["reactivated"] == []


def test_render_shows_satisfied_and_reactivated_tags():
    for seed in range(30):
        clauses, n_vars, hints, trace, _ = _instance(seed=seed, n_vars=8, alpha=5.0)
        if not any(e["satisfied"] for e in trace["events"]):
            continue
        vocab = Vocab(n_vars, clauses.shape[0], use_clause_status=True)
        tokens = encode_instance(clauses, n_vars, hints, trace, vocab)
        text = render(tokens, vocab)
        assert "SATISFIED:" in text
        return
    pytest.fail("no seed in range produced a SATISFIED event to render")
