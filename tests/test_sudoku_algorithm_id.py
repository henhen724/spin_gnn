"""The baselines decide what the next_node probe result means, so they are
checked against hand-computed state rather than against each other.
"""
import numpy as np
import torch

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_algorithm_id import _known_before, baseline_predictions, score_instance
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab

CFG = SudokuReasoningConfig(n_nodes=24, degree=5, n_colors=4, rho=0.6, rho_max=0.6, seed=0)


def _instance(i=0):
    return build_instances(CFG, n_instances=i + 1, seed_start=777)[i]


def test_known_before_excludes_the_deduction_being_predicted():
    """The state must be what DD had BEFORE emitting this node -- including
    the node itself would hand the answer to every baseline."""
    adj, _colors, hints, trace = _instance()
    for r, rd in enumerate(trace["rounds"]):
        for k in range(len(rd)):
            known = _known_before(trace, r, k)
            assert int(rd[k][0]) not in known
            # earlier siblings in the same round ARE known
            for j in range(k):
                assert int(rd[j][0]) in known
            # later siblings are not
            for j in range(k + 1, len(rd)):
                assert int(rd[j][0]) not in known
            assert set(hints) <= set(known)


def test_baselines_only_guess_uncolored_nodes():
    adj, _colors, _hints, trace = _instance()
    rng = np.random.default_rng(0)
    for r, rd in enumerate(trace["rounds"]):
        for k in range(len(rd)):
            known = _known_before(trace, r, k)
            for name, guess in baseline_predictions(adj, trace, CFG.n_nodes, r, k, rng).items():
                assert guess not in known, (name, guess)
                assert 0 <= guess < CFG.n_nodes


def test_ascending_after_exploits_within_round_sorting():
    """Rounds are emitted sorted, so after emitting node v the next one is the
    smallest uncolored id above v whenever the round has more members. This is
    exactly the free accuracy a non-DD rule gets, and the reason the next_node
    probe needs a baseline."""
    adj, _colors, _hints, trace = _instance()
    rng = np.random.default_rng(0)
    hits = total = 0
    for r, rd in enumerate(trace["rounds"]):
        for k in range(1, len(rd)):
            base = baseline_predictions(adj, trace, CFG.n_nodes, r, k, rng)
            if not base:
                continue
            total += 1
            hits += base["ascending_after"] == int(rd[k][0])
    assert total > 0
    # Not asserting a specific rate, only that the heuristic is a real
    # contender -- if it were near zero there would be nothing to control for.
    assert hits / total > 0.3, f"ascending_after only {hits}/{total}"


def test_score_instance_reports_only_disagreements():
    """Rows where a baseline agrees with DD carry no information and must not
    dilute the vote."""
    vocab = build_vocab(CFG)
    from spin_gnn.config import SudokuReasoningModelConfig
    from spin_gnn.sat_transformer2 import SATReasoningTransformer
    torch.manual_seed(0)
    mc = SudokuReasoningModelConfig(d_model=16, num_layers=1, nhead=2, dim_feedforward=32)
    model = SATReasoningTransformer(
        vocab_size=vocab.size, d_model=mc.d_model, nhead=mc.nhead, num_layers=mc.num_layers,
        dim_feedforward=mc.dim_feedforward, max_len=4096, norm_first=True,
    ).eval()

    adj, _colors, _hints, trace = _instance()
    rows = score_instance(model, adj, trace, vocab, "cpu", np.random.default_rng(0))
    assert rows
    for r in rows:
        assert r["dd_answer"] != r["baseline_answer"]
        assert not (r["model_follows_dd"] and r["model_follows_baseline"])
    assert {r["baseline"] for r in rows} <= {
        "ascending", "ascending_after", "random_uncolored", "most_known_nbrs"}
