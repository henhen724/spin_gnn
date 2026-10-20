"""End-to-end smoke test + checkpoint round-trip for train_sudoku_transformer,
mirroring test_train_sat_transformer2.py's structure. Tiny scale -- checks
the pipeline runs and every metric key exists/is in a sane range, not that
the model actually learns anything at this size."""
from dataclasses import replace

import pytest
import torch

from spin_gnn.config import SudokuReasoningConfig, SudokuReasoningModelConfig, SudokuReasoningTrainConfig
from spin_gnn.train_sudoku_transformer import (
    build_instances, build_vocab, eval_solve_by_rho, load_checkpoint, save_checkpoint,
    solve_rate, train_sudoku_model,
)


def _tiny_configs():
    # n_nodes=24 (not smaller): planted_regular_graph's multi-edge removal
    # needs enough stubs per class pair to find a simple pairing -- N=12 at
    # this degree/n_colors hits "could not remove multi-edges" too often.
    sudoku_cfg = SudokuReasoningConfig(n_nodes=24, degree=4, n_colors=3, rho=0.5, rho_max=0.5, seed=0)
    model_cfg = SudokuReasoningModelConfig(d_model=16, num_layers=1, nhead=2, dim_feedforward=32)
    train_cfg = SudokuReasoningTrainConfig(n_train_instances=20, n_val_instances=10, batch_size=4, epochs=2)
    return sudoku_cfg, model_cfg, train_cfg


def test_same_seed_gives_identical_training_run():
    sudoku_cfg, model_cfg, train_cfg = _tiny_configs()
    result1 = train_sudoku_model(sudoku_cfg, model_cfg, train_cfg)
    result2 = train_sudoku_model(sudoku_cfg, model_cfg, train_cfg)

    for (n1, p1), (n2, p2) in zip(result1["model"].named_parameters(), result2["model"].named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p1, p2, msg=f"parameter {n1} differs between same-seed runs")
    assert result1["history"] == result2["history"]


def test_build_instances_and_vocab_consistent():
    sudoku_cfg = SudokuReasoningConfig(n_nodes=24, degree=4, n_colors=3, rho=0.5, rho_max=0.5)
    vocab = build_vocab(sudoku_cfg)
    assert vocab.n_nodes == 24
    instances = build_instances(sudoku_cfg, n_instances=15, seed_start=0)
    assert len(instances) == 15
    for adj, colors, hints, trace in instances:
        assert adj.shape == (24, 4)
        assert len(hints) == round(0.5 * 24)


def test_train_sudoku_model_runs_and_reports_sane_metrics():
    sudoku_cfg, model_cfg, train_cfg = _tiny_configs()
    result = train_sudoku_model(sudoku_cfg, model_cfg, train_cfg)

    assert len(result["history"]) == train_cfg.epochs
    for record in result["history"]:
        assert 0.0 <= record["accuracy"] <= 1.0
        assert record["nll"] >= 0.0
    assert 0.0 <= result["baseline_accuracy"] <= 1.0
    sm = result["solve_metrics"]
    assert 0.0 <= sm["solve_rate"] <= 1.0
    assert 0.0 <= sm["outcome_accuracy"] <= 1.0
    assert 0.0 <= sm["well_formed_rate"] <= 1.0
    assert sm["solve_rate"] <= sm["outcome_accuracy"] + 1e-9


def test_checkpoint_roundtrip(tmp_path):
    sudoku_cfg, model_cfg, train_cfg = _tiny_configs()
    result = train_sudoku_model(sudoku_cfg, model_cfg, train_cfg)
    out_dir = tmp_path / "ckpt"
    save_checkpoint(
        str(out_dir), sudoku_cfg, model_cfg, result["model"], result["vocab"], result["model_max_len"],
        result["baseline_accuracy"], result["solve_metrics"], result["history"],
    )
    model2, vocab2, config2 = load_checkpoint(str(out_dir))
    assert vocab2.n_nodes == sudoku_cfg.n_nodes
    for p1, p2 in zip(result["model"].parameters(), model2.parameters()):
        torch.testing.assert_close(p1.cpu(), p2.cpu())


def test_random_hint_mode_produces_both_outcomes_and_exact_density():
    """The whole point of hint_mode="random": unlike the removal-path default,
    it must produce genuinely unresolvable instances, which is what makes a
    hint-density solvability transition exist in the data. Swept across rho
    because any single density can land entirely on one side."""
    outcomes = set()
    for rho in (0.2, 0.5, 0.9):
        cfg = SudokuReasoningConfig(
            n_nodes=24, degree=4, n_colors=3, rho=rho, rho_max=rho, hint_mode="random",
        )
        instances = build_instances(cfg, n_instances=12, seed_start=0)
        assert len(instances) == 12
        for adj, colors, hints, trace in instances:
            # exactly round(rho * n_nodes) hints, unlike removal mode where the
            # path's own minimal density can overshoot the target
            assert len(hints) == round(rho * 24)
            for node, color in hints.items():
                assert colors[node] == color
            outcomes.add(bool(trace["resolved"]))
    assert outcomes == {True, False}, f"random hints must yield both outcomes, got {outcomes}"


def test_removal_mode_is_unchanged_and_always_resolvable():
    """Guards the default path against the hint_mode refactor."""
    cfg = SudokuReasoningConfig(n_nodes=24, degree=4, n_colors=3, rho=0.5, rho_max=0.5)
    assert cfg.hint_mode == "removal"
    for adj, colors, hints, trace in build_instances(cfg, n_instances=10, seed_start=0):
        assert trace["resolved"]
        assert not trace["unresolved_nodes"]


def test_unknown_hint_mode_raises():
    cfg = SudokuReasoningConfig(n_nodes=24, degree=4, n_colors=3, rho=0.5, rho_max=0.5, hint_mode="bogus")
    try:
        build_instances(cfg, n_instances=1, seed_start=0)
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("unknown hint_mode must raise")


def test_random_hint_mode_trains_end_to_end():
    """Unresolved instances exercise the UNRESOLVED branch of the grammar and
    of solve_rate's outcome check, which the removal-path default never hits."""
    sudoku_cfg, model_cfg, train_cfg = _tiny_configs()
    sudoku_cfg = replace(sudoku_cfg, rho=0.3, rho_max=0.9, hint_mode="random")
    result = train_sudoku_model(sudoku_cfg, model_cfg, train_cfg)
    sm = result["solve_metrics"]
    assert 0.0 <= sm["solve_rate"] <= 1.0
    assert 0.0 <= sm["outcome_accuracy"] <= 1.0


def test_truncated_rate_catches_a_starved_generation_budget():
    """A tiny max_new_tokens must register as truncation, not as a reasoning
    failure. This is the confound that would otherwise fake a transition:
    deeper traces are longer, so a budget that only binds at low hint density
    produces a solve_rate collapse with no reasoning content at all."""
    sudoku_cfg, model_cfg, train_cfg = _tiny_configs()
    result = train_sudoku_model(sudoku_cfg, model_cfg, train_cfg)
    model, vocab = result["model"], result["vocab"]
    instances = result["val_instances"]

    starved = solve_rate(model, instances, vocab, "cpu", max_new_tokens=3)
    assert starved["truncated_rate"] == 1.0, starved
    assert starved["solve_rate"] == 0.0

    # The other half of the metric needs a model that actually terminates. The
    # toy model above never emits SOLVED at any budget, so it cannot
    # distinguish "ran out of room" from "cannot reason" -- a stub that always
    # emits SOLVED can, and that is exactly the case that must NOT be counted.
    class _AlwaysSolved:
        pos_embed = torch.zeros(result["model_max_len"], 1)

        def eval(self):
            return self

        def __call__(self, seq):
            logits = torch.full((1, seq.shape[1], vocab.size), -10.0)
            logits[..., vocab.SOLVED] = 10.0
            return logits

    stub = solve_rate(_AlwaysSolved(), instances, vocab, "cpu", max_new_tokens=result["model_max_len"])
    assert stub["truncated_rate"] == 0.0, stub


def test_eval_by_rho_reports_realized_density_and_deduction_time():
    """Below the greedy path's own floor a requested rho cannot be delivered,
    so realized_rho must reveal the floor instead of echoing the request --
    otherwise sub-floor bins look like distinct densities when they are one
    point measured repeatedly. dd_mean_T must also rise as rho falls, since
    that divergence is the transition being looked for."""
    sudoku_cfg, model_cfg, train_cfg = _tiny_configs()
    result = train_sudoku_model(sudoku_cfg, model_cfg, train_cfg)
    records = eval_solve_by_rho(
        result["model"], result["vocab"], "cpu", sudoku_cfg,
        rho_grid=[0.95, 0.7, 0.1], n_per_bin=8, seed_start=12345,
        max_new_tokens=result["model_max_len"],
    )
    by_rho = {r["rho"]: r for r in records}
    assert by_rho[0.95]["realized_rho"] == pytest.approx(0.95, abs=0.06)
    # The 0.1 request is unreachable, so it must come back at the path's floor,
    # well above what was asked for. The floor itself is config-dependent (it
    # sits near 0.20 at this d=4/q=3 toy setting but near 0.46 at the d=5/q=4
    # setting the real models use -- 3 colors means DD forces a node from only
    # 2 distinct neighbour colors, so it tolerates much sparser hints), hence
    # the assertion is "above the request", not a fixed density.
    assert by_rho[0.1]["realized_rho"] > 0.15, by_rho[0.1]["realized_rho"]
    assert by_rho[0.1]["dd_mean_T"] > by_rho[0.95]["dd_mean_T"], records
    # Removal-path instances are resolvable by construction at every density,
    # so a solve_rate drop can never be the instance becoming unsolvable.
    assert all(r["dd_resolved_frac"] == 1.0 for r in records)
