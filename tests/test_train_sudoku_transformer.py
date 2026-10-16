"""End-to-end smoke test + checkpoint round-trip for train_sudoku_transformer,
mirroring test_train_sat_transformer2.py's structure. Tiny scale -- checks
the pipeline runs and every metric key exists/is in a sane range, not that
the model actually learns anything at this size."""
import torch

from spin_gnn.config import SudokuReasoningConfig, SudokuReasoningModelConfig, SudokuReasoningTrainConfig
from spin_gnn.train_sudoku_transformer import (
    build_instances, build_vocab, load_checkpoint, save_checkpoint, train_sudoku_model,
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
