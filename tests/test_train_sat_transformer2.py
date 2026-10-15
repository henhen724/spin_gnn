"""End-to-end smoke test + checkpoint round-trip for train_sat_transformer2,
mirroring test_train_sat_transformer.py's structure. Tiny scale (few
instances, few epochs) -- this checks the pipeline runs and every metric
key exists/is in a sane range, not that the model actually learns anything
at this size."""
import numpy as np
import torch

from spin_gnn.config import SATReasoningConfig, SATReasoningModelConfig, SATReasoningTrainConfig
from spin_gnn.train_sat_transformer2 import (
    build_instances, build_vocab, load_checkpoint, save_checkpoint, train_sat2_model,
)


def _tiny_configs():
    sat_cfg = SATReasoningConfig(n_vars=6, k=3, alpha=1.5, n_hints=0, seed=0)
    model_cfg = SATReasoningModelConfig(d_model=16, num_layers=1, nhead=2, dim_feedforward=32)
    train_cfg = SATReasoningTrainConfig(n_train_instances=20, n_val_instances=10, batch_size=4, epochs=2)
    return sat_cfg, model_cfg, train_cfg


def test_same_seed_gives_identical_training_run():
    """Regression test for the noise fix: train_sat2_model now calls
    torch.manual_seed(sat_cfg.seed) before model construction, so two runs
    with the same seed must produce bit-identical weights and history on
    CPU (no GPU nondeterminism here) -- this is exactly the ~3-25x
    run-to-run solve_rate noise seen in every sweep so far, now eliminated
    for same-seed reruns."""
    sat_cfg, model_cfg, train_cfg = _tiny_configs()
    result1 = train_sat2_model(sat_cfg, model_cfg, train_cfg)
    result2 = train_sat2_model(sat_cfg, model_cfg, train_cfg)

    for (name1, p1), (name2, p2) in zip(result1["model"].named_parameters(), result2["model"].named_parameters()):
        assert name1 == name2
        torch.testing.assert_close(p1, p2, msg=f"parameter {name1} differs between same-seed runs")
    assert result1["history"] == result2["history"]


def test_different_seed_gives_different_init():
    sat_cfg1, model_cfg, train_cfg = _tiny_configs()
    sat_cfg2 = SATReasoningConfig(**{**sat_cfg1.__dict__, "seed": 1})
    result1 = train_sat2_model(sat_cfg1, model_cfg, train_cfg)
    result2 = train_sat2_model(sat_cfg2, model_cfg, train_cfg)

    any_different = any(
        not torch.equal(p1, p2)
        for p1, p2 in zip(result1["model"].parameters(), result2["model"].parameters())
    )
    assert any_different, "different seeds should not coincidentally produce identical weights"


def test_build_instances_and_vocab_consistent():
    sat_cfg = SATReasoningConfig(n_vars=8, k=3, alpha=2.0)
    vocab = build_vocab(sat_cfg)
    assert vocab.m_clauses == round(2.0 * 8)
    instances = build_instances(sat_cfg, n_instances=15, seed_start=0)
    assert len(instances) == 15
    for clauses, n_vars, hints, trace in instances:
        assert clauses.shape == (vocab.m_clauses, 8)
        assert n_vars == 8
        assert hints == {}


def test_train_sat2_model_runs_and_reports_sane_metrics():
    sat_cfg, model_cfg, train_cfg = _tiny_configs()
    result = train_sat2_model(sat_cfg, model_cfg, train_cfg)

    assert len(result["history"]) == train_cfg.epochs
    for record in result["history"]:
        assert 0.0 <= record["accuracy"] <= 1.0
        assert record["nll"] >= 0.0
    assert 0.0 <= result["baseline_accuracy"] <= 1.0
    sm = result["solve_metrics"]
    assert 0.0 <= sm["solve_rate"] <= 1.0
    assert 0.0 <= sm["outcome_accuracy"] <= 1.0
    assert 0.0 <= sm["well_formed_rate"] <= 1.0
    assert sm["solve_rate"] <= sm["outcome_accuracy"] + 1e-9  # solving requires the outcome to be right first


def test_checkpoint_roundtrip(tmp_path):
    sat_cfg, model_cfg, train_cfg = _tiny_configs()
    result = train_sat2_model(sat_cfg, model_cfg, train_cfg)
    out_dir = str(tmp_path / "ckpt")
    save_checkpoint(
        out_dir, sat_cfg, model_cfg, result["model"], result["vocab"], result["model_max_len"],
        result["baseline_accuracy"], result["solve_metrics"], result["history"],
    )
    model2, vocab2, config2 = load_checkpoint(out_dir, device="cpu")
    assert vocab2.size == result["vocab"].size
    assert config2["nhead"] == model_cfg.nhead
    assert config2["num_layers"] == model_cfg.num_layers

    x = torch.randint(0, vocab2.size, (2, 5))
    with torch.no_grad():
        out1 = result["model"](x, valid_mask=torch.ones(2, 5, dtype=torch.bool))
        out2 = model2(x, valid_mask=torch.ones(2, 5, dtype=torch.bool))
    torch.testing.assert_close(out1, out2)


def test_checkpoint_roundtrip_norm_first(tmp_path):
    sat_cfg, model_cfg, train_cfg = _tiny_configs()
    model_cfg.norm_first = True
    result = train_sat2_model(sat_cfg, model_cfg, train_cfg)
    assert result["model"].encoder.norm is not None

    out_dir = str(tmp_path / "ckpt_pre")
    save_checkpoint(
        out_dir, sat_cfg, model_cfg, result["model"], result["vocab"], result["model_max_len"],
        result["baseline_accuracy"], result["solve_metrics"], result["history"],
    )
    model2, vocab2, config2 = load_checkpoint(out_dir, device="cpu")
    assert config2["norm_first"] is True
    assert model2.encoder.norm is not None

    x = torch.randint(0, vocab2.size, (2, 5))
    with torch.no_grad():
        out1 = result["model"](x, valid_mask=torch.ones(2, 5, dtype=torch.bool))
        out2 = model2(x, valid_mask=torch.ones(2, 5, dtype=torch.bool))
    torch.testing.assert_close(out1, out2)


def test_old_style_checkpoint_without_norm_first_key_defaults_to_post_norm(tmp_path):
    """load_checkpoint must still work on a config.json written before
    norm_first existed (every checkpoint from the original 72+14-config
    sweeps)."""
    import json

    sat_cfg, model_cfg, train_cfg = _tiny_configs()
    result = train_sat2_model(sat_cfg, model_cfg, train_cfg)
    out_dir = str(tmp_path / "ckpt_old")
    save_checkpoint(
        out_dir, sat_cfg, model_cfg, result["model"], result["vocab"], result["model_max_len"],
        result["baseline_accuracy"], result["solve_metrics"], result["history"],
    )
    config_path = f"{out_dir}/config.json"
    with open(config_path) as f:
        config = json.load(f)
    del config["norm_first"]
    with open(config_path, "w") as f:
        json.dump(config, f)

    model2, vocab2, config2 = load_checkpoint(out_dir, device="cpu")
    assert model2.encoder.norm is None
