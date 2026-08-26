"""Regression test for train_sat_transformer.py's length-capping paths.

satisfying_completion_rate operates on the *raw* trace list (not through
SATTraceDataset), so nothing else caps sequence length before it's fed to
the model -- a trace longer than the model's positional-embedding capacity
must be filtered out here specifically, or it crashes. This exact bug
(train_cfg.max_trace_tokens capped the training dataset but not this path)
took down every task of a 12-way hyperparameter sweep in practice."""
import numpy as np
import torch

from spin_gnn.config import SATConfig, SATModelConfig
from spin_gnn.sat_transformer import SATTransformer
from spin_gnn.train_sat_transformer import load_checkpoint, save_checkpoint, satisfying_completion_rate


def test_satisfying_completion_rate_skips_traces_longer_than_model_max_len():
    n_vars = 4
    model = SATTransformer(n_vars, d_model=8, num_layers=1, nhead=1, max_len=5)

    short_tokens = np.zeros((4, n_vars + 1), dtype=np.float32)  # input len 3 <= max_len=5, kept
    long_tokens = np.zeros((10, n_vars + 1), dtype=np.float32)  # input len 9 > max_len=5, must be skipped
    clauses = np.zeros((1, n_vars), dtype=np.int8)
    clauses[0, 0] = 1  # trivially satisfied by x1=True

    traces = [
        {"satisfiable": True, "tokens": short_tokens, "clauses": clauses},
        {"satisfiable": True, "tokens": long_tokens, "clauses": clauses},
    ]

    # Would raise (ValueError: sequence length exceeds max_len) if the long
    # trace weren't filtered out before being fed to the model.
    rate = satisfying_completion_rate(model, traces, n_vars, device="cpu")
    assert rate is not None


def test_checkpoint_roundtrips_dim_feedforward(tmp_path):
    """save_checkpoint must persist the *resolved* dim_feedforward (reading
    it off the model itself), not the 0="auto" sentinel that may have been
    passed in via SATModelConfig -- otherwise load_checkpoint would
    reconstruct a differently-shaped model and fail to load the state dict."""
    sat_cfg = SATConfig(n_vars=4, k=2, alpha=2.0)

    explicit_cfg = SATModelConfig(d_model=8, num_layers=1, nhead=2, dim_feedforward=17)
    explicit_model = SATTransformer(
        sat_cfg.n_vars, d_model=explicit_cfg.d_model, num_layers=explicit_cfg.num_layers,
        nhead=explicit_cfg.nhead, max_len=5, dim_feedforward=explicit_cfg.dim_feedforward or None,
    )
    save_checkpoint(explicit_model, [], str(tmp_path / "explicit"), sat_cfg, explicit_cfg)
    _, explicit_saved = load_checkpoint(str(tmp_path / "explicit"))
    assert explicit_saved["dim_feedforward"] == 17

    auto_cfg = SATModelConfig(d_model=8, num_layers=1, nhead=2, dim_feedforward=0)
    auto_model = SATTransformer(
        sat_cfg.n_vars, d_model=auto_cfg.d_model, num_layers=auto_cfg.num_layers,
        nhead=auto_cfg.nhead, max_len=5, dim_feedforward=auto_cfg.dim_feedforward or None,
    )
    save_checkpoint(auto_model, [], str(tmp_path / "auto"), sat_cfg, auto_cfg)
    reloaded_auto, auto_saved = load_checkpoint(str(tmp_path / "auto"))
    assert auto_saved["dim_feedforward"] == 4 * auto_cfg.d_model  # SATTransformer's own default

    # And the reload must actually succeed (right shapes) and produce a
    # working model, not just report the right number.
    x = torch.zeros(1, 3, sat_cfg.n_vars + 1)
    reloaded_auto(x)
