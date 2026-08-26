"""Sanity check: on the full (unsparsified) graph, SpinGNN(exact=True,
hidden_dim=0, num_layers=1) must reproduce the analytic Bayes-optimal
next-spin posterior from spin_gnn.simulate exactly (up to float error).

This validates the edge_index/edge_attr convention in graph.py and the
message-passing reduction in model.LagCoupledInputConv against the
closed-form kinetic-Ising posterior -- run this before trusting any learned
result, especially after touching graph.py or model.py.

Run with: python -m pytest tests/test_exact_recovery.py -q
(inside an sh_dev / job allocation with the project venv active -- see
scripts/setup_env.sh. Do not run on the Sherlock login node.)
"""
import numpy as np
import torch

from spin_gnn.config import DisorderConfig
from spin_gnn.dataset import spin_to_onehot
from spin_gnn.graph import build_spin_graph
from spin_gnn.model import SpinGNN
from spin_gnn.simulate import generate_realization


def test_exact_recovery_matches_analytic_posterior():
    cfg = DisorderConfig(n_spins=40, context_length=3, beta=0.7, seed=0)
    realization = generate_realization(cfg, t_steps=20, seed=1)
    spin_graph = build_spin_graph(realization["J"], cfg.context_length, keep_prob=1.0, seed=2)

    model = SpinGNN(cfg.context_length, hidden_dim=0, num_layers=1, exact=True, beta_init=cfg.beta)
    model.eval()

    t = 15
    history = realization["S"][t - cfg.context_length : t]
    x = torch.from_numpy(np.transpose(spin_to_onehot(history), (1, 0, 2)).astype(np.float32))
    edge_index = torch.from_numpy(spin_graph.edge_index)
    edge_attr = torch.from_numpy(spin_graph.edge_attr)

    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
    p_up_model = torch.softmax(logits, dim=-1)[:, 1].numpy()
    p_up_analytic = realization["P"][t]

    np.testing.assert_allclose(p_up_model, p_up_analytic, atol=1e-5)
