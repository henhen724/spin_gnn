"""Forward/backward sanity check for SpinTransformer. Unlike SpinGNN there's
no linear special case to check exactly (attention + FFN are nonlinear by
construction), so this checks shapes, that a forward+backward pass runs
without error, and that gradients are finite -- the level of rigor that's
actually meaningful for this architecture."""
import torch

from spin_gnn.transformer_baseline import (
    SpinTransformer,
    count_params,
    find_matching_d_model,
    gnn_param_count,
)


def test_forward_shape_and_backward():
    b, k, n = 3, 5, 7
    model = SpinTransformer(context_length=k, n_spins=n, d_model=16, num_layers=2)
    x = torch.zeros(b, k, n, 2)
    x[..., 1] = 1.0  # arbitrary valid one-hot ("all up")

    logits = model(x)
    assert logits.shape == (b, n, 2)

    loss = logits.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_find_matching_d_model_hits_target_closely():
    k, n, num_layers = 4, 32, 2
    target = gnn_param_count(k, hidden_dim=32, num_layers=num_layers)
    d = find_matching_d_model(target, k, n, num_layers=num_layers)
    achieved = count_params(SpinTransformer(k, n, d_model=d, num_layers=num_layers))
    assert abs(achieved - target) / target < 0.05
