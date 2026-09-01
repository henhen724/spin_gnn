"""Forward/backward sanity check + causal-mask verification for
SATTransformer. As with test_transformer_shapes.py for SpinTransformer,
there's no closed-form ground truth to check exactly here (self-attention +
FFN are nonlinear), so this checks shapes and that a forward+backward pass
runs with finite gradients -- plus, new here since this model is genuinely
autoregressive (unlike SpinTransformer's fixed-window case), that the causal
mask really blocks information from later positions: changing only the last
input token must not change any earlier position's output logits."""
import numpy as np
import pytest
import torch

from spin_gnn.sat_transformer import SATTransformer, generate_trace, sat_accuracy, sat_nll_loss


def test_forward_shape_and_backward():
    b, l, n = 3, 7, 5
    model = SATTransformer(n_vars=n, d_model=16, num_layers=2, nhead=2, max_len=32)
    x = torch.randint(-1, 2, (b, l, n + 1)).float()
    target = torch.randint(-1, 2, (b, l, n + 1)).float()
    mask = torch.ones(b, l, dtype=torch.bool)

    logits = model(x, valid_mask=mask)
    assert logits.shape == (b, l, n + 1, 3)

    loss = sat_nll_loss(logits, target, mask)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()

    acc = sat_accuracy(logits, target, mask)
    assert 0.0 <= acc <= 1.0


def test_causal_mask_blocks_future_positions():
    n, l = 4, 6
    model = SATTransformer(n_vars=n, d_model=16, num_layers=2, nhead=2, max_len=32)
    model.eval()  # disable dropout so the two forward passes are directly comparable

    torch.manual_seed(0)
    x = torch.randint(-1, 2, (1, l, n + 1)).float()
    x_perturbed = x.clone()
    last = x_perturbed[0, -1]
    # Guarantee the last token actually changes regardless of what it drew:
    # zeros flip to 1, +-1 negate.
    x_perturbed[0, -1] = torch.where(last == 0, torch.ones_like(last), -last)

    with torch.no_grad():
        logits = model(x)
        logits_perturbed = model(x_perturbed)

    torch.testing.assert_close(logits[:, :-1], logits_perturbed[:, :-1])
    assert not torch.allclose(logits[:, -1], logits_perturbed[:, -1])


def test_generate_trace_shape_and_greedy_determinism():
    n, t0, new = 5, 3, 4
    model = SATTransformer(n_vars=n, d_model=16, num_layers=2, nhead=2, max_len=32)
    model.eval()
    prompt = np.random.default_rng(0).integers(-1, 2, size=(t0, n + 1)).astype(np.float32)

    out1 = generate_trace(model, prompt, max_new_tokens=new, temperature=0.0)
    out2 = generate_trace(model, prompt, max_new_tokens=new, temperature=0.0)

    assert out1.shape == (t0 + new, n + 1)
    np.testing.assert_array_equal(out1[:t0], prompt)  # prompt is preserved verbatim
    np.testing.assert_array_equal(out1, out2)  # greedy (temperature=0) decoding is deterministic
    assert set(np.unique(out1)).issubset({-1.0, 0.0, 1.0})


def test_generate_trace_stops_at_max_len():
    n = 3
    model = SATTransformer(n_vars=n, d_model=16, num_layers=1, nhead=1, max_len=5)
    model.eval()
    prompt = np.zeros((4, n + 1), dtype=np.float32)

    out = generate_trace(model, prompt, max_new_tokens=10, temperature=0.0)

    assert out.shape[0] == 5  # stopped once it hit max_len, not 4 + 10


def test_low_rank_qk_uses_custom_attention_path():
    """qk_dim != d_model must route through LowRankSelfAttention (a plain
    ModuleList), not nn.TransformerEncoder -- the two paths have different
    internal structure (.layers vs. direct indexing), which is exactly what
    train_sat_transformer.py's save_checkpoint/sat_hparam_sweep.py branch on."""
    model_full_rank = SATTransformer(n_vars=4, d_model=16, num_layers=2, nhead=2, max_len=32)
    model_low_rank = SATTransformer(n_vars=4, d_model=16, num_layers=2, nhead=2, max_len=32, qk_dim=8)

    assert model_full_rank._custom_attn is False
    assert model_full_rank.qk_dim == 16
    assert model_low_rank._custom_attn is True
    assert model_low_rank.qk_dim == 8


def test_forward_shape_and_backward_low_rank_qk():
    """Same check as test_forward_shape_and_backward, but through
    LowRankSelfAttention/LowRankEncoderLayer (qk_dim < d_model, so the
    per-head Q/K rank is strictly smaller than the per-head V/output rank)."""
    b, l, n = 3, 7, 5
    model = SATTransformer(n_vars=n, d_model=16, num_layers=2, nhead=2, max_len=32, qk_dim=8)
    x = torch.randint(-1, 2, (b, l, n + 1)).float()
    target = torch.randint(-1, 2, (b, l, n + 1)).float()
    mask = torch.ones(b, l, dtype=torch.bool)

    logits = model(x, valid_mask=mask)
    assert logits.shape == (b, l, n + 1, 3)

    loss = sat_nll_loss(logits, target, mask)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_causal_mask_blocks_future_positions_low_rank_qk():
    """Same check as test_causal_mask_blocks_future_positions, through the
    low-rank Q/K attention path."""
    n, l = 4, 6
    model = SATTransformer(n_vars=n, d_model=16, num_layers=2, nhead=2, max_len=32, qk_dim=8)
    model.eval()  # disable dropout so the two forward passes are directly comparable

    torch.manual_seed(0)
    x = torch.randint(-1, 2, (1, l, n + 1)).float()
    x_perturbed = x.clone()
    last = x_perturbed[0, -1]
    x_perturbed[0, -1] = torch.where(last == 0, torch.ones_like(last), -last)

    with torch.no_grad():
        logits = model(x)
        logits_perturbed = model(x_perturbed)

    torch.testing.assert_close(logits[:, :-1], logits_perturbed[:, :-1])
    assert not torch.allclose(logits[:, -1], logits_perturbed[:, -1])


def test_qk_dim_not_divisible_by_nhead_raises():
    with pytest.raises(ValueError, match="qk_dim"):
        SATTransformer(n_vars=4, d_model=16, num_layers=1, nhead=3, max_len=8, qk_dim=10)
