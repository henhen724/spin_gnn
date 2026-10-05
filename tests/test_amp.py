"""Checks for spin_gnn/amp.py: denoisers against brute force, and AMP on a
small teacher-student instance against the MAP baseline.

Run with: python -m pytest tests/test_amp.py -q (inside a job allocation,
see scripts/slurm/run_tests.sbatch -- not on the login node).
"""
import numpy as np
import torch

from spin_gnn.amp import (AMPConfig, _gauss_hermite, map_estimate, output_denoiser, predictive_nll,
                          prior_denoiser, run_amp, teacher_data, _student_prior)
from spin_gnn.config import DisorderConfig

DT = torch.float64


def _log_z_brute(y, omega, v, beta):
    h = np.linspace(omega - 12 * np.sqrt(v), omega + 12 * np.sqrt(v), 200001)
    dens = np.exp(-((h - omega) ** 2) / (2 * v)) / np.sqrt(2 * np.pi * v)
    return np.log(np.trapz(dens / (1 + np.exp(-2 * beta * y * h)), h))


def test_output_denoiser_matches_finite_differences():
    quad = _gauss_hermite(64, "cpu", DT)
    beta, eps = 0.8, 1e-4
    for y, omega, v in [(1.0, 0.3, 0.5), (-1.0, 1.2, 2.0), (1.0, -2.0, 0.05)]:
        args = lambda o: (torch.tensor([y], dtype=DT), torch.tensor([o], dtype=DT), torch.tensor([v], dtype=DT))
        g, dg, log_z = output_denoiser(*args(omega), beta, quad)
        assert abs(float(log_z) - _log_z_brute(y, omega, v, beta)) < 1e-8
        lp = _log_z_brute(y, omega + eps, v, beta)
        lm = _log_z_brute(y, omega - eps, v, beta)
        l0 = _log_z_brute(y, omega, v, beta)
        assert abs(float(g) - (lp - lm) / (2 * eps)) < 1e-6
        assert abs(float(dg) - (lp - 2 * l0 + lm) / eps**2) < 1e-4


def test_output_denoiser_small_variance_limit_is_continuous():
    quad = _gauss_hermite(48, "cpu", DT)
    y, om = torch.tensor([1.0, -1.0], dtype=DT), torch.tensor([0.4, 0.4], dtype=DT)
    g_a, dg_a, _ = output_denoiser(y, om, torch.full((2,), 1e-6, dtype=DT), 0.7, quad)
    g_b, dg_b, _ = output_denoiser(y, om, torch.full((2,), 1e-9, dtype=DT), 0.7, quad)
    assert torch.allclose(g_a, g_b, atol=1e-5) and torch.allclose(dg_a, dg_b, atol=1e-4)


def test_prior_denoiser_matches_closed_form():
    rng = np.random.default_rng(0)
    k = 3
    a = rng.standard_normal((k, k))
    sigma = a @ a.T + 0.1 * np.eye(k)
    b = rng.standard_normal((5, k, k))
    lam = b @ b.transpose(0, 2, 1)
    gam = rng.standard_normal((5, k))
    w, c = prior_denoiser(torch.tensor(gam), torch.tensor(lam), torch.tensor(sigma))
    c_ref = np.linalg.inv(np.linalg.inv(sigma) + lam)
    w_ref = (c_ref @ gam[..., None])[..., 0]
    assert np.allclose(c.numpy(), c_ref, atol=1e-10) and np.allclose(w.numpy(), w_ref, atol=1e-10)
    # zero prior covariance (masked self-coupling) -> exactly zero posterior
    w0, c0 = prior_denoiser(torch.tensor(gam), torch.tensor(lam), torch.zeros(k, k, dtype=DT))
    assert float(w0.abs().max()) == 0.0 and float(c0.abs().max()) == 0.0


def test_amp_learns_teacher_and_tracks_map():
    cfg = DisorderConfig(n_spins=60, context_length=3, beta=1.0, gamma_kind="power_law", gamma_param=1.0)
    j, _, x, y = teacher_data(cfg, n_sequences=4, t_steps=120, seed=0)  # alpha = T/N = 8
    out = run_amp(x, y, cfg, AMPConfig(max_iter=300, tol=1e-8), j_true=j)
    hist = out["history"]
    assert hist[-1]["delta"] < 1e-6, "AMP did not converge"
    assert hist[-1]["mse"] < 0.6, hist[-1]

    w_map = map_estimate(x, y, _student_prior(cfg, AMPConfig().student), cfg.beta)
    j_t = torch.as_tensor(j)
    mse_map = float(((w_map - j_t) ** 2).sum() / (j_t**2).sum())
    # Posterior mean should be at least about as good as MAP in MSE.
    assert hist[-1]["mse"] < mse_map + 0.02, (hist[-1]["mse"], mse_map)

    _, _, x_te, y_te = teacher_data(cfg, n_sequences=2, t_steps=200, seed=1, j=j)
    nll_teacher = predictive_nll(x_te, y_te, j_t, beta=cfg.beta)
    nll_amp = predictive_nll(x_te, y_te, out["w_hat"], out["C"], beta=cfg.beta)
    assert nll_teacher < nll_amp < np.log(2)
