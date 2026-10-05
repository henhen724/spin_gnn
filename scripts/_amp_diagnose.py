"""Diagnose AMP non-convergence at large alpha: design structure vs damping vs beta."""
import numpy as np, torch
from spin_gnn.amp import AMPConfig, run_amp, teacher_data, map_estimate, _student_prior
from spin_gnn.config import DisorderConfig

def stats(x):
    s = x[:, :, 0]
    m = s.mean(0)
    ac1 = ((x[:, :, 0] - m) * (x[:, :, 1] - m)).mean(0) / s.var(0).clip(1e-12)  # within-window lag-1
    xc = x.reshape(len(x), -1)
    # top eigenvalue of sample-sample gram vs iid expectation
    g = (xc @ xc.T) / xc.shape[1]
    ev = np.linalg.eigvalsh(g)
    return dict(mean_abs_mag=float(np.abs(m).mean()), lag1_autocorr=float(np.nanmean(ac1)),
                gram_top_eig=float(ev[-1]), gram_ratio_top_mean=float(ev[-1] / ev.mean()))

def run(cfg, alpha, damping, design="teacher", seed=0):
    n = cfg.n_spins
    T = int(alpha * n)
    stride = cfg.context_length if design == "stride" else 1
    j, _, x, y = teacher_data(cfg, int(np.ceil(T * stride / 500)), 500, seed=seed)
    x, y = x[::stride][:T], y[::stride][:T]
    if design == "iid":
        rng = np.random.default_rng(seed + 7)
        x = rng.choice([-1.0, 1.0], size=x.shape)
        h = np.einsum("ijk,tjk->ti", j, x)
        y = np.where(rng.random(h.shape) < 1 / (1 + np.exp(-2 * cfg.beta * h)), 1.0, -1.0)
    elif design == "block":
        # keep each spin's real K-lag window statistics, but resample windows
        # independently over (t, j): no shift-overlap between samples, no cross-spin correlation
        rng = np.random.default_rng(seed + 3)
        idx = rng.integers(0, T, size=(T, n))
        x = x[idx, np.arange(n)[None, :]]
        h = np.einsum("ijk,tjk->ti", j, x)
        y = np.where(rng.random(h.shape) < 1 / (1 + np.exp(-2 * cfg.beta * h)), 1.0, -1.0)
    out = run_amp(x, y, cfg, AMPConfig(max_iter=400, damping=damping), j_true=j)
    h = out["history"][-1]
    wm = map_estimate(x, y, _student_prior(cfg, AMPConfig().student), cfg.beta)
    jt = torch.as_tensor(j)
    mm = float(((wm - jt) ** 2).sum() / (jt ** 2).sum())
    st = stats(x)
    print(f"beta={cfg.beta} gamma={cfg.gamma_kind}{cfg.gamma_param} alpha={alpha} damp={damping} design={design:8s} "
          f"iters={h['iter']+1} delta={h['delta']:.2e} amp_mse={h['mse']:.4f} map_mse={mm:.4f} q/qs={h['q_trace']:.2f}/{h['qs_trace']:.2f} {st}", flush=True)

for beta in [1.0, 0.5]:
    cfg = DisorderConfig(n_spins=100, context_length=4, beta=beta, gamma_kind="power_law", gamma_param=1.0)
    for design in ["iid", "block", "stride", "teacher"]:
        run(cfg, 16, 0.5, design)
    for d in [0.8, 0.95]:
        run(cfg, 16, d, "teacher")
cfg = DisorderConfig(n_spins=100, context_length=4, beta=1.0, gamma_kind="identity")
for design in ["iid", "teacher"]:
    run(cfg, 16, 0.5, design)
