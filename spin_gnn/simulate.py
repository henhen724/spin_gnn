"""Exact simulation of the kinetic Ising variant of the disordered spin model
in Zavatone-Veth & Pehlevan (2026), https://doi.org/10.1088/1751-8121/ae38a4.

E(s_t | s_{t-1},...,s_{t-K}) = - s_t^T sum_k J_k s_{t-k}

Because this energy is linear in s_t, under the uniform measure on
{-1,+1}^N the conditional p(s_t | context) factorizes exactly across sites:

    p(s_t,i = +1 | context) = sigmoid(2 * beta * h_t,i),
    h_t,i = sum_{k=1}^K sum_j J_k[i, j] * s_{t-k,j}

This gives an exact, tractable ground truth (the Bayes-optimal next-spin
predictor) to benchmark the self-supervised GNN against, and note that h_t is
itself exactly a linear graph-convolution of the K-step spin history — see
spin_gnn/model.py for why a single linear message-passing layer on the full
graph should recover it exactly.

This module is pure numpy and cheap (O(N^2 K) per step): fine for N in the
few-hundreds to low-thousands range on a normal-partition CPU job, no GPU
needed. Do not run this on the Sherlock login node for anything past a toy
size -- submit via scripts/slurm/generate_data.sbatch or grab `sh_dev` first.
"""
import numpy as np

from .config import DisorderConfig


def _nearest_psd(a, eps=1e-8):
    """Project a symmetric matrix onto the PSD cone by clipping negative
    eigenvalues. Needed for the power_law kind below: |k-k'|^-alpha Toeplitz
    matrices are valid covariances for some (alpha, K) but not guaranteed for
    all -- e.g. alpha=1 (slow decay) can go indefinite as K grows."""
    a_sym = (a + a.T) / 2
    eigvals, eigvecs = np.linalg.eigh(a_sym)
    clipped = np.clip(eigvals, eps, None)
    return (eigvecs * clipped) @ eigvecs.T


def build_gamma(k, kind="identity", param=0.0):
    """Lag-correlation matrix Gamma_{k,k'}.

    - "identity": Gamma = I, lags uncorrelated (Gamma_{k,k'} = delta_{k,k'}).
    - "ones": Gamma = all-ones matrix -- every lag pair has correlation 1, i.e.
      J_k is (numerically, up to the eps floor below) the same matrix for
      every lag k. Rank-1 and singular, so it's PSD-projected like power_law.
    - "alternating": the paper's example, Gamma = I + param*(1-I)*(-1)^(k+k').
    - "power_law": Gamma_kk = 1, Gamma_{k,k'} = |k-k'|^-param for k != k' --
      long-range lag correlation.
    "alternating"/"ones"/"power_law" are all projected to the nearest PSD
    matrix since the raw construction isn't guaranteed PSD for every
    (kind, param, K).
    """
    if kind == "identity":
        return np.eye(k)
    if kind == "ones":
        return _nearest_psd(np.ones((k, k)))
    if kind == "alternating":
        idx = np.arange(1, k + 1)
        off = (-1.0) ** (idx[:, None] + idx[None, :])
        gamma = np.eye(k) + param * (1 - np.eye(k)) * off
        return _nearest_psd(gamma)
    if kind == "power_law":
        idx = np.arange(1, k + 1).astype(float)
        diff = np.abs(idx[:, None] - idx[None, :])
        with np.errstate(divide="ignore"):
            off_diag = diff ** (-param)
        gamma = np.where(diff == 0, 1.0, off_diag)
        return _nearest_psd(gamma)
    raise ValueError(f"unknown gamma kind: {kind!r}")


def sample_disorder(cfg: DisorderConfig, rng: np.random.Generator):
    """Draw one quenched realization of the coupling tensor J, shape (N, N, K)."""
    n, k = cfg.n_spins, cfg.context_length
    gamma = build_gamma(k, cfg.gamma_kind, cfg.gamma_param)
    l = np.linalg.cholesky(gamma)
    z = rng.standard_normal((n, n, k))
    j_corr = z @ l.T
    j = (cfg.coupling_scale / np.sqrt(n)) * j_corr
    idx = np.arange(n)
    j[idx, idx, :] = 0.0
    return j, gamma


def local_field(j, history):
    """history: (K, N) array of the K most recent spin configs, oldest first
    (history[0] = s_{t-K}, history[K-1] = s_{t-1}). Returns h_t, the (N,)
    local field sum_{lag=1}^K J[:, :, lag-1] @ s_{t-lag}.

    J's last axis is lag-1-first (index 0 = lag 1); history is oldest-first.
    This function is where those two opposite conventions get reconciled
    (`history[k - lag]`) -- graph.py's edge_attr reverses J's axis to match
    history's convention instead, so keep that in sync if this loop changes.
    """
    k = j.shape[-1]
    h = np.zeros(j.shape[0])
    for lag in range(1, k + 1):
        s_lag = history[k - lag]  # s_{t-lag}
        h += j[:, :, lag - 1] @ s_lag
    return h


def rollout(j, cfg: DisorderConfig, rng: np.random.Generator, t_steps, init=None):
    """Simulate t_steps of the kinetic Ising chain.

    Returns
    -------
    s : (t_steps, N) array of +-1, the generated trajectory (does not include
        the K burn-in/context configs used to seed it).
    p : (t_steps, N) array of the exact P(s_t,i = +1 | context) at each step --
        the ground-truth Bayes-optimal next-spin distribution, for evaluation
        only (never fed to the GNN).
    """
    n, k = cfg.n_spins, cfg.context_length
    if init is None:
        history = rng.choice([-1.0, 1.0], size=(k, n))
    else:
        history = np.asarray(init, dtype=float).copy()
        assert history.shape == (k, n)

    s = np.empty((t_steps, n))
    p = np.empty((t_steps, n))
    for t in range(t_steps):
        h = local_field(j, history)
        p_up = 1.0 / (1.0 + np.exp(-2.0 * cfg.beta * h))
        s_t = np.where(rng.random(n) < p_up, 1.0, -1.0)
        s[t] = s_t
        p[t] = p_up
        history = np.roll(history, shift=-1, axis=0)
        history[-1] = s_t
    return s, p


def generate_realization(cfg: DisorderConfig, t_steps, seed):
    rng = np.random.default_rng(seed)
    j, gamma = sample_disorder(cfg, rng)
    s, p = rollout(j, cfg, rng, t_steps)
    return {"J": j, "Gamma": gamma, "S": s, "P": p, "cfg": cfg, "seed": seed}


def save_realization(path, realization):
    cfg = realization["cfg"]
    np.savez_compressed(
        path,
        J=realization["J"],
        Gamma=realization["Gamma"],
        S=realization["S"],
        P=realization["P"],
        seed=realization["seed"],
        n_spins=cfg.n_spins,
        context_length=cfg.context_length,
        beta=cfg.beta,
        gamma_kind=cfg.gamma_kind,
        gamma_param=cfg.gamma_param,
        coupling_scale=cfg.coupling_scale,
    )


def load_realization(path):
    d = np.load(path)
    cfg = DisorderConfig(
        n_spins=int(d["n_spins"]),
        context_length=int(d["context_length"]),
        beta=float(d["beta"]),
        gamma_kind=str(d["gamma_kind"]),
        gamma_param=float(d["gamma_param"]),
        coupling_scale=float(d["coupling_scale"]),
        seed=int(d["seed"]),
    )
    return {"J": d["J"], "Gamma": d["Gamma"], "S": d["S"], "P": d["P"], "cfg": cfg, "seed": int(d["seed"])}


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="e.g. $SCRATCH/spin_gnn/data")
    ap.add_argument("--n-spins", type=int, default=200)
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma-kind", choices=["identity", "ones", "alternating", "power_law"], default="identity")
    ap.add_argument("--gamma-param", type=float, default=0.0)
    ap.add_argument("--coupling-scale", type=float, default=1.0)
    ap.add_argument("--t-steps", type=int, default=512)
    ap.add_argument("--n-realizations", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base_cfg = DisorderConfig(
        n_spins=args.n_spins,
        context_length=args.context_length,
        beta=args.beta,
        gamma_kind=args.gamma_kind,
        gamma_param=args.gamma_param,
        coupling_scale=args.coupling_scale,
    )
    for i in range(args.n_realizations):
        seed = args.seed_start + i
        realization = generate_realization(base_cfg, args.t_steps, seed)
        out_path = os.path.join(args.out_dir, f"realization_{seed:05d}.npz")
        save_realization(out_path, realization)
        print(f"wrote {out_path}")
