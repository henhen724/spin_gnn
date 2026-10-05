"""Learning curves for the teacher-student kinetic Ising model via AMP
(spin_gnn/amp.py): sweep the sample ratio alpha = T_total / N and record
AMP's and MAP's reconstruction error and held-out NLL gap to the teacher.

Usage (inside a job, see scripts/slurm/amp_sweep.sbatch):
    python -m spin_gnn.amp_sweep --out amp_results/sweep.csv --alphas 1 2 4 8 16
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from .amp import AMPConfig, StudentConfig, _student_prior, map_estimate, overlaps, predictive_nll, run_amp, teacher_data
from .config import DisorderConfig


def run_point(cfg, alpha, seed, amp, t_seq, test_steps, do_map):
    n_total = int(round(alpha * cfg.n_spins))
    n_seq = max(1, int(np.ceil(n_total / t_seq)))
    j, _, x, y = teacher_data(cfg, n_seq, t_seq, seed=seed)
    x, y = x[:n_total], y[:n_total]
    _, _, x_te, y_te = teacher_data(cfg, max(1, test_steps // t_seq), t_seq, seed=10_000 + seed, j=j)

    t0 = time.time()
    out = run_amp(x, y, cfg, amp, j_true=j)
    j_t = torch.as_tensor(j, device=amp.device, dtype=amp.dtype)
    beta = out["beta"]
    m, q, qs = overlaps(out["w_hat"], j_t)
    last = out["history"][-1]
    row = {
        "n_spins": cfg.n_spins, "context_length": cfg.context_length, "beta": cfg.beta,
        "gamma_kind": cfg.gamma_kind, "gamma_param": cfg.gamma_param, "alpha": alpha, "seed": seed,
        "amp_iters": last["iter"] + 1, "amp_delta": last["delta"], "amp_final_damping": last["damping"], "amp_mse": last["mse"],
        "amp_m_trace": float(torch.trace(m)), "amp_q_trace": float(torch.trace(q)),
        "qs_trace": float(torch.trace(qs)), "amp_seconds": time.time() - t0,
        "nll_teacher": predictive_nll(x_te, y_te, j_t, beta=cfg.beta),
        "nll_amp": predictive_nll(x_te, y_te, out["w_hat"], out["C"], beta=beta),
    }
    if do_map:
        w_map = map_estimate(x, y, _student_prior(cfg, amp.student), beta, device=amp.device, dtype=amp.dtype)
        row["map_mse"] = float(((w_map - j_t) ** 2).sum() / (j_t**2).sum())
        row["nll_map"] = predictive_nll(x_te, y_te, w_map, beta=beta)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-spins", type=int, default=200)
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma-kind", default="power_law", choices=["identity", "ones", "alternating", "power_law"])
    ap.add_argument("--gamma-param", type=float, default=1.0)
    ap.add_argument("--coupling-scale", type=float, default=1.0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1, 2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--t-seq", type=int, default=500, help="length of each training trajectory")
    ap.add_argument("--test-steps", type=int, default=2000)
    ap.add_argument("--damping", type=float, default=0.8)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--student-beta", type=float, default=None)
    ap.add_argument("--student-gamma-kind", default=None)
    ap.add_argument("--student-gamma-param", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = DisorderConfig(n_spins=args.n_spins, context_length=args.context_length, beta=args.beta,
                         gamma_kind=args.gamma_kind, gamma_param=args.gamma_param,
                         coupling_scale=args.coupling_scale)
    amp = AMPConfig(max_iter=args.max_iter, damping=args.damping, device=args.device,
                    student=StudentConfig(beta=args.student_beta, gamma_kind=args.student_gamma_kind,
                                          gamma_param=args.student_gamma_param))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = None
    with open(args.out, "w", newline="") as fh:
        for alpha in args.alphas:
            for seed in args.seeds:
                row = run_point(cfg, alpha, seed, amp, args.t_seq, args.test_steps, not args.no_map)
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                fh.flush()
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main()
