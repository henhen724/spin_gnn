"""One-off plot for the d_model scaling sweep (nhead=4 fixed).

Compares against the nhead sweep's d_model=64,nhead=4 point. Enrichment
numbers are copied from the corresponding sudoku_attention_distance
summary.json files (baseline: 48267874; dmodel128/dmodel256 analysis jobs
48449673/48449674).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d_model = [64, 128, 256]
head_dim = [16, 32, 64]
mean_o1_solve_rate = [0.980, 0.896, 0.842]
agg_solve_rate = [0.72, 0.7733333333333333, 0.8466666666666667]
solve_rate_k0 = [0.98, 0.55, 0.24]  # d=64 k0 from earlier L12/ff1024 sweep row
final_val_nll = [1.2, 1.6747804066394552, 2.412636435513031]

enrichment_by_layer = {
    64: [1.1800297373646045, 0.9185781537940083, 1.079046243893263, 1.1686119031404325,
         1.6590321965721389, 1.7247475752686032, 1.5277825124802675, 1.4618423412817418,
         1.1028781277968325, 1.0204622359689943, 1.3291745478766779, 0.8919322991099964],
    128: [1.059374654810914, 1.0131814556617738, 0.7881234869684568, 1.4392904444218646,
          1.1686854025522224, 1.027414553470763, 1.0030937068884516, 0.9924824991321539,
          0.9491681966939194, 2.8871699409278104, 1.183811858669829, 1.4754213668606349],
    256: [1.0961432668580642, 0.8910833295998267, 0.46164131154660787, 0.9062403318646443,
          1.0239282341242737, 0.8503206856471254, 0.948710183507761, 0.9296407289064528,
          0.883063095650866, 1.163485401536499, 2.4412463046388693, 2.131150888757019],
}
mean_enrichment = [float(np.mean(enrichment_by_layer[d])) for d in d_model]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.plot(d_model, mean_o1_solve_rate, "o-", label="mean_o1_solve_rate (near-trivial)")
ax.plot(d_model, solve_rate_k0, "s-", label="solve_rate_k0 (0 uncolored, hardest near-trivial)")
ax.plot(d_model, agg_solve_rate, "^-", label="agg_solve_rate (full rho range)")
ax.set_xscale("log", base=2)
ax.set_xticks(d_model)
ax.set_xticklabels([f"{d}\n(head_dim={hd})" for d, hd in zip(d_model, head_dim)])
ax.set_xlabel("d_model (nhead=4 fixed)")
ax.set_ylabel("solve rate")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)
ax.set_title("Solve rate vs d_model")

ax = axes[1]
ax.plot(d_model, mean_enrichment, "o-", color="tab:green")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="untargeted baseline")
ax.set_xscale("log", base=2)
ax.set_xticks(d_model)
ax.set_xticklabels([f"{d}\n(head_dim={hd})" for d, hd in zip(d_model, head_dim)])
ax.set_xlabel("d_model (nhead=4 fixed)")
ax.set_ylabel("mean attention-targeting enrichment")
ax.legend(fontsize=8)
ax.set_title("Mean enrichment vs d_model: flat-to-down,\nnot up, as head_dim grows")

ax = axes[2]
for d in d_model:
    ax.plot(range(12), enrichment_by_layer[d], "o-", label=f"d_model={d}")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("layer")
ax.set_ylabel("enrichment")
ax.legend(fontsize=8)
ax.set_title("Per-layer enrichment: larger d_model concentrates\ninto a few spiky late layers, most layers degrade")

fig.suptitle("Scaling d_model at fixed nhead=4 does not sharpen id-matching attention", fontsize=12)
fig.tight_layout()
fig.savefig("notebooks/sudoku_dmodel_sweep.png", dpi=150)
print("wrote notebooks/sudoku_dmodel_sweep.png")
