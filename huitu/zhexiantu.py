import matplotlib.pyplot as plt
import numpy as np

# =========================
# Data
# =========================
models = ["ProstT5", "Ankh", "ProtT5", "ESM2", "ESM1b", "ProtBert"]

pro70 = {
    "F1":    [42.64, 41.08, 41.47, 38.21, 40.71, 33.99],
    "MCC":   [0.3715, 0.3646, 0.3552, 0.3219, 0.3455, 0.2818],
    "AUC":   [0.8922, 0.8965, 0.8882, 0.8747, 0.8806, 0.8644],
    "AUPRC": [0.4496, 0.4550, 0.4280, 0.3879, 0.4162, 0.3594],
}

atp41 = {
    "F1":    [67.67, 65.56, 66.46, 65.03, 62.74, 49.37],
    "MCC":   [0.6730, 0.6524, 0.6670, 0.6488, 0.6285, 0.5035],
    "AUC":   [0.9702, 0.9761, 0.9662, 0.9632, 0.9525, 0.9319],
    "AUPRC": [0.7073, 0.6744, 0.6992, 0.6731, 0.6519, 0.4829],
}

# Convert F1 from percentage to decimal
pro70["F1"] = [v / 100 for v in pro70["F1"]]
atp41["F1"] = [v / 100 for v in atp41["F1"]]

# =========================
# Plot style
# =========================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 1.1,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

metric_styles = {
    "F1":    {"color": "#D55E00", "marker": "o"},
    "MCC":   {"color": "#0072B2", "marker": "s"},
    "AUC":   {"color": "#009E73", "marker": "^"},
    "AUPRC": {"color": "#CC79A7", "marker": "D"},
}

x = np.arange(len(models))

# =========================
# Figure
# =========================
fig, axes = plt.subplots(
    1, 2,
    figsize=(10.2, 4.1),
    sharey=True,
    constrained_layout=True
)

panels = [
    (axes[0], pro70, "(a) Macromolecular Interfaces (pro70)"),
    (axes[1], atp41, "(b) Localized Micro-pockets (ATP41)")
]

for ax, data, title in panels:
    for metric, style in metric_styles.items():
        ax.plot(
            x,
            data[metric],
            label=metric,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=6.2,
            markeredgewidth=0.8,
            markeredgecolor="white",
            alpha=0.96
        )

    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=28, ha="right")
    ax.set_ylim(0.25, 1.02)
    ax.set_yticks(np.arange(0.3, 1.05, 0.1))
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Highlight the strongest embeddings visually
    ax.axvspan(-0.35, 1.35, color="gray", alpha=0.08, zorder=0)
    ax.text(
        0.5, 0.275,
        "Dominant embeddings",
        ha="center",
        va="bottom",
        fontsize=8,
        color="dimgray"
    )

axes[0].set_ylabel("Metric value")
axes[1].legend(
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(0.02, 0.02),
    ncol=2
)

fig.suptitle(
    "Performance comparison of individual PLM embeddings",
    fontsize=13,
    fontweight="bold",
    y=1.04
)

# =========================
# Save
# =========================
plt.savefig("Figure_X_single_embedding_lineplot.pdf", bbox_inches="tight")
plt.savefig("Figure_X_single_embedding_lineplot.png", dpi=600, bbox_inches="tight")
plt.show()