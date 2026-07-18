import matplotlib.pyplot as plt
import numpy as np

# =========================
# Data
# =========================
models = ["ProstT5", "Ankh", "ProtT5", "ESM2", "ESM1b", "ProtBert"]

pro70 = {
    "SEN":    [39.16, 34.81, 38.71, 35.28, 38.77, 29.85],
    "MCC":    [0.3715, 0.3646, 0.3552, 0.3219, 0.3455, 0.2818],
    "AUC":    [0.8922, 0.8965, 0.8882, 0.8747, 0.8806, 0.8644],
    "AUPRC":  [0.4496, 0.4550, 0.4280, 0.3879, 0.4162, 0.3594],
}

# =========================================================================
# Panel B: Localized Small-Molecule Pockets (ATP387/ATP41)
# =========================================================================
atp41 = {
    "SEN":    [62.11, 59.09, 57.21, 57.65, 54.07, 38.88],
    "MCC":    [0.6730, 0.6524, 0.6670, 0.6488, 0.6285, 0.5035],
    "AUC":    [0.9702, 0.9761, 0.9662, 0.9632, 0.9525, 0.9319],
    "AUPRC":  [0.7073, 0.6744, 0.6992, 0.6731, 0.6519, 0.4829],
}

# Convert F1 from percentage to decimal
pro70["SEN"] = [v / 100 for v in pro70["SEN"]]
atp41["SEN"] = [v / 100 for v in atp41["SEN"]]

# =========================
# Style
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
    "SEN":    {"color": "#D55E00", "marker": "o"},
    "MCC":   {"color": "#0072B2", "marker": "s"},
    "AUC":   {"color": "#009E73", "marker": "^"},
    "AUPRC": {"color": "#CC79A7", "marker": "D"},
}

x = np.arange(len(models))

# =========================
# Helper function
# =========================
def plot_broken_panel(ax_top, ax_bottom, data, title):
    for metric, style in metric_styles.items():
        for ax in [ax_top, ax_bottom]:
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

    # Upper and lower y-axis ranges
    ax_top.set_ylim(0.84, 1.00)
    ax_bottom.set_ylim(0.25, 0.74)

    # Hide connected spines
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)

    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_bottom.tick_params(axis="x", rotation=28)

    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(models, rotation=28, ha="right")

    # Grid
    ax_top.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax_bottom.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    # Remove top/right spines
    for ax in [ax_top, ax_bottom]:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # Add broken-axis diagonal marks
    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs = dict(transform=ax_bottom.transAxes, color="black", clip_on=False, linewidth=1.0)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Add omitted-region note
    ax_bottom.text(
        0.02, 1.03,
        "y-axis break: 0.74–0.84 omitted",
        transform=ax_bottom.transAxes,
        fontsize=8,
        color="dimgray"
    )

    # Highlight ProstT5 and Ankh
    for ax in [ax_top, ax_bottom]:
        ax.axvspan(-0.35, 1.35, color="gray", alpha=0.08, zorder=0)

    ax_top.set_title(title, pad=8, fontweight="bold")


# =========================
# Figure with broken y-axis
# =========================
fig = plt.figure(figsize=(10.5, 4.8))

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1.0, 2.2],
    hspace=0.06,
    wspace=0.18
)

ax1_top = fig.add_subplot(gs[0, 0])
ax1_bottom = fig.add_subplot(gs[1, 0], sharex=ax1_top)

ax2_top = fig.add_subplot(gs[0, 1], sharey=ax1_top)
ax2_bottom = fig.add_subplot(gs[1, 1], sharex=ax2_top, sharey=ax1_bottom)

plot_broken_panel(
    ax1_top,
    ax1_bottom,
    pro70,
    "(a) Macromolecular Interfaces (pro70)"
)

plot_broken_panel(
    ax2_top,
    ax2_bottom,
    atp41,
    "(b) Localized Micro-pockets (ATP41)"
)

# Shared labels
ax1_bottom.set_ylabel("Metric value")
ax1_top.set_ylabel("")

# Hide repeated y labels on right panel
plt.setp(ax2_top.get_yticklabels(), visible=False)
plt.setp(ax2_bottom.get_yticklabels(), visible=False)

# Legend
handles, labels = ax2_top.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4
)

fig.suptitle(
    "Performance comparison of individual PLM embeddings",
    fontsize=13,
    fontweight="bold",
    y=1.08
)

plt.savefig("Figure_X_single_embedding_broken_axis.pdf", bbox_inches="tight")
plt.savefig("Figure_X_single_embedding_broken_axis.png", dpi=600, bbox_inches="tight")
plt.show()