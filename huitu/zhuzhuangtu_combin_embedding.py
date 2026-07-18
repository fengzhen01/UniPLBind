import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# =========================================================
# 1. Data
#    Note:
#    SEN / PRE / F1 are given in percentage form in the table
#    MCC / AUC / AUPRC are given in decimal form
# =========================================================

strategies = [
    "ProstT5 + Ankh",
    "ProstT5 + ProtT5",
    "ProstT5 + ESM2",
    "ProstT5 + ESM1b",
    "ProstT5 + ProtBert",
    "Ankh + ESM2",
    "Ankh + ProtT5",
    "ESM2 + ProtBert",
    "ESM1b + ProtBert",
]

metrics = ["SEN", "PRE", "F1", "MCC", "AUC", "AUPRC"]

# -------- Panel A: pro70 --------
pro70_mean = {
    "ProstT5 + Ankh": [42.2, 48.3, 44.8, 0.396, 0.899, 0.482],
    "ProstT5 + ProtT5": [41.5, 47.6, 44.1, 0.388, 0.895, 0.472],
    "ProstT5 + ESM2": [37.3, 49.2, 42.3, 0.375, 0.887, 0.464],
    "ProstT5 + ESM1b": [37.9, 50.2, 43.2, 0.385, 0.891, 0.477],
    "ProstT5 + ProtBert": [40.6, 49.7, 44.6, 0.396, 0.897, 0.474],
    "Ankh + ESM2": [38.3, 43.6, 40.4, 0.348, 0.880, 0.415],
    "Ankh + ProtT5": [39.2, 46.6, 42.1, 0.369, 0.891, 0.450],
    "ESM2 + ProtBert": [37.1, 44.1, 40.0, 0.345, 0.884, 0.412],
    "ESM1b + ProtBert": [33.8, 45.9, 38.8, 0.338, 0.882, 0.418],
}
pro70_std = {
    "ProstT5 + Ankh": [4.6, 2.3, 1.5, 0.011, 0.002, 0.006],
    "ProstT5 + ProtT5": [4.9, 2.5, 1.4, 0.010, 0.002, 0.005],
    "ProstT5 + ESM2": [2.8, 2.5, 0.8, 0.004, 0.002, 0.004],
    "ProstT5 + ESM1b": [1.6, 0.5, 1.0, 0.008, 0.005, 0.001],
    "ProstT5 + ProtBert": [2.0, 1.3, 0.7, 0.005, 0.002, 0.003],
    "Ankh + ESM2": [7.0, 1.9, 2.9, 0.024, 0.004, 0.009],
    "Ankh + ProtT5": [6.1, 3.6, 2.0, 0.011, 0.001, 0.005],
    "ESM2 + ProtBert": [5.3, 2.6, 1.9, 0.012, 0.002, 0.003],
    "ESM1b + ProtBert": [3.4, 0.9, 2.1, 0.016, 0.002, 0.011],
}

# -------- Panel B: ATP41 --------
atp41_mean = {
    "ProstT5 + Ankh": [65.2, 72.9, 68.8, 0.682, 0.971, 0.717],
    "ProstT5 + ProtT5": [60.3, 78.7, 68.3, 0.683, 0.968, 0.713],
    "ProstT5 + ESM2": [61.1, 76.4, 67.9, 0.677, 0.967, 0.712],
    "ProstT5 + ESM1b": [60.0, 75.2, 66.7, 0.665, 0.965, 0.709],
    "ProstT5 + ProtBert": [57.4, 78.5, 66.2, 0.664, 0.969, 0.707],
    "Ankh + ESM2": [60.2, 76.5, 67.4, 0.672, 0.969, 0.695],
    "Ankh + ProtT5": [57.0, 80.0, 66.4, 0.668, 0.969, 0.703],
    "ESM2 + ProtBert": [56.9, 76.3, 65.3, 0.652, 0.963, 0.676],
    "ESM1b + ProtBert": [55.4, 76.1, 64.1, 0.642, 0.960, 0.659],
}
atp41_std = {
    "ProstT5 + Ankh": [2.9, 3.3, 0.7, 0.007, 0.002, 0.004],
    "ProstT5 + ProtT5": [2.3, 2.0, 0.7, 0.005, 0.003, 0.006],
    "ProstT5 + ESM2": [2.6, 2.1, 1.1, 0.010, 0.002, 0.003],
    "ProstT5 + ESM1b": [2.2, 2.3, 0.8, 0.007, 0.005, 0.009],
    "ProstT5 + ProtBert": [2.8, 1.7, 1.4, 0.010, 0.002, 0.008],
    "Ankh + ESM2": [0.8, 1.0, 0.6, 0.006, 0.003, 0.008],
    "Ankh + ProtT5": [5.0, 1.0, 3.2, 0.027, 0.002, 0.011],
    "ESM2 + ProtBert": [0.7, 1.4, 0.7, 0.008, 0.003, 0.004],
    "ESM1b + ProtBert": [2.1, 1.3, 0.9, 0.007, 0.006, 0.010],
}


# =========================================================
# 2. Utility functions
# =========================================================

def convert_to_decimal(arr):
    """
    Convert SEN / PRE / F1 from percentage to decimal.
    Metrics order: SEN, PRE, F1, MCC, AUC, AUPRC
    """
    arr = np.array(arr, dtype=float)
    arr[:3] = arr[:3] / 100.0
    return arr


def build_matrix(mean_dict, std_dict, strategy_order):
    mean_mat = np.array([convert_to_decimal(mean_dict[s]) for s in strategy_order])
    std_mat = np.array([convert_to_decimal(std_dict[s]) for s in strategy_order])
    return mean_mat, std_mat


def make_monochrome_colors(n, highlight_idx=None, highlight_color="#C43C39"):
    """
    Elegant academic blue-gray gradient + one highlighted strategy
    """
    # academic blue gradient
    cmap = cm.get_cmap("Blues")
    base = [cmap(v) for v in np.linspace(0.35, 0.85, n)]

    # slightly desaturate / grey-blue feel
    colors = []
    for c in base:
        r, g, b, a = c
        colors.append((0.92 * r, 0.95 * g, 1.00 * b, a))

    if highlight_idx is not None:
        colors[highlight_idx] = highlight_color
    return colors


def plot_grouped_bar(
        mean_dict,
        std_dict,
        title,
        save_name,
        strategy_order,
        highlight_strategy="ProstT5 + Ankh",
        highlight_mode="strategy",  # "strategy" or "per_metric"
        highlight_color="#C43C39"
):
    """
    highlight_mode:
        - "strategy": highlight one whole strategy (default: ProstT5 + Ankh)
        - "per_metric": highlight the highest bar within each metric group
    """
    mean_mat, std_mat = build_matrix(mean_dict, std_dict, strategy_order)

    n_strategies = len(strategy_order)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.08

    fig, ax = plt.subplots(figsize=(13.5, 5.6))

    # Global style
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    highlight_idx = strategy_order.index(highlight_strategy)
    colors = make_monochrome_colors(n_strategies, highlight_idx, highlight_color=highlight_color)

    # Draw bars
    for i, strategy in enumerate(strategy_order):
        pos = x + (i - (n_strategies - 1) / 2) * width

        # Default color
        bar_color = colors[i]

        # If per-metric highlight is used, colors are set metric-wise later
        if highlight_mode == "strategy":
            current_colors = [bar_color] * n_metrics
        else:
            current_colors = []
            for j in range(n_metrics):
                best_idx = np.argmax(mean_mat[:, j])
                if i == best_idx:
                    current_colors.append(highlight_color)
                else:
                    current_colors.append(colors[i])

        bars = ax.bar(
            pos,
            mean_mat[i],
            width=width,
            yerr=std_mat[i],
            capsize=2.3,
            linewidth=0.6,
            edgecolor="white",
            color=current_colors,
            alpha=0.95,
            label=strategy,
            error_kw=dict(elinewidth=0.8, ecolor="dimgray")
        )

        # Optional subtle edge emphasis for highlighted strategy
        if highlight_mode == "strategy" and i == highlight_idx:
            for b in bars:
                b.set_linewidth(1.0)
                b.set_edgecolor("black")

    # Axes formatting
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    ax.set_ylim(0.30, 1.02)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add note on scaling
    ax.text(
        0.995, 1.015,
        "Note: SEN/PRE/F1 converted from % to proportion",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="dimgray"
    )

    # Legend
    ax.legend(
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.20),
        loc="upper center"
    )

    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{save_name}.pdf", bbox_inches="tight")
    plt.show()


# =========================================================
# 3. Plot
#    Default: highlight ProstT5 + Ankh
#    If you want highest bar in each metric highlighted,
#    change highlight_mode to "per_metric"
# =========================================================

plot_grouped_bar(
    mean_dict=pro70_mean,
    std_dict=pro70_std,
    title="Fusion strategies on macromolecular interfaces (pro70)",
    save_name="fusion_pro70_grouped_bar",
    strategy_order=strategies,
    highlight_strategy="ProstT5 + Ankh",
    highlight_mode="strategy",  # or "per_metric"
    highlight_color="#C43C39"  # academic red
)

plot_grouped_bar(
    mean_dict=atp41_mean,
    std_dict=atp41_std,
    title="Fusion strategies on localized micro-pockets (ATP41)",
    save_name="fusion_atp41_grouped_bar",
    strategy_order=strategies,
    highlight_strategy="ProstT5 + Ankh",
    highlight_mode="strategy",  # or "per_metric"
    highlight_color="#C43C39"
)