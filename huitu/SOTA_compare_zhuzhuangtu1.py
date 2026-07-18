import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. 数据准备
# =========================
data = {
    "PPI-Test70": {
        "methods": ["SCRIBER", "DeepPPISP", "DELPHI", "EnsemPPIS", "ALLSites", "UniPLBind"],
        "MCC":   [0.159, 0.206, 0.236, 0.277, 0.319, 0.397],
        "AUROC": [0.635, 0.671, 0.690, 0.719, 0.755, 0.898],
        "AUPRC": [0.307, 0.320, 0.360, 0.405, 0.438, 0.480],
        # 枣红色系：由浅到深
        "colors": ["#F4D6D0", "#E9B3A8", "#D98778", "#C35D52", "#A53A39", "#7A1F1E"]
    },

    "DPI-Test129": {
        "methods": ["GraphBind", "iDRNA-ITF", "CLAPE-DB", "EGPDI", "IPDLPre", "ALLSites", "UniPLBind"],
        "MCC":   [0.311, 0.401, 0.373, 0.522, 0.492, 0.480, 0.527],
        "AUROC": [0.841, 0.883, 0.855, 0.941, 0.914, 0.927, 0.967],
        "AUPRC": [0.489, np.nan, 0.495, np.nan, 0.506, 0.517, 0.538],
        # 青蓝色系：由浅到深
        "colors": ["#D6EEF5", "#B3DCEB", "#8CC6DA", "#63ADC7", "#378EAF", "#166D92", "#0A4F72"]
    },

    "RPI-Test117": {
        "methods": ["GraphBind", "iDRNA-ITF", "CLAPE-RB", "IPDLPre", "ALLSites", "UniPLBind"],
        "MCC":   [0.202, 0.236, 0.208, 0.291, 0.303, 0.330],
        "AUROC": [0.734, 0.760, 0.784, 0.814, 0.853, 0.921],
        "AUPRC": [0.179, np.nan, 0.199, 0.255, np.nan, 0.291],
        # 青色系：由浅到深
        "colors": ["#D8F5F2", "#B4EBE4", "#87DDD2", "#55C8BE", "#1CA79F", "#007C7A"]
    }
}


# =========================
# 2. 绘图函数
# =========================
def plot_metric(metric_name, display_name=None, save_path=None):
    """
    metric_name: 数据中真实指标名，如 'MCC', 'AUROC', 'AUPRC'
    display_name: 图中显示名字，如 'AUC'
    """
    if display_name is None:
        display_name = metric_name

    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = list(data.keys())
    group_centers = np.arange(len(datasets))   # 三组数据的中心
    max_methods = max(len(data[d]["methods"]) for d in datasets)
    bar_width = 0.10

    for i, dataset in enumerate(datasets):
        methods = data[dataset]["methods"]
        values = data[dataset][metric_name]
        colors = data[dataset]["colors"]
        n_methods = len(methods)

        # 每组内部柱子围绕中心对称分布
        offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * bar_width

        for j, (method, value, color) in enumerate(zip(methods, values, colors)):
            if np.isnan(value):
                continue

            x = group_centers[i] + offsets[j]

            ax.bar(
                x, value,
                width=bar_width * 0.9,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=method if i == 0 else None   # 图例只在第一组加入一次
            )

            # 数值标注
            ax.text(
                x, value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90
            )

    # 坐标轴与标题
    ax.set_xticks(group_centers)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel(display_name, fontsize=12)
    ax.set_xlabel("Datasets", fontsize=12)
    ax.set_title(f"Comparison of {display_name} on Macromolecular Binding Site Datasets",
                 fontsize=14)

    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 去掉顶部和右侧边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 图例放在右侧
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9,
        frameon=False
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")

    plt.show()


# =========================
# 3. 生成三张图
# =========================

# 图1：MCC
plot_metric("MCC", display_name="MCC", save_path="Figure_MCC.png")

# 图2：AUC（按 AUROC 处理）
plot_metric("AUROC", display_name="AUC", save_path="Figure_AUC.png")

# 图3：AUPRC
plot_metric("AUPRC", display_name="AUPRC", save_path="Figure_AUPRC.png")