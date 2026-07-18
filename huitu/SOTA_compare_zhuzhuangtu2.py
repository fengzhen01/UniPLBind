import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 数据准备
# =========================
datasets = {
    "PepPI-Test639": {
        "methods": ["PepBind", "PepBCL", "DeepProSite", "ALLSites", "UniPLBind"],
        "MCC": [0.348, 0.312, 0.397, 0.316, 0.380],
        "AUROC": [0.767, 0.804, 0.861, 0.817, 0.928],
        "AUPRC": [np.nan, np.nan, np.nan, np.nan, 0.361],
        "colors": ["#F2D4CF", "#E8B7AF", "#DA8E84", "#C85F56", "#A83C3A"]
    },
    "COACH355": {
        "methods": ["GraphBind", "DeepProSite", "CLAPE-SMB", "SOPE-MsL", "UniPLBind"],
        "MCC": [0.403, 0.394, 0.388, 0.444, 0.459],
        "AUROC": [0.860, 0.810, 0.831, 0.900, 0.935],
        "AUPRC": [0.363, 0.419, 0.414, 0.450, 0.454],
        "colors": ["#D7EEF6", "#B6DDEB", "#8EC5DA", "#5EA9C7", "#358CB1"]
    },
    "ATP-41": {
        "methods": ["TargetS", "ATPseq", "DELIA", "E2EATP", "ATP-Pred", "UniPLBind"],
        "MCC": [0.580, 0.639, 0.612, 0.649, 0.648, 0.682],
        "AUROC": [0.872, 0.878, 0.906, 0.913, 0.948, 0.971],
        "AUPRC": [np.nan, np.nan, np.nan, np.nan, 0.685, 0.717],
        "colors": ["#D8F5F2", "#B8ECE6", "#8ADFD6", "#54C9BE", "#1FA89F", "#007F7F"]
    }
}

# =========================
# 2. 横向柱状图绘制函数
# =========================
def plot_metric_horizontal(metric_name, datasets, save_path=None):
    fig, ax = plt.subplots(figsize=(11, 9))
    y_positions, y_labels, bar_colors, values = [], [], [], []

    current_y = 0
    group_centers = []

    for dataset_name, data in datasets.items():
        methods = data["methods"]
        metric_values = data[metric_name]
        colors = data["colors"]

        start_y = current_y
        for method, value, color in zip(methods, metric_values, colors):
            if not np.isnan(value):
                y_positions.append(current_y)
                y_labels.append(method)
                values.append(value)
                bar_colors.append(color)
                current_y += 1

        end_y = current_y - 1
        group_centers.append((dataset_name, (start_y + end_y)/2))
        current_y += 1.2  # 组间空隙

    bars = ax.barh(y_positions, values, color=bar_colors, edgecolor='none')

    # 添加数值标签
    for bar, value in zip(bars, values):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, f"{value:.3f}",
                va="center", ha="left", fontsize=12)

    # y轴标签
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=12)

    # 横坐标
    ax.set_xlabel(metric_name, fontsize=14, fontweight='bold')

    # 设置横坐标范围
    if metric_name == "MCC" or metric_name == "AUPRC":
        ax.set_xlim(0, 1.0)
    else:  # AUROC
        ax.set_xlim(0, 1.0)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # 网格
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 标注组名
    for dataset_name, center_y in group_centers:
        ax.text(-0.15, center_y, dataset_name, va="center", ha="right",
                fontsize=12, fontweight="normal", transform=ax.get_yaxis_transform())

    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.show()

# =========================
# 3. 绘制三张图
# =========================
plot_metric_horizontal("MCC", datasets, save_path="Pep_SMB_MCC.png")
plot_metric_horizontal("AUROC", datasets, save_path="Pep_SMB_AUROC.png")
plot_metric_horizontal("AUPRC", datasets, save_path="Pep_SMB_AUPRC.png")