import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 数据准备
# =========================
datasets = {

    "PPI-Test70": {
        "methods": ["SCRIBER", "DeepPPISP", "DELPHI", "EnsemPPIS", "ALLSites", "UniPLBind"],
        "MCC": [0.159, 0.206, 0.236, 0.277, 0.319, 0.397],
        "AUROC": [0.635, 0.671, 0.690, 0.719, 0.755, 0.898],
        "AUPRC": [0.307, 0.320, 0.360, 0.405, 0.438, 0.480],
        # 枣红色系，由浅到深
        "colors": ["#F2D4CF", "#E8B7AF", "#DA8E84", "#C85F56", "#A83C3A", "#7F1D1D"]
    },
    "DPI-Test129": {
        "methods": ["GraphBind", "iDRNA-ITF", "CLAPE-DB", "EGPDI", "IPDLPre", "ALLSites", "UniPLBind"],
        "MCC": [0.311, 0.401, 0.373, 0.522, 0.492, 0.480, 0.527],
        "AUROC": [0.841, 0.883, 0.855, 0.941, 0.914, 0.927, 0.967],
        "AUPRC": [0.489, np.nan, 0.495, np.nan, 0.506, 0.517, 0.538],
        # 青蓝色系，由浅到深
        "colors": ["#D7EEF6", "#B6DDEB", "#8EC5DA", "#5EA9C7", "#358CB1", "#166D93", "#0A4F70"]
    },
    "RPI-Test117": {
        "methods": ["GraphBind", "iDRNA-ITF", "CLAPE-RB", "IPDLPre", "ALLSites", "UniPLBind"],
        "MCC": [0.202, 0.236, 0.208, 0.291, 0.303, 0.330],
        "AUROC": [0.734, 0.760, 0.784, 0.814, 0.853, 0.921],
        "AUPRC": [0.179, np.nan, 0.199, 0.255, np.nan, 0.291],
        # 青色系，由浅到深
        "colors": ["#D8F5F2", "#B8ECE6", "#8ADFD6", "#54C9BE", "#1FA89F", "#007F7F"]
    }

}


# =========================
# 2. 绘制单个指标的横向柱状图
# =========================
def plot_metric_horizontal(metric_name, datasets, save_path=None):
    fig, ax = plt.subplots(figsize=(11, 9))

    y_positions = []
    y_labels = []
    bar_colors = []
    values = []

    current_y = 0
    group_centers = []

    for dataset_name, data in datasets.items():
        methods = data["methods"]
        metric_values = data[metric_name]
        colors = data["colors"]

        start_y = current_y

        for method, value, color in zip(methods, metric_values, colors):
            if not np.isnan(value):  # 缺失值不绘制
                y_positions.append(current_y)
                y_labels.append(method)
                values.append(value)
                bar_colors.append(color)
                current_y += 1

        end_y = current_y - 1
        group_centers.append((dataset_name, (start_y + end_y) / 2))
        current_y += 1.2  # 组间空隙

    # 绘制柱状图（无黑色边框）
    bars = ax.barh(y_positions, values, color=bar_colors, edgecolor='none', linewidth=0)

    # 添加数值标签，字体变大
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=18  # 字体变大
        )

    # y轴方法标签，字体变大
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=18)

    # 横坐标设置
    ax.set_xlabel(metric_name, fontsize=18, fontweight='bold')  # 横坐标变大加粗

    # 根据指标设置不同的横坐标范围
    if metric_name == "MCC" or metric_name == "AUPRC":
        ax.set_xlim(0, 0.6)
    else:  # AUROC
        ax.set_xlim(0, 1.0)

    # 刻度值变大
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    # 标题
    ax.set_title(f"Comparison of {metric_name} across macromolecular binding site datasets",
                 fontsize=18, fontweight="bold")

    # 网格线
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # 去掉上边框和右边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 在左侧标出数据集名称，字体变大，不加粗
    for dataset_name, center_y in group_centers:
        ax.text(
            -0.22, center_y,
            dataset_name,
            va="center",
            ha="right",
            fontsize=18,  # 字体变大
            fontweight="normal",  # 不加粗
            transform=ax.get_yaxis_transform()
        )

    # 反转y轴，让第一组在上面
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")

    plt.show()


# =========================
# 3. 分别绘制 MCC / AUROC / AUPRC
# =========================

plot_metric_horizontal("MCC", datasets, save_path="Macromolecular_MCC_comparison.png")
plot_metric_horizontal("AUROC", datasets, save_path="Macromolecular_AUROC_comparison.png")
plot_metric_horizontal("AUPRC", datasets, save_path="Macromolecular_AUPRC_comparison.png")