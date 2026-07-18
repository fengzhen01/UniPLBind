import matplotlib.pyplot as plt
import numpy as np

# =========================================================================
# 1. 数据配置（共 8 种策略，6个参数群组，百分比指标均已转换为 0-1 小数）
# =========================================================================
strategies = [
    "ProstT5", "ProstT5 + Ankh", "ProstT5 + ProtT5", "ProstT5 + ESM2",
    "ProstT5 + ESM1b", "ProstT5 + ProtBert", "Ankh + ESM2", "Ankh + ProtT5"
]

metrics = ["SEN", "PRE", "F1", "MCC", "AUC", "AUPRC"]

# -------------------------------------------------------------------------
# 数据源 A: pro70 (大分子) - 已补全 ProstT5 单模型的 Mean 与 SD
# -------------------------------------------------------------------------
pro70_mean = np.array([
    [39.2 / 100, 42.2 / 100, 41.5 / 100, 37.3 / 100, 37.9 / 100, 40.6 / 100, 38.3 / 100, 39.2 / 100],  # SEN
    [46.8 / 100, 48.3 / 100, 47.6 / 100, 49.2 / 100, 50.2 / 100, 49.7 / 100, 43.6 / 100, 46.6 / 100],  # PRE
    [42.4 / 100, 44.8 / 100, 44.1 / 100, 42.3 / 100, 43.2 / 100, 44.6 / 100, 40.4 / 100, 42.1 / 100],  # F1
    [0.371, 0.396, 0.388, 0.375, 0.384, 0.396, 0.348, 0.369],  # MCC
    [0.892, 0.899, 0.895, 0.887, 0.891, 0.897, 0.880, 0.891],  # AUC
    [0.450, 0.482, 0.472, 0.464, 0.477, 0.474, 0.415, 0.450]  # AUPRC
])
pro70_sd = np.array([
    [4.5 / 100, 4.6 / 100, 4.9 / 100, 2.8 / 100, 1.6 / 100, 2.0 / 100, 7.0 / 100, 6.1 / 100],  # SEN
    [1.6 / 100, 2.3 / 100, 2.5 / 100, 2.5 / 100, 0.5 / 100, 1.3 / 100, 1.9 / 100, 3.6 / 100],  # PRE
    [1.5 / 100, 1.4 / 100, 1.4 / 100, 0.8 / 100, 1.0 / 100, 0.7 / 100, 2.9 / 100, 2.0 / 100],  # F1
    [0.013, 0.011, 0.010, 0.004, 0.008, 0.005, 0.024, 0.011],  # MCC
    [0.001, 0.002, 0.002, 0.002, 0.005, 0.002, 0.004, 0.001],  # AUC
    [0.004, 0.006, 0.005, 0.004, 0.001, 0.003, 0.009, 0.005]  # AUPRC
])

# -------------------------------------------------------------------------
# 数据源 B: ATP41 (小分子) - 已补全 ProstT5 单模型的 Mean 与 SD
# -------------------------------------------------------------------------
atp41_mean = np.array([
    [62.1 / 100, 65.2 / 100, 60.3 / 100, 61.1 / 100, 60.0 / 100, 57.4 / 100, 60.2 / 100, 57.0 / 100],  # SEN
    [74.4 / 100, 72.9 / 100, 78.7 / 100, 76.4 / 100, 75.2 / 100, 78.5 / 100, 76.5 / 100, 80.0 / 100],  # PRE
    [67.7 / 100, 68.8 / 100, 68.3 / 100, 67.9 / 100, 66.7 / 100, 66.2 / 100, 67.4 / 100, 66.4 / 100],  # F1
    [0.673, 0.682, 0.683, 0.677, 0.665, 0.664, 0.672, 0.668],  # MCC
    [0.970, 0.971, 0.968, 0.967, 0.965, 0.969, 0.969, 0.969],  # AUC
    [0.707, 0.717, 0.713, 0.711, 0.709, 0.707, 0.695, 0.703]  # AUPRC
])
atp41_sd = np.array([
    [1.9 / 100, 2.9 / 100, 2.3 / 100, 2.6 / 100, 2.2 / 100, 2.8 / 100, 0.8 / 100, 5.0 / 100],  # SEN
    [2.1 / 100, 3.3 / 100, 2.0 / 100, 2.1 / 100, 2.3 / 100, 1.7 / 100, 1.0 / 100, 1.0 / 100],  # PRE
    [0.5 / 100, 0.7 / 100, 0.7 / 100, 1.1 / 100, 0.8 / 100, 1.4 / 100, 0.6 / 100, 3.2 / 100],  # F1
    [0.004, 0.007, 0.005, 0.010, 0.007, 0.010, 0.006, 0.027],  # MCC
    [0.002, 0.002, 0.003, 0.002, 0.005, 0.002, 0.003, 0.002],  # AUC
    [0.006, 0.004, 0.006, 0.003, 0.009, 0.008, 0.008, 0.011]  # AUPRC
])

# =========================================================================
# 2. 定制色彩系统（去边框扁平化风格）
# =========================================================================
# 大分子 (pro70): 红色标尺柱 + 经典学术蓝渐变系
colors_pro70 = ['#D32F2F', '#1E4A75', '#336BA2', '#4D8CC4', '#70AEE0', '#99CCF5', '#C2E3FF', '#DBEFFF']

# 小分子 (ATP41): 红色标尺柱 + 森林深绿渐变系
colors_atp41 = ['#D32F2F', '#006D2C', '#238B45', '#41AB5D', '#74C476', '#A1D99B', '#C7E9C0', '#E5F5E0']

# =========================================================================
# 3. 全局基础排版参数
# =========================================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10.5,
    "axes.linewidth": 1.1,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =========================================================================
# 4. 核心绘图复用函数
# =========================================================================
def draw_single_canvas_plot(data_mean, data_sd, color_palette, title_text, file_prefix):
    fig, ax = plt.subplots(figsize=(13.0, 6.2), dpi=300)

    n_metrics = len(metrics)
    n_strategies = len(strategies)

    total_width = 0.82
    single_width = total_width / n_strategies
    x_indexes = np.arange(n_metrics)

    # 循环渲染各个柱群
    for j in range(n_strategies):
        bar_positions = x_indexes - (total_width / 2) + (j * single_width) + (single_width / 2)

        # 💡 已修改：edgecolor='none' 且 linewidth=0，完美隐去柱子外框线
        rects = ax.bar(
            bar_positions, data_mean[:, j], width=single_width,
            yerr=data_sd[:, j], color=color_palette[j],
            edgecolor='none', linewidth=0,
            error_kw=dict(ecolor='black', lw=0.6, capsize=1.5, capthick=0.6)
        )

        # 渲染 90 度旋转的柱头数值标签
        for i, rect in enumerate(rects):
            height = rect.get_height()
            sd_val = data_sd[i, j]
            label_y = height + sd_val + 0.015  # 动态浮于误差棒上方，避免重叠

            ax.text(
                rect.get_x() + rect.get_width() / 2., label_y,
                f"{height:.3f}",
                ha='center', va='bottom',
                rotation=90,
                fontsize=7.5,
                color='#111111'
            )

    # 图表装饰与美化
    ax.set_title(title_text, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Value (Decimal)", fontsize=12)
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0.0, 1.15)  # 留出顶部安全距离给标签文字

    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 💡 已修改：图例方块的边框线也同步隐去，保持极致纯净的视觉统一
    custom_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_palette[m], edgecolor='none', linewidth=0) for m in
                      range(n_strategies)]
    fig.legend(
        handles=custom_handles, labels=strategies,
        loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=4,  # 扩展为4列，排版更对称紧凑
        frameon=False, fontsize=10.5
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.76)

    # 高清文件储存
    plt.savefig(f"{file_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{file_prefix}.png", dpi=600, bbox_inches="tight")

    # 展示窗口
    plt.show()


# =========================================================================
# 5. 执行控制流
# =========================================================================
if __name__ == "__main__":
    # 1. 第一张图：大分子 pro70 单图展现
    print(">>> 正在展示第一张大图：大分子（pro70）。最左侧红柱为带误差棒的 ProstT5。")
    draw_single_canvas_plot(
        data_mean=pro70_mean,
        data_sd=pro70_sd,
        color_palette=colors_pro70,
        title_text="Comprehensive Model & Fusion Performance Comparison on Macromolecular Interfaces (pro70)",
        file_prefix="Figure_Baseline_pro70_Final"
    )

    # 2. 第二张图：小分子 ATP41 单图展现
    print(">>> 正在展示第二张大图：小分子（ATP41）。最左侧红柱为带误差棒的 ProstT5。")
    draw_single_canvas_plot(
        data_mean=atp41_mean,
        data_sd=atp41_sd,
        color_palette=colors_atp41,
        title_text="Comprehensive Model & Fusion Performance Comparison on Localized Micro-pockets (ATP41)",
        file_prefix="Figure_Baseline_ATP41_Final"
    )

    print(">>> 最终完美版图表已成功导出。")