import matplotlib.pyplot as plt
import numpy as np

# =========================================================================
# 1. 共享的基础数据配置
# =========================================================================
strategies = [
    "ProstT5 + Ankh", "ProstT5 + ProtT5", "ProstT5 + ESM2",
    "ProstT5 + ESM1b", "ProstT5 + ProtBert", "Ankh + ESM2",
    "Ankh + ProtT5", "ESM2 + ProtBert", "ESM1b + ProtBert"
]
metrics = ["SEN", "MCC", "AUC", "AUPRC"]

# -------------------------------------------------------------------------
# 数据源 A: pro70 (大分子)
# -------------------------------------------------------------------------
pro70_mean = np.array([
    [42.2 / 100, 41.5 / 100, 37.3 / 100, 37.9 / 100, 40.6 / 100, 38.3 / 100, 39.2 / 100, 37.1 / 100, 33.8 / 100],
    [0.396, 0.388, 0.375, 0.384, 0.396, 0.348, 0.369, 0.345, 0.338],
    [0.899, 0.895, 0.887, 0.891, 0.897, 0.880, 0.891, 0.884, 0.882],
    [0.482, 0.472, 0.464, 0.477, 0.474, 0.415, 0.450, 0.412, 0.418]
])
pro70_sd = np.array([
    [4.6 / 100, 4.9 / 100, 2.8 / 100, 1.6 / 100, 2.0 / 100, 7.0 / 100, 6.1 / 100, 5.3 / 100, 3.4 / 100],
    [0.011, 0.010, 0.004, 0.008, 0.005, 0.024, 0.011, 0.012, 0.016],
    [0.002, 0.002, 0.002, 0.005, 0.002, 0.004, 0.001, 0.002, 0.002],
    [0.006, 0.005, 0.004, 0.001, 0.003, 0.009, 0.005, 0.003, 0.011]
])

# -------------------------------------------------------------------------
# 数据源 B: ATP41 (小分子)
# -------------------------------------------------------------------------
atp41_mean = np.array([
    [65.2 / 100, 60.3 / 100, 61.1 / 100, 60.0 / 100, 57.4 / 100, 60.2 / 100, 57.0 / 100, 56.9 / 100, 55.4 / 100],
    [0.682, 0.683, 0.677, 0.665, 0.664, 0.672, 0.668, 0.652, 0.642],
    [0.971, 0.968, 0.967, 0.965, 0.969, 0.969, 0.969, 0.963, 0.960],
    [0.717, 0.713, 0.711, 0.709, 0.707, 0.695, 0.703, 0.676, 0.659]
])
atp41_sd = np.array([
    [2.9 / 100, 2.3 / 100, 2.6 / 100, 2.2 / 100, 2.8 / 100, 0.8 / 100, 5.0 / 100, 0.7 / 100, 2.1 / 100],
    [0.007, 0.005, 0.010, 0.007, 0.010, 0.006, 0.027, 0.008, 0.007],
    [0.002, 0.003, 0.002, 0.005, 0.002, 0.003, 0.002, 0.003, 0.006],
    [0.004, 0.006, 0.003, 0.009, 0.008, 0.008, 0.011, 0.004, 0.010]
])

# =========================================================================
# 2. 全局基础排版样式更新
# =========================================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =========================================================================
# 3. 核心绘图复用函数
# =========================================================================
def draw_perf_plot(data_mean, data_sd, color_palette, title_text, file_prefix):
    fig, ax = plt.subplots(figsize=(6.2,5.2), dpi=300)

    n_metrics = len(metrics)
    n_strategies = len(strategies)
    total_width = 0.8
    single_width = total_width / n_strategies
    x_indexes = np.arange(n_metrics)

    # 循环绘制 9 种策略的柱子
    for j in range(n_strategies):
        bar_positions = x_indexes - (total_width / 2) + (j * single_width) + (single_width / 2)
        ax.bar(
            bar_positions, data_mean[:, j], width=single_width,
            yerr=data_sd[:, j], color=color_palette[j],
            edgecolor='black', linewidth=0.5,
            error_kw=dict(ecolor='black', lw=0.6, capsize=1.5, capthick=0.6)
        )

    # 细节修饰
    ax.set_title(title_text, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Metric Value (Decimal)", fontsize=11)
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 动态构建独立的顶部图例
    custom_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_palette[m], edgecolor='black', linewidth=0.5) for m in
                      range(n_strategies)]
    fig.legend(
        handles=custom_handles, labels=strategies,
        loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3,
        frameon=False, fontsize=9.5
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)

    # 保存出版级别的高清矢量文件
    plt.savefig(f"{file_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{file_prefix}.png", dpi=600, bbox_inches="tight")

    # 激活渲染窗口
    plt.show()


# =========================================================================
# 4. 顺次执行显示（通过 plt.show() 的阻塞机制实现前后交替显示）
# =========================================================================
if __name__ == "__main__":
    # 1. 弹出并显示大分子图 (pro70) - 选用学术蓝渐变系
    colors_pro70 = ['#0A2540', '#1E4A75', '#336BA2', '#4D8CC4', '#70AEE0', '#99CCF5', '#C2E3FF', '#DBEFFF', '#EDF7FF']
    print(">>> 正在展示第一张图：大分子（pro70）。请在查看完成后，关闭该图片窗口以加载下一张。")
    draw_perf_plot(
        data_mean=pro70_mean,
        data_sd=pro70_sd,
        color_palette=colors_pro70,
        title_text="Performance on Macromolecular Interfaces (pro70)",
        file_prefix="Figure_Fusion_pro70_Single"
    )

    # 2. 当你关闭上面那张图的窗口后，以下代码才会触发，弹出第二张小分子图 (ATP41) - 选用学术灰渐变系
    colors_atp41 = ['#1A1A1A', '#3D3D3D', '#545454', '#6B6B6B', '#828282', '#999999', '#B0B0B0', '#C7C7C7', '#DEDEDE']
    print(">>> 正在展示第二张图：小分子（ATP41）。")
    draw_perf_plot(
        data_mean=atp41_mean,
        data_sd=atp41_sd,
        color_palette=colors_atp41,
        title_text="Performance on Localized Micro-pockets (ATP41)",
        file_prefix="Figure_Fusion_ATP41_Single"
    )

    print(">>> 所有图表已绘制并保存完毕。")