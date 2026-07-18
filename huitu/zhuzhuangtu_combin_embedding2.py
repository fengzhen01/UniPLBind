import matplotlib.pyplot as plt
import numpy as np

# =========================================================================
# 1. 数据配置（6个参数群组，百分比指标均已转换为 0-1 小数）
# =========================================================================
strategies = [
    "ProstT5 + Ankh", "ProstT5 + ProtT5", "ProstT5 + ESM2",
    "ProstT5 + ESM1b", "ProstT5 + ProtBert", "Ankh + ESM2",
    "Ankh + ProtT5", "ESM2 + ProtBert", "ESM1b + ProtBert"
]

metrics = ["SEN", "PRE", "F1", "MCC", "AUC", "AUPRC"]

# 数据源 A: pro70 (大分子)
pro70_mean = np.array([
    [42.2 / 100, 41.5 / 100, 37.3 / 100, 37.9 / 100, 40.6 / 100, 38.3 / 100, 39.2 / 100, 37.1 / 100, 33.8 / 100],
    [48.3 / 100, 47.6 / 100, 49.2 / 100, 50.2 / 100, 49.7 / 100, 43.6 / 100, 46.6 / 100, 44.1 / 100, 45.9 / 100],
    [44.8 / 100, 44.1 / 100, 42.3 / 100, 43.2 / 100, 44.6 / 100, 40.4 / 100, 42.1 / 100, 40.0 / 100, 38.8 / 100],
    [0.396, 0.388, 0.375, 0.384, 0.396, 0.348, 0.369, 0.345, 0.338],
    [0.899, 0.895, 0.887, 0.891, 0.897, 0.880, 0.891, 0.884, 0.882],
    [0.482, 0.472, 0.464, 0.477, 0.474, 0.415, 0.450, 0.412, 0.418]
])
pro70_sd = np.array([
    [4.6 / 100, 4.9 / 100, 2.8 / 100, 1.6 / 100, 2.0 / 100, 7.0 / 100, 6.1 / 100, 5.3 / 100, 3.4 / 100],
    [2.3 / 100, 2.5 / 100, 2.5 / 100, 0.5 / 100, 1.3 / 100, 1.9 / 100, 3.6 / 100, 2.6 / 100, 0.9 / 100],
    [1.5 / 100, 1.4 / 100, 0.8 / 100, 1.0 / 100, 0.7 / 100, 2.9 / 100, 2.0 / 100, 1.9 / 100, 2.1 / 100],
    [0.011, 0.010, 0.004, 0.008, 0.005, 0.024, 0.011, 0.012, 0.016],
    [0.002, 0.002, 0.002, 0.005, 0.002, 0.004, 0.001, 0.002, 0.002],
    [0.006, 0.005, 0.004, 0.001, 0.003, 0.009, 0.005, 0.003, 0.011]
])

# 数据源 B: ATP41 (小分子)
atp41_mean = np.array([
    [65.2 / 100, 60.3 / 100, 61.1 / 100, 60.0 / 100, 57.4 / 100, 60.2 / 100, 57.0 / 100, 56.9 / 100, 55.4 / 100],
    [72.9 / 100, 78.7 / 100, 76.4 / 100, 75.2 / 100, 78.5 / 100, 76.5 / 100, 80.0 / 100, 76.3 / 100, 76.1 / 100],
    [68.8 / 100, 68.3 / 100, 67.9 / 100, 66.7 / 100, 66.2 / 100, 67.4 / 100, 66.4 / 100, 65.3 / 100, 64.1 / 100],
    [0.682, 0.683, 0.677, 0.665, 0.664, 0.672, 0.668, 0.652, 0.642],
    [0.971, 0.968, 0.967, 0.965, 0.969, 0.969, 0.969, 0.963, 0.960],
    [0.717, 0.713, 0.711, 0.709, 0.707, 0.695, 0.703, 0.676, 0.659]
])
atp41_sd = np.array([
    [2.9 / 100, 2.3 / 100, 2.6 / 100, 2.2 / 100, 2.8 / 100, 0.8 / 100, 5.0 / 100, 0.7 / 100, 2.1 / 100],
    [3.3 / 100, 2.0 / 100, 2.1 / 100, 2.3 / 100, 1.7 / 100, 1.0 / 100, 1.0 / 100, 1.4 / 100, 1.3 / 100],
    [0.7 / 100, 0.7 / 100, 1.1 / 100, 0.8 / 100, 1.4 / 100, 0.6 / 100, 3.2 / 100, 0.7 / 100, 0.9 / 100],
    [0.007, 0.005, 0.010, 0.007, 0.010, 0.006, 0.027, 0.008, 0.007],
    [0.002, 0.003, 0.002, 0.005, 0.002, 0.003, 0.002, 0.003, 0.006],
    [0.004, 0.006, 0.003, 0.009, 0.008, 0.008, 0.011, 0.004, 0.010]
])

# =========================================================================
# 2. 全局样式配置
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
# 3. 核心单图多群组绘图函数（含柱头文本旋转）
# =========================================================================
def draw_single_canvas_plot(data_mean, data_sd, color_palette, title_text, file_prefix):
    # 横向加长画布 (13.5 x 6.2) 保证柱头上的小数值标签有充足的排版空间
    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=300)

    n_metrics = len(metrics)
    n_strategies = len(strategies)

    total_width = 0.82
    single_width = total_width / n_strategies
    x_indexes = np.arange(n_metrics)

    # 核心循环绘制柱状图
    for j in range(n_strategies):
        bar_positions = x_indexes - (total_width / 2) + (j * single_width) + (single_width / 2)

        rects = ax.bar(
            bar_positions, data_mean[:, j], width=single_width,
            yerr=data_sd[:, j], color=color_palette[j],
            edgecolor='black', linewidth=0.5,
            error_kw=dict(ecolor='black', lw=0.6, capsize=1.5, capthick=0.6)
        )

        # 为当前策略（当前颜色的这一列柱子）添加旋转 90 度的数值标签
        for i, rect in enumerate(rects):
            height = rect.get_height()
            sd_val = data_sd[i, j]
            # 计算标签的 Y 轴高度偏移量（放置在柱子本体加上标准差误差棒的顶端稍高处）
            label_y = height + sd_val + 0.015

            ax.text(
                rect.get_x() + rect.get_width() / 2., label_y,
                f"{height:.3f}",  # 格式化保留三位小数
                ha='center', va='bottom',
                rotation=90,  # 核心要求：逆时针旋转 90 度
                fontsize=6,  # 微型字体防止拥挤
                color='#111111'  # 确保高清晰度可读性
            )

    # 图表修饰
    ax.set_title(title_text, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Value (Decimal)", fontsize=12)
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0.0, 1.15)  # 适当调高 Y 轴上限，防止旋转后的数值标签超出顶格边界

    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 构建顶部规整图例
    custom_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_palette[m], edgecolor='black', linewidth=0.5) for m in
                      range(n_strategies)]
    fig.legend(
        handles=custom_handles, labels=strategies,
        loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3,
        frameon=False, fontsize=10.5
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.76)  # 留出顶部安全距离给图例

    # 高清多格式保存
    plt.savefig(f"{file_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{file_prefix}.png", dpi=600, bbox_inches="tight")

    # 阻塞式交替显示
    plt.show()


# =========================================================================
# 4. 执行控制流
# =========================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 图 1：大分子 pro70 单图展示（学术蓝渐变，王牌最深蓝高亮）
    # -------------------------------------------------------------------------
    colors_pro70 = ['#0A2540', '#1E4A75', '#336BA2', '#4D8CC4', '#70AEE0', '#99CCF5', '#C2E3FF', '#DBEFFF', '#EDF7FF']
    print(">>> 正在展示第一张大图：大分子（pro70），横轴包含全部6个参数群组。")
    print(">>> 【提示】：查看完大分子图后，请关闭弹出的图片窗口，代码会自动加载下一张小分子图。")

    draw_single_canvas_plot(
        data_mean=pro70_mean,
        data_sd=pro70_sd,
        color_palette=colors_pro70,
        title_text="Comprehensive Fusion Strategy Comparison on Macromolecular Interfaces (pro70)",
        file_prefix="Figure_Fusion_pro70_AllInOne"
    )

    # -------------------------------------------------------------------------
    # 图 2：小分子 ATP41 单图展示（全面更换：深绿到浅绿渐变，王牌森林深绿高亮）
    # -------------------------------------------------------------------------
    colors_atp41_green = ['#00441B', '#006D2C', '#238B45', '#41AB5D', '#74C476', '#A1D99B', '#C7E9C0', '#E5F5E0',
                          '#F7FCF5']
    print(">>> 正在展示第二张大图：小分子（ATP41），颜色已更新为绿色渐变系。")

    draw_single_canvas_plot(
        data_mean=atp41_mean,
        data_sd=atp41_sd,
        color_palette=colors_atp41_green,
        title_text="Comprehensive Fusion Strategy Comparison on Localized Micro-pockets (ATP41)",
        file_prefix="Figure_Fusion_ATP41_AllInOne"
    )

    print(">>> 绘图全部顺利结束。")