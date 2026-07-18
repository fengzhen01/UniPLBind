import numpy as np
import matplotlib.pyplot as plt

# 1. 定义融合策略和核心指标（两组图共有 8 种交集模型以便上下完美对齐）
strategies = [
    "ProstT5 + Ankh", "ProstT5 + ProtT5", "ProstT5 + ESM2",
    "ProstT5 + ESM1b", "ProstT5 + ProtBert", "Ankh + ESM2",
    "Ankh + ProtT5", "ESM2 + ProtBert"
]
metrics = ["PRE", "F1", "MCC", "AUPRC"]

# 2. 录入数据（全部归一化到 0.0 - 1.0 范围）
pro70_matrix = np.array([
    [0.4828, 0.4506, 0.3958, 0.4819],  # ProstT5 + Ankh
    [0.4756, 0.4435, 0.3877, 0.4720],  # ProstT5 + ProtT5
    [0.4923, 0.4246, 0.3753, 0.4643],  # ProstT5 + ESM2
    [0.5025, 0.4320, 0.3847, 0.4768],  # ProstT5 + ESM1b
    [0.4971, 0.4469, 0.3960, 0.4737],  # ProstT5 + ProtBert
    [0.4357, 0.4077, 0.3477, 0.4148],  # Ankh + ESM2
    [0.4657, 0.4255, 0.3688, 0.4504],  # Ankh + ProtT5
    [0.4406, 0.4031, 0.3446, 0.4123]   # ESM2 + ProtBert
])

atp41_matrix = np.array([
    [0.7295, 0.6887, 0.6824, 0.7167],  # ProstT5 + Ankh
    [0.7873, 0.6830, 0.6831, 0.7130],  # ProstT5 + ProtT5
    [0.7637, 0.6791, 0.6765, 0.7116],  # ProstT5 + ESM2
    [0.7524, 0.6675, 0.6647, 0.7086],  # ProstT5 + ESM1b
    [0.7850, 0.6631, 0.6645, 0.7069],  # ProstT5 + ProtBert
    [0.7648, 0.6734, 0.6720, 0.6951],  # Ankh + ESM2
    [0.8000, 0.6652, 0.6682, 0.7034],  # Ankh + ProtT5
    [0.7626, 0.6518, 0.6521, 0.6764]   # ESM2 + ProtBert
])

# 3. 设置专属调色盘：让目标组合最突出，其余灰色过渡
colors = ["#E64B35", "#797D7F", "#95A5A6", "#BDC3C7", "#ECF0F1", "#94A3B8", "#CBD5E1", "#E2E8F0"]

x = np.arange(len(metrics))  # 4个指标大分组的位置
total_width = 0.8
bar_width = total_width / len(strategies)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# 绘制函数
def draw_panel(ax, matrix, title):
    for i in range(len(strategies)):
        # 计算每个柱子在分组内部的偏移量
        offset = (i - len(strategies)/2) * bar_width + bar_width/2
        ax.bar(x + offset, matrix[i], bar_width, label=strategies[i], color=colors[i], edgecolor='black', linewidth=0.3)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_ylim(0.2, 0.9)  # 裁剪空余底盘，让高低对比极度明显
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 绘制上下两个 Panel
draw_panel(ax1, pro70_matrix, "(a) Multi-PLM Fusion Strategies on Macromolecular Test Set (test_pro70)")
draw_panel(ax2, atp41_matrix, "(b) Multi-PLM Fusion Strategies on Localized Pocket Test Set (ATP41)")

# 设置 X 轴大标签
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=12, fontweight='bold')

# 放置全局图例在最底下
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=10, frameon=False)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("fusion_embedding_bars.png", dpi=300)
plt.show()