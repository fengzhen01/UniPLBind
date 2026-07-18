import matplotlib.pyplot as plt

# 1. 准备数据（按 ProstT5 -> ProtBert 的性能趋势降序排列横轴）
models = ["ProstT5", "Ankh", "ProtT5", "ESM2", "ESM1b", "ProtBert"]

# (a) 大分子 pro70 数据
pro70_data = {
    "F1": [42.64, 41.08, 41.47, 38.21, 40.71, 33.99],
    "MCC": [0.3715 * 100, 0.3646 * 100, 0.3552 * 100, 0.3219 * 100, 0.3455 * 100, 0.2818 * 100],  # 放大100倍与百分比对齐
    "AUC": [0.8922 * 100, 0.8965 * 100, 0.8882 * 100, 0.8747 * 100, 0.8806 * 100, 0.8644 * 100],
    "AUPRC": [0.4496 * 100, 0.4550 * 100, 0.4280 * 100, 0.3879 * 100, 0.4162 * 100, 0.3594 * 100]
}

# (b) 小分子 ATP41 数据
atp41_data = {
    "F1": [67.72, 65.56, 66.46, 65.03, 62.74, 49.37],
    "MCC": [0.6730 * 100, 0.6524 * 100, 0.6670 * 100, 0.6488 * 100, 0.6285 * 100, 0.5035 * 100],
    "AUC": [0.9702 * 100, 0.9761 * 100, 0.9662 * 100, 0.9632 * 100, 0.9525 * 100, 0.9319 * 100],
    "AUPRC": [0.7073 * 100, 0.6744 * 100, 0.6992 * 100, 0.6731 * 100, 0.6519 * 100, 0.4829 * 100]
}

metrics = ["F1", "MCC", "AUC", "AUPRC"]
colors = ["#2F4F4F", "#4682B4", "#D96459", "#8C4646"]  # 莫兰迪高级质感配色
markers = ["o", "s", "^", "D"]

# 2. 开始绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# 绘制左图 (a) pro70
for metric, color, marker in zip(metrics, colors, markers):
    ax1.plot(models, pro70_data[metric], label=metric, color=color, marker=marker, linewidth=2, markersize=8)
ax1.set_title("(a) Macromolecular Interfaces (test_pro70)", fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel("Score (%)", fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.tick_params(labelsize=11)

# 绘制右图 (b) ATP41
for metric, color, marker in zip(metrics, colors, markers):
    ax2.plot(models, atp41_data[metric], label=metric, color=color, marker=marker, linewidth=2, markersize=8)
ax2.set_title("(b) Localized Micro-pockets (ATP41)", fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.tick_params(labelsize=11)

# 统一精简样式（IEEE风格：去顶右边框）
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels(models, rotation=15)

# 合并图例到正上方
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=12, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("single_embedding_trends.png", dpi=300)
plt.show()