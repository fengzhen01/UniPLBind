import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib import rcParams

# 设置全局字体和样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2


def plot_residue_score_map_with_strips_enhanced(
        csv_file,
        protein_name,
        save_path=None,
        threshold=0.5,
        high_conf=0.7,
        smooth_window=None,
        color_scheme='modern',  # 'modern' or 'classic'
        show_residue_labels=False
):
    """
    增强版的残基得分映射图（仅保留modern和classic两种配色）

    Parameters:
    -----------
    color_scheme : str
        配色方案: 'modern' 或 'classic'
    show_residue_labels : bool
        是否显示残基索引标签
    """

    # 定义配色方案（True的配色使用modern方案）
    MODERN_TRUE_COLORS = {
        'true_cmap': ['#ECF0F1', '#2C3E50'],
        'cat_colors': {
            'TN': '#BDC3C7',        # RGB: (189, 195, 199)  → 灰色
            'TP': '#27AE60',        # RGB: (39, 174, 96)    → 绿色
            'FP': '#F39C12',        # RGB: (243, 156, 18)   → 橙色
            'FN': '#E74C3C'         # RGB: (231, 76, 60)    → 红色
        }
    }

    COLOR_SCHEMES = {
        'modern': {
            'score_line': '#2C3E50',        # rgb(44, 62, 80)
            'score_fill': '#3498DB',        # rgb(52, 152, 219)
            'score_high': '#E74C3C',        # rgb(231, 76, 60)
            'true_cmap': MODERN_TRUE_COLORS['true_cmap'],
            'cat_colors': MODERN_TRUE_COLORS['cat_colors'],
            'threshold': '#E67E22',        #  rgb(230, 126, 34)
            'bg_color': '#F8F9FA',         #  rgb(248, 249, 250)
            'grid_color': '#E8ECEF'        #  rgb(232, 236, 239)
        },
        'classic': {
            'score_line': '#1a1a2e',
            'score_fill': '#16213e',
            'score_high': '#e94560',
            'true_cmap': MODERN_TRUE_COLORS['true_cmap'],  # 使用modern的True配色
            'cat_colors': MODERN_TRUE_COLORS['cat_colors'],  # 使用modern的category配色
            'threshold': '#e94560',
            'bg_color': '#ffffff',
            'grid_color': '#E8ECEF'
        }
    }

    colors = COLOR_SCHEMES[color_scheme]

    # 读取数据
    df = pd.read_csv(csv_file)
    sub = df[df['protein'] == protein_name].copy()

    if sub.empty:
        raise ValueError(f'Protein "{protein_name}" not found in {csv_file}.')

    sub = sub.sort_values('residue_index')

    x = sub['residue_index'].values
    prob = sub['pred_prob'].values
    true_label = sub['true_label'].values.astype(int)
    pred_label = sub['pred_label'].values.astype(int)

    # 自动计算category
    if 'category' not in sub.columns:
        categories = []
        for y_true, y_pred in zip(true_label, pred_label):
            if y_true == 1 and y_pred == 1:
                categories.append('TP')
            elif y_true == 0 and y_pred == 0:
                categories.append('TN')
            elif y_true == 0 and y_pred == 1:
                categories.append('FP')
            else:
                categories.append('FN')
        sub['category'] = categories

    category = sub['category'].values

    # 编码
    cat_to_code = {'TN': 0, 'TP': 1, 'FP': 2, 'FN': 3}
    cat_code = np.array([cat_to_code[c] for c in category]).reshape(1, -1)
    true_arr = true_label.reshape(1, -1)

    # 创建图形 - 增加顶部空间用于图例
    # fig = plt.figure(figsize=(12, 6.5))
    # gs = fig.add_gridspec(
    #     nrows=4,
    #     ncols=1,
    #     height_ratios=[0.6, 4.5, 0.5, 0.5],  # 增加图例行
    #     hspace=0.05
    # )

    fig = plt.figure(figsize=(12, 5.5))  # 整体高度也相应降低
    gs = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[0.1, 3.0, 0.5, 0.5],  # 第1个（图例）压到最小，第2个（得分图）从4.5降到3.0
        hspace=0.05
    )

    # 图例放在最顶部
    # ax_legend = fig.add_subplot(gs[0])
    # ax_legend.axis('off')  # 隐藏坐标轴

    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax3 = fig.add_subplot(gs[3], sharex=ax1)

    # 设置背景色
    ax1.set_facecolor(colors['bg_color'])

    # ----------------------------
    # 顶部：图例
    # ----------------------------
    # 创建图例元素
    # legend_elements_top = [
    #     plt.Line2D([0], [0], color=colors['score_line'], linewidth=2, label='Predicted binding score'),
    #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['score_high'],
    #                markersize=8, label='True binding residues'),
    #     plt.Line2D([0], [0], color=colors['threshold'], linewidth=1.5, linestyle='--',
    #                label=f'Threshold = {threshold:.2f}'),
    #     Patch(facecolor=colors['score_fill'], alpha=0.15, label='Binding region'),
    #     Patch(facecolor=colors['score_high'], alpha=0.06, label=f'High confidence (≥{high_conf:.2f})')
    # ]

    # if smooth_window is not None and smooth_window > 1:
    #     legend_elements_top.append(
    #         plt.Line2D([0], [0], color=colors['score_fill'], linewidth=1.2, linestyle=':',
    #                    alpha=0.6, label=f'Smoothed (window={smooth_window})')
    #     )
    #
    # ax_legend.legend(
    #     handles=legend_elements_top,
    #     frameon=True,
    #     fontsize=8,
    #     ncol=3,
    #     loc='center',
    #     fancybox=False,
    #     edgecolor='#CCCCCC',
    #     facecolor='white',
    #     framealpha=0.9
    # )

    # ----------------------------
    # 中间：得分曲线
    # ----------------------------
    # 填充区域（得分高于阈值）
    ax1.fill_between(
        x, threshold, prob,
        where=(prob >= threshold),
        color=colors['score_fill'],
        alpha=0.15,
        interpolate=True
    )

    # 得分曲线
    ax1.plot(
        x,
        prob,
        linewidth=2.0,
        color=colors['score_line'],
        zorder=3
    )

    # 实际结合位点标记 - 去掉白色边缘
    true_pos = x[true_label == 1]
    true_prob = prob[true_label == 1]
    ax1.scatter(
        true_pos,
        true_prob,
        s=35,
        marker='o',
        color=colors['score_high'],
        edgecolors='none',  # 去掉白色边缘
        zorder=4
    )

    # 阈值线
    ax1.axhline(
        threshold,
        linestyle='--',
        linewidth=1.5,
        color=colors['threshold'],
        alpha=0.7
    )

    # 高置信区域
    ax1.axhspan(
        high_conf, 1.0,
        alpha=0.06,
        color=colors['score_high']
    )

    # 可选平滑曲线
    if smooth_window is not None and smooth_window > 1:
        smooth_prob = (
            pd.Series(prob)
            .rolling(window=smooth_window, center=True, min_periods=1)
            .mean()
            .values
        )
        ax1.plot(
            x,
            smooth_prob,
            linewidth=1.2,
            linestyle=':',
            color=colors['score_fill'],
            alpha=0.6
        )

    # 美化坐标轴
    ax1.set_ylabel('Binding Score', fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(x.min() - 1, x.max() + 1)

    # 添加网格
    ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, color=colors['grid_color'])
    ax1.set_axisbelow(True)

    # 标题
    ax1.set_title(f'{protein_name} - Residue Binding Prediction',
                  fontsize=13, fontweight='bold', pad=15)

    # 添加统计信息
    n_tp = np.sum(category == 'TP')
    n_tn = np.sum(category == 'TN')
    n_fp = np.sum(category == 'FP')
    n_fn = np.sum(category == 'FN')
    accuracy = (n_tp + n_tn) / len(category)

    stats_text = (
        f'TP: {n_tp}  TN: {n_tn}  FP: {n_fp}  FN: {n_fn}  '
        f'Acc: {accuracy:.2%}'
    )
    ax1.text(
        0.98, 0.02, stats_text,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')
    )

    # 确保x轴刻度标签可见（调整与下方子图的间距）
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # ----------------------------
    # 中间：真实标签
    # ----------------------------
    true_cmap = ListedColormap(colors['true_cmap'])
    true_norm = BoundaryNorm([-0.5, 0.5, 1.5], true_cmap.N)

    ax2.imshow(
        true_arr,
        aspect='auto',
        cmap=true_cmap,
        norm=true_norm,
        extent=[x.min() - 0.5, x.max() + 0.5, 0, 1],
        zorder=1
    )

    ax2.set_yticks([])
    ax2.set_ylabel('True', rotation=0, labelpad=20, va='center',
                   fontsize=9, fontweight='bold')
    ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # 添加边框
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('#999999')

    # ----------------------------
    # 下方：分类条带
    # ----------------------------
    cat_colors = [colors['cat_colors'][cat] for cat in ['TN', 'TP', 'FP', 'FN']]
    cat_cmap = ListedColormap(cat_colors)
    cat_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cat_cmap.N)

    ax3.imshow(
        cat_code,
        aspect='auto',
        cmap=cat_cmap,
        norm=cat_norm,
        extent=[x.min() - 0.5, x.max() + 0.5, 0, 1],
        zorder=1
    )

    ax3.set_yticks([])
    ax3.set_ylabel('Class', rotation=0, labelpad=20, va='center',
                   fontsize=9, fontweight='bold')
    ax3.set_xlabel('Residue Index', fontsize=11, fontweight='bold')

    # 显示残基标签
    if show_residue_labels and len(x) <= 100:
        step = max(1, len(x) // 30)
        ax3.set_xticks(x[::step])
        ax3.set_xticklabels(x[::step].astype(int), rotation=45, ha='right', fontsize=7)
    else:
        ax3.tick_params(axis='x', labelsize=8)

    # 添加边框
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('#999999')

    # 分类图例（放在底部）
    legend_elements_bottom = [
        Patch(facecolor=colors['cat_colors']['TN'], label='TN', edgecolor='white', linewidth=0.5),
        Patch(facecolor=colors['cat_colors']['TP'], label='TP', edgecolor='white', linewidth=0.5),
        Patch(facecolor=colors['cat_colors']['FP'], label='FP', edgecolor='white', linewidth=0.5),
        Patch(facecolor=colors['cat_colors']['FN'], label='FN', edgecolor='white', linewidth=0.5)
    ]

    ax3.legend(
        handles=legend_elements_bottom,
        frameon=True,
        fontsize=8,
        ncol=4,
        loc='upper right',
        bbox_to_anchor=(1.0, -0.55),
        fancybox=False,
        edgecolor='#CCCCCC',
        facecolor='white',
        framealpha=0.9
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    plt.show()

    return fig


# 批量绘制函数
def batch_plot_enhanced(csv_file, protein_list, color_scheme='modern'):
    """批量绘制增强版本"""
    for protein_name in protein_list:
        save_path = f'enhanced_{color_scheme}_PPI_score_map_{protein_name}.png'
        print(f"绘制: {protein_name} (配色: {color_scheme})")
        try:
            plot_residue_score_map_with_strips_enhanced(
                csv_file=csv_file,
                protein_name=protein_name,
                save_path=save_path,
                threshold=0.5,
                high_conf=0.7,
                # smooth_window=3,
                color_scheme=color_scheme,
                show_residue_labels=True
            )
        except Exception as e:
            print(f"Error plotting {protein_name}: {e}")


# 测试代码
if __name__ == "__main__":
    # 测试数据
    csv_file = 'case_study1_PPI_1_2026-06-13 18_04_54.490447(test).csv'
    # csv_file = 'case_study_ATP_3_2026-06-13 17_47_26.048495(test).csv'
    # csv_file = 'case_study_DNA_3_2026-06-23 10_32_09.779475(test).csv'
    # test_protein = '1jtd__B'
    test_protein = '3d7v__A'
    # test_protein = '4RQV_A'
    # test_protein = '6dt7_B_DNA'
    # test_protein = '5gpc_A_DNA'

    # 测试modern配色
    print("\n=== Testing Modern Color Scheme ===")
    try:
        plot_residue_score_map_with_strips_enhanced(
            csv_file=csv_file,
            protein_name=test_protein,
            save_path='test_modern_fixed.png',
            color_scheme='modern',
            show_residue_labels=True,
            # smooth_window=3
        )
    except Exception as e:
        print(f"Error with modern: {e}")

    # 测试classic配色
    print("\n=== Testing Classic Color Scheme ===")
    try:
        plot_residue_score_map_with_strips_enhanced(
            csv_file=csv_file,
            protein_name=test_protein,
            save_path='test_classic_fixed.png',
            color_scheme='classic',
            show_residue_labels=True,
            # smooth_window=3
        )
    except Exception as e:
        print(f"Error with classic: {e}")