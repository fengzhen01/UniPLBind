import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib import rcParams
import seaborn as sns

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
        color_scheme='modern',  # 'modern', 'classic', 'warm', 'cool', 'vibrant'
        show_residue_labels=False,
        highlight_misclassified=False
):
    """
    增强版的残基得分映射图

    Parameters:
    -----------
    color_scheme : str
        配色方案: 'modern', 'classic', 'warm', 'cool', 'vibrant'
    show_residue_labels : bool
        是否显示残基索引标签
    highlight_misclassified : bool
        是否高亮错误分类的残基
    """

    # 定义配色方案
    COLOR_SCHEMES = {
        'modern': {
            'score_line': '#2C3E50',
            'score_fill': '#3498DB',
            'score_high': '#E74C3C',
            'true_cmap': ['#ECF0F1', '#2C3E50'],
            'cat_colors': {
                'TN': '#BDC3C7',
                'TP': '#27AE60',
                'FP': '#F39C12',
                'FN': '#E74C3C'
            },
            'threshold': '#E67E22',
            'bg_color': '#F8F9FA'
        },
        'classic': {
            'score_line': '#1a1a2e',
            'score_fill': '#16213e',
            'score_high': '#e94560',
            'true_cmap': ['#f5f5f5', '#1a1a2e'],
            'cat_colors': {
                'TN': '#d9d9d9',
                'TP': '#2ca02c',
                'FP': '#ff7f0e',
                'FN': '#d62728'
            },
            'threshold': '#e94560',
            'bg_color': '#ffffff'
        },
        'warm': {
            'score_line': '#5D4037',
            'score_fill': '#FF6F00',
            'score_high': '#D32F2F',
            'true_cmap': ['#FFF3E0', '#5D4037'],
            'cat_colors': {
                'TN': '#FFE0B2',
                'TP': '#FF8F00',
                'FP': '#E65100',
                'FN': '#B71C1C'
            },
            'threshold': '#D84315',
            'bg_color': '#FFF8E1'
        },
        'cool': {
            'score_line': '#0D47A1',
            'score_fill': '#1976D2',
            'score_high': '#E53935',
            'true_cmap': ['#E3F2FD', '#0D47A1'],
            'cat_colors': {
                'TN': '#BBDEFB',
                'TP': '#1E88E5',
                'FP': '#FB8C00',
                'FN': '#C62828'
            },
            'threshold': '#EF6C00',
            'bg_color': '#F5F9FF'
        },
        'vibrant': {
            'score_line': '#2C3E50',
            'score_fill': '#8E44AD',
            'score_high': '#E74C3C',
            'true_cmap': ['#F8F9FA', '#2C3E50'],
            'cat_colors': {
                'TN': '#95A5A6',
                'TP': '#2ECC71',
                'FP': '#F1C40F',
                'FN': '#E74C3C'
            },
            'threshold': '#E67E22',
            'bg_color': '#FDFEFE'
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

    # 创建图形
    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[4.5, 0.4, 0.4],
        hspace=0.08
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # 设置背景色
    ax1.set_facecolor(colors['bg_color'])

    # ----------------------------
    # 上方：得分曲线
    # ----------------------------
    # 填充区域（得分高于阈值）
    ax1.fill_between(
        x, threshold, prob,
        where=(prob >= threshold),
        color=colors['score_fill'],
        alpha=0.15,
        interpolate=True,
        label='Binding region'
    )

    # 得分曲线
    ax1.plot(
        x,
        prob,
        linewidth=2.0,
        color=colors['score_line'],
        label='Predicted binding score',
        zorder=3
    )

    # 实际结合位点标记
    true_pos = x[true_label == 1]
    true_prob = prob[true_label == 1]
    ax1.scatter(
        true_pos,
        true_prob,
        s=35,
        marker='o',
        color=colors['score_high'],
        edgecolors='white',
        linewidth=1.5,
        zorder=4,
        label='True binding residues'
    )

    # 如果有错误分类，高亮显示
    if highlight_misclassified:
        mis_idx = (true_label != pred_label)
        if np.any(mis_idx):
            ax1.scatter(
                x[mis_idx],
                prob[mis_idx],
                s=50,
                marker='X',
                color='#FF6B6B',
                edgecolors='black',
                linewidth=1,
                zorder=5,
                label='Misclassified',
                alpha=0.8
            )

    # 阈值线
    ax1.axhline(
        threshold,
        linestyle='--',
        linewidth=1.5,
        color=colors['threshold'],
        alpha=0.7,
        label=f'Threshold = {threshold:.2f}'
    )

    # 高置信区域
    ax1.axhspan(
        high_conf, 1.0,
        alpha=0.06,
        color=colors['score_high'],
        label=f'High confidence (≥{high_conf:.2f})'
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
            alpha=0.6,
            label=f'Smoothed (window={smooth_window})'
        )

    # 美化坐标轴
    ax1.set_ylabel('Binding Score', fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(x.min() - 1, x.max() + 1)

    # 添加网格
    ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)

    # 标题
    ax1.set_title(f'{protein_name} - Residue Binding Prediction',
                  fontsize=13, fontweight='bold', pad=15)

    # 图例（放在左上角或自定义位置）
    legend = ax1.legend(
        frameon=True,
        fontsize=8,
        ncol=2,
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
        fancybox=False,
        edgecolor='#CCCCCC',
        facecolor='white',
        framealpha=0.9
    )
    legend.get_frame().set_linewidth(0.5)

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

    # 添加残基标签
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

    # 自定义图例
    legend_elements = [
        Patch(facecolor=colors['cat_colors']['TN'], label='TN', edgecolor='white', linewidth=0.5),
        Patch(facecolor=colors['cat_colors']['TP'], label='TP', edgecolor='white', linewidth=0.5),
        Patch(facecolor=colors['cat_colors']['FP'], label='FP', edgecolor='white', linewidth=0.5),
        Patch(facecolor=colors['cat_colors']['FN'], label='FN', edgecolor='white', linewidth=0.5)
    ]

    ax3.legend(
        handles=legend_elements,
        frameon=True,
        fontsize=8,
        ncol=4,
        loc='upper right',
        bbox_to_anchor=(1.0, -0.6),
        fancybox=False,
        edgecolor='#CCCCCC',
        facecolor='white',
        framealpha=0.9
    )

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
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    plt.show()

    return fig


# 使用示例
def batch_plot_with_enhancements(csv_file, protein_list, color_scheme='modern'):
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
                smooth_window=3,
                color_scheme=color_scheme,
                show_residue_labels=True,
                highlight_misclassified=True
            )
        except Exception as e:
            print(f"Error plotting {protein_name}: {e}")


# 测试单个蛋白质
if __name__ == "__main__":
    # 测试数据
    # csv_file = 'case_study_PPI_3_2026-06-13 17_47_26.048495(test).csv'
    csv_file = 'case_study1_PPI_1_2026-06-13 18_04_54.490447(test).csv'
    # test_protein = '1jtd__B'
    test_protein = '3d7v__A'

    # 测试不同配色方案
    for scheme in ['modern', 'classic', 'warm', 'cool', 'vibrant']:
        print(f"\nTesting color scheme: {scheme}")
        try:
            plot_residue_score_map_with_strips_enhanced(
                csv_file=csv_file,
                protein_name=test_protein,
                save_path=f'test_{scheme}_map.png',
                color_scheme=scheme,
                show_residue_labels=True,
                highlight_misclassified=True
            )
        except Exception as e:
            print(f"Error with {scheme}: {e}")