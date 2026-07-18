import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


def plot_residue_score_map_with_strips(
    csv_file,
    protein_name,
    save_path=None,
    threshold=0.5,
    high_conf=0.7,
    smooth_window=None
):
    df = pd.read_csv(csv_file)
    sub = df[df['protein'] == protein_name].copy()

    if sub.empty:
        raise ValueError(f'Protein "{protein_name}" not found in {csv_file}.')

    sub = sub.sort_values('residue_index')

    x = sub['residue_index'].values
    prob = sub['pred_prob'].values
    true_label = sub['true_label'].values.astype(int)
    pred_label = sub['pred_label'].values.astype(int)

    # 如果 CSV 中没有 category，则自动计算
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

    # category 编码
    cat_to_code = {
        'TN': 0,
        'TP': 1,
        'FP': 2,
        'FN': 3
    }
    cat_code = np.array([cat_to_code[c] for c in category]).reshape(1, -1)

    true_arr = true_label.reshape(1, -1)

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[4.0, 0.35, 0.35],
        hspace=0.12
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # ----------------------------
    # 上方：predicted binding score
    # ----------------------------
    ax1.plot(
        x,
        prob,
        linewidth=1.5,
        label='Raw binding score'
    )

    ax1.scatter(
        x[true_label == 1],
        prob[true_label == 1],
        s=22,
        marker='o',
        label='True binding residues'
    )

    ax1.axhline(
        threshold,
        linestyle='--',
        linewidth=1.0,
        label=f'Threshold = {threshold}'
    )

    ax1.axhspan(
        high_conf,
        1.0,
        alpha=0.08,
        label=f'High-confidence region ($p \\geq {high_conf}$)'
    )

    # 可选：平滑曲线，只用于辅助观察
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
            label=f'Moving average (window={smooth_window})'
        )

    ax1.set_ylabel('Binding score')
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_xlim(x.min(), x.max())
    ax1.set_title(protein_name)
    ax1.legend(frameon=False, fontsize=8, ncol=2, loc='upper right')

    # ----------------------------
    # 中间：true label strip
    # ----------------------------
    true_cmap = ListedColormap(['white', 'black'])
    true_norm = BoundaryNorm([-0.5, 0.5, 1.5], true_cmap.N)

    ax2.imshow(
        true_arr,
        aspect='auto',
        cmap=true_cmap,
        norm=true_norm,
        extent=[x.min() - 0.5, x.max() + 0.5, 0, 1]
    )

    ax2.set_yticks([])
    ax2.set_ylabel('True', rotation=0, labelpad=25, va='center')
    ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # ----------------------------
    # 下方：TP/TN/FP/FN strip
    # ----------------------------
    cat_cmap = ListedColormap([
        '#d9d9d9',  # TN
        '#2ca02c',  # TP
        '#ff7f0e',  # FP
        '#d62728'   # FN
    ])
    cat_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cat_cmap.N)

    ax3.imshow(
        cat_code,
        aspect='auto',
        cmap=cat_cmap,
        norm=cat_norm,
        extent=[x.min() - 0.5, x.max() + 0.5, 0, 1]
    )

    ax3.set_yticks([])
    ax3.set_ylabel('Class', rotation=0, labelpad=25, va='center')
    ax3.set_xlabel('Residue index')

    legend_elements = [
        Patch(facecolor='#d9d9d9', label='TN'),
        Patch(facecolor='#2ca02c', label='TP'),
        Patch(facecolor='#ff7f0e', label='FP'),
        Patch(facecolor='#d62728', label='FN')
    ]

    ax3.legend(
        handles=legend_elements,
        frameon=False,
        fontsize=8,
        ncol=4,
        loc='upper right',
        bbox_to_anchor=(1.0, -0.55)
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()


# 完整的蛋白质列表  ATP
protein_list = [
    '3J8Y_K', '3J94_A', '4D79_C', '4RQV_A', '4RV7_C', '4RX6_B',
    '4S1K_A', '4WVY_A', '4WZY_A', '4X2D_A', '4XBR_A', '4XHO_A',
    '4XJX_B', '4XRU_A', '4XRU_B', '4XVU_H', '4YB7_A', '4YDS_A',
    '4ZGN_A', '4ZS4_B', '5A98_A', '5A99_A', '5APB_A', '5BSM_A',
    '5BUR_A', '5COU_A', '5CYR_B', '5D15_B', '5D1O_A', '5D6J_A',
    '5D9H_B', '5DB4_A', '5DN3_A', '5DN6_A', '5E3I_B', '5E84_F',
    '5ECK_A', '5EOU_B', '5ETL_D', '5EWG_A', '5F1C_B'
]

# 完整的蛋白质列表   PPI
protein_list = [
    '1ak4__D', '1b6c__A', '1f60__B', '1jtd__B', '1k5d_A_B', '1ktz__A', '1mah__F', '1ml0__D', '1nrj__A', '1nrj__B',
    '1pxv__A', '1qa9__A', '1svd__M', '1u0s__A', '1z0j__B', '1zbd__B', '2a9k__A', '2b42__A', '2d5r__A', '2doh__X',
    '2f91__A', '2fp4__A', '2g2u__B', '2h0d__B', '2p7v__A', '2v4z__B', '2v8s__V', '2v9t__A', '3ANW__B', '3AYH__B',
    '3B08__B', '3MDB__C', '3MMY__A', '3OUR__B', '3PH0__A', '3PV6__A', '3SHG__A', '3UVJ__A', '3VDO__B', '3VU9__A',
    '3VZ9__D', '3W2W__B', '3ZEU__D', '3ZKQ__A', '3beg__B', '3c5t__A', '3cf4__G', '3cx8__B', '3czu__B', '3d7v__A',
    '3dlq__I', '3e1z__B', '3e5a__A', '3gni__A', '4APX__A', '4AWX__B', '4BH6__D', '4BH6__L', '4BI8__A', '4BI8__B',
    '4BJJ__B', '4DFC__B', '4FQ0__B', '4H3K__A', '4H3K__B', '4IU2__A', '4JE3__B', '4KT3__B', '4KT6__B', '4apx__A',  # 注意 4APX__A 已存在，此处为原列表中的小写 a，按原样保留
    '4awx__B', '4bh6__D', '4bh6__L', '4bi8__A', '4bi8__B', '4bjj__B', '4dfc__B', '4fq0__B', '4h3k__A', '4h3k__B',
    '4iu2__A', '4je3__B', '4kt3__B', '4kt6__B', '7cei__A'
]

# 完整的蛋白质列表   DPI
protein_list = [
    '5dy0_A_DNA', '5fgp_A_DNA', '5ejk_A_DNA', '5dac_A_DNA', '5d2s_A_DNA', '5eyb_A_DNA',
    '5iud_D_DNA', '4zm2_B_DNA', '5j3e_A_DNA', '5hr4_J_DNA', '5cr2_A_DNA', '5hrt_A_DNA',
    '5i44_B_DNA', '5j37_E_DNA', '5hnk_A_DNA', '5k5q_B_DNA', '5f55_A_DNA', '5d8c_B_DNA',
    '5kbj_A_DNA', '5itt_A_DNA', '5fd3_A_DNA', '5t9j_B_DNA', '5dlo_A_DNA', '5dwb_A_DNA',
    '5jre_D_DNA', '5f7q_E_DNA', '5jub_A_DNA', '5gke_A_DNA', '5fmp_A_DNA', '5l2x_A_DNA',
    '5trd_A_DNA', '5h1c_A_DNA', '5tgx_A_DNA', '5jjv_B_DNA', '5k7z_A_DNA', '5l6l_D_DNA',
    '5fdk_A_DNA', '5lej_B_DNA', '5tkz_A_DNA', '5hrf_A_DNA', '5m0r_A_DNA', '5m1s_D_DNA',
    '5hso_C_DNA', '5udb_4_DNA', '5udb_9_DNA', '5udb_D_DNA', '5ui5_V_DNA', '5g5t_A_DNA',
    '5gpc_A_DNA', '5j2y_A_DNA', '5gzb_A_DNA', '5u1j_B_DNA', '5t4i_B_DNA', '5k5l_G_DNA',
    '5mhk_D_DNA', '5h58_B_DNA', '5jlt_A_DNA', '5nj8_A_DNA', '5vc8_A_DNA', '5w7g_A_DNA',
    '5w7g_B_DNA', '5oa1_U_DNA', '5w1c_B_DNA', '5h3r_A_DNA', '5o63_B_DNA', '5xfq_A_DNA',
    '5uc6_A_DNA', '5oqn_B_DNA', '5oqo_A_DNA', '5lxu_A_DNA', '5odl_A_DNA', '5vhv_A_DNA',
    '5xvn_B_DNA', '5xvn_F_DNA', '6erp_F_DNA', '5wx9_A_DNA', '5x11_E_DNA', '5ybb_D_DNA',
    '5o6b_A_DNA', '5v9x_A_DNA', '5yi2_A_DNA', '5mpf_B_DNA', '6c0w_K_DNA', '6eu0_E_DNA',
    '6eu0_P_DNA', '6eu0_V_DNA', '6eu1_O_DNA', '5yx2_A_DNA', '5vxn_C_DNA', '5vfx_F_DNA',
    '5xrz_A_DNA', '6fml_G_DNA', '6fml_I_DNA', '6emy_B_DNA', '6ged_A_DNA', '5yzy_C_DNA',
    '6c2s_A_DNA', '5uk7_H_DNA', '5vhe_A_DNA', '5orq_A_DNA', '5vl9_A_DNA', '6f2s_H_DNA',
    '5ond_A_DNA', '5zr1_A_DNA', '5zr1_B_DNA', '5zr1_C_DNA', '5zr1_E_DNA', '6dt7_A_DNA',
    '6dt7_B_DNA', '6bkg_A_DNA', '6g1t_A_DNA', '6gdr_A_DNA', '6c31_A_DNA', '6h8q_A_DNA',
    '5zyu_A_DNA', '6dt1_A_DNA', '5yej_A_DNA', '5zmn_A_DNA', '6cg8_E_DNA', '6eko_A_DNA',
    '6f5f_A_DNA', '6e33_A_DNA', '6en8_A_DNA', '6ee8_M_DNA', '6mzm_A_DNA', '6mzm_B_DNA',
    '6gys_A_DNA', '6gys_C_DNA', '6gys_E_DNA'
]

# CSV 文件路径（您的原始文件）
# csv_file = 'case_study_PPI_3_2026-06-13 17_47_26.048495(test).csv'
# csv_file = 'case_study1_PPI_1_2026-06-13 18_04_54.490447(test).csv'
# csv_file = 'case_study_PPI_0_2026-06-16 15_52_21.349233.csv'
csv_file = 'case_study_DNA_3_2026-06-23 10_32_09.779475(test).csv'

# 批量绘制并保存
for protein_name in protein_list:
    save_path = f'case_PPI_score_map_{protein_name}.png'
    print(f"正在绘制: {protein_name}")
    try:
        plot_residue_score_map_with_strips(
            csv_file=csv_file,
            protein_name=protein_name,
            save_path=save_path
        )
        print(f"已保存: {save_path}")
    except Exception as e:
        print(f"绘制 {protein_name} 时出错: {e}")