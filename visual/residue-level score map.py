import pandas as pd
import matplotlib.pyplot as plt

def plot_residue_score_map(csv_file, protein_name, save_path=None):
    df = pd.read_csv(csv_file)
    sub = df[df['protein'] == protein_name].copy()

    x = sub['residue_index'].values
    prob = sub['pred_prob'].values
    label = sub['true_label'].values

    plt.figure(figsize=(10, 2.8))

    # predicted probability curve
    plt.plot(x, prob, linewidth=1.5, label='Predicted probability')

    # true binding residues
    binding_x = x[label == 1]
    binding_y = prob[label == 1]
    plt.scatter(binding_x, binding_y, s=18, label='True binding residues')

    # threshold
    plt.axhline(0.5, linestyle='--', linewidth=1, label='Threshold = 0.5')

    plt.xlabel('Residue index')
    plt.ylabel('Binding probability')
    plt.ylim(-0.02, 1.02)
    plt.title(protein_name)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()


# 完整的蛋白质列表
protein_list = [
    '3J8Y_K', '3J94_A', '4D79_C', '4RQV_A', '4RV7_C', '4RX6_B',
    '4S1K_A', '4WVY_A', '4WZY_A', '4X2D_A', '4XBR_A', '4XHO_A',
    '4XJX_B', '4XRU_A', '4XRU_B', '4XVU_H', '4YB7_A', '4YDS_A',
    '4ZGN_A', '4ZS4_B', '5A98_A', '5A99_A', '5APB_A', '5BSM_A',
    '5BUR_A', '5COU_A', '5CYR_B', '5D15_B', '5D1O_A', '5D6J_A',
    '5D9H_B', '5DB4_A', '5DN3_A', '5DN6_A', '5E3I_B', '5E84_F',
    '5ECK_A', '5EOU_B', '5ETL_D', '5EWG_A', '5F1C_B'
]

# CSV 文件路径（您的原始文件）
# csv_file = 'case_study_PPI_0_2026-06-13 10_18_34.687456.csv'
csv_file = 'case_study_DNA_3_2026-06-23 10_32_09.779475(test).csv'

# 批量绘制并保存
for protein_name in protein_list:
    save_path = f'case_PPI_score_map_{protein_name}.png'
    print(f"正在绘制: {protein_name}")
    try:
        plot_residue_score_map(
            csv_file=csv_file,
            protein_name=protein_name,
            save_path=save_path
        )
        print(f"已保存: {save_path}")
    except Exception as e:
        print(f"绘制 {protein_name} 时出错: {e}")