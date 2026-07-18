import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_probability_heatmap(csv_file, protein_name, save_path=None):
    df = pd.read_csv(csv_file)
    sub = df[df['protein'] == protein_name].copy()

    if sub.empty:
        raise ValueError(f'Protein "{protein_name}" not found in {csv_file}.')

    sub = sub.sort_values('residue_index')

    prob = sub['pred_prob'].values.reshape(1, -1)
    label = sub['true_label'].values.reshape(1, -1)

    plt.figure(figsize=(12, 1.9))

    # 使用与示例图类似的渐变方式：
    # low score: dark purple/blue
    # middle score: green/cyan
    # high score: yellow
    im = plt.imshow(
        prob,
        aspect='auto',
        cmap='viridis',
        vmin=0,
        vmax=1
    )

    cbar = plt.colorbar(im)
    cbar.set_label('Predicted binding probability', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # 标记真实结合残基
    true_pos = np.where(label.flatten() == 1)[0]

    plt.scatter(
        true_pos,
        np.zeros_like(true_pos),
        marker='|',
        s=220,
        color='#1f77b4',
        linewidth=1.5,
        label='True binding residues'
    )

    plt.yticks([])
    plt.xlabel('Residue index', fontsize=13)
    plt.title(protein_name, fontsize=15)

    # plt.legend(
    #     frameon=False,
    #     fontsize=10,
    #     loc='upper right'
    # )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()


plot_probability_heatmap(
    csv_file='case_study1_PPI_1_2026-06-13 18_04_54.490447(test).csv',
    # csv_file='case_study_ATP_3_2026-06-13 17_47_26.048495(test).csv',
    # csv_file='case_study_DNA_3_2026-06-23 10_32_09.779475(test).csv',
    # protein_name='4RQV_A',
    protein_name='3d7v__A',
    # protein_name='6dt7_B_DNA',
    save_path=r'D:/xiangmu/3NucGMTL-main/visual/case_heatmap_6dt7_B_DNA.png'
)