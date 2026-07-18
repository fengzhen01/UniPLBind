import numpy as np
import pandas as pd


def analyze_binding_dataset(file_path):
    protein_lengths = []
    binding_counts = []
    nonbinding_counts = []
    binding_ratios_per_protein = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) % 2 != 0:
        raise ValueError("文件行数不是偶数，请检查是否每个蛋白都有序列行和标签行。")

    for i in range(0, len(lines), 2):
        seq_line = lines[i]
        label_line = lines[i + 1]

        seq_parts = seq_line.split()
        label_parts = label_line.split()

        if len(seq_parts) < 2 or len(label_parts) < 2:
            raise ValueError(f"第 {i+1} 行或第 {i+2} 行格式不正确。")

        protein_id_seq = seq_parts[0]
        sequence = seq_parts[1]

        protein_id_label = label_parts[0]
        labels = label_parts[1]

        if protein_id_seq != protein_id_label:
            print(f"警告：序列ID与标签ID不一致：{protein_id_seq} vs {protein_id_label}")

        seq_len = len(sequence)
        label_len = len(labels)

        if seq_len != label_len:
            print(
                f"警告：{protein_id_seq} 序列长度与标签长度不一致，"
                f"sequence={seq_len}, label={label_len}"
            )

        # 为避免长度不一致导致统计错误，取二者较小长度
        min_len = min(seq_len, label_len)
        labels = labels[:min_len]

        binding = labels.count("1")
        nonbinding = labels.count("0")
        total = binding + nonbinding

        if total == 0:
            continue

        protein_lengths.append(total)
        binding_counts.append(binding)
        nonbinding_counts.append(nonbinding)
        binding_ratios_per_protein.append(binding / total * 100)

    protein_chains = len(protein_lengths)
    total_binding = sum(binding_counts)
    total_nonbinding = sum(nonbinding_counts)
    total_residues = total_binding + total_nonbinding

    avg_binding_percent = np.mean(binding_ratios_per_protein)
    sd_binding_percent = np.std(binding_ratios_per_protein, ddof=1)

    ratio1 = total_binding / total_residues * 100
    avg_length = np.mean(protein_lengths)

    result = {
        "Protein chains": protein_chains,
        "Binding residues": total_binding,
        "Non-binding residues": total_nonbinding,
        "Avg Binding residues(± SD)(%)": f"{avg_binding_percent:.2f}(± {sd_binding_percent:.2f})",
        "Ratio1 (%)": round(ratio1, 2),
        "Average length": round(avg_length)
    }

    return result


# 修改为你的文件路径
file_path = r"ATP41.txt"

result = analyze_binding_dataset(file_path)

for k, v in result.items():
    print(f"{k}: {v}")