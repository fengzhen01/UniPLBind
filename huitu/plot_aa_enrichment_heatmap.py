import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 基本配置
# ============================================================

# 标准 20 种氨基酸；此顺序会作为热图横坐标
AA_ORDER = [
    "A", "R", "N", "D", "C",
    "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V"
]

# Laplace pseudocount，避免某些氨基酸在某类残基中未出现时出现 log(0)
PSEUDOCOUNT = 1.0

# 输出文件夹
OUTPUT_DIR = Path(r"D:/xiangmu/3NucGMTL-main/huitu/aa_enrichment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 数据集路径配置
#
# 每一行对应一个 benchmark protocol。
# 同一 protocol 中训练集与测试集合并，仅用于描述性统计。
#
# 请按照你的真实路径修改。
# ============================================================

DATASET_CONFIG = [
    {
        "name": "PPI-352/70",
        "files": [
            r"D:/xiangmu/3NucGMTL-main/DataSet/PPI/Train352-Test70/PPI-Train352.txt",
            r"D:/xiangmu/3NucGMTL-main/DataSet/PPI/Train352-Test70/PPI-Test70.txt",
        ],
    },
    {
        "name": "PPI-335/60",
        "files": [
            r"D:/xiangmu/3NucGMTL-main/DataSet/PPI/Train335-Test60-Test315/PPI-Train335.txt",
            r"D:/xiangmu/3NucGMTL-main/DataSet/PPI/Train335-Test60-Test315/PPI-Test60.txt",
        ],
    },
    {
        "name": "DPI-573/129",
        "files": [
            r"D:/xiangmu/3NucGMTL-main/DataSet/DNA/Train/DNA_Train.txt",
            r"D:/xiangmu/3NucGMTL-main/DataSet/DNA/Test/DNA_Test.txt",
        ],
    },
    {
        "name": "RPI-495/117",
        "files": [
            r"D:/xiangmu/3NucGMTL-main/DataSet/RNA/Train/RNA_Train.txt",
            r"D:/xiangmu/3NucGMTL-main/DataSet/RNA/Test/RNA_Test.txt",
        ],
    },
    {
        "name": "PepPI-640/639",
        "files": [
            r"D:/xiangmu/3NucGMTL-main/DataSet/PepPI/PepPI-Train640.txt",
            r"D:/xiangmu/3NucGMTL-main/DataSet/PepPI/PepPI-Test639.txt",
        ],
    },
    {
        "name": "CHEN11/COACH355",
        "files": [
            r"D:/xiangmu/3NucGMTL-main/DataSet/SMB/SMB_Train.txt",
            r"D:/xiangmu/3NucGMTL-main/DataSet/SMB/SMB_Test.txt",
        ],
    },
    {
        "name": "ATP-226/17",
        "files": [
            r"D:/xiangmu/3NucGMTL-main/DataSet/ATP1/ATP226.txt",
            r"D:/xiangmu/3NucGMTL-main/DataSet/ATP1/ATP17.txt",
        ],
    },
    {
        "name": "ATP-387/41",
        "files": [
            r"D:/xiangmu/3NucGMTL-main/DataSet/ATP1/ATP387.txt",
            r"D:/xiangmu/3NucGMTL-main/DataSet/ATP1/ATP41.txt",
        ],
    },
    {
        "name": "ATP-542/202",
        "files": [
            r"D:/xiangmu/3NucGMTL-main/DataSet/ATP1/ATP542.txt",
            r"D:/xiangmu/3NucGMTL-main/DataSet/ATP1/ATP202.txt",
        ],
    },
]


# ============================================================
# 3. 读取两行一个蛋白的数据文件
# ============================================================

def split_id_and_content(line):
    """
    将一行拆成:
        protein_id, sequence_or_label

    例如:
        test639_19 MVKQIES...
        test639_19 0000100...

    返回:
        protein_id, content
    """
    parts = line.strip().split()

    if len(parts) < 2:
        raise ValueError(f"Cannot parse line:\n{line}")

    protein_id = parts[0]
    content = "".join(parts[1:])

    return protein_id, content


def read_sequence_label_records(txt_file):
    """
    读取两行一个蛋白的数据集文件。

    返回列表，每个元素为:
        {
            'protein': protein_id,
            'sequence': sequence,
            'labels': labels
        }
    """
    txt_file = Path(txt_file)

    if not txt_file.exists():
        raise FileNotFoundError(f"Dataset file not found:\n{txt_file}")

    with open(txt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) % 2 != 0:
        raise ValueError(
            f"{txt_file.name} has {len(lines)} non-empty lines, "
            "but the number of lines should be even."
        )

    records = []

    for i in range(0, len(lines), 2):
        seq_id, sequence = split_id_and_content(lines[i])
        label_id, labels = split_id_and_content(lines[i + 1])

        sequence = sequence.upper()
        labels = labels.strip()

        if seq_id != label_id:
            print(
                f"[Warning] Protein IDs differ at lines {i + 1}-{i + 2}: "
                f"{seq_id} vs {label_id}"
            )

        if len(sequence) != len(labels):
            raise ValueError(
                f"Length mismatch in {txt_file.name}, protein {seq_id}: "
                f"sequence length = {len(sequence)}, "
                f"label length = {len(labels)}"
            )

        if not set(labels).issubset({"0", "1"}):
            raise ValueError(
                f"Invalid labels in {txt_file.name}, protein {seq_id}. "
                "Labels must contain only 0 and 1."
            )

        records.append({
            "protein": seq_id,
            "sequence": sequence,
            "labels": labels
        })

    return records


# ============================================================
# 4. 统计一个数据文件中的结合与非结合残基氨基酸组成
# ============================================================

def count_amino_acids_in_file(txt_file):
    """
    统计一个 txt 文件中：
        binding residues (label=1)
        non-binding residues (label=0)

    返回：
        positive_counts, negative_counts, summary
    """
    records = read_sequence_label_records(txt_file)

    positive_counts = Counter({aa: 0 for aa in AA_ORDER})
    negative_counts = Counter({aa: 0 for aa in AA_ORDER})
    unknown_counts = Counter()

    n_chains = 0
    n_binding = 0
    n_nonbinding = 0
    n_unknown = 0

    for record in records:
        n_chains += 1

        sequence = record["sequence"]
        labels = record["labels"]

        for aa, label in zip(sequence, labels):
            if aa not in AA_ORDER:
                unknown_counts[aa] += 1
                n_unknown += 1
                continue

            if label == "1":
                positive_counts[aa] += 1
                n_binding += 1
            else:
                negative_counts[aa] += 1
                n_nonbinding += 1

    summary = {
        "n_chains": n_chains,
        "n_binding_residues": n_binding,
        "n_nonbinding_residues": n_nonbinding,
        "n_unknown_residues": n_unknown,
        "unknown_amino_acids": dict(unknown_counts)
    }

    return positive_counts, negative_counts, summary


# ============================================================
# 5. 合并一个 benchmark protocol 的 train/test 文件
# ============================================================

def summarize_protocol(protocol_name, file_list, pseudocount=1.0):
    """
    对一个 benchmark protocol 合并统计其 train/test 文件。

    计算：
        f_pos(aa) = (count_pos(aa) + alpha) /
                    (sum_pos + 20*alpha)

        f_neg(aa) = (count_neg(aa) + alpha) /
                    (sum_neg + 20*alpha)

        enrichment(aa) = log2(f_pos(aa) / f_neg(aa))
    """
    total_positive = Counter({aa: 0 for aa in AA_ORDER})
    total_negative = Counter({aa: 0 for aa in AA_ORDER})

    total_chains = 0
    total_binding = 0
    total_nonbinding = 0
    total_unknown = 0
    all_unknown = Counter()

    for txt_file in file_list:
        pos_counts, neg_counts, file_summary = count_amino_acids_in_file(txt_file)

        total_positive.update(pos_counts)
        total_negative.update(neg_counts)

        total_chains += file_summary["n_chains"]
        total_binding += file_summary["n_binding_residues"]
        total_nonbinding += file_summary["n_nonbinding_residues"]
        total_unknown += file_summary["n_unknown_residues"]
        all_unknown.update(file_summary["unknown_amino_acids"])

    pos_total = sum(total_positive[aa] for aa in AA_ORDER)
    neg_total = sum(total_negative[aa] for aa in AA_ORDER)

    if pos_total == 0 or neg_total == 0:
        raise ValueError(
            f"{protocol_name}: positive or negative residue count is zero."
        )

    pos_freq = {}
    neg_freq = {}
    log2_enrichment = {}

    for aa in AA_ORDER:
        pos_freq[aa] = (
            total_positive[aa] + pseudocount
        ) / (
            pos_total + pseudocount * len(AA_ORDER)
        )

        neg_freq[aa] = (
            total_negative[aa] + pseudocount
        ) / (
            neg_total + pseudocount * len(AA_ORDER)
        )

        log2_enrichment[aa] = np.log2(
            pos_freq[aa] / neg_freq[aa]
        )

    pos_ratio = (
        total_binding / (total_binding + total_nonbinding) * 100
    )

    protocol_summary = {
        "protocol": protocol_name,
        "n_chains": total_chains,
        "n_binding_residues": total_binding,
        "n_nonbinding_residues": total_nonbinding,
        "positive_ratio_percent": pos_ratio,
        "n_unknown_residues": total_unknown,
        "unknown_amino_acids": str(dict(all_unknown))
    }

    long_rows = []

    for aa in AA_ORDER:
        long_rows.append({
            "protocol": protocol_name,
            "amino_acid": aa,
            "binding_count": total_positive[aa],
            "nonbinding_count": total_negative[aa],
            "binding_frequency": pos_freq[aa],
            "nonbinding_frequency": neg_freq[aa],
            "log2_enrichment_binding_vs_nonbinding": log2_enrichment[aa]
        })

    return protocol_summary, long_rows


# ============================================================
# 6. 绘制富集热图
# ============================================================

def plot_enrichment_heatmap(enrichment_df, output_dir):
    """
    绘制:
        rows    = benchmark protocols
        columns = 20 amino acids
        values  = log2 enrichment
    """
    matrix = enrichment_df.pivot(
        index="protocol",
        columns="amino_acid",
        values="log2_enrichment_binding_vs_nonbinding"
    )

    # 确保氨基酸列顺序固定
    matrix = matrix[AA_ORDER]

    row_names = matrix.index.tolist()
    values = matrix.values

    max_abs = float(np.max(np.abs(values)))

    # 色轴至少覆盖 [-0.5, 0.5]，保证图不至于过度放大极小差异
    vmax = max(0.5, np.ceil(max_abs * 10) / 10)
    vmin = -vmax

    fig_height = max(5.0, 0.62 * len(row_names) + 2.0)

    fig, ax = plt.subplots(figsize=(13.0, fig_height))

    im = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    ax.set_xticks(np.arange(len(AA_ORDER)))
    ax.set_xticklabels(AA_ORDER, fontsize=11)

    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=10)

    ax.set_xlabel("Amino acid", fontsize=12)
    ax.set_ylabel("Benchmark protocol", fontsize=12)

    # 在每个格子中标出数值
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]

            text_color = (
                "white" if abs(value) > 0.55 * vmax else "black"
            )

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_color
            )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.02
    )

    cbar.set_label(
        r"$\log_2$ enrichment (binding vs. non-binding)",
        fontsize=11
    )

    ax.set_title(
        "Amino-acid enrichment profiles of binding residues",
        fontsize=13,
        pad=12
    )

    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()

    png_path = output_dir / "amino_acid_enrichment_heatmap.png"
    pdf_path = output_dir / "amino_acid_enrichment_heatmap.pdf"

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Heatmap PNG saved to: {png_path}")
    print(f"Heatmap PDF saved to: {pdf_path}")


# ============================================================
# 7. 主程序
# ============================================================

def main():
    protocol_summary_rows = []
    enrichment_long_rows = []

    for item in DATASET_CONFIG:
        protocol_name = item["name"]
        file_list = item["files"]

        print("\n" + "=" * 70)
        print(f"Processing: {protocol_name}")

        protocol_summary, long_rows = summarize_protocol(
            protocol_name=protocol_name,
            file_list=file_list,
            pseudocount=PSEUDOCOUNT
        )

        protocol_summary_rows.append(protocol_summary)
        enrichment_long_rows.extend(long_rows)

        print(
            f"Chains={protocol_summary['n_chains']}, "
            f"Binding={protocol_summary['n_binding_residues']}, "
            f"Non-binding={protocol_summary['n_nonbinding_residues']}, "
            f"Pos. ratio={protocol_summary['positive_ratio_percent']:.2f}%"
        )

        if protocol_summary["n_unknown_residues"] > 0:
            print(
                f"[Warning] Unknown residues: "
                f"{protocol_summary['unknown_amino_acids']}"
            )

    # 保存 protocol 级统计
    protocol_summary_df = pd.DataFrame(protocol_summary_rows)

    summary_csv = OUTPUT_DIR / "protocol_amino_acid_summary.csv"
    protocol_summary_df.to_csv(summary_csv, index=False)

    # 保存每个氨基酸的长表统计
    enrichment_df = pd.DataFrame(enrichment_long_rows)

    enrichment_csv = OUTPUT_DIR / "amino_acid_enrichment_long.csv"
    enrichment_df.to_csv(enrichment_csv, index=False)

    print("\n" + "=" * 70)
    print(f"Protocol summary saved to: {summary_csv}")
    print(f"Amino-acid enrichment table saved to: {enrichment_csv}")

    # 绘制主热图
    plot_enrichment_heatmap(
        enrichment_df=enrichment_df,
        output_dir=OUTPUT_DIR
    )

    print("\nFinished.")


if __name__ == "__main__":
    main()