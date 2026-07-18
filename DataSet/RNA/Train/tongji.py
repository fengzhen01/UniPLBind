from pathlib import Path


def count_labels_in_dataset(txt_file):
    num_0 = 0
    num_1 = 0
    num_proteins = 0
    total_residues = 0

    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            content = parts[1].strip()

            # 只统计标签行：该行内容只由 0 和 1 组成
            if set(content).issubset({'0', '1'}):
                num_0 += content.count('0')
                num_1 += content.count('1')
                total_residues += len(content)
                num_proteins += 1

    ratio_pn = num_1 / num_0 if num_0 > 0 else 0
    pos_ratio = num_1 / total_residues * 100 if total_residues > 0 else 0

    output_text = (
        f"Number of proteins: {num_proteins}\n"
        f"Number of binding residues, numP: {num_1}\n"
        f"Number of non-binding residues, numN: {num_0}\n"
        f"Total residues: {total_residues}\n"
        f"Ratio = numP / numN: {ratio_pn:.6f}\n"
        f"Positive ratio = numP / total residues: {pos_ratio:.2f}%\n"
    )

    print(output_text)

    # 保存到当前程序文件所在文件夹下的 jieguo.txt
    (Path(__file__).resolve().parent / 'jieguo1.txt').write_text(output_text, encoding='utf-8')

    return {
        "num_proteins": num_proteins,
        "numP": num_1,
        "numN": num_0,
        "total_residues": total_residues,
        "ratio_numP_numN": ratio_pn,
        "positive_ratio_percent": pos_ratio
    }


result = count_labels_in_dataset(
    txt_file=r"RNA_Train.txt"
)