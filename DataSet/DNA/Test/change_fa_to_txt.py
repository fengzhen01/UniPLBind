def convert_fa_to_txt(input_fa, output_txt):
    records = []

    with open(input_fa, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            raise ValueError(f"格式错误：第 {i+1} 行应以 '>' 开头，但实际为：{lines[i]}")

        protein_id = lines[i][1:].strip()   # 去掉 >
        sequence = lines[i + 1].strip()
        label = lines[i + 2].strip()

        if len(sequence) != len(label):
            print(
                f"警告：{protein_id} 序列长度与标签长度不一致，"
                f"sequence={len(sequence)}, label={len(label)}"
            )

        records.append((protein_id, sequence, label))
        i += 3

    with open(output_txt, "w", encoding="utf-8") as f:
        for protein_id, sequence, label in records:
            f.write(f"{protein_id}\t{sequence}\n")
            f.write(f"{protein_id}\t{label}\n")

    print(f"转换完成：共转换 {len(records)} 条蛋白序列")
    print(f"输出文件：{output_txt}")


# 示例使用
input_fa = "DPI-Test181.fa"
output_txt = "DPI-Test181.txt"

convert_fa_to_txt(input_fa, output_txt)