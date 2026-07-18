def reformat_rna_data(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if line.strip():  # 跳过空行
                parts = line.split(maxsplit=1)  # 分割 ID 和序列/二进制字符串
                if len(parts) == 2:
                    old_id = parts[0]
                    # 在最后一个字母和 "RNA" 之间插入下划线
                    if old_id.endswith("_RNA"):
                        base = old_id[:-4]  # 去掉 "_RNA"
                        if len(base) > 0:
                            # 在最后一个字母前插入下划线（如 "5kl1B" → "5kl1_B"）
                            new_base = base[:-1] + "_" + base[-1]
                            new_id = new_base + "_RNA"
                        else:
                            new_id = old_id  # 如果格式异常，保持原样
                    else:
                        new_id = old_id  # 如果没有 "_RNA"，保持原样
                    # 写入新格式的行
                    f_out.write(f"{new_id}\t{parts[1]}")
                else:
                    f_out.write(line)  # 如果格式不对，保持原样

# 使用示例
input_filename = "RNA_Train.txt"  # 替换为你的输入文件名
output_filename = "RNA_Train.txt"  # 替换为输出文件名
reformat_rna_data(input_filename, output_filename)