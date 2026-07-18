def process_rna_data(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        while True:
            # 读取四行一组的数据
            header = f_in.readline().strip()  # 第一行 (e.g., ">4csu_2")
            sequence = f_in.readline().strip()  # 第二行 (氨基酸序列)
            binary_str = f_in.readline().strip()  # 第三行 (二进制字符串)
            _ = f_in.readline()  # 第四行 (丢弃)

            if not header:  # 文件结束
                break

            # 处理标题行
            # 去掉 ">"，并合并数字部分（如 "4csu_2" → "4csu2_RNA"）
            header = header[1:]  # 去掉 ">"
            if '_' in header:
                base_id, num = header.split('_', 1)  # 分割成 "4csu" 和 "2"
                processed_header = f"{base_id}{num}_DNA"  # 合并成 "4csu2_RNA"
            else:
                processed_header = f"{header}_DNA"  # 如果没有下划线，直接加 "_RNA"

            # 写入处理后的数据
            f_out.write(f"{processed_header} {sequence}\n")
            f_out.write(f"{processed_header} {binary_str}\n")


# 使用示例
input_filename = "DNA-573_Train.txt"  # 替换为你的输入文件名
output_filename = "DNA_Train1.txt"  # 替换为你想要的输出文件名
process_rna_data(input_filename, output_filename)