# 文件名示例: PPI-Train335.fa
input_file = "PPI-Test60.fa"
output_file = "PPI-Test60.txt"

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    lines = f_in.readlines()
    # 每三行一组
    for i in range(0, len(lines), 3):
        id_line = lines[i].strip()
        seq_line = lines[i + 1].strip()
        label_line = lines[i + 2].strip()

        # 处理 ID，只保留序列名 + chain，例如 ">2fcwA" -> "2fcw_A"
        # 假设最后一个字符是 chain
        if id_line.startswith('>'):
            seq_id = id_line[1:-1] + '_' + id_line[-1]
        else:
            seq_id = id_line

        # 写入序列行
        f_out.write(f"{seq_id}  {seq_line}\n")
        # 写入标签行
        f_out.write(f"{seq_id}  {label_line}\n")