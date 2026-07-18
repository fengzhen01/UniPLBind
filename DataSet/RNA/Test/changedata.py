def process_rbp_data(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        while True:
            # 读取三行一组数据
            header = infile.readline().strip()
            if not header:  # 如果读到文件末尾
                break
            sequence = infile.readline().strip()
            binding = infile.readline().strip()

            # 处理header行
            if header.startswith('>'):
                protein_id = header[1:].replace('_', '')  # 去掉>和_
            else:
                protein_id = header.replace('_', '')  # 如果没有>，直接去掉_

            # 写入新的两行格式
            outfile.write(f"{protein_id}_RNA\t{sequence}\n")
            outfile.write(f"{protein_id}_RNA\t{binding}\n")


# 使用示例
input_filename = "RNA-117_Test.txt"  # 替换为你的输入文件名
output_filename = "RNA_Test1.txt"  # 输出文件名
process_rbp_data(input_filename, output_filename)