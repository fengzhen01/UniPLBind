#!/usr/bin/env python3
"""
检查聚类后缺失的序列
从原始文件 atp-549.txt 中提取所有序列ID，与聚类结果 atp-549_30.fasta 中的ID对比，
找出缺失的序列ID
"""

import re


def parse_atp_txt(filename):
    """
    解析 atp-549.txt 文件，提取所有序列ID
    格式：每行包含 ID、序列、ID、数字串
    """
    ids = set()
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            # 每4行为一组：第一行是ID+序列，但根据你的格式，应该是每行独立
            for i, line in enumerate(lines):
                line = line.strip()
                if line and len(line.split()) >= 2:
                    # 格式：ID 序列 或者 ID 序列 ID 数字串
                    parts = line.split()
                    if parts:
                        # 第一个字段是ID
                        seq_id = parts[0]
                        ids.add(seq_id)
        return ids
    except FileNotFoundError:
        print(f"错误：找不到文件 {filename}")
        return set()


def parse_fasta_ids(filename):
    """
    解析 FASTA 文件，提取所有序列ID（>开头的行）
    """
    ids = set()
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    # 提取ID（去掉>号，并去除可能的空格和描述）
                    seq_id = line[1:].strip().split()[0]
                    ids.add(seq_id)
        return ids
    except FileNotFoundError:
        print(f"错误：找不到文件 {filename}")
        return set()


def main():
    # 文件名
    original_file = "ATP549.txt"
    clustered_file = "ATP549_30.fasta"

    print("=" * 60)
    print("序列缺失检查程序")
    print("=" * 60)

    # 解析两个文件
    print(f"\n1. 正在读取原始文件: {original_file}")
    original_ids = parse_atp_txt(original_file)
    print(f"   原始文件中的序列总数: {len(original_ids)}")
    print(f"   前10个ID示例: {list(original_ids)[:10]}")

    print(f"\n2. 正在读取聚类结果文件: {clustered_file}")
    clustered_ids = parse_fasta_ids(clustered_file)
    print(f"   聚类结果中的序列总数: {len(clustered_ids)}")
    print(f"   前10个ID示例: {list(clustered_ids)[:10]}")

    # 找出缺失的序列
    missing_ids = original_ids - clustered_ids

    # 输出结果
    print("\n" + "=" * 60)
    print("3. 检查结果")
    print("=" * 60)

    if missing_ids:
        print(f"\n❌ 发现 {len(missing_ids)} 条序列在聚类后缺失：")
        print("-" * 60)

        # 排序后输出
        for seq_id in sorted(missing_ids):
            print(f"   {seq_id}")

        # 保存到文件
        output_file = "missing_sequences.txt"
        with open(output_file, 'w') as f:
            f.write(f"缺失的序列ID列表（共{len(missing_ids)}条）:\n")
            f.write("=" * 50 + "\n")
            for seq_id in sorted(missing_ids):
                f.write(f"{seq_id}\n")
        print(f"\n✅ 缺失序列列表已保存到: {output_file}")

        # 显示统计信息
        print("\n统计信息：")
        print(f"   原始序列总数: {len(original_ids)}")
        print(f"   聚类后序列数: {len(clustered_ids)}")
        print(f"   缺失序列数量: {len(missing_ids)}")
        print(f"   保留比例: {(len(clustered_ids) / len(original_ids) * 100):.2f}%")

    else:
        print("\n✅ 所有序列都存在于聚类结果中，没有缺失！")
        print(f"   原始序列总数: {len(original_ids)}")
        print(f"   聚类后序列数: {len(clustered_ids)}")

    # 可选：显示聚类结果中多出来的序列（正常情况下不应该有）
    extra_ids = clustered_ids - original_ids
    if extra_ids:
        print(f"\n⚠️  警告：聚类结果中有 {len(extra_ids)} 条序列不在原始文件中：")
        for seq_id in sorted(extra_ids)[:10]:  # 只显示前10个
            print(f"   {seq_id}")
        if len(extra_ids) > 10:
            print(f"   ... 等共{len(extra_ids)}条")


if __name__ == "__main__":
    main()