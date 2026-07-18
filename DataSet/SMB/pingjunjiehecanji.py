import numpy as np
import math


def calculate_binding_site_statistics(filename):
    """
    计算蛋白质结合位点残基的统计信息

    参数:
        filename: 数据文件名

    返回:
        包含统计信息的字典
    """
    binding_site_counts = []  # 存储每条链的结合位点数量

    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

            # 每三行为一组数据
            for i in range(0, len(lines), 3):
                if i + 2 >= len(lines):
                    break  # 确保有完整的三行

                # 第一行: 蛋白质标识符 (例如: >1d2a_B)
                # 第二行: 氨基酸序列 (我们不需要这个来计算统计)
                # 第三行: 结合位点标注 (0/1字符串)
                binding_sites_line = lines[i + 2].strip()

                # 计算该链中结合位点的数量 (1的数量)
                count = binding_sites_line.count('1')
                binding_site_counts.append(count)

        # 将列表转换为numpy数组以便计算统计量
        counts_array = np.array(binding_site_counts)

        # 计算基本统计量
        total_chains = len(binding_site_counts)
        mean_count = np.mean(counts_array)
        std_dev = np.std(counts_array, ddof=1)  # 样本标准差

        # 计算正负标准差（mean ± std）
        positive_std = mean_count + std_dev
        negative_std = mean_count - std_dev

        # 输出详细统计信息
        print(f"数据统计结果:")
        print(f"总蛋白质链数: {total_chains}")
        print(f"平均结合位点残基数: {mean_count:.2f}")
        print(f"标准差: {std_dev:.2f}")
        print(f"平均值+标准差: {positive_std:.2f}")
        print(f"平均值-标准差: {negative_std:.2f}")
        print(f"最大值: {np.max(counts_array)}")
        print(f"最小值: {np.min(counts_array)}")
        print(f"中位数: {np.median(counts_array):.2f}")

        # 返回统计结果
        return {
            'total_chains': total_chains,
            'mean': mean_count,
            'std_dev': std_dev,
            'mean_plus_std': positive_std,
            'mean_minus_std': negative_std,
            'max': np.max(counts_array),
            'min': np.min(counts_array),
            'median': np.median(counts_array),
            'all_counts': binding_site_counts
        }

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{filename}'")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


# 使用示例
if __name__ == "__main__":
    # 假设你的数据文件名为 'train.txt'
    filename = 'SMB_Test.txt'

    # 计算统计信息
    results = calculate_binding_site_statistics(filename)

    if results:
        print("\n详细统计:")
        print(f"结合位点数量列表: {results['all_counts']}")
        print(f"平均结合位点残基数: {results['mean']:.2f} ± {results['std_dev']:.2f}")


