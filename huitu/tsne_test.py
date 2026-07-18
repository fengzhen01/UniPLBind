import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# 5种配色方案
COLOR_SCHEMES = {
    'scheme1': {
        'neg': '#1f77b4',  # 蓝色
        'pos': '#DA3C5D',  # 红色
        'name': 'Classic Blue-Red'
    },
    'scheme2': {
        'neg': '#2ca02c',  # 绿色
        'pos': '#d62728',  # 红色
        'name': 'Green-Red'
    },
    'scheme3': {
        'neg': '#9467bd',  # 紫色
        'pos': '#ff7f0e',  # 橙色
        'name': 'Purple-Orange'
    },
    'scheme4': {
        'neg': '#17becf',  # 青色
        'pos': '#e377c2',  # 粉色
        'name': 'Cyan-Pink'
    },
    'scheme5': {
        'neg': '#7f7f7f',  # 灰色
        'pos': '#bcbd22',  # 黄绿色
        'name': 'Gray-Lime'
    }
}


def generate_test_data(n_samples=500, n_features=50, seed=42):
    """生成随机测试数据"""
    np.random.seed(seed)

    # 生成两类数据，使其在t-SNE中有所区分
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    # 正类：围绕中心点聚集
    X_pos = np.random.randn(n_pos, n_features) * 1.0 + 2.0
    # 负类：分散在另一区域
    X_neg = np.random.randn(n_neg, n_features) * 1.0 - 2.0

    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [0] * n_neg)

    # 打乱数据
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def tsne_plot_with_colors(X, y, title, outfile, color_scheme, perplexity=30, lr=200, seed=42):
    """使用指定配色方案绘制t-SNE图"""
    n = X.shape[0]

    if n < 50:
        print(f"[t-SNE] Skip {title}: too few samples, n={n}")
        return

    if perplexity >= n:
        perplexity = max(5, min(30, n // 3))
        print(f"[t-SNE] Adjust perplexity to {perplexity} for {title}")

    # 计算t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=lr,
        init='pca',
        random_state=seed
    )

    Z = tsne.fit_transform(X)
    y = np.asarray(y, dtype=int)
    neg_mask = (y == 0)
    pos_mask = (y == 1)

    # 获取配色
    neg_color = color_scheme['neg']
    pos_color = color_scheme['pos']
    scheme_name = color_scheme['name']

    # 绘图
    plt.figure(figsize=(8, 6))

    plt.scatter(
        Z[neg_mask, 0],
        Z[neg_mask, 1],
        s=5,
        alpha=0.35,
        c=neg_color,
        label='Non-binding'
    )

    plt.scatter(
        Z[pos_mask, 0],
        Z[pos_mask, 1],
        s=8,
        alpha=0.70,
        c=pos_color,
        label='Binding'
    )

    plt.title(f'{title} [{scheme_name}]', fontsize=12)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(markerscale=3, frameon=False)
    plt.tight_layout()

    # 保存图片
    stem = os.path.splitext(outfile)[0]
    png_path = f"{stem}_{scheme_name.replace(' ', '_').lower()}.png"
    eps_path = f"{stem}_{scheme_name.replace(' ', '_').lower()}.eps"

    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.close()

    print(f"[t-SNE] saved PNG: {png_path}")
    print(f"[t-SNE] saved EPS: {eps_path}")


def test_all_color_schemes():
    """测试所有配色方案"""
    print("=" * 60)
    print("Testing 5 Color Schemes for t-SNE Visualization")
    print("=" * 60)

    # 生成测试数据
    X, y = generate_test_data(n_samples=500, n_features=50, seed=42)
    print(f"Generated test data: X.shape={X.shape}, y.shape={y.shape}")
    print(f"Positive samples: {np.sum(y == 1)}, Negative samples: {np.sum(y == 0)}")
    print("-" * 60)

    # 测试每种配色方案
    for scheme_name, color_scheme in COLOR_SCHEMES.items():
        print(f"\nTesting {scheme_name}: {color_scheme['name']}")
        print(f"  Colors: neg={color_scheme['neg']}, pos={color_scheme['pos']}")

        tsne_plot_with_colors(
            X,
            y,
            f't-SNE of Test Data',
            f'test_tsne_{scheme_name}.png',
            color_scheme,
            perplexity=30,
            lr=200,
            seed=42
        )

    print("\n" + "=" * 60)
    print("All color schemes tested successfully!")
    print("=" * 60)


def visualize_comparison():
    """在一个图中对比所有配色方案"""
    print("Generating comparison figure...")

    X, y = generate_test_data(n_samples=200, n_features=50, seed=42)

    # 计算t-SNE一次，用于所有子图
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                init='pca', random_state=42)
    Z = tsne.fit_transform(X)
    y = np.asarray(y, dtype=int)
    neg_mask = (y == 0)
    pos_mask = (y == 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (scheme_name, color_scheme) in enumerate(COLOR_SCHEMES.items()):
        if idx >= 6:
            break
        ax = axes[idx]

        neg_color = color_scheme['neg']
        pos_color = color_scheme['pos']
        scheme_name_display = color_scheme['name']

        ax.scatter(
            Z[neg_mask, 0],
            Z[neg_mask, 1],
            s=5,
            alpha=0.35,
            c=neg_color,
            label='Non-binding'
        )

        ax.scatter(
            Z[pos_mask, 0],
            Z[pos_mask, 1],
            s=8,
            alpha=0.70,
            c=pos_color,
            label='Binding'
        )

        ax.set_title(f'Scheme {idx + 1}: {scheme_name_display}', fontsize=12)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend(markerscale=3, frameon=False)
        ax.grid(True, alpha=0.3)

    # 移除多余的子图
    for idx in range(len(COLOR_SCHEMES), 6):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig('color_schemes_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Comparison figure saved as 'color_schemes_comparison.png'")


if __name__ == "__main__":
    # 测试所有配色方案
    test_all_color_schemes()

    # 生成对比图
    visualize_comparison()

    print("\n" + "=" * 60)
    print("Color schemes overview:")
    print("=" * 60)
    for i, (name, scheme) in enumerate(COLOR_SCHEMES.items(), 1):
        print(f"{i}. {scheme['name']}: Negative={scheme['neg']}, Positive={scheme['pos']}")