import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shap
from torch.utils.data import Dataset


# ============================================================
# 0. 基本配置：这里只需要按你的实际路径修改
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 两个 PLM 的名称与维度 ----------
PLM1_NAME = "ProstT5"
PLM1_DIM = 1024

PLM2_NAME = "Ankh"
PLM2_DIM = 1536

INPUT_DIM = PLM1_DIM + PLM2_DIM   # 2560


# ---------- 测试集文件 ----------
# 文件格式应为每个蛋白两行：
# protein_id   sequence
# protein_id   0/1 label sequence
# TEST_LIST_FILE = r"E:/fengzhen/NucGMTL-main/DataSet/DNA/Test/DNA_Test.txt"
TEST_LIST_FILE = r"E:/fengzhen/NucGMTL-main/DataSet/PPI/Train352-Test70/PPI-Test70.txt"


# ---------- ProstT5 与 Ankh embedding 文件夹 ----------
# 每个蛋白应对应：
# <embedding_dir>/<protein_id>.data
#
# 请按你的实际文件夹修改。
# PLM1_EMBEDDING_DIR = r"E:/fengzhen/embedding_DNA/ProstT5_embedding_573+129+181/"
# PLM2_EMBEDDING_DIR = r"E:/fengzhen/embedding_DNA/Ankh_embedding_573+129/"
PLM1_EMBEDDING_DIR = r"E:/fengzhen/embedding_PPI/ProstT5_embedding_352+70/"
PLM2_EMBEDDING_DIR = r"E:/fengzhen/embedding_PPI/Ankh_embedding_352+70/"


# ---------- 五个重复运行得到的 checkpoint ----------
CHECKPOINT_PATHS = [
    # r"E:/fengzhen/NucGMTL-main/RGMTL/Result_ATP_0_2026-06-22 10_49_25.912853.pkl",
    # r"E:/fengzhen/NucGMTL-main/RGMTL/Result_ATP_1_2026-06-22 10_49_25.912853.pkl",
    # r"E:/fengzhen/NucGMTL-main/RGMTL/Result_ATP_2_2026-06-22 10_49_25.912853.pkl",
    # r"E:/fengzhen/NucGMTL-main/RGMTL/Result_ATP_3_2026-06-22 10_49_25.912853.pkl",
    # r"E:/fengzhen/NucGMTL-main/RGMTL/Result_DNA_4_2026-06-22 18_39_11.891406.pkl",
    # r"E:/fengzhen/NucGMTL-main/RGMTL/Result_PPI_2_2026-06-23 08_57_02.635648.pkl",
    # r"E:/fengzhen/NucGMTL-main/RGMTL/Result_PPI_4_2026-06-27 18_54_34.365366.pkl",
    r"E:/fengzhen/NucGMTL-main/RGMTL/Result_DNA_1_2026-06-30 08_04_16.882886.pkl",
    # r"E:/fengzhen/NucGMTL-main/RGMTL/Result_ATP_3_2026-06-22 10_49_25.912853.pkl",
]


# ---------- 总输出文件夹 ----------
OUTPUT_ROOT = Path(
    r"E:/fengzhen/NucGMTL-main/RGMTL/shap_PPI_five_runs"
)


# ---------- SHAP 参数 ----------
RANDOM_SEED = 42

# background protein 数量
N_BACKGROUND = 10

# 希望解释的蛋白数量。
# 若测试集只有 41 个蛋白，实际会变为 31 个 explained proteins。
N_EXPLAIN = 100

# 所有蛋白统一截取的最大长度
LCAP = 128

# 先测试时可设置为 50；最终论文图建议使用 200 或更高
NSAMPLES = 200

# dumbbell 图中展示多少个代表性蛋白
N_DISPLAY = 20


# ============================================================
# 1. 读取测试集中的蛋白名称
# ============================================================

def read_protein_names(two_line_file):
    """
    读取每两行一个蛋白的数据文件中的 protein ID。

    文件形式：
    protein_A  sequence_A
    protein_A  labels_A
    protein_B  sequence_B
    protein_B  labels_B
    ...
    """
    with open(two_line_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) % 2 != 0:
        raise ValueError(
            f"Expected an even number of non-empty lines in {two_line_file}, "
            f"but got {len(lines)}."
        )

    protein_names = []

    for i in range(0, len(lines), 2):
        parts = lines[i].split()

        if len(parts) < 1:
            raise ValueError(f"Cannot parse protein ID from line: {lines[i]}")

        protein_names.append(parts[0])

    return protein_names


# ============================================================
# 2. 仅用于 SHAP 的 embedding 数据集
#    不需要读取 label，也不需要 VNet。
# ============================================================

class EmbeddingOnlyDataset(Dataset):
    """
    读取 ProstT5 + Ankh embedding，并在特征维度拼接。

    每个蛋白读取：
        ProstT5: [L1, 1024]
        Ankh:    [L2, 1536]

    最终返回：
        [min(L1, L2), 2560]
    """

    def __init__(
        self,
        protein_names,
        plm1_embedding_dir,
        plm2_embedding_dir,
        plm1_dim,
        plm2_dim
    ):
        self.protein_names = protein_names
        self.plm1_embedding_dir = Path(plm1_embedding_dir)
        self.plm2_embedding_dir = Path(plm2_embedding_dir)
        self.plm1_dim = plm1_dim
        self.plm2_dim = plm2_dim

    def __len__(self):
        return len(self.protein_names)

    def __getitem__(self, index):
        protein_name = self.protein_names[index]

        plm1_file = self.plm1_embedding_dir / f"{protein_name}.data"
        plm2_file = self.plm2_embedding_dir / f"{protein_name}.data"

        if not plm1_file.exists():
            raise FileNotFoundError(f"PLM-1 embedding not found:\n{plm1_file}")

        if not plm2_file.exists():
            raise FileNotFoundError(f"PLM-2 embedding not found:\n{plm2_file}")

        plm1 = pd.read_csv(plm1_file, header=None).values.astype(np.float32)
        plm2 = pd.read_csv(plm2_file, header=None).values.astype(np.float32)

        if plm1.ndim != 2 or plm1.shape[1] != self.plm1_dim:
            raise ValueError(
                f"{protein_name}: expected PLM-1 shape [L, {self.plm1_dim}], "
                f"but got {plm1.shape}."
            )

        if plm2.ndim != 2 or plm2.shape[1] != self.plm2_dim:
            raise ValueError(
                f"{protein_name}: expected PLM-2 shape [L, {self.plm2_dim}], "
                f"but got {plm2.shape}."
            )

        min_len = min(plm1.shape[0], plm2.shape[0])

        plm1 = plm1[:min_len]
        plm2 = plm2[:min_len]

        features = np.concatenate([plm1, plm2], axis=1)

        return torch.from_numpy(features).float(), protein_name


# ============================================================
# 3. 模型结构定义
#
# 注意：
# 这里的 AttentionModel、FeatureExtractor 和 Module
# 必须与你训练并保存 .pkl 时的结构完全一致。
# ============================================================

class AttentionModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)

        # 使用 float 标量，避免 CPU/GPU device mismatch
        self.norm_factor = out_dim ** (-0.5)

    def forward(self, plms1, seq_lengths):
        """
        plms1: [B, L, C]
        seq_lengths: [B]
        """
        q = self.q(plms1)
        k = self.k(plms1)
        v = self.v(plms1)

        attention_score = torch.bmm(q, k.permute(0, 2, 1))
        attention_score = attention_score * self.norm_factor

        attention = self.masked_softmax(attention_score, seq_lengths)

        output = torch.bmm(attention, v)

        return output + v

    @staticmethod
    def create_src_lengths_mask(batch_size, src_lengths):
        max_src_len = int(src_lengths.max().item())

        src_indices = torch.arange(
            max_src_len,
            device=src_lengths.device
        ).unsqueeze(0)

        src_indices = src_indices.expand(batch_size, max_src_len)

        src_lengths_expand = src_lengths.unsqueeze(1).expand(
            batch_size,
            max_src_len
        )

        return (src_indices < src_lengths_expand).int()

    def masked_softmax(self, scores, src_lengths):
        """
        scores: [B, L, L]
        """
        batch_size, seq_len, _ = scores.size()

        mask = self.create_src_lengths_mask(
            batch_size,
            src_lengths.to(scores.device)
        )

        mask = mask.unsqueeze(2)  # [B, L, 1]

        scores = scores.permute(0, 2, 1)
        scores = scores.masked_fill(mask == 0, -np.inf)
        scores = scores.permute(0, 2, 1)

        return F.softmax(scores.float(), dim=-1)


class FeatureExtractor(nn.Module):
    def __init__(self, inputdim):
        super().__init__()

        self.inputdim = inputdim

        # Multi-scale CNN branch: kernel size = 1
        self.ms1cnn1 = nn.Conv1d(inputdim, 512, 1, padding="same")
        self.ms1cnn2 = nn.Conv1d(512, 256, 1, padding="same")
        self.ms1cnn3 = nn.Conv1d(256, 128, 1, padding="same")

        # Multi-scale CNN branch: kernel size = 3
        self.ms2cnn1 = nn.Conv1d(inputdim, 512, 3, padding="same")
        self.ms2cnn2 = nn.Conv1d(512, 256, 3, padding="same")
        self.ms2cnn3 = nn.Conv1d(256, 128, 3, padding="same")

        # Multi-scale CNN branch: kernel size = 5
        self.ms3cnn1 = nn.Conv1d(inputdim, 512, 5, padding="same")
        self.ms3cnn2 = nn.Conv1d(512, 256, 5, padding="same")
        self.ms3cnn3 = nn.Conv1d(256, 128, 5, padding="same")

        self.relu = nn.ReLU(True)

        self.AttentionModel1 = AttentionModel(512, 128)
        self.AttentionModel2 = AttentionModel(256, 128)
        self.AttentionModel3 = AttentionModel(128, 128)

    def forward(self, prot_input, seq_lengths):
        """
        prot_input: [B, L, C]
        """
        prot_input_share = prot_input.permute(0, 2, 1)  # [B, C, L]

        # First CNN stage
        m1 = self.relu(self.ms1cnn1(prot_input_share))
        m2 = self.relu(self.ms2cnn1(prot_input_share))
        m3 = self.relu(self.ms3cnn1(prot_input_share))

        att = (m1 + m2 + m3).permute(0, 2, 1)
        s1 = self.AttentionModel1(att, seq_lengths)

        # Second CNN stage
        m1 = self.relu(self.ms1cnn2(m1))
        m2 = self.relu(self.ms2cnn2(m2))
        m3 = self.relu(self.ms3cnn2(m3))

        att = (m1 + m2 + m3).permute(0, 2, 1)
        s2 = self.AttentionModel2(att, seq_lengths)

        # Third CNN stage
        m1 = self.relu(self.ms1cnn3(m1))
        m2 = self.relu(self.ms2cnn3(m2))
        m3 = self.relu(self.ms3cnn3(m3))

        att = (m1 + m2 + m3).permute(0, 2, 1)
        s3 = self.AttentionModel3(att, seq_lengths)

        mscnn = (m1 + m2 + m3).permute(0, 2, 1)
        attention_feature = s1 + s2 + s3

        return mscnn + attention_feature


class Module(nn.Module):
    """
    当前单任务结合位点预测模型。

    checkpoint 参数名应类似：
        feature_extractor.ms1cnn1.weight
        ...
        task_fc.0.weight
        ...
    """

    def __init__(self, inputdim, istrain=False):
        super().__init__()

        self.istrain = istrain
        self.inputdim = inputdim

        self.feature_extractor = FeatureExtractor(inputdim)

        self.task_fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, prot_input, data_lengths):
        features = self.feature_extractor(prot_input, data_lengths)
        output = self.task_fc(features)  # [B, L, 1]
        return output.squeeze(-1)  # [B, L]


# ============================================================
# 4. SHAP 前向包装器
#
# 将残基级预测 [B, L, 2] 汇总为蛋白级输出 [B, 1]。
# ============================================================
class MeanPositiveLogitForSHAP(nn.Module):
    """
    将每条蛋白中所有残基的 binding logit 取平均，
    作为蛋白级 SHAP 输出。

    输入:
        [B, L, C]

    输出:
        [B, 1]
    """

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x_in):
        x_in = x_in.to(DEVICE)

        fixed_lengths = torch.full(
            (x_in.size(0),),
            x_in.size(1),
            dtype=torch.long,
            device="cpu"
        )

        logits = self.model(x_in, fixed_lengths)  # [B, L]

        # 单输出模型中，每个 logit 对应 binding class
        return logits.mean(dim=1, keepdim=True)   # [B, 1]


# ============================================================
# 5. 检查 checkpoint 是否与当前 Module 结构匹配
# ============================================================

def inspect_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state_dict", checkpoint)
        )
    else:
        state_dict = checkpoint

    keys = list(state_dict.keys())

    print("\n" + "=" * 70)
    print(f"Checkpoint inspection:\n{checkpoint_path}")

    print("\nFirst 10 parameter names:")
    for key in keys[:10]:
        print("   ", key)

    if any(key.startswith("feature_extractor.") for key in keys):
        print("\n[OK] Current single-task architecture detected.")
        print("     Prefixes: feature_extractor.* and task_fc.*")

    elif any(key.startswith("ShardEncoder.") for key in keys):
        print("\n[Warning] Old multi-task architecture detected.")
        print("          Prefixes: ShardEncoder.* and tasks_fcs.*")
        print("          This checkpoint cannot be loaded by the current Module.")

    else:
        print("\n[Warning] Checkpoint architecture could not be identified.")

    print("=" * 70 + "\n")


# ============================================================
# 6. 加载模型 checkpoint
# ============================================================

def load_model_for_shap(model_path, input_dim):
    model = Module(input_dim, istrain=False).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state_dict", checkpoint)
        )
    else:
        state_dict = checkpoint

    # 保持 strict=True，防止错误 checkpoint 被悄悄载入
    model.load_state_dict(state_dict, strict=True)

    model.eval()

    return model


# ============================================================
# 7. 固定五次 SHAP 使用同一批蛋白
# ============================================================

def prepare_fixed_shap_inputs(
    dataset,
    n_background,
    n_explain,
    lcap,
    random_seed
):
    """
    一次性固定：
        background proteins
        explained proteins
        Lcap_eff

    所有五个 checkpoint 共用该批数据。
    """
    if len(dataset) < 2:
        raise ValueError("The test dataset contains fewer than two proteins.")

    rng = np.random.default_rng(random_seed)

    pick_n = min(len(dataset), n_background + n_explain)

    selected_indices = rng.choice(
        len(dataset),
        size=pick_n,
        replace=False
    )

    selected_lengths = []

    for idx in selected_indices:
        x_i, _ = dataset[idx]
        selected_lengths.append(x_i.shape[0])

    lcap_eff = min(lcap, min(selected_lengths))

    feature_list = []
    selected_names = []

    for idx in selected_indices:
        x_i, protein_name = dataset[idx]

        feature_list.append(x_i[:lcap_eff].float())
        selected_names.append(protein_name)

    x_all = torch.stack(feature_list, dim=0)  # [B, Lcap_eff, C]

    bg_n = min(n_background, pick_n - 1)
    ex_n = pick_n - bg_n

    if ex_n <= 0:
        raise ValueError("No proteins remain for SHAP explanation.")

    bg = x_all[:bg_n].clone()
    ex = x_all[bg_n:bg_n + ex_n].clone()

    background_names = selected_names[:bg_n]
    explained_names = selected_names[bg_n:bg_n + ex_n]

    manifest_df = pd.DataFrame({
        "protein": selected_names,
        "dataset_index": selected_indices,
        "role": ["background"] * bg_n + ["explained"] * ex_n,
        "cropped_length": [lcap_eff] * pick_n
    })

    return bg, ex, background_names, explained_names, lcap_eff, manifest_df


# ============================================================
# 8. 绘图函数
# ============================================================

def save_histogram(
    plm1_ratio,
    plm1_name,
    out_path,
    n_explained
):
    """
    自适应范围的 histogram。
    若分布很集中，也不会只出现一个宽柱。
    """
    mean_ratio = float(np.mean(plm1_ratio))
    median_ratio = float(np.median(plm1_ratio))

    data_min = float(np.min(plm1_ratio))
    data_max = float(np.max(plm1_ratio))

    # 若范围过窄，给横轴两端增加小范围边界
    lower = max(0.0, data_min - 0.6)
    upper = min(100.0, data_max + 0.6)

    if upper - lower < 1.0:
        lower = max(0.0, mean_ratio - 0.8)
        upper = min(100.0, mean_ratio + 0.8)

    n_bins = min(10, max(5, int(np.sqrt(len(plm1_ratio)) + 2)))
    bins = np.linspace(lower, upper, n_bins + 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    ax.hist(
        plm1_ratio,
        bins=bins,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85
    )

    ax.axvline(
        50,
        color="gray",
        linestyle=":",
        linewidth=1.4,
        label="Equal attribution (50%)"
    )

    ax.axvline(
        mean_ratio,
        color="#D62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean = {mean_ratio:.1f}%"
    )

    ax.axvline(
        median_ratio,
        color="black",
        linestyle="-",
        linewidth=1.4,
        label=f"Median = {median_ratio:.1f}%"
    )

    ax.set_xlabel(
        f"Relative {plm1_name} attribution (% of total |SHAP|)"
    )

    ax.set_ylabel("Number of proteins")

    ax.set_title(
        f"Distribution of {plm1_name} attribution\n"
        f"across explained proteins (n = {n_explained})"
    )

    ax.legend(frameon=False, fontsize=8.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_boxstrip(
    plm1_ratio,
    plm1_name,
    out_path,
    random_seed
):
    """
    对于分布高度集中的结果，
    boxplot + jittered individual points 比 histogram 更适合。
    """
    mean_ratio = float(np.mean(plm1_ratio))
    median_ratio = float(np.median(plm1_ratio))

    std_ratio = (
        float(np.std(plm1_ratio, ddof=1))
        if len(plm1_ratio) > 1
        else 0.0
    )

    rng = np.random.default_rng(random_seed)

    fig, ax = plt.subplots(figsize=(4.8, 5.0))

    bp = ax.boxplot(
        [plm1_ratio],
        positions=[1],
        widths=0.42,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops=dict(color="black", linewidth=1.8),
        meanprops=dict(
            color="#D62728",
            linewidth=1.8,
            linestyle="--"
        ),
        boxprops=dict(
            facecolor="#4C78A8",
            alpha=0.45
        ),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0)
    )

    jitter_x = 1 + rng.uniform(
        low=-0.08,
        high=0.08,
        size=len(plm1_ratio)
    )

    ax.scatter(
        jitter_x,
        plm1_ratio,
        s=34,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.35,
        alpha=0.85,
        zorder=3
    )

    ax.axhline(
        50,
        color="gray",
        linestyle=":",
        linewidth=1.4,
        label="Equal attribution (50%)"
    )

    ax.axhline(
        mean_ratio,
        color="#D62728",
        linestyle="--",
        linewidth=1.6,
        label=f"Mean = {mean_ratio:.1f}%"
    )

    ax.axhline(
        median_ratio,
        color="black",
        linestyle="-",
        linewidth=1.3,
        label=f"Median = {median_ratio:.1f}%"
    )

    ax.set_xlim(0.65, 1.35)
    ax.set_ylim(0, 100)

    ax.set_xticks([1])
    ax.set_xticklabels([plm1_name])

    ax.set_ylabel("Relative attribution (% of total |SHAP|)")

    ax.set_title(
        f"{plm1_name} attribution across explained proteins\n"
        f"SD = {std_ratio:.2f}%"
    )

    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_dumbbell_plot(
    summary_df,
    plm1_name,
    plm2_name,
    out_path,
    n_display
):
    """
    单蛋白层面的双 PLM relative attribution dumbbell plot。
    """
    rank_df = summary_df.sort_values(
        by=f"{plm1_name}_ratio_percent",
        ascending=True
    ).reset_index(drop=True)

    n_show = min(n_display, len(rank_df))

    selected_row_idx = np.linspace(
        0,
        len(rank_df) - 1,
        n_show,
        dtype=int
    )

    plot_df = rank_df.iloc[selected_row_idx].copy().reset_index(drop=True)

    y_pos = np.arange(len(plot_df))

    fig_height = max(4.5, 0.34 * len(plot_df) + 1.5)

    fig, ax = plt.subplots(figsize=(7.0, fig_height))

    for i in range(len(plot_df)):
        x1 = plot_df.loc[i, f"{plm1_name}_ratio_percent"]
        x2 = plot_df.loc[i, f"{plm2_name}_ratio_percent"]

        ax.plot(
            [x1, x2],
            [y_pos[i], y_pos[i]],
            color="lightgray",
            linewidth=1.25,
            zorder=1
        )

    ax.scatter(
        plot_df[f"{plm1_name}_ratio_percent"],
        y_pos,
        s=48,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.35,
        label=plm1_name,
        zorder=3
    )

    ax.scatter(
        plot_df[f"{plm2_name}_ratio_percent"],
        y_pos,
        s=48,
        color="#F58518",
        edgecolor="black",
        linewidth=0.35,
        label=plm2_name,
        zorder=3
    )

    ax.axvline(
        50,
        color="gray",
        linestyle=":",
        linewidth=1.3
    )

    ax.set_xlim(0, 100)

    ax.set_xlabel("Relative attribution (% of total |SHAP|)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["protein"].tolist(), fontsize=8.5)

    ax.set_title(
        f"Protein-level relative attribution of {plm1_name} and {plm2_name}"
    )

    ax.legend(frameon=False, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 9. 对单个 checkpoint 计算 SHAP
# ============================================================

def analyze_one_checkpoint(
    checkpoint_path,
    run_id,
    out_dir,
    bg_cpu,
    ex_cpu,
    explained_names,
    lcap_eff,
    plm1_name,
    plm1_dim,
    plm2_name,
    plm2_dim,
    nsamples,
    random_seed,
    n_display
):
    """
    对一个 checkpoint 运行 SHAP，并保存该 run 的图和 CSV。
    """
    checkpoint_path = Path(checkpoint_path)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    input_dim = plm1_dim + plm2_dim

    print("\n" + "=" * 80)
    print(f"Running SHAP for run {run_id}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print("=" * 80)

    model = load_model_for_shap(
        model_path=str(checkpoint_path),
        input_dim=input_dim
    )

    fwrap = MeanPositiveLogitForSHAP(model)

    # 每次 run 固定随机种子，使 GradientExplainer 的采样更可比较
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    bg = bg_cpu.to(DEVICE)
    ex = ex_cpu.to(DEVICE)

    print(f"Background proteins: {bg.size(0)}")
    print(f"Explained proteins: {ex.size(0)}")
    print(f"Effective cropped length: {lcap_eff}")
    print(f"Input dimension: {input_dim}")
    print(f"{plm1_name}: {plm1_dim} dimensions")
    print(f"{plm2_name}: {plm2_dim} dimensions")

    # GradientExplainer 适用于 PyTorch 深度模型
    explainer = shap.GradientExplainer(fwrap, bg)

    shap_values = explainer.shap_values(
        ex,
        nsamples=nsamples
    )

    # 不同 SHAP 版本的返回形式可能略有不同
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values, dtype=np.float32)

    # 有的版本会返回 [B, L, C, 1]
    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    expected_shape = tuple(ex.shape)

    if tuple(shap_values.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected SHAP shape: {shap_values.shape}; "
            f"expected: {expected_shape}"
        )

    # --------------------------------------------------------
    # 计算双 PLM 的总绝对 SHAP attribution
    # --------------------------------------------------------
    abs_shap = np.abs(shap_values)

    plm1_abs = abs_shap[:, :, :plm1_dim].sum(axis=(1, 2))
    plm2_abs = abs_shap[:, :, plm1_dim:].sum(axis=(1, 2))

    total_abs = plm1_abs + plm2_abs + 1e-12

    plm1_ratio = 100.0 * plm1_abs / total_abs
    plm2_ratio = 100.0 * plm2_abs / total_abs

    summary_df = pd.DataFrame({
        "protein": explained_names,
        f"{plm1_name}_abs_shap": plm1_abs,
        f"{plm2_name}_abs_shap": plm2_abs,
        f"{plm1_name}_ratio_percent": plm1_ratio,
        f"{plm2_name}_ratio_percent": plm2_ratio
    })

    summary_df.insert(0, "checkpoint", checkpoint_path.name)
    summary_df.insert(1, "run_id", run_id)

    # --------------------------------------------------------
    # 保存 CSV
    # --------------------------------------------------------
    csv_path = out_dir / "plm_attribution_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    # --------------------------------------------------------
    # 保存图 A：Histogram
    # --------------------------------------------------------
    save_histogram(
        plm1_ratio=plm1_ratio,
        plm1_name=plm1_name,
        out_path=out_dir / "shap_plm1_ratio_histogram.png",
        n_explained=len(summary_df)
    )

    # --------------------------------------------------------
    # 保存图 A 的替代表达：Boxplot + individual points
    # --------------------------------------------------------
    save_boxstrip(
        plm1_ratio=plm1_ratio,
        plm1_name=plm1_name,
        out_path=out_dir / "shap_plm1_ratio_boxstrip.png",
        random_seed=random_seed
    )

    # --------------------------------------------------------
    # 保存图 B：Dumbbell plot
    # --------------------------------------------------------
    save_dumbbell_plot(
        summary_df=summary_df,
        plm1_name=plm1_name,
        plm2_name=plm2_name,
        out_path=out_dir / "shap_plm_attribution_dumbbell.png",
        n_display=n_display
    )

    # --------------------------------------------------------
    # 当前 run 的统计摘要
    # --------------------------------------------------------
    result = {
        "run_id": run_id,
        "checkpoint": checkpoint_path.name,
        "n_explained_proteins": len(summary_df),

        f"{plm1_name}_mean_ratio_percent": float(np.mean(plm1_ratio)),
        f"{plm1_name}_median_ratio_percent": float(np.median(plm1_ratio)),
        f"{plm1_name}_std_ratio_percent": float(
            np.std(plm1_ratio, ddof=1)
        ) if len(plm1_ratio) > 1 else 0.0,
        f"{plm1_name}_min_ratio_percent": float(np.min(plm1_ratio)),
        f"{plm1_name}_max_ratio_percent": float(np.max(plm1_ratio)),

        f"{plm2_name}_mean_ratio_percent": float(np.mean(plm2_ratio)),
        f"{plm2_name}_median_ratio_percent": float(np.median(plm2_ratio)),
        f"{plm2_name}_std_ratio_percent": float(
            np.std(plm2_ratio, ddof=1)
        ) if len(plm2_ratio) > 1 else 0.0,
        f"{plm2_name}_min_ratio_percent": float(np.min(plm2_ratio)),
        f"{plm2_name}_max_ratio_percent": float(np.max(plm2_ratio))
    }

    print(
        f"[Run {run_id}] {plm1_name}: "
        f"{result[f'{plm1_name}_mean_ratio_percent']:.2f}% "
        f"± {result[f'{plm1_name}_std_ratio_percent']:.2f}%"
    )

    print(
        f"[Run {run_id}] {plm2_name}: "
        f"{result[f'{plm2_name}_mean_ratio_percent']:.2f}% "
        f"± {result[f'{plm2_name}_std_ratio_percent']:.2f}%"
    )

    print(f"Saved: {csv_path}")

    # 释放显存
    del model
    del fwrap
    del explainer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary_df, result


# ============================================================
# 10. 五次运行总体归因图
# ============================================================

def save_five_runs_mean_plot(
    run_summary_df,
    plm1_name,
    plm2_name,
    out_path
):
    """
    展示五个 checkpoint 的平均归因比例。
    """
    x = np.arange(len(run_summary_df))

    plm1_means = run_summary_df[
        f"{plm1_name}_mean_ratio_percent"
    ].values

    plm2_means = run_summary_df[
        f"{plm2_name}_mean_ratio_percent"
    ].values

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    ax.plot(
        x,
        plm1_means,
        marker="o",
        linewidth=1.8,
        markersize=6,
        color="#4C78A8",
        label=plm1_name
    )

    ax.plot(
        x,
        plm2_means,
        marker="o",
        linewidth=1.8,
        markersize=6,
        color="#F58518",
        label=plm2_name
    )

    ax.axhline(
        50,
        color="gray",
        linestyle=":",
        linewidth=1.3,
        label="Equal attribution (50%)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Run {int(i)}" for i in run_summary_df["run_id"].values]
    )

    ax.set_ylim(0, 100)

    ax.set_xlabel("Independently trained checkpoint")
    ax.set_ylabel("Mean relative attribution (% of total |SHAP|)")

    ax.set_title(
        f"Stability of PLM attribution across repeated training runs"
    )

    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 11. 主程序：依次分析 5 个 checkpoint
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Output folder: {OUTPUT_ROOT}")

    # --------------------------------------------------------
    # 1. 检查 checkpoint
    # --------------------------------------------------------
    valid_checkpoints = []

    for checkpoint_path in CHECKPOINT_PATHS:
        if not Path(checkpoint_path).exists():
            print(f"[Skip] Checkpoint not found:\n{checkpoint_path}")
            continue

        inspect_checkpoint(checkpoint_path)
        valid_checkpoints.append(checkpoint_path)

    if len(valid_checkpoints) == 0:
        raise RuntimeError(
            "No valid checkpoint was found. "
            "Please check CHECKPOINT_PATHS."
        )

    # --------------------------------------------------------
    # 2. 读取测试集 embedding
    # --------------------------------------------------------
    protein_names = read_protein_names(TEST_LIST_FILE)

    print(f"Number of proteins in the test set: {len(protein_names)}")

    dataset = EmbeddingOnlyDataset(
        protein_names=protein_names,
        plm1_embedding_dir=PLM1_EMBEDDING_DIR,
        plm2_embedding_dir=PLM2_EMBEDDING_DIR,
        plm1_dim=PLM1_DIM,
        plm2_dim=PLM2_DIM
    )

    # --------------------------------------------------------
    # 3. 固定同一批 SHAP 输入
    # --------------------------------------------------------
    bg_cpu, ex_cpu, bg_names, explained_names, lcap_eff, manifest_df = (
        prepare_fixed_shap_inputs(
            dataset=dataset,
            n_background=N_BACKGROUND,
            n_explain=N_EXPLAIN,
            lcap=LCAP,
            random_seed=RANDOM_SEED
        )
    )

    manifest_path = OUTPUT_ROOT / "selected_proteins_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print("\n" + "=" * 80)
    print("Fixed SHAP input configuration")
    print("=" * 80)
    print(f"Background proteins: {len(bg_names)}")
    print(f"Explained proteins: {len(explained_names)}")
    print(f"Effective cropped length: {lcap_eff}")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 80)

    # --------------------------------------------------------
    # 4. 逐个 checkpoint 分析
    # --------------------------------------------------------
    all_protein_level_results = []
    run_level_results = []

    for run_id, checkpoint_path in enumerate(valid_checkpoints):
        checkpoint_stem = Path(checkpoint_path).stem

        run_out_dir = OUTPUT_ROOT / f"run{run_id}_{checkpoint_stem}"

        summary_df, run_result = analyze_one_checkpoint(
            checkpoint_path=checkpoint_path,
            run_id=run_id,
            out_dir=run_out_dir,
            bg_cpu=bg_cpu,
            ex_cpu=ex_cpu,
            explained_names=explained_names,
            lcap_eff=lcap_eff,
            plm1_name=PLM1_NAME,
            plm1_dim=PLM1_DIM,
            plm2_name=PLM2_NAME,
            plm2_dim=PLM2_DIM,
            nsamples=NSAMPLES,
            random_seed=RANDOM_SEED,
            n_display=N_DISPLAY
        )

        all_protein_level_results.append(summary_df)
        run_level_results.append(run_result)

    # --------------------------------------------------------
    # 5. 汇总五个 checkpoint 的蛋白级数据
    # --------------------------------------------------------
    all_protein_df = pd.concat(
        all_protein_level_results,
        axis=0,
        ignore_index=True
    )

    all_protein_csv = OUTPUT_ROOT / "all_runs_protein_level_attribution.csv"
    all_protein_df.to_csv(all_protein_csv, index=False)

    # --------------------------------------------------------
    # 6. 汇总五个 checkpoint 的 run-level 数据
    # --------------------------------------------------------
    run_summary_df = pd.DataFrame(run_level_results)

    run_summary_csv = OUTPUT_ROOT / "five_runs_shap_summary.csv"
    run_summary_df.to_csv(run_summary_csv, index=False)

    # --------------------------------------------------------
    # 7. 五个 checkpoint 的总体平均结果
    # --------------------------------------------------------
    plm1_run_means = run_summary_df[
        f"{PLM1_NAME}_mean_ratio_percent"
    ].values

    plm2_run_means = run_summary_df[
        f"{PLM2_NAME}_mean_ratio_percent"
    ].values

    plm1_overall_mean = float(np.mean(plm1_run_means))
    plm1_overall_std = float(
        np.std(plm1_run_means, ddof=1)
    ) if len(plm1_run_means) > 1 else 0.0

    plm2_overall_mean = float(np.mean(plm2_run_means))
    plm2_overall_std = float(
        np.std(plm2_run_means, ddof=1)
    ) if len(plm2_run_means) > 1 else 0.0

    summary_text = (
        "\n============================================================\n"
        "SHAP summary across independently trained checkpoints\n"
        "============================================================\n"
        f"Number of checkpoints: {len(run_summary_df)}\n"
        f"Number of explained proteins per checkpoint: {len(explained_names)}\n"
        f"Effective cropped length: {lcap_eff}\n\n"
        f"{PLM1_NAME} mean attribution across runs: "
        f"{plm1_overall_mean:.2f}% ± {plm1_overall_std:.2f}%\n"
        f"{PLM2_NAME} mean attribution across runs: "
        f"{plm2_overall_mean:.2f}% ± {plm2_overall_std:.2f}%\n"
        "============================================================\n"
    )

    print(summary_text)

    summary_text_path = OUTPUT_ROOT / "five_runs_shap_summary.txt"

    with open(summary_text_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # --------------------------------------------------------
    # 8. 保存五次 run 的平均归因图
    # --------------------------------------------------------
    save_five_runs_mean_plot(
        run_summary_df=run_summary_df,
        plm1_name=PLM1_NAME,
        plm2_name=PLM2_NAME,
        out_path=OUTPUT_ROOT / "five_runs_mean_attribution.png"
    )

    print("\nAll SHAP analyses completed.")
    print(f"Protein-level data: {all_protein_csv}")
    print(f"Run-level summary: {run_summary_csv}")
    print(f"Overall summary: {summary_text_path}")


if __name__ == "__main__":
    main()