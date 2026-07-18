import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from LossFunction.focalLoss import FocalLoss_v2
from Resnet1 import VNet
import torch.multiprocessing
import datetime
import copy
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

try:
    from torch.func import functional_call
except ImportError:
    # 兼容较早版本 PyTorch
    from torch.nn.utils.stateless import functional_call

def read_data_file_trip(istraining):
    """读取SMB数据文件"""
    if istraining:
        # file_path = 'D:/fengzhen/1NucGMTL-main/DataSet/SMB/SMB_Train.txt'
        file_path = 'E:/fengzhen/NucGMTL-main/DataSet/ATP/387_41/ATP387.txt'
    else:
        file_path = 'E:/fengzhen/NucGMTL-main/DataSet/ATP/387_41/ATP41.txt'
    # if istraining:
    #     # file_path = 'D:/fengzhen/1NucGMTL-main/DataSet/SMB/SMB_Train.txt'
    #     file_path = 'E:/fengzhen/NucGMTL-main/DataSet/DNA/Train/DNA_Train.txt'
    # else:
    #     file_path = 'E:/fengzhen/NucGMTL-main/DataSet/DNA/Test/DNA_Test.txt'

    with open(file_path, 'r') as f:
        data = f.readlines()

    results = []
    block = len(data) // 2
    for index in range(block):
        item = data[index * 2 + 0].split()
        name = item[0].strip()
        results.append(name)

    return results


def coll_paddding(batch_traindata):
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)
    feature_plms = []
    train_y = []
    task_ids = []

    for data in batch_traindata:
        feature_plms.append(data[0])

        train_y.append(data[1])
        task_ids.append(data[2])
    data_length = [len(data) for data in feature_plms]

    feature_plms = torch.nn.utils.rnn.pad_sequence(feature_plms, batch_first=True, padding_value=0)
    train_y = torch.nn.utils.rnn.pad_sequence(train_y, batch_first=True, padding_value=0)
    task_ids = torch.nn.utils.rnn.pad_sequence(task_ids, batch_first=True, padding_value=0)
    return feature_plms, train_y, task_ids, torch.tensor(data_length)


class BioinformaticsDataset(Dataset):
    # X: list of filename
    def __init__(self, X):
        self.X = X

    def __getitem__(self, index):
        filename = self.X[index]
        # esm_embedding1280 prot_embedding  esm_embedding2560 msa_embedding
        df0 = pd.read_csv('E:/fengzhen/embedding_ATP1/ProstT5_embedding_387_41/' + filename + '.data', header=None)
        prot0 = df0.values.astype(float).tolist()
        prot0 = torch.tensor(prot0)

        df1 = pd.read_csv('E:/fengzhen/embedding_ATP1/Ankh_embedding_387_41/' + filename + '.data', header=None)
        prot1 = df1.values.astype(float).tolist()
        prot1 = torch.tensor(prot1)

        # combine two plms
        # prot = torch.cat((prot0, prot1), dim=1)  # prot0 #

        min_len = min(prot0.size(0), prot1.size(0))      # 需要
        prot = torch.cat((prot0[:min_len], prot1[:min_len]), dim=1)     # 需要

        df2 = pd.read_csv('E:/fengzhen/embedding_ATP1/label_387_41/' + filename + '.label', header=None)
        label = df2.values.astype(int).tolist()
        label = torch.tensor(label)
        # reduce 2D to 1D
        label = torch.squeeze(label)

        label = label[:min_len]       # 截断 label，使其与 prot 对齐     添加
        task_id_label = torch.ones(prot.shape[0], dtype=int) * 0

        return prot, label, task_id_label

    def __len__(self):
        return len(self.X)


class AttentionModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AttentionModel, self).__init__()
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self._norm_fact = 1 / torch.sqrt(torch.tensor(out_dim))

    def forward(self, plms1, seqlengths):
        Q = self.q(plms1)
        K = self.k(plms1)
        V = self.v(plms1)
        atten = self.masked_softmax((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact, seqlengths)
        output = torch.bmm(atten, V)
        return output + V

    def create_src_lengths_mask(self, batch_size: int, src_lengths):
        max_src_len = int(src_lengths.max())
        src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
        src_indices = src_indices.expand(batch_size, max_src_len)
        src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
        # returns [batch_size, max_seq_len]
        return (src_indices < src_lengths).int().detach()

    def masked_softmax(self, scores, src_lengths, src_length_masking=True):
        if src_length_masking:
            bsz, src_len, max_src_len = scores.size()
            src_mask = self.create_src_lengths_mask(bsz, src_lengths)
            src_mask = src_mask.unsqueeze(2)
            src_mask = src_mask.to(scores.device)  # <-- 关键
            scores = scores.permute(0, 2, 1)
            scores = scores.masked_fill(src_mask == 0, -np.inf)
            scores = scores.permute(0, 2, 1)
        return F.softmax(scores.float(), dim=-1)


class FeatureExtractor(nn.Module):
    def __init__(self, inputdim):
        super(FeatureExtractor, self).__init__()
        self.inputdim = inputdim

        self.ms1cnn1 = nn.Conv1d(self.inputdim, 512, 1, padding='same')
        self.ms1cnn2 = nn.Conv1d(512, 256, 1, padding='same')
        self.ms1cnn3 = nn.Conv1d(256, 128, 1, padding='same')

        self.ms2cnn1 = nn.Conv1d(self.inputdim, 512, 3, padding='same')
        self.ms2cnn2 = nn.Conv1d(512, 256, 3, padding='same')
        self.ms2cnn3 = nn.Conv1d(256, 128, 3, padding='same')

        self.ms3cnn1 = nn.Conv1d(self.inputdim, 512, 5, padding='same')
        self.ms3cnn2 = nn.Conv1d(512, 256, 5, padding='same')
        self.ms3cnn3 = nn.Conv1d(256, 128, 5, padding='same')

        self.relu = nn.ReLU(True)

        self.AttentionModel1 = AttentionModel(512, 128)
        self.AttentionModel2 = AttentionModel(256, 128)
        self.AttentionModel3 = AttentionModel(128, 128)

    def forward(self, prot_input, seqlengths):
        prot_input_share = prot_input.permute(0, 2, 1)

        m1 = self.relu(self.ms1cnn1(prot_input_share))
        m2 = self.relu(self.ms2cnn1(prot_input_share))
        m3 = self.relu(self.ms3cnn1(prot_input_share))

        att = m1 + m2 + m3
        att = att.permute(0, 2, 1)
        s1 = self.AttentionModel1(att, seqlengths)

        m1 = self.relu(self.ms1cnn2(m1))
        m2 = self.relu(self.ms2cnn2(m2))
        m3 = self.relu(self.ms3cnn2(m3))

        att = m1 + m2 + m3
        att = att.permute(0, 2, 1)
        s2 = self.AttentionModel2(att, seqlengths)

        m1 = self.relu(self.ms1cnn3(m1))
        m2 = self.relu(self.ms2cnn3(m2))
        m3 = self.relu(self.ms3cnn3(m3))

        att = m1 + m2 + m3
        att = att.permute(0, 2, 1)
        s3 = self.AttentionModel3(att, seqlengths)

        mscnn = m1 + m2 + m3
        mscnn = mscnn.permute(0, 2, 1)
        s = s1 + s2 + s3

        return mscnn + s


class Module(nn.Module):
    def __init__(self, inputdim, istrain):
        super(Module, self).__init__()
        self.istrain = istrain
        self.inputdim = inputdim
        self.feature_extractor = FeatureExtractor(self.inputdim)

        # 单任务：每个残基输出一个 logit
        self.task_fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, prot_input, datalengths):
        features = self.feature_extractor(prot_input, datalengths)
        output = self.task_fc(features)      # [B, L, 1]
        return output.squeeze(-1)            # [B, L]


def build_valid_mask(lengths, max_len, device):
    """
    根据每条蛋白质真实长度构建有效残基掩码。
    True 表示真实残基，False 表示 padding 位置。
    """
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    return positions < lengths.to(device).unsqueeze(1)


def get_valid_residues(logits, labels, lengths):
    """
    删除 padding 位置，只保留真实残基的预测值和标签。
    logits: [B, L]
    labels: [B, L]
    """
    mask = build_valid_mask(lengths, labels.size(1), logits.device)
    valid_logits = logits[mask]
    valid_labels = labels[mask].float()
    return valid_logits, valid_labels


def binary_focal_loss_per_residue(
    logits,
    labels,
    # alpha_pos=2.0,
    alpha_pos=10.0,
    alpha_neg=1.0,
    gamma=2.0
):
    """
    单任务二分类逐残基 Focal Loss。
    logits: [N]
    labels: [N], values in {0, 1}
    """
    labels = labels.float()

    bce_loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none"
    )

    probs = torch.sigmoid(logits)
    p_t = probs * labels + (1.0 - probs) * (1.0 - labels)

    alpha_t = alpha_pos * labels + alpha_neg * (1.0 - labels)

    focal_loss = alpha_t * (1.0 - p_t).pow(gamma) * bce_loss
    return focal_loss


@torch.no_grad()
def evaluate_validation_loss(
    model,
    val_loader,
    # alpha_pos=2.0,
    alpha_pos=10.0,
    alpha_neg=1.0,
    gamma=2.0
):
    """
    在严格独立的 validation set 上计算未加权 focal loss，
    用于早停与保存最佳模型。
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_residues = 0

    for prot_xs, data_ys, _, lengths in val_loader:
        prot_xs = prot_xs.to(device)
        data_ys = data_ys.to(device)
        lengths = lengths.to("cpu")

        logits = model(prot_xs, lengths)
        valid_logits, valid_labels = get_valid_residues(
            logits,
            data_ys,
            lengths
        )

        losses = binary_focal_loss_per_residue(
            valid_logits,
            valid_labels,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            gamma=gamma
        )

        total_loss += losses.sum().item()
        total_residues += losses.numel()

    if was_training:
        model.train()

    if total_residues == 0:
        return float("inf")

    return total_loss / total_residues



# 假设 Module, FeatureExtractor, BioinformaticsDataset, coll_paddding 已经定义
# VNet 类也已经定义
# device 初始化
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

# -------------------------- Train with VNet --------------------------
def train_vnet(
    itrainfile,
    modelstoreapl,
    valfile,
    # alpha_pos=2.0,
    alpha_pos=10.0,
    alpha_neg=1.0,
    gamma=2.0,
    warmup_epochs=3
):
    """
    正确的双层优化动态残基加权训练。

    Inner loop:
        weighted training loss -> virtual parameters

    Outer loop:
        validation loss of virtual model -> V-Net update

    Final step:
        updated V-Net weights -> actual primary-model update
    """
    if valfile is None or len(valfile) == 0:
        raise ValueError("A non-empty held-out validation file list is required.")

    if len(itrainfile) == 0:
        raise ValueError("Training file list is empty.")

    # 主预测模型
    model = Module(1536 + 1024, True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 动态权重网络
    vnet = VNet(
        input=1,
        hidden1=100,
        output=1,
        num_classes=1
    ).to(device)

    optimizer_vnet = torch.optim.Adam(vnet.parameters(), lr=1e-4)

    epochs = 30
    # epochs = 1
    patience = 5
    weight_floor = 0.1

    # 严格分开的训练集与 meta-validation set
    train_set = BioinformaticsDataset(itrainfile)
    val_set = BioinformaticsDataset(valfile)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=coll_paddding
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=coll_paddding
    )

    best_val_loss = float("inf")
    best_epo = -1
    counter = 0

    for epo in range(epochs):
        model.train()
        vnet.train()

        epoch_loss_sum = 0.0
        epoch_residue_count = 0

        val_iterator = iter(val_loader)

        for prot_xs, data_ys, _, lengths in train_loader:
            prot_xs = prot_xs.to(device)
            data_ys = data_ys.to(device)
            lengths = lengths.to("cpu")

            # ------------------------------------------------------
            # Warm-up: 前 warmup_epochs 个 epoch 仅使用普通 Focal Loss
            # ------------------------------------------------------
            if epo < warmup_epochs:
                logits = model(prot_xs, lengths)

                valid_logits, valid_labels = get_valid_residues(
                    logits,
                    data_ys,
                    lengths
                )

                loss_per_residue = binary_focal_loss_per_residue(
                    valid_logits,
                    valid_labels,
                    alpha_pos=alpha_pos,
                    alpha_neg=alpha_neg,
                    gamma=gamma
                )

                final_loss = loss_per_residue.mean()

                optimizer.zero_grad(set_to_none=True)
                final_loss.backward()
                optimizer.step()

                epoch_loss_sum += loss_per_residue.detach().sum().item()
                epoch_residue_count += loss_per_residue.numel()
                continue

            # ======================================================
            # Step 1: Inner loop -- construct virtual model theta'
            # ======================================================
            logits = model(prot_xs, lengths)

            valid_logits, valid_labels = get_valid_residues(
                logits,
                data_ys,
                lengths
            )

            loss_per_residue = binary_focal_loss_per_residue(
                valid_logits,
                valid_labels,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                gamma=gamma
            )

            # V-Net 根据每个残基当前 loss 生成动态权重
            weights_inner = vnet(
                loss_per_residue.detach().unsqueeze(1)
            ).squeeze(1)

            # 保留你原来的权重下限设置
            weights_inner = torch.clamp(
                weights_inner,
                min=weight_floor,
                max=1.0
            )

            weighted_inner_loss = torch.mean(
                weights_inner * loss_per_residue
            )

            # 获取主模型参数
            base_params = OrderedDict(model.named_parameters())
            base_buffers = OrderedDict(model.named_buffers())

            # 计算 inner loss 对 theta 的梯度；
            # create_graph=True 保证 theta' 对 V-Net 参数 Phi 保持可微
            grads = torch.autograd.grad(
                weighted_inner_loss,
                tuple(base_params.values()),
                create_graph=True
            )

            inner_lr = optimizer.param_groups[0]["lr"]

            # 真正构造 virtual parameters theta'
            fast_params = OrderedDict(
                (
                    name,
                    param - inner_lr * grad
                )
                for (name, param), grad in zip(
                    base_params.items(),
                    grads
                )
            )

            # ======================================================
            # Step 2: Outer loop -- update V-Net using validation loss
            # ======================================================
            try:
                val_prot, val_label, _, val_lengths = next(val_iterator)
            except StopIteration:
                val_iterator = iter(val_loader)
                val_prot, val_label, _, val_lengths = next(val_iterator)

            val_prot = val_prot.to(device)
            val_label = val_label.to(device)
            val_lengths = val_lengths.to("cpu")

            # Validation 时关闭 dropout，使 meta loss 更稳定
            previous_mode = model.training
            model.eval()

            fast_state = OrderedDict()
            fast_state.update(base_buffers)
            fast_state.update(fast_params)

            # functional_call 使用 theta' 前向，而不修改原模型 theta
            virtual_val_logits = functional_call(
                model,
                fast_state,
                (val_prot, val_lengths)
            )

            if previous_mode:
                model.train()

            valid_val_logits, valid_val_labels = get_valid_residues(
                virtual_val_logits,
                val_label,
                val_lengths
            )

            meta_loss_per_residue = binary_focal_loss_per_residue(
                valid_val_logits,
                valid_val_labels,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                gamma=gamma
            )

            meta_loss = meta_loss_per_residue.mean()

            # meta loss 通过 theta'(Phi) 反向传播，更新 V-Net
            optimizer.zero_grad(set_to_none=True)
            optimizer_vnet.zero_grad(set_to_none=True)

            meta_loss.backward()
            optimizer_vnet.step()

            # ======================================================
            # Step 3: 用更新后的 V-Net 权重更新主模型 theta
            # ======================================================
            logits = model(prot_xs, lengths)

            valid_logits, valid_labels = get_valid_residues(
                logits,
                data_ys,
                lengths
            )

            final_loss_per_residue = binary_focal_loss_per_residue(
                valid_logits,
                valid_labels,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                gamma=gamma
            )

            # 使用更新后的 V-Net 重新计算权重
            with torch.no_grad():
                final_weights = vnet(
                    final_loss_per_residue.detach().unsqueeze(1)
                ).squeeze(1)

                final_weights = torch.clamp(
                    final_weights,
                    min=weight_floor,
                    max=1.0
                )

            final_loss = torch.mean(
                final_weights * final_loss_per_residue
            )

            optimizer.zero_grad(set_to_none=True)
            final_loss.backward()
            optimizer.step()

            epoch_loss_sum += final_loss_per_residue.detach().sum().item()
            epoch_residue_count += final_loss_per_residue.numel()

        # 使用 held-out validation loss 保存最佳模型
        train_loss = epoch_loss_sum / max(epoch_residue_count, 1)

        val_loss = evaluate_validation_loss(
            model,
            val_loader,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            gamma=gamma
        )

        print(
            f"Epoch {epo + 1}/{epochs} | "
            f"Train focal loss: {train_loss:.6f} | "
            f"Validation focal loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), modelstoreapl)
            best_val_loss = val_loss
            best_epo = epo + 1
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(
        f"Best validation loss: {best_val_loss:.6f} | "
        f"Best epoch: {best_epo}"
    )

    return model, vnet

# -------------------------- Test --------------------------
def test(modelstoreapl, testfile=None):
    model = Module(1536 + 1024, False).to(device)
    model.load_state_dict(
        torch.load(modelstoreapl, map_location=device)
    )
    model.eval()

    if testfile is None:
        test_files = read_data_file_trip(False)
    else:
        test_files = testfile

    test_set = BioinformaticsDataset(test_files)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=coll_paddding
    )

    predicted_probs = []
    labels_actual = []
    labels_predicted = []

    with torch.no_grad():
        for prot_xs, data_ys, _, lengths in test_loader:
            prot_xs = prot_xs.to(device)
            data_ys = data_ys.to(device)
            lengths = lengths.to("cpu")

            logits = model(prot_xs, lengths)       # [B, L]
            probs = torch.sigmoid(logits)          # [B, L]
            preds = (probs >= 0.5).long()          # [B, L]

            # 只保留真实残基，排除 padding
            for batch_idx, seq_len in enumerate(lengths.tolist()):
                valid_probs = probs[batch_idx, :seq_len]
                valid_preds = preds[batch_idx, :seq_len]
                valid_labels = data_ys[batch_idx, :seq_len]

                predicted_probs.extend(
                    valid_probs.detach().cpu().tolist()
                )
                labels_predicted.extend(
                    valid_preds.detach().cpu().tolist()
                )
                labels_actual.extend(
                    valid_labels.detach().cpu().tolist()
                )

    return (
        np.asarray(labels_actual, dtype=np.int64),
        np.asarray(labels_predicted, dtype=np.int64),
        np.asarray(predicted_probs, dtype=np.float64)
    )

# -------------------------- Metrics --------------------------
def printresult(ligand, actual_label, predict_prob, predict_label):
    print('\n---------', ligand, '-------------')
    auc = metrics.roc_auc_score(actual_label, predict_prob)
    precision_1, recall_1, threshold_1 = metrics.precision_recall_curve(actual_label, predict_prob)
    aupr_1 = metrics.auc(recall_1, precision_1)
    acc = metrics.accuracy_score(actual_label, predict_label)
    tn, fp, fn, tp = metrics.confusion_matrix(actual_label, predict_label).ravel()
    mcc = metrics.matthews_corrcoef(actual_label, predict_label)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn)
    f1score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    youden = sensitivity + specificity - 1

    print(f'acc: {acc}, sensitivity: {sensitivity}, specificity: {specificity}, precision: {precision}')
    print(f'MCC: {mcc}, F1: {f1score}, AUC: {auc}, AUPR: {aupr_1}, Youden: {youden}')
    return sensitivity, specificity, acc, precision, mcc, auc, aupr_1

def plot_roc_pr_curves(actual_label, predict_prob, run_index, save_dir, ligand="PPI"):
    """
    为第 run_index 次重复训练保存一张 ROC 曲线和一张 PR 曲线。
    """

    actual_label = np.asarray(actual_label, dtype=np.int64).ravel()
    predict_prob = np.asarray(predict_prob, dtype=np.float64).ravel()

    if len(np.unique(actual_label)) < 2:
        print(f"Run {run_index + 1}: only one class is present; ROC/PR curves were not generated.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # ---------------- ROC curve ----------------
    fpr, tpr, _ = metrics.roc_curve(actual_label, predict_prob)
    roc_auc = metrics.auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    # ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot(
        fpr,
        tpr,
        color="#DA3C5D",  # 曲线颜色
        linewidth=1.2,  # 曲线线宽
        label=f"ROC (AUC = {roc_auc:.4f})"
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Random classifier"
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{ligand} ROC Curve — Run {run_index + 1}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    roc_path = os.path.join(
        save_dir,
        f"{ligand}_run_{run_index + 1}_ROC.png"
    )
    fig.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- PR curve ----------------
    precision, recall, _ = metrics.precision_recall_curve(
        actual_label,
        predict_prob
    )

    # recall 通常从 1 降至 0；反向后更适合绘图和计算梯形 AUPRC
    pr_auc = metrics.auc(recall[::-1], precision[::-1])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(
        recall[::-1],
        precision[::-1],
        color="#DA3C5D",  # 曲线颜色 红色
        linewidth=1.2,  # 曲线线宽
        label=f"PR curve (AUPRC = {pr_auc:.4f})"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{ligand} Precision–Recall Curve — Run {run_index + 1}")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    pr_path = os.path.join(
        save_dir,
        f"{ligand}_run_{run_index + 1}_PR.png"
    )
    fig.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Run {run_index + 1} curves saved:")
    print(f"  ROC: {roc_path}")
    print(f"  PR : {pr_path}")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")

    # 单任务：固定为DNA
    tasks = ['DNA']

    # 重复训练次数
    circle = 5
    timestamp = str(datetime.datetime.now()).replace(':', '_')
    totalkv = {task: [] for task in tasks}
    storename = '_'.join(tasks)
    print(f"Task: {storename}")

    curve_dir = os.path.join(
        "RGMTL",
        "Curves",
        f"{storename}_{timestamp}"
    )
    os.makedirs(curve_dir, exist_ok=True)

    # ------------------------ 读取训练数据 ------------------------
    all_train_files = read_data_file_trip(True)

    rng = np.random.default_rng(2026)
    permutation = rng.permutation(len(all_train_files))

    val_size = max(1, int(0.2 * len(all_train_files)))

    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]

    val_files = [all_train_files[idx] for idx in val_indices]
    train_files = [all_train_files[idx] for idx in train_indices]

    print(
        f"Training proteins: {len(train_files)} | "
        f"Meta-validation proteins: {len(val_files)}"
    )

    # ------------------------ 循环训练 ------------------------
    for i in range(circle):
        storeapl = f'RGMTL/Result_{storename}_{i}_{timestamp}.pkl'
        print(f"\n=== Training iteration {i + 1}/{circle} ===")

        train_vnet(
            train_files,
            storeapl,
            valfile=val_files,
            # alpha_pos=2.0,  # 当前 DNA 设置对应原 alpha=[1.0, 2.0]
            # alpha_pos=5.0,  # 当前 DNA 设置对应原 alpha=[1.0, 2.0]
            alpha_pos=10.0,  # 当前 ATP 设置对应原 alpha=[1.0, 2.0]
            alpha_neg=1.0,
            gamma=2.0,
            warmup_epochs=3
        )

        # 测试集预测
        labels_actual, labels_predicted, predicted_probs = test(storeapl)

        # 计算并输出指标
        sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult(
            'PPI',
            labels_actual,
            predicted_probs,
            labels_predicted
        )

        # 保存本次重复训练对应的 ROC 和 PR 曲线
        plot_roc_pr_curves(
            actual_label=labels_actual,
            predict_prob=predicted_probs,
            run_index=i,
            save_dir=curve_dir,
            ligand="ATP-41"
        )

        totalkv['DNA'].append(
            [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------ 保存结果 ------------------------
    result_txt = f'RGMTL/Result_{storename}_{timestamp}.txt'
    with open(result_txt, 'w') as f:
        for nuc in tasks:
            np.savetxt(f, totalkv[nuc], delimiter=',', footer=f'Above is record {nuc}', fmt='%s')
            m = np.mean(totalkv[nuc], axis=0)
            np.savetxt(f, [m], delimiter=',', footer=f'----------Above is AVG ------- {nuc}', fmt='%s')
    print(f"All results saved to {result_txt}")