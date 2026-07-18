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


def read_data_file_trip(istraining):
    """读取SMB数据文件"""
    if istraining:
        # file_path = 'D:/fengzhen/1NucGMTL-main/DataSet/SMB/SMB_Train.txt'
        file_path = 'E:/fengzhen/NucGMTL-main/DataSet/ATP/387_41/ATP387.txt'
    else:
        file_path = 'E:/fengzhen/NucGMTL-main/DataSet/ATP/387_41/ATP41.txt'
    # if istraining:
    #     file_path = '/DataSet/PPI/Train352-Test70/PPI-Train352.txt'
    # else:
    #     file_path = 'E:/fengzhen/NucGMTL-main/DataSet/PPI/Train352-Test70/PPI-Test70.txt'

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

        # 单任务：只有一个任务头
        self.task_fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, prot_input, datalengths):
        features = self.feature_extractor(prot_input, datalengths)
        # 单任务：直接返回一个输出
        output = self.task_fc(features)
        return output


# 假设 Module, FeatureExtractor, BioinformaticsDataset, coll_paddding 已经定义
# VNet 类也已经定义
# device 初始化
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

# -------------------------- Train with VNet --------------------------
def train_vnet(itrainfile, modelstoreapl, valfile=None):
    # 主模型
    # model = Module(1024, True).to(device)
    # model = Module(1024 + 1024, True).to(device)
    # model = Module(1280 + 1536, True).to(device)
    model = Module(1536 + 1024, True).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # VNet
    vnet = VNet(input=1, hidden1=100, output=1, num_classes=1).to(device)
    # optimizer_vnet = torch.optim.Adam(vnet.parameters(), lr=1e-5)
    optimizer_vnet = torch.optim.Adam(vnet.parameters(), lr=1e-4)

    epochs = 30
    # epochs = 3
    # epochs = 40
    patience = 5

    # 数据集
    train_files = read_data_file_trip(True)
    train_set = BioinformaticsDataset(train_files)
    # train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True,
    #                           collate_fn=coll_paddding, num_workers=16, pin_memory=True,
    #                           persistent_workers=True)
    train_loader = DataLoader(dataset=train_set, batch_size=16,
                              pin_memory=False,  # <- 改成 False
                              shuffle=True,
                              num_workers=0,  # <- 改成 0
                              collate_fn=coll_paddding)

    if valfile is None:
        valfile = train_files[:len(train_files)//5]
    val_set = BioinformaticsDataset(valfile)
    val_loader = DataLoader(dataset=val_set, batch_size=16, shuffle=True,
                            collate_fn=coll_paddding, num_workers=8, pin_memory=True,
                            persistent_workers=True)

    # best_val_loss = 1e5
    best_val_loss = 3000
    best_epo = 0
    counter = 0

    for epo in range(epochs):
        model.train()
        epoch_loss_train = 0
        nb_train = 0

        for prot_xs, data_ys, taskids, lengths in train_loader:
            prot_xs, data_ys, lengths = prot_xs.to(device), data_ys.to(device), lengths.to('cpu')

            # 主模型正向
            # ===== 主模型前向 =====
            outputs = model(prot_xs, lengths)

            # 展平
            outputs_flat = outputs.view(-1, 2)
            labels_flat = data_ys.view(-1)

            # ===== 使用 Focal Loss 逐样本版本 =====
            # alpha = torch.tensor([1.0, 50.0]).to(device)
            # alpha = torch.tensor([1.0, 5.0]).to(device)
            # alpha = torch.tensor([1.0, 2.0]).to(device)
            alpha = torch.tensor([1.0, 10.0]).to(device)

            ce_loss = F.cross_entropy(outputs_flat, labels_flat, reduction='none')
            pt = torch.exp(-ce_loss)

            alpha_t = alpha[labels_flat.long()]
            gamma = 2
            loss_per_residue = alpha_t * (1 - pt) ** gamma * ce_loss
            loss_per_residue = loss_per_residue.view(-1, 1)

            # VNet 输出每个残基权重
            if epo < 3:
                v_lambda = torch.ones_like(loss_per_residue).squeeze(1).to(device)
            else:
                v_lambda = vnet(loss_per_residue.detach(),
                                torch.zeros_like(loss_per_residue).to(device),  # dummy num
                                torch.zeros_like(loss_per_residue).to(device))  # dummy c
                v_lambda = v_lambda.squeeze(1)

                # 限制权重范围，避免过小导致梯度消失
                v_lambda = torch.clamp(v_lambda, min=0.1, max=1.0)

            weighted_loss = torch.mean(loss_per_residue.squeeze() * v_lambda)

            # Meta-update VNet using validation batch
            val_prot, val_label, _, val_lengths = next(iter(val_loader))
            val_prot, val_label = val_prot.to(device), val_label.to(device)
            val_lengths = val_lengths.to('cpu')

            meta_model = copy.deepcopy(model)
            meta_model.train()
            grads = torch.autograd.grad(weighted_loss, list(meta_model.parameters()),
                                        create_graph=True, allow_unused=True)
            lr_inner = 0.0001
            for p, g in zip(meta_model.parameters(), grads):
                if g is not None:
                    p.data -= lr_inner * g

            val_outputs = meta_model(val_prot, val_lengths)
            val_outputs = val_outputs.view(-1, 2)
            val_label_flat = val_label.view(-1)
            # meta_val_loss = nn.CrossEntropyLoss(reduction='mean')(val_outputs, val_label_flat)
            # val_alpha = torch.tensor([1.0, 50.0]).to(device)
            # val_alpha = torch.tensor([1.0, 5.0]).to(device)
            # val_alpha = torch.tensor([1.0, 2.0]).to(device)
            val_alpha = torch.tensor([1.0, 10.0]).to(device)

            ce_val = F.cross_entropy(val_outputs, val_label_flat, reduction='none')
            pt_val = torch.exp(-ce_val)
            alpha_val = val_alpha[val_label_flat.long()]

            # meta_val_loss = (alpha_val * (1 - pt_val) ** gamma * ce_val).mean()
            meta_val_loss = (alpha_val * (1 - pt_val) ** gamma * ce_val).mean()

            optimizer_vnet.zero_grad()
            meta_val_loss.backward()
            optimizer_vnet.step()

            # ===== 更新主模型（非常关键）=====
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            epoch_loss_train += weighted_loss.item()
            nb_train += 1

        epoch_loss_avg = epoch_loss_train / nb_train
        print('Epoch', epo, 'Avg weighted loss:', epoch_loss_avg)

        if epoch_loss_avg < best_val_loss:
            torch.save(model.state_dict(), modelstoreapl)
            best_val_loss = epoch_loss_avg
            best_epo = epo
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print('Best loss:', best_val_loss, 'Best epoch:', best_epo)
    return model, vnet

# -------------------------- Test --------------------------
def test(modelstoreapl, testfile=None):
    # model = Module(1024, False).to(device)
    # model = Module(1024 + 1024, False).to(device)
    # model = Module(1280 + 1536, False).to(device)
    model = Module(1536 + 1024, False).to(device)
    model.load_state_dict(torch.load(modelstoreapl))
    model.eval()

    if testfile is None:
        test_files = read_data_file_trip(False)
    else:
        test_files = testfile

    test_set = BioinformaticsDataset(test_files)
    test_loader = DataLoader(dataset=test_set, batch_size=16,
                             pin_memory=False,   # <- 改成 False
                             shuffle=False,
                             num_workers=0,      # <- 改成 0
                             collate_fn=coll_paddding)

    predicted_probs = []
    labels_actual = []
    labels_predicted = []

    with torch.no_grad():
        for prot_xs, data_ys, taskids, lengths in test_loader:
            prot_xs, data_ys = prot_xs.to(device), data_ys.to(device)
            lengths = lengths.to('cpu')

            # 前向
            outputs = model(prot_xs, lengths)

            # 展平 batch
            outputs_flat = outputs.view(-1, 2)

            # softmax 概率
            pred_probs = F.softmax(outputs_flat, dim=1).cpu()

            # 保存预测结果
            predicted_probs.extend(pred_probs[:, 1])  # 正类概率
            labels_predicted.extend(torch.argmax(pred_probs, dim=1).cpu())
            labels_actual.extend(data_ys.view(-1).cpu())

    return labels_actual, labels_predicted, predicted_probs

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

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")

    # 单任务：固定为DNA
    tasks = ['PPI']

    # 重复训练次数
    circle = 5
    timestamp = str(datetime.datetime.now()).replace(':', '_')
    totalkv = {task: [] for task in tasks}
    storename = '_'.join(tasks)
    print(f"Task: {storename}")

    # ------------------------ 读取训练数据 ------------------------
    train_files = read_data_file_trip(True)
    val_files = train_files[:len(train_files)//5]  # 验证集

    # ------------------------ 循环训练 ------------------------
    for i in range(circle):
        storeapl = f'RGMTL/Result_{storename}_{i}_{timestamp}.pkl'
        print(f"\n=== Training iteration {i+1}/{circle} ===")
        # 传入训练数据文件列表和模型保存路径
        train_vnet(train_files, storeapl, valfile=val_files)

        # 测试
        labels_actual, labels_predicted, predicted_probs = test(storeapl)
        sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult(
            'PPI', labels_actual, predicted_probs, labels_predicted
        )
        totalkv['PPI'].append([sensitivity, specificity, acc, precision, mcc, auc, aupr_1])
        torch.cuda.empty_cache()

    # ------------------------ 保存结果 ------------------------
    result_txt = f'RGMTL/Result_{storename}_{timestamp}.txt'
    with open(result_txt, 'w') as f:
        for nuc in tasks:
            np.savetxt(f, totalkv[nuc], delimiter=',', footer=f'Above is record {nuc}', fmt='%s')
            m = np.mean(totalkv[nuc], axis=0)
            np.savetxt(f, [m], delimiter=',', footer=f'----------Above is AVG ------- {nuc}', fmt='%s')
    print(f"All results saved to {result_txt}")