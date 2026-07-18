import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from LossFunction.focalLoss import FocalLoss_v2
import torch.multiprocessing
import datetime

# def count_parameters(model: torch.nn.Module):         # 新
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total, trainable


def read_data_file_trip(istraining):
    """读取SMB数据文件"""
    if istraining:
        # file_path = 'D:/fengzhen/1NucGMTL-main/DataSet/SMB/SMB_Train.txt'
        file_path = 'E:/fengzhen/NucGMTL-main/DataSet/ATP/387_41/ATP387.txt'
    else:
        file_path = 'E:/fengzhen/NucGMTL-main/DataSet/ATP/387_41/ATP41.txt'

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

        # # combine two plms
        # prot = torch.cat((prot0, prot1), dim=1)  # prot0#

        min_len = min(prot0.size(0), prot1.size(0))      # 需要
        prot = torch.cat((prot0[:min_len], prot1[:min_len]), dim=1)     # 需要

        df2 = pd.read_csv('E:/fengzhen/embedding_ATP1/label_387_41/' + filename + '.label', header=None)
        label = df2.values.astype(int).tolist()
        label = torch.tensor(label)
        # reduce 2D to 1D
        label = torch.squeeze(label)
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
        # scores [batchsize,L*L]
        if src_length_masking:
            bsz, src_len, max_src_len = scores.size()
            # compute masks
            src_mask = self.create_src_lengths_mask(bsz, src_lengths)
            src_mask = torch.unsqueeze(src_mask, 2)
            # Fill pad positions with -inf
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


def train(itrainfile, modelstoreapl):
    # 单任务：tasklen固定为1
    model = Module(1536+ 1024, True)
    model = model.to(device)

    # total_params, trainable_params = count_parameters(model)       # 新
    # print(f"[Model Params] Total: {total_params:,} | Trainable: {trainable_params:,}")    # 新

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 30

    per_cls_weights = torch.FloatTensor([0.15, 0.85]).to(device)
    fcloss = FocalLoss_v2(alpha=per_cls_weights, gamma=2)

    model.train()

    file = read_data_file_trip(True)
    # 单任务：数据集初始化简化
    train_set = BioinformaticsDataset(file)
    train_loader = DataLoader(dataset=train_set, batch_size=16, pin_memory=True,
                              persistent_workers=True, shuffle=True, num_workers=16,
                              collate_fn=coll_paddding)

    best_val_loss = 3000
    best_epo = 0
    patience = 5
    counter = 0

    for epo in range(epochs):
        epoch_loss_train = 0
        nb_train = 0
        for prot_xs, data_ys, taskids, lengths in train_loader:
            outputs = model(prot_xs.to(device), lengths.to(device))
            data_ys = data_ys.to(device)
            lengths = lengths.to('cpu')

            outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=True)
            data_ys = torch.nn.utils.rnn.pack_padded_sequence(data_ys, lengths, batch_first=True)

            loss = fcloss(outputs.data, data_ys.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train = epoch_loss_train + loss.item()
            nb_train += 1

        epoch_loss_avg = epoch_loss_train / nb_train
        print('epo ', epo, ' epoch_loss_avg,', epoch_loss_avg)

        if best_val_loss > epoch_loss_avg:
            model_fn = modelstoreapl
            torch.save(model.state_dict(), model_fn)
            best_val_loss = epoch_loss_avg
            best_epo = epo
            if epo % 10 == 0:
                print('epo ', epo, " Save model, best_val_loss: ", best_val_loss)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    print('best loss,', best_val_loss, 'best epo,', best_epo)


def test(itestfile, modelstoreapl):
    model = Module(1536 + 1024, False)
    model = model.to(device)
    model.load_state_dict(torch.load(modelstoreapl))
    model.eval()

    file = read_data_file_trip(False)
    test_set = BioinformaticsDataset(file)
    test_load = DataLoader(dataset=test_set, batch_size=32,
                           num_workers=16, pin_memory=True, persistent_workers=True,
                           collate_fn=coll_paddding)

    print("==========================Test RESULT================================")

    predicted_probs = []
    labels_actual = []
    labels_predicted = []

    with torch.no_grad():
        for prot_xs, data_ys, taskids, lengths in test_load:
            outputs = model(prot_xs.to(device), lengths.to(device))
            outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths.to('cpu'), batch_first=True)
            data_ys = torch.nn.utils.rnn.pack_padded_sequence(data_ys, lengths, batch_first=True)

            pred_probs = torch.nn.functional.softmax(outputs.data, dim=1)
            pred_probs = pred_probs.to('cpu')

            predicted_probs.extend(pred_probs[:, 1])
            labels_actual.extend(data_ys.data)
            labels_predicted.extend(torch.argmax(pred_probs, dim=1))

    sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult('SMB', labels_actual,
                                                                             predicted_probs, labels_predicted)
    tmresult = {'SMB': [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]}

    return tmresult


def printresult(ligand, actual_label, predict_prob, predict_label):
    print('\n---------', ligand, '-------------')
    auc = metrics.roc_auc_score(actual_label, predict_prob)
    precision_1, recall_1, threshold_1 = metrics.precision_recall_curve(actual_label, predict_prob)
    aupr_1 = metrics.auc(recall_1, precision_1)
    acc = metrics.accuracy_score(actual_label, predict_label)
    print('acc ', acc)
    print('balanced_accuracy ', metrics.balanced_accuracy_score(actual_label, predict_label))
    tn, fp, fn, tp = metrics.confusion_matrix(actual_label, predict_label).ravel()
    print('tn, fp, fn, tp ', tn, fp, fn, tp)
    mcc = metrics.matthews_corrcoef(actual_label, predict_label)
    print('MCC ', mcc)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1score = 2 * precision * recall / (precision + recall)
    youden = sensitivity + specificity - 1
    print('sensitivity ', sensitivity)
    print('specificity ', specificity)
    print('precision ', precision)
    print('recall ', recall)
    print('f1score ', f1score)
    print('youden ', youden)
    print('auc', auc)
    print('AUPR ', aupr_1)
    print('---------------END------------')
    return sensitivity, specificity, acc, precision, mcc, auc, aupr_1


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")

    # 单任务：固定为SMB
    tasks = ['SMB']

    circle = 5
    a = str(datetime.datetime.now())
    a = a.replace(':', '_')

    totalkv = {task: [] for task in tasks}
    storename = '_'.join(p for p in tasks)
    print(storename)

    for i in range(circle):
        storeapl = 'RGMTL/Result_' + storename + '_' + str(i) + '_' + a + '.pkl'
        train(tasks, storeapl)
        tmresult = test(tasks, storeapl)
        for task in tasks:
            totalkv[task].append(tmresult[task])
        torch.cuda.empty_cache()

    with open('RGMTL/Result_' + storename + '_' + a + '.txt', 'w') as f:
        for nuc in tasks:
            np.savetxt(f, totalkv[nuc], delimiter=',', footer='Above is  record ' + nuc, fmt='%s')
            m = np.mean(totalkv[nuc], axis=0)
            np.savetxt(f, [m], delimiter=',', footer='----------Above is AVG -------' + nuc, fmt='%s')