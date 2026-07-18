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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os


def read_data_file_trip(tasknames, istraining):
    results = []
    for taskname in tasknames:
        if istraining:
            f = open('D:/fengzhen/1NucGMTL-main/DataSet/SMB/' + taskname + '_Train.txt')
        else:
            f = open('D:/fengzhen/1NucGMTL-main/DataSet/SMB/' + taskname + '_Test.txt')
        data = f.readlines()
        f.close()
        tmpresults = []
        block = len(data) // 2
        for index in range(block):
            item = data[index * 2 + 0].split()
            name = item[0].strip()
            tmpresults.append(name)
        results.extend(tmpresults)
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
    def __init__(self, X, tasks):
        self.X = X
        self.Tasks = tasks

    def __getitem__(self, index):
        filename = self.X[index]
        df0 = pd.read_csv('D:/fengzhen/1embedding/prostT5_embedding_SMB/' + filename + '.data', header=None)
        prot0 = df0.values.astype(float).tolist()
        prot0 = torch.tensor(prot0)     # 改
        df1 = pd.read_csv('D:/fengzhen/1embedding/Ankh_embedding_SMB/' + filename + '.data', header=None)    # 需要
        prot1 = df1.values.astype(float).tolist()     # 需要
        prot = torch.tensor(prot1)     # 需要

        min_len = min(prot0.size(0), prot1.size(0))  # 需要
        prot = torch.cat((prot0[:min_len], prot1[:min_len]), dim=1)  # 需要

        df2 = pd.read_csv('D:/fengzhen/1embedding/label/' + filename + '.label', header=None)
        label = df2.values.astype(int).tolist()
        # label = df2.values.astype(np.float64).tolist()
        label = torch.tensor(label)
        # reduce 2D to 1D
        label = torch.squeeze(label)
        taskid = 0
        find = False
        for taskname in self.Tasks:
            if '_' + taskname in filename:
                find = True
                break
            taskid += 1
        if not find:
            taskid = 0
        task_id_label = torch.ones(prot.shape[0], dtype=int) * taskid

        return prot, label, task_id_label

    def __len__(self):
        return len(self.X)


class Task_shared(nn.Module):
    def __init__(self, inputdim):
        super(Task_shared, self).__init__()
        self.inputdim = inputdim

        # Multi-scale CNN only, no BiLSTM
        self.ms1cnn1 = nn.Conv1d(self.inputdim, 512, 3, padding='same')
        self.ms1cnn2 = nn.Conv1d(512, 256, 3, padding='same')
        self.ms1cnn3 = nn.Conv1d(256, 128, 3, padding='same')

        self.ms2cnn1 = nn.Conv1d(self.inputdim, 512, 5, padding='same')
        self.ms2cnn2 = nn.Conv1d(512, 256, 5, padding='same')
        self.ms2cnn3 = nn.Conv1d(256, 128, 5, padding='same')

        self.ms3cnn1 = nn.Conv1d(self.inputdim, 512, 7, padding='same')
        self.ms3cnn2 = nn.Conv1d(512, 256, 7, padding='same')
        self.ms3cnn3 = nn.Conv1d(256, 128, 7, padding='same')

        self.relu = nn.ReLU(True)

    def forward(self, prot_input, seqlengths=None):
        # prot_input: [batch, seq, dim]
        prot_input_share = prot_input.permute(0, 2, 1)  # [batch, dim, seq]

        # 第一层卷积
        m1 = self.relu(self.ms1cnn1(prot_input_share))
        m2 = self.relu(self.ms2cnn1(prot_input_share))
        m3 = self.relu(self.ms3cnn1(prot_input_share))

        # 第二层卷积
        m1 = self.relu(self.ms1cnn2(m1))
        m2 = self.relu(self.ms2cnn2(m2))
        m3 = self.relu(self.ms3cnn2(m3))

        # 第三层卷积
        m1 = self.relu(self.ms1cnn3(m1))
        m2 = self.relu(self.ms2cnn3(m2))
        m3 = self.relu(self.ms3cnn3(m3))

        # 特征融合
        fused = (m1 + m2 + m3).permute(0, 2, 1)  # [batch, seq, 128]

        # 缓存特征用于t-SNE可视化
        self.cache = {
            'plm': prot_input,  # PLM embeddings
            'final': fused  # Multi-scale CNN features (final features)
        }

        return fused


class MTLModule(nn.Module):
    def __init__(self, inputdim, istrain, tasklen):
        super(MTLModule, self).__init__()
        self.istrain = istrain
        self.inputdim = inputdim
        self.tasklen = tasklen
        self.ShardEncoder = Task_shared(self.inputdim)

        self.tasks_fcs = nn.ModuleList()

        for i in range(self.tasklen):
            self.tasks_fcs.append(nn.Sequential(nn.Linear(128, 512),
                                                nn.Dropout(0.5),
                                                nn.Linear(512, 64),
                                                nn.Dropout(0.5),
                                                nn.Linear(64, 2)))

    def forward(self, prot_input, datalengths):
        sharedembedding = self.ShardEncoder(prot_input, datalengths)

        task_outs = []
        for i in range(self.tasklen):
            task_embeddingi = self.tasks_fcs[i](sharedembedding)
            task_outs.append(task_embeddingi)
        return task_outs


def train(itrainfile, modelstoreapl):
    model = MTLModule(1024 + 1536, True, len(itrainfile))
    # model = MTLModule( 1536 , True, len(itrainfile))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 30
    # epochs = 25

    per_cls_weights = torch.FloatTensor([0.15, 0.85]).to(device)

    fcloss = FocalLoss_v2(alpha=per_cls_weights, gamma=2)

    model.train()

    file = read_data_file_trip(itrainfile, True)
    train_set = BioinformaticsDataset(file, itrainfile)
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
            task_outs = model(prot_xs.to(device), lengths.to(device))
            data_ys = data_ys.to(device)
            taskids = taskids.to(device)
            lengths = lengths.to('cpu')
            for i in range(len(itrainfile)):
                task_outs[i] = torch.nn.utils.rnn.pack_padded_sequence(task_outs[i], lengths, batch_first=True)
            data_ys = torch.nn.utils.rnn.pack_padded_sequence(data_ys, lengths, batch_first=True)

            taskids = torch.nn.utils.rnn.pack_padded_sequence(taskids, lengths, batch_first=True)

            loss_task = 0
            for i in range(len(itrainfile)):
                indexs = torch.nonzero(taskids.data == i).squeeze()
                pred = task_outs[i].data[indexs]
                lbs = data_ys.data[indexs]
                if lbs.shape[0] > 0:
                    fc = fcloss(pred, lbs)
                    loss_task += fc
            optimizer.zero_grad()
            loss_task.backward()
            optimizer.step()
            epoch_loss_train = epoch_loss_train + loss_task.item()
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


def test(itestfile, modelstoreapl, tsne_max_points=5000, tsne_outdir='tsne_plots'):
    # === 1) 模型 ===
    model = MTLModule(1536 + 1024, True, len(itestfile))
    # model = MTLModule( 1536, True, len(itestfile))
    model = model.to(device)
    state = torch.load(modelstoreapl, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # === 2) 输出目录 ===
    os.makedirs(tsne_outdir, exist_ok=True)

    # === 3) DataLoader ===
    file = read_data_file_trip(itestfile, False)
    test_set = BioinformaticsDataset(file, itestfile)
    test_load = DataLoader(dataset=test_set, batch_size=32,
                           num_workers=16, pin_memory=True, persistent_workers=True,
                           collate_fn=coll_paddding)

    print("==========================Test RESULT================================")

    # 评测容器
    predicted_probs = [[] for _ in range(len(itestfile))]
    labels_actual = [[] for _ in range(len(itestfile))]
    labels_predicted = [[] for _ in range(len(itestfile))]
    tmresult = {}

    # t-SNE 容器 - 只保留需要的特征
    all_plm_feats, all_final_feats = [], []
    all_labels = []

    with torch.no_grad():
        for prot_xs, data_ys, taskids, lengths in test_load:
            prot_xs_dev = prot_xs.to(device)
            lengths_dev = lengths.to(device)

            # 推理
            task_outs = model(prot_xs_dev, lengths_dev)

            # ====== 抓取 cache ======
            try:
                cache = model.ShardEncoder.cache  # 从Task_shared模块获取缓存
            except AttributeError as e:
                raise RuntimeError("找不到 model.ShardEncoder.cache，请检查 MTLModule 中 encoder 的属性名。") from e

            # 只获取需要的特征
            plm_batch = cache['plm'].detach().cpu().numpy()  # [B,L,Cplm] - PLM embeddings
            final_batch = cache['final'].detach().cpu().numpy()  # [B,L,128] - Multi-scale CNN features (final features)

            labels_batch = data_ys.cpu().numpy()  # [B,L]
            lens = lengths.cpu().numpy().tolist()

            # 按真实长度截断并收集
            for b, L in enumerate(lens):
                all_plm_feats.append(plm_batch[b, :L, :])
                all_final_feats.append(final_batch[b, :L, :])
                all_labels.append(labels_batch[b, :L])

            # ====== 原有的 pack + 评测 ======
            lengths_cpu = lengths.cpu()
            for i in range(len(itestfile)):
                task_outs[i] = torch.nn.utils.rnn.pack_padded_sequence(task_outs[i], lengths_cpu, batch_first=True)

            data_ys_packed = torch.nn.utils.rnn.pack_padded_sequence(data_ys.to(device), lengths_cpu, batch_first=True)
            taskids_packed = torch.nn.utils.rnn.pack_padded_sequence(taskids.to(device), lengths_cpu, batch_first=True)

            for i in range(len(itestfile)):
                indexs = torch.nonzero(taskids_packed.data == i).squeeze()
                if indexs.numel() == 0:
                    continue
                task_pred = task_outs[i].data[indexs]
                lbs = data_ys_packed.data[indexs].to('cpu')
                task_pred = F.softmax(task_pred, dim=1).to('cpu')
                predicted_probs[i].extend(task_pred[:, 1])
                labels_actual[i].extend(lbs)
                labels_predicted[i].extend(torch.argmax(task_pred, dim=1))

    # === 4) 打印指标 ===
    itask_names = itestfile
    for id_, task_name in enumerate(itask_names):
        sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult(
            task_name, labels_actual[id_], predicted_probs[id_], labels_predicted[id_]
        )
        tmresult[task_name] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]

    # === 5) 采样函数（每个表示单独采样并返回匹配的 y） ===
    def _stack_and_sample(feat_list, label_list, max_points=5000, seed=42):
        X = np.vstack(feat_list)  # [N, D]
        y = np.concatenate(label_list)  # [N]
        if X.shape[0] > max_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=max_points, replace=False)
            X, y = X[idx], y[idx]
        return X, y.astype(int, copy=False)

    X_plm, y_plm = _stack_and_sample(all_plm_feats, all_labels, tsne_max_points)
    X_final, y_final = _stack_and_sample(all_final_feats, all_labels, tsne_max_points)

    # === 6) t-SNE 绘图（带异常捕获和样本数检查） ===
    def _tsne_plot(X, y, title, outfile, perplexity=30, lr=200, seed=42):
        try:
            n = X.shape[0]
            if n < 50:
                print(f"[t-SNE] 跳过 {title}：样本过少 (n={n})")
                return
            if perplexity >= n:
                perplexity = max(5, min(30, n // 3))
                print(f"[t-SNE] {title} 的 perplexity 自动调整为 {perplexity} (n={n})")

            tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr,
                        init='pca', random_state=seed)
            Z = tsne.fit_transform(X)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 6))
            y = np.asarray(y, dtype=int)
            # 防止类别不是0/1导致索引异常
            pos_mask = (y == 1)
            neg_mask = (y == 0)
            plt.scatter(Z[neg_mask, 0], Z[neg_mask, 1], s=5, alpha=0.35, c='#1f77b4', label='Non-binding')
            plt.scatter(Z[pos_mask, 0], Z[pos_mask, 1], s=8, alpha=0.65, c='#DA3C5D', label='Binding')
            plt.title(title)
            plt.legend(markerscale=3)
            plt.tight_layout()

            # 获取文件名主干
            stem = os.path.splitext(outfile)[0]

            # 保存PNG格式
            png_save_path = os.path.join(tsne_outdir, f"{stem}.png")
            plt.savefig(png_save_path, dpi=300, bbox_inches='tight')
            print(f"[t-SNE] saved as PNG: {png_save_path}")

            # 保存EPS格式
            eps_save_path = os.path.join(tsne_outdir, f"{stem}.eps")
            plt.savefig(eps_save_path, format="eps", bbox_inches='tight')
            print(f"[t-SNE] saved as EPS: {eps_save_path}")

            plt.close()
            print(f"[t-SNE] saved as EPS: {eps_save_path}")

        except Exception as e:
            print(f"[t-SNE] 绘制 {title} 失败：{e}")

    # 只绘制两个t-SNE图
    _tsne_plot(X_plm, y_plm, 't-SNE of PLM Embeddings', 'tsne_plm.png')
    _tsne_plot(X_final, y_final, 't-SNE of Multi-scale CNN Features', 'tsne_cnn.png')

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
    # torch.cuda.set_device(1)
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")

    trainfiles1 = ['D:/fengzhen/1NucGMTL-main/DataSet/SMB/SMB_Train.txt']
    testfiles1 = ['D:/fengzhen/1NucGMTL-main/DataSet/SMB/SMB_Test.txt']

    task1 = ['ADP', 'ATP', 'AMP']  # 能量代谢组
    task2 = ['GDP', 'ADP', 'ATP', 'GTP']
    task3 = ['GTP', 'GDP', 'ATP']

    task4 = ['SMB']

    tasks = task4  # change taskid for different task

    circle = 5
    a = str(datetime.datetime.now())
    a = a.replace(':', '_')

    totalkv = {task: [] for task in tasks}
    storename = '_'.join(p for p in tasks)
    print(storename)
    for i in range(circle):

        storeapl = 'RGMTL1/Result_' + storename + '_' + str(i) + '_' + a + '.pkl'
        train(tasks, storeapl)
        tmresult = test(tasks, storeapl)
        for task in tasks:
            totalkv[task].append(tmresult[task])
        torch.cuda.empty_cache()

    with open('RGMTL1/Result_' + storename + '_' + a + '.txt', 'w') as f:
        for nuc in tasks:
            np.savetxt(f, totalkv[nuc], delimiter=',', footer='Above is  record ' + nuc, fmt='%s')
            m = np.mean(totalkv[nuc], axis=0)
            np.savetxt(f, [m], delimiter=',', footer='----------Above is AVG -------' + nuc, fmt='%s')