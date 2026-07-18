import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from LossFunction.focalLoss import FocalLoss_v2
import torch.multiprocessing
# from losses import TripletCenterLoss, FocalLoss
import datetime
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def read_data_file_trip(tasknames, istraining):
    results = []
    for taskname in tasknames:
        if istraining:
            # f = open('D:/fengzhen/NucGMTL-main/DataSet/Nuc1892/Train/'+taskname+'30_Train.txt')
            f = open('D:/fengzhen/1NucGMTL-main/DataSet/SMB/' + taskname + '_Train.txt')
        else:
            # f = open('D:/fengzhen/NucGMTL-main/DataSet/Nuc1892/Test/' + taskname + '30_Test.txt')
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
        # esm_embedding1280 prot_embedding  esm_embedding2560 msa_embedding
        # df0 = pd.read_csv('D:/fengzhen/NucGMTL-main/embedding/prot_embedding/' + filename + '.data', header=None)
        # df0 = pd.read_csv('D:/fengzhen/embedding/prot_embedding_DNA_RNA/' + filename + '.data', header=None)
        df0 = pd.read_csv('D:/fengzhen/1embedding/prostT5_embedding_SMB/' + filename + '.data', header=None)
        prot0 = df0.values.astype(float).tolist()
        prot0 = torch.tensor(prot0)     # 改

        # print("prot0 shape:", prot0.shape)  # 例如 torch.Size([batch, 1023, features])

        # df1 = pd.read_csv('D:/fengzhen/NucGMTL-main/embedding/esm_embedding1280/' + filename + '.data', header=None)
        # df1 = pd.read_csv('D:/fengzhen/embedding/esm_embedding1280_DNA_RNA/' + filename + '.data', header=None)

        df1 = pd.read_csv('D:/fengzhen/1embedding/Ankh_embedding_SMB/' + filename + '.data', header=None)    # 需要
        prot1 = df1.values.astype(float).tolist()     # 需要
        prot1 = torch.tensor(prot1)     # 需要

        # print("prot1 shape:", prot1.shape)    # 例如 torch.Size([batch, 1038, features])

        min_len = min(prot0.size(0), prot1.size(0))      # 需要
        prot = torch.cat((prot0[:min_len], prot1[:min_len]), dim=1)     # 需要


        # combine two plms
        # prot = torch.cat((prot0, prot1), dim=1)  # prot0#  需要
        # prot = prot0  # prot0#  需要

        # df2= pd.read_csv('D:/fengzhen/embedding/prot_embedding_DNA_RNA/'+  filename+'.label', header=None)
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
        atten=self.masked_softmax((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact,seqlengths)
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


class Task_shared(nn.Module):
    def __init__(self,inputdim):
        super(Task_shared,self).__init__()
        self.inputdim=inputdim


        self.ms1cnn1=nn.Conv1d(self.inputdim,512,1,padding='same')
        self.ms1cnn2=nn.Conv1d(512,256,1,padding='same')
        self.ms1cnn3=nn.Conv1d(256,128,1,padding='same')


        self.ms2cnn1 = nn.Conv1d(self.inputdim, 512, 3, padding='same')
        self.ms2cnn2 = nn.Conv1d(512, 256, 3, padding='same')
        self.ms2cnn3 = nn.Conv1d(256, 128, 3, padding='same')


        self.ms3cnn1 = nn.Conv1d(self.inputdim, 512, 5, padding='same')
        self.ms3cnn2 = nn.Conv1d(512, 256, 5, padding='same')
        self.ms3cnn3 = nn.Conv1d(256, 128, 5, padding='same')

        self.relu=nn.ReLU(True)

        self.AttentionModel1 = AttentionModel(512, 128)
        self.AttentionModel2 = AttentionModel(256, 128)
        self.AttentionModel3 = AttentionModel(128, 128)



    def forward(self,prot_input,seqlengths):

        prot_input_share = prot_input.permute(0, 2, 1)

        m1=self.relu(self.ms1cnn1(prot_input_share))
        m2 = self.relu(self.ms2cnn1(prot_input_share))
        m3 = self.relu(self.ms3cnn1(prot_input_share))

        att=m1+m2+m3
        att=att.permute(0,2,1)
        s1=self.AttentionModel1(att, seqlengths)

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

        mscnn=m1+m2+m3
        mscnn=mscnn.permute(0,2,1)
        s=s1+s2+s3

        return mscnn+s


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
    model = MTLModule( 1024 + 1536, True, len(itrainfile))
    # model = MTLModule(1280, True, len(itrainfile))
    # model = MTLModule(1024,True,len(itrainfile))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # epochs = 30
    epochs = 10
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


def test(itestfile, modelstoreapl):
    # model = MTLModule(1280, False, len(itestfile))
    model = MTLModule(1536 + 1024, True, len(itestfile))
    # model = MTLModule(1024,False,len(itestfile))
    model = model.to(device)
    model.load_state_dict(torch.load(modelstoreapl))
    model.eval()
    tmresult = {}

    file = read_data_file_trip(itestfile, False)
    test_set = BioinformaticsDataset(file, itestfile)
    test_load = DataLoader(dataset=test_set, batch_size=32,
                           num_workers=16, pin_memory=True, persistent_workers=True, collate_fn=coll_paddding)

    print("==========================Test RESULT================================")

    predicted_probs = [[] for i in range(len(itestfile))]
    labels_actual = [[] for i in range(len(itestfile))]
    labels_predicted = [[] for i in range(len(itestfile))]

    with torch.no_grad():
        for prot_xs, data_ys, taskids, lengths in test_load:
            task_outs = model(prot_xs.to(device), lengths.to(device))
            for i in range(len(itestfile)):
                task_outs[i] = torch.nn.utils.rnn.pack_padded_sequence(task_outs[i], lengths.to('cpu'),
                                                                       batch_first=True)

            data_ys = torch.nn.utils.rnn.pack_padded_sequence(data_ys, lengths, batch_first=True)
            taskids = torch.nn.utils.rnn.pack_padded_sequence(taskids, lengths, batch_first=True)

            for i in range(len(itestfile)):
                indexs = torch.nonzero(taskids.data == i).squeeze()
                task_pred = task_outs[i].data[indexs]
                lbs = data_ys.data[indexs]
                task_pred = torch.nn.functional.softmax(task_pred, dim=1)
                task_pred = task_pred.to('cpu')
                if lbs.shape[0] > 0:
                    predicted_probs[i].extend(task_pred[:, 1])
                    labels_actual[i].extend(lbs)
                    labels_predicted[i].extend(torch.argmax(task_pred, dim=1))

        itask_names = itestfile
        itaskid = [i for i in range(len(itask_names))]
        for id, task_name in zip(itaskid, itask_names):
            sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult(task_name, labels_actual[id],
                                                                                     predicted_probs[id],
                                                                                     labels_predicted[id])
            tmresult[task_name] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]
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

    # trainfiles1 = ['../DataSet/Train/ATP30_Train.txt',
    #                '../DataSet/Train/ADP30_Train.txt',
    #                ]
    # trainfiles1 = ['D:/fengzhen/NucGMTL-main/DataSet/DNA/Train/DNA_Train.txt',    # DNA和RNA联合训练
    #                'D:/fengzhen/NucGMTL-main/DataSet/RNA/Train/RNA_Train.txt']
    trainfiles1 = ['D:/fengzhen/1NucGMTL-main/DataSet/SMB/SMB_Train.txt']
    # testfiles1 = ['../DataSet/Test/ATP30_Test.txt',
    #               '../DataSet/Test/ADP30_Test.txt',
    #               ]
    # testfiles1 = ['D:/fengzhen/NucGMTL-main/DataSet/DNA/Test/DNA_Test.txt',     # DNA和RNA联合训练
    #               'D:/fengzhen/NucGMTL-main/DataSet/RNA/Test/RNA_Test.txt']
    testfiles1 = ['D:/fengzhen/1NucGMTL-main/DataSet/SMB/SMB_Test.txt']

    task1 = ['ADP', 'ATP', 'AMP']  # 能量代谢组
    task2 = ['GDP', 'ADP', 'ATP', 'GTP']
    task3 = ['GTP', 'GDP', 'ATP']

    task4 = ['SMB']

    tasks = task4  # change taskid for different task

    # circle = 5
    circle = 1
    # circle=2
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


    import os, numpy as np, torch
    import matplotlib.pyplot as plt
    import shap

    out_dir = 'D:/fengzhen/1NucGMTL-main/shap_subset_out'

    # os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # ==== 0) 基本参数 ====
    seed = 42
    n_bg = 20  # 背景样本数（用于估计期望）
    n_explain = 8  # 需要解释的样本数
    Lcap = 256  # 统一裁剪到的残基长度上限（会与可用最短长度再取 min）
    task_idx = 0  # 解释第几个任务/输出
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ==== 1) 构建并加载模型 ====
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTLModule(1024 + 1536, True, len(tasks)).to(dev)
    model_path = r'D:/fengzhen/1NucGMTL-main/RGMTL1/Result_SMB_2_2025-09-05 15_42_25.464198.pkl'

    state = torch.load(model_path, map_location=dev)
    # 兼容两种保存方式
    state_dict = state.get("state_dict", state)
    # 若保存时带有前缀比如 "model.", 可在此做 strip：
    # state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # ==== 2) 抽取少量固定长度的数据 ====
    names = read_data_file_trip(tasks, istraining=False)
    ds = BioinformaticsDataset(names, tasks)

    assert len(ds) > 0, "测试集为空，无法进行 SHAP 可视化。"

    rng = np.random.default_rng(seed)
    pick_n = min(len(ds), n_bg + n_explain)
    pick_idx = rng.choice(len(ds), size=pick_n, replace=False)

    # 先做一次“长度探测”，确定统一的 Lcap（不越界）
    lengths = []
    first_x, _, _ = ds[pick_idx[0]]
    C = first_x.shape[1]
    for idx in pick_idx:
        x_i, _, _ = ds[idx]
        lengths.append(x_i.shape[0])
    Lcap_eff = min(Lcap, min(lengths))  # 实际裁剪长度

    X_list, y_list = [], []
    for idx in pick_idx:
        x, y, _ = ds[idx]  # x: [L, C], y: [L]
        X_list.append(x[:Lcap_eff, :])  # 裁剪到统一长度
        y_list.append(y[:Lcap_eff])

    X = torch.stack(X_list, dim=0)  # [B, Lcap_eff, C]
    y = torch.stack(y_list, dim=0)  # [B, Lcap_eff]

    B = X.size(0)
    # 重新计算 bg/ex 的切分，防越界
    bg_n = min(n_bg, B)
    ex_n = min(n_explain, max(0, B - bg_n))
    assert ex_n > 0, "解释样本数量为 0：请增大 n_explain 或减少 n_bg，或保证测试集样本数充足。"

    bg = X[:bg_n].clone()
    ex = X[bg_n:bg_n + ex_n].clone()
    ey = y[bg_n:bg_n + ex_n].clone()


    # ==== 3) 前向包装器：固定 lengths = Lcap_eff ====
    class ForwardForSHAP(torch.nn.Module):
        def __init__(self, base_model, task_idx=0):
            super().__init__()
            self.model = base_model
            self.task_idx = task_idx

        def forward(self, X_in):
            # X_in: [B, Lcap_eff, C]
            X_in = X_in.to(dev)
            lengths = torch.full((X_in.size(0),), X_in.size(1),
                                 dtype=torch.long, device=dev)  # 全等长
            outs = self.model(X_in, lengths)  # list of [B, Lcap_eff, 2]
            # 返回阳性类 logit（或概率，保持一致即可）
            return outs[self.task_idx][..., 1]  # [B, Lcap_eff]


    fwrap = ForwardForSHAP(model, task_idx=task_idx)

    # ==== 4) 先试 DeepExplainer，失败则回退 KernelExplainer ====
    bg_flat = bg.reshape(bg.size(0), -1).cpu().numpy()  # [B_bg, Lcap_eff*C]
    ex_flat = ex.reshape(ex.size(0), -1).cpu().numpy()  # [B_ex, Lcap_eff*C]


    # 标量输出：将每个样本的 [Lcap_eff] 位置分数做均值（或 sum）
    def f_np_flat_scalar(x_flat):
        x = torch.tensor(x_flat, dtype=torch.float32, device=dev).view(-1, Lcap_eff, C)
        with torch.no_grad():
            out = fwrap(x)  # [B, Lcap_eff]
            out_scalar = out.mean(dim=1)  # [B] 也可改为 .sum(dim=1)
        return out_scalar.detach().cpu().numpy()


    explainer = shap.KernelExplainer(f_np_flat_scalar, bg_flat)
    # nsamples 可根据速度/稳定性调节（更小更快）
    shap_vals_flat = explainer.shap_values(ex_flat, nsamples=100)

    # 兼容不同 shap 版本的返回类型
    if isinstance(shap_vals_flat, list):
        shap_vals_flat = shap_vals_flat[0]

    shap_vals = np.asarray(shap_vals_flat, dtype=np.float32).reshape(ex.size(0), Lcap_eff, C)  # [B_ex, Lcap_eff, C]

    # ==== 5) 美化可视化：残基级 |SHAP| 热图 + PLM 贡献（升级版） ====
    X_np = ex.cpu().numpy()
    y_np = ey.cpu().numpy()

    # 统一一些美观参数（白底、细网格、柔和文字）
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    # 1) 残基层面重要性（按通道聚合）：统一色标避免不同样本间难以比较
    imp_r = np.abs(shap_vals).sum(axis=2)  # [B_ex, L, C] → [B_ex, L]
    # 用全体样本的 99 分位作为 vmax，抑制极端值，使颜色层次更清楚
    vmax = float(np.percentile(imp_r, 99))
    vmin = 0.0


    # 简单的 Top-K 峰选（带非极大值抑制），用于在图上标出“模型最关心的位置”
    def topk_nms(arr, k=8, radius=2):
        # arr: [L]
        idx_sorted = np.argsort(arr)[::-1]
        selected = []
        for j in idx_sorted:
            if all(abs(j - s) > radius for s in selected):
                selected.append(j)
                if len(selected) >= k:
                    break
        return np.array(sorted(selected))


    show_k = min(4, imp_r.shape[0])
    for i in range(show_k):
        fig, ax = plt.subplots(figsize=(11, 2.6))
        # 采用更清晰的 colormap（如 'viridis' 或 'magma'）；白底更好分辨
        im = ax.imshow(imp_r[i][np.newaxis, :], aspect='auto',
                       cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')

        # 让横轴更易读：每隔 10 个刻度标注一次
        L_here = imp_r[i].shape[0]
        xticks = np.arange(0, L_here, max(1, L_here // 10))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])
        ax.set_yticks([])

        # 在真阳性位置画半透明青色带，比细竖线更显眼
        pos_idx = np.where(y_np[i] == 1)[0]
        for p in pos_idx:
            # ax.axvspan(p - 0.5, p + 0.5, color='cyan', alpha=0.25, linewidth=0)
            ax.axvspan(p - 0.5, p + 0.5, color='red', alpha=0.25, linewidth=0)

        # 标出 Top-K 峰（洋红色竖线），帮助快速定位最重要的位置
        peaks = topk_nms(imp_r[i], k=8, radius=2)
        for p in peaks:
            # ax.axvline(p, color='magenta', alpha=0.9, linewidth=1.0)
            ax.axvline(p, color='green', alpha=0.9, linewidth=1.0)

        # 可选：在顶部加一个微型刻度条，增强对比阅读（不是必须）
        ax.set_xlabel('Residue index')
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label('|SHAP| per residue')

        # 标题里直接告诉读者这张图的关键信息
        ax.set_title(f"Residue-level importance (sample {i}) — cyan: positives, magenta: top peaks")

        # 轻微网格（每 10 个残基虚线），不抢眼
        for g in range(0, L_here, 10):
            ax.axvline(g - 0.5, color='k', alpha=0.06, linewidth=0.7, linestyle='--')

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"subset_shap_residue_heat_beautified_{i}.png"))
        plt.close(fig)


    # 2) PLM 贡献柱状图：用“百分比 + 数值标注”，更直观比较 ProstT5 vs Ankh
    def bar_label(ax, bars, fmt='{:.1f}%'):
        for b in bars:
            h = b.get_height()
            ax.annotate(fmt.format(h),
                        xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)


    # 每个样本一张 + 汇总一张（跨样本平均）
    if C == (1024 + 1536):
        palette = ['#4C78A8', '#F58518']  # ProstT5 / Ankh
        # 汇总图（跨样本）
        sv_abs_all = np.abs(shap_vals)  # [B, L, C]
        contrib_prost_all = sv_abs_all[:, :, :1024].sum(axis=(1, 2))  # [B]
        contrib_ankh_all = sv_abs_all[:, :, 1024:].sum(axis=(1, 2))  # [B]
        total_all = contrib_prost_all + contrib_ankh_all + 1e-12
        mean_pct_prost = float((100.0 * contrib_prost_all / total_all).mean())
        mean_pct_ankh = float((100.0 * contrib_ankh_all / total_all).mean())

        # fig, ax = plt.subplots(figsize=(4.6, 3.2))
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        bars = ax.bar(['ProstT5', 'Ankh'], [mean_pct_prost, mean_pct_ankh], color=palette, width=0.6)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Percent of total |SHAP| (%)')
        ax.set_title('PLM contribution (mean across explained samples)')
        bar_label(ax, bars)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'subset_shap_plm_contrib_overall_pct_beautified.png'))
        plt.close(fig)

        # 单样本图（前 show_k 个样本）
        for i in range(show_k):
            sv = np.abs(shap_vals[i])  # [L, C]
            imp_prost = sv[:, :1024].sum()
            imp_ankh = sv[:, 1024:].sum()
            total = imp_prost + imp_ankh + 1e-12
            vals_pct = [100.0 * imp_prost / total, 100.0 * imp_ankh / total]

            fig, ax = plt.subplots(figsize=(4.2, 3.0))
            bars = ax.bar(['ProstT5', 'Ankh'], vals_pct, color=palette, width=0.6)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Percent of total |SHAP| (%)')
            ax.set_title(f"PLM contribution (sample {i})")
            bar_label(ax, bars)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"subset_shap_plm_contrib_pct_beautified_{i}.png"))
            plt.close(fig)

    print(f"[SHAP] Done. Beautified figures saved to: {out_dir}")

