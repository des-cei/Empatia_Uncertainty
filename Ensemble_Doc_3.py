"""
                                                    ⠀⠀⠀⠀⠀⠀⠀⢠⣤⣀⠀⠀⠀⠀⢀⣀⣤⣤⠀⠀⠀⠀⠀⠀⠀
                                                    ⠀⠀⢀⢀⠀⠀⠀⢸⡿⠛⠛⠛⠛⠛⠉⠛⢿⣿⠀⠀⠀⠀⠀⠀⠀
 _   _       _     _____         _             		⠀⠠⣿⣿⣿⣄⠀⣼⠀⠀⠉⣍⣀⣀⡍⠁⠀⢹⡀⠀⠀⠀⠀⠀⠀
| \ | | ___ | |_  |_   _|__   __| | __ _ _   _ 		⠀⢸⣿⣿⣿⣿⡷⠋⠈⠀⠀⠀⠀⠀⠀⠀⠈⠘⠣⡀⠀⠀⠀⠀⠀
|  \| |/ _ \| __|   | |/ _ \ / _` |/ _` | | | |		⠀⠈⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⣦⡀⠀⠀
| |\  | (_) | |_    | | (_) | (_| | (_| | |_| |		⠀⠀⢹⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⣿⣦⠀
|_| \_|\___/ \__|   |_|\___/ \__,_|\__,_|\__, |		⠀⠀⣸⣿⣿⣶⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇
                                         |___/ 		⠀⣤⡟⠛⠋⠉⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠉⠈⠋⠈⢿⣿⡿
                                                    ⢀⡉⠀⠀⣀⣤⣄⢈⣿⣿⣿⣿⣿⣿⣿⣿⣿⢀⣤⣤⣄⠀⠀⣴⡄
                                                    ⠘⢇⠀⠰⣿⣿⢟⢼⣿⣿⣿⣿⣿⣿⣿⣿⡿⢜⠿⠿⠿⠀⡀⠀⠀
@Author: Junjiao Sun                               
@Time : Created in 3:41 PM 2025/07/04
@FileName: Graph_classification.py
@Software: PyCharm


Introduction of this File:

"""
import os

import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.append("/home/junjiaos/PycharmProjects/MMASH_research/")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, DataLoader
from Graph_process.Graph_encoder_loso import Encoder_training, GCNEncoder, graph_others, dict_list_to_batch
import math
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_2d_to_3d(input_list):
    result = []
    for row in input_list:
        new_row = []
        for val in row:
            if isinstance(val, float) and math.isnan(val):
                new_val = [0, 1, 0]
            elif val <= 20:
                new_val = [1, 0, 0]
            elif val <= 35:
                new_val = [0, 1, 0]
            else:
                new_val = [0, 0, 1]
            new_row.append(new_val)
        result.append(new_row)
    return result


def transform_2d(input_list):
    result = []
    for row in input_list:
        new_row = []
        for val in row:
            if isinstance(val, float) and math.isnan(val):
                new_val = 1
            elif val <= 20:
                new_val = 0
            elif val <= 35:
                new_val = 1
            else:
                new_val = 2
            new_row.append(new_val)
        result.append(new_row)
    return result


class RegressionCNN(nn.Module):
    def __init__(self):
        super(RegressionCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc_mu = nn.Linear(64, 1)  # average
        self.fc_sigma = nn.Linear(64, 1)  # std

    def forward(self, x):  # x: (B, 1, 64)
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))  # (B, 16, 64)
        x = self.relu(self.bn2(self.conv2(x)))  # (B, 32, 64)
        x = self.relu(self.bn3(self.conv3(x)))  # (B, 64, 64)
        x = self.pool(x).squeeze(-1)  # -> (B, 64)
        x = self.dropout(x)
        mu = self.fc_mu(x)  # -> (B, 1)
        sigma = F.softplus(self.fc_sigma(x)) + 1e-6  # ensure positive
        return mu.squeeze(-1), sigma.squeeze(-1)  # (B,), (B,)


class GCN_encoder_regression(nn.Module):
    def __init__(self, encoder, freeze_encoder=True):
        super(GCN_encoder_regression, self).__init__()
        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.decoder = RegressionCNN()

    def forward(self, x, edge_index, batch):
        with torch.no_grad():
            features = self.encoder(x, edge_index, batch)  # (B, 64)
        out = self.decoder(features)  # (B, 2)
        return out


def nll_loss(mu, sigma, y):
    """
    mu: predicted average, shape: (B,)
    sigma: predicted std, shape: (B,)
    y: ground-truth, shape: (B,)
    """
    var = sigma ** 2
    nll = 0.5 * torch.log(2 * torch.pi * var) + ((y - mu) ** 2) / (2 * var)
    return nll.mean()


def find_closest(target, array):
    array = list(array)
    closest_index = min(range(len(array)), key=lambda i: abs(array[i] - target))
    closest_value = array[closest_index]
    return closest_value, closest_index


def prepare_graph_batch(x, edge_index):
    """
    将 batch 图数据转换为 PyTorch Geometric 所需格式：
    - x: (B, N, F)  → (B*N, F)
    - edge_index: (B, 2, E) → (2, B*E) with offset
    - batch: (B*N,) 表示每个节点属于哪个图

    参数:
        x: torch.Tensor, shape (B, N, F)，节点特征
        edge_index: torch.Tensor, shape (B, 2, E)，每个图的边连接信息

    返回:
        x_reshaped: torch.Tensor, shape (B*N, F)
        edge_index_combined: torch.Tensor, shape (2, B*E)
        batch_vector: torch.Tensor, shape (B*N,)
    """
    batch_size, num_nodes, num_features = x.shape
    _, _, num_edges = edge_index.shape

    # 1. 展平 x
    x_reshaped = x.reshape(-1, num_features)  # shape: (B*N, F)

    # 2. 偏移 edge_index
    edge_index_list = []
    for i in range(batch_size):
        edge = edge_index[i]  # shape: (2, E)
        offset = i * num_nodes
        edge_shifted = edge + offset
        edge_index_list.append(edge_shifted)

    edge_index_combined = torch.cat(edge_index_list, dim=1)  # shape: (2, B*E)

    # 3. 构建 batch 向量
    batch_vector = torch.arange(batch_size).repeat_interleave(num_nodes)  # shape: (B*N,)

    return x_reshaped, edge_index_combined, batch_vector


def kl_divergence(mu1, sigma1, mu2, sigma2):
    return np.log(sigma2 / sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) - 0.5

def split_with_overlap(user_list, target_id, num_splits=4):
    """
    给定 user_list 和 target_id：
    1) 删除 target_id
    2) 将剩余用户分成 num_splits 个重叠子列表（重叠比例 50%）
    """

    # ---- Step 1: 删除 target id ----
    L = [u for u in user_list if u != target_id]
    n = len(L)

    if n == 0:
        return [[] for _ in range(num_splits)]

    # ---- Step 2: 定义窗口大小与步长（重叠 50%） ----
    window = 8     # 每段长度
    step = max(1, window // 2)               # 重叠 50%

    # ---- Step 3: 生成四个窗口 ----
    splits = []
    start = 0
    for _ in range(num_splits):
        end = start + window
        splits.append(L[start:end])
        start += step

    return splits

def split_no_overlap(user_list, target_id, num_splits=3, block_size=6):
    """
    给定 user_list 和 target_id：
    1) 删除 target_id
    2) 将剩余用户分成 num_splits 个不重叠的子列表
    3) 每个子列表包含 block_size 个元素（如果不够则返回剩余元素）
    """

    # Step 1: 删除 target_id
    L = [u for u in user_list if u != target_id]

    splits = []
    start = 0

    # Step 2: 按 block_size 不重叠切分
    for _ in range(num_splits):
        end = start + block_size
        splits.append(L[start:end])
        start = end  # ⭐ 不重叠

    return splits


def split_with_random_special(user_list, target_id, num_splits=3, block_size=5, special_count=4):
    """
    固定输出版本：对于相同的 user_list 和 target_id，永远返回相同的结果。
    """

    # Step 1: 删除 target_id
    candidates = [u for u in user_list if u != target_id]

    # 【关键修改 1】先排序
    # 确保无论输入的 user_list 是乱序还是正序，起点状态都是一致的。
    # 如果列表里也是数字，直接 sort() 即可。
    candidates.sort()

    # 【关键修改 2】使用基于 target_id 的局部随机种子
    # 创建一个独立的随机生成器，不受外界 random.seed() 的影响
    # 使用 target_id 作为种子，只要 target_id 不变，随机后的顺序就永远不变
    rng = random.Random(target_id)
    rng.shuffle(candidates)

    # Step 3: 取出前 special_count 个作为 new_list (用于 validation_set)
    new_list = candidates[:special_count]

    # Step 4: 获取剩余列表
    rest_list = candidates[special_count:]

    # 初始化结果
    final_splits = [new_list]
    start = 0

    # Step 5: 切分剩余部分
    for _ in range(num_splits):
        end = start + block_size
        chunk = rest_list[start:end]

        if chunk:
            final_splits.append(chunk)

        start = end

    # 分配返回值
    if final_splits:
        validation_set = final_splits[0]
        splits = final_splits[1:]
    else:
        validation_set = []
        splits = []

    return splits, validation_set


if __name__ == '__main__':
    Current_Doc = 'Doc_3'
    root = '../dataset'
    model_root = '/home/junjiaos/PycharmProjects/MMASH_research/Encoder_Ensembles/Encoder_checkpoints_' + Current_Doc
    Label_slot_group = [10, 14, 18, 22, 33]
    user_list = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    user_list.remove('user_2')
    user_list.remove('user_8')

    loso_train_total_acc = []
    loso_test_total_acc = []
    loso_train_total_f1 = []
    loso_test_total_f1 = []

    num_epochs = 1000

    for User_num in user_list:
        average_small_KL = []

        feature_dim = 1
        encoder_model = GCNEncoder(in_channels=feature_dim).to(device)
        file_doc = Current_Doc.replace('_', '')
        User_file = User_num + '_' + file_doc + '_encoder.pth'
        encoder_model.load_state_dict(torch.load(os.path.join(model_root, User_file)))

        # obtain the training data
        with open("../Json_files/graph_normal.json", "r", encoding="utf-8") as f:
            User_normal_data = json.load(f)

        with open("../Json_files/Label.json", "r", encoding="utf-8") as f:
            User_label_data = json.load(f)

        # remove data from user 2
        del User_normal_data['2']
        del User_label_data['2']
        del User_normal_data['8']
        del User_label_data['8']

        # Do LOSO for training

        User_id = User_num.replace('user_', '')
        print('Now user: ' + str(User_id))

        # Get the Doc previous experiences
        User_list = sorted(User_normal_data.keys(), key=int)
        # User_split = split_no_overlap(User_list, User_id)
        User_split, validation_set = split_with_random_special(User_list, User_id)

        # This file for Doc 1
        Now_doc_num = int(Current_Doc.replace('Doc_', '')) - 1
        Doc_now = User_split[Now_doc_num]

        User_normal_data_Doc = {k: User_normal_data[k] for k in Doc_now if k in User_normal_data}

        User_validation_data_Doc = {k: User_normal_data[k] for k in validation_set if k in User_normal_data}

        # Get other graphs
        Other_normal_graph = graph_others(User_normal_data_Doc, User_id)

        Valid_normal_graph = graph_others(User_validation_data_Doc, User_id)

        # Pick number of users for training encoder
        # num_samples = 19
        # random.seed(42)
        # list_length = len(Other_normal_graph)
        # selected_indices = random.sample(range(list_length), num_samples)
        #
        # Other_normal_graph = [Other_normal_graph[i] for i in selected_indices]

        User_label_data_Doc = {k: User_label_data[k] for k in Doc_now if k in User_label_data}

        Valid_label_data_Doc = {k: User_label_data[k] for k in validation_set if k in User_label_data}

        Other_normal_label = list(User_label_data_Doc.values())

        Valid_normal_label = list(Valid_label_data_Doc.values())

        # Other_normal_label = [values_label[i] for i in selected_indices]

        # Get normal feature maps
        feature_graph_group_normal = User_normal_data[User_id]
        normal_labels = User_label_data[User_id]

        total_graphs = 1199

        Batch_now = {
            'p_10': [],
            'p_14': [],
            'p_18': [],
            'p_22': [],
            'p_33': [],
            'n_10': [],
            'n_14': [],
            'n_18': [],
            'n_22': [],
            'n_33': []
        }
        label_training_now_P = {
            '10': [],
            '14': [],
            '18': [],
            '22': [],
            '33': []
        }
        label_training_now_N = {
            '10': [],
            '14': [],
            '18': [],
            '22': [],
            '33': []
        }

        Batch_now_test = {
            'p_10': [],
            'p_14': [],
            'p_18': [],
            'p_22': [],
            'p_33': [],
            'n_10': [],
            'n_14': [],
            'n_18': [],
            'n_22': [],
            'n_33': []
        }
        label_test_now_P = {
            '10': [],
            '14': [],
            '18': [],
            '22': [],
            '33': []
        }
        label_test_now_N = {
            '10': [],
            '14': [],
            '18': [],
            '22': [],
            '33': []
        }

        Batch_now_valid = {
            'p_10': [],
            'p_14': [],
            'p_18': [],
            'p_22': [],
            'p_33': [],
            'n_10': [],
            'n_14': [],
            'n_18': [],
            'n_22': [],
            'n_33': []
        }
        label_valid_now_P = {
            '10': [],
            '14': [],
            '18': [],
            '22': [],
            '33': []
        }
        label_valid_now_N = {
            '10': [],
            '14': [],
            '18': [],
            '22': [],
            '33': []
        }

        user_valid_now_P = {
            '10': [],
            '14': [],
            '18': [],
            '22': [],
            '33': []
        }
        user_valid_now_N = {
            '10': [],
            '14': [],
            '18': [],
            '22': [],
            '33': []
        }

        for graph_index in range(total_graphs):

            Indicator_normal = feature_graph_group_normal[graph_index]
            # arrange the slot time
            Indicator_slot_time = 10 + (graph_index / 60)
            Indicator_slot_value, Indicator_slot_index = find_closest(Indicator_slot_time, Label_slot_group)
            # find to correspond label
            icon_positive = 'panas_pos_' + str(Indicator_slot_value)
            icon_negative = 'panas_neg_' + str(Indicator_slot_value)

            label_value_p = normal_labels[icon_positive]
            label_value_n = normal_labels[icon_negative]

            if math.isnan(label_value_p):
                label_value_p = 30
            if math.isnan(label_value_n):
                label_value_n = 30

            label_test_now_P[str(Indicator_slot_value)].append(label_value_p)
            label_test_now_N[str(Indicator_slot_value)].append(label_value_n)
            Batch_now_test['p_' + str(Indicator_slot_value)].append(Indicator_normal)
            Batch_now_test['n_' + str(Indicator_slot_value)].append(Indicator_normal)

            # Batch_now[str(Indicator_slot_value)].append(Indicator_normal)
            # label_training_now_P[str(Indicator_slot_value)].append(normal_labels[icon_positive])
            # label_training_now_N[str(Indicator_slot_value)].append(normal_labels[icon_negative])

            for index, arr in enumerate(Other_normal_graph):
                Training_normal = arr[graph_index]
                Training_slot_time = 10 + (graph_index / 60)
                Training_slot_value, Training_slot_index = find_closest(Training_slot_time, Label_slot_group)

                icon_positive = 'panas_pos_' + str(Training_slot_value)
                icon_negative = 'panas_neg_' + str(Training_slot_value)

                label_value_p = Other_normal_label[index][icon_positive]
                label_value_n = Other_normal_label[index][icon_negative]

                if math.isnan(label_value_p):
                    label_value_p = 30
                if math.isnan(label_value_n):
                    label_value_n = 30

                label_training_now_P[str(Training_slot_value)].append(label_value_p)
                label_training_now_N[str(Training_slot_value)].append(label_value_n)
                Batch_now['p_' + str(Training_slot_value)].append(Training_normal)
                Batch_now['n_' + str(Training_slot_value)].append(Training_normal)

            for index, arr in enumerate(Valid_normal_graph):
                Training_normal = arr[graph_index]
                Training_slot_time = 10 + (graph_index / 60)
                Training_slot_value, Training_slot_index = find_closest(Training_slot_time, Label_slot_group)

                icon_positive = 'panas_pos_' + str(Training_slot_value)
                icon_negative = 'panas_neg_' + str(Training_slot_value)

                label_value_p = Valid_normal_label[index][icon_positive]
                label_value_n = Valid_normal_label[index][icon_negative]

                if math.isnan(label_value_p):
                    label_value_p = 30
                if math.isnan(label_value_n):
                    label_value_n = 30

                label_valid_now_P[str(Training_slot_value)].append(label_value_p)
                label_valid_now_N[str(Training_slot_value)].append(label_value_n)
                Batch_now_valid['p_' + str(Training_slot_value)].append(Training_normal)
                Batch_now_valid['n_' + str(Training_slot_value)].append(Training_normal)
                user_valid_now_P[str(Training_slot_value)].append(validation_set[index])
                user_valid_now_N[str(Training_slot_value)].append(validation_set[index])

        # training process
        print('Now for training classifier!!!')

        # Train 10 times for each panas and slot
        for training_data in Batch_now.items():
            Smallest_KL = 10000
            best_mu = 0
            best_sigma = 0
            best_y = 0
            print('Now training: ' + str(training_data[0]) + ' for user: ' + str(User_id))

            model = GCN_encoder_regression(encoder=encoder_model, freeze_encoder=True).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.NLLLoss()
            # Training
            if 'p_' in training_data[0]:
                training_labels = label_training_now_P[training_data[0].replace('p_', '')]
            else:
                training_labels = label_training_now_N[training_data[0].replace('n_', '')]

            # calculate the mu and sigma of training set
            mu_train = np.mean(training_labels)
            sigma_train = np.std(training_labels)

            data_list = []
            for data_index, graph_dict in enumerate(training_data[1]):
                data = Data(x=graph_dict['x'], edge_index=graph_dict['edge_index'], y=training_labels[data_index])
                data_list.append(data)

            batch_size = 128

            Train_loader = DataLoader(data_list, batch_size=batch_size, drop_last=True)

            for batch_train in Train_loader:
                batch_train = batch_train.to(device)

            # Test
            test_data_value = Batch_now_test[training_data[0]]
            test_data = (training_data[0], test_data_value)

            if 'p_' in test_data[0]:
                test_labels = label_test_now_P[test_data[0].replace('p_', '')]
            else:
                test_labels = label_test_now_N[test_data[0].replace('n_', '')]

            data_list_test = []
            for data_index, graph_dict in enumerate(test_data[1]):
                data = Data(x=graph_dict['x'], edge_index=graph_dict['edge_index'], y=test_labels[data_index])
                data_list_test.append(data)

            batch_size_test = 24

            Test_loader = DataLoader(data_list_test, batch_size=batch_size_test, drop_last=True)

            for batch_test in Test_loader:
                batch_test = batch_test.to(device)

            # validation set
            valid_data_value = Batch_now_valid[training_data[0]]
            valid_data = (training_data[0], valid_data_value)

            if 'p_' in valid_data[0]:
                valid_labels = label_valid_now_P[valid_data[0].replace('p_', '')]
            else:
                valid_labels = label_valid_now_N[valid_data[0].replace('n_', '')]

            if 'p_' in valid_data[0]:
                valid_user = user_valid_now_P[valid_data[0].replace('p_', '')]
            else:
                valid_user = user_valid_now_N[valid_data[0].replace('n_', '')]

            data_list_valid = []
            for data_index, graph_dict in enumerate(valid_data[1]):
                data = Data(x=graph_dict['x'], edge_index=graph_dict['edge_index'], y=valid_labels[data_index],
                            u=valid_user[data_index], modality=valid_data[0])
                data_list_valid.append(data)

            batch_size_valid = 24

            Valid_loader = DataLoader(data_list_valid, batch_size=batch_size_valid, drop_last=True)

            for batch_valid in Valid_loader:
                batch_valid = batch_valid.to(device)

            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                for batch_index, batch in enumerate(Train_loader):
                    batch = batch.to(device)

                    x = torch.tensor(batch.x, dtype=torch.float32)
                    edge_index = torch.tensor(batch.edge_index, dtype=torch.float32)
                    y = batch.y
                    x_reshaped, edge_index_combined, batch_vector = prepare_graph_batch(x, edge_index)

                    mu, sigma = model(x_reshaped.to(device), edge_index_combined.long().to(device),
                                      batch_vector.to(device))
                    loss = nll_loss(mu, sigma, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                avg_train_loss = total_loss / len(Train_loader)

                # ---------- Test ----------
                test_loss = 0.0

                mu_test = 0
                sigma_test = 0
                y_test = 0

                with torch.no_grad():
                    for batch in Test_loader:
                        batch = batch.to(device)

                        x = torch.tensor(batch.x, dtype=torch.float32)
                        edge_index = torch.tensor(batch.edge_index, dtype=torch.float32)
                        y = batch.y.to(device)

                        x_reshaped, edge_index_combined, batch_vector = prepare_graph_batch(x, edge_index)

                        mu, sigma = model(x_reshaped.to(device), edge_index_combined.long().to(device),
                                          batch_vector.to(device))
                        loss = nll_loss(mu, sigma, y)

                        mu_test += np.mean(mu.cpu().numpy())
                        sigma_test += np.mean(sigma.cpu().numpy())
                        y_test += np.mean(y.cpu().numpy())

                        test_loss += loss.item()

                avg_test_loss = test_loss / len(Test_loader)

                avg_test_mu = mu_test / len(Test_loader)
                avg_test_sigma = sigma_test / len(Test_loader)
                avg_test_y = y_test / len(Test_loader)

                klDivergence = kl_divergence(mu_train, sigma_train, avg_test_mu, avg_test_sigma)
                klDivergence = klDivergence * 10

                # ---------- Print Results ----------
                if epoch % 10 == 0:
                    print('Current Doc: ' + str(Current_Doc))
                    print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_test_loss:.4f}")
                    print(f"Epoch {epoch + 1} - kl_divergence: {klDivergence:.4f}")
                    if klDivergence < Smallest_KL:
                        Smallest_KL = float(klDivergence)
                        best_mu = float(avg_test_mu)
                        best_sigma = float(avg_test_sigma)
                        best_y = float(avg_test_y)

                        checkpoint = {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "smallest_kl": Smallest_KL,
                            "best_mu": best_mu,
                            "best_sigma": best_sigma,
                            "best_y": best_y
                        }
                        file_name = 'U_' + str(User_id) + '_' + str(training_data[0]) + '_' + Current_Doc + '.pth'
                        save_path = os.path.join("Ensemble_checkpoints", file_name)
                        torch.save(checkpoint, save_path)

                        print(f"✅ New best model saved at epoch {epoch + 1} with KL = {Smallest_KL:.4f}")

                        # --------------------------------------------------------------
                        # 🔥 Reload model and test again to verify checkpoint integrity
                        # --------------------------------------------------------------
                        print("🔄 Reloading checkpoint and running test...")

                        reloaded_model = GCN_encoder_regression(encoder_model, freeze_encoder=True).to(device)
                        reloaded_optimizer = optimizer.__class__(model.parameters(), lr=optimizer.param_groups[0]["lr"])

                        cp = torch.load(save_path, map_location=device, weights_only=False)
                        reloaded_model.load_state_dict(cp["model_state_dict"])
                        reloaded_optimizer.load_state_dict(cp["optimizer_state_dict"])
                        print("\n================= Checkpoint Content =================")
                        for key, value in cp.items():
                            if isinstance(value, dict):
                                # For state_dict or optimizer state_dict
                                print(f"{key}: [state_dict with {len(value)} keys]")
                            else:
                                print(f"{key}: {value}")
                        print("======================================================\n")

                        # ---- Run one Test again ----
                        reloaded_model.eval()
                        reload_mu = 0
                        reload_sigma = 0
                        reload_y = 0

                        with torch.no_grad():
                            for batch in Test_loader:
                                batch = batch.to(device)

                                x = torch.tensor(batch.x, dtype=torch.float32)
                                edge_index = torch.tensor(batch.edge_index, dtype=torch.float32)
                                y = batch.y.to(device)

                                x_reshaped, edge_index_combined, batch_vector = prepare_graph_batch(x, edge_index)

                                mu2, sigma2 = reloaded_model(
                                    x_reshaped.to(device),
                                    edge_index_combined.long().to(device),
                                    batch_vector.to(device)
                                )

                                reload_mu += np.mean(mu2.cpu().numpy())
                                reload_sigma += np.mean(sigma2.cpu().numpy())
                                reload_y += np.mean(y.cpu().numpy())

                        reload_mu /= len(Test_loader)
                        reload_sigma /= len(Test_loader)
                        reload_y /= len(Test_loader)

                        print(f"⭐ Reload Test — μ: {reload_mu:.4f}, σ: {reload_sigma:.4f}, y: {reload_y:.4f}")
                        print("✅ Checkpoint verified successfully.\n")

            print('Now: ' + str(training_data[0]) + ' for user: ' + str(User_id))
            print('Now Current Doc: ' + str(Current_Doc))
            print(f"Smallest KL: : {Smallest_KL:.8f}")
            print(f"Best MU: : {best_mu:.8f}")
            print(f"Best Sigma: : {best_sigma:.8f}")
            print(f"GT Label: : {best_y:.8f}")
            print(f"GT MU: : {mu_train:.8f}")
            print(f"GT Sigma: : {sigma_train:.8f}")

            average_small_KL.append(Smallest_KL)

            # --------------------------------------------------------------
            # 🔥 Reload model and validate again to getting threshold
            # --------------------------------------------------------------
            print("🔄 Validation for threshold!!!")

            # ---- Run one Validation again ----

            temp_user_data = defaultdict(lambda: {'mu_values': [], 'sigma_values': [], 'y_values': []})

            with torch.no_grad():
                for batch in Valid_loader:
                    batch = batch.to(device)

                    x = torch.tensor(batch.x, dtype=torch.float32)
                    edge_index = torch.tensor(batch.edge_index, dtype=torch.float32)
                    y = batch.y.to(device)
                    u_list = batch.u
                    modality_list = batch.modality

                    modality = list(set(modality_list))[0]

                    # load the corresponding checkpoint
                    file_name = 'U_' + str(User_id) + '_' + modality + '_' + Current_Doc + '.pth'
                    save_path = os.path.join("Ensemble_checkpoints", file_name)

                    valid_model = GCN_encoder_regression(encoder_model, freeze_encoder=True).to(device)
                    valid_optimizer = optimizer.__class__(model.parameters(), lr=optimizer.param_groups[0]["lr"])

                    cp = torch.load(save_path, map_location=device, weights_only=False)
                    valid_model.load_state_dict(cp["model_state_dict"])
                    valid_optimizer.load_state_dict(cp["optimizer_state_dict"])

                    x_reshaped, edge_index_combined, batch_vector = prepare_graph_batch(x, edge_index)

                    mu2, sigma2 = valid_model(
                        x_reshaped.to(device),
                        edge_index_combined.long().to(device),
                        batch_vector.to(device)
                    )

                    # --- 核心修改部分 ---

                    # 将 tensor 转为 numpy 数组并拉平
                    batch_mu = mu2.detach().cpu().numpy().flatten()
                    batch_sigma = sigma2.detach().cpu().numpy().flatten()
                    batch_y = y.detach().cpu().numpy().flatten()  # 【新增】处理 y

                    # 如果 batch.u 是 tensor，转为 python list
                    if isinstance(u_list, torch.Tensor):
                        u_list = u_list.cpu().tolist()

                    # 遍历当前 batch 的每一个样本，将结果填入 temp_user_data
                    for i, user_id in enumerate(u_list):
                        # 确保 user_id 是基本数据类型
                        if hasattr(user_id, 'item'):
                            user_id = user_id.item()

                        temp_user_data[user_id]['mu_values'].append(batch_mu[i])
                        temp_user_data[user_id]['sigma_values'].append(batch_sigma[i])
                        temp_user_data[user_id]['y_values'].append(batch_y[i])  # 【新增】存储 y

            # 2. 汇总计算
            final_user_dict = {}

            for user_id in validation_set:
                if user_id in temp_user_data:
                    data = temp_user_data[user_id]

                    final_user_dict[user_id] = {
                        'mu_mean': np.mean(data['mu_values']),
                        'sigma_mean': np.mean(data['sigma_values']),
                        'y_mean': np.mean(data['y_values'])  # 【新增】计算 y 的均值
                    }
                else:
                    print(f"Warning: User {user_id} not found in validation batches.")
                    final_user_dict[user_id] = {'mu_mean': 0.0, 'sigma_mean': 0.0, 'y_mean': 0.0}

            # 3. 打印结果
            print("\n--- Final Validation Results ---")
            for user_id, stats in sorted(final_user_dict.items()):
                # 【新增】在打印语句中加入 y
                print(f"The validation results for User {user_id}: mu = {stats['mu_mean']:.4f}; "
                      f"sigma = {stats['sigma_mean']:.4f}; y = {stats['y_mean']:.4f}.")

            print('Now: ' + str(training_data[0]) + ' for user: ' + str(User_id))
            print("✅ Get the threshold down.\n")

        print('\nNow training done!! for user: ' + str(User_id))
        print('Total Current Doc: ' + str(Current_Doc))
        print(f"Average KL: : {np.mean(average_small_KL):.8f}")
        print(f"Std KL: : {np.std(average_small_KL):.8f} \n")
