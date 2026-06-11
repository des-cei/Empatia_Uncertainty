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

import numpy as np
from sklearn import metrics

sys.path.append("/home/junjiaos/PycharmProjects/MMASH_research/")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
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
                new_val = 0
            elif val <= 27:
                new_val = 0
            else:
                new_val = 1
            new_row.append(new_val)
        result.append(new_row)
    return result

class CNN1DClassifier(nn.Module):
    def __init__(self):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)  # (B, 8, 64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, padding=1)  # (B, 1, 64)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):  # x shape: (B, 64)
        x = x.unsqueeze(1)  # (B, 1, 64)
        x = self.conv1(x)   # (B, 8, 64)
        x = self.relu(x)
        x = self.conv2(x)   # (B, 1, 64)
        x = x.squeeze(1)    # (B, 64)
        x = self.fc(x)      # (B, 2)
        return x

class CNN1DClassifier_v2(nn.Module):
    def __init__(self):
        super(CNN1DClassifier_v2, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):  # x: (B, 64)
        x = x.unsqueeze(1)  # (B, 1, 64)
        x = self.relu(self.bn1(self.conv1(x)))  # (B, 16, 64)
        x = self.relu(self.bn2(self.conv2(x)))  # (B, 32, 64)
        x = self.relu(self.bn3(self.conv3(x)))  # (B, 64, 64)
        x = self.pool(x).squeeze(-1)  # (B, 64)
        x = self.dropout(x)
        return self.fc(x)  # (B, 2)


class GCN_encoder_classifier(nn.Module):
    def __init__(self, encoder, freeze_encoder=True):
        super(GCN_encoder_classifier, self).__init__()
        self.encoder = encoder  # 使用已有的 encoder 实例

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # self.classifier = CNN1DClassifier()
        self.classifier = CNN1DClassifier_v2()

    def forward(self, x, edge_index, batch):
        with torch.no_grad():
            features = self.encoder(x, edge_index, batch)  # (B, 64)
        out = self.classifier(features)  # (B, 2
        return out


def train(model, dataloader, label_training, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for index_train, batch in enumerate(dataloader):
        batch = batch.to(device)

        x = batch.x
        edge_index = batch.edge_index
        batch_index = batch.batch

        labels = torch.tensor(label_training[index_train], dtype=torch.long).to(device)  # shape: (batch_size,)

        optimizer.zero_grad()
        outputs = model(x, edge_index, batch_index)  # (B, 3)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')  # 可选 'micro', 'weighted'
    confusion_matrix = metrics.confusion_matrix(all_labels, all_preds, labels=[0, 1])

    return avg_loss, acc, f1, confusion_matrix

# @torch.no_grad()
def evaluate(model, dataloader, label_test, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for index_test, batch in enumerate(dataloader):
        batch = batch.to(device)

        x = batch.x
        edge_index = batch.edge_index
        batch_index = batch.batch

        labels = torch.tensor(label_test[index_test], dtype=torch.long).to(device)

        outputs = model(x, edge_index, batch_index)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')  # 可以改成'micro'或'weighted'
    a = np.max(all_labels)
    b = np.max(all_preds)

    confusion_matrix = metrics.confusion_matrix(all_labels, all_preds, labels=[0, 1])

    return avg_loss, acc, f1, confusion_matrix

def find_closest(target, array):
    array = list(array)
    closest_index = min(range(len(array)), key=lambda i: abs(array[i] - target))
    closest_value = array[closest_index]
    return closest_value, closest_index

if __name__ == '__main__':
    root = '../dataset'
    model_root = '../Encoder_checkpoints'
    Label_slot_group = [10, 14, 18, 22, 33]
    user_list = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    user_list.remove('user_2')

    loso_train_total_acc = []
    loso_test_total_acc = []
    loso_train_total_f1 = []
    loso_test_total_f1 = []

    for User_num in user_list:
        feature_dim = 1
        encoder_model = GCNEncoder(in_channels=feature_dim).to(device)
        encoder_model.load_state_dict(torch.load(os.path.join(model_root, (User_num + '_weights.pth'))))

        # obtain the training data
        with open("../Json_files/graph_normal.json", "r", encoding="utf-8") as f:
            User_normal_data = json.load(f)

        with open("../Json_files/Label.json", "r", encoding="utf-8") as f:
            User_label_data = json.load(f)

        # remove data from user 2
        del User_normal_data['2']
        del User_label_data['2']

        # Do LOSO for training

        User_id = User_num.replace('user_', '')
        print('Now user: ' + str(User_id))

        # Get other graphs
        Other_normal_graph = graph_others(User_normal_data, User_id)

        # Pick number of users for training encoder
        num_samples = 20
        random.seed(42)
        list_length = len(Other_normal_graph)
        selected_indices = random.sample(range(list_length), num_samples)

        Other_normal_graph = [Other_normal_graph[i] for i in selected_indices]
        values_label = list(User_label_data.values())

        Other_normal_label = [values_label[i] for i in selected_indices]

        # Get normal feature maps
        feature_graph_group_normal = User_normal_data[User_id]
        normal_labels = User_label_data[User_id]

        # Generate batch

        Batch_list = []
        Batch_list_test = []
        All_label_training_P = []
        All_label_test_P = []
        All_label_training_N = []
        All_label_test_N = []

        total_graphs = len(feature_graph_group_normal)

        for graph_index in range(total_graphs):
            Batch_now = []
            Batch_now_test = []

            label_training_now_P = []
            label_test_now_P = []
            label_training_now_N = []
            label_test_now_N = []

            Indicator_normal = feature_graph_group_normal[graph_index]
            # arrange the slot time
            Indicator_slot_time = 10 + (graph_index / 60)
            Indicator_slot_value, Indicator_slot_index = find_closest(Indicator_slot_time, Label_slot_group)
            # find to correspond label
            icon_positive = 'panas_pos_' + str(Indicator_slot_value)
            icon_negative = 'panas_neg_' + str(Indicator_slot_value)

            Batch_now.append(Indicator_normal)
            label_training_now_P.append(normal_labels[icon_positive])
            label_training_now_N.append(normal_labels[icon_negative])
            Batch_now_test.append(Indicator_normal)
            label_test_now_P.append(normal_labels[icon_positive])
            label_test_now_N.append(normal_labels[icon_negative])

            candidates = []
            start = (graph_index + 300) % 1000
            candidates.append(start)

            for _ in range(2):
                next_idx = (candidates[-1] + 300) % 1000
                candidates.append(next_idx)

            idx = random.choice(candidates)
            for index, arr in enumerate(Other_normal_graph):
                if idx > len(arr):
                    idx = len(arr) - 1
                selected_normal = arr[idx]

                Batch_now.append(selected_normal)
                # arrange the slot time
                selected_slot_time = 10 + (idx / 60)
                selected_slot_value, selected_slot_index = find_closest(selected_slot_time, Label_slot_group)
                # find to correspond label
                icon_positive = 'panas_pos_' + str(selected_slot_value)
                icon_negative = 'panas_neg_' + str(selected_slot_value)
                label_training_now_P.append(Other_normal_label[index][icon_positive])
                label_training_now_N.append(Other_normal_label[index][icon_negative])


                # if index == 0 or index == 1 or index == 2 or index == 3 or index == 4:
                #     Batch_now_test.append(selected_normal)
                #     label_test_now_P.append(Other_normal_label[index][icon_positive])
                #     label_test_now_N.append(Other_normal_label[index][icon_negative])

            Batch_list.append(Batch_now)
            Batch_list_test.append(Batch_now_test)
            All_label_training_P.append(label_training_now_P)
            All_label_training_N.append(label_training_now_N)
            All_label_test_P.append(label_test_now_P)
            All_label_test_N.append(label_test_now_N)

        # training process
        print('Now for training classifier!!!')

        # All_label_training_P_trans = transform_2d_to_3d(All_label_training_P)
        # All_label_training_N_trans = transform_2d_to_3d(All_label_training_N)
        # All_label_test_P_trans = transform_2d_to_3d(All_label_test_P)
        # All_label_test_N_trans = transform_2d_to_3d(All_label_test_N)

        All_label_training_P_trans = transform_2d(All_label_training_P)
        All_label_training_N_trans = transform_2d(All_label_training_N)
        All_label_test_P_trans = transform_2d(All_label_test_P)
        All_label_test_N_trans = transform_2d(All_label_test_N)

        # transfer dict to Data
        Data_batch_list = []
        for batch_item in Batch_list:
            batch_data = dict_list_to_batch(batch_item).to(device)
            Data_batch_list.append(batch_data)

        Data_batch_list_test = []
        for batch_item in Batch_list_test:
            batch_data = dict_list_to_batch(batch_item).to(device)
            Data_batch_list_test.append(batch_data)

        model = GCN_encoder_classifier(encoder=encoder_model, freeze_encoder=True).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        num_epochs = 20
        train_total_acc = []
        test_total_acc = []
        train_total_f1 = []
        test_total_f1 = []

        for epoch in range(num_epochs):
            train_loss, train_acc, train_f1, confusion_matrix_train = train(model, Data_batch_list,
                                          All_label_training_P_trans, optimizer, criterion, device)
            val_loss, val_acc, val_f1, confusion_matrix_test = \
                evaluate(model, Data_batch_list_test, All_label_test_P_trans, criterion, device)

            # print(f"Epoch {epoch + 1}: "
            #       f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
            print(f"Epoch {epoch + 1}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Train f1={train_f1:.4f}, \n"
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val f1={val_f1:.4f}")
            print('confusion_matrix train:')
            print(confusion_matrix_train)
            print('confusion_matrix test:')
            print(confusion_matrix_test)

            train_total_acc.append(train_acc)
            test_total_acc.append(val_acc)
            train_total_f1.append(train_f1)
            test_total_f1.append(val_f1)

        print('Training process of classification Done!!')
        print('User Done: ' + str(User_id))
        print(f"Train Total Acc={np.max(train_total_acc):.4f}")
        print(f"Test Total Acc={np.max(test_total_acc):.4f}")
        print(f"Train Total f1={np.max(train_total_f1):.4f}")
        print(f"Test Total f1={np.max(test_total_f1):.4f}")

        loso_train_total_acc.append(np.max(train_total_acc))
        loso_test_total_acc.append(np.max(test_total_acc))
        loso_train_total_f1.append(np.max(train_total_f1))
        loso_test_total_f1.append(np.max(test_total_f1))

    print('Training All process Done!!!')
    print(f"All Train Total Acc={np.mean(loso_train_total_acc):.4f}")
    print(f"All Test Total Acc={np.mean(loso_test_total_acc):.4f}")
    print(f"All Train Total f1={np.mean(loso_train_total_f1):.4f}")
    print(f"All Test Total f1={np.mean(loso_test_total_f1):.4f}")

    print(f"All Train Total Acc Std ={np.std(loso_train_total_acc):.4f}")
    print(f"All Test Total Acc Std ={np.std(loso_test_total_acc):.4f}")
    print(f"All Train Total f1 Std ={np.std(loso_train_total_f1):.4f}")
    print(f"All Test Total f1 Std ={np.std(loso_test_total_f1):.4f}")

