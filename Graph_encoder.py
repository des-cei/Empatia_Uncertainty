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
@Time : Created in 12:41 PM 2025/06/17   
@FileName: Graph_encoder.py                           
@Software: PyCharm


Introduction of this File:

"""
import json
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch_geometric.data import Data, Batch

from Map_process.Map_encoder import nt_bxent_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=64):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Pool to get graph-level embedding
        x = self.projector(x)
        return x

def dict_list_to_batch(dict_graphs):
    data_list = []
    for g in dict_graphs:
        x = g['x']
        edge_index = g['edge_index']

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)
    batch = Batch.from_data_list(data_list)
    return batch

def graph_others(User_graph_dict, user_id):
    User_map_list = []
    for User_num, User_value in User_graph_dict.items():
        now_user = User_num
        if now_user != user_id:
            User_map_list.append(User_value)

    return User_map_list

if __name__ == '__main__':
    root = '../dataset'

    user_list = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]

    with open("../Json_files/graph_normal.json", "r", encoding="utf-8") as f:
        User_normal_data = json.load(f)
    with open("../Json_files/graph_Gaussian.json", "r", encoding="utf-8") as f:
        User_Gaussian_data = json.load(f)
    with open("../Json_files/graph_Jitter.json", "r", encoding="utf-8") as f:
        User_Jitter_data = json.load(f)
    with open("../Json_files/graph_White.json", "r", encoding="utf-8") as f:
        User_White_data = json.load(f)
# Do LOSO for training
    for User_num in user_list:
        # User_id = 18
        User_id = User_num.replace('user_', '')
        print('Now user: ' + str(User_id))

        # Get other graphs
        Other_normal_graph = graph_others(User_normal_data, User_id)
        Other_Gaussian_graph = graph_others(User_Gaussian_data, User_id)
        Other_Jitter_graph = graph_others(User_Jitter_data, User_id)
        Other_White_graph = graph_others(User_White_data, User_id)

        # Pick number of users for training encoder
        num_samples = 3
        list_length = len(Other_normal_graph)
        selected_indices = random.sample(range(list_length), num_samples)

        Other_normal_graph = [Other_normal_graph[i] for i in selected_indices]
        Other_Gaussian_graph = [Other_Gaussian_graph[i] for i in selected_indices]
        Other_Jitter_graph = [Other_Jitter_graph[i] for i in selected_indices]
        Other_White_graph = [Other_White_graph[i] for i in selected_indices]

        # Get normal feature maps
        feature_graph_group_normal = User_normal_data[User_id]

        # Get Gaussian feature maps
        feature_graph_group_Gaussian = User_Gaussian_data[User_id]

        # Get Jitter feature maps
        feature_graph_group_Jitter = User_Jitter_data[User_id]

        # Get White feature maps
        feature_graph_group_White = User_White_data[User_id]

        # Generate batch

        Batch_list = []

        total_graphs = len(feature_graph_group_normal)

        for graph_index in range(total_graphs):
            Batch_now = []

            Indicator_normal = feature_graph_group_normal[graph_index]
            Indicator_Gaussian = feature_graph_group_Gaussian[graph_index]
            Indicator_White = feature_graph_group_Jitter[graph_index]
            Indicator_Jitter = feature_graph_group_White[graph_index]
            Batch_now.extend([Indicator_normal, Indicator_Gaussian, Indicator_White, Indicator_Jitter])

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
                selected_Gaussian = Other_Gaussian_graph[index][idx]
                selected_Jitter = Other_Jitter_graph[index][idx]
                selected_White = Other_White_graph[index][idx]

                # selected = np.zeros((arr.shape[0], arr.shape[1]))
                # selected = torch.tensor(selected, dtype=torch.float32).to(device)

                Batch_now.extend([selected_normal, selected_Gaussian, selected_Jitter, selected_White])

            Batch_list.append(Batch_now)

        # training process
        print('Now for training encoder!!!')
        positive_pairs = []
        group_size = 4
        for i in range(0, len(Batch_list[0]), group_size):
            group = list(range(i, i + group_size))
            anchor = group[0]
            for other in group[1:]:
                positive_pairs.append((anchor, other))

        # transfer dict to Data
        Data_batch_list = []
        for batch_item in Batch_list:
            batch_data = dict_list_to_batch(batch_item).to(device)
            Data_batch_list.append(batch_data)

        batch = dict_list_to_batch(Batch_list[0])
        feature_dim = batch.x.size(1)
        print("Feature dimension:", feature_dim)

        model = GCNEncoder(in_channels=feature_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        num_epochs = 100
        model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            total_positive = 0
            total_negative = 0
            for batch in Data_batch_list:
                embeddings = model(batch.x, batch.edge_index, batch.batch)

                loss = nt_bxent_loss(embeddings, positive_pairs)
                # loss = nt_xent_loss(embeddings, positive_pairs)

                z = F.normalize(embeddings, dim=1)
                sim_matrix = torch.matmul(z, z.T)

                positive_sims = []
                for i, j in positive_pairs:
                    positive_sims.append(sim_matrix[i, j].item())

                positive_sims = torch.tensor(positive_sims, device=device)
                # print(f"Positive pairs similarity mean: {positive_sims.mean().item():.6f}")
                total_positive += positive_sims.mean()

                mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
                for i, j in positive_pairs:
                    mask[i, j] = True
                    mask[j, i] = True

                negative_sims = sim_matrix[~mask]

                negative_mean = negative_sims.mean().item()
                negative_std = negative_sims.std().item()
                # print(f"Negative pairs similarity mean: {negative_mean}")
                total_negative += negative_mean

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(Batch_list)
            avg_positive = total_positive / len(Batch_list)
            avg_negative = total_negative / len(Batch_list)
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f} - Positive: {avg_positive:.4f} "
                  f"- Negative: {avg_negative:.4f}")


