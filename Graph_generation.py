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
@Time : Created in 10:53 AM 2025/06/02   
@FileName: Graph_generation.py                           
@Software: PyCharm


Introduction of this File:

"""

import os

import numpy as np
import torch

from Actigraph_extract import Read_Actigraph
from Activity_extract import Read_Activity
from Merge_data import merge_data
from RR_extract import Read_RR
from torch_geometric.data import Data
from Signal_augmentation.Actigraph_aug import Actigraph_Gaussian, Actigraph_Jitter
from Signal_augmentation.RR_aug import RR_Gaussian, RR_Jitter

def graph_gen_less(merged_data):
    if not isinstance(merged_data, torch.Tensor):
        merged_data = torch.tensor(merged_data, dtype=torch.float)

    num_features, num_time_points = merged_data.shape

    graphs = []

    for t in range(num_time_points):
        node_feats = []  # will be size (21, 1)

        # Original nodes 0-12
        for i in range(num_features):
            node_feats.append(merged_data[i, t].unsqueeze(0))  # shape: (1,)

        # Virtual nodes:
        # activity node (index 18) = avg(0~17)
        ActiGraph_val = merged_data[0:13, t].mean().unsqueeze(0)
        node_feats.append(ActiGraph_val)

        # root node (index 22) = avg(0~20)
        root_val = merged_data[0:16, t].mean().unsqueeze(0)
        node_feats.append(root_val)

        x = torch.stack(node_feats)  # shape: (15, 1)

        # Edges
        edge_index = []

        # ActiGraph node (21) --> 0~17
        for i in range(13):
            edge_index.append([16, i])  # ActiGraph -> node i

        # root node (22) --> ActiGraph, 18,19,20
        edge_index.append([17, 16])  # root -> activity
        edge_index.append([17, 15])
        edge_index.append([17, 14])
        edge_index.append([17, 13])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        graph = Data(x=x, edge_index=edge_index)
        graphs.append(graph)

    return graphs

def graph_gen(merged_data):
    if not isinstance(merged_data, torch.Tensor):
        merged_data = torch.tensor(merged_data, dtype=torch.float)

    num_features, num_time_points = merged_data.shape

    graphs = []

    for t in range(num_time_points):
        node_feats = []  # will be size (21, 1)

        # Original nodes 0-12
        for i in range(num_features):
            node_feats.append(merged_data[i, t].unsqueeze(0))  # shape: (1,)

        # Virtual nodes:
        # activity node (index 18) = avg(0~17)
        ActiGraph_val = merged_data[0:18, t].mean().unsqueeze(0)
        node_feats.append(ActiGraph_val)

        # root node (index 22) = avg(0~20)
        root_val = merged_data[0:21, t].mean().unsqueeze(0)
        node_feats.append(root_val)

        x = torch.stack(node_feats)  # shape: (15, 1)

        # Edges
        edge_index = []

        # ActiGraph node (21) --> 0~17
        for i in range(18):
            edge_index.append([21, i])  # ActiGraph -> node i

        # root node (22) --> ActiGraph, 18,19,20
        edge_index.append([22, 21])  # root -> activity
        edge_index.append([22, 20])
        edge_index.append([22, 19])
        edge_index.append([22, 18])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        graph = Data(x=x, edge_index=edge_index)
        graphs.append(graph)

    return graphs

if __name__ == '__main__':
    root = 'dataset'

    user_list = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    for User_num in user_list:
        # User_id = 18
        User_id = User_num.replace('user_', '')
        print('Now user: ' + str(User_id))
        path = os.path.join(root, ('user_' + str(User_id)))
        Actigraph_file = os.path.join(path, 'Actigraph.csv')
        RR_file = os.path.join(path, 'RR.csv')
        Activity_file = os.path.join(path, 'Activity.csv')
        data_Actigraph = Actigraph_Gaussian(Actigraph_file)
        data_RR = RR_Gaussian(RR_file)
        data_Activity = Read_Activity(Activity_file)
        data_merge_all = merge_data(data_Actigraph, data_RR, data_Activity)

        Graph_group = graph_gen(data_merge_all)

        print(1)
