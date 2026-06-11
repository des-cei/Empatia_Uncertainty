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
@Time : Created in 11:28 AM 2025/06/17   
@FileName: Pack_json_graph.py                           
@Software: PyCharm


Introduction of this File:

"""
import os
import torch
from Actigraph_extract import Read_Actigraph
from Activity_extract import Read_Activity
from Graph_generation import graph_gen
from Merge_data import merge_data
from RR_extract import Read_RR
from Reduce_dim_min import Reduce_data_map
from Signal_augmentation.Actigraph_aug import Actigraph_Gaussian, Actigraph_Jitter, Actigraph_White
from Signal_augmentation.RR_aug import RR_Gaussian, RR_Jitter, RR_White
import json
from torch_geometric.data import Data

def dict_to_data(data_dict):
    data = {}
    for key, value in data_dict.items():
        data[key] = torch.tensor(value)
    return Data(**data)

def data_to_dict(data):
    data_dict = {}

    for key in data.keys():
        item = data[key]
        if isinstance(item, torch.Tensor):
            data_dict[key] = item.tolist()
        else:
            # 如果有其他非Tensor类型，比如字符串、整数
            data_dict[key] = item

    return data_dict

if __name__ == '__main__':
    root = '../dataset'
    User_map_dict = {}
    user_list = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    for User_num in user_list:
        # User_id = 18
        User_id = User_num.replace('user_', '')
        print('Now user: ' + str(User_id))
        path = os.path.join(root, ('user_' + str(User_id)))
        Actigraph_file = os.path.join(path, 'Actigraph.csv')
        RR_file = os.path.join(path, 'RR.csv')
        Activity_file = os.path.join(path, 'Activity.csv')
        data_Actigraph = Read_Actigraph(Actigraph_file)
        data_RR = Read_RR(RR_file)
        data_Activity = Read_Activity(Activity_file)
        data_merge_all = merge_data(data_Actigraph, data_RR, data_Activity)

        data_reduce_all = Reduce_data_map(data_merge_all)
        print('Number of min: ' + str(data_reduce_all.shape[1]))

        Graph_group = graph_gen(data_reduce_all)

        data_list_json = [data_to_dict(d) for d in Graph_group]

        User_map_dict[User_id] = data_list_json

    with open('../Json_files/graph_normal.json', 'w') as f:
        json.dump(User_map_dict, f)

    with open('../Json_files/graph_normal.json', 'r') as f:
        raw_data = json.load(f)

    User_map_dict_load = {}

    for key, value in raw_data.items():
        value_load = [dict_to_data(d) for d in value]
        User_map_dict_load[key] = value_load

    print(1)
