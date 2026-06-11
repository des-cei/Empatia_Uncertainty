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
@Time : Created in 11:10 AM 2025/05/30   
@FileName: Feature_map_generation.py                           
@Software: PyCharm


Introduction of this File:

"""
import os

import numpy as np

from Actigraph_extract import Read_Actigraph
from Activity_extract import Read_Activity
from Merge_data import merge_data
from RR_extract import Read_RR
from Reduce_dim_min import Reduce_data_map
from Signal_augmentation.Actigraph_aug import Actigraph_Gaussian, Actigraph_Jitter
from Signal_augmentation.RR_aug import RR_Gaussian, RR_Jitter


def feature_map_gen(merged_data):
    # split the feature array to 2D maps
    feature_map_group = np.zeros((merged_data.shape[0], merged_data.shape[0]))
    range_col = merged_data.shape[0]
    for i in range(merged_data.shape[1]):
        if (i + 1) % range_col == 0:
            now_map = merged_data[:, (i + 1 - range_col):(i + 1)]
            feature_map_group = np.dstack((feature_map_group, now_map))

    feature_map_group = feature_map_group[:, :, 1:]

    return feature_map_group


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

        data_reduce_all = Reduce_data_map(data_merge_all)

        feature_map_group = feature_map_gen(data_merge_all)
        print('Number of feature map: ' + str(feature_map_group.shape[2]))
