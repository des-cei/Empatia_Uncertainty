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
@Time : Created in 5:20 PM 2025/05/29   
@FileName: Merge_data.py                           
@Software: PyCharm


Introduction of this File:

"""
import os

import numpy as np

from Actigraph_extract import Read_Actigraph
from Activity_extract import Read_Activity
from RR_extract import Read_RR


def merge_data(data_Actigraph, data_RR, data_Activity):
    # remove day
    data_Actigraph = np.vstack((data_Actigraph[0:10], data_Actigraph[11]))
    data_RR = np.vstack((data_RR[0], data_RR[2]))
    data_Activity = np.vstack((data_Activity[0], data_Activity[2]))
    # take the maximum start time and minimum end time to normalize them
    start_time_all = np.max([data_Actigraph[10, 0], data_RR[1, 0], data_Activity[1, 0]])
    end_time_all = np.min([data_Actigraph[10, -1], data_RR[1, -1], data_Activity[1, -1]])
    # normalize three array to same arrange
    mask = (data_Actigraph[10] >= start_time_all) & (data_Actigraph[10] <= end_time_all)
    data_Actigraph_nor = data_Actigraph[:, mask]
    mask = (data_RR[1] >= start_time_all) & (data_RR[1] <= end_time_all)
    data_RR_nor = data_RR[:, mask]
    mask = (data_Activity[1] >= start_time_all) & (data_Activity[1] <= end_time_all)
    data_Activity_nor = data_Activity[:, mask]
    # stack three array
    data_merge_all = np.vstack((data_Actigraph_nor[:10, :], data_RR_nor[0, :], data_Activity_nor[:, :]))

    return data_merge_all

def merge_data_NoActi(data_Actigraph, data_RR):
    # remove day
    data_Actigraph = np.vstack((data_Actigraph[0:10], data_Actigraph[11]))
    # data_RR = np.vstack((data_RR[0], data_RR[2]))
    # take the maximum start time and minimum end time to normalize them
    start_time_all = np.max([data_Actigraph[10, 0], data_RR[2, 0]])
    end_time_all = np.min([data_Actigraph[10, -1], data_RR[2, -1]])
    # normalize three array to same arrange
    mask = (data_Actigraph[10] >= start_time_all) & (data_Actigraph[10] <= end_time_all)
    data_Actigraph_nor = data_Actigraph[:, mask]
    mask = (data_RR[2] >= start_time_all) & (data_RR[2] <= end_time_all)
    data_RR_nor = data_RR[:, mask]
    # stack three array
    data_merge_all = np.vstack((data_Actigraph_nor[:10, :], data_RR_nor[:, :]))

    return data_merge_all

if __name__ == '__main__':
    root = 'dataset'
    User_num = 1
    path = os.path.join(root, ('user_' + str(User_num)))
    Actigraph_file = os.path.join(path, 'Actigraph.csv')
    RR_file = os.path.join(path, 'RR.csv')
    Activity_file = os.path.join(path, 'Activity.csv')
    data_Actigraph = Read_Actigraph(Actigraph_file)
    data_RR = Read_RR(RR_file)
    data_merge_all = merge_data(data_Actigraph, data_RR)
