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
@Time : Created in 3:48 PM 2025/05/29   
@FileName: Activity_extract.py                           
@Software: PyCharm


Introduction of this File:

"""

import datetime
import os.path
import pandas as pd
import numpy as np

def Read_Activity(excel_file):
    df = pd.read_csv(excel_file, header=0)
    data = df.iloc[0:, [1, 4]].to_numpy()
    data = data[~np.isnan(data).any(axis=1)]
    time_start = df.iloc[0:, 2].tolist()
    time_end = df.iloc[0:, 3].tolist()
    time_start_trans_list = []
    time_end_trans_list = []

    for t in time_start:
        try:
            if t == '24:00':
                t = '23:59'
            t_list = datetime.datetime.strptime(t, "%H:%M").time()
            t_trans = t_list.hour * 3600 + t_list.minute * 60
            time_start_trans_list.append(t_trans)
        except:
            continue
    for t in time_end:
        try:
            if t == '24:00:00' or t == '24:00':
                t = '23:59'
            t_list = datetime.datetime.strptime(t, "%H:%M").time()
            t_trans = t_list.hour * 3600 + t_list.minute * 60
            time_end_trans_list.append(t_trans)
        except:
            continue
    time_start_trans_array = np.array(time_start_trans_list)
    time_end_trans_array = np.array(time_end_trans_list)
    data_out = np.hstack((data, time_start_trans_array.reshape(-1, 1), time_end_trans_array.reshape(-1, 1)))
    for index, day in enumerate(data_out[:, 1]):
        if day != 1:
            data_out[index, 2] = data_out[index, 2] + 86400
            data_out[index, 3] = data_out[index, 3] + 86400
    # transfer the array to a new array with every second as y axis
    data_trans = np.transpose(data_out)
    start_time = data_trans[2, 0]
    end_time = data_trans[3, -1]
    time_array = np.arange(start_time, (end_time + 1))
    data_reConstruct = np.full((2, (int(end_time - start_time) + 1)), -1)
    data_reConstruct = np.vstack((data_reConstruct, time_array))
    for value in data_out:
        item = value[0]
        day = value[1]
        s_time = value[2]
        e_time = value[3]
        for i in range(data_reConstruct.shape[1]):
            value_real = data_reConstruct[:, i]
            if s_time <= value_real[2] < e_time:
                data_reConstruct[0, i] = item
                data_reConstruct[1, i] = day
    data_reConstruct = data_reConstruct[:, :-1]
    return data_reConstruct

if __name__ == '__main__':
    root = 'dataset'
    User_num = 1
    path = os.path.join(root, ('user_' + str(User_num)))
    file = os.path.join(path, 'Activity.csv')
    data_Activity = Read_Activity(file)
    print(1)