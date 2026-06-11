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
@Time : Created in 12:36 PM 2025/05/29   
@FileName: Actigraph_extract.py                           
@Software: PyCharm


Introduction of this File:

"""
import datetime
import os.path
import pandas as pd
import numpy as np


def Read_Actigraph(excel_file):
    df = pd.read_csv(excel_file, header=0)
    data = df.iloc[0:, 1:12].to_numpy()
    time = df.iloc[0:, 12].tolist()
    time_trans_list = []

    for t in time:
        t_list = datetime.datetime.strptime(t, "%H:%M:%S").time()
        t_trans = t_list.hour * 3600 + t_list.minute * 60 + t_list.second
        time_trans_list.append(t_trans)
    time_trans_array = np.array(time_trans_list)
    data_out = np.hstack((data, time_trans_array.reshape(-1, 1)))
    for index, day in enumerate(data_out[:, 10]):
        if day != 1:
            data_out[index, 11] = data_out[index, 11] + 86400

    data_trans = np.transpose(data_out)
    # normalize the array according to time
    start_time = data_trans[11, 0]
    end_time = data_trans[11, -1]

    time_array = np.arange(start_time, end_time + 1)  # shape: (T,)
    data_reConstruct = np.zeros((11, len(time_array)))

    time_to_index = {t: i for i, t in enumerate(time_array)}

    for value in data_out:
        t = int(value[-1])
        idx = time_to_index.get(t)
        if idx is not None:
            data_reConstruct[:, idx] = value[:11]
    data_reConstruct = np.vstack((data_reConstruct, time_array))
    for i in range(1, data_reConstruct.shape[1]):
        current_col = data_reConstruct[:11, i]

        if np.all(current_col == 0):
            data_reConstruct[:11, i] = data_reConstruct[:11, i - 1]

    return data_reConstruct

if __name__ == '__main__':
    root = 'dataset'
    User_num = 1
    path = os.path.join(root, ('user_' + str(User_num)))
    file = os.path.join(path, 'Actigraph.csv')
    data_Actigraph = Read_Actigraph(file)
    print(1)