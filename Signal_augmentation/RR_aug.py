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
@Time : Created in 12:57 PM 2025/06/06   
@FileName: RR_aug.py                           
@Software: PyCharm


Introduction of this File:

"""
import datetime
import os.path
import pandas as pd
import numpy as np
import torch


def add_white_noise(
        signal: torch.Tensor,
        noise_level: float = 1.0,
        clamp_range: tuple = None,
        round_result: bool = False,
        preserve_zero: bool = False,
) -> torch.Tensor:
    # 生成均匀分布噪声，范围[-noise_level, noise_level]
    noise = (torch.rand_like(signal) * 2 - 1) * noise_level

    if preserve_zero:
        mask = signal != 0
        noise = noise * mask

    noisy_signal = signal + noise

    if round_result:
        noisy_signal = torch.round(noisy_signal)

    if clamp_range:
        noisy_signal = torch.clamp(noisy_signal, clamp_range[0], clamp_range[1])

    return noisy_signal


def add_gaussian_noise(
        signal: torch.Tensor,
        std: float = 1.0,
        clamp_range: tuple = None,
        round_result: bool = False,
        preserve_zero: bool = False,
) -> torch.Tensor:
    noise = torch.randn_like(signal) * std

    if preserve_zero:
        mask = signal != 0
        noise = noise * mask

    noisy_signal = signal + noise

    if round_result:
        noisy_signal = torch.round(noisy_signal)

    if clamp_range:
        noisy_signal = torch.clamp(noisy_signal, clamp_range[0], clamp_range[1])

    return noisy_signal


def add_jitter(
        signal: torch.Tensor,
        std: float = 0.05,
        clamp_range: tuple = None,
        round_result: bool = False,
        preserve_zero: bool = False
) -> torch.Tensor:
    noise = torch.randn_like(signal) * std

    if preserve_zero:
        mask = signal != 0
        noise = noise * mask

    noisy_signal = signal + noise

    if round_result:
        noisy_signal = torch.round(noisy_signal)

    if clamp_range:
        noisy_signal = torch.clamp(noisy_signal, clamp_range[0], clamp_range[1])

    return noisy_signal


def RR_Gaussian(excel_file):
    df = pd.read_csv(excel_file, header=0)
    data = df.iloc[0:, 1:3].to_numpy()
    # add gaussian noise
    data_ibs = data[:, 0]
    data_ibs_tensor = torch.from_numpy(data_ibs).float()
    data_ibs_noisy = add_gaussian_noise(data_ibs_tensor, std=0.05, preserve_zero=True)
    data_ibs_noisy_arr = data_ibs_noisy.numpy()

    time = df.iloc[0:, 3].tolist()
    time_trans_list = []

    for t in time:
        t_list = datetime.datetime.strptime(t, "%H:%M:%S").time()
        t_trans = t_list.hour * 3600 + t_list.minute * 60 + t_list.second
        time_trans_list.append(t_trans)
    time_trans_array = np.array(time_trans_list)

    data_out = np.vstack((data_ibs_noisy_arr, data[:, 1], time_trans_array))

    data_out = np.transpose(data_out)

    for index, day in enumerate(data_out[:, 1]):
        if day != 1:
            data_out[index, 2] = data_out[index, 2] + 86400

    data_trans = np.transpose(data_out)
    # ---------------
    data_now = data_trans[:2, :]
    timestamps = data_trans[2, :]

    unique_ts = np.unique(timestamps)

    new_data = np.zeros((3, len(unique_ts)))

    for i, ts in enumerate(unique_ts):
        mask = (timestamps == ts)
        group = data_now[:, mask]  # shape: (2, n)
        mean_vals = np.mean(group, axis=1)  # shape: (2,)

        new_data[:2, i] = mean_vals
        new_data[2, i] = ts

    # time normalization
    new_out = np.transpose(new_data)
    start_time = data_trans[2, 0]
    end_time = data_trans[2, -1]

    time_array = np.arange(start_time, end_time + 1)  # shape: (T,)
    data_reConstruct = np.zeros((2, len(time_array)))

    time_to_index = {t: i for i, t in enumerate(time_array)}

    for value in new_out:
        t = int(value[-1])
        idx = time_to_index.get(t)
        if idx is not None:
            data_reConstruct[:, idx] = value[:2]
    data_reConstruct = np.vstack((data_reConstruct, time_array))
    for i in range(1, data_reConstruct.shape[1]):
        current_col = data_reConstruct[:2, i]

        if np.all(current_col == 0):
            data_reConstruct[:2, i] = data_reConstruct[:2, i - 1]

    return data_reConstruct


def RR_Jitter(excel_file):
    df = pd.read_csv(excel_file, header=0)
    data = df.iloc[0:, 1:3].to_numpy()
    # add gaussian noise
    data_ibs = data[:, 0]
    data_ibs_tensor = torch.from_numpy(data_ibs).float()
    data_ibs_noisy = add_jitter(data_ibs_tensor, std=0.025, preserve_zero=True)
    data_ibs_noisy_arr = data_ibs_noisy.numpy()

    time = df.iloc[0:, 3].tolist()
    time_trans_list = []

    for t in time:
        t_list = datetime.datetime.strptime(t, "%H:%M:%S").time()
        t_trans = t_list.hour * 3600 + t_list.minute * 60 + t_list.second
        time_trans_list.append(t_trans)
    time_trans_array = np.array(time_trans_list)

    data_out = np.vstack((data_ibs_noisy_arr, data[:, 1], time_trans_array))

    data_out = np.transpose(data_out)

    for index, day in enumerate(data_out[:, 1]):
        if day != 1:
            data_out[index, 2] = data_out[index, 2] + 86400

    data_trans = np.transpose(data_out)
    # ---------------
    data_now = data_trans[:2, :]
    timestamps = data_trans[2, :]

    unique_ts = np.unique(timestamps)

    new_data = np.zeros((3, len(unique_ts)))

    for i, ts in enumerate(unique_ts):
        mask = (timestamps == ts)
        group = data_now[:, mask]  # shape: (2, n)
        mean_vals = np.mean(group, axis=1)  # shape: (2,)

        new_data[:2, i] = mean_vals
        new_data[2, i] = ts

    # time normalization
    new_out = np.transpose(new_data)
    start_time = data_trans[2, 0]
    end_time = data_trans[2, -1]

    time_array = np.arange(start_time, end_time + 1)  # shape: (T,)
    data_reConstruct = np.zeros((2, len(time_array)))

    time_to_index = {t: i for i, t in enumerate(time_array)}

    for value in new_out:
        t = int(value[-1])
        idx = time_to_index.get(t)
        if idx is not None:
            data_reConstruct[:, idx] = value[:2]
    data_reConstruct = np.vstack((data_reConstruct, time_array))
    for i in range(1, data_reConstruct.shape[1]):
        current_col = data_reConstruct[:2, i]

        if np.all(current_col == 0):
            data_reConstruct[:2, i] = data_reConstruct[:2, i - 1]

    return data_reConstruct

def RR_White(excel_file):
    df = pd.read_csv(excel_file, header=0)
    data = df.iloc[0:, 1:3].to_numpy()
    # add gaussian noise
    data_ibs = data[:, 0]
    data_ibs_tensor = torch.from_numpy(data_ibs).float()
    data_ibs_noisy = add_white_noise(data_ibs_tensor, noise_level=0.05, preserve_zero=True)
    data_ibs_noisy_arr = data_ibs_noisy.numpy()

    time = df.iloc[0:, 3].tolist()
    time_trans_list = []

    for t in time:
        t_list = datetime.datetime.strptime(t, "%H:%M:%S").time()
        t_trans = t_list.hour * 3600 + t_list.minute * 60 + t_list.second
        time_trans_list.append(t_trans)
    time_trans_array = np.array(time_trans_list)

    data_out = np.vstack((data_ibs_noisy_arr, data[:, 1], time_trans_array))

    data_out = np.transpose(data_out)

    for index, day in enumerate(data_out[:, 1]):
        if day != 1:
            data_out[index, 2] = data_out[index, 2] + 86400

    data_trans = np.transpose(data_out)
    # ---------------
    data_now = data_trans[:2, :]
    timestamps = data_trans[2, :]

    unique_ts = np.unique(timestamps)

    new_data = np.zeros((3, len(unique_ts)))

    for i, ts in enumerate(unique_ts):
        mask = (timestamps == ts)
        group = data_now[:, mask]  # shape: (2, n)
        mean_vals = np.mean(group, axis=1)  # shape: (2,)

        new_data[:2, i] = mean_vals
        new_data[2, i] = ts

    # time normalization
    new_out = np.transpose(new_data)
    start_time = data_trans[2, 0]
    end_time = data_trans[2, -1]

    time_array = np.arange(start_time, end_time + 1)  # shape: (T,)
    data_reConstruct = np.zeros((2, len(time_array)))

    time_to_index = {t: i for i, t in enumerate(time_array)}

    for value in new_out:
        t = int(value[-1])
        idx = time_to_index.get(t)
        if idx is not None:
            data_reConstruct[:, idx] = value[:2]
    data_reConstruct = np.vstack((data_reConstruct, time_array))
    for i in range(1, data_reConstruct.shape[1]):
        current_col = data_reConstruct[:2, i]

        if np.all(current_col == 0):
            data_reConstruct[:2, i] = data_reConstruct[:2, i - 1]

    return data_reConstruct

if __name__ == '__main__':
    root = '../dataset'
    User_num = 1
    path = os.path.join(root, ('user_' + str(User_num)))
    file = os.path.join(path, 'RR.csv')
    # data_RR_Gaussian = RR_Gaussian(file)
    data_RR_Jitter = RR_Jitter(file)
    print(1)
