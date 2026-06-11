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
@Time : Created in 10:23 AM 2025/06/06   
@FileName: Actigraph_aug.py                           
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

def Actigraph_Gaussian(excel_file):
    df = pd.read_csv(excel_file, header=0)
    data = df.iloc[0:, 0:12].to_numpy()
    time = df.iloc[0:, 12].tolist()
    time_trans_list = []

    Axis_3D = df.iloc[0:, 1:4].to_numpy()
    HR = df.iloc[0:, 5].to_numpy()
    Vector_Magnitude = df.iloc[0:, 10].to_numpy()
    # add Actigraph_Gaussian
    axis_tensor = torch.from_numpy(Axis_3D).float()
    axis_noisy = add_gaussian_noise(axis_tensor, std=5, preserve_zero=True, round_result=True)
    axis_noisy_arr = axis_noisy.numpy()

    hr_tensor = torch.from_numpy(HR).float()
    hr_noisy = add_gaussian_noise(hr_tensor, std=2.0, clamp_range=(40, 200), round_result=True)
    hr_noisy_arr = hr_noisy.numpy()

    # calculate new Vector_Magnitude
    Vector_Magnitude_reCal = np.linalg.norm(axis_noisy_arr, axis=1)

    # integrate
    for t in time:
        t_list = datetime.datetime.strptime(t, "%H:%M:%S").time()
        t_trans = t_list.hour * 3600 + t_list.minute * 60 + t_list.second
        time_trans_list.append(t_trans)
    time_trans_array = np.array(time_trans_list)

    data_Re = np.vstack((np.transpose(axis_noisy_arr), data[:, 4], np.transpose(hr_noisy_arr), np.transpose(data[:, 6:10]),
                         np.transpose(Vector_Magnitude_reCal), data[:, -1], time_trans_array))
    data_out = np.transpose(data_Re)
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

def Actigraph_Jitter(excel_file):
    df = pd.read_csv(excel_file, header=0)
    data = df.iloc[0:, 0:12].to_numpy()
    time = df.iloc[0:, 12].tolist()
    time_trans_list = []

    Axis_3D = df.iloc[0:, 1:4].to_numpy()
    HR = df.iloc[0:, 5].to_numpy()
    Vector_Magnitude = df.iloc[0:, 10].to_numpy()
    # add Actigraph_Jitter
    axis_tensor = torch.from_numpy(Axis_3D).float()
    axis_noisy = add_jitter(axis_tensor, std=2.5, preserve_zero=True, round_result=True)
    axis_noisy_arr = axis_noisy.numpy()

    hr_tensor = torch.from_numpy(HR).float()
    hr_noisy = add_jitter(hr_tensor, std=1.5, clamp_range=(40, 200), round_result=True)
    hr_noisy_arr = hr_noisy.numpy()

    # calculate new Vector_Magnitude
    Vector_Magnitude_reCal = np.linalg.norm(axis_noisy_arr, axis=1)

    # integrate
    for t in time:
        t_list = datetime.datetime.strptime(t, "%H:%M:%S").time()
        t_trans = t_list.hour * 3600 + t_list.minute * 60 + t_list.second
        time_trans_list.append(t_trans)
    time_trans_array = np.array(time_trans_list)

    data_Re = np.vstack(
        (np.transpose(axis_noisy_arr), data[:, 4], np.transpose(hr_noisy_arr), np.transpose(data[:, 6:10]),
         np.transpose(Vector_Magnitude_reCal), data[:, -1], time_trans_array))
    data_out = np.transpose(data_Re)
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

def Actigraph_White(excel_file):
    df = pd.read_csv(excel_file, header=0)
    data = df.iloc[0:, 0:12].to_numpy()
    time = df.iloc[0:, 12].tolist()
    time_trans_list = []

    Axis_3D = df.iloc[0:, 1:4].to_numpy()
    HR = df.iloc[0:, 5].to_numpy()
    Vector_Magnitude = df.iloc[0:, 10].to_numpy()
    # add Actigraph_Gaussian
    axis_tensor = torch.from_numpy(Axis_3D).float()
    axis_noisy = add_white_noise(axis_tensor, noise_level=5, preserve_zero=True, round_result=True)
    axis_noisy_arr = axis_noisy.numpy()

    hr_tensor = torch.from_numpy(HR).float()
    hr_noisy = add_white_noise(hr_tensor, noise_level=2.0, clamp_range=(40, 200), round_result=True)
    hr_noisy_arr = hr_noisy.numpy()

    # calculate new Vector_Magnitude
    Vector_Magnitude_reCal = np.linalg.norm(axis_noisy_arr, axis=1)

    # integrate
    for t in time:
        t_list = datetime.datetime.strptime(t, "%H:%M:%S").time()
        t_trans = t_list.hour * 3600 + t_list.minute * 60 + t_list.second
        time_trans_list.append(t_trans)
    time_trans_array = np.array(time_trans_list)

    data_Re = np.vstack((np.transpose(axis_noisy_arr), data[:, 4], np.transpose(hr_noisy_arr), np.transpose(data[:, 6:10]),
                         np.transpose(Vector_Magnitude_reCal), data[:, -1], time_trans_array))
    data_out = np.transpose(data_Re)
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
    root = '../dataset'
    User_num = 1
    path = os.path.join(root, ('user_' + str(User_num)))
    file = os.path.join(path, 'Actigraph.csv')
    # data_Actigraph_Gaussian = Actigraph_Gaussian(file)
    data_Actigraph_Jitter = Actigraph_Jitter(file)
