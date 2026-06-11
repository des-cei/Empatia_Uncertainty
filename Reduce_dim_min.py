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
@Time : Created in 7:03 PM 2025/06/06   
@FileName: Reduce_dim_min.py                           
@Software: PyCharm


Introduction of this File:

"""

import os

import numpy as np

from Actigraph_extract import Read_Actigraph
from Activity_extract import Read_Activity
from Merge_data import merge_data
from RR_extract import Read_RR
from Signal_augmentation.Actigraph_aug import Actigraph_Gaussian, Actigraph_Jitter
from Signal_augmentation.RR_aug import RR_Gaussian, RR_Jitter

def majority_per_minute_all_channels(axis_4xn: np.ndarray) -> np.ndarray:
    """
    输入:
        axis_4xn: np.ndarray of shape (4, n)，其中每行是一个0/1二值信号，n 是 60 的倍数
    输出:
        np.ndarray of shape (4, n//60)，每行每分钟的多数类（0 或 1）
    """
    c, n = axis_4xn.shape
    assert c == 4, "输入必须是 (4, n) 的格式"
    assert n % 60 == 0, "时间轴长度必须是 60 的倍数"

    num_minutes = n // 60
    result = []

    for i in range(c):  # 对每一行分别处理
        channel = axis_4xn[i, :]  # shape: (n,)
        reshaped = channel.reshape(num_minutes, 60)  # shape: (分钟数, 60)
        majority = (reshaped.sum(axis=1) >= 30).astype(int)  # shape: (分钟数,)
        result.append(majority)

    return np.stack(result, axis=0)  # shape: (4, n//60)

def majority_per_minute(data_1xn: np.ndarray) -> np.ndarray:
    """
    输入:
        data_1xn: np.ndarray of shape (1, n)，其中 n 是 60 的倍数，且值为 0 或 1
    输出:
        np.ndarray of shape (1, n//60)，每分钟的多数类（0 或 1）
    """
    assert data_1xn.ndim == 2 and data_1xn.shape[0] == 1, "输入必须是 shape (1, n)"
    n = data_1xn.shape[1]
    assert n % 60 == 0, "时间轴长度必须是 60 的倍数"

    num_minutes = n // 60
    reshaped = data_1xn.reshape(num_minutes, 60)  # shape: (分钟数, 60)
    majority = (reshaped.sum(axis=1) >= 30).astype(int)  # shape: (分钟数,)

    return majority.reshape(1, -1)  # shape: (1, n//60)

def sum_per_minute(arr_1xn: np.ndarray) -> np.ndarray:
    """
    输入:
        arr_1xn: np.ndarray of shape (1, n)，n 是 60 的倍数
    输出:
        np.ndarray of shape (1, n//60)，每分钟一个和
    """
    assert arr_1xn.ndim == 2 and arr_1xn.shape[0] == 1, "输入应为 shape (1, n)"
    _, n = arr_1xn.shape
    assert n % 60 == 0, "长度必须是 60 的倍数"

    arr = arr_1xn.reshape(1, n // 60, 60)     # shape: (1, 分钟数, 60)
    summed = arr.sum(axis=2)                  # 每分钟求和 → shape: (1, 分钟数)
    return summed

def avg_per_minute(arr_1xn: np.ndarray) -> np.ndarray:
    """
    输入:
        arr_1xn: np.ndarray of shape (1, n)，n 是 60 的倍数
    输出:
        np.ndarray of shape (1, n//60)，每分钟一个和
    """
    assert arr_1xn.ndim == 2 and arr_1xn.shape[0] == 1, "输入应为 shape (1, n)"
    _, n = arr_1xn.shape
    assert n % 60 == 0, "长度必须是 60 的倍数"

    arr = arr_1xn.reshape(1, n // 60, 60)
    avg = arr.mean(axis=2)
    return avg

def summarize_axis_minutes(axis_3xn: np.ndarray) -> np.ndarray:
    """
    输入:
        axis_3xn: np.ndarray of shape (3, n)，其中 n 是 60 的倍数，表示 X/Y/Z 三轴加速度
    输出:
        np.ndarray of shape (9, n//60)，每分钟一个 9 维特征向量
    """
    c, n = axis_3xn.shape
    assert c == 3, "输入必须是 (3, n) 的格式"
    assert n % 60 == 0, "时间轴长度必须是 60 的倍数"

    num_minutes = n // 60
    result = []

    for i in range(num_minutes):
        start = i * 60
        end = start + 60
        segment = axis_3xn[:, start:end]       # shape: (3, 60)
        segment_t = segment.T                  # shape: (60, 3)

        # 均值与标准差
        mean = segment_t.mean(axis=0)          # (3,)
        std = segment_t.std(axis=0)            # (3,)

        # 向量模长（每个时刻的 √(x² + y² + z²)）
        vmag = np.linalg.norm(segment_t, axis=1)  # (60,)
        vmag_mean = vmag.mean()
        vmag_std = vmag.std()
        vmag_max = vmag.max()

        feature = np.concatenate([mean, std, [vmag_mean, vmag_std, vmag_max]])  # (9,)
        result.append(feature)

    return np.stack(result, axis=1)  # 输出 shape: (9, n//60)

def summarize_single_channel_minutes(data_1xn: np.ndarray) -> np.ndarray:
    """
    输入:
        data_1xn: np.ndarray of shape (1, n)，n 是 60 的倍数，表示单通道数据（如 vmag）
    输出:
        np.ndarray of shape (3, n//60)，每分钟三个特征：均值、标准差、最大值
    """
    c, n = data_1xn.shape
    assert c == 1, "输入必须是 (1, n) 的格式"
    assert n % 60 == 0, "时间轴长度必须是 60 的倍数"

    num_minutes = n // 60
    result = []

    for i in range(num_minutes):
        start = i * 60
        end = start + 60
        segment = data_1xn[0, start:end]  # shape: (60,)

        mean = segment.mean()
        std = segment.std()
        max_val = segment.max()

        feature = np.array([mean, std, max_val])  # shape: (3,)
        result.append(feature)

    return np.stack(result, axis=1)  # shape: (3, n//60)


def extract_minutes(timestamps_1xn: np.ndarray) -> np.ndarray:
    """
    输入:
        timestamps_1xn: np.ndarray of shape (1, n)，连续的秒级时间戳（n 是 60 的倍数）
    输出:
        np.ndarray of shape (1, n//60)，每分钟一个值（例如每分钟第一个时间戳）
    """
    assert timestamps_1xn.ndim == 2 and timestamps_1xn.shape[0] == 1, "输入必须是 shape (1, n)"
    n = timestamps_1xn.shape[1]
    assert n % 60 == 0, "时间长度必须是 60 的倍数"

    timestamps = timestamps_1xn[0]
    assert np.all(np.diff(timestamps) == 1), "时间戳必须是连续的秒数"
    assert timestamps[0] % 3600 == 0 and (timestamps[-1] + 1) % 3600 == 0, "首尾必须能被 3600 整除"

    minute_values = timestamps[::60]  # 每分钟取第一个秒值
    return minute_values.reshape(1, -1)  # shape: (1, n//60)

def Reduce_data_map(merged_data):
    # trim the data only remain hour.
    hour_list = []
    merged_data_trans = np.transpose(merged_data)
    for index, value in enumerate(merged_data_trans):
        if value[12] % 3600 == 0:
            hour_list.append(index)
    merged_data_trim = merged_data[:, hour_list[0]:hour_list[-1]]
    # reduce the dimension of axis
    axis_data = merged_data_trim[:3, :]
    axis_data_reduced = summarize_axis_minutes(axis_data)
    # reduce the dimension of step
    step_data = merged_data_trim[3:4, :]
    step_data_reduced = sum_per_minute(step_data)
    # reduce the dimension of HR
    HR_data = merged_data_trim[4:5, :]
    HR_data_reduced = avg_per_minute(HR_data)
    # reduce the dimension of Inclinometer
    Inclinometer_data = merged_data_trim[5:9, :]
    Inclinometer_data_reduced = majority_per_minute_all_channels(Inclinometer_data)
    # reduce the dimension of Vector
    VM_data = merged_data_trim[9:10, :]
    VM_data_reduced = summarize_single_channel_minutes(VM_data)
    # reduce the dimension of RR
    RR_data = merged_data_trim[10:11, :]
    RR_data_reduced = avg_per_minute(RR_data)
    # reduce the dimension of Activity
    Activity_data = merged_data_trim[11:12, :]
    Activity_data_reduced = majority_per_minute(Activity_data)
    # reduce the dimension of time
    time_data = merged_data_trim[12:, :]
    time_data_reduced = extract_minutes(time_data)
    # Integrate them
    merged_data_reduced = np.vstack((axis_data_reduced, step_data_reduced, HR_data_reduced, Inclinometer_data_reduced,
                                     VM_data_reduced, RR_data_reduced, Activity_data_reduced, time_data_reduced))
    return merged_data_reduced

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
        print('Number of min: ' + str(data_reduce_all.shape[1]))