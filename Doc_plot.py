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
@Time : Created in 2:59 PM 2025/12/11   
@FileName: Doc_plot.py                           
@Software: PyCharm


Introduction of this File:

"""
import os.path
import matplotlib.patches as mpatches
import numpy as np

from Ensembles_Threshold.Diagnose_Docs import calculate_uncertainty_dict, generate_diagnostic_dict
from Ensembles_Threshold.Read_threshold import read_doc_threshold, calculate_uncertainty_thresholds
from Read_Doc import read_doc
import matplotlib.pyplot as plt


def parse_time_keys(d):
    times = []
    for k in d.keys():
        t = int(k.split('_')[1])  # 'T_10' -> 10
        times.append((k, t))
    return sorted(times, key=lambda x: x[1])


def plot_type_shadow(target_type, data, time_pairs, real_times, doc_keys, user_id, x_positions, Diagnosis_Result):
    """
    修改点：让背景色块之间有空隙，不再连成一片。
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    num_docs = len(doc_keys)
    offsets = np.linspace(-0.25, 0.25, num_docs)

    # 颜色定义
    case_colors = {
        "Case 0": "#eaffea",  # 浅绿
        "Case 1": "#fff8c4",  # 浅黄
        "Case 2": "#ffdddd"  # 浅红
    }
    default_color = "white"

    # =========================================================
    # 1. 绘制分离的背景阴影 (Block Shading)
    # =========================================================

    # ⭐ 核心修改：定义色块的半宽
    # 如果设为 0.5，则色块相连；设为 0.4，则中间有 0.2 的空隙
    half_width = 0.4

    for (t_key, _), x_pos in zip(time_pairs, x_positions):

        # 获取状态码
        status_code = "Unknown"
        try:
            diag_info = Diagnosis_Result[user_id][t_key][target_type]
            status_code = diag_info.get("status_code", "Unknown")
        except KeyError:
            pass

        bg_color = case_colors.get(status_code, default_color)

        # ⭐ 修改绘制范围：留出空隙
        ax.axvspan(
            x_pos - half_width,  # 左边界
            x_pos + half_width,  # 右边界
            color=bg_color,
            alpha=0.6,
            zorder=0,
            edgecolor=None  # 确保没有边框线，看起来更干净
        )

    # =========================================================
    # 2. 绘制数据点 (保持不变)
    # =========================================================
    for offset, doc in zip(offsets, doc_keys):
        mus = []
        sigmas = []

        for t_key, _ in time_pairs:
            try:
                entry = data[t_key][target_type][doc]
                mus.append(entry["predicted_mu"])
                sigmas.append(entry["predicted_sigma"])
            except KeyError:
                mus.append(np.nan)
                sigmas.append(np.nan)

        shifted_x = x_positions + offset

        ax.errorbar(
            shifted_x, mus, yerr=sigmas,
            fmt='o', markersize=6, capsize=3,
            label=f"{doc}", zorder=3
        )

    # =========================================================
    # 3. 绘制 GT (保持不变)
    # =========================================================
    gts = []
    for t_key, _ in time_pairs:
        try:
            gts.append(data[t_key][target_type]["GT"])
        except KeyError:
            gts.append(np.nan)

    ax.plot(x_positions, gts, 'o-r', linewidth=2, markersize=8, label="GT", zorder=4)

    # =========================================================
    # 4. 图表装饰
    # =========================================================
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(t) for t in real_times])

    ax.set_xlabel("Time Step")
    ax.set_ylabel(f"{target_type} value")
    ax.set_title(f'{target_type.upper()} Analysis: Predicted vs GT (User: {user_id})')

    # 网格线设为虚线，稍微淡一点
    ax.grid(True, linestyle=':', alpha=0.6, zorder=1)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    patch_case0 = mpatches.Patch(color=case_colors['Case 0'], label='Case 0: Trusted')
    patch_case1 = mpatches.Patch(color=case_colors['Case 1'], label='Case 1: Data Uncertainty')
    patch_case2 = mpatches.Patch(color=case_colors['Case 2'], label='Case 2: Expert Referral')
    handles.extend([patch_case0, patch_case1, patch_case2])

    ax.legend(handles=handles, loc='best', fontsize='small')

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join('Plot_Doc', ('User_' + str(user_id) + '_Emotion_' + str(target_type.upper()))))


if __name__ == '__main__':
    Data = read_doc('Ensemble_Doc1.log', 'Ensemble_Doc2.log', 'Ensemble_Doc3.log',
                    '/home/junjiaos/PycharmProjects/MMASH_research/Graph_process/Graph_NLL721.log')

    Data_train = read_doc('Ensemble_Doc1.log', 'Ensemble_Doc2.log', 'Ensemble_Doc3.log',
                          '/home/junjiaos/PycharmProjects/MMASH_research/Graph_process/Graph_NLL721.log')
    Data_threshold = read_doc_threshold('Ensemble_Doc1.log', 'Ensemble_Doc2.log', 'Ensemble_Doc3.log')
    thresholds_dict = calculate_uncertainty_thresholds(Data_threshold, mode='median', percentile=80)
    Uncertainty_Data = calculate_uncertainty_dict(Data_train, mode='median')
    Diagnosis_Result = generate_diagnostic_dict(Uncertainty_Data, thresholds_dict)

    for user_id, user_data in Data.items():
        time_pairs = parse_time_keys(user_data)  # [( 'T_10', 10), ...]
        time_labels = [t for _, t in time_pairs]  # [10,14,18,22,33]

        doc_keys = ['Doc1', 'Doc2', 'Doc3']

        real_times = np.array([t for _, t in time_pairs])

        x_positions = np.arange(len(real_times))

        plot_type_shadow('p', user_data, time_pairs, real_times, doc_keys, user_id, x_positions, Diagnosis_Result)
        plot_type_shadow('n', user_data, time_pairs, real_times, doc_keys, user_id, x_positions, Diagnosis_Result)
        # print(1)
