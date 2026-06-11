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
@Time : Created in 4:00 PM 2025/12/11   
@FileName: Doc_ensemble.py                           
@Software: PyCharm


Introduction of this File:

"""
import os

import numpy as np
from matplotlib import pyplot as plt

from Ensembles_1215.Read_Doc import read_doc
import matplotlib.patches as mpatches
from Ensembles_Threshold.Diagnose_Docs import calculate_uncertainty_dict, generate_diagnostic_dict
from Ensembles_Threshold.Read_threshold import read_doc_threshold, calculate_uncertainty_thresholds


def parse_time_keys(d):
    times = []
    for k in d.keys():
        t = int(k.split('_')[1])
        times.append((k, t))
    return sorted(times, key=lambda x: x[1])


def plot_type_shadow(target_type, time_pairs, real_times, user_id, x_positions, Diagnosis_Result):
    # 创建两个子图，共享 X 轴
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # =========================================================
    # 0. 数据提取 (Data Extraction)
    # =========================================================
    # 仅提取不确定性指标，不再提取 GT
    now_aleatoric_list = []
    thresh_aleatoric_list = []
    now_epistemic_list = []
    thresh_epistemic_list = []

    for t_key, _ in time_pairs:
        try:
            # 从 Diagnosis_Result 中提取
            details = Diagnosis_Result[user_id][t_key][target_type]["details"]

            now_aleatoric_list.append(details.get("now_aleatoric", np.nan))
            thresh_aleatoric_list.append(details.get("threshold_aleatoric", np.nan))

            now_epistemic_list.append(details.get("now_epistemic", np.nan))
            thresh_epistemic_list.append(details.get("threshold_epistemic", np.nan))
        except KeyError:
            # 数据缺失处理
            now_aleatoric_list.append(np.nan)
            thresh_aleatoric_list.append(np.nan)
            now_epistemic_list.append(np.nan)
            thresh_epistemic_list.append(np.nan)

    # =========================================================
    # 1. 背景阴影绘制逻辑 (通用)
    # =========================================================
    case_colors = {
        "Case 0": "#eaffea",  # 浅绿
        "Case 1": "#fff8c4",  # 浅黄
        "Case 2": "#ffdddd"   # 浅红
    }
    default_color = "white"
    half_width = 0.4  # 背景条宽度控制 (留空隙)

    def draw_background(ax_target):
        for (t_key, _), x_pos in zip(time_pairs, x_positions):
            status_code = "Unknown"
            try:
                diag_info = Diagnosis_Result[user_id][t_key][target_type]
                status_code = diag_info.get("status_code", "Unknown")
            except KeyError:
                pass

            bg_color = case_colors.get(status_code, default_color)
            ax_target.axvspan(
                x_pos - half_width,
                x_pos + half_width,
                color=bg_color,
                alpha=0.6,
                zorder=0,
                edgecolor=None
            )

    # 为两个子图都画上背景
    draw_background(ax1)
    draw_background(ax2)

    # =========================================================
    # 2. 上图: Epistemic Analysis (System/Model)
    # =========================================================
    # 使用 紫色系 代表 Epistemic
    color_epi = '#9467bd' # 紫色

    ax1.plot(x_positions, now_epistemic_list, 's-', color=color_epi, linewidth=2, label="Epistemic (Current)", zorder=3)
    ax1.plot(x_positions, thresh_epistemic_list, '--', color=color_epi, linewidth=2, alpha=0.7, label="Epistemic Threshold", zorder=3)

    ax1.set_ylabel("Variance Value")
    ax1.set_title(f"Epistemic Uncertainty (System Disagreement) - User: {user_id} - Emotion: {target_type}")
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 制作 Ax1 图例 (包含背景含义)
    handles1, _ = ax1.get_legend_handles_labels()
    # 添加背景色块说明
    patch_c2 = mpatches.Patch(color=case_colors['Case 2'], label='Case 2: Expert Referral (Triggered by Epistemic)')
    handles1.append(patch_c2)

    ax1.legend(handles=handles1, loc='upper left', fontsize='small')

    # =========================================================
    # 3. 下图: Aleatoric Analysis (Data)
    # =========================================================
    # 使用 蓝色系 代表 Aleatoric
    color_ale = '#1f77b4' # 蓝色

    ax2.plot(x_positions, now_aleatoric_list, 'o-', color=color_ale, linewidth=2, label="Aleatoric (Current)", zorder=3)
    ax2.plot(x_positions, thresh_aleatoric_list, '--', color=color_ale, linewidth=2, alpha=0.7, label="Aleatoric Threshold", zorder=3)

    ax2.set_ylabel("Variance Value")
    ax2.set_xlabel("Time Step")
    ax2.set_title(f"Aleatoric Uncertainty (Data Complexity) - User: {user_id} - Emotion: {target_type}")
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 设置 X 轴刻度
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([str(t) for t in real_times])

    # 制作 Ax2 图例
    handles2, _ = ax2.get_legend_handles_labels()
    # 添加背景色块说明
    patch_c1 = mpatches.Patch(color=case_colors['Case 1'], label='Case 1: Data Uncertainty (Triggered by Aleatoric)')
    patch_c0 = mpatches.Patch(color=case_colors['Case 0'], label='Case 0: Trusted')
    handles2.extend([patch_c1, patch_c0])

    ax2.legend(handles=handles2, loc='upper left', fontsize='small')

    # =========================================================
    # 4. 调整布局
    # =========================================================
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join('Plot_ensemble', ('User_' + str(user_id) + '_Emotion_' + str(target_type.upper()))))


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
        time_pairs = parse_time_keys(user_data)
        real_times = np.array([t for _, t in time_pairs])  # [10,14,18,22,33]
        x_positions = np.arange(len(real_times))  # 等距坐标: 0,1,2,3,4

        plot_type_shadow('p', time_pairs, real_times, user_id, x_positions, Diagnosis_Result)
        plot_type_shadow('n', time_pairs, real_times, user_id, x_positions, Diagnosis_Result)

        # print(1)
