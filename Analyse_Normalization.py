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
@Time : Created in 5:23 PM 2025/07/28   
@FileName: Analyse_Normalization.py                           
@Software: PyCharm


Introduction of this File:

"""

import os.path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from dtw import dtw
from fastdtw import fastdtw


if __name__ == '__main__':
    All_plot_dict = defaultdict(lambda: defaultdict(dict))
    with open("/home/junjiaos/PycharmProjects/MMASH_research/Graph_process/Graph_NLL721.log",
              "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line_index, line in enumerate(lines):
            if 'Now: ' in line:
                line_list = line.split(' ')
                User_num = line_list[-1].replace('\n', '')
                Emotion_tag_list = line_list[1].split('_')
                Emotion_tag = Emotion_tag_list[0]
                Emotion_slot = Emotion_tag_list[1]
                # load the value of plot
                value_dict = {}
                Smallest_KL_list = lines[line_index + 1].split(': : ')
                Best_MU_list = lines[line_index + 2].split(': : ')
                Best_Sigma_list = lines[line_index + 3].split(': : ')
                GT_Label_list = lines[line_index + 4].split(': : ')
                GT_MU_list = lines[line_index + 5].split(': : ')
                GT_Sigma_list = lines[line_index + 6].split(': : ')
                value_dict[Smallest_KL_list[0]] = float(Smallest_KL_list[1])
                value_dict[Best_MU_list[0]] = float(Best_MU_list[1])
                value_dict[Best_Sigma_list[0]] = float(Best_Sigma_list[1])
                value_dict[GT_Label_list[0]] = float(GT_Label_list[1])
                value_dict[GT_MU_list[0]] = float(GT_MU_list[1])
                value_dict[GT_Sigma_list[0]] = float(GT_Sigma_list[1])

                All_plot_dict[User_num][Emotion_tag][Emotion_slot] = value_dict
            if 'Average KL' in line:
                All_plot_dict[User_num]['KL'] = float(line.replace('Average KL: : ', ''))
    for user_data in All_plot_dict.items():
        if len(user_data[1]['p']) == 5 and len(user_data[1]['n']) == 5:
            all_plot_judging = 0
            all_abs_list = []
            for plot_data in user_data[1].items():
                if plot_data[0] == 'p' or plot_data[0] == 'n':
                    mu_predicted_list = []
                    sigma_predicted_list = []
                    mu_GT_list = []
                    sigma_GT_list = []
                    label_real_list = []
                    x_list = []
                    kl_list = []
                    for plot_value in plot_data[1].items():
                        x_list.append(plot_value[0])
                        mu_predicted_list.append(plot_value[1]['Best MU'])
                        sigma_predicted_list.append(plot_value[1]['Best Sigma'])
                        mu_GT_list.append(plot_value[1]['GT MU'])
                        sigma_GT_list.append(plot_value[1]['GT Sigma'])
                        label_real_list.append(plot_value[1]['GT Label'])
                        kl_list.append(plot_value[1]['Smallest KL'])
                    # dtw
                    if len(mu_predicted_list) > 1:
                        mu_pred = np.array(mu_predicted_list).reshape(-1, 1)
                        sigma_pred = np.array(sigma_predicted_list).reshape(-1, 1)
                        label_real = np.array(label_real_list).reshape(-1, 1)

                        scaler = MinMaxScaler()
                        mu_pred_norm = scaler.fit_transform(mu_pred).flatten()
                        sigma_pred_norm = scaler.fit_transform(sigma_pred).flatten()
                        label_real_norm = scaler.fit_transform(label_real).flatten()

                        mu_array = np.array(mu_pred_norm).flatten().astype(float)
                        label_array = np.array(label_real_norm).flatten().astype(float)
                        sigma_array = np.array(sigma_pred_norm).flatten().astype(float)

                        mu_dtw_distance, _ = fastdtw(mu_array, label_array, dist=2)

                        print(f"User: {user_data[0]}, Type: {plot_data[0]}")
                        print(f"  DTW(mu_pred vs label_real): {mu_dtw_distance:.4f}")
                        print('\n')

                        plt.figure(figsize=(10, 4))
                        plt.plot(x_list, mu_pred_norm, label='mu_pred (normalized)', marker='o')
                        plt.plot(x_list, label_real_norm, label='label_real (normalized)', marker='s')
                        plt.title(
                            f"User {user_data[0]} | {plot_data[0]} | DTW(mu): {mu_dtw_distance:.3f}")
                        plt.xlabel("X")
                        plt.ylabel("Normalized Value")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        # plt.show()
                        plt.savefig(os.path.join('DTW_plot', ('User_' + str(user_data[0]) +
                                                                          '_Emotion_' + str.upper(
                                    plot_data[0]) + '.png')))
                        print(1)
