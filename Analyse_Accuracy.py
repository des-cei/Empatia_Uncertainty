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
@Time : Created in 10:47 AM 2025/07/28   
@FileName: Analyse_Accuracy.py                           
@Software: PyCharm


Introduction of this File:

"""

import os.path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

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
            all_delta_list = []
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
                    # Judging the label if in the range
                    now_plot_judging = 0
                    abs_list = []
                    delta_list = []
                    for index_label, real_label in enumerate(label_real_list):
                        mu = mu_predicted_list[index_label]
                        sigma = sigma_predicted_list[index_label]
                        abs_now_label = abs(real_label - mu_predicted_list[index_label])
                        if (real_label >= mu - sigma) and (real_label <= mu + sigma):
                            now_plot_judging += 1
                            all_plot_judging += 1
                        else:
                            delta_list.append(abs_now_label)
                            all_delta_list.append(abs_now_label)
                        abs_list.append(abs_now_label)
                        all_abs_list.append(abs_now_label)
                    mean_val = (np.sum(delta_list) / 5) if delta_list else 0
                    print('User: ' + str(user_data[0]) + ' -- Emotion: ' + str.upper(plot_data[0]) +
                          ' -- In-Range Probability: ' + str(now_plot_judging / 5) + '\n' +
                          'Average Distance: ' + str(np.mean(abs_list)) + '\n' +
                          'Average Delta: ' + str(mean_val) + '\n' +
                          'Average Delta PANAS: ' + str(f"{(mean_val / 45) * 100:.2f}%") + '\n')
            all_mean_val = (np.sum(all_delta_list) / 10) if all_delta_list else 0
            print('All User: ' + str(user_data[0]) + ' -- All In-Range Probability: ' + str(all_plot_judging / 10) + '\n' +
                  'Average Distance: ' + str(np.mean(all_abs_list)) + '\n' +
                  'Average Delta: ' + str(all_mean_val) + '\n' +
                  'Average Delta PANAS: ' + str(f"{(all_mean_val / 45) * 100:.2f}%") + '\n' + '\n')
