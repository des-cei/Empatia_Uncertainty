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
@Time : Created in 11:33 AM 2025/07/21   
@FileName: Prediction_plots.py                           
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

                    # drawing the plot
                    x = np.array(x_list)
                    mu_pred = np.array(mu_predicted_list)
                    sigma_pred = np.array(sigma_predicted_list)
                    mu_gt = np.array(mu_GT_list)
                    sigma_gt = np.array(sigma_GT_list)
                    label_real = np.array(label_real_list)

                    plt.figure(figsize=(12, 6))

                    plt.plot(x, mu_pred, label='Predicted μ', color='blue')
                    plt.fill_between(x, mu_pred - sigma_pred, mu_pred + sigma_pred, color='blue', alpha=0.2,
                                     label='Predicted ±σ')

                    # plt.plot(x, mu_gt, label='GT μ', color='green')
                    # plt.fill_between(x, mu_gt - sigma_gt, mu_gt + sigma_gt, color='green', alpha=0.2, label='GT ±σ')

                    plt.plot(x, label_real, label='Real Label', color='red', linestyle='--', marker='o')

                    plt.xlabel('Time Slot')
                    plt.ylabel('Panas')
                    plt.title('User: ' + str(user_data[0]) + ' -- Emotion: ' + str.upper(plot_data[0]) +
                              ' -- Average Emotion KL: ' + str(round(np.mean(kl_list), 8)) + '\n' +
                              'User KL divergence: ' + str(user_data[1]['KL']))
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    # plt.show()
                    plt.savefig(os.path.join('Plot_Prediction_noGT', ('User_' + str(user_data[0]) +
                                                                 '_Emotion_' + str.upper(plot_data[0]) + '.png')))
                    print(1)
