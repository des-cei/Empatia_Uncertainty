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
@Time : Created in 12:43 PM 2025/12/11
@FileName: Read_Doc.py
@Software: PyCharm


Introduction of this File:

"""
from collections import defaultdict
import json

import numpy as np


def load_docs(doc_paths):
    """
    输入:  doc_paths = [path1, path2, path3, path4]
    输出:  四个 list，对应四个文件的每一行（自动去掉换行符）
    """
    doc_lists = []

    for path in doc_paths:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            doc_lists.append(lines)

    return doc_lists


def read_doc_threshold(doc1_p, doc2_p, doc3_p):
    Doc1_path = doc1_p
    Doc2_path = doc2_p
    Doc3_path = doc3_p

    doc1, doc2, doc3 = load_docs([
        Doc1_path,
        Doc2_path,
        Doc3_path
    ])

    Data = {}
    doc_lists = [doc1, doc2, doc3]

    for doc_id, doc in enumerate(doc_lists, start=1):
        for line_index, line in enumerate(doc):
            if 'Final Validation Results' in line:
                U1_line_list = doc[line_index + 1].split(' ')
                U2_line_list = doc[line_index + 2].split(' ')
                U3_line_list = doc[line_index + 3].split(' ')
                U4_line_list = doc[line_index + 4].split(' ')

                U1_num = int(U1_line_list[5].replace(':', ''))
                U2_num = int(U2_line_list[5].replace(':', ''))
                U3_num = int(U3_line_list[5].replace(':', ''))
                U4_num = int(U4_line_list[5].replace(':', ''))

                U1_mu = float(U1_line_list[8].replace(';', ''))
                U2_mu = float(U2_line_list[8].replace(';', ''))
                U3_mu = float(U3_line_list[8].replace(';', ''))
                U4_mu = float(U4_line_list[8].replace(';', ''))

                U1_sigma = float(U1_line_list[11].replace(';', ''))
                U2_sigma = float(U2_line_list[11].replace(';', ''))
                U3_sigma = float(U3_line_list[11].replace(';', ''))
                U4_sigma = float(U4_line_list[11].replace(';', ''))

                U1_y = int(float(U1_line_list[14].rstrip('.')))
                U2_y = int(float(U2_line_list[14].rstrip('.')))
                U3_y = int(float(U3_line_list[14].rstrip('.')))
                U4_y = int(float(U4_line_list[14].rstrip('.')))

                line_info_list = doc[line_index + 5].split(' ')

                line_p_Em_T = line_info_list[1].split('_')
                now_emotion = line_p_Em_T[0]
                timestamp = 'T_' + line_p_Em_T[1]
                user_id = 'U_' + line_info_list[-1]

                if user_id not in Data:
                    Data[user_id] = {}
                if timestamp not in Data[user_id]:
                    Data[user_id][timestamp] = {}
                if now_emotion not in Data[user_id][timestamp]:
                    Data[user_id][timestamp][now_emotion] = {}

                # --- 新增代码开始 ---

                # 1. 将解析出的四组数据打包成一个列表，方便遍历
                # 这样不用写四遍相同的赋值逻辑
                batch_results = [
                    {'num': U1_num, 'mu': U1_mu, 'sigma': U1_sigma, 'y': U1_y},
                    {'num': U2_num, 'mu': U2_mu, 'sigma': U2_sigma, 'y': U2_y},
                    {'num': U3_num, 'mu': U3_mu, 'sigma': U3_sigma, 'y': U3_y},
                    {'num': U4_num, 'mu': U4_mu, 'sigma': U4_sigma, 'y': U4_y}
                ]

                # 2. 遍历这4个User的数据并填入字典
                for res in batch_results:
                    # 构造 Sub User Key (例如: U_13)
                    sub_user_key = 'V_' + str(res['num'])

                    # 构造 Doc Key (例如: Doc1)
                    doc_key = 'Doc' + str(doc_id)

                    # 确保 sub_user_key 在当前路径下存在
                    target_path = Data[user_id][timestamp][now_emotion]
                    if sub_user_key not in target_path:
                        target_path[sub_user_key] = {}

                    # 3. 在对应的 Doc 下面写入 mu, sigma, y
                    target_path[sub_user_key][doc_key] = {
                        'mu': res['mu'],
                        'sigma': res['sigma'],
                        'y': res['y']
                    }
    return Data


def calculate_uncertainty_thresholds(data_dict, mode, percentile=90):

    # 初始化结果字典，用于存储每个 emotion 对应的阈值
    thresholds_dict = {}

    # --- 1. 遍历多层嵌套字典 ---
    for target_u, time_dict in data_dict.items():
        # 确保结果字典中有对应的 key
        if target_u not in thresholds_dict:
            thresholds_dict[target_u] = {}

        for timestamp, emo_dict in time_dict.items():
            if timestamp not in thresholds_dict[target_u]:
                thresholds_dict[target_u][timestamp] = {}

            # --- 关键修改点：在此循环开始时处理特定 emotion ---
            for emotion, val_users_dict in emo_dict.items():

                # 【修改点 1】: 在进入 emotion 循环时，清空收集列表
                # 这样收集到的只是当前 user、当前 timestamp、当前 emotion 下的数据
                current_aleatoric_scores = []
                current_epistemic_scores = []

                # --- 2. 处理该 emotion 下的每个验证样本 ---
                for val_u_key, docs_data in val_users_dict.items():

                    # 跳过非字典项 (比如之前存的 'GT' 标签)
                    if not isinstance(docs_data, dict):
                        continue

                    mus = []
                    sigmas = []

                    for doc_key, metrics in docs_data.items():
                        if isinstance(metrics, dict):
                            mus.append(metrics['mu'])
                            sigmas.append(metrics['sigma'])

                    if len(mus) > 0:
                        mus = np.array(mus)
                        sigmas = np.array(sigmas)

                        u_aleatoric = np.mean(sigmas ** 2)

                        # Using mean to calculate Var
                        if mode == 'average':
                            u_epistemic = np.var(mus)

                        # Using median to calculate Var
                        if mode == 'median':
                            u_epistemic = np.mean((mus - np.median(mus)) ** 2)

                        current_aleatoric_scores.append(u_aleatoric)
                        current_epistemic_scores.append(u_epistemic)

                if current_aleatoric_scores:
                    t_a = np.percentile(current_aleatoric_scores, percentile)
                    t_e = np.percentile(current_epistemic_scores, percentile)
                else:
                    t_a, t_e = 0.0, 0.0

                thresholds_dict[target_u][timestamp][emotion] = {
                    't_aleatoric': t_a,
                    't_epistemic': t_e,
                    'sample_count': len(current_aleatoric_scores)  # 方便debug，记录用了多少个样本算的
                }

    return thresholds_dict

if __name__ == '__main__':
    Data = read_doc_threshold('Ensemble_Doc1.log', 'Ensemble_Doc2.log', 'Ensemble_Doc3.log')
    thresholds_dict = calculate_uncertainty_thresholds(Data, mode='median', percentile=90)

