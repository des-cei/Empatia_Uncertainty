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
@Time : Created in 5:44 PM 2026/01/22   
@FileName: Diagnose_Docs.py                           
@Software: PyCharm


Introduction of this File:

"""
import json

from Read_Doc import read_doc
from Read_threshold import read_doc_threshold, calculate_uncertainty_thresholds

import numpy as np
import json


def calculate_uncertainty_dict(source_data, mode):
    """
    遍历 source_data，仅使用 Doc1, Doc2, Doc3 计算不确定性，
    并返回一个新的字典存放结果。
    """
    # 初始化结果字典
    uncertainty_dict = {}

    # 1. 遍历 User 层 (U_9)
    for user_id, time_dict in source_data.items():
        if user_id not in uncertainty_dict:
            uncertainty_dict[user_id] = {}

        # 2. 遍历 Timestamp 层 (T_10)
        for timestamp, emo_dict in time_dict.items():
            if timestamp not in uncertainty_dict[user_id]:
                uncertainty_dict[user_id][timestamp] = {}

            # 3. 遍历 Emotion 层 (p, n)
            for emotion, content_dict in emo_dict.items():

                # --- 数据收集阶段 ---
                mus = []
                sigmas = []

                target_docs = ['Doc1', 'Doc2', 'Doc3']

                for doc_name in target_docs:
                    if doc_name in content_dict:
                        data = content_dict[doc_name]
                        mus.append(data['predicted_mu'])
                        sigmas.append(data['predicted_sigma'])

                if len(mus) > 0:
                    mus_arr = np.array(mus)
                    sigmas_arr = np.array(sigmas)

                    now_aleatoric = np.mean(sigmas_arr ** 2)

                    # Using mean to calculate Var
                    if mode == 'average':
                        now_epistemic = np.var(mus_arr)

                    # Using median to calculate Var
                    if mode == 'median':
                        now_epistemic = np.mean((mus_arr - np.median(mus_arr)) ** 2)

                    # Using Median Absolute Deviation
                    # now_epistemic = np.median(np.abs(mus_arr - np.median(mus_arr)))

                    uncertainty_dict[user_id][timestamp][emotion] = {
                        'now_aleatoric': float(now_aleatoric),
                        'now_epistemic': float(now_epistemic)
                    }
                else:
                    # 如果该条目下没有 Doc1-3 的数据，给个默认值或跳过
                    uncertainty_dict[user_id][timestamp][emotion] = {
                        'now_aleatoric': 0.0,
                        'now_epistemic': 0.0
                    }

    return uncertainty_dict


def generate_diagnostic_dict(measurements, thresholds):
    diagnostic_report = {}

    for user_id, time_dict in measurements.items():
        if user_id not in diagnostic_report:
            diagnostic_report[user_id] = {}

        for timestamp, emo_dict in time_dict.items():
            if timestamp not in diagnostic_report[user_id]:
                diagnostic_report[user_id][timestamp] = {}

            for emotion, metrics in emo_dict.items():

                now_a = metrics.get('now_aleatoric', 0)
                now_e = metrics.get('now_epistemic', 0)

                try:
                    t_data = thresholds[user_id][timestamp][emotion]
                    t_a = t_data['t_aleatoric']
                    t_e = t_data['t_epistemic']
                except KeyError:
                    diagnostic_report[user_id][timestamp][emotion] = {
                        "status": "Unknown",
                        "reason": "Threshold not found for this key"
                    }
                    continue

                status_code = ""
                status_msg = ""
                reason = ""

                if now_e > t_e:
                    status_code = "Case 2"
                    status_msg = "System Failure (High Disagreement)"
                    reason = f"Epistemic ({now_e:.2f}) > Threshold ({t_e:.2f})"

                elif now_a > t_a:
                    status_code = "Case 1"
                    status_msg = "Data Complexity (High Uncertainty)"
                    reason = f"Aleatoric ({now_a:.2f}) > Threshold ({t_a:.2f})"

                else:
                    status_code = "Case 0"
                    status_msg = "Trusted Diagnosis"
                    reason = "Values within normal range"

                diagnostic_report[user_id][timestamp][emotion] = {
                    "status_code": status_code,  # Case 1, Case 2, Case 0
                    "message": status_msg,
                    "details": {
                        "now_aleatoric": now_a,
                        "threshold_aleatoric": t_a,
                        "now_epistemic": now_e,
                        "threshold_epistemic": t_e
                    },
                    "reason_short": reason
                }

    return diagnostic_report


def print_console_table(diagnosis_result):
    """
    直接在控制台打印带有颜色的诊断状态表。
    """

    # --- 1. 定义控制台颜色代码 (ANSI Escape Codes) ---
    class Colors:
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'  # Case 0
        YELLOW = '\033[93m'  # Case 1
        RED = '\033[91m'  # Case 2
        ENDC = '\033[0m'  # Reset
        BOLD = '\033[1m'

    # --- 2. 辅助函数：格式化单元格内容 ---
    def format_cell(status_code):
        if status_code == "Case 0":
            return f"{Colors.GREEN}Case 0 (Trusted){Colors.ENDC}"
        elif status_code == "Case 1":
            return f"{Colors.YELLOW}Case 1 (Data Unc.){Colors.ENDC}"
        elif status_code == "Case 2":
            return f"{Colors.RED}Case 2 (Sys. Fail){Colors.ENDC}"
        else:
            return "N/A"

    # --- 3. 打印表头 ---
    # 使用 f-string 设置固定宽度: <10 表示左对齐占10格, <25 表示左对齐占25格
    print("-" * 70)
    print(f"{Colors.BOLD}{'User ID':<10} {'Time Step':<12} {'Emotion (p)':<25} {'Emotion (n)':<25}{Colors.ENDC}")
    print("-" * 70)

    # --- 4. 排序并遍历数据 ---
    # 按照 User 数字大小排序
    sorted_users = sorted(diagnosis_result.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else x)

    for user_id in sorted_users:
        time_data = diagnosis_result[user_id]

        # 按照 Time 数字大小排序
        sorted_times = sorted(time_data.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else x)

        for i, timestamp in enumerate(sorted_times):
            emotions = time_data[timestamp]

            status_p = emotions.get('p', {}).get('status_code', 'N/A')
            status_n = emotions.get('n', {}).get('status_code', 'N/A')

            text_p = format_cell(status_p)
            text_n = format_cell(status_n)

            display_user = user_id if i == 0 else ""

            # 简单对齐方案：
            print(f"{display_user:<10} {timestamp:<12} {text_p:<35} {text_n:<35}")

        # 每个 User 结束后加个空行或分割线
        print("-" * 70)


def analyze_distance(diagnosis_result):
    """
    在原有的 Diagnosis_Result 基础上，增加定量分析字段。
    """
    # 严重程度阈值设定 (例如：超标 10% 算 Low, 50% 算 High)
    SEVERITY_LEVELS = [
        (0.0, "Safe"),  # 未超标
        (0.1, "Low"),  # 超标 < 10%
        (0.5, "Medium"),  # 超标 10% - 50%
        (1.0, "High"),  # 超标 50% - 100%
        (float('inf'), "Critical")  # 超标 > 100%
    ]

    def get_severity(ratio):
        if ratio <= 0: return "Safe"
        for limit, label in SEVERITY_LEVELS:
            if ratio <= limit:
                return label
        return "Critical"

    # 遍历字典
    analysis_report = {}

    for user_id, time_dict in diagnosis_result.items():
        analysis_report[user_id] = {}
        for timestamp, emo_dict in time_dict.items():
            analysis_report[user_id][timestamp] = {}
            for emotion, info in emo_dict.items():

                # 获取原始数据
                details = info.get('details', {})
                if not details: continue

                # 1. 提取数值
                now_a = details.get('now_aleatoric', 0)
                thr_a = details.get('threshold_aleatoric', 1e-6)  # 防止除零

                now_e = details.get('now_epistemic', 0)
                thr_e = details.get('threshold_epistemic', 1e-6)

                # 2. 计算 Aleatoric 距离 (针对 Case 1)
                diff_a = now_a - thr_a
                ratio_a = diff_a / thr_a
                severity_a = get_severity(ratio_a)

                # 3. 计算 Epistemic 距离 (针对 Case 2)
                diff_e = now_e - thr_e
                ratio_e = diff_e / thr_e
                severity_e = get_severity(ratio_e)

                # 4. 存入新结构
                analysis_report[user_id][timestamp][emotion] = {
                    "original_status": info.get('status_code'),
                    "quantitative": {
                        "aleatoric": {
                            "val": now_a,
                            "thr": thr_a,
                            "diff": round(diff_a, 4),
                            "ratio_percent": round(ratio_a * 100, 2),  # 转换为百分比
                            "severity": severity_a
                        },
                        "epistemic": {
                            "val": now_e,
                            "thr": thr_e,
                            "diff": round(diff_e, 4),
                            "ratio_percent": round(ratio_e * 100, 2),
                            "severity": severity_e
                        }
                    }
                }

    return analysis_report

if __name__ == '__main__':
    Data_train = read_doc('Ensemble_Doc1.log', 'Ensemble_Doc2.log', 'Ensemble_Doc3.log',
                    '/home/junjiaos/PycharmProjects/MMASH_research/Graph_process/Graph_NLL721.log')
    Data_threshold = read_doc_threshold('Ensemble_Doc1.log', 'Ensemble_Doc2.log', 'Ensemble_Doc3.log')
    thresholds_dict = calculate_uncertainty_thresholds(Data_threshold, mode='median', percentile=80)
    Uncertainty_Data = calculate_uncertainty_dict(Data_train, mode='median')
    Diagnosis_Result = generate_diagnostic_dict(Uncertainty_Data, thresholds_dict)
    # print_console_table(Diagnosis_Result)
    report = analyze_distance(Diagnosis_Result)
    print(json.dumps(report, indent=4))
    # print(json.dumps(Diagnosis_Result, indent=4))
