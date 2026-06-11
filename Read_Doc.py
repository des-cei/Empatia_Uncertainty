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


def read_doc(doc1_p, doc2_p, doc3_p, docS_p):
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
            if 'Now Current Doc:' in line:
                line_previous = doc[line_index - 1]
                line_p_split = line_previous.split(' ')
                line_p_Em_T = line_p_split[1].split('_')
                now_emotion = line_p_Em_T[0]
                timestamp = 'T_' + line_p_Em_T[1]
                user_id = 'U_' + line_p_split[-1]
                predicted_mu = float(doc[line_index + 2].split(' ')[-1])
                predicted_sigma = float(doc[line_index + 3].split(' ')[-1])
                GT_label = float(doc[line_index + 4].split(' ')[-1])
                if user_id not in Data:
                    Data[user_id] = {}
                if timestamp not in Data[user_id]:
                    Data[user_id][timestamp] = {}
                if now_emotion not in Data[user_id][timestamp]:
                    Data[user_id][timestamp][now_emotion] = {}
                if 'GT' not in Data[user_id][timestamp][now_emotion]:
                    Data[user_id][timestamp][now_emotion]['GT'] = GT_label
                if doc_id not in Data[user_id][timestamp][now_emotion]:
                    Data[user_id][timestamp][now_emotion]['Doc' + str(doc_id)] = {'predicted_mu': predicted_mu, 'predicted_sigma': predicted_sigma}

    All_plot_dict = defaultdict(lambda: defaultdict(dict))
    with open(docS_p,
              "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line_index, line in enumerate(lines):
            if 'Now: ' in line:
                line_list = line.split(' ')
                User_num = line_list[-1].replace('\n', '')
                user_id = 'U_' + User_num
                Emotion_tag_list = line_list[1].split('_')
                Emotion_tag = Emotion_tag_list[0]
                now_emotion = Emotion_tag
                Emotion_slot = Emotion_tag_list[1]
                timestamp = 'T_' + Emotion_slot
                # load the value of plot
                value_dict = {}
                Smallest_KL_list = lines[line_index + 1].split(': : ')
                Best_MU_list = lines[line_index + 2].split(': : ')
                Best_Sigma_list = lines[line_index + 3].split(': : ')
                GT_Label_list = lines[line_index + 4].split(': : ')
                GT_MU_list = lines[line_index + 5].split(': : ')
                GT_Sigma_list = lines[line_index + 6].split(': : ')

                Senior_Doc_mu = float(Best_MU_list[1])
                Senior_Doc_sigma = float(Best_Sigma_list[1])

                if Data.get(user_id, {}).get(timestamp, {}).get(now_emotion) is not None:
                    if 'Doc_senior' not in Data[user_id][timestamp][now_emotion]:
                        Data[user_id][timestamp][now_emotion]['Doc_senior'] = {'predicted_mu': Senior_Doc_mu, 'predicted_sigma': Senior_Doc_sigma}
                else:
                    continue
    return Data

if __name__ == '__main__':
    Data = read_doc('Ensemble_Doc1.log', 'Ensemble_Doc2.log', 'Ensemble_Doc3.log',
                    '/home/junjiaos/PycharmProjects/MMASH_research/Graph_process/Graph_NLL721.log')
