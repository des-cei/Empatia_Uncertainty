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
@Time : Created in 11:16 AM 2025/10/08   
@FileName: P_value_calculate.py                           
@Software: PyCharm


Introduction of this File:

"""

if __name__ == '__main__':
    data = {
        "P": [
            {"user": 9, "prob": 0.6, "dist": 5.3758, "delta": 4.0958, "panas": 9.10},
            {"user": 18, "prob": 0.8, "dist": 4.7207, "delta": 1.5521, "panas": 3.45},
            {"user": 7, "prob": 0.2, "dist": 9.1139, "delta": 8.2877, "panas": 18.42},
            {"user": 19, "prob": 0.8, "dist": 3.7064, "delta": 1.0579, "panas": 2.35},
            {"user": 13, "prob": 0.6, "dist": 3.5527, "delta": 2.1514, "panas": 4.78},
            {"user": 10, "prob": 1.0, "dist": 1.8113, "delta": 0.0000, "panas": 0.00},
            {"user": 17, "prob": 1.0, "dist": 2.3925, "delta": 0.0000, "panas": 0.00},
            {"user": 1, "prob": 0.2, "dist": 6.6953, "delta": 6.2678, "panas": 13.93},
            {"user": 20, "prob": 1.0, "dist": 3.0861, "delta": 0.0000, "panas": 0.00},
            {"user": 11, "prob": 0.4, "dist": 7.0053, "delta": 5.6244, "panas": 12.50},
            {"user": 5, "prob": 0.6, "dist": 5.2132, "delta": 3.0796, "panas": 6.84},
            {"user": 14, "prob": 0.6, "dist": 5.0988, "delta": 3.2408, "panas": 7.20},
            {"user": 4, "prob": 1.0, "dist": 3.0340, "delta": 0.0000, "panas": 0.00},
            {"user": 3, "prob": 0.2, "dist": 8.5889, "delta": 7.4329, "panas": 16.52},
            {"user": 16, "prob": 1.0, "dist": 2.5792, "delta": 0.0000, "panas": 0.00},
            {"user": 15, "prob": 0.8, "dist": 4.3071, "delta": 2.1857, "panas": 4.86},
            {"user": 6, "prob": 0.8, "dist": 3.8320, "delta": 1.6569, "panas": 3.68},
            {"user": 12, "prob": 1.0, "dist": 3.6362, "delta": 0.0000, "panas": 0.00},
            {"user": 22, "prob": 1.0, "dist": 2.8579, "delta": 0.0000, "panas": 0.00},
            {"user": 21, "prob": 1.0, "dist": 3.1939, "delta": 0.0000, "panas": 0.00},
        ],
        "N": [
            {"user": 9, "prob": 1.0, "dist": 2.7336, "delta": 0.0000, "panas": 0.00},
            {"user": 18, "prob": 0.8, "dist": 2.5464, "delta": 1.7011, "panas": 3.78},
            {"user": 7, "prob": 1.0, "dist": 1.7020, "delta": 0.0000, "panas": 0.00},
            {"user": 19, "prob": 1.0, "dist": 1.8808, "delta": 0.0000, "panas": 0.00},
            {"user": 13, "prob": 0.6, "dist": 4.7321, "delta": 3.7580, "panas": 8.35},
            {"user": 10, "prob": 0.4, "dist": 5.4706, "delta": 5.2127, "panas": 11.58},
            {"user": 17, "prob": 0.4, "dist": 3.8633, "delta": 2.8132, "panas": 6.25},
            {"user": 1, "prob": 0.8, "dist": 3.2794, "delta": 0.5925, "panas": 1.32},
            {"user": 20, "prob": 0.4, "dist": 6.7302, "delta": 5.6311, "panas": 12.51},
            {"user": 11, "prob": 0.8, "dist": 2.2636, "delta": 0.9241, "panas": 2.05},
            {"user": 5, "prob": 0.6, "dist": 3.3345, "delta": 2.8788, "panas": 6.40},
            {"user": 14, "prob": 0.6, "dist": 2.6421, "delta": 1.4376, "panas": 3.19},
            {"user": 4, "prob": 0.8, "dist": 1.7711, "delta": 0.5040, "panas": 1.12},
            {"user": 3, "prob": 0.6, "dist": 3.2563, "delta": 1.5256, "panas": 3.39},
            {"user": 16, "prob": 0.8, "dist": 2.6083, "delta": 0.8062, "panas": 1.79},
            {"user": 15, "prob": 0.6, "dist": 2.8954, "delta": 1.7053, "panas": 3.79},
            {"user": 6, "prob": 0.8, "dist": 2.6845, "delta": 0.7270, "panas": 1.62},
            {"user": 12, "prob": 0.8, "dist": 1.6825, "delta": 1.1891, "panas": 2.64},
            {"user": 22, "prob": 0.8, "dist": 2.9891, "delta": 1.8491, "panas": 4.11},
            {"user": 21, "prob": 1.0, "dist": 2.1047, "delta": 0.0000, "panas": 0.00},
        ],
        "All": [
            {"user": 9, "prob": 0.8, "dist": 4.0547, "delta": 2.0479, "panas": 4.55},
            {"user": 18, "prob": 0.8, "dist": 3.6335, "delta": 1.6266, "panas": 3.61},
            {"user": 7, "prob": 0.6, "dist": 5.4080, "delta": 4.1439, "panas": 9.21},
            {"user": 19, "prob": 0.9, "dist": 2.7936, "delta": 0.5289, "panas": 1.18},
            {"user": 13, "prob": 0.6, "dist": 4.1424, "delta": 2.9547, "panas": 6.57},
            {"user": 10, "prob": 0.7, "dist": 3.6409, "delta": 2.6064, "panas": 5.79},
            {"user": 17, "prob": 0.7, "dist": 3.1279, "delta": 1.4066, "panas": 3.13},
            {"user": 1, "prob": 0.5, "dist": 4.9874, "delta": 3.4302, "panas": 7.62},
            {"user": 20, "prob": 0.7, "dist": 4.9081, "delta": 2.8156, "panas": 6.26},
            {"user": 11, "prob": 0.6, "dist": 4.6344, "delta": 3.2743, "panas": 7.28},
            {"user": 5, "prob": 0.6, "dist": 4.2739, "delta": 2.9792, "panas": 6.62},
            {"user": 14, "prob": 0.6, "dist": 3.8704, "delta": 2.3392, "panas": 5.20},
            {"user": 4, "prob": 0.9, "dist": 2.4025, "delta": 0.2520, "panas": 0.56},
            {"user": 3, "prob": 0.4, "dist": 5.9226, "delta": 4.4792, "panas": 9.95},
            {"user": 16, "prob": 0.9, "dist": 2.5937, "delta": 0.4031, "panas": 0.90},
            {"user": 15, "prob": 0.7, "dist": 3.6012, "delta": 1.9455, "panas": 4.32},
            {"user": 6, "prob": 0.8, "dist": 3.2583, "delta": 1.1920, "panas": 2.65},
            {"user": 12, "prob": 0.9, "dist": 2.6593, "delta": 0.5945, "panas": 1.32},
            {"user": 22, "prob": 0.9, "dist": 2.9235, "delta": 0.9246, "panas": 2.05},
            {"user": 21, "prob": 1.0, "dist": 2.6493, "delta": 0.0000, "panas": 0.00},
        ],
    }

    from scipy.stats import ttest_rel
    import numpy as np

    # 示例：比较 P 和 N 的 In-Range Probability
    p_prob = {d["user"]: d["prob"] for d in data["P"]}
    n_prob = {d["user"]: d["prob"] for d in data["N"]}
    common_users = sorted(set(p_prob) & set(n_prob))

    p_arr = np.array([p_prob[u] for u in common_users])
    n_arr = np.array([n_prob[u] for u in common_users])

    t_stat, p_value = ttest_rel(p_arr, n_arr)
    print(f"Paired t-test for In-Range Probability (P vs N): p = {p_value:.4f}")

    # 示例：比较 P 和 N 的 Average Distance
    p_dist = np.array([d["dist"] for d in data["P"]])
    n_dist = np.array([d["dist"] for d in data["N"]])

    t_stat, p_value = ttest_rel(p_dist, n_dist)
    print(f"Paired t-test for Avg Distance (P vs N): p = {p_value:.4f}")

    # 示例：比较 P 和 N 的 Avg. Delta
    p_dist = np.array([d["delta"] for d in data["P"]])
    n_dist = np.array([d["delta"] for d in data["N"]])

    t_stat, p_value = ttest_rel(p_dist, n_dist)
    print(f"Paired t-test for Avg. Delta (P vs N): p = {p_value:.4f}")

