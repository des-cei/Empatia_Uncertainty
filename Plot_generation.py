import os
import numpy as np
import matplotlib.pyplot as plt

# ========== 数据：直接写在代码里 ==========
data = {
    "P": [
        {"user": 9,  "prob": 0.6, "dist": 5.3758, "delta": 4.0958, "panas": 9.10},
        {"user": 18, "prob": 0.8, "dist": 4.7207, "delta": 1.5521, "panas": 3.45},
        {"user": 7,  "prob": 0.2, "dist": 9.1139, "delta": 8.2877, "panas": 18.42},
        {"user": 19, "prob": 0.8, "dist": 3.7064, "delta": 1.0579, "panas": 2.35},
        {"user": 13, "prob": 0.6, "dist": 3.5527, "delta": 2.1514, "panas": 4.78},
        {"user": 10, "prob": 1.0, "dist": 1.8113, "delta": 0.0000, "panas": 0.00},
        {"user": 17, "prob": 1.0, "dist": 2.3925, "delta": 0.0000, "panas": 0.00},
        {"user": 1,  "prob": 0.2, "dist": 6.6953, "delta": 6.2678, "panas": 13.93},
        {"user": 20, "prob": 1.0, "dist": 3.0861, "delta": 0.0000, "panas": 0.00},
        {"user": 11, "prob": 0.4, "dist": 7.0053, "delta": 5.6244, "panas": 12.50},
        {"user": 5,  "prob": 0.6, "dist": 5.2132, "delta": 3.0796, "panas": 6.84},
        {"user": 14, "prob": 0.6, "dist": 5.0988, "delta": 3.2408, "panas": 7.20},
        {"user": 4,  "prob": 1.0, "dist": 3.0340, "delta": 0.0000, "panas": 0.00},
        {"user": 3,  "prob": 0.2, "dist": 8.5889, "delta": 7.4329, "panas": 16.52},
        {"user": 16, "prob": 1.0, "dist": 2.5792, "delta": 0.0000, "panas": 0.00},
        {"user": 15, "prob": 0.8, "dist": 4.3071, "delta": 2.1857, "panas": 4.86},
        {"user": 6,  "prob": 0.8, "dist": 3.8320, "delta": 1.6569, "panas": 3.68},
        {"user": 12, "prob": 1.0, "dist": 3.6362, "delta": 0.0000, "panas": 0.00},
        {"user": 22, "prob": 1.0, "dist": 2.8579, "delta": 0.0000, "panas": 0.00},
        {"user": 21, "prob": 1.0, "dist": 3.1939, "delta": 0.0000, "panas": 0.00},
    ],
    "N": [
        {"user": 9,  "prob": 1.0, "dist": 2.7336, "delta": 0.0000, "panas": 0.00},
        {"user": 18, "prob": 0.8, "dist": 2.5464, "delta": 1.7011, "panas": 3.78},
        {"user": 7,  "prob": 1.0, "dist": 1.7020, "delta": 0.0000, "panas": 0.00},
        {"user": 19, "prob": 1.0, "dist": 1.8808, "delta": 0.0000, "panas": 0.00},
        {"user": 13, "prob": 0.6, "dist": 4.7321, "delta": 3.7580, "panas": 8.35},
        {"user": 10, "prob": 0.4, "dist": 5.4706, "delta": 5.2127, "panas": 11.58},
        {"user": 17, "prob": 0.4, "dist": 3.8633, "delta": 2.8132, "panas": 6.25},
        {"user": 1,  "prob": 0.8, "dist": 3.2794, "delta": 0.5925, "panas": 1.32},
        {"user": 20, "prob": 0.4, "dist": 6.7302, "delta": 5.6311, "panas": 12.51},
        {"user": 11, "prob": 0.8, "dist": 2.2636, "delta": 0.9241, "panas": 2.05},
        {"user": 5,  "prob": 0.6, "dist": 3.3345, "delta": 2.8788, "panas": 6.40},
        {"user": 14, "prob": 0.6, "dist": 2.6421, "delta": 1.4376, "panas": 3.19},
        {"user": 4,  "prob": 0.8, "dist": 1.7711, "delta": 0.5040, "panas": 1.12},
        {"user": 3,  "prob": 0.6, "dist": 3.2563, "delta": 1.5256, "panas": 3.39},
        {"user": 16, "prob": 0.8, "dist": 2.6083, "delta": 0.8062, "panas": 1.79},
        {"user": 15, "prob": 0.6, "dist": 2.8954, "delta": 1.7053, "panas": 3.79},
        {"user": 6,  "prob": 0.8, "dist": 2.6845, "delta": 0.7270, "panas": 1.62},
        {"user": 12, "prob": 0.8, "dist": 1.6825, "delta": 1.1891, "panas": 2.64},
        {"user": 22, "prob": 0.8, "dist": 2.9891, "delta": 1.8491, "panas": 4.11},
        {"user": 21, "prob": 1.0, "dist": 2.1047, "delta": 0.0000, "panas": 0.00},
    ],
    "All": [
        {"user": 9,  "prob": 0.8, "dist": 4.0547, "delta": 2.0479, "panas": 4.55},
        {"user": 18, "prob": 0.8, "dist": 3.6335, "delta": 1.6266, "panas": 3.61},
        {"user": 7,  "prob": 0.6, "dist": 5.4080, "delta": 4.1439, "panas": 9.21},
        {"user": 19, "prob": 0.9, "dist": 2.7936, "delta": 0.5289, "panas": 1.18},
        {"user": 13, "prob": 0.6, "dist": 4.1424, "delta": 2.9547, "panas": 6.57},
        {"user": 10, "prob": 0.7, "dist": 3.6409, "delta": 2.6064, "panas": 5.79},
        {"user": 17, "prob": 0.7, "dist": 3.1279, "delta": 1.4066, "panas": 3.13},
        {"user": 1,  "prob": 0.5, "dist": 4.9874, "delta": 3.4302, "panas": 7.62},
        {"user": 20, "prob": 0.7, "dist": 4.9081, "delta": 2.8156, "panas": 6.26},
        {"user": 11, "prob": 0.6, "dist": 4.6344, "delta": 3.2743, "panas": 7.28},
        {"user": 5,  "prob": 0.6, "dist": 4.2739, "delta": 2.9792, "panas": 6.62},
        {"user": 14, "prob": 0.6, "dist": 3.8704, "delta": 2.3392, "panas": 5.20},
        {"user": 4,  "prob": 0.9, "dist": 2.4025, "delta": 0.2520, "panas": 0.56},
        {"user": 3,  "prob": 0.4, "dist": 5.9226, "delta": 4.4792, "panas": 9.95},
        {"user": 16, "prob": 0.9, "dist": 2.5937, "delta": 0.4031, "panas": 0.90},
        {"user": 15, "prob": 0.7, "dist": 3.6012, "delta": 1.9455, "panas": 4.32},
        {"user": 6,  "prob": 0.8, "dist": 3.2583, "delta": 1.1920, "panas": 2.65},
        {"user": 12, "prob": 0.9, "dist": 2.6593, "delta": 0.5945, "panas": 1.32},
        {"user": 22, "prob": 0.9, "dist": 2.9235, "delta": 0.9246, "panas": 2.05},
        {"user": 21, "prob": 1.0, "dist": 2.6493, "delta": 0.0000, "panas": 0.00},
    ],
}

# ========== 工具：获取三组共同存在的 user，并按升序 ==========
def get_common_users():
    users_sets = [set(r["user"] for r in data[emo]) for emo in ["P", "N", "All"]]
    common = sorted(set.intersection(*users_sets))
    return common

# ========== 工具函数：计算组内统一 y 轴范围 ==========
def get_group_ylim(metric_key, is_prob=False, is_percent=False):
    """返回四类指标各自统一的 y 轴范围（基于三组共同用户）。"""
    common_users = get_common_users()
    vals = []
    for emo in ["P", "N", "All"]:
        for row in data[emo]:
            if row["user"] in common_users:
                v = row[metric_key]
                if is_prob:  # 概率转百分比
                    v *= 100.0
                vals.append(v)
    vmin = 0.0
    vmax = max(vals) if vals else 1.0
    if is_prob:
        return (0, 100)
    elif is_percent:
        # 你之前给 PANAS 设过较紧的范围，这里保留；如需自动伸缩可改成 (0, vmax*1.1)
        return (0, 20)
    return (0, vmax * 1.1 if vmax > 0 else 1.0)

# ========== 画图函数：带 μ 虚线 + σ 全覆盖阴影 ==========
def plot_metric_group(metric_key, ylabel, filename_prefix, is_prob=False, is_percent=False):
    """同一指标（Prob/Dist/Delta/PANAS）画三张图（P/N/All），共用同一个 y 轴范围。"""
    os.makedirs('Plot_Analyse', exist_ok=True)
    ylim = get_group_ylim(metric_key, is_prob=is_prob, is_percent=is_percent)
    common_users = get_common_users()  # 只画共同用户，自动剔除缺失者

    for emo in ["P", "N", "All"]:
        # 仅保留共同用户并按 user 升序
        emo_rows = sorted([r for r in data[emo] if r["user"] in common_users], key=lambda r: r["user"])
        users = [r["user"] for r in emo_rows]
        values = [r[metric_key] * (100.0 if is_prob else 1.0) for r in emo_rows]

        values_arr = np.array(values, dtype=float)
        mean_val = float(np.mean(values_arr)) if len(values_arr) else 0.0
        std_val  = float(np.std(values_arr)) if len(values_arr) else 0.0

        # 用类别型 x 轴：位置为 0..n-1，标签显示真实 user 编号
        x = np.arange(len(users))

        plt.figure(figsize=(11, 6))
        plt.bar(x, values, color="skyblue", edgecolor="black")

        # μ 虚线
        label_suffix = "%" if (is_prob or is_percent) else ""
        plt.axhline(mean_val, color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {mean_val:.2f}{label_suffix}")

        # σ 阴影（全覆盖）：用 axhspan；对于非负量可按需截断到 0
        lower = mean_val - std_val
        upper = mean_val + std_val
        if (is_prob or is_percent) and lower < 0:
            lower = 0  # 百分比不画到负值
        plt.axhspan(lower, upper, color="red", alpha=0.18, label=f"Std = {std_val:.2f}{label_suffix}")

        # 轴标签与刻度：显示具体 user 数字
        plt.xlabel("User")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} (Emotion {emo})")
        plt.ylim(ylim)
        plt.xticks(x, [str(u) for u in users], rotation=0)  # 显示具体 user 编号

        plt.legend(loc="best")
        plt.tight_layout()
        outname = os.path.join('Plot_Analyse', f"{filename_prefix}_{emo}.png")
        plt.savefig(outname, dpi=300)
        plt.close()
        print(f"Saved: {outname}  | mean={mean_val:.3f}{label_suffix}, std={std_val:.3f}{label_suffix}, ylim={ylim}, users={users}")

# ========== 生成 12 张图 ==========
# 1) In-Range Probability（百分比、组内统一 0-100）
plot_metric_group("prob", "In-Range Probability (%)", "in_range_prob", is_prob=True)

# 2) Average Distance（组内统一）
plot_metric_group("dist", "Average Distance", "avg_distance")

# 3) Average Delta（组内统一）
plot_metric_group("delta", "Average Delta", "avg_delta")

# 4) Average Delta PANAS（百分比、组内统一）
plot_metric_group("panas", "Average Delta PANAS (%)", "avg_delta_panas", is_percent=True)
