import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 从 CSV 读取数据 ----------------------
df = pd.read_csv("PGD_SAC_humanoid-V5_N30_report.csv")

epsilon = df["Epsilon"].values
asr = df["ASR(%)"].values

# 解析 Reward 和置信区间
reward_mean = []
reward_ci = []
for s in df["Reward"]:
    val, ci = s.split(" ± ")
    reward_mean.append(float(val))
    reward_ci.append(float(ci))
reward_mean = np.array(reward_mean)
reward_ci = np.array(reward_ci)

# -----------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax1 = plt.subplots(figsize=(10, 6))

# ========== 左Y轴：Reward ==========
color1 = "#1f77b4"
ax1.set_xlabel("扰动边界 ε", fontsize=14)
ax1.set_ylabel("累积奖励 Reward", color=color1, fontsize=14, rotation=90, labelpad=12)

ax1.plot(epsilon, reward_mean, '-o', color=color1, linewidth=2.5, markersize=7, label="Reward")
ax1.fill_between(epsilon, reward_mean - reward_ci, reward_mean + reward_ci,
                 color=color1, alpha=0.2, label="置信区间")

ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax1.grid(alpha=0.3)

# ========== 右Y轴：ASR ==========
ax2 = ax1.twinx()
color2 = "#ff4c59"
ax2.set_ylabel("攻击成功率 ASR (%)", color=color2, fontsize=14, rotation=90, labelpad=12)
ax2.plot(epsilon, asr, '-s', color=color2, linewidth=2.5, markersize=7, label="ASR")
ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)

# ========== 只改这里：图例 ==========
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.95, 0.8))

plt.title("ASR 与 Reward 随扰动强度 ε 的变化曲线", fontsize=16, pad=15)
plt.tight_layout()
plt.savefig("图4-3_ASR_Reward曲线_干净版.png", dpi=300, bbox_inches="tight")
plt.show()