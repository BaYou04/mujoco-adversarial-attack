import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据 (请确保这两个 CSV 文件与脚本在同一目录下)
df_pgd = pd.read_csv('BB_PGD_TQC_Humanoid_N30_report.csv')
df_mpgd = pd.read_csv('BB_MI_PGD_TQC_Humanoid_report copy.csv')

# 2. 数据清洗函数
def clean_reward(val):
    if isinstance(val, str):
        return float(val.split('±')[0].strip())
    return val

def clean_asr(val):
    if isinstance(val, str):
        return float(val.replace('%', '').strip())
    return val

# 分别清洗两份数据的 Reward 和 ASR
df_pgd['Reward_num'] = df_pgd['Reward'].apply(clean_reward)
df_pgd['ASR_num'] = df_pgd['ASR(%)'].apply(clean_asr)

df_mpgd['Reward_num'] = df_mpgd['Reward'].apply(clean_reward)
df_mpgd['ASR_num'] = df_mpgd['ASR(%)'].apply(clean_asr)

# 3. 数据对齐与合并 (Inner Join)
# 这一步确保只保留两个文件中都有测试过的 Epsilon (例如自动过滤掉单方面测试的 0.005, 0.018 等)
df_merge = pd.merge(df_pgd[['Epsilon', 'Reward_num', 'ASR_num']], 
                    df_mpgd[['Epsilon', 'Reward_num', 'ASR_num']], 
                    on='Epsilon', how='inner', suffixes=('_PGD', '_MPGD'))

# 排序并截取 0.05 及以内的数据
df_merge = df_merge.sort_values('Epsilon').reset_index(drop=True)
df_merge = df_merge[df_merge['Epsilon'] <= 0.05]

# 4. 绘图准备
x = np.arange(len(df_merge['Epsilon']))
eps_labels = [f"{eps:.3f}" for eps in df_merge['Epsilon']]

# ================= 图 1：Reward 折线图 =================
plt.figure(figsize=(10, 6))

plt.plot(x, df_merge['Reward_num_PGD'], marker='o', linestyle='-', color='blue', label='PGD Reward', linewidth=2)
plt.plot(x, df_merge['Reward_num_MPGD'], marker='s', linestyle='-', color='orange', label='M-PGD Reward', linewidth=2)

plt.xlabel('Epsilon (Perturbation Threshold)', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.xticks(x, eps_labels)
plt.title('Comparison of PGD and M-PGD: Reward on Humanoid TQC', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存图 1
plt.savefig('zFig4-3_Reward_Line_TQC.png', dpi=300)
plt.close()

# ================= 图 2：ASR 柱状图 =================
plt.figure(figsize=(10, 6))
width = 0.35

plt.bar(x - width/2, df_merge['ASR_num_PGD'], width, label='PGD ASR (%)', color='skyblue', alpha=0.9, edgecolor='black')
plt.bar(x + width/2, df_merge['ASR_num_MPGD'], width, label='M-PGD ASR (%)', color='orange', alpha=0.9, edgecolor='black')

plt.xlabel('Epsilon (Perturbation Threshold)', fontsize=12)
plt.ylabel('Attack Success Rate (ASR) %', fontsize=12)
plt.xticks(x, eps_labels)
plt.title('Comparison of PGD and M-PGD: Attack Success Rate on Humanoid TQC', fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存图 2
plt.savefig('zFig4-3_ASR_Bar_TQC.png', dpi=300)
plt.close()

print("✅ 图表已成功生成：zFig4-3_Reward_Line_TQC.png 和 zFig4-3_ASR_Bar_TQC.png")