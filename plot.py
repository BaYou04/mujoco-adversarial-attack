import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df1 = pd.read_csv('data1.csv')
df2 = pd.read_csv('data2.csv')

def parse_reward(reward_str):
    parts = reward_str.split('±')
    mean = float(parts[0].strip())
    std = float(parts[1].strip())
    return mean, std

def parse_asr(asr_str):
    return float(asr_str.replace('%', '').strip())

for df in [df1, df2]:
    rewards = df['Reward'].apply(parse_reward)
    df['Reward_mean'] = [r[0] for r in rewards]
    df['Reward_std'] = [r[1] for r in rewards]
    df['ASR_float'] = df['ASR'].apply(parse_asr)

# Plot 1: Epsilon vs Reward
plt.figure(figsize=(10, 6))
plt.plot(df1['Epsilon'], df1['Reward_mean'], label='SAC (data1)', marker='o')
plt.fill_between(df1['Epsilon'], df1['Reward_mean'] - df1['Reward_std'], df1['Reward_mean'] + df1['Reward_std'], alpha=0.2)

plt.plot(df2['Epsilon'], df2['Reward_mean'], label='TQC (data2)', marker='s')
plt.fill_between(df2['Epsilon'], df2['Reward_mean'] - df2['Reward_std'], df2['Reward_mean'] + df2['Reward_std'], alpha=0.2)

plt.xlabel('Epsilon')
# Make the y-label vertical and upright
plt.ylabel('奖\n励', rotation=0, labelpad=20, va='center')
plt.title('不同扰动下的奖励')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('reward_vs_epsilon.png')
plt.close()

# Plot 2: Epsilon vs ASR
plt.figure(figsize=(10, 6))
plt.plot(df1['Epsilon'], df1['ASR_float'], label='SAC (data1)', marker='o')
plt.plot(df2['Epsilon'], df2['ASR_float'], label='TQC (data2)', marker='s')

plt.xlabel('Epsilon')
# Make the y-label vertical and upright
plt.ylabel('攻\n击\n成\n功\n率\n(%)', rotation=0, labelpad=20, va='center')
plt.title('不同扰动下的攻击成功率')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('asr_vs_epsilon.png')
plt.close()