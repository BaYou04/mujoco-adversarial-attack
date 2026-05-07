import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import pandas as pd
from scipy import stats
from sb3_contrib import TQC 
from tqdm import tqdm  # 引入进度条库

# ===================== [配置区] =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "Humanoid-v5"
MODEL_PATH = "./model/humanoid-v5-TQC.zip" 

EXPERIMENT_NAME = "FGSM_TQC_humanoid-V5"  # 实验标签
N_EPISODES = 30                        # 每个 epsilon 测试 30 次
# 确保第一个必须是 0.0，用于自动获取基准分
EPSILON_LIST = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05]

VIDEO_FOLDER = "./attack_videos_tqc"
# ====================================================

class FGSMAttacker:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.epsilon = 0
        self.latencies = []

    def attack(self, obs):
        if self.epsilon == 0: return obs
        
        start_t = time.time()
        # 针对 TQC 的 Actor 网络进行梯度攻击
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).requires_grad_(True)
        
        # TQC 的 policy.actor 输出动作
        current_action = self.model.policy.actor(obs_t.unsqueeze(0))
        
        with torch.no_grad():
            clean_action = self.model.policy.actor(obs_t.unsqueeze(0)).detach()
        
        loss = F.mse_loss(current_action, clean_action)
        self.model.policy.actor.zero_grad()
        loss.backward()
        
        delta = self.epsilon * obs_t.grad.data.sign().cpu().numpy()
        adv_obs = np.clip(obs + delta, self.env.observation_space.low, self.env.observation_space.high)
        
        self.latencies.append((time.time() - start_t) * 1000)
        return adv_obs

def calculate_ci(data):
    """计算 95% 置信区间"""
    if len(data) < 2: return 0
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + 0.95) / 2., len(data) - 1)
    return h

def run_experiment():
    total_start_time = time.time()  # 记录实验开始总时间
    print(f"📡 启动 TQC-FGSM 扫描 | 模型: {os.path.basename(MODEL_PATH)}")
    
    model = TQC.load(MODEL_PATH, device=DEVICE)
    summary_data = []
    
    # 动态基准初始化
    dynamic_clean_benchmark = None

    if not os.path.exists(VIDEO_FOLDER):
        os.makedirs(VIDEO_FOLDER)

    # 总体进度条
    pbar_eps = tqdm(EPSILON_LIST, desc="总体进度", position=0)

    for eps in pbar_eps:
        video_prefix = f"{EXPERIMENT_NAME}_eps{eps}"
        raw_env = gym.make(ENV_NAME, render_mode="rgb_array")
        env = RecordVideo(
            raw_env, 
            video_folder=VIDEO_FOLDER, 
            name_prefix=video_prefix,
            episode_trigger=lambda x: x == 0
        )
        
        attacker = FGSMAttacker(model, env)
        attacker.epsilon = eps
        
        ep_rewards = []
        rpr_list = []

        # 内部回合进度条
        pbar_batch = tqdm(range(N_EPISODES), desc=f" ε={eps:.3f}", position=1, leave=False)

        for _ in pbar_batch:
            obs, _ = env.reset()
            done = False
            total_r = 0
            
            while not done:
                adv_obs = attacker.attack(obs)
                
                if eps > 0:
                    rpr = (np.linalg.norm(adv_obs - obs) / (np.linalg.norm(obs) + 1e-8)) * 100
                    rpr_list.append(rpr)
                
                action, _ = model.predict(adv_obs, deterministic=True)
                obs, r, term, trun, _ = env.step(action)
                total_r += r
                done = term or trun
            
            ep_rewards.append(total_r)
            pbar_batch.set_postfix({"last_r": f"{total_r:.1f}"})  # 实时显示得分

        env.close()

        # --- 核心统计逻辑 ---
        mean_r = np.mean(ep_rewards)
        ci_h = calculate_ci(ep_rewards)
        
        # 自动捕获基准分
        if eps == 0.0:
            dynamic_clean_benchmark = mean_r
            asr_val = 0.0
        else:
            # 使用实时基准判断是否攻击成功（下降 50%）
            success_threshold = dynamic_clean_benchmark * 0.5
            asr_val = np.mean([1 if r < success_threshold else 0 for r in ep_rewards]) * 100

        avg_rpr = np.mean(rpr_list) if eps > 0 else 0
        avg_lat = np.mean(attacker.latencies) if attacker.latencies else 0
        
        summary_data.append({
            "Epsilon": f"{eps:.3f}",
            "Reward": f"{mean_r:.1f} ± {ci_h:.1f}",
            "RPR(%)": f"{avg_rpr:.3f}%",
            "ASR(%)": f"{asr_val:.1f}%",
            "Latency(ms)": f"{avg_lat:.3f}"
        })

    # 保存 CSV
    df = pd.DataFrame(summary_data)
    csv_name = f"{EXPERIMENT_NAME}_report.csv"
    df.to_csv(csv_name, index=False)
    
    total_duration = time.time() - total_start_time  # 计算总时长
    
    print("\n" + "="*80)
    print(df.to_string(index=False))
    print("="*80)
    print(f"✅ 实验完成！总耗时: {total_duration/60:.2f} 分钟")  # 显示总时间
    print(f"📂 报告已存至: {csv_name}")
    print(f"🎯 基准分: {dynamic_clean_benchmark:.2f}")

if __name__ == "__main__":
    run_experiment()