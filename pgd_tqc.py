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
from stable_baselines3 import SAC # 如果测TQC，改为 from sb3_contrib import TQC
from sb3_contrib import TQC
from tqdm import tqdm

# ===================== [快速配置区] =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "Humanoid-v5"
MODEL_PATH = "./model/humanoid-v5-TQC.zip" # 确认路径

EXPERIMENT_NAME = "PGD_TQC_humanoid-V5_N101" # 结果文件前缀
N_EPISODES = 3        # 每个 Eps 跑 30 回合（快速平衡速度与统计意义）
EPSILON_LIST = [0.00,  0.01,  0.015,  0.02,  0.03, 0.05] # 略过不敏感的点
    
PGD_ITER = 10          # 10步 PGD，兼顾强度与时间
VIDEO_FOLDER = "./attack_videos_tqc_pgd1"
# ====================================================

class PGDAttacker:
    def __init__(self, model, env, n_iter=5):
        self.model = model
        self.env = env
        self.epsilon = 0
        self.n_iter = n_iter
        self.latencies = []

    def attack(self, obs):
        if self.epsilon == 0: return obs
        
        start_t = time.time()
        alpha = self.epsilon / 8.0
        
        # 1. 确保模型处于评估模式，避免破坏 BN/Dropout 层
        policy_net = self.model.actor if hasattr(self.model, 'actor') else self.model.policy.actor
        policy_net.eval() 

        # 2. 随机初始化 (Madry 原始 PGD 论文建议)
        adv_obs = obs.copy() + np.random.uniform(-self.epsilon, self.epsilon, obs.shape)
        adv_obs = np.clip(adv_obs, obs - self.epsilon, obs + self.epsilon)
        
        obs_raw = obs.copy()
        obs_raw_tensor = torch.as_tensor(obs_raw, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # 预先获取干净动作，减少循环内计算
        with torch.no_grad():
            # 建议使用 deterministic=True 获取确定性动作
            clean_action = policy_net(obs_raw_tensor).detach()

        for _ in range(self.n_iter):
            obs_t = torch.as_tensor(adv_obs, dtype=torch.float32, device=DEVICE).requires_grad_(True)
            
            # 3. 这里的输出必须包含梯度
            current_action = policy_net(obs_t.unsqueeze(0))
            
            # 计算 MSE Loss，让当前动作尽可能偏离原始动作
            loss = F.mse_loss(current_action, clean_action)
            
            # 4. 正确的清零位置
            policy_net.zero_grad()
            if obs_t.grad is not None:
                obs_t.grad.zero_grad()
                
            loss.backward()
            
            # 5. 梯度上升 (Gradient Ascent)
            grad_sign = obs_t.grad.data.sign().cpu().numpy()
            adv_obs = adv_obs + alpha * grad_sign
            
            # 6. 投影到 epsilon 球内并裁剪到环境边界
            delta = np.clip(adv_obs - obs_raw, -self.epsilon, self.epsilon)
            adv_obs = np.clip(obs_raw + delta, self.env.observation_space.low, self.env.observation_space.high)
        
        self.latencies.append((time.time() - start_t) * 1000)
        return adv_obs

def calculate_ci(data):
    if len(data) < 2: return 0
    return stats.sem(data) * stats.t.ppf((1 + 0.95) / 2., len(data) - 1)

def run_experiment():
    print(f"📡 启动快速 PGD 扫描 | 模型: {os.path.basename(MODEL_PATH)}")
    
    # 自动识别算法类型加载
    if "TQC" in MODEL_PATH:
        model = TQC.load(MODEL_PATH, device=DEVICE)
    else:
        model = SAC.load(MODEL_PATH, device=DEVICE)

    summary_data = []
    dynamic_benchmark = None
    if not os.path.exists(VIDEO_FOLDER): os.makedirs(VIDEO_FOLDER)

    # 总体进度条
    pbar_eps = tqdm(EPSILON_LIST, desc="总体进度", position=0)
    
    for eps in pbar_eps:
        video_prefix = f"{EXPERIMENT_NAME}_eps{eps}"
        raw_env = gym.make(ENV_NAME, render_mode="rgb_array")
        
        # 每个 epsilon 录制第一个回合 (episode_id == 0)
        env = RecordVideo(
            raw_env, 
            video_folder=VIDEO_FOLDER, 
            name_prefix=video_prefix,
            episode_trigger=lambda x: x == 0
        )
        
        attacker = PGDAttacker(model, env, n_iter=PGD_ITER)
        attacker.epsilon = eps
        ep_rewards = []
        rpr_list = []

        # 内部回合进度条
        pbar_batch = tqdm(range(N_EPISODES), desc=f" ε={eps:.3f}", position=1, leave=False)
        
        for _ in pbar_batch:
            obs, _ = env.reset()
            done, total_r = False, 0
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
            pbar_batch.set_postfix({"last_r": f"{total_r:.1f}"})

        env.close()

        # 统计分析
        mean_r = np.mean(ep_rewards)
        ci_h = calculate_ci(ep_rewards)
        if eps == 0.0: dynamic_benchmark = mean_r
        
        asr_val = np.mean([1 if r < dynamic_benchmark * 0.5 else 0 for r in ep_rewards]) * 100 if eps > 0 else 0
        avg_rpr = np.mean(rpr_list) if eps > 0 else 0
        avg_lat = np.mean(attacker.latencies) if attacker.latencies else 0
        
        summary_data.append({
            "Epsilon": f"{eps:.3f}",
            "Reward": f"{mean_r:.1f} ± {ci_h:.1f}",
            "RPR(%)": f"{avg_rpr:.3f}%",
            "ASR(%)": f"{asr_val:.1f}%",
            "Latency(ms)": f"{avg_lat:.3f}"
        })

    # 输出结果表格
    df = pd.DataFrame(summary_data)
    csv_filename = f"{EXPERIMENT_NAME}_report.csv"
    df.to_csv(csv_filename, index=False)
    print("\n" + "="*80)
    print(df.to_string(index=False))
    print("="*80)
    print(f"✅ 实验完成！视频已保存至 {VIDEO_FOLDER}，报告见 {csv_filename}")

if __name__ == "__main__":
    run_experiment()