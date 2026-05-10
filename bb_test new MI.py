import warnings
warnings.filterwarnings("ignore")
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import pandas as pd
from scipy import stats
from tqdm import tqdm
from sb3_contrib import TQC
from stable_baselines3 import SAC

# ===================== [1. 路径与参数配置] =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "Humanoid-v5"
EXPERT_PATH = "model/humanoid-v5-SAC.zip"

# 影子模型路径
SHADOW_PATHS = {
    "50K":   "translearning/model/shadow_humanoid_SAC_model_50K.pth",
    "200K":  "translearning/model/shadow_humanoid_SAC_model_200K.pth",
    "500K":  "translearning/model/shadow_humanoid_SAC_model_500K.pth"
}

EXPERIMENT_NAME = "BB_MI_PGD_SAC_Humanoid 3"
N_EPISODES = 3       # 每组 Epsilon 测试 30 次以获得稳定的置信区间
EPSILON_LIST = [0.00, 0.010, 0.012, 0.015, 0.018, 0.02, 0.023, 0.025, 0.027, 0.03, 0.035, 0.05]
PGD_ITER = 10         # 迭代次数
VIDEO_FOLDER = "./bb_mi_sac_pgd_videos 3"

# ===================== [2. 影子模型架构] =====================
class SubstituteModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SubstituteModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

# ===================== [3. MI-PGD 集成攻击核心类] =====================
class EnsemblePGDAttacker:
    def __init__(self, shadow_models, epsilon=0, n_iter=10, mu=0.9):
        self.models = shadow_models
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.mu = mu  # 动量衰减因子
        self.latencies = [] 

        # 冻结影子模型参数以加速攻击
        for m in self.models:
            for param in m.parameters():
                param.requires_grad = False

    def attack(self, obs):
        if self.epsilon == 0: return obs
        
        start_t = time.time()
        alpha = self.epsilon / 8.0 
        obs_raw = obs.copy()
        
        # 1. 初始随机化 (PGD 特性)
        adv_obs = obs.copy() + np.random.uniform(-self.epsilon, self.epsilon, obs.shape)
        
        # 2. 初始化动量累加器
        momentum_g = torch.zeros_like(torch.as_tensor(obs, dtype=torch.float32, device=DEVICE))
        
        # 3. 预计算影子模型在干净状态下的基准动作
        clean_obs_t = torch.as_tensor(obs_raw, dtype=torch.float32, device=DEVICE)
        clean_actions = []
        with torch.no_grad():
            for m in self.models:
                m.eval()
                clean_actions.append(m(clean_obs_t.unsqueeze(0)))

        # 4. MI-PGD 迭代循环
        for _ in range(self.n_iter):
            obs_t = torch.as_tensor(adv_obs, dtype=torch.float32, device=DEVICE).requires_grad_(True)
            
            joint_loss = 0
            for i, m in enumerate(self.models):
                m.eval()
                action_pred = m(obs_t.unsqueeze(0))
                # 目标：最大化影子模型与原始动作的 MSE 偏差
                joint_loss += F.mse_loss(action_pred, clean_actions[i])
            
            joint_loss.backward()
            
            # --- 动量机制核心步骤 ---
            grad = obs_t.grad.data
            
            # 步骤 A: 梯度归一化 (L1 Norm)
            grad_l1_norm = torch.norm(grad, p=1)
            if grad_l1_norm > 1e-8:
                grad = grad / grad_l1_norm
            
            # 步骤 B: 累加动量
            momentum_g = self.mu * momentum_g + grad
            
            # 步骤 C: 沿着动量符号方向步进
            grad_sign = momentum_g.sign().cpu().numpy()
            adv_obs = adv_obs + alpha * grad_sign
            
            # 5. 投影约束 (L-infinity)
            delta = np.clip(adv_obs - obs_raw, -self.epsilon, self.epsilon)
            adv_obs = obs_raw + delta

        self.latencies.append((time.time() - start_t) * 1000) 
        return adv_obs

def calculate_ci(data):
    """计算 95% 置信区间"""
    if len(data) < 2: return 0
    return stats.sem(data) * stats.t.ppf((1 + 0.95) / 2., len(data) - 1)

# ===================== [4. 实验主循环] =====================
def run_experiment():
    print(f"🚀 启动集成 MI-PGD 扫描 | 环境: {ENV_NAME} | 设备: {DEVICE}")
    
    # 加载黑盒专家模型
    if not os.path.exists(EXPERT_PATH):
        print(f"❌ 错误: 找不到专家模型 {EXPERT_PATH}")
        return
    expert = SAC.load(EXPERT_PATH, device=DEVICE)
    
    env_temp = gym.make(ENV_NAME)
    s_dim = env_temp.observation_space.shape[0]
    a_dim = env_temp.action_space.shape[0]
    env_temp.close()

    # 加载影子模型
    shadow_models = []
    for name, path in SHADOW_PATHS.items():
        if os.path.exists(path):
            m = SubstituteModel(s_dim, a_dim).to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.eval()
            shadow_models.append(m)
            print(f"✅ 已加载影子模型: {name}")
        else:
            print(f"⚠️ 警告: 找不到影子模型 {path}")

    summary_data = []
    clean_baseline_reward = None
    if not os.path.exists(VIDEO_FOLDER): os.makedirs(VIDEO_FOLDER)

    pbar_eps = tqdm(EPSILON_LIST, desc="[Total Progress]", position=0)
    
    for eps in pbar_eps:
        pbar_eps.set_description(f"Testing Epsilon: {eps:.3f}")
        
        raw_env = gym.make(ENV_NAME, render_mode="rgb_array")
        env = RecordVideo(raw_env, video_folder=VIDEO_FOLDER, 
                          name_prefix=f"{EXPERIMENT_NAME}_eps{eps}",
                          episode_trigger=lambda x: x == 0,
                          disable_logger=True)
        
        attacker = EnsemblePGDAttacker(shadow_models, epsilon=eps, n_iter=PGD_ITER)
        ep_rewards = []
        rpr_list = []

        pbar_episodes = tqdm(range(N_EPISODES), desc=f"  -> Episodes (eps={eps})", position=1, leave=False)
        
        for ep_idx in pbar_episodes:
            obs, _ = env.reset()
            done, total_r = False, 0
            
            while not done:
                # 调用带动量的攻击方法
                adv_obs = attacker.attack(obs)
                
                if eps > 0:
                    rpr = (np.linalg.norm(adv_obs - obs) / (np.linalg.norm(obs) + 1e-8)) * 100
                    rpr_list.append(rpr)
                
                # 专家模型在被污染的观测值下做出决策
                action, _ = expert.predict(adv_obs, deterministic=True)
                obs, r, term, trun, _ = env.step(action)
                total_r += r
                done = term or trun
            
            ep_rewards.append(total_r)
            pbar_episodes.set_postfix({"Last_Rew": f"{total_r:.1f}"})

        env.close()

        # 数据统计
        mean_r = np.mean(ep_rewards)
        ci_h = calculate_ci(ep_rewards)
        if eps == 0.0: clean_baseline_reward = mean_r
        
        # 指标计算
        asr_val = np.mean([1 if r < clean_baseline_reward * 0.5 else 0 for r in ep_rewards]) * 100 if eps > 0 else 0
        avg_rpr = np.mean(rpr_list) if eps > 0 else 0
        avg_lat = np.mean(attacker.latencies) if attacker.latencies else 0
        
        summary_data.append({
            "Epsilon": f"{eps:.3f}",
            "Reward": f"{mean_r:.1f} ± {ci_h:.1f}",
            "RPR(%)": f"{avg_rpr:.4f}%",
            "ASR(%)": f"{asr_val:.1f}%",
            "Latency(ms)": f"{avg_lat:.3f}"
        })

    # 保存报表
    df = pd.DataFrame(summary_data)
    df.to_csv(f"{EXPERIMENT_NAME}_report.csv", index=False)
    print("\n" + "="*85)
    print(df.to_string(index=False))
    print("="*85)
    print(f"📊 报告已保存至: {EXPERIMENT_NAME}_report.csv")

if __name__ == "__main__":
    run_experiment()