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
EXPERT_PATH = "model/humanoid-v5-TQC.zip"

# 影子模型路径 (请确保路径正确)
SHADOW_PATHS = {
    "50K":   "translearning/model/shadow_humanoid_SAC_model_50K.pth",
    "200K":  "translearning/model/shadow_humanoid_SAC_model_200K.pth",
    "500K":  "translearning/model/shadow_humanoid_SAC_model_500K.pth"
}

EXPERIMENT_NAME = "BB_PGD_SAC_Humanoid_N300"
N_EPISODES = 3        # 每组 Epsilon 测试 30 次
EPSILON_LIST = [0.00, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
# EPSILON_LIST = [0.00]
PGD_ITER = 10          # PGD 迭代次数
VIDEO_FOLDER = "./bb_tqc_pgd_videos1"

# ===================== [2. 模型定义] =====================
class SubstituteModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SubstituteModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

# ===================== [3. 集成攻击核心类] =====================
class EnsemblePGDAttacker:
    def __init__(self, shadow_models, epsilon=0, n_iter=10):
        self.models = shadow_models
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.latencies = [] 

        for m in self.models:
            for param in m.parameters():
                param.requires_grad = False

    def attack(self, obs):
        if self.epsilon == 0: return obs
        
        start_t = time.time()
        alpha = self.epsilon / 8.0 
        obs_raw = obs.copy()
        
        # 初始扰动 (保持在 epsilon 球内)
        adv_obs = obs.copy() + np.random.uniform(-self.epsilon, self.epsilon, obs.shape)
        
        # 1. 迭代外：获取各个学徒模型在干净观测下的“基准动作”
        clean_obs_t = torch.as_tensor(obs_raw, dtype=torch.float32, device=DEVICE)
        clean_actions = []

        # 这里使用 no_grad() 是为了节省显存并加速，因为获取基准动作不需要算梯度
        with torch.no_grad():
            for m in self.models:
                m.eval()
                clean_actions.append(m(clean_obs_t.unsqueeze(0)))

        # 2. 进入 PGD 寻找对抗扰动的迭代循环
        for _ in range(self.n_iter):
            # 开启 requires_grad，准备计算输入数据的梯度
            obs_t = torch.as_tensor(adv_obs, dtype=torch.float32, device=DEVICE).requires_grad_(True)
            
            joint_loss = 0
            
            # 3. 遍历模型，计算集成 Loss（一对一锚定计算 MSE）
            for i, m in enumerate(self.models):
                m.eval()
                action_pred = m(obs_t.unsqueeze(0))
                
                # 核心改动：计算当前动作与该模型【自己】基准动作的 MSE 并累加
                joint_loss += torch.nn.functional.mse_loss(action_pred, clean_actions[i])
                
            # 4. 反向传播，求 joint_loss 对输入 obs_t 的梯度
            joint_loss.backward()
            
            # 5. 梯度上升：沿着使得 MSE 增加的方向，以步长 alpha 施加扰动
            grad_sign = obs_t.grad.data.sign().cpu().numpy()
            adv_obs = adv_obs + alpha * grad_sign
            
            # 6. L_inf 投影：将扰动强行拉回到 [-epsilon, epsilon] 的隐蔽范围内
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
    print(f"📡 启动集成 PGD 扫描 | 环境: {ENV_NAME} | 设备: {DEVICE}")
    
    # 加载专家模型
    if not os.path.exists(EXPERT_PATH):
        print(f"❌ 错误: 找不到专家模型 {EXPERT_PATH}")
        return
    expert = SAC.load(EXPERT_PATH, device=DEVICE)
    
    env_temp = gym.make(ENV_NAME)
    s_dim = env_temp.observation_space.shape[0]
    a_dim = env_temp.action_space.shape[0]
    env_temp.close()

    # 加载三个影子模型
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

    # --- 大进度条：遍历 Epsilon 列表 ---
    pbar_eps = tqdm(EPSILON_LIST, desc="[Total Progress]", position=0)
    
    for eps in pbar_eps:
        pbar_eps.set_description(f"Testing Epsilon: {eps:.3f}")
        
        raw_env = gym.make(ENV_NAME, render_mode="rgb_array")
        # 仅录制每个 Epsilon 的第一个回合
        env = RecordVideo(raw_env, video_folder=VIDEO_FOLDER, 
                          name_prefix=f"{EXPERIMENT_NAME}_eps{eps}",
                          episode_trigger=lambda x: x == 0,
                          disable_logger=True)
        
        attacker = EnsemblePGDAttacker(shadow_models, epsilon=eps, n_iter=PGD_ITER)
        ep_rewards = []
        rpr_list = []

        # --- 小进度条：遍历 Episode ---
        # position=1 使其显示在大进度条下方
        pbar_episodes = tqdm(range(N_EPISODES), desc=f"  -> Episodes (eps={eps})", position=1, leave=False)
        
        for ep_idx in pbar_episodes:
            obs, _ = env.reset()
            done, total_r = False, 0
            
            while not done:
                adv_obs = attacker.attack(obs)
                
                if eps > 0:
                    rpr = (np.linalg.norm(adv_obs - obs) / (np.linalg.norm(obs) + 1e-8)) * 100
                    rpr_list.append(rpr)
                
                action, _ = expert.predict(adv_obs, deterministic=True)
                obs, r, term, trun, _ = env.step(action)
                total_r += r
                done = term or trun
            
            ep_rewards.append(total_r)
            # 实时更新小进度条右侧的 Reward
            pbar_episodes.set_postfix({"Last_Rew": f"{total_r:.1f}"})

        env.close()

        # --- 数据统计 ---
        mean_r = np.mean(ep_rewards)
        ci_h = calculate_ci(ep_rewards)
        
        if eps == 0.0: clean_baseline_reward = mean_r
        
        # ASR: 下降超过基准值的 50% 视为攻击成功
        asr_val = np.mean([1 if r < clean_baseline_reward * 0.5 else 0 for r in ep_rewards]) * 100 if eps > 0 else 0
        avg_rpr = np.mean(rpr_list) if eps > 0 else 0
        avg_lat = np.mean(attacker.latencies) if attacker.latencies else 0
        
        summary_data.append({
            "Epsilon": f"{eps:.3f}",
            "Reward": f"{mean_r:.1f} ± {ci_h:.1f}",
            "RPR(%)": f"{avg_rpr:.3f}%",
            "ASR(%)": f"{asr_val:.1f}%",
            "Latency(ms)": f"{avg_lat:.3f}"
        })

    # 保存并打印最终报表
    df = pd.DataFrame(summary_data)
    df.to_csv(f"{EXPERIMENT_NAME}_report.csv", index=False)
    print("\n" + "="*85)
    print(df.to_string(index=False))
    print("="*85)
    print(f"📊 报告已保存至: {EXPERIMENT_NAME}_report.csv")

if __name__ == "__main__":
    run_experiment()