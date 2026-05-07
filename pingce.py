import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from sb3_contrib import TQC
from tqdm import tqdm

# ===================== [1. 配置] =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERT_PATH = "model/halfcheetah-v5-SAC.zip"
SHADOW_PATH = "translearning/model/shadow_hc_SAC_model_200K.pth"

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

# ===================== [3. 核心评估函数] =====================
def run_multiple_diagnostics(env, expert, shadow, n_episodes=30):
    print("\n" + "📊" * 15)
    print(f" 开始影子模型动作一致性评估 (共 {n_episodes} 轮)")
    print("📊" * 15)

    shadow.eval() # 纯评估模式
    all_episode_rewards = []
    all_episode_act_cos = []  
    all_episode_mse = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_act_cosines, ep_mses = [], []
        
        # 这一版速度极快，tqdm 可以实时显示
        pbar = tqdm(desc=f"Episode {ep+1}/{n_episodes}", unit="step", leave=False)
        
        while not done:
            # --- 1. 获取动作 ---
            state_t = torch.FloatTensor(obs.astype(np.float32)).to(DEVICE).unsqueeze(0)
            
            with torch.no_grad():
                # 影子模型预测
                sha_act_t = shadow(state_t)
                sha_act = sha_act_t.cpu().numpy().flatten()
                
                # 专家模型预测
                exp_act, _ = expert.predict(obs, deterministic=True)
            
            # --- 2. 计算动作向量指标 ---
            # A. 动作方向相似度 (Cosine Similarity)
            act_norm = (np.linalg.norm(exp_act) * np.linalg.norm(sha_act)) + 1e-8
            act_sim = np.dot(exp_act, sha_act) / act_norm
            ep_act_cosines.append(act_sim)
            
            # B. 动作偏差 (MSE)
            ep_mses.append(np.mean((exp_act - sha_act)**2))

            # --- 3. 环境推进 (由影子模型独立控制，测试其生存能力) ---
            obs, r, term, trunc, _ = env.step(sha_act)
            ep_reward += r
            done = term or trunc
            pbar.update(1)
            
        pbar.close()
        
        # 记录每轮结果
        all_episode_rewards.append(ep_reward)
        all_episode_act_cos.append(np.mean(ep_act_cosines))
        all_episode_mse.append(np.mean(ep_mses))
        
        print(f" ✅ Ep {ep+1:02d}: Reward={ep_reward:>8.1f} | Act_Cos={np.mean(ep_act_cosines):.4f} | MSE={np.mean(ep_mses):.6f}")

    # --- 4. 汇总展示 ---
    print("\n" + "═"*75)
    print(f"📈 影子模型性能汇总报告 ({os.path.basename(SHADOW_PATH)})")
    print("═"*75)
    print(f" 1. 平均总奖励 (Mean Reward):      {np.mean(all_episode_rewards):>10.2f} ± {np.std(all_episode_rewards):.2f}")
    print(f" 2. 平均动作偏差 (Mean MSE):       {np.mean(all_episode_mse):>10.8f}")
    print(f" 3. 动作方向相似度 (Action Cos):    {np.mean(all_episode_act_cos):>10.4f}")
    print(f"    (注：1.0 为完全一致，-1.0 为完全相反)")
    print("═"*75 + "\n")

# ===================== [4. 执行入口] =====================
if __name__ == "__main__":
    # 环境初始化
    env = gym.make("HalfCheetah-v5")
    
    # 模型加载
    if not os.path.exists(EXPERT_PATH):
        print(f"❌ 找不到专家模型: {EXPERT_PATH}"); exit()
        
    expert = SAC.load(EXPERT_PATH, device=DEVICE)
    
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    shadow = SubstituteModel(s_dim, a_dim).to(DEVICE)
    
    if os.path.exists(SHADOW_PATH):
        shadow.load_state_dict(torch.load(SHADOW_PATH, map_location=DEVICE))
        print(f"📂 已成功加载影子权重: {SHADOW_PATH}")
    else:
        print(f"❌ 找不到影子权重: {SHADOW_PATH}"); exit()

    # --- 这里可以自由调大评估次数 ---
    EVAL_COUNT = 20 
    
    run_multiple_diagnostics(env, expert, shadow, n_episodes=EVAL_COUNT)
    
    env.close()