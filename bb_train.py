import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from sb3_contrib import TQC
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ===================== [1. 配置锁定] =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径配置
EXPERT_MODEL_PATH = "model/halfcheetah-v5-TQC.zip" 
DATA_SAVE_PATH = "translearning/data/hc_TQC_data_1000K.npz"
SHADOW_MODEL_PATH = "translearning/model/shadow_hc_TQC_model_1000K.pth"

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

# ===================== [3. 核心功能函数] =====================

def collect_and_save_data(env, expert_model, n_samples=1000000):
    print(f"[*] 正在采集数据 ({n_samples} 步)...")
    
    # 预分配内存（使用 float32 节省一半空间，且与 PyTorch 默认匹配）
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    states = np.zeros((n_samples, s_dim), dtype=np.float32)
    actions = np.zeros((n_samples, a_dim), dtype=np.float32)
    
    obs, _ = env.reset()
    for i in tqdm(range(n_samples), desc="数据采集进度"):
        states[i] = obs.astype(np.float32)
        action, _ = expert_model.predict(obs, deterministic=True)
        actions[i] = action.astype(np.float32)
        
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc: 
            obs, _ = env.reset()
    
    # 保存数据
    np.savez(DATA_SAVE_PATH, states=states, actions=actions)
    print(f"[+] 数据已成功存至: {DATA_SAVE_PATH}")
    return states, actions

def train_shadow(states, actions, s_dim, a_dim):
    print(f"[*] 训练影子模型 (Device: {DEVICE})...")
    model = SubstituteModel(s_dim, a_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    X = torch.from_numpy(states).float() # 确保是 float32
    Y = torch.from_numpy(actions).float()
    loader = DataLoader(TensorDataset(X, Y), batch_size=512, shuffle=True)
    
    pbar = tqdm(range(100), desc="模型训练进度")
    for epoch in pbar:
        l_sum = 0
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            l_sum += loss.item()
        
        if epoch % 10 == 0:
            pbar.set_postfix({"Loss": f"{l_sum/len(loader):.6f}"})
    
    torch.save(model.state_dict(), SHADOW_MODEL_PATH)
    print(f"[+] 影子模型已存至: {SHADOW_MODEL_PATH}")
    return model

# ===================== [4. 评估模块] =====================
def evaluate_shadow(env, expert, shadow):
    print("\n" + "="*50)
    print("📋 影子模型评估报告")
    print("="*50)
    shadow.eval()
    mse_list, cos_list = [], []
    
    obs, _ = env.reset()
    print("[*] 正在对比专家与影子的动作一致性...")
    for _ in range(1000): # 对比1000步
        with torch.no_grad():
            exp_act, _ = expert.predict(obs, deterministic=True)
            sha_act = shadow(torch.FloatTensor(obs).to(DEVICE).unsqueeze(0)).cpu().numpy().flatten()
            
            # 1. 计算 MSE
            mse_list.append(np.mean((exp_act - sha_act)**2))
            
            # 2. 计算方向相似度 (Cosine Similarity)
            sim = np.dot(exp_act, sha_act) / (np.linalg.norm(exp_act) * np.linalg.norm(sha_act) + 1e-8)
            cos_list.append(sim)
            
            obs, _, term, trunc, _ = env.step(exp_act)
            if term or trunc: obs, _ = env.reset()

    print(f"1. 动作偏差 (MSE): {np.mean(mse_list):.6f}")
    print(f"2. 方向相似度 (Cosine Sim): {np.mean(cos_list):.4f}")
    
    # 指标 3：影子模型自己的行走能力
    print("[*] 正在测试影子模型的独立行走能力...")
    total_r = 0; obs, _ = env.reset(); d = False; steps = 0
    while not d and steps < 1000:
        with torch.no_grad():
            a = shadow(torch.FloatTensor(obs).to(DEVICE).unsqueeze(0)).cpu().numpy().flatten()
        obs, r, term, trunc, _ = env.step(a)
        total_r += r; d = term or trunc; steps += 1
        
    print(f"3. 影子模型独立运行奖励: {total_r:.2f}")
    print("="*50 + "\n")

# ===================== [5. 执行入口] =====================
if __name__ == "__main__":
    # 核心修正：环境必须与模型匹配，均为 Humanoid
    env = gym.make("HalfCheetah-v5")
    
    if not os.path.exists(EXPERT_MODEL_PATH):
        print(f"❌ 错误：找不到专家模型文件 {EXPERT_MODEL_PATH}")
    else:
        expert = TQC.load(EXPERT_MODEL_PATH, device=DEVICE)
        print(f"✅ 成功加载 HalfCheetah-TQC 专家模型")

        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]
        print(f"📊 环境维度: Observation {s_dim} | Action {a_dim}")

        # 1. 采集数据
        s_data, a_data = collect_and_save_data(env, expert, n_samples=1000000)
        
        # 2. 训练影子模型
        shadow = train_shadow(s_data, a_data, s_dim, a_dim)
        
        # 3. 运行评估
        evaluate_shadow(env, expert, shadow)
        
        env.close()