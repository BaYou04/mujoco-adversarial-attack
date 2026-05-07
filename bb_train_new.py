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

# 路径配置 (严格保留原路径)
EXPERT_MODEL_PATH = "model/humanoid-v5-SAC.zip" 
DATA_SAVE_PATH = "ACmodel/data/humanoid_SAC_data_500K.npz"
SHADOW_MODEL_PATH = "ACmodel/model/shadow_humanoid_SAC_model_500K.pth"

# ===================== [2. 模型定义 (新增 Critic)] =====================
class SubstituteModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SubstituteModel, self).__init__()
        
        # 影子 Actor 网络: State -> Action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        
        # 影子 Critic 网络: (State, Action) -> Q-Value
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        # 默认前向传播保留给 Actor
        return self.actor(state)

    def get_q_value(self, state, action):
        # Critic 前向传播：拼接状态与动作
        x = torch.cat([state, action], dim=-1)
        return self.critic(x)

# ===================== [3. 核心功能函数] =====================

def collect_and_save_data(env, expert_model, n_samples=1000000):
    print(f"[*] 正在采集数据 ({n_samples} 步)...")
    
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    states = np.zeros((n_samples, s_dim), dtype=np.float32)
    actions = np.zeros((n_samples, a_dim), dtype=np.float32)
    # 新增：预分配 Q 值内存
    q_targets = np.zeros((n_samples, 1), dtype=np.float32)
    
    obs, _ = env.reset()
    for i in tqdm(range(n_samples), desc="数据采集进度"):
        states[i] = obs.astype(np.float32)
        action, _ = expert_model.predict(obs, deterministic=True)
        actions[i] = action.astype(np.float32)
        
        # 知识蒸馏：提取专家模型 Critic 的 Q 值
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        act_t = torch.FloatTensor(action).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_out = expert_model.critic(obs_t, act_t)
            if isinstance(q_out, torch.Tensor):
                # TQC 格式: [batch_size, n_critics, n_quantiles]
                # 先对最后一个维度(分位数)求均值，再在第1个维度(多个Critic)中取最小值
                q_val = q_out.mean(dim=-1).min(dim=1)[0].item()
            else:
                # SAC 格式: tuple of [batch_size, 1]
                q_val = torch.min(torch.stack(q_out), dim=0)[0].item()
            q_targets[i] = q_val
        
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc: 
            obs, _ = env.reset()
    
    # 保存数据时加入 q_targets
    np.savez(DATA_SAVE_PATH, states=states, actions=actions, q_targets=q_targets)
    print(f"[+] 数据已成功存至: {DATA_SAVE_PATH}")
    return states, actions, q_targets

def train_shadow(states, actions, q_targets, s_dim, a_dim):
    print(f"[*] 训练影子模型 (Device: {DEVICE})...")
    model = SubstituteModel(s_dim, a_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    X = torch.from_numpy(states).float()
    Y = torch.from_numpy(actions).float()
    Q = torch.from_numpy(q_targets).float() # 新增 Q 值张量
    
    # DataLoader 增加一个 Q 值输出
    loader = DataLoader(TensorDataset(X, Y, Q), batch_size=512, shuffle=True)
    
    pbar = tqdm(range(100), desc="模型训练进度")
    for epoch in pbar:
        l_sum = 0
        model.train()
        for bx, by, bq in loader:
            bx, by, bq = bx.to(DEVICE), by.to(DEVICE), bq.to(DEVICE)
            optimizer.zero_grad()
            
            # 1. Actor 损失：拟合动作边界
            out_actions = model.actor(bx)
            actor_loss = criterion(out_actions, by)
            
            # 2. Critic 损失：拟合价值评估 (输入状态和专家动作)
            out_q = model.get_q_value(bx, by)
            critic_loss = criterion(out_q, bq)
            
            # 联合反向传播
            total_loss = actor_loss + critic_loss
            total_loss.backward()
            optimizer.step()
            
            l_sum += total_loss.item()
        
        if epoch % 10 == 0:
            pbar.set_postfix({"Total_Loss": f"{l_sum/len(loader):.6f}"})
    
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
    q_error_list = [] # 新增：评估 Critic 拟合精度
    
    obs, _ = env.reset()
    print("[*] 正在对比专家与影子的特征一致性...")
    for _ in range(1000):
        with torch.no_grad():
            # 动作对比
            exp_act, _ = expert.predict(obs, deterministic=True)
            sha_act = shadow(torch.FloatTensor(obs).to(DEVICE).unsqueeze(0)).cpu().numpy().flatten()
            
            mse_list.append(np.mean((exp_act - sha_act)**2))
            sim = np.dot(exp_act, sha_act) / (np.linalg.norm(exp_act) * np.linalg.norm(sha_act) + 1e-8)
            cos_list.append(sim)
            
            # Q值对比评估
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            act_t = torch.FloatTensor(exp_act).unsqueeze(0).to(DEVICE)
            
            q_out = expert.critic(obs_t, act_t)
            if isinstance(q_out, torch.Tensor):
                # 兼容 TQC 格式
                exp_q = q_out.mean(dim=-1).min(dim=1)[0].item()
            else:
                # 兼容 SAC 格式
                exp_q = torch.min(torch.stack(q_out), dim=0)[0].item()
                
            sha_q = shadow.get_q_value(obs_t, act_t).item()
            q_error_list.append(abs(exp_q - sha_q))
            
            obs, _, term, trunc, _ = env.step(exp_act)
            if term or trunc: obs, _ = env.reset()

    print(f"1. 动作偏差 (MSE): {np.mean(mse_list):.6f}")
    print(f"2. 方向相似度 (Cosine Sim): {np.mean(cos_list):.4f}")
    print(f"3. 价值预测平均绝对误差 (MAE): {np.mean(q_error_list):.4f}")
    
    print("[*] 正在测试影子模型的独立行走能力...")
    total_r = 0; obs, _ = env.reset(); d = False; steps = 0
    while not d and steps < 1000:
        with torch.no_grad():
            a = shadow(torch.FloatTensor(obs).to(DEVICE).unsqueeze(0)).cpu().numpy().flatten()
        obs, r, term, trunc, _ = env.step(a)
        total_r += r; d = term or trunc; steps += 1
        
    print(f"4. 影子模型独立运行奖励: {total_r:.2f}")
    print("="*50 + "\n")

# ===================== [5. 执行入口] =====================
if __name__ == "__main__":
    env = gym.make("Humanoid-v5")
    
    if not os.path.exists(EXPERT_MODEL_PATH):
        print(f"❌ 错误：找不到专家模型文件 {EXPERT_MODEL_PATH}")
    else:
        expert = SAC.load(EXPERT_MODEL_PATH, device=DEVICE)
        print(f"✅ 成功加载 HalfCheetah-SAC 专家模型")

        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]
        print(f"📊 环境维度: Observation {s_dim} | Action {a_dim}")

        # 1. 采集数据 (接收三个返回值)
        s_data, a_data, q_data = collect_and_save_data(env, expert, n_samples=500000)
        
        # 2. 训练影子模型 (传入 Q 值数据)
        shadow = train_shadow(s_data, a_data, q_data, s_dim, a_dim)
        
        # 3. 运行评估
        evaluate_shadow(env, expert, shadow)
        
        env.close()
