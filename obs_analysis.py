import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sb3_contrib import TQC
from stable_baselines3 import SAC
from tqdm import tqdm

# ===================== [1. 网络架构定义] =====================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(nn.Linear(dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, dim))
    def forward(self, x): return x + 0.1 * self.net(x)

class ATNGenerator(nn.Module):
    def __init__(self, obs_dim):
        super(ATNGenerator, self).__init__()
        self.model = nn.Sequential(ResBlock(obs_dim), ResBlock(obs_dim), nn.Linear(obs_dim, obs_dim))
    def forward(self, obs, epsilon):
        delta = torch.tanh(self.model(obs))
        return obs + epsilon * delta

class SubstituteModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SubstituteModel, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim), nn.Tanh())
    def forward(self, x): return self.net(x)

# ===================== [2. 攻击算法包装器] =====================

def pgd_attack_wb(obs, model, epsilon, env, device, n_iter=10):
    if epsilon == 0: return obs
    alpha = epsilon / 8.0
    policy_net = model.policy.actor
    policy_net.eval()
    
    adv_obs = obs.copy() + np.random.uniform(-epsilon, epsilon, obs.shape)
    adv_obs = np.clip(adv_obs, obs - epsilon, obs + epsilon)
    obs_raw = obs.copy()
    obs_raw_tensor = torch.as_tensor(obs_raw, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        clean_action = policy_net(obs_raw_tensor).detach()

    for _ in range(n_iter):
        obs_t = torch.as_tensor(adv_obs, dtype=torch.float32, device=device).requires_grad_(True)
        current_action = policy_net(obs_t.unsqueeze(0))
        loss = F.mse_loss(current_action, clean_action)
        policy_net.zero_grad()
        if obs_t.grad is not None: obs_t.grad.zero_grad()
        loss.backward()
        
        grad_sign = obs_t.grad.data.sign().cpu().numpy()[0]
        adv_obs = adv_obs + alpha * grad_sign
        delta = np.clip(adv_obs - obs_raw, -epsilon, epsilon)
        adv_obs = np.clip(obs_raw + delta, env.observation_space.low, env.observation_space.high)
    return adv_obs

def pgd_attack_bb(obs, shadow_models, epsilon, device, n_iter=10):
    if epsilon == 0 or not shadow_models: return obs
    alpha = epsilon / 8.0
    obs_raw = obs.copy()
    adv_obs = obs.copy() + np.random.uniform(-epsilon, epsilon, obs.shape)
    for _ in range(n_iter):
        obs_t = torch.as_tensor(adv_obs, dtype=torch.float32, device=device).requires_grad_(True)
        joint_loss = 0
        for m in shadow_models:
            m.eval()
            action_pred = m(obs_t.unsqueeze(0))
            joint_loss += torch.norm(action_pred) 
        joint_loss.backward()
        grad_sign = obs_t.grad.data.sign().cpu().numpy()[0]
        adv_obs = adv_obs + alpha * grad_sign
        delta = np.clip(adv_obs - obs_raw, -epsilon, epsilon)
        adv_obs = obs_raw + delta
    return adv_obs

def atn_attack_bb(obs, atn, epsilon, device):
    if epsilon == 0: return obs
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        adv_obs_t = atn(obs_t, epsilon)
    return adv_obs_t.cpu().numpy()[0]

# ===================== [3. 数据采集逻辑] =====================

def collect_analysis_data(env_id, model, atn, shadow_models, device, epsilon=0.03):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=42)
    
    data_micro = {'clean':[], 'pgd_wb':[], 'pgd_bb':[], 'atn':[]}
    print(f"🔍 正在采集微观扰动对比数据 (Epsilon = {epsilon})...")
    for _ in tqdm(range(200), desc="Micro Collection"):
        data_micro['clean'].append(obs)
        data_micro['pgd_wb'].append(pgd_attack_wb(obs, model, epsilon, env, device))
        data_micro['pgd_bb'].append(pgd_attack_bb(obs, shadow_models, epsilon, device))
        data_micro['atn'].append(atn_attack_bb(obs, atn, epsilon, device))
        
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc: break

    print("\n🔍 正在采集宏观状态流形数据 (收集 Clean, PGD 和 ATN)...")
    def run_traj(attack_name):
        o, _ = env.reset(seed=42) 
        traj = []
        for _ in range(400): 
            traj.append(o)
            if attack_name == 'clean': adv_o = o
            elif attack_name == 'pgd_wb': adv_o = pgd_attack_wb(o, model, epsilon, env, device)
            elif attack_name == 'atn': adv_o = atn_attack_bb(o, atn, epsilon, device)
            
            action, _ = model.predict(adv_o, deterministic=True)
            o, _, term, trunc, _ = env.step(action)
            if term or trunc: break
        return np.array(traj)

    # 为了画双子图，把 pgd_wb 加回来
    data_macro = {
        'clean': run_traj('clean'),
        'pgd_wb': run_traj('pgd_wb'),
        'atn': run_traj('atn')
    }
    env.close()
    for k in data_micro: data_micro[k] = np.array(data_micro[k])
    return data_micro, data_macro

# ===================== [4. 综合绘图引擎] =====================

def plot_all_figures(micro, macro, atn, device, epsilon=0.03, target_dim=5):
    save_dir = "./obs_analysis_figs1"
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # --- 图 1：时序波形特征谱 ---
    print("\n📈 [1/4] 绘制图 1: 关键观测维度时序波形...")
    plt.figure(figsize=(12, 6))
    steps = min(150, len(micro['clean'])) 
    x_axis = np.arange(steps)
    
    plt.plot(x_axis, micro['clean'][:steps, target_dim], label='Clean (Ground Truth)', color='black', linewidth=2.5)
    plt.plot(x_axis, micro['pgd_wb'][:steps, target_dim], label='PGD (White-box)', color='red', alpha=0.7)
    plt.plot(x_axis, micro['atn'][:steps, target_dim], label='ATN (Black-box, Ours)', color='green', linewidth=2.5, linestyle='--')
    
    plt.title(f'Temporal Perturbation Signature on Humanoid Obs Dimension [{target_dim}]', fontweight='bold', fontsize=18)
    plt.xlabel('Time Steps', fontsize=16)
    plt.ylabel('Observation Sensor Value', fontsize=16)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig1_Temporal.png', dpi=300)
    plt.close()

    # --- 图 2：扰动稀疏性热力图 ---
    print("📈 [2/4] 绘制图 2: 扰动维度的空间分布...")
    plt.figure(figsize=(14, 5))
    
    delta_pgd_wb = np.mean(np.abs(micro['pgd_wb'] - micro['clean']), axis=0)
    delta_pgd_bb = np.mean(np.abs(micro['pgd_bb'] - micro['clean']), axis=0)
    delta_atn = np.mean(np.abs(micro['atn'] - micro['clean']), axis=0)
    
    dims = np.arange(30) 
    width = 0.3
    
    plt.bar(dims - width, delta_pgd_wb[:30], width=width, color='red', alpha=0.6, label='PGD-WB Mean |Δ|')
    plt.bar(dims, delta_pgd_bb[:30], width=width, color='purple', alpha=0.6, label='PGD-BB Mean |Δ|')
    plt.bar(dims + width, delta_atn[:30], width=width, color='green', alpha=0.8, label='ATN Mean |Δ| (Ours)')
    
    plt.title('Perturbation Sparsity & Targeting Across Observation Dimensions', fontweight='bold')
    plt.xlabel('Observation Dimension Index (0-29)', fontsize=20)
    plt.ylabel('Mean Absolute Perturbation', fontsize=20)
    plt.yscale('log') 
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig2_Sparsity.png', dpi=300)
    plt.close()

    # --- 🌟图 3：状态空间流形偏移 (双子图对比版) ---
    print("📈 [3/4] 绘制图 3: 状态空间流形偏移 t-SNE (双子图对比计算中...)")
    traj_clean = macro['clean']
    traj_pgd = macro['pgd_wb']
    traj_atn = macro['atn']
    
    len_c = len(traj_clean)
    len_p = len(traj_pgd)
    
    # 把三者拼在一起降维，确保左右两图的坐标系和背景(Clean)完全一致
    all_states = np.vstack([traj_clean, traj_pgd, traj_atn])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(all_states)
    
    emb_clean = embedded[:len_c]
    emb_pgd = embedded[len_c : len_c+len_p]
    emb_atn = embedded[len_c+len_p:]
    
    # 创建 1x2 并排画板
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    # 左图 A：PGD 的流形破坏
    axes[0].scatter(emb_clean[:, 0], emb_clean[:, 1], c='blue', label='Clean Trajectory', alpha=0.3, s=20)
    axes[0].scatter(emb_pgd[:, 0], emb_pgd[:, 1], c='red', label='PGD Induced States', marker='x', alpha=0.6, s=30)
    axes[0].set_title('Clean Manifold vs. PGD Deviation', fontweight='bold')
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    axes[0].legend()
    
    # 右图 B：ATN 的流形破坏
    axes[1].scatter(emb_clean[:, 0], emb_clean[:, 1], c='blue', label='Clean Trajectory', alpha=0.3, s=20)
    axes[1].scatter(emb_atn[:, 0], emb_atn[:, 1], c='green', label='ATN Induced States (Ours)', marker='^', alpha=0.9, s=40)
    axes[1].set_title('Clean Manifold vs. ATN Deviation', fontweight='bold')
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig3_tSNE_Comparative.png', dpi=300)
    plt.close()

    # --- 图 4：特征矩阵视觉隐蔽性条形码 ---
    print("📈 [4/4] 绘制图 4: 特征矩阵视觉隐蔽性条形码...")
    obs = micro['clean'][0]
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad(): adv_obs_t = atn(obs_t, epsilon)
    adv_obs = adv_obs_t.cpu().numpy()[0]
    
    delta = adv_obs - obs
    delta_amplified = delta * 50 
    
    matrix_shape = (12, 29)
    obs_matrix = obs.reshape(matrix_shape)
    delta_matrix = delta_amplified.reshape(matrix_shape)
    adv_matrix = adv_obs.reshape(matrix_shape)
    vmin, vmax = np.min(obs_matrix), np.max(obs_matrix)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = 'viridis' 
    
    im0 = axes[0].imshow(obs_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title('Clean State Observation', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    im1 = axes[1].imshow(delta_matrix, cmap='bwr', aspect='auto') 
    axes[1].set_title(f'ATN Perturbation (Amplified x50)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    im2 = axes[2].imshow(adv_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[2].set_title(f'Adversarial State (Epsilon={epsilon})', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im0, cax=cbar_ax, label='Sensor Value Magnitude')
    
    plt.suptitle("Visual Imperceptibility of ATN Attack in State Space", fontsize=18, fontweight='bold', y=1.05)
    plt.savefig(f'{save_dir}/Fig4_Imperceptibility_Barcode.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 完美！带 PGD 并排对比图的 4 张学术图表已全部生成！")

# ===================== [5. 启动入口] =====================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 初始化观测空间分析系统... 设备: {device}")
    
    env_id = "Humanoid-v5"
    model_path = "./model/humanoid-v5-SAC.zip"
    model = SAC.load(model_path, device=device)
    
    temp_env = gym.make(env_id)
    obs_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()
    
    atn = ATNGenerator(obs_dim).to(device)
    atn.load_state_dict(torch.load("./atn_humanoid_model/atn_humanoid_model.pth", map_location=device))
    atn.eval()
    
    shadow_paths = [
        "translearning/model/shadow_humanoid_SAC_model_50K.pth",
        "translearning/model/shadow_humanoid_SAC_model_200K.pth",
        "translearning/model/shadow_humanoid_SAC_model_500K.pth"
    ]
    shadow_models = []
    for path in shadow_paths:
        if os.path.exists(path):
            m = SubstituteModel(obs_dim, action_dim).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            shadow_models.append(m)
    
    epsilon_test = 0.03  
    target_dimension = 5 
    
    micro_data, macro_data = collect_analysis_data(env_id, model, atn, shadow_models, device, epsilon=epsilon_test)
    plot_all_figures(micro_data, macro_data, atn, device, epsilon=epsilon_test, target_dim=target_dimension)