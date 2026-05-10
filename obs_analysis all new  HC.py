import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sb3_contrib import TQC
from stable_baselines3 import SAC
from tqdm import tqdm

# ===================== [0. 用户自由配置区] =====================
USER_CONFIG = {
    "env_id": "HalfCheetah-v5",    # 🌟 修改为 HalfCheetah 环境
    "epsilon": 0.03,               
    "fig4_epsilon": 0.05,          
    "target_dim": 5,               
    
    "fig2_selected_dims": list(range(17)), 
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    "atn_model_path": "./atn_halfcheetah_model/atn_halfcheetah_model.pth",
    "target_model_path": "./model/halfcheetah-v5-SAC.zip",
    
    # 🌟 将影子模型严格拆分为两组
    "shadow_paths_pgd": [
        "translearning/model/shadow_hc_SAC_model_200K.pth",
        "translearning/model/shadow_hc_SAC_model_500K.pth",
        "translearning/model/shadow_hc_SAC_model_1000K.pth"
    ],
    "shadow_paths_atn": [
        "translearning/model/shadow_hc_SAC_model_50K.pth",
        "translearning/model/shadow_hc_SAC_model_200K.pth"
    ]
}

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
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), 
            nn.Linear(256, 256), nn.ReLU(), 
            nn.Linear(256, action_dim), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

# ===================== [2. 攻击算法包装器] =====================
def fgsm_attack_wb(obs, model, epsilon, env, device):
    if epsilon == 0: return obs
    policy_net = model.policy.actor
    policy_net.eval()
    
    obs_raw = obs.copy()
    obs_t = torch.as_tensor(obs_raw, dtype=torch.float32, device=device).requires_grad_(True)
    
    with torch.no_grad():
        clean_action = policy_net(torch.as_tensor(obs_raw, dtype=torch.float32, device=device).unsqueeze(0)).detach()
        
    current_action = policy_net(obs_t.unsqueeze(0))
    loss = F.mse_loss(current_action, clean_action)
    
    policy_net.zero_grad()
    if obs_t.grad is not None: obs_t.grad.zero_grad()
    loss.backward()
    
    grad_sign = obs_t.grad.data.sign().cpu().numpy()[0]
    adv_obs = obs_raw + epsilon * grad_sign
    adv_obs = np.clip(adv_obs, env.observation_space.low, env.observation_space.high)
    return adv_obs

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

# ===================== [3. 核心时机数据采集逻辑] =====================
# 🌟 接收两套独立的影子模型列表
def collect_combined_analysis_data(env_id, model, atn, shadow_models_pgd, shadow_models_atn, device, epsilon):
    env = gym.make(env_id)
    
    print(f"🔍 [1/2] 正在采集微观状态序列 (Epsilon = {epsilon})...")
    obs, _ = env.reset(seed=42)
    base_trajectory = []
    
    for _ in range(250):
        base_trajectory.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc: break
        
    data_micro = {
        'clean': np.array(base_trajectory),
        'fgsm_wb': [],
        'pgd_wb': [],
        'pgd_bb': [],
        'atn_full': [],
        'atn_sparse': [],
        'atn_triggered': [] 
    }
    
    burn_in_mses = []
    mu = 0.0
    
    for ep_steps, raw_obs in enumerate(tqdm(base_trajectory, desc="Micro Calculation")):
        data_micro['fgsm_wb'].append(fgsm_attack_wb(raw_obs, model, epsilon, env, device))
        data_micro['pgd_wb'].append(pgd_attack_wb(raw_obs, model, epsilon, env, device))
        
        # 🌟 PGD 黑盒攻击使用 shadow_models_pgd
        data_micro['pgd_bb'].append(pgd_attack_bb(raw_obs, shadow_models_pgd, epsilon, device))
        
        obs_t = torch.FloatTensor(raw_obs).unsqueeze(0).to(device)
        with torch.no_grad():
            adv_obs_t = atn(obs_t, epsilon)
            adv_obs_np = adv_obs_t.cpu().numpy()[0]
            
            shadow_mses = []
            # 🌟 ATN 触发判定使用 shadow_models_atn
            for shadow_m in shadow_models_atn:
                s_clean_act = shadow_m(obs_t)
                s_adv_act = shadow_m(adv_obs_t)
                shadow_mses.append(torch.mean(torch.pow(s_clean_act - s_adv_act, 2)).item())
            current_mse = np.mean(shadow_mses) if shadow_models_atn else 0.0

        data_micro['atn_full'].append(adv_obs_np)
        
        executed_obs = raw_obs.copy()
        is_triggered = False
        
        if ep_steps < 100:
            burn_in_mses.append(current_mse)
        else:
            if ep_steps == 100:
                mu = np.mean(burn_in_mses) if burn_in_mses else 0.0
            
            if current_mse > mu and epsilon > 0:
                executed_obs = adv_obs_np
                is_triggered = True
                
        data_micro['atn_sparse'].append(executed_obs)
        data_micro['atn_triggered'].append(is_triggered)

    for k in ['fgsm_wb', 'pgd_wb', 'pgd_bb', 'atn_full', 'atn_sparse', 'atn_triggered']:
        data_micro[k] = np.array(data_micro[k])

    print("\n🔍 [2/2] 正在运行真实对抗环境 Rollout 采集流形数据...")
    def run_macro_traj(attack_type):
        o, _ = env.reset(seed=42)
        traj = []
        burn_in_mses = []
        mu = 0.0
        
        for step in range(400):
            traj.append(o)
            executed_obs = o.copy()
            
            if attack_type == 'pgd_wb':
                executed_obs = pgd_attack_wb(o, model, epsilon, env, device)
            elif attack_type == 'atn_full':
                obs_t = torch.FloatTensor(o).unsqueeze(0).to(device)
                with torch.no_grad():
                    executed_obs = atn(obs_t, epsilon).cpu().numpy()[0]
            elif attack_type == 'atn_sparse':
                obs_t = torch.FloatTensor(o).unsqueeze(0).to(device)
                with torch.no_grad():
                    adv_obs_t = atn(obs_t, epsilon)
                    # 🌟 宏观演化中的 ATN 触发判定同样使用 shadow_models_atn
                    shadow_mses = [torch.mean(torch.pow(m(obs_t) - m(adv_obs_t), 2)).item() for m in shadow_models_atn]
                    current_mse = np.mean(shadow_mses) if shadow_models_atn else 0.0
                if step < 100:
                    burn_in_mses.append(current_mse)
                else:
                    if step == 100: mu = np.mean(burn_in_mses) if burn_in_mses else 0.0
                    if current_mse > mu and epsilon > 0:
                        executed_obs = adv_obs_t.cpu().numpy()[0]

            action, _ = model.predict(executed_obs, deterministic=True)
            o, _, term, trunc, _ = env.step(action)
            if term or trunc: break
        return np.array(traj)

    data_macro = {
        'clean': run_macro_traj('clean'),
        'pgd_wb': run_macro_traj('pgd_wb'),
        'atn_full': run_macro_traj('atn_full'),
        'atn_sparse': run_macro_traj('atn_sparse')
    }
    
    env.close()
    return data_micro, data_macro

# ===================== [4. 综合绘图引擎] =====================
def plot_combined_figures(micro, macro, cfg, atn_model, target_model, device):
    epsilon = cfg['epsilon']
    env_id = cfg['env_id']
    target_dim = cfg['target_dim']
    save_dir = "./obs_analysis_combined_figs SAC"
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    print(f"\n📈 [1/4] 绘制图 1-1: 维度 [{target_dim}] 时序波形 (Clean vs PGD vs Full ATN)...")
    plt.figure(figsize=(14, 6))
    steps = min(150, len(micro['clean'])) 
    x_axis = np.arange(steps)
    
    plt.plot(x_axis, micro['clean'][:steps, target_dim], label='Clean State (Ground Truth)', color='black', linewidth=2.5, zorder=3)
    plt.plot(x_axis, micro['pgd_wb'][:steps, target_dim], label='PGD-WB', color='#D55E00', alpha=0.6, zorder=2)
    plt.plot(x_axis, micro['atn_full'][:steps, target_dim], label='ATN Full (Ours)', color='#009E73', linewidth=2.5, linestyle='--', zorder=4)
    
    plt.axvline(x=100, color='gray', linestyle='-.', alpha=0.8, label='Burn-in Ends (Step 100)')
    plt.title(f'Temporal Perturbation (Full ATN) on Dim [{target_dim}] (Eps={epsilon})', fontweight='bold', fontsize=18)
    plt.xlabel('Time Steps', fontsize=16)
    plt.ylabel('Sensor Value', fontsize=16)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig1-1_Temporal_Full.png', dpi=300)
    plt.close()

    print(f"📈 [1/4] 绘制图 1-2: 维度 [{target_dim}] 时序波形 (Clean vs PGD vs Sparse ATN)...")
    plt.figure(figsize=(14, 6))
    
    plt.plot(x_axis, micro['clean'][:steps, target_dim], label='Clean State (Ground Truth)', color='black', linewidth=2.5, zorder=3)
    plt.plot(x_axis, micro['pgd_wb'][:steps, target_dim], label='PGD-WB', color='#D55E00', alpha=0.6, zorder=2)
    plt.plot(x_axis, micro['atn_sparse'][:steps, target_dim], label='Sparse ATN (Ours)', color='#0072B2', linewidth=2.5, linestyle='--', zorder=4)
    
    plt.axvline(x=100, color='gray', linestyle='-.', alpha=0.8, label='Burn-in Ends (Step 100)')
    plt.title(f'Temporal Perturbation (Sparse ATN) on Dim [{target_dim}] (Eps={epsilon})', fontweight='bold', fontsize=18)
    plt.xlabel('Time Steps', fontsize=16)
    plt.ylabel('Sensor Value', fontsize=16)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig1-2_Temporal_Sparse.png', dpi=300)
    plt.close()

    print("📈 [2/4] 绘制图 2: 五种攻击类型的扰动空间分布...")
    plt.figure(figsize=(18, 6)) 
    
    burn_in_steps = 100
    if len(micro['clean']) > burn_in_steps:
        post_clean = micro['clean'][burn_in_steps:]
        delta_fgsm_wb = np.mean(np.abs(micro['fgsm_wb'][burn_in_steps:] - post_clean), axis=0)
        delta_pgd_wb = np.mean(np.abs(micro['pgd_wb'][burn_in_steps:] - post_clean), axis=0)
        delta_pgd_bb = np.mean(np.abs(micro['pgd_bb'][burn_in_steps:] - post_clean), axis=0)
        delta_atn_full = np.mean(np.abs(micro['atn_full'][burn_in_steps:] - post_clean), axis=0)
        delta_atn_sparse = np.mean(np.abs(micro['atn_sparse'][burn_in_steps:] - post_clean), axis=0)
    else:
        delta_fgsm_wb = np.mean(np.abs(micro['fgsm_wb'] - micro['clean']), axis=0)
        delta_pgd_wb = np.mean(np.abs(micro['pgd_wb'] - micro['clean']), axis=0)
        delta_pgd_bb = np.mean(np.abs(micro['pgd_bb'] - micro['clean']), axis=0)
        delta_atn_full = np.mean(np.abs(micro['atn_full'] - micro['clean']), axis=0)
        delta_atn_sparse = np.mean(np.abs(micro['atn_sparse'] - micro['clean']), axis=0)
    
    fig2_dims = cfg.get('fig2_selected_dims', list(range(30)))
    total_env_dims = len(delta_pgd_wb)
    valid_dims = [d for d in fig2_dims if d < total_env_dims]
    
    x_indices = np.arange(len(valid_dims))
    width = 0.15 
    
    plt.bar(x_indices - 2*width, delta_fgsm_wb[valid_dims], width=width, color='#E69F00', alpha=0.8, label='FGSM-WB')
    plt.bar(x_indices - width, delta_pgd_wb[valid_dims], width=width, color='#D55E00', alpha=0.7, label='PGD-WB')
    plt.bar(x_indices, delta_pgd_bb[valid_dims], width=width, color='#CC79A7', alpha=0.7, label='PGD-BB')
    plt.bar(x_indices + width, delta_atn_full[valid_dims], width=width, color='#009E73', alpha=0.8, label='ATN Full (Ours)')
    plt.bar(x_indices + 2*width, delta_atn_sparse[valid_dims], width=width, color='#0072B2', alpha=0.8, label='Sparse ATN (Ours)')
    
    plt.xticks(x_indices, valid_dims, fontsize=12)
    plt.title(f'Perturbation Sparsity Across {len(valid_dims)} Selected Dimensions', fontweight='bold')
    plt.xlabel('Observation Dimension Index', fontsize=16)
    plt.ylabel('Mean Absolute Perturbation', fontsize=16)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig2_Combined_Sparsity.png', dpi=300)
    plt.close()

# ---------------- 图 3：状态空间流形偏移 (三子图) ----------------
    print("📈 [3/4] 绘制图 3: 状态空间流形偏移 t-SNE (三子图版，支持独立裁剪)...")
    traj_clean = macro['clean']
    traj_pgd = macro['pgd_wb']
    traj_atn_full = macro['atn_full']
    traj_atn_sparse = macro['atn_sparse'][100:]  
    
    len_c, len_p, len_af = len(traj_clean), len(traj_pgd), len(traj_atn_full)
    all_states = np.vstack([traj_clean, traj_pgd, traj_atn_full, traj_atn_sparse])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(all_states)
    
    emb_clean = embedded[:len_c]
    emb_pgd = embedded[len_c : len_c+len_p]
    emb_atn_full = embedded[len_c+len_p : len_c+len_p+len_af]
    emb_atn_sparse = embedded[len_c+len_p+len_af:]
    
    # 🌟 1. 计算全局坐标轴范围，稍微留出 5% 的边缘空白
    x_min, x_max = np.min(embedded[:, 0]), np.max(embedded[:, 0])
    y_min, y_max = np.min(embedded[:, 1]), np.max(embedded[:, 1])
    x_margin, y_margin = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
    
    # 🌟 2. 移除 sharex=True, sharey=True，让每张图都有完整的独立边框和刻度
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # 🌟 3. 为所有子图统一施加绝对一致的坐标范围，并加上横纵坐标轴标签
    for ax in axes:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlabel('t-SNE Component 1', fontsize=14)
        ax.set_ylabel('t-SNE Component 2', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    axes[0].scatter(emb_clean[:, 0], emb_clean[:, 1], c='blue', label='Clean', alpha=0.3, s=20)
    axes[0].scatter(emb_pgd[:, 0], emb_pgd[:, 1], c='red', label='PGD-WB Deviation', marker='x', alpha=0.6, s=30)
    axes[0].set_title('Clean vs. PGD-WB', fontweight='bold', fontsize=16)
    axes[0].legend(fontsize=12)
    
    axes[1].scatter(emb_clean[:, 0], emb_clean[:, 1], c='blue', label='Clean', alpha=0.3, s=20)
    axes[1].scatter(emb_atn_full[:, 0], emb_atn_full[:, 1], c='green', label='Full ATN Deviation', marker='^', alpha=0.9, s=40)
    axes[1].set_title('Clean vs. Full ATN', fontweight='bold', fontsize=16)
    axes[1].legend(fontsize=12)
    
    axes[2].scatter(emb_clean[:, 0], emb_clean[:, 1], c='blue', label='Clean', alpha=0.3, s=20)
    axes[2].scatter(emb_atn_sparse[:, 0], emb_atn_sparse[:, 1], c='#BB5533', label='Sparse ATN Deviation', marker='v', alpha=0.9, s=40)
    axes[2].set_title('Clean vs. Sparse ATN', fontweight='bold', fontsize=16)
    axes[2].legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig3_tSNE_Combined.png', dpi=300)
    plt.close()

    print("📈 [4/4] 绘制图 4: 特征矩阵视觉隐蔽性条形码 (基于 FGSM-WB 最坏情况)...")
    
    fig4_eps = cfg.get('fig4_epsilon', epsilon)
    target_idx = int(len(micro['clean']) * 0.1) 
    obs = micro['clean'][target_idx]
    
    temp_env_fig4 = gym.make(env_id)
    adv_obs = fgsm_attack_wb(obs, target_model, fig4_eps, temp_env_fig4, device)
    temp_env_fig4.close()
    
    delta = adv_obs - obs
    delta_amplified = delta * 50 
    
    import math
    total_dim = obs.shape[0]
    cols = int(math.ceil(math.sqrt(total_dim)))
    rows = int(math.ceil(total_dim / cols))
    pad_size = (rows * cols) - total_dim
    
    def pad_and_reshape(arr):
        padded = np.pad(arr, (0, pad_size), mode='constant', constant_values=0)
        return padded.reshape((rows, cols))

    obs_matrix = pad_and_reshape(obs)
    delta_matrix = pad_and_reshape(delta_amplified)
    adv_matrix = pad_and_reshape(adv_obs)
    vmin, vmax = np.min(obs_matrix), np.max(obs_matrix)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = 'viridis' 
    
    im0 = axes[0].imshow(obs_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title('Clean State Observation', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    im1 = axes[1].imshow(delta_matrix, cmap='bwr', aspect='auto') 
    axes[1].set_title(f'FGSM-WB Perturbation (Amplified x50)\nStep: {target_idx}', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    im2 = axes[2].imshow(adv_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[2].set_title(f'Adversarial State (Epsilon={fig4_eps})', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im0, cax=cbar_ax, label='Sensor Value Magnitude')
    
    plt.suptitle(f"Worst-Case Visual Imperceptibility via FGSM ({env_id})", fontsize=18, fontweight='bold', y=1.05)
    plt.savefig(f'{save_dir}/Fig4_Combined_Imperceptibility.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n🎉 完美！所有图表已输出至目录: {save_dir}/")

# ===================== [5. 启动入口] =====================
if __name__ == "__main__":
    cfg = USER_CONFIG
    print(f"🚀 启动观测空间综合对比分析系统... 设备: {cfg['device']}")
    
    try:
        target_model = SAC.load(cfg['target_model_path'], device=cfg['device'])
    except:
        print(f"⚠️ 无法使用SAC加载目标模型，尝试使用TQC...")
        target_model = TQC.load(cfg['target_model_path'], device=cfg['device'])
    
    temp_env = gym.make(cfg['env_id'])
    obs_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()
    
    atn = ATNGenerator(obs_dim).to(cfg['device'])
    if os.path.exists(cfg['atn_model_path']):
        atn.load_state_dict(torch.load(cfg['atn_model_path'], map_location=cfg['device']))
        print("✅ 成功加载 ATN 模型。")
    atn.eval()
    
    # 🌟 独立加载 PGD 黑盒影子模型
    shadow_models_pgd = []
    for path in cfg['shadow_paths_pgd']:
        if os.path.exists(path):
            m = SubstituteModel(obs_dim, action_dim).to(cfg['device'])
            m.load_state_dict(torch.load(path, map_location=cfg['device']))
            m.eval()
            shadow_models_pgd.append(m)
    print(f"✅ 成功加载 {len(shadow_models_pgd)} 个 PGD 黑盒影子模型。")

    # 🌟 独立加载 ATN 稀疏触发影子模型
    shadow_models_atn = []
    for path in cfg['shadow_paths_atn']:
        if os.path.exists(path):
            m = SubstituteModel(obs_dim, action_dim).to(cfg['device'])
            m.load_state_dict(torch.load(path, map_location=cfg['device']))
            m.eval()
            shadow_models_atn.append(m)
    print(f"✅ 成功加载 {len(shadow_models_atn)} 个 ATN 稀疏触发影子模型。")
    
    micro_data, macro_data = collect_combined_analysis_data(
        cfg['env_id'], target_model, atn, shadow_models_pgd, shadow_models_atn, cfg['device'], cfg['epsilon']
    )
    
    plot_combined_figures(micro_data, macro_data, cfg, atn, target_model, cfg['device'])