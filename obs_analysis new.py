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
    "env_id": "Humanoid-v5",       
    "epsilon": 0.03,              
    "target_dim": 5,               # 指定图 1 中用于绘制时序波形的具体观测维度 (0-based)
    
    # 🌟 自由指定图2中想要展示的观测维度列表
    "fig2_selected_dims": list(range(30)), 
    
    # 🌟 图2的扰动均值计算方式
    # True: 仅计算触发攻击时的瞬时扰动幅度 (不包含静默期的0，数值会更高)
    # False: 计算整条轨迹(包含静默期)的平均扰动幅度 (体现全局时序隐蔽性)
    "fig2_calc_only_triggered": False,
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "atn_model_path": "./atn_humanoid_model/atn_humanoid_model.pth",
    "target_model_path": "./model/humanoid-v5-SAC.zip",
    "shadow_paths": [
        "translearning/model/shadow_humanoid_SAC_model_500K.pth",
        "translearning/model/shadow_humanoid_TQC_model_500K.pth"
    ]
}

# ===================== [1. 网络架构定义] =====================
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x): return self.net(x)

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

# ===================== [3. 核心时机数据采集逻辑] =====================
def collect_timed_analysis_data(env_id, model, atn, shadow_models, device, epsilon):
    env = gym.make(env_id)
    
    # ---------------- 阶段 A：采集固定基准轨迹（用于图1, 图2, 图4）----------------
    print(f"🔍 [1/2] 正在采集基准状态序列与计算时机触发 (Epsilon = {epsilon})...")
    obs, _ = env.reset(seed=42)
    base_trajectory = []
    
    for _ in range(250):
        base_trajectory.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc: break
        
    data_micro = {
        'clean': np.array(base_trajectory),
        'pgd_wb': [],
        'atn_applied': [],
        'atn_triggered': [] 
    }
    
    burn_in_mses = []
    mu = 0.0
    
    for ep_steps, raw_obs in enumerate(tqdm(base_trajectory, desc="Micro Calculation")):
        pgd_adv = pgd_attack_wb(raw_obs, model, epsilon, env, device)
        data_micro['pgd_wb'].append(pgd_adv)
        
        obs_t = torch.FloatTensor(raw_obs).unsqueeze(0).to(device)
        with torch.no_grad():
            adv_obs_t = atn(obs_t, epsilon)
            adv_obs_np = adv_obs_t.cpu().numpy()[0]
            
            shadow_mses = []
            for shadow_m in shadow_models:
                s_clean_act = shadow_m(obs_t)
                s_adv_act = shadow_m(adv_obs_t)
                shadow_mses.append(torch.mean(torch.pow(s_clean_act - s_adv_act, 2)).item())
            current_mse = np.mean(shadow_mses) if shadow_models else 0.0

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
                
        data_micro['atn_applied'].append(executed_obs)
        data_micro['atn_triggered'].append(is_triggered)

    data_micro['pgd_wb'] = np.array(data_micro['pgd_wb'])
    data_micro['atn_applied'] = np.array(data_micro['atn_applied'])
    data_micro['atn_triggered'] = np.array(data_micro['atn_triggered'])

    # ---------------- 阶段 B：采集真实对抗下的演化轨迹（用于图3 t-SNE）----------------
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
            elif attack_type == 'atn_timed':
                obs_t = torch.FloatTensor(o).unsqueeze(0).to(device)
                with torch.no_grad():
                    adv_obs_t = atn(obs_t, epsilon)
                    shadow_mses = [torch.mean(torch.pow(m(obs_t) - m(adv_obs_t), 2)).item() for m in shadow_models]
                    current_mse = np.mean(shadow_mses) if shadow_models else 0.0
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
        'atn_timed': run_macro_traj('atn_timed')
    }
    
    env.close()
    return data_micro, data_macro

# ===================== [4. 综合绘图引擎] =====================
def plot_all_figures(micro, macro, cfg):
    epsilon = cfg['epsilon']
    target_dim = cfg['target_dim']
    fig2_dims = cfg['fig2_selected_dims']
    env_id = cfg['env_id']
    
    save_dir = "./obs_analysis_timed_figs1"
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # --- 图 1：时序波形特征谱 ---
    print(f"\n📈 [1/4] 绘制图 1: 维度 [{target_dim}] 时序波形 (0-149步)...")
    plt.figure(figsize=(14, 6))
    
    start_step = 0
    end_step = min(150, len(micro['clean'])) 
    x_axis = np.arange(start_step, end_step)
    
    plt.plot(x_axis, micro['clean'][start_step:end_step, target_dim], label='Clean (Ground Truth)', color='black', linewidth=2.5, zorder=3)
    plt.plot(x_axis, micro['pgd_wb'][start_step:end_step, target_dim], label='PGD (Continuous White-box)', color='red', alpha=0.5, zorder=2)
    plt.plot(x_axis, micro['atn_applied'][start_step:end_step, target_dim], label='ATN (Timed Black-box, Ours)', color='green', linewidth=2.5, linestyle='--', zorder=4)
    
    plt.axvline(x=100, color='blue', linestyle='-.', alpha=0.6, label='Burn-in Ends (Step 100)')
    
    plt.title(f'Temporal Perturbation & Timed Triggers on {env_id} Obs Dim [{target_dim}] (Eps={epsilon})', fontweight='bold', fontsize=18)
    plt.xlabel('Time Steps', fontsize=18)
    plt.ylabel(f'Sensor Value (Dim {target_dim})', fontsize=18)
    plt.legend(loc='lower right')
    plt.xlim(start_step, end_step)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig1_Temporal_Timed.png', dpi=300)
    plt.close()

    # --- 图 2：扰动稀疏性热力图 (支持自定义离散维度与均值模式切换) ---
    print("📈 [2/4] 绘制图 2: 自定义扰动维度的空间分布...")
    plt.figure(figsize=(14, 5))
    
    # ================= 🌟 采用后半程公平对比逻辑 🌟 =================
    burn_in_steps = 100
    
    # 确保轨迹长度超过了静默期
    if len(micro['clean']) > burn_in_steps:
        # 截取从第100步到最后的所有数据
        post_clean = micro['clean'][burn_in_steps:]
        post_pgd = micro['pgd_wb'][burn_in_steps:]
        post_atn = micro['atn_applied'][burn_in_steps:]
        
        # PGD 在后半程的平均扰动
        delta_pgd_wb_full = np.mean(np.abs(post_pgd - post_clean), axis=0)
        pgd_label = 'PGD-WB Mean |Δ| (Post Burn-in)'
        
        # ATN 根据用户配置决定计算方式 (仅触发时刻 or 后半程平均)
        if cfg['fig2_calc_only_triggered']:
            # 仅提取真正触发了攻击的那些时间步的索引 (必须是在100步之后触发的)
            triggered_idxs = np.where(micro['atn_triggered'][burn_in_steps:])[0] 
            if len(triggered_idxs) > 0:
                active_atn_obs = post_atn[triggered_idxs]
                active_clean_obs = post_clean[triggered_idxs]
                delta_atn_full = np.mean(np.abs(active_atn_obs - active_clean_obs), axis=0)
                atn_label = 'ATN Mean |Δ| (Triggered Instants Only)'
            else:
                delta_atn_full = np.zeros(micro['clean'].shape[1])
                atn_label = 'ATN Mean |Δ| (No Triggers Occurred)'
        else:
            # ATN 在后半程的平均扰动（包含了算法主动潜伏的0，体现了真实稀疏性）
            delta_atn_full = np.mean(np.abs(post_atn - post_clean), axis=0)
            atn_label = 'ATN Mean |Δ| (Post Burn-in, Ours)'
    else:
        # 防错：如果轨迹太短
        delta_pgd_wb_full = np.zeros(micro['clean'].shape[1])
        delta_atn_full = np.zeros(micro['clean'].shape[1])
        pgd_label = 'PGD-WB Mean |Δ|'
        atn_label = 'ATN Mean |Δ|'
    # =================================================================
    
    total_env_dims = len(delta_pgd_wb_full)
    valid_dims = [d for d in fig2_dims if d < total_env_dims]
    if not valid_dims:
        print("⚠️ 警告：您选择的维度不合法或为空，默认展示前10个维度。")
        valid_dims = list(range(min(10, total_env_dims)))
        
    val_pgd = delta_pgd_wb_full[valid_dims]
    val_atn = delta_atn_full[valid_dims]
    
    x_indices = np.arange(len(valid_dims))
    width = 0.35
    
    plt.bar(x_indices - width/2, val_pgd, width=width, color='red', alpha=0.6, label=pgd_label)
    plt.bar(x_indices + width/2, val_atn, width=width, color='green', alpha=0.8, label=atn_label)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.title(f'Perturbation Sparsity (Temporal & Spatial) Across {len(valid_dims)} Selected Dimensions', fontweight='bold')
    plt.xlabel('Selected Observation Dimension Index', fontsize=20)
    plt.ylabel('Mean Absolute Applied Perturbation', fontsize=20)
    plt.yscale('log') 
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig2_Sparsity_Timed.png', dpi=300)
    plt.close()

    # --- 图 3：状态空间流形偏移 ---
    print("📈 [3/4] 绘制图 3: 状态空间流形偏移 t-SNE...")
    traj_clean = macro['clean']
    traj_pgd = macro['pgd_wb']
    traj_atn = macro['atn_timed'][100:]
    
    len_c, len_p = len(traj_clean), len(traj_pgd)
    all_states = np.vstack([traj_clean, traj_pgd, traj_atn])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(all_states)
    
    emb_clean = embedded[:len_c]
    emb_pgd = embedded[len_c : len_c+len_p]
    emb_atn = embedded[len_c+len_p:]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    axes[0].scatter(emb_clean[:, 0], emb_clean[:, 1], c='blue', label='Clean Trajectory', alpha=0.3, s=20)
    axes[0].scatter(emb_pgd[:, 0], emb_pgd[:, 1], c='red', label='PGD Deviation', marker='x', alpha=0.6, s=30)
    axes[0].set_title('A: Clean Manifold vs. Continuous PGD', fontweight='bold')
    axes[0].legend()
    
    axes[1].scatter(emb_clean[:, 0], emb_clean[:, 1], c='blue', label='Clean Trajectory', alpha=0.3, s=20)
    axes[1].scatter(emb_atn[:, 0], emb_atn[:, 1], c='green', label='ATN Deviation (Ours)', marker='^', alpha=0.9, s=40)
    axes[1].set_title('B: Clean Manifold vs. Timed ATN', fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig3_tSNE_Comparative_Timed.png', dpi=300)
    plt.close()

    # --- 图 4：特征矩阵视觉隐蔽性条形码 ---
    print("📈 [4/4] 绘制图 4: 特征矩阵视觉隐蔽性条形码...")
    
    trigger_idxs = np.where(micro['atn_triggered'])[0]
    if len(trigger_idxs) > 0:
        target_idx = trigger_idxs[0]
        print(f"   => 在 Step {target_idx} 捕获到 ATN 攻击触发，用于绘制图4。")
    else:
        target_idx = -1
        print(f"   => ⚠️ 当前轨迹未触发攻击，使用最后一帧绘制。")
        
    obs = micro['clean'][target_idx]
    adv_obs = micro['atn_applied'][target_idx]
    delta = adv_obs - obs
    delta_amplified = delta * 50 
    
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
    axes[1].set_title(f'ATN Perturbation (Amplified x50)\nStep: {target_idx}', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    im2 = axes[2].imshow(adv_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[2].set_title(f'Adversarial State (Epsilon={epsilon})', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im0, cax=cbar_ax, label='Sensor Value Magnitude')
    
    plt.suptitle(f"Visual Imperceptibility of Timed ATN Attack ({env_id})", fontsize=18, fontweight='bold', y=1.05)
    plt.savefig(f'{save_dir}/Fig4_Imperceptibility_Barcode_Timed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 完美！充分考虑攻击时机的 4 张可视化图表已全部生成至: {save_dir}/")

# ===================== [5. 启动入口] =====================
if __name__ == "__main__":
    cfg = USER_CONFIG
    print(f"🚀 启动【考虑攻击时机】的观测空间分析系统... 设备: {cfg['device']}")
    print(f"⚙️ 当前配置 -> 环境: {cfg['env_id']} | Eps: {cfg['epsilon']} | 目标维度: {cfg['target_dim']}")
    
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
    
    shadow_models = []
    for path in cfg['shadow_paths']:
        if os.path.exists(path):
            m = Actor(obs_dim, action_dim).to(cfg['device'])
            m.load_state_dict(torch.load(path, map_location=cfg['device']))
            m.eval()
            shadow_models.append(m)
    print(f"✅ 成功加载 {len(shadow_models)} 个影子探测器用于判决时机。")
    
    micro_data, macro_data = collect_timed_analysis_data(
        cfg['env_id'], target_model, atn, shadow_models, cfg['device'], cfg['epsilon']
    )
    
    plot_all_figures(micro_data, macro_data, cfg)