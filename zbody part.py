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
from sb3_contrib import TQC
from stable_baselines3 import SAC
from tqdm import tqdm

# ===================== [0. User Config] =====================
USER_CONFIG = {
    "env_id": "Humanoid-v5",
    "epsilon": 0.03,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    "atn_model_path": "./atn_humanoid_model/atn_humanoid_model.pth",
    "target_model_path": "./model/humanoid-v5-TQC.zip",
    "shadow_paths": [
        "translearning/model/shadow_humanoid_TQC_model_50K.pth",
        "translearning/model/shadow_humanoid_TQC_model_200K.pth",
        "translearning/model/shadow_humanoid_TQC_model_500K.pth"
    ]
}

# ===================== [0.1 Body Part Mapping (中文版)] =====================
HUMANOID_BODY_PARTS = {
    '躯干与头部\n(86维)': np.concatenate([
        np.arange(0, 8), np.arange(22, 31), np.arange(45, 75),
        np.arange(175, 193), np.arange(253, 256), np.arange(270, 288)
    ]),
    '右腿\n(78维)': np.concatenate([
        np.arange(8, 12), np.arange(31, 35), np.arange(75, 105),
        np.arange(193, 211), np.arange(256, 260), np.arange(288, 306)
    ]),
    '左腿\n(78维)': np.concatenate([
        np.arange(12, 16), np.arange(35, 39), np.arange(105, 135),
        np.arange(211, 229), np.arange(260, 264), np.arange(306, 324)
    ]),
    '右臂\n(53维)': np.concatenate([
        np.arange(16, 19), np.arange(39, 42), np.arange(135, 155),
        np.arange(229, 241), np.arange(264, 267), np.arange(324, 336)
    ]),
    '左臂\n(53维)': np.concatenate([
        np.arange(19, 22), np.arange(42, 45), np.arange(155, 175),
        np.arange(241, 253), np.arange(267, 270), np.arange(336, 348)
    ])
}

# 用于控制台输出表格的标签
CN_LABELS = ["躯体（包含头部）", "右腿", "左腿", "右臂", "左臂"]
CN_DIMS = [86, 78, 78, 53, 53]

# ===================== [1. Model Architecture] =====================
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

# ===================== [2. Attack Functions] =====================
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

# ===================== [3. Data Collection] =====================
def collect_micro_data_only(env_id, model, atn, shadow_models, device, epsilon):
    env = gym.make(env_id)
    print(f"🔍 正在采集状态序列 (Epsilon = {epsilon})...")
    obs, _ = env.reset(seed=42)
    base_trajectory = []
    
    for _ in range(250):
        base_trajectory.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc: break
        
    data_micro = {
        'clean': np.array(base_trajectory),
        'fgsm_wb': [], 'pgd_wb': [], 'pgd_bb': [],
        'atn_full': [], 'atn_sparse': []
    }
    
    for ep_steps, raw_obs in enumerate(tqdm(base_trajectory, desc="计算攻击扰动")):
        data_micro['fgsm_wb'].append(fgsm_attack_wb(raw_obs, model, epsilon, env, device))
        data_micro['pgd_wb'].append(pgd_attack_wb(raw_obs, model, epsilon, env, device))
        data_micro['pgd_bb'].append(pgd_attack_bb(raw_obs, shadow_models, epsilon, device))
        
        obs_t = torch.FloatTensor(raw_obs).unsqueeze(0).to(device)
        with torch.no_grad():
            adv_obs_t = atn(obs_t, epsilon)
            adv_obs_np = adv_obs_t.cpu().numpy()[0]
        data_micro['atn_full'].append(adv_obs_np)
        data_micro['atn_sparse'].append(adv_obs_np)

    for k in data_micro.keys():
        data_micro[k] = np.array(data_micro[k])

    env.close()
    return data_micro

# ===================== [4. Plotting] =====================
def plot_and_print_results(micro, cfg):
    save_dir = "./zbodypart_analysis_figs"
    os.makedirs(save_dir, exist_ok=True)
    
    # 🌟 关键修正：先加载 seaborn，再强行覆盖字体设置！
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 强制优先使用系统自带的微软雅黑
    plt.rcParams['axes.unicode_minus'] = False                       # 确保负号不变成方块

    delta_fgsm_wb = np.mean(np.abs(micro['fgsm_wb'] - micro['clean']), axis=0)
    delta_pgd_wb = np.mean(np.abs(micro['pgd_wb'] - micro['clean']), axis=0)
    delta_pgd_bb = np.mean(np.abs(micro['pgd_bb'] - micro['clean']), axis=0)
    delta_atn_full = np.mean(np.abs(micro['atn_full'] - micro['clean']), axis=0)
    delta_atn_sparse = np.mean(np.abs(micro['atn_sparse'] - micro['clean']), axis=0)

    part_names = list(HUMANOID_BODY_PARTS.keys())
    agg_fgsm = [np.mean(delta_fgsm_wb[dims]) for dims in HUMANOID_BODY_PARTS.values()]
    agg_pgd_wb = [np.mean(delta_pgd_wb[dims]) for dims in HUMANOID_BODY_PARTS.values()]
    agg_pgd_bb = [np.mean(delta_pgd_bb[dims]) for dims in HUMANOID_BODY_PARTS.values()]
    agg_atn_full = [np.mean(delta_atn_full[dims]) for dims in HUMANOID_BODY_PARTS.values()]
    agg_atn_sparse = [np.mean(delta_atn_sparse[dims]) for dims in HUMANOID_BODY_PARTS.values()]

    print("\n" + "="*50)
    print("📊 Markdown 表格数据")
    print("="*50)
    all_aggs = {
        "白盒 FGSM": agg_fgsm,
        "白盒 PGD": agg_pgd_wb,
        "黑盒 PGD": agg_pgd_bb,
        "全局 ATN (本文)": agg_atn_full,
        "稀疏 ATN (本文)": agg_atn_sparse
    }
    for method_name, agg_data in all_aggs.items():
        print(f"\n### 方法：{method_name}")
        print("身体部位|包含维度数|扰动大小")
        for i in range(5):
            print(f"{CN_LABELS[i]}|{CN_DIMS[i]}|{agg_data[i]:.6f}")

    # 绘制宽屏图表防止拥挤
    fig, ax = plt.subplots(figsize=(18, 8))
    x = np.arange(len(part_names))
    width = 0.15

    rects1 = ax.bar(x - 2*width, agg_fgsm, width, label='白盒 FGSM', color='#E69F00')
    rects2 = ax.bar(x - width, agg_pgd_wb, width, label='白盒 PGD', color='#D55E00')
    rects3 = ax.bar(x, agg_pgd_bb, width, label='黑盒 PGD', color='#CC79A7')
    rects4 = ax.bar(x + width, agg_atn_full, width, label='全局 ATN (本文)', color='#009E73')
    rects5 = ax.bar(x + 2*width, agg_atn_sparse, width, label='稀疏 ATN (本文)', color='#0072B2')

    # 🌟 恢复横向水平数值标注
    label_kwargs = {'fmt': '%.3f', 'padding': 4, 'rotation': 0, 'fontsize': 9, 'fontweight': 'bold'}
    ax.bar_label(rects1, **label_kwargs)
    ax.bar_label(rects2, **label_kwargs)
    ax.bar_label(rects3, **label_kwargs)
    ax.bar_label(rects4, **label_kwargs)
    ax.bar_label(rects5, **label_kwargs)

    ax.set_title('各身体部位的平均对抗扰动幅度对比', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Humanoid 身体部位分类', fontsize=16)
    ax.set_ylabel('平均绝对扰动值', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(part_names, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.05)
    ax.legend(fontsize=12, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/body_part_analysis_CN.png', dpi=300)
    plt.close()
    print(f"\n✅ 全中文图表已保存至: {save_dir}/body_part_analysis_CN.png")

# ===================== [Main] =====================
if __name__ == "__main__":
    cfg = USER_CONFIG
    print(f"🚀 Running on {cfg['device']}")
    
    try:
        target_model = SAC.load(cfg['target_model_path'], device=cfg['device'])
    except:
        target_model = TQC.load(cfg['target_model_path'], device=cfg['device'])

    temp_env = gym.make(cfg['env_id'])
    obs_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()

    atn = ATNGenerator(obs_dim).to(cfg['device'])
    if os.path.exists(cfg['atn_model_path']):
        atn.load_state_dict(torch.load(cfg['atn_model_path'], map_location=cfg['device']))
    atn.eval()

    shadow_models = []
    for path in cfg['shadow_paths']:
        if os.path.exists(path):
            m = SubstituteModel(obs_dim, action_dim).to(cfg['device'])
            m.load_state_dict(torch.load(path, map_location=cfg['device']))
            m.eval()
            shadow_models.append(m)

    micro_data = collect_micro_data_only(cfg['env_id'], target_model, atn, shadow_models, cfg['device'], cfg['epsilon'])
    plot_and_print_results(micro_data, cfg)