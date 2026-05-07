import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import numpy as np
import os
import time
import csv
from tqdm import tqdm
from stable_baselines3 import SAC
from sb3_contrib import TQC

# ================= 1. 网络架构定义 =================
# A. 影子模型架构 (用于后台推演)
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x): return self.net(x)

# B. ATN 对抗生成器架构
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, dim)
        )
    def forward(self, x): return x + 0.1 * self.net(x)

class ATNGenerator(nn.Module):
    def __init__(self, obs_dim):
        super(ATNGenerator, self).__init__()
        self.model = nn.Sequential(
            ResBlock(obs_dim), ResBlock(obs_dim), nn.Linear(obs_dim, obs_dim)
        )
    def forward(self, obs, epsilon):
        delta = torch.tanh(self.model(obs))
        return obs + epsilon * delta


# ================= 2. 核心黑盒评估逻辑 =================
# 注意：传入的是 active_shadows（专属于当前算法的同源影子探测器）
def run_evaluation(model_name, target_model, active_shadows, atn, epsilon, episodes=50, device="cpu", video_root=None, baseline_reward=None):
    all_rewards, all_rpr, latencies = [], [], []
    attack_rates = [] 
    success_count = 0
    
    current_video_dir = None
    if video_root and epsilon > 0:
        current_video_dir = os.path.join(video_root, f"{model_name}_eps_{epsilon:.3f}_smart_blackbox")
        os.makedirs(current_video_dir, exist_ok=True)

    base_env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    if current_video_dir:
        env = RecordVideo(base_env, video_folder=current_video_dir, 
                          episode_trigger=lambda x: x == 0, disable_logger=True)
    else:
        env = base_env

    pbar = tqdm(range(episodes), desc=f" {model_name} [Eps:{epsilon:.3f}]", leave=False)
    for ep_idx in pbar:
        obs, _ = env.reset()
        ep_reward, ep_steps = 0, 0
        ep_rpr_list = []
        ep_attack_count = 0 
        
        burn_in_mses = []    
        mu = 0.0             
        
        while True:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            start_t = time.perf_counter()
            
            with torch.no_grad():
                # 1. ATN 生成带有扰动的对抗观测值
                adv_obs_t = atn(obs_t, epsilon)
                adv_obs_np = adv_obs_t.cpu().numpy()
                
                # ================= [算法感知特化推演] =================
                # 2. 仅利用同源影子模型计算动作偏差，计算量减半，信噪比更高！
                shadow_mses = []
                for shadow_m in active_shadows:
                    s_clean_act = shadow_m(obs_t)
                    s_adv_act = shadow_m(adv_obs_t)
                    mse_val = torch.mean(torch.pow(s_clean_act - s_adv_act, 2)).item()
                    shadow_mses.append(mse_val)
                
                # 同源影子的平均偏差 -> 瞬时本能偏差
                current_mse = np.mean(shadow_mses) if len(shadow_mses) > 0 else 0.0
                # =======================================================
            
            latencies.append((time.perf_counter() - start_t) * 1000)
            
            # ================= 稀疏攻击时机判定逻辑 =================
            executed_obs_np = obs_t.cpu().numpy()  # 默认不攻击，发干净状态
            applied_rpr = 0.0               
            
            if ep_steps < 100:
                burn_in_mses.append(current_mse)
            else:
                if ep_steps == 100:
                    mu = np.mean(burn_in_mses) if len(burn_in_mses) > 0 else 0.0
                
                # 触发条件：当前状态偏差突变 > 平稳期均值
                if current_mse > mu and epsilon > 0:
                    executed_obs_np = adv_obs_np   # 拦截！替换为假状态
                    ep_attack_count += 1
                    applied_rpr = (torch.norm(adv_obs_t - obs_t, p=2) / (torch.norm(obs_t, p=2) + 1e-8)).item()
            # ========================================================
            
            ep_rpr_list.append(applied_rpr)
            
            # ================= [黑盒执行] =================
            # 3. 目标黑盒模型仅作为执行器，接收判决后的状态
            executed_action, _ = target_model.predict(executed_obs_np, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(executed_action[0])
            # ============================================
            
            ep_reward += reward
            ep_steps += 1
            
            if terminated or truncated:
                if baseline_reward is not None:
                    if ep_steps < 800 or ep_reward < (baseline_reward * 0.5):
                        success_count += 1
                break
        
        all_rewards.append(ep_reward)
        all_rpr.append(np.mean(ep_rpr_list))
        attack_rates.append((ep_attack_count / ep_steps) * 100)
        pbar.set_postfix({"Rew": f"{ep_reward:.0f}"})
    
    env.close()
    return {
        "rew": np.mean(all_rewards),
        "asr": (success_count / episodes) * 100,
        "rpr": np.mean(all_rpr) * 100,
        "lat": np.mean(latencies),
        "atk_rate": np.mean(attack_rates) 
    }


# ================= 3. 主程序 =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 初始化【算法感知特化黑盒】评估系统... 设备: {device}")
    
    res_dir = "./atn_cheetah_timing_results real"
    video_dir = os.path.join(res_dir, "videos new")
    os.makedirs(video_dir, exist_ok=True)

    # 1. 初始化维度
    temp_env = gym.make("HalfCheetah-v5")
    obs_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()

    # 2. 加载 ATN 攻击者模型
    atn = ATNGenerator(obs_dim).to(device)
    atn_model_path = "./atn_cheetah_model/atn_cheetah_model.pth"
    if os.path.exists(atn_model_path):
        atn.load_state_dict(torch.load(atn_model_path, map_location=device))
    atn.eval()

    # 3. 加载完整影子模型阵列
    base = "translearning/"
    shadow_paths = [
        base + f"model/shadow_hc_SAC_model_{k}.pth" for k in ["50K","200K"]
    ] + [
        base + f"model/shadow_hc_TQC_model_{k}.pth" for k in ["50K","200K"]
    ]
    
    print("📦 正在挂载全系影子模型阵列...")
    shadow_ensemble = []
    for p in shadow_paths:
        if os.path.exists(p):
            m = Actor(obs_dim, action_dim).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            m.eval()
            shadow_ensemble.append(m)
        else:
            print(f"⚠️ 找不到影子模型: {p}")
            
    print(f"✅ 成功加载 {len(shadow_ensemble)} 个影子模型。\n")

    # 4. 正式评估
    target_configs = {"SAC": "model/halfcheetah-v5-SAC.zip"}
    epsilons = [0.010, 0.020,0.025, 0.030,0.035, 0.040, 0.050]
    
    csv_file = os.path.join(res_dir, "smart_timing_metrics SAC.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Epsilon', 'Reward', 'Drop_Rate(%)', 'ASR(%)', 'Avg_RPR(%)', 'Latency(ms)', 'Atk_Rate(%)'])

        for name, path in target_configs.items():
            print(f"🎬 开始狙击目标黑盒: {name}")
            if not os.path.exists(path):
                print(f"⚠️ 找不到目标模型: {path}，跳过...")
                continue
                
            target_model = SAC.load(path, device=device) if name == "SAC" else TQC.load(path, device=device)
            
            # ================= [核心机制：部署同源侦察兵] =================
            if name == "SAC":
                print(" 🎯 检测到目标为 SAC 架构，调遣 [SAC 影子模型] 进行特化侦察...")
                active_shadows = shadow_ensemble[:2]
            else:
                print(" 🎯 检测到目标为 TQC 架构，调遣 [TQC 影子模型] 进行特化侦察...")
                active_shadows = shadow_ensemble[2:]
            # ===============================================================
            
            print(f" 🔍 正在校准基准分数...")
            base_data = run_evaluation(name, target_model, active_shadows, atn, 0.0, episodes=20, device=device)
            baseline_rew = base_data['rew']
            print(f" 📊 黑盒基准奖励: {baseline_rew:.2f}")
            
            for eps in epsilons:
                res = run_evaluation(name, target_model, active_shadows, atn, eps, episodes=50, device=device, 
                                     video_root=video_dir, baseline_reward=baseline_rew)
                
                drop_rate = (1 - res['rew'] / baseline_rew) * 100 if baseline_rew > 0 else 0
                
                writer.writerow([
                    name, f"{eps:.3f}", f"{res['rew']:.2f}", f"{max(0, drop_rate):.2f}", 
                    f"{res['asr']:.2f}", f"{res['rpr']:.4f}", f"{res['lat']:.4f}", f"{res['atk_rate']:.2f}"
                ])
                print(f" ✅ Eps {eps:.3f} | Rew: {res['rew']:.0f} | ASR: {res['asr']:.1f}% | 攻击频率: {res['atk_rate']:.1f}% | 延迟: {res['lat']:.2f}ms")

    print(f"\n✨ 算法感知特化评估执行完毕！结果已存档至: {csv_file}")