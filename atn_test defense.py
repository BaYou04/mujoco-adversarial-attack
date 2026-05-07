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

# ================= 1. ATN 网络架构 =================
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

# ================= 2. 核心评估逻辑 =================
def run_evaluation(model_name, model, atn, epsilon, defense_type="none", episodes=50, device="cpu", video_root=None, baseline_reward=None):
    all_rewards, all_rpr, latencies = [], [], []
    success_count = 0
    
    # 确定视频存放的具体子路径 (加入防御类型以便区分)
    current_video_dir = None
    if video_root and epsilon > 0:
        current_video_dir = os.path.join(video_root, f"{model_name}_{defense_type}_eps_{epsilon:.3f}")
        os.makedirs(current_video_dir, exist_ok=True)

    # 初始化环境
    base_env = gym.make("Humanoid-v5", render_mode="rgb_array")
    
    # 如果开启视频录制，包装环境 (只录制第 0 回合)
    if current_video_dir:
        env = RecordVideo(base_env, video_folder=current_video_dir, 
                          episode_trigger=lambda x: x == 0, disable_logger=True)
    else:
        env = base_env

    pbar = tqdm(range(episodes), desc=f" {model_name} [Eps:{epsilon:.3f}|Def:{defense_type}]", leave=False)
    for ep_idx in pbar:
        obs, _ = env.reset()
        ep_reward, ep_steps = 0, 0
        ep_rpr_list = []
        
        while True:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            start_t = time.perf_counter()
            with torch.no_grad():
                # 1. ATN 生成对抗观测值
                adv_obs_t = atn(obs_t, epsilon)
                adv_obs_np = adv_obs_t.cpu().numpy()
                
                # ================= 防御策略植入点 =================
                if defense_type == "quantize":
                    # 防御1：降低精度，破坏微小梯度
                    adv_obs_np = np.round(adv_obs_np, decimals=2)
                elif defense_type == "noise":
                    # 防御2：注入随机高斯噪声，打乱扰动方向
                    noise = np.random.normal(0, 0.01, size=adv_obs_np.shape).astype(np.float32)
                    adv_obs_np = adv_obs_np + noise
                # ==================================================
                
                # 2. 目标模型基于处理后的观测值做出决策
                action, _ = model.predict(adv_obs_np, deterministic=True)
            
            latencies.append((time.perf_counter() - start_t) * 1000)
            
            # 计算 RPR (依然使用原始对抗样本和原样本计算，反映攻击者实际施加的扰动)
            pert = torch.norm(adv_obs_t - obs_t, p=2) / (torch.norm(obs_t, p=2) + 1e-8)
            ep_rpr_list.append(pert.item())
            
            obs, reward, terminated, truncated, _ = env.step(action[0])
            ep_reward += reward
            ep_steps += 1
            
            if terminated or truncated:
                # 双准则判定：摔倒 (steps < 800) 或 奖励减半
                if baseline_reward is not None:
                    if ep_steps < 800 or ep_reward < (baseline_reward * 0.5):
                        success_count += 1
                break
        
        all_rewards.append(ep_reward)
        all_rpr.append(np.mean(ep_rpr_list))
        pbar.set_postfix({"Rew": f"{ep_reward:.0f}"})
    
    env.close()
    return {
        "rew": np.mean(all_rewards),
        "asr": (success_count / episodes) * 100,
        "rpr": np.mean(all_rpr) * 100,
        "lat": np.mean(latencies)
    }

# ================= 3. 主程序 =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 初始化评估系统... 设备: {device}")
    
    # 结果目录设置
    res_dir = "./atn_detailed_results1"
    video_dir = os.path.join(res_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # 环境参数
    temp_env = gym.make("Humanoid-v5")
    obs_dim = temp_env.observation_space.shape[0]
    temp_env.close()

    # 加载攻击者模型
    atn = ATNGenerator(obs_dim).to(device)
    atn_model_path = "./atn_humanoid_model/atn_humanoid_model.pth"
    if os.path.exists(atn_model_path):
        atn.load_state_dict(torch.load(atn_model_path, map_location=device))
    atn.eval()

    target_configs = {"SAC": "model/humanoid-v5-SAC.zip", "TQC": "model/humanoid-v5-TQC.zip"}
    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
    defense_methods = ["none", "quantize", "noise"]
    
    # CSV 初始化 (增加 Defense 列)
    csv_file = os.path.join(res_dir, "full_epsilon_defense_metrics.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Defense', 'Epsilon', 'Reward', 'Drop_Rate(%)', 'ASR(%)', 'Avg_RPR(%)', 'Latency(ms)'])

        for name, path in target_configs.items():
            print(f"\n🎬 评估模型: {name}")
            if not os.path.exists(path):
                print(f"⚠️ 找不到模型文件: {path}，跳过...")
                continue
                
            model = SAC.load(path, device=device) if name == "SAC" else TQC.load(path, device=device)
            
            # 1. 跑一次无攻击获取 Baseline (固定为 none 防御)
            print(f" 🔍 正在校准基准分数...")
            base_data = run_evaluation(name, model, atn, 0.0, defense_type="none", episodes=20, device=device)
            baseline_rew = base_data['rew']
            print(f" 📊 基准奖励: {baseline_rew:.2f}")
            
            # 2. 正式开始多维度测试 (循环 Epsilon 和 防御策略)
            for eps in epsilons:
                print(f"\n ⚡ 正在测试 Epsilon: {eps:.3f}")
                for defense in defense_methods:
                    # 传入 video_dir 确保录制视频
                    res = run_evaluation(name, model, atn, eps, defense_type=defense, episodes=50, device=device, 
                                         video_root=video_dir, baseline_reward=baseline_rew)
                    
                    drop_rate = (1 - res['rew'] / baseline_rew) * 100 if baseline_rew > 0 else 0
                    
                    # 记录结果
                    writer.writerow([
                        name, 
                        defense,
                        f"{eps:.3f}", 
                        f"{res['rew']:.2f}", 
                        f"{max(0, drop_rate):.2f}", 
                        f"{res['asr']:.2f}", 
                        f"{res['rpr']:.4f}", 
                        f"{res['lat']:.4f}"
                    ])
                    print(f"   🛡️ 防御: {defense:<8} | Rew: {res['rew']:>5.0f} | Drop: {drop_rate:>5.1f}% | ASR: {res['asr']:>5.1f}%")

    print(f"\n✨ 评估流执行完毕！")
    print(f"📈 指标文件: {csv_file}")
    print(f"📹 视频证据目录: {video_dir}")