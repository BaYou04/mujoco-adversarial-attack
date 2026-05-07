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
def run_evaluation(model_name, model, atn, epsilon, episodes=50, device="cpu", video_root=None, baseline_reward=None):
    all_rewards, all_rpr, latencies = [], [], []
    attack_rates = [] # 新增：记录每回合的攻击频率
    success_count = 0
    
    # 确定视频存放的具体子路径
    current_video_dir = None
    if video_root and epsilon > 0:
        current_video_dir = os.path.join(video_root, f"{model_name}_eps_{epsilon:.3f}")
        os.makedirs(current_video_dir, exist_ok=True)

    # 初始化环境
    base_env = gym.make("Humanoid-v5", render_mode="rgb_array")
    
    # 如果开启视频录制，包装环境 (只录制第 0 回合)
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
        ep_attack_count = 0  # 记录当前回合触发攻击的次数
        
        burn_in_mses = []    # 隐匿分析期收集的 MSE
        mu = 0.0             # 动态基线（均值）
        
        while True:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            start_t = time.perf_counter()
            
            with torch.no_grad():
                # 1. ATN 生成带有扰动的对抗观测值
                adv_obs_t = atn(obs_t, epsilon)
                adv_obs_np = adv_obs_t.cpu().numpy()
                
                # 2. 后台推演：分别计算干净观测和对抗观测对应的动作
                clean_action, _ = model.predict(obs_t.cpu().numpy(), deterministic=True) # 替换为 obs_t
                adv_action, _ = model.predict(adv_obs_np, deterministic=True)
                
                # 3. 计算动作偏差 MSE
                current_mse = np.mean((clean_action - adv_action) ** 2)
            
            latencies.append((time.perf_counter() - start_t) * 1000)
            
            # ================= 稀疏攻击时机判定逻辑 =================
            executed_action = clean_action  # 默认执行干净动作
            applied_rpr = 0.0               # 默认本步未施加扰动，RPR 为 0
            
            if ep_steps < 100:
                # 阶段一：隐匿分析期 (前100步)
                burn_in_mses.append(current_mse)
            else:
                # 阶段二：伺机狙击期 (100步之后)
                if ep_steps == 100:
                    # 恰好第100步时，计算基线均值 mu
                    mu = np.mean(burn_in_mses) if len(burn_in_mses) > 0 else 0.0
                
                # 触发条件：当前动作偏差 > 前100步的平均偏差
                # 注意：如果是 Baseline 测试 (epsilon=0)，current_mse 为 0，不会触发攻击
                if current_mse > mu and epsilon > 0:
                    executed_action = adv_action
                    ep_attack_count += 1
                    # 只有真正发动攻击时，才记录产生的相对扰动大小
                    applied_rpr = (torch.norm(adv_obs_t - obs_t, p=2) / (torch.norm(obs_t, p=2) + 1e-8)).item()
            # ========================================================
            
            ep_rpr_list.append(applied_rpr)
            
            # 将最终裁决的动作发给环境执行
            obs, reward, terminated, truncated, _ = env.step(executed_action[0])
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
        # 计算本回合的攻击频率 (触发攻击次数 / 总步数)
        attack_rates.append((ep_attack_count / ep_steps) * 100)
        pbar.set_postfix({"Rew": f"{ep_reward:.0f}"})
    
    env.close()
    return {
        "rew": np.mean(all_rewards),
        "asr": (success_count / episodes) * 100,
        "rpr": np.mean(all_rpr) * 100,
        "lat": np.mean(latencies),
        "atk_rate": np.mean(attack_rates)  # 返回攻击频率
    }

# ================= 3. 主程序 =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 初始化评估系统... 设备: {device}")
    
    # 结果目录设置
    res_dir = "./atn_timing_results"
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
    epsilons = [ 0.025, 0.035]
    
    # CSV 初始化 (增加了 Atk_Rate(%) 列)
    csv_file = os.path.join(res_dir, "full_epsilon_metrics1.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Epsilon', 'Reward', 'Drop_Rate(%)', 'ASR(%)', 'Avg_RPR(%)', 'Latency(ms)', 'Atk_Rate(%)'])

        for name, path in target_configs.items():
            print(f"\n🎬 评估模型: {name}")
            if not os.path.exists(path):
                print(f"⚠️ 找不到模型: {path}，跳过...")
                continue
                
            model = SAC.load(path, device=device) if name == "SAC" else TQC.load(path, device=device)
            
            # 1. 跑一次无攻击获取 Baseline
            print(f" 🔍 正在校准基准分数...")
            base_data = run_evaluation(name, model, atn, 0.0, episodes=20, device=device)
            baseline_rew = base_data['rew']
            print(f" 📊 基准奖励: {baseline_rew:.2f}")
            
            # 2. 正式开始多维度测试
            for eps in epsilons:
                res = run_evaluation(name, model, atn, eps, episodes=50, device=device, 
                                     video_root=video_dir, baseline_reward=baseline_rew)
                
                drop_rate = (1 - res['rew'] / baseline_rew) * 100 if baseline_rew > 0 else 0
                
                # 记录结果
                writer.writerow([
                    name, 
                    f"{eps:.3f}", 
                    f"{res['rew']:.2f}", 
                    f"{max(0, drop_rate):.2f}", 
                    f"{res['asr']:.2f}", 
                    f"{res['rpr']:.4f}", 
                    f"{res['lat']:.4f}",
                    f"{res['atk_rate']:.2f}"  # 写入攻击频率
                ])
                print(f" ✅ Eps {eps:.3f} | Rew: {res['rew']:.0f} | ASR: {res['asr']:.1f}% | Atk_Rate: {res['atk_rate']:.1f}%")

    print(f"\n✨ 评估流执行完毕！")
    print(f"📈 指标文件: {csv_file}")
    print(f"📹 视频证据目录: {video_dir}")