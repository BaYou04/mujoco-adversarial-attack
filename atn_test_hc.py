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
import io
import zipfile

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
    success_count = 0
    
    # 确定视频存放的具体子路径
    current_video_dir = None
    if video_root is not None:
        current_video_dir = os.path.join(video_root, f"{model_name}_eps_{epsilon:.3f}")
        os.makedirs(current_video_dir, exist_ok=True)

    # 初始化环境 (HalfCheetah-v5)
    base_env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    
    # 配置录像：只录制每个测试组的第 0 个回合
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
        
        while True:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            start_t = time.perf_counter()
            
            with torch.no_grad():
                # 生成对抗观测值
                adv_obs_t = atn(obs_t, epsilon)
                # 目标模型预测动作
                action, _ = model.predict(adv_obs_t.cpu().numpy(), deterministic=True)
            
            latencies.append((time.perf_counter() - start_t) * 1000)
            
            # 计算相对扰动率 RPR
            pert = torch.norm(adv_obs_t - obs_t, p=2) / (torch.norm(obs_t, p=2) + 1e-8)
            ep_rpr_list.append(pert.item())
            
            obs, reward, terminated, truncated, _ = env.step(action[0])
            ep_reward += reward
            ep_steps += 1
            
            if terminated or truncated:
                # 判定攻击成功：根据步数或奖励跌幅
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
    print(f"🚀 启动评估系统... 当前设备: {device}")
    
    # 路径配置
    res_dir = "./atn_cheetah_results"
    video_dir = os.path.join(res_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # 获取观测维度
    temp_env = gym.make("HalfCheetah-v5")
    obs_dim = temp_env.observation_space.shape[0]
    temp_env.close()

    # 1. 加载训练好的 ATN 生成器
    atn = ATNGenerator(obs_dim).to(device)
    atn_model_path = os.path.join(res_dir, "atn_cheetah_model.pth") # 确保文件名与你保存的一致
    atn.load_state_dict(torch.load(atn_model_path, map_location=device))
    atn.eval()

    # 2. 目标模型配置
    target_configs = {
        "SAC": "model/halfcheetah-v5-SAC.zip", 
        "TQC": "model/halfcheetah-v5-TQC.zip"
    }
    
    # 3. 攻击强度列表 (不含 0.0，因为 0.0 单独校准)
    epsilons = [0.030, 0.040]
    
    # 4. CSV 数据初始化
    csv_file = os.path.join(res_dir, "full_epsilon_metrics1.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Epsilon', 'Reward', 'Drop_Rate(%)', 'ASR(%)', 'Avg_RPR(%)', 'Latency(ms)'])

        for name, path in target_configs.items():
            print(f"\n🎬 正在评估目标模型: {name}")
            
            # 加载被攻击模型
            if name == "SAC":
                model = SAC.load(path, device=device)
            else:
                model = TQC.load(path, device=device)
            
            # --- 第一步：基准校准 (录制正常视频) ---
            print(f" 🔍 正在执行基准测试并录制视频...")
            base_data = run_evaluation(name, model, atn, 0.0, episodes=20, device=device, video_root=video_dir)
            baseline_rew = base_data['rew']
            print(f" 📊 {name} 基准奖励: {baseline_rew:.2f}")

            # 写入基准行
            writer.writerow([name, 0.0, f"{baseline_rew:.2f}", 0.0, 0.0, 0.0, f"{base_data['lat']:.4f}"])
            
            # --- 第二步：正式攻击测试 ---
            for eps in epsilons:
                res = run_evaluation(name, model, atn, eps, episodes=50, device=device, 
                                     video_root=video_dir, baseline_reward=baseline_rew)
                
                # 计算奖励下降率
                drop_rate = (1 - res['rew'] / baseline_rew) * 100 if baseline_rew > 0 else 0
                
                # 写入 CSV
                writer.writerow([
                    name, 
                    f"{eps:.3f}", 
                    f"{res['rew']:.2f}", 
                    f"{max(0, drop_rate):.2f}", 
                    f"{res['asr']:.2f}", 
                    f"{res['rpr']:.4f}", 
                    f"{res['lat']:.4f}"
                ])
                print(f" ✅ Eps {eps:.3f} | Rew: {res['rew']:.0f} | Drop: {drop_rate:.1f}% | ASR: {res['asr']:.1f}%")

    print(f"\n✨ 评估流全部完成！")
    print(f"📈 指标统计表: {csv_file}")
    print(f"📹 视频录像目录: {video_dir}")