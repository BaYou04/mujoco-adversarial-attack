import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import numpy as np
import os

# ================= ATN 攻击网络 =================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, dim)
        )
    def forward(self, x):
        return x + 0.1 * self.net(x)

class ATNGenerator(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.model = nn.Sequential(ResBlock(obs_dim), ResBlock(obs_dim), nn.Linear(obs_dim, obs_dim))
    def forward(self, obs, epsilon):
        delta = torch.tanh(self.model(obs))
        return obs + epsilon * delta

# ================= 单回合评估 + 录视频 =================
def evaluate_humanoid_attack(model_path, atn_path, epsilon=0.03, video_save_dir="./humanoid_attack_video"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(video_save_dir, exist_ok=True)

    # 加载环境
    env = gym.make("Humanoid-v5", render_mode="rgb_array", terminate_when_unhealthy=False)
    obs_dim = env.observation_space.shape[0]

    # 加载攻击模型
    atn = ATNGenerator(obs_dim).to(device)
    atn.load_state_dict(torch.load(atn_path, map_location=device))
    atn.eval()

    # 加载目标策略（默认用 SAC）
    from stable_baselines3 import SAC
    model = SAC.load(model_path, device=device)

    # 包装视频录制（只录这 1 回合）
    env = RecordVideo(
        env, video_folder=video_save_dir,
        episode_trigger=lambda x: x == 0,
        disable_logger=True
    )

    # 开始单回合攻击
    print(f"🎬 开始单回合攻击 Humanoid | eps={epsilon}")
    obs, _ = env.reset()
    total_reward = 0
    step = 0

    while step < 400:  # 最多 1000 步
        # 生成对抗观测
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            adv_obs = atn(obs_t, epsilon)
        adv_obs = adv_obs.cpu().numpy()

        # 动作推理
        action, _ = model.predict(adv_obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action[0])

        total_reward += reward
        step += 1

        # 只在自然结束时停止（不提前终止）
        if terminated or truncated:
            break

    env.close()
    print(f"✅ 回合结束 | 总步数={step} | 总奖励={total_reward:.2f}")
    print(f"📹 视频已保存到: {video_save_dir}")
    return total_reward

# ================= 主程序 =================
if __name__ == "__main__":
    # 你只需要改这 3 个路径！
    TARGET_MODEL_PATH = "model/humanoid-v5-SAC.zip"  # 你的目标模型
    ATK_MODEL_PATH    = "atn_humanoid_model/atn_humanoid_model.pth"     # 你的攻击网络
    ATTACK_EPSILON    = 0.03                         # 攻击强度

    evaluate_humanoid_attack(
        model_path=TARGET_MODEL_PATH,
        atn_path=ATK_MODEL_PATH,
        epsilon=ATTACK_EPSILON,
        video_save_dir="./zhumanoid_attack_video"
    )