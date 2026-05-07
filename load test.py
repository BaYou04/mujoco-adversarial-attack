"""
本地已训练 SAC 模型运行 + MuJoCo 可视化窗口
支持 Humanoid-v5 / HalfCheetah-v5
"""
import gymnasium as gym
from stable_baselines3 import SAC
import time

# ====================== 你只需要改这里 ======================
ENV_NAME = "HalfCheetah-v5"       # 或者 "Humanoid-v5"
MODEL_PATH = "model/halfcheetah-v5-SAC.zip"  # 例如 "sac_halfcheetah_v5.zip"
# ==========================================================

def run_trained_model():
    # 创建带可视化窗口的环境
    env = gym.make(ENV_NAME, render_mode="human")

    # 加载你本地已经训练好的 SAC 模型
    model = SAC.load(MODEL_PATH, env=env)

    print("模型加载成功！开始运行...")

    # 运行 10 轮，自动循环
    for episode in range(10):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            time.sleep(0.01)

        print(f"Episode {episode+1} 奖励: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    run_trained_model()