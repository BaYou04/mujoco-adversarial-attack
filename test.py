import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from tqdm import tqdm

def pgd_attack(obs, model, env, epsilon=0.01, n_iter=10):
    """
    基于动作 MSE 最大化的 PGD 白盒攻击核心函数 (维度安全稳定版)
    """
    if epsilon == 0:
        return obs
        
    alpha = epsilon / 8.0
    device = model.device
    
    # 1. 获取干净环境下的确定性动作 (作为攻击偏离的基准)
    # 注意: 显式扩展 batch 维度变成 [1, obs_dim]，绝对防止 SB3 内部 flatten 报错
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        # 这里加上 deterministic=True 保证基准动作是稳定的
        clean_action = model.policy.actor(obs_tensor, deterministic=True).detach()

    # 2. 随机初始化扰动起点 (在 Numpy 层面操作)
    adv_obs = obs.copy() + np.random.uniform(-epsilon, epsilon, obs.shape)
    adv_obs = np.clip(adv_obs, obs - epsilon, obs + epsilon)
    adv_obs = np.clip(adv_obs, env.observation_space.low, env.observation_space.high)

    # 3. PGD 多步迭代
    for _ in range(n_iter):
        # 将 adv_obs 转化为 [1, obs_dim] 的 2D tensor 参与求导
        adv_obs_tensor = torch.tensor(adv_obs, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
        
        # 前向传播得到当前对抗观测下的动作 (传入 2D tensor，底层畅通无阻)
        current_action = model.policy.actor(adv_obs_tensor, deterministic=True)
        
        # 计算与干净动作的 MSE Loss，迫使动作向相反/错误的方向偏移
        loss = F.mse_loss(current_action, clean_action)
        
        # 清空梯度并反向传播
        model.policy.zero_grad()
        if adv_obs_tensor.grad is not None:
            adv_obs_tensor.grad.zero_grad()
        loss.backward()
        
        # 提取梯度符号并放缩，squeeze(0) 安全地将 [1, obs_dim] 降回 [obs_dim] 给 Numpy 计算
        perturbation = alpha * adv_obs_tensor.grad.sign().cpu().numpy().squeeze(0)
        adv_obs = adv_obs + perturbation
        
        # 投影回 epsilon 球内，并确保不超过环境合法的观测边界
        delta = np.clip(adv_obs - obs, -epsilon, epsilon)
        adv_obs = np.clip(obs + delta, env.observation_space.low, env.observation_space.high)

    return adv_obs

def main():
    # --- 参数配置区 ---
    MODEL_PATH = "model/humanoid-v5-SAC.zip"  # 请替换为您本地的真实路径
    EPSILON = 0.0
    PGD_ITER = 10                             # PGD 迭代次数
    TOTAL_STEPS = 500

    # --- 环境初始化 ---
    env = gym.make(
        "Humanoid-v5",
        render_mode="human",
        terminate_when_unhealthy=False,
        max_episode_steps=TOTAL_STEPS
    )

    try:
        # --- 模型加载 ---
        print(f"正在加载 SAC 模型: {MODEL_PATH} ...")
        model = SAC.load(MODEL_PATH, env=env)
        
        obs, info = env.reset()
        total_reward = 0.0

        # --- 攻击与评估循环 ---
        for step in tqdm(range(TOTAL_STEPS), desc=f"PGD 攻击测试 (Eps={EPSILON}, Iter={PGD_ITER})"):
            # 替换为 PGD 攻击方法，注意需要传入 env 获取边界限制
            adv_obs = pgd_attack(obs, model, env, epsilon=EPSILON, n_iter=PGD_ITER)
            
            action, _states = model.predict(adv_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward

            if terminated or truncated:
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\n\n检测到手动中断，正在安全退出并清理环境...")
        
    except Exception as e:
        print(f"\n\n运行中发生异常: {e}")
        print("正在安全清理环境...")
        
    finally:
        env.close()
        print("\n" + "="*40)
        print("环境已完全清理释放。")
        print(f"当前已累积真实奖励: {total_reward:.2f}")
        print("="*40)

if __name__ == "__main__":
    main()