import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================= 1. 网络架构定义 =================
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

# ================= 2. 核心训练逻辑 =================
def train_ensemble_atn(obs_path, sac_paths, tqc_paths, epsilon=0.03, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建带时间戳的目录，防止覆盖之前的实验结果
    save_dir = "./atn_ensemble_results/"
    os.makedirs(save_dir, exist_ok=True)
    
    # A. 数据加载
    data = np.load(obs_path)
    obs_data = data['obs'] if 'obs' in data.files else data[data.files[0]]
    obs_tensor = torch.FloatTensor(obs_data).to(device)
    loader = DataLoader(TensorDataset(obs_tensor), batch_size=1024, shuffle=True)
    
    obs_dim = obs_data.shape[1]
    action_dim = 17

    # B. 加载影子模型
    def load_group(paths):
        models = []
        for p in paths:
            m = Actor(obs_dim, action_dim).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            m.eval()
            for param in m.parameters(): param.requires_grad = False
            models.append(m)
        return models

    print("📦 Loading 6 Shadow Models...")
    sac_group = load_group(sac_paths)
    tqc_group = load_group(tqc_paths)

    # C. 初始化
    generator = ATNGenerator(obs_dim).to(device)
    optimizer = optim.Adam(generator.parameters(), lr=3e-4)
    
    # 用于绘图的历史记录
    history = {'loss': [], 'sac_mse': [], 'tqc_mse': []}
    
    # D. 进度条控制
    pbar_total = tqdm(range(1, epochs + 1), desc="🚀 Total Progress", colour='cyan')
    
    generator.train()
    for epoch in pbar_total:
        e_sac_mse, e_tqc_mse, e_loss = 0, 0, 0
        pbar_batch = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        
        for batch in pbar_batch:
            obs_batch = batch[0]
            optimizer.zero_grad()
            
            adv_obs = generator(obs_batch, epsilon)
            sim_loss = torch.mean(torch.norm(adv_obs - obs_batch, p=2, dim=1))
            
            # 计算两组算法的动作偏差
            s_mse = sum([torch.mean(torch.pow(m(adv_obs)-m(obs_batch), 2)) for m in sac_group]) / len(sac_group)
            t_mse = sum([torch.mean(torch.pow(m(adv_obs)-m(obs_batch), 2)) for m in tqc_group]) / len(tqc_group)

            # 联合损失 (150为放大系数，0.8/1.5为算法均衡权重)
            loss = sim_loss - 150.0 * (0.8 * s_mse + 1.5 * t_mse)
            
            loss.backward()
            optimizer.step()
            
            e_sac_mse += s_mse.item()
            e_tqc_mse += t_mse.item()
            e_loss += loss.item()
            
            pbar_batch.set_postfix({"Loss": f"{loss.item():.2f}", "S": f"{s_mse:.4f}", "T": f"{t_mse:.4f}"})

        # 记录 Epoch 平均值
        avg_s = e_sac_mse / len(loader)
        avg_t = e_tqc_mse / len(loader)
        avg_l = e_loss / len(loader)
        
        history['sac_mse'].append(avg_s)
        history['tqc_mse'].append(avg_t)
        history['loss'].append(avg_l)
        
        pbar_total.set_postfix({"Loss": f"{avg_l:.2f}", "S_M": f"{avg_s:.4f}", "T_M": f"{avg_t:.4f}"})

    # E. 保存模型与绘图
    torch.save(generator.state_dict(), os.path.join(save_dir, "atn_ensemble_model123.pth"))
    
    # 绘制曲线图
    plt.figure(figsize=(12, 5))
    
    # 子图 1: Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Total Loss', color='blue')
    plt.title('Training Loss (Convergence)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.legend()

    # 子图 2: MSE 距离曲线 (SAC vs TQC)
    plt.subplot(1, 2, 2)
    plt.plot(history['sac_mse'], label='SAC Avg MSE', color='green')
    plt.plot(history['tqc_mse'], label='TQC Avg MSE', color='orange')
    plt.title('Action Deviation (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Distance')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    report_path = os.path.join(save_dir, "ensemble_training_report123.png")
    plt.savefig(report_path)
    
    print(f"\n✅ 训练完成！")
    print(f"📊 训练曲线图已保存至: {report_path}")
    print(f"💾 模型已保存至: {os.path.join(save_dir, 'atn_ensemble_model.pth')}")

if __name__ == "__main__":
    base = "translearning/"
    config = {
        "obs_path": base + "data/humanoid_SAC_data_500k.npz",
        "sac_paths": [base + f"model/shadow_humanoid_SAC_model_{k}.pth" for k in ["50K", "200K", "500K"]],
        "tqc_paths": [base + f"model/shadow_humanoid_TQC_model_{k}.pth" for k in ["50K", "200K", "500K"]],
        "epsilon": 0.03,
        "epochs": 100
    }
    train_ensemble_atn(**config)