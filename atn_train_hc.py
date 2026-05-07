import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import zipfile
import io
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
def train_cheetah_atn(obs_path, sac_paths, tqc_paths, epsilon=0.05, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "./atn_cheetah_results/"
    os.makedirs(save_dir, exist_ok=True)
    
    # A. 数据加载
    print(f"📂 Loading data: {obs_path}")
    data = np.load(obs_path)
    obs_data = data['obs'] if 'obs' in data.files else data[data.files[0]]
    # 如果内存压力大，建议只用 500K: obs_data = obs_data[:500000]
    obs_tensor = torch.FloatTensor(obs_data).to(device)
    loader = DataLoader(TensorDataset(obs_tensor), batch_size=1024, shuffle=True)
    
    obs_dim = obs_data.shape[1]
    action_dim = 6 
    print(f"📊 Detect Dimensions | Obs: {obs_dim}, Action: {action_dim}")

    # C. 加载混合模型 (专家 .zip + 影子 .pth)
    def load_hybrid_group(paths):
        models = []
        for p in paths:
            m = Actor(obs_dim, action_dim).to(device)
            if p.endswith('.zip'):
                with zipfile.ZipFile(p, 'r') as archive:
                    with archive.open('policy.pth') as f:
                        state_dict = torch.load(io.BytesIO(f.read()), map_location=device)
                        
                        # 根据你提供的打印结果进行的精准映射
                        try:
                            # 映射到你定义的 Actor(nn.Sequential) 结构
                            new_state_dict = {
                                "net.0.weight": state_dict["actor.latent_pi.0.weight"],
                                "net.0.bias":   state_dict["actor.latent_pi.0.bias"],
                                "net.2.weight": state_dict["actor.latent_pi.2.weight"],
                                "net.2.bias":   state_dict["actor.latent_pi.2.bias"],
                                "net.4.weight": state_dict["actor.mu.weight"],
                                "net.4.bias":   state_dict["actor.mu.bias"]
                            }
                            m.load_state_dict(new_state_dict)
                            print(f"✅ 专家模型 {os.path.basename(p)} 映射成功!")
                        except KeyError as e:
                            print(f"❌ 映射再次失败！请检查该模型的 Key。缺失: {e}")
                            # 如果 TQC 的命名不一样，这里会打印出来
                            print(f"模型实测 Keys: {list(state_dict.keys())[:10]}")
                            raise e
            else:
                # 影子模型 (.pth) 保持原样
                m.load_state_dict(torch.load(p, map_location=device))
                print(f"✅ 影子模型 {os.path.basename(p)} 加载成功!")
            
            m.eval()
            for param in m.parameters(): param.requires_grad = False
            models.append(m)
        return models
    
    print(f"📦 Loading {len(sac_paths) + len(tqc_paths)} Hybrid Models...")
    sac_group = load_hybrid_group(sac_paths)
    tqc_group = load_hybrid_group(tqc_paths)

    generator = ATNGenerator(obs_dim).to(device)
    optimizer = optim.Adam(generator.parameters(), lr=3e-4)
    
    history = {'loss': [], 'sac_mse': [], 'tqc_mse': []}
    pbar_total = tqdm(range(1, epochs + 1), desc="🚀 Training ATN (Cheetah)", colour='green')
    
    for epoch in pbar_total:
        e_sac_mse, e_tqc_mse, e_loss = 0, 0, 0
        generator.train()
        for batch in loader:
            obs_batch = batch[0]
            optimizer.zero_grad()
            
            adv_obs = generator(obs_batch, epsilon)
            sim_loss = torch.mean(torch.norm(adv_obs - obs_batch, p=2, dim=1))
            
            s_mse = sum([torch.mean(torch.pow(m(adv_obs)-m(obs_batch), 2)) for m in sac_group]) / len(sac_group)
            t_mse = sum([torch.mean(torch.pow(m(adv_obs)-m(obs_batch), 2)) for m in tqc_group]) / len(tqc_group)

            # 损失函数：如果影子模型太差，建议把 150 改成 500 以获得更明显的梯度
            loss = sim_loss - 150.0 * (s_mse + t_mse)
            
            loss.backward()
            optimizer.step()
            
            e_sac_mse += s_mse.item(); e_tqc_mse += t_mse.item(); e_loss += loss.item()

        avg_s = e_sac_mse / len(loader); avg_t = e_tqc_mse / len(loader); avg_l = e_loss / len(loader)
        history['sac_mse'].append(avg_s); history['tqc_mse'].append(avg_t); history['loss'].append(avg_l)
        pbar_total.set_postfix({"L": f"{avg_l:.2f}", "S_M": f"{avg_s:.4f}", "T_M": f"{avg_t:.4f}"})

    # ================= E. 保存模型与【绘图逻辑】 =================
    # 1. 保存模型
    torch.save(generator.state_dict(), os.path.join(save_dir, "atn_cheetah_hybrid.pth"))
    
    # 2. 绘制训练曲线
    print("\n📊 Generating training plots...")
    plt.figure(figsize=(12, 5))
    
    # 左图：总损失
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Total Loss', color='tab:red')
    plt.title('ATN Total Loss Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 右图：动作偏差 (MSE)
    plt.subplot(1, 2, 2)
    plt.plot(history['sac_mse'], label='SAC Mean MSE', color='tab:green')
    plt.plot(history['tqc_mse'], label='TQC Mean MSE', color='tab:orange')
    plt.title('Action Deviation (Attack Strength)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    report_path = os.path.join(save_dir, "training_report.png")
    plt.savefig(report_path)
    plt.close() # 释放内存
    
    print(f"✅ 训练完成！\n💾 模型已保存至: {save_dir}atn_cheetah_hybrid.pth\n📈 曲线图已生成: {report_path}")

if __name__ == "__main__":
    base = "translearning/" 
    config = {
        "obs_path": base + "data/hc_SAC_data_1000k.npz", 
        "sac_paths": [
            "model/halfcheetah-v5-SAC.zip",         
            base + "model/shadow_hc_SAC_model_1000K.pth"   
        ],
        "tqc_paths": [
            "model/halfcheetah-v5-TQC.zip",         
            base + "model/shadow_hc_TQC_model_1000K.pth"   
        ],
        "epsilon": 0.03,
        "epochs": 50
    }
    train_cheetah_atn(**config)