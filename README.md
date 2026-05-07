# 强化学习对抗性攻击与防御（MuJoCo 实验集）

本仓库实现并评估了针对连续控制任务（主要为 `HalfCheetah-v5` 与 `Humanoid-v5`）的对抗攻击方法，包含白盒/黑盒攻击（PGD、FGSM、ATN）、影子模型训练与观测空间分析脚本。

## 快速概览
- 主要算法：`SAC`, `TQC`。
- 攻击方法：`PGD`（迭代投影梯度）、`FGSM`（单步符号梯度）、`ATN`（生成式扰动网络）。
- 防御/黑盒：影子模型（Shadow Model）、ATN 防御、黑盒评估脚本。

## 重要脚本与使用示例

- `pgd_sac.py` — 对 SAC/TQC 模型执行 PGD 白盒攻击并导出报告。
	- 关键配置（文件顶部）：`ENV_NAME`, `MODEL_PATH`, `EXPERIMENT_NAME`, `N_EPISODES`, `EPSILON_LIST`, `PGD_ITER`, `VIDEO_FOLDER`。
	- 运行：
		```bash
		python pgd_sac.py
		```
	- 输出：`{EXPERIMENT_NAME}_report.csv` 与 `VIDEO_FOLDER` 下的视频（第一个 episode）。

- `fgsm_sac.py` — 对 SAC 模型执行 FGSM 攻击。
	- 关键配置（文件顶部）：`ENV_NAME`, `MODEL_PATH`, `EXPERIMENT_NAME`, `N_EPISODES`, `EPSILON_LIST`, `VIDEO_FOLDER`。
	- 运行：
		```bash
		python fgsm_sac.py
		```
	-  输出：`{EXPERIMENT_NAME}_report.csv` 与 `VIDEO_FOLDER`。

- `atn_train.py` / `atn_train_hc.py` — 训练 ATN 生成器（防御/攻击器）。
	- 训练流程：加载专家模型 → 收集样本 → 训练 ATN → 保存模型（例如 `atn_humanoid_model.pth`）。
	- 运行：
		```bash
		python atn_train.py
		```

- `atn_test.py` / `atn_test_hc.py` — 使用训练好的 ATN 对模型做评估并记录 `full_epsilon_metrics.csv`，视频保存至 `atn_cheetah_results real/videos`（示例路径）。
	- 关键点：脚本会先跑 `epsilon=0` 获取基准分，再逐个 eps 评估并记录 ASR、平均 RPR、延迟等指标。

- `bb_train.py` — 为黑盒攻击训练影子模型（数据采集并训练替代网络）。
	- 关键配置：`EXPERT_MODEL_PATH`, `DATA_SAVE_PATH`, `SHADOW_MODEL_PATH`。
	- 运行：
		```bash
		python bb_train.py
		```

- `bb_test.py` — 使用已训练的影子模型进行集成黑盒 PGD（Ensemble PGD）攻击并输出 `{EXPERIMENT_NAME}_report.csv`。
	- 关键配置：`EXPERT_PATH`, `SHADOW_PATHS`, `EXPERIMENT_NAME`, `N_EPISODES`, `EPSILON_LIST`, `PGD_ITER`, `VIDEO_FOLDER`。
	- 运行：
		```bash
		python bb_test.py
		```

- `obs_analysis.py` — 采集观测扰动样本并绘制 4 张分析图（时序、稀疏性热力图、t-SNE 对比、条形码可视化），输出到 `./obs_analysis_figs1`。
	- 运行：
		```bash
		python obs_analysis.py
		```

- `plot.py` / `plot PGD vs M-PGD.py` — 使用已有 CSV (`data1.csv`, `data2.csv`) 绘制奖励与 ASR 曲线，输出图片 `reward_vs_epsilon.png`、`asr_vs_epsilon.png`。

## 输出约定
- 报告 CSV：脚本通常保存为 `{EXPERIMENT_NAME}_report.csv` 或 `*_report.csv`，列包含 `Epsilon`, `Reward`, `RPR(%)`, `ASR(%)`, `Latency(ms)` 或更丰富的聚合信息。
- 视频：每个 `VIDEO_FOLDER` 通常只保存每个 epsilon 的第 1 个 episode（便于示例回放）。
- ATN 评估会在结果目录写入 `full_epsilon_metrics.csv`。

## 环境与依赖
建议使用 Python 3.10+，并安装以下核心包：

```
gymnasium
stable-baselines3
sb3-contrib
torch
numpy
pandas
matplotlib
scipy
scikit-learn
seaborn
tqdm

# 可选（用于渲染与视频）：
opencv-python
imageio
```

可通过 `pip` 安装：

```bash
pip install gymnasium stable-baselines3 sb3-contrib torch numpy pandas matplotlib scipy scikit-learn seaborn tqdm opencv-python imageio
```

注意：若需 GPU 支持，请安装对应 CUDA 版本的 `torch`（参考 PyTorch 官方安装页）。

## 修改常用配置
- 修改脚本顶部的变量即可（例如 `MODEL_PATH`、`ENV_NAME`、`EPSILON_LIST`、`N_EPISODES`、`VIDEO_FOLDER` 等）。

示例（修改 `pgd_sac.py`）：
```python
ENV_NAME = "Humanoid-v5"
MODEL_PATH = "./model/humanoid-v5-SAC.zip"
EPSILON_LIST = [0.0, 0.01, 0.02, 0.03]
PGD_ITER = 10
N_EPISODES = 30
```

## 建议的运行顺序（实验复现）
1. 准备并放置专家模型到 `model/`（例如 `humanoid-v5-SAC.zip` 或 `halfcheetah-v5-TQC.zip`）。
2. 如需黑盒攻击，先运行 `python bb_train.py` 训练影子模型。
3. 运行 `python pgd_sac.py` / `python fgsm_sac.py` 进行白盒攻击评估。
4. 运行 `python atn_train.py`（如需）然后 `python atn_test.py` 评估 ATN。
5. 使用 `python obs_analysis.py` 与 `python plot.py` 生成分析图与最终曲线图。

## 常见问题
- 如果脚本找不到模型文件，确认 `MODEL_PATH` 或 `EXPERT_MODEL_PATH` 指向正确路径。
- 若视频无法保存，检查 `VIDEO_FOLDER` 的写权限与磁盘空间。



