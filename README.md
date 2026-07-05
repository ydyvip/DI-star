# DI-star: 星际争霸II AI 分布式训练平台

> 已训练出宗师级 AI！支持 StarCraft II 最新 5.0 版本

## 项目简介

DI-star 是专为星际争霸II（StarCraft II）打造的**大规模游戏 AI 分布式训练平台**。本项目包含：

- [x] 对战演示与测试代码（可与 AI 对战！）
- [x] 预训练 SL（监督学习）和 RL（强化学习）AI 代理（当前仅支持虫族 vs 虫族）
- [x] 监督学习和强化学习训练代码
- [x] 有限资源（单台 PC）训练基线与指导
- [x] 与 [Harstem (YouTube)](https://www.youtube.com/watch?v=fvQF-24IpXs&t=813s) 对战过
- [ ] 更强的预训练 RL 代理（开发中）

## 环境要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **Python** | 3.8+ | 3.10-3.12 |
| **PyTorch** | 1.7.1+ | 2.0+ (CUDA 11.8+) |
| **CUDA** | 11.0 | 11.8 或 12.1 |
| **GPU** | 可选（CPU 也可运行）| NVIDIA RTX 3060+ |
| **星际争霸II** | 任意版本 | 最新 5.0+ 版本 |
| **操作系统** | Windows 10+/Linux/MacOS | Windows 11 / Ubuntu 22.04 |

> **GPU 提示**: 实时对战中使用 GPU 推理可以大幅提升 AI 反应速度。没有 CUDA 也可以运行，但性能会显著降低。

## 快速开始

### 第一步：安装星际争霸II

1. 访问 [starcraft2.com](https://starcraft2.com) 下载并安装星际争霸II
2. 确保游戏能正常运行一次
3. 系统会自动检测安装路径，通常位于：
   - Windows: `C:\Program Files (x86)\StarCraft II`
   - MacOS: `/Applications/StarCraft II`
   - Linux: `~/StarCraftII`

> 如果安装路径非默认位置，请添加环境变量 `SC2PATH` 指向安装目录。

### 第二步：安装 DI-star

```bash
# 克隆本仓库（已适配 5.0 版本）
git clone https://github.com/ydyvip/DI-star.git
cd DI-star

# 安装依赖
pip install -r requirements.txt
# 或
pip install -e .
```

### 第三步：下载 AI 模型

```bash
# 下载默认 RL 模型（推荐）
python -m distar.bin.download_model --name rl_model

# 可选模型：
# rl_model      - 强化学习模型，宗师级水平（默认）
# sl_model      - 监督学习模型，钻石级水平
# Abathur       - 喜欢出飞龙
# Brakk         - 喜欢小狗毒爆一波
# Dehaka        - 蟑螂火蟑螂打法
# Zagara        - 蟑螂一波
```

### 第四步：启动对战！

#### 模式 1：玩家 vs AI（人机对战）
```bash
# AI 控制一方，另一方由真人操作
python -m distar.bin.play --game_type human_vs_agent
```

运行后会启动两个星际争霸II窗口：
- 第一个窗口：AI 自动控制
- 第二个窗口：玩家操作（全屏，可正常游戏）

#### 模式 2：AI vs AI（代理互相对战）
```bash
# 两个 AI 互相较量
python -m distar.bin.play --game_type agent_vs_agent

# 指定不同模型对战
python -m distar.bin.play --game_type agent_vs_agent --model1 rl_model --model2 Abathur
```

#### 模式 3：AI vs 电脑（挑战内置 AI）
```bash
# 挑战内置精英难度电脑
python -m distar.bin.play --game_type agent_vs_bot
```

#### 无 GPU 运行选项
```bash
# 如果没有 CUDA GPU，添加 --cpu 参数
python -m distar.bin.play --cpu --game_type agent_vs_agent
```

## CUDA GPU 加速配置

### 检查 CUDA 可用性
```bash
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"
```

### 确保 GPU 模式启用
编辑 `distar/bin/user_config.yaml`：
```yaml
actor:
  use_cuda: True    # 开启 GPU 推理
  gpu_batch_inference: False
```

### 常见问题

| 问题 | 解决方案 |
|------|---------|
| `CUDA out of memory` | 降低游戏画质设置，或减小 batch size |
| `indices should be on cpu` | 已在本版本中修复，确保使用最新代码 |
| SC2 连接失败 | SC2 5.0 启动时有短暂关闭，已添加重试逻辑 |
| `np.int` 不存在 | numpy 2.0+ 兼容性已修复 |

## 训练自己的 AI

## 引用

```latex
@misc{distar,
    title={DI-star: An Open-sourse Reinforcement Learning Framework for StarCraftII},
    author={DI-star Contributors},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/ydyvip/DI-star}},
    year={2024},
}
```

## 许可证

本项目基于 Apache 2.0 许可证开源。
