# EchoNet-Dynamic 实验指南

## 快速开始

### 1. 服务器登录后初始化
```bash
# 进入项目目录
cd /root/autodl-tmp/cardio/dynamic-master/dynamic-master

# 激活环境
conda activate torch_env

# 安装项目 (首次运行)
pip install -e . -q
pip install tensorboard -q
```

### 2. 运行实验

#### 方式一：完整实验套件 (推荐)
```bash
# 运行所有实验 (约12小时)
python scripts/run_experiments.py --mode full

# 或分阶段运行:
python scripts/run_experiments.py --mode reproduce   # 论文复现 (~3小时)
python scripts/run_experiments.py --mode core        # 核心消融 (~6小时)
python scripts/run_experiments.py --mode temporal    # 时序消融 (~3小时)
```

#### 方式二：快速验证 (调试用)
```bash
# 快速对比实验 (约20分钟)
python scripts/hyperparameter_tuning.py --mode quick --epochs 5 --subset 500
```

---

## 实验内容

### 第一部分：论文复现 (~3小时)
| 实验名 | 配置 | 目标 |
|--------|------|------|
| 00_paper_reproduction | R(2+1)D-18, pretrained, 45 epochs | R² ≈ 0.81, MAE ≈ 4.1% |

### 第二部分：核心消融实验 (~6小时)
| 实验名 | 变量 | 说明 |
|--------|------|------|
| 01_pretrain_yes | 预训练 | Kinetics-400预训练 |
| 02_pretrain_no | 预训练 | 随机初始化 |
| 03_arch_r2plus1d | 架构 | R(2+1)D: 时空分解 |
| 04_arch_r3d | 架构 | R3D: 完整3D卷积 |
| 05_arch_mc3 | 架构 | MC3: 混合卷积 |
| 06_lr_1e-3 | 学习率 | 较大学习率 |
| 07_lr_1e-4 | 学习率 | 论文默认 |
| 08_lr_1e-5 | 学习率 | 较小学习率 |

### 第三部分：时序消融实验 (~3小时)
| 实验名 | 变量 | 说明 |
|--------|------|------|
| 09_frames_16 | 帧数 | 16帧 (覆盖~1秒) |
| 10_frames_32 | 帧数 | 32帧 (论文) |
| 11_frames_64 | 帧数 | 64帧 (覆盖~4秒) |
| 12_period_1 | 采样 | 连续采样 |
| 13_period_2 | 采样 | 隔帧采样 (论文) |
| 14_period_4 | 采样 | 稀疏采样 |

---

## 断点保护

实验具有自动断点保护功能：

### 实验中断后恢复
```bash
# 自动从上次中断处继续 (默认开启，无需额外参数)
python scripts/run_experiments.py --mode full

# 如果要强制重新开始，使用 --no_resume
python scripts/run_experiments.py --mode full --no_resume
```

### 检查点文件
- `output/experiments/suite_progress.json` - 套件级进度
- `output/experiments/<exp_name>/checkpoint.pt` - 每个实验的检查点
- `output/experiments/<exp_name>/best.pt` - 最佳模型

---

## 查看结果

### TensorBoard可视化
```bash
# 启动TensorBoard (在服务器上)
tensorboard --logdir=output/experiments --port 6006 --bind_all

# 然后在本地浏览器打开:
# http://<服务器IP>:6006
```

### 命令行查看
```bash
# 查看实验汇总
cat output/experiments/final_summary.csv

# 查看单个实验结果
cat output/experiments/01_pretrain_yes/results.json
```

### 结果文件说明
```
output/experiments/
├── suite_progress.json      # 整体进度
├── final_summary.csv        # 所有实验汇总表
├── 00_paper_reproduction/
│   ├── tensorboard/         # TensorBoard日志
│   ├── history.csv          # 训练历史
│   ├── results.json         # 最终结果
│   ├── best.pt              # 最佳模型
│   └── checkpoint.pt        # 断点文件
└── ...
```

---

## 服务器优化配置

### 已优化参数
- `batch_size=32` (充分利用32GB显存)
- `num_workers=8` (充分利用多核CPU)
- 自动混合精度 (可选)

### 显存监控
```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi
```

---

## 预期结果

### 论文复现目标
| 指标 | 论文值 | 预期范围 |
|------|--------|----------|
| R² | 0.81 | 0.78-0.83 |
| MAE | 4.1% | 3.8-4.5% |
| RMSE | 5.3% | 5.0-5.8% |

### 消融实验预期
- **预训练 vs 从头训练**: 预训练应提升 ~5-10% R²
- **R(2+1)D vs R3D**: R(2+1)D略优 (~1-2%)
- **学习率**: 1e-4 最稳定
- **帧数**: 32帧是最佳平衡点
- **采样间隔**: period=2 覆盖完整心动周期

---

## 常见问题

### Q: 实验中断了怎么办？
```bash
python scripts/run_experiments.py --mode all --resume
```

### Q: 如何只重新运行某个实验？
编辑 `suite_progress.json`，将该实验的 `completed` 设为 `false`

### Q: 显存不足？
减小 batch_size:
```bash
python scripts/run_experiments.py --mode all --batch_size 16
```

### Q: 如何下载结果到本地？
```bash
# 在本地运行
scp -r user@server:/root/autodl-tmp/cardio/dynamic-master/dynamic-master/output ./
```

---

## 时间估算

| 模式 | 实验数 | 总Epochs | 预计时间 |
|------|--------|----------|----------|
| reproduce | 1 | 45 | ~3小时 |
| core | 8 | 120 | ~6小时 |
| temporal | 6 | 90 | ~3小时 |
| full | 15 | 255 | ~12小时 |

*基于 batch_size=32, 32GB VRAM 估算*
