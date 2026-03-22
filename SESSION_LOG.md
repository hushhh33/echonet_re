# EchoNet-Dynamic 项目会话记录

## 日期: 2026-03-21

## 项目背景
- Stanford EchoNet-Dynamic (Nature 2020)
- 基于深度学习的心脏超声视频分析
- 任务: 左心室分割 + 射血分数(EF)预测

## 已完成工作

### 1. 环境配置
- 本地环境: `conda activate torch_env`
- 服务器路径: `/root/autodl-tmp/cardio/dynamic-master/dynamic-master`
- 配置文件: `echonet.cfg` 指向数据目录

### 2. 代码修复
- `hyperparameter_tuning.py`: 修复变量遮蔽bug (subset -> stat_subset)
- `hyperparameter_tuning.py`: 添加 `flush_print()` 解决输出缓冲问题
- `run_experiments.py`: 完整重写，加入断点保护

### 3. 核心文件
| 文件 | 功能 |
|------|------|
| `scripts/run_experiments.py` | 完整实验套件 (15个实验) |
| `scripts/hyperparameter_tuning.py` | 快速调参工具 |
| `scripts/experiment_config.py` | 实验配置定义 |
| `setup_server.sh` | 服务器初始化脚本 |
| `EXPERIMENT_GUIDE.md` | 实验操作指南 |
| `EXPERIMENT_REPORT.md` | 给导师的报告 |

### 4. 实验设计
- **论文复现**: 45 epochs, 目标 R²≈0.81
- **核心消融**: 预训练/架构/学习率对比
- **时序消融**: 帧数/采样间隔对比

### 5. 关键概念理解
- R(2+1)D: 时空分解卷积 (2D空间 + 1D时间)
- frames: 输入视频片段长度
- period: 帧采样间隔
- Kinetics-400: 视频动作识别预训练数据集

## 服务器配置
- 内存: 62GB RAM
- 显存: 32GB VRAM
- 推荐: batch_size=32, num_workers=8

## 运行命令
```bash
# 完整实验
python scripts/run_experiments.py --mode full

# 断点恢复 (默认自动恢复)
python scripts/run_experiments.py --mode full

# TensorBoard
tensorboard --logdir=output/experiments --port 6006
```

## 预期产出
- 论文复现结果验证
- 消融实验对比表格
- TensorBoard可视化图表
- 可用于毕设/报告的完整实验数据
