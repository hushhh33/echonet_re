# EchoNet-Dynamic 复现指南

## 项目简介

EchoNet-Dynamic 是斯坦福大学开发的深度学习模型，用于：
1. **左心室语义分割** - 逐帧分割心脏超声视频中的左心室
2. **射血分数(EF)预测** - 从视频片段预测心脏射血分数
3. **心肌病评估** - 基于逐搏预测评估心功能

论文：[Video-based AI for beat-to-beat assessment of cardiac function](https://www.nature.com/articles/s41586-020-2145-8) (Nature, 2020)

---

## 环境配置

### 已完成
- [x] Python 环境 (base)
- [x] 依赖包安装
- [x] 数据集下载 (`F:/cardio/EchoNet-Dynamic/`)

### 数据集结构
```
F:/cardio/EchoNet-Dynamic/
├── FileList.csv          # 视频文件列表和标签 (EF值等)
├── VolumeTracings.csv    # 左心室轮廓标注
└── Videos/               # 10,030个超声心动图视频 (.avi)
```

### 配置文件
`echonet.cfg` 已配置数据路径：
```
data_dir = F:/cardio/EchoNet-Dynamic
```

---

## 复现步骤

### 步骤1：左心室语义分割

**命令：**
```bash
echonet segmentation --save_video
```

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | deeplabv3_resnet50 | 分割模型架构 |
| `--pretrained` | True | 使用ImageNet预训练权重 |
| `--batch_size` | 20 | 批次大小 |
| `--num_epochs` | 50 | 训练轮数 |
| `--save_video` | False | 是否保存分割视频 |

**输出目录：** `output/segmentation/deeplabv3_resnet50_random/`
- `log.csv` - 训练/验证损失记录
- `best.pt` - 最佳模型权重
- `size.csv` - 每帧左心室大小估计
- `videos/` - 分割可视化视频

**预计时间：** ~4-6小时 (GPU)

---

### 步骤2：射血分数预测

**命令：**
```bash
echonet video
```

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | r2plus1d_18 | 视频模型 (R(2+1)D) |
| `--frames` | 32 | 输入帧数 |
| `--period` | 2 | 采样间隔 |
| `--pretrained` | True | 使用Kinetics预训练权重 |
| `--batch_size` | 20 | 批次大小 |
| `--num_epochs` | 45 | 训练轮数 |

**输出目录：** `output/video/r2plus1d_18_32_2_pretrained/`
- `log.csv` - 训练/验证损失记录
- `best.pt` - 最佳模型权重
- `test_predictions.csv` - 测试集EF预测结果

**预计时间：** ~6-8小时 (GPU)

---

### 步骤3：模型评估

训练完成后，检查以下指标：

**分割任务：**
- Dice系数
- IoU (交并比)

**EF预测任务：**
- MAE (平均绝对误差) - 论文结果: 4.1%
- R² (决定系数) - 论文结果: 0.81

---

## 快速测试 (可选)

如果想快速验证环境，可以用较少数据测试：

```bash
# 分割 - 减少epoch
echonet segmentation --num_epochs 1 --num_workers 0

# 视频 - 减少epoch
echonet video --num_epochs 1 --num_workers 0
```

---

## 常见问题

### Q1: CUDA内存不足
减小 `--batch_size`，如 `--batch_size 8`

### Q2: Windows多进程问题
添加 `--num_workers 0`

### Q3: 训练中断后继续
模型会自动保存checkpoint，重新运行命令会从头开始（暂不支持断点续训）

---

## 输出文件说明

### size.csv (分割输出)
| 列名 | 说明 |
|------|------|
| FileName | 视频文件名 |
| Frame | 帧编号 |
| Size | 左心室面积 (像素) |
| ComputerSmall | 模型预测的收缩末期帧 |

### test_predictions.csv (视频输出)
| 列名 | 说明 |
|------|------|
| FileName | 视频文件名 |
| EF | 真实射血分数 |
| Prediction | 模型预测值 |

---

## 参考资料

- 官方仓库: https://github.com/echonet/dynamic
- 数据集: https://echonet.github.io/dynamic/
- 论文: https://doi.org/10.1038/s41586-020-2145-8
我正在通过学习echonet来深度了解深度学习项目，让我们深入代码，我希望学会一些代码上的技巧和熟练度，你要教会我这个代码的构思和逻辑。从数据处理开始，先整理学习框架，再深入代码细节。先给一个大纲然后深入。