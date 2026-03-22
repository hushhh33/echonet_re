# EchoNet-Dynamic 复现与调参实验报告

## 1. 项目背景

### 1.1 论文信息
- **标题**: Video-based AI for beat-to-beat assessment of cardiac function
- **发表**: Nature, March 2020
- **机构**: Stanford University
- **引用**: Ouyang et al., Nature 2020, DOI: 10.1038/s41586-020-2145-8

### 1.2 核心任务
| 任务 | 模型 | 输出 |
|------|------|------|
| 左心室语义分割 | DeepLabV3 + ResNet50 | 逐帧分割掩码 |
| 射血分数(EF)预测 | R(2+1)D-18 | EF百分比 |
| 心肌病评估 | 基于上述两者 | 逐搏心功能评估 |

### 1.3 数据集
- **名称**: EchoNet-Dynamic Dataset
- **规模**: 10,030 个心脏超声视频
- **视图**: 心尖四腔心切面 (A4C)
- **标注**: 射血分数值 + 左心室轮廓

---

## 2. 复现结果

### 2.1 环境配置
```
Conda环境: torch_env
PyTorch: 2.5.1+cu121
CUDA: 12.1
```

### 2.2 分割任务结果 (已完成)
| 指标 | 训练集 | 验证集 | 论文结果 |
|------|--------|--------|----------|
| Dice | 0.971 | 0.921 | ~0.92 |
| 最佳Epoch | - | 13 | - |
| 训练时间 | ~6小时 | - | - |

**观察**: 训练后期出现轻微过拟合，验证loss在epoch 13后开始上升。

---

## 3. 深度学习核心知识点

### 3.1 模型架构

#### R(2+1)D (视频理解)
```
3D卷积 → 2D空间卷积 + 1D时序卷积 (分解)

优势:
- 参数量减少
- 非线性增加
- 时空特征分离学习
```

#### DeepLabV3 (语义分割)
```
ResNet Backbone → ASPP (空洞空间金字塔) → 输出

关键技术:
- 空洞卷积 (Dilated Convolution)
- 多尺度特征融合
```

### 3.2 训练技巧

| 技巧 | 说明 | 代码位置 |
|------|------|----------|
| 预训练初始化 | Kinetics-400 / ImageNet | `pretrained=True` |
| SGD + Momentum | 动量优化器 | `momentum=0.9` |
| Weight Decay | L2正则化 | `weight_decay=1e-4` |
| StepLR | 学习率阶梯衰减 | `step_size=15` |
| 数据标准化 | 计算训练集mean/std | `get_mean_and_std()` |
| 偏置初始化 | EF均值初始化输出偏置 | `bias.data[0] = 55.6` |

### 3.3 评估指标
- **分割**: Dice系数、IoU
- **回归**: MAE (平均绝对误差)、R² (决定系数)、RMSE

---

## 4. 超参数调优实验设计

### 4.1 实验组合

| 实验名称 | 模型 | 预训练 | LR | Frames | Period |
|----------|------|--------|-----|--------|--------|
| baseline_pretrained | r2plus1d_18 | Yes | 1e-4 | 32 | 2 |
| baseline_scratch | r2plus1d_18 | No | 1e-4 | 32 | 2 |
| r3d_pretrained | r3d_18 | Yes | 1e-4 | 32 | 2 |
| mc3_pretrained | mc3_18 | Yes | 1e-4 | 32 | 2 |
| lr_1e-3 | r2plus1d_18 | Yes | 1e-3 | 32 | 2 |
| lr_1e-5 | r2plus1d_18 | Yes | 1e-5 | 32 | 2 |
| frames_16 | r2plus1d_18 | Yes | 1e-4 | 16 | 2 |
| frames_64 | r2plus1d_18 | Yes | 1e-4 | 64 | 2 |
| period_1 | r2plus1d_18 | Yes | 1e-4 | 32 | 1 |
| period_4 | r2plus1d_18 | Yes | 1e-4 | 32 | 4 |

### 4.2 运行命令

```bash
# 激活环境
conda activate torch_env
cd F:/cardio/dynamic-master/dynamic-master

# 快速验证 (5 epochs)
python scripts/hyperparameter_tuning.py --mode quick --num_epochs 5

# 推荐实验组合
python scripts/hyperparameter_tuning.py --mode recommended --num_epochs 45

# 查看TensorBoard
tensorboard --logdir=output/tuning_runs

# 分析结果
python scripts/analyze_results.py --tuning_dir output/tuning_runs
```

### 4.3 预期对比结论

| 对比维度 | 预期结论 |
|----------|----------|
| 预训练 vs 从头训练 | 预训练收敛更快，最终性能更好 |
| 学习率 | 1e-4 为最优，过大不稳定，过小收敛慢 |
| 帧数 | 更多帧数捕获更完整心动周期，但计算成本增加 |
| 采样间隔 | 影响时间分辨率，需权衡 |

---

## 5. 近五年领域发展 (2020-2025)

### 5.1 模型架构演进

| 时期 | 架构 | 代表工作 |
|------|------|----------|
| 2020 | 3D CNN | EchoNet-Dynamic (R(2+1)D) |
| 2021-22 | Video Transformer | UVT, UltraSwin |
| 2023-24 | Vision Transformer | ViViEchoformer, Video Swin Transformer |
| 2024-25 | Foundation Model | EchoFM, Echo-VisionFM |

### 5.2 关键技术突破

#### (1) Video Transformer
- **ViViEchoformer**: 直接用Video Vision Transformer回归EF
- **Video Swin Transformer (V-SwinT)**: 3D shifted windows，时序一致性更好
- **UltraSwin**: 层次化Vision Transformer用于EF估计

#### (2) 自监督学习
- **EchoCLR** (2024, Yale): 对比学习 + 帧重排序预训练
  - 仅需<1000样本即可达到良好性能
  - 比传统初始化提升5-40%
  - GitHub: CarDS-Yale/EchoCLR

- **EchoFM**: 290,000+视频预训练的基础模型
  - 时空一致性掩码策略
  - 周期驱动对比学习

#### (3) 多任务学习
- JAMA 2025: 完整AI心超解读系统
- 多任务深度学习同时完成多个诊断任务

### 5.3 可借鉴的改进方向

| 改进方向 | 具体方法 | 预期收益 |
|----------|----------|----------|
| **模型升级** | Video Swin Transformer | 更好的时序建模 |
| **自监督预训练** | EchoCLR方法 | 标签效率提升 |
| **数据增强** | 时间维度增强、MixUp | 泛化能力提升 |
| **学习策略** | 余弦退火、Warmup | 训练稳定性 |
| **多任务** | 分割+EF联合学习 | 特征共享增益 |

---

## 6. 改进实验建议

### 6.1 短期可实现 (1-2周)

```python
# 1. 余弦退火学习率
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)

# 2. 数据增强
transforms = [
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=10),
    ColorJitter(brightness=0.1, contrast=0.1),
    # 时间维度: 随机帧采样
]

# 3. MixUp增强
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam
```

### 6.2 中期目标 (1-2月)

1. **替换骨干网络**: R(2+1)D → Video Swin Transformer
2. **自监督预训练**: 实现简化版EchoCLR
3. **多任务学习**: 分割+EF预测联合训练

### 6.3 参考代码库

| 项目 | 地址 | 用途 |
|------|------|------|
| EchoCLR | github.com/CarDS-Yale/EchoCLR | 自监督学习 |
| Video Swin | github.com/SwinTransformer/Video-Swin-Transformer | Transformer骨干 |
| timm | github.com/huggingface/pytorch-image-models | 预训练模型 |

---

## 7. 总结与展望

### 7.1 本次复现收获

1. **框架理解**: PyTorch训练流程、DataLoader、模型定义
2. **调参经验**: 学习率、batch size、正则化的影响
3. **工程实践**: TensorBoard可视化、checkpoint保存、实验管理

### 7.2 关键学习点

- 预训练模型在医学影像中的迁移学习价值
- 视频理解模型的时空分解思想
- 超参数调优的系统化方法

### 7.3 后续计划

1. 完成调参实验，对比不同配置
2. 尝试Video Transformer架构
3. 探索自监督预训练方法

---

## 参考文献

1. Ouyang et al. "Video-based AI for beat-to-beat assessment of cardiac function" Nature 2020
2. Holste et al. "EchoCLR: Contrastive self-supervised learning for echocardiography" Communications Medicine 2024
3. Reynaud et al. "Ultrasound Video Transformers for Cardiac Ejection Fraction Estimation" MICCAI 2021
4. "Artificial intelligence-enhanced echocardiography in cardiovascular disease management" Nature Reviews Cardiology 2025

---

*报告生成时间: 2026-03-21*
