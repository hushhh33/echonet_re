"""
EchoNet-Dynamic 完整实验方案 (学术级)
====================================

实验分为三部分：
1. 精确复现 - 复现论文结果
2. 核心Ablation - 验证关键组件
3. 扩展探索 - 额外发现

预计时间：
- 精确复现: ~3小时 (45 epochs)
- 核心Ablation: ~4小时 (8个实验 x 15 epochs)
- 扩展探索: ~2小时 (4个实验 x 15 epochs)
总计: ~9小时
"""

# ============== 第一部分：精确复现论文 ==============
REPRODUCTION_EXPERIMENT = [
    {
        "name": "paper_reproduction",
        "model_name": "r2plus1d_18",
        "pretrained": True,
        "lr": 1e-4,
        "frames": 32,
        "period": 2,
        "weight_decay": 1e-4,
        "epochs": 45,  # 论文设置
        "description": "精确复现论文结果，目标 R²≈0.81, MAE≈4.1%"
    }
]

# ============== 第二部分：核心Ablation Study ==============
ABLATION_EXPERIMENTS = [
    # --- Ablation 1: 预训练的影响 ---
    {"name": "ablation_pretrain_yes", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4,
     "group": "pretrain", "description": "预训练 (Kinetics-400)"},

    {"name": "ablation_pretrain_no", "model_name": "r2plus1d_18", "pretrained": False,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4,
     "group": "pretrain", "description": "随机初始化"},

    # --- Ablation 2: 模型架构 ---
    {"name": "ablation_arch_r2plus1d", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4,
     "group": "architecture", "description": "R(2+1)D: 时空分解"},

    {"name": "ablation_arch_r3d", "model_name": "r3d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4,
     "group": "architecture", "description": "R3D: 完整3D卷积"},

    {"name": "ablation_arch_mc3", "model_name": "mc3_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4,
     "group": "architecture", "description": "MC3: 混合卷积"},

    # --- Ablation 3: 学习率 ---
    {"name": "ablation_lr_1e-3", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-3, "frames": 32, "period": 2, "weight_decay": 1e-4,
     "group": "learning_rate", "description": "学习率 1e-3"},

    {"name": "ablation_lr_1e-4", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4,
     "group": "learning_rate", "description": "学习率 1e-4 (论文)"},

    {"name": "ablation_lr_1e-5", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-5, "frames": 32, "period": 2, "weight_decay": 1e-4,
     "group": "learning_rate", "description": "学习率 1e-5"},

    # --- Ablation 4: 时序长度 (frames × period) ---
    {"name": "ablation_frames_16", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 16, "period": 2, "weight_decay": 1e-4,
     "group": "temporal", "description": "16帧 (覆盖32帧≈1秒)"},

    {"name": "ablation_frames_32", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4,
     "group": "temporal", "description": "32帧 (覆盖64帧≈2秒) 论文"},

    {"name": "ablation_frames_64", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 64, "period": 2, "weight_decay": 1e-4,
     "group": "temporal", "description": "64帧 (覆盖128帧≈4秒)"},

    # --- Ablation 5: 采样间隔 ---
    {"name": "ablation_period_1", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 1, "weight_decay": 1e-4,
     "group": "sampling", "description": "每帧采样 (高时间分辨率)"},

    {"name": "ablation_period_2", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4,
     "group": "sampling", "description": "隔帧采样 (论文)"},

    {"name": "ablation_period_4", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 4, "weight_decay": 1e-4,
     "group": "sampling", "description": "每4帧采样 (长时间跨度)"},
]

# ============== 第三部分：扩展探索 (可选) ==============
EXTENSION_EXPERIMENTS = [
    # 数据量影响
    {"name": "ext_data_25pct", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4,
     "subset": 1865, "description": "25%训练数据"},

    {"name": "ext_data_50pct", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4,
     "subset": 3730, "description": "50%训练数据"},

    # 正则化
    {"name": "ext_wd_1e-3", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-3,
     "description": "更强正则化"},

    {"name": "ext_wd_1e-5", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-5,
     "description": "更弱正则化"},
]


# ============== 汇报用：核心实验组合 ==============
# 如果时间有限，只跑这些
CORE_EXPERIMENTS = [
    # 1. 复现
    {"name": "01_reproduction", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4},

    # 2. 预训练对比
    {"name": "02_no_pretrain", "model_name": "r2plus1d_18", "pretrained": False,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4},

    # 3. 最佳学习率验证
    {"name": "03_lr_1e-3", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-3, "frames": 32, "period": 2, "weight_decay": 1e-4},

    # 4. 架构对比 (选一个)
    {"name": "04_r3d", "model_name": "r3d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4},
]

# 时间估算
TIME_ESTIMATES = """
服务器配置: 32GB VRAM, batch_size=32

实验时间估算:
├── 精确复现 (45 epochs): ~3小时
├── 核心Ablation (14个 × 15 epochs): ~7小时
├── 扩展探索 (4个 × 15 epochs): ~2小时
└── 总计: ~12小时

推荐策略:
1. 首先运行 CORE_EXPERIMENTS (4个, ~2小时) 验证环境
2. 然后运行精确复现 (1个, ~3小时) 获得论文级结果
3. 最后运行完整Ablation (可选, ~7小时) 深入分析
"""

if __name__ == "__main__":
    print(TIME_ESTIMATES)
    print("\n可用实验组:")
    print(f"  REPRODUCTION_EXPERIMENT: {len(REPRODUCTION_EXPERIMENT)}个")
    print(f"  ABLATION_EXPERIMENTS: {len(ABLATION_EXPERIMENTS)}个")
    print(f"  EXTENSION_EXPERIMENTS: {len(EXTENSION_EXPERIMENTS)}个")
    print(f"  CORE_EXPERIMENTS: {len(CORE_EXPERIMENTS)}个 (推荐先跑)")
