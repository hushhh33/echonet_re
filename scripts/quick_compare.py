"""
快速测试脚本 - 预训练 vs 从头训练对比
=====================================
特点：
1. 数据加载时显示进度条
2. 内存优化（较小batch_size）
3. 详细的训练进度显示
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import sklearn.metrics
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
import echonet


def get_mean_and_std_with_progress(dataset, samples=128, batch_size=8):
    """计算数据集均值和标准差（带进度条）"""
    print("\n[1/4] 计算数据集统计信息...")

    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False).tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=True
    )

    n = 0
    s1 = 0.
    s2 = 0.

    for (x, *_) in tqdm(dataloader, desc="计算mean/std", unit="batch"):
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x ** 2, dim=1).numpy()

    mean = (s1 / n).astype(np.float32)
    std = np.sqrt(s2 / n - (s1 / n) ** 2).astype(np.float32)

    print(f"   Mean: {mean}, Std: {std}")
    return mean, std


class ProgressDataset(torch.utils.data.Dataset):
    """带进度显示的数据集包装器"""

    def __init__(self, dataset, desc="加载数据"):
        self.dataset = dataset
        self.desc = desc
        self._pbar = None
        self._count = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def run_quick_experiment(pretrained=True, num_epochs=3, batch_size=4):
    """运行快速实验"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print(f"实验配置:")
    print(f"  - 设备: {device}")
    print(f"  - 预训练: {'是' if pretrained else '否 (从头训练)'}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch Size: {batch_size}")
    print("="*60)

    # 模型
    print("\n[0/4] 加载模型...")
    if pretrained:
        weights = torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1
        print("   使用 Kinetics-400 预训练权重")
    else:
        weights = None
        print("   随机初始化权重")

    model = torchvision.models.video.r2plus1d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6

    if device.type == "cuda":
        model = nn.DataParallel(model)
    model.to(device)
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 数据
    data_dir = echonet.config.DATA_DIR
    print(f"\n   数据目录: {data_dir}")

    # 先创建一个小数据集用于计算统计信息
    temp_dataset = echonet.datasets.Echo(root=data_dir, split="train")
    print(f"   训练集大小: {len(temp_dataset)} 个视频")

    mean, std = get_mean_and_std_with_progress(temp_dataset, samples=64, batch_size=batch_size)
    del temp_dataset  # 释放内存

    kwargs = {"target_type": "EF", "mean": mean, "std": std, "length": 32, "period": 2}

    print("\n[2/4] 准备数据集...")
    train_dataset = echonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12)
    val_dataset = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   验证集: {len(val_dataset)} 样本")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15)

    # 训练
    print("\n[3/4] 开始训练...")
    history = []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        # Train
        model.train()
        train_loss = 0
        train_n = 0

        pbar = tqdm(train_loader, desc=f"训练", unit="batch",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X).view(-1)
            loss = nn.functional.mse_loss(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            train_n += X.size(0)
            pbar.set_postfix({"loss": f"{train_loss/train_n:.4f}"})

        # Validate
        model.eval()
        val_loss = 0
        val_n = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"验证", unit="batch",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            for X, y in pbar:
                X, y = X.to(device), y.to(device)
                out = model(X).view(-1)
                loss = nn.functional.mse_loss(out, y)
                val_loss += loss.item() * X.size(0)
                val_n += X.size(0)
                all_preds.extend(out.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        val_r2 = sklearn.metrics.r2_score(all_targets, all_preds)
        val_mae = sklearn.metrics.mean_absolute_error(all_targets, all_preds)

        scheduler.step()

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss / train_n,
            "val_loss": val_loss / val_n,
            "val_r2": val_r2,
            "val_mae": val_mae
        }
        history.append(epoch_result)

        print(f"\n结果: train_loss={train_loss/train_n:.4f}, "
              f"val_loss={val_loss/val_n:.4f}, R²={val_r2:.4f}, MAE={val_mae:.2f}%")

        if val_loss / val_n < best_val_loss:
            best_val_loss = val_loss / val_n
            print("   ★ 新的最佳模型!")

    # 保存结果
    print("\n[4/4] 保存结果...")
    output_dir = Path("output/quick_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / f"{'pretrained' if pretrained else 'scratch'}_{num_epochs}ep.json"
    with open(result_file, "w") as f:
        json.dump({
            "config": {
                "pretrained": pretrained,
                "num_epochs": num_epochs,
                "batch_size": batch_size
            },
            "history": history,
            "best_val_loss": best_val_loss
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"  - 结果保存至: {result_file}")
    print(f"  - 最佳验证Loss: {best_val_loss:.4f}")
    print(f"  - 最终R²: {history[-1]['val_r2']:.4f}")
    print(f"  - 最终MAE: {history[-1]['val_mae']:.2f}%")
    print(f"{'='*60}")

    return history, best_val_loss


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EchoNet快速对比实验")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="使用预训练权重")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数 (默认: 3)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小 (默认: 4，减少内存占用)")
    args = parser.parse_args()

    history, best_loss = run_quick_experiment(
        pretrained=args.pretrained,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
