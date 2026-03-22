"""
EchoNet-Dynamic 超参数调参脚本
==============================
功能：
1. 多组超参数组合自动实验
2. TensorBoard可视化
3. 预训练vs从头训练对比
4. 结果自动汇总

使用方法：
    conda activate torch_env
    cd F:/cardio/dynamic-master/dynamic-master
    python scripts/hyperparameter_tuning.py --mode quick --epochs 3

TensorBoard查看：
    tensorboard --logdir=output/tuning_runs
"""

import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 设置缓存目录到项目内
os.environ['TORCH_HOME'] = str(Path(__file__).parent.parent / '.cache')

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
import echonet


def flush_print(*args, **kwargs):
    """立即输出，不缓冲"""
    print(*args, **kwargs)
    sys.stdout.flush()


# ============== 推荐实验组合 ==============
RECOMMENDED_EXPERIMENTS = [
    # 实验1: 基线 (论文配置) - 预训练
    {"model_name": "r2plus1d_18", "pretrained": True, "lr": 1e-4,
     "frames": 32, "period": 2, "weight_decay": 1e-4, "name": "baseline_pretrained"},

    # 实验2: 从头训练对比
    {"model_name": "r2plus1d_18", "pretrained": False, "lr": 1e-4,
     "frames": 32, "period": 2, "weight_decay": 1e-4, "name": "baseline_scratch"},
]

# ============== 快速实验 (用于验证) ==============
QUICK_EXPERIMENTS = [
    {"model_name": "r2plus1d_18", "pretrained": True, "lr": 1e-4,
     "frames": 32, "period": 2, "weight_decay": 1e-4, "name": "pretrained"},
    {"model_name": "r2plus1d_18", "pretrained": False, "lr": 1e-4,
     "frames": 32, "period": 2, "weight_decay": 1e-4, "name": "scratch"},
]


class TensorBoardTrainer:
    """带TensorBoard的训练器"""

    def __init__(self, config, output_dir, data_dir=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = data_dir or echonet.config.DATA_DIR

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        flush_print(f"[设备] {self.device}")

        # 记录配置
        self.writer.add_text("config", json.dumps(config, indent=2))

    def build_model(self):
        """构建模型"""
        model_name = self.config["model_name"]
        pretrained = self.config["pretrained"]

        flush_print(f"[模型] {model_name}, 预训练={pretrained}")

        # 获取预训练权重
        if pretrained:
            weights_map = {
                "r2plus1d_18": torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1,
                "r3d_18": torchvision.models.video.R3D_18_Weights.KINETICS400_V1,
                "mc3_18": torchvision.models.video.MC3_18_Weights.KINETICS400_V1,
            }
            weights = weights_map.get(model_name)
        else:
            weights = None

        model = torchvision.models.video.__dict__[model_name](weights=weights)

        # 修改输出层
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.fc.bias.data[0] = 55.6  # EF平均值初始化

        if self.device.type == "cuda":
            model = nn.DataParallel(model)
        model.to(self.device)

        param_count = sum(p.numel() for p in model.parameters())
        flush_print(f"[参数量] {param_count:,}")

        return model

    def get_dataloaders(self, batch_size=4, num_workers=0, subset=None):
        """获取数据加载器"""
        flush_print(f"[数据] 加载中...")

        # 计算数据集均值和标准差 (使用少量样本加速)
        temp_dataset = echonet.datasets.Echo(root=self.data_dir, split="train")

        # 使用64个样本计算统计信息
        indices = np.random.choice(len(temp_dataset), min(64, len(temp_dataset)), replace=False).tolist()
        stat_subset = torch.utils.data.Subset(temp_dataset, indices)
        dataloader = torch.utils.data.DataLoader(stat_subset, batch_size=batch_size, num_workers=0)

        n, s1, s2 = 0, 0., 0.
        for (x, *_) in tqdm(dataloader, desc="计算统计信息"):
            x = x.transpose(0, 1).contiguous().view(3, -1)
            n += x.shape[1]
            s1 += torch.sum(x, dim=1).numpy()
            s2 += torch.sum(x ** 2, dim=1).numpy()

        mean = (s1 / n).astype(np.float32)
        std = np.sqrt(s2 / n - (s1 / n) ** 2).astype(np.float32)

        del temp_dataset, stat_subset  # 释放内存

        kwargs = {
            "target_type": "EF",
            "mean": mean,
            "std": std,
            "length": self.config["frames"],
            "period": self.config["period"],
        }

        train_dataset = echonet.datasets.Echo(
            root=self.data_dir, split="train", **kwargs, pad=12
        )
        val_dataset = echonet.datasets.Echo(
            root=self.data_dir, split="val", **kwargs
        )

        # 使用子集加速训练
        if subset and subset < len(train_dataset):
            indices = np.random.choice(len(train_dataset), subset, replace=False).tolist()
            train_dataset = torch.utils.data.Subset(train_dataset, indices)

        flush_print(f"[数据] 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False
        )

        return train_loader, val_loader

    def train_epoch(self, model, dataloader, optimizer, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        n = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [训练]")
        for X, y in pbar:
            X = X.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()
            outputs = model(X).view(-1)
            loss = nn.functional.mse_loss(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            n += X.size(0)

            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(y.cpu().numpy())

            pbar.set_postfix({"loss": f"{total_loss/n:.4f}"})

        avg_loss = total_loss / n
        r2 = sklearn.metrics.r2_score(all_targets, all_preds)
        mae = sklearn.metrics.mean_absolute_error(all_targets, all_preds)

        return {"loss": avg_loss, "r2": r2, "mae": mae}

    @torch.no_grad()
    def validate(self, model, dataloader, epoch):
        """验证"""
        model.eval()
        total_loss = 0
        n = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [验证]")
        for X, y in pbar:
            X = X.to(self.device)
            y = y.to(self.device)

            outputs = model(X).view(-1)
            loss = nn.functional.mse_loss(outputs, y)

            total_loss += loss.item() * X.size(0)
            n += X.size(0)

            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        avg_loss = total_loss / n
        r2 = sklearn.metrics.r2_score(all_targets, all_preds)
        mae = sklearn.metrics.mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(all_targets, all_preds))

        return {"loss": avg_loss, "r2": r2, "mae": mae, "rmse": rmse}

    def train(self, num_epochs=5, batch_size=4, num_workers=0, subset=None):
        """完整训练流程"""
        flush_print(f"\n{'='*60}")
        flush_print(f"实验: {self.config.get('name', 'unnamed')}")
        flush_print(f"配置: {self.config}")
        if subset:
            flush_print(f"使用子集: {subset} 样本")
        flush_print(f"{'='*60}\n")

        # 构建模型和数据
        model = self.build_model()
        train_loader, val_loader = self.get_dataloaders(
            batch_size=batch_size, num_workers=num_workers, subset=subset
        )

        # 优化器和调度器
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config["lr"],
            momentum=0.9,
            weight_decay=self.config["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15)

        # 训练记录
        best_val_loss = float("inf")
        best_epoch = 0
        history = []

        for epoch in range(num_epochs):
            start_time = time.time()

            # 训练
            train_metrics = self.train_epoch(model, train_loader, optimizer, epoch)

            # 验证
            val_metrics = self.validate(model, val_loader, epoch)

            scheduler.step()

            # 记录到TensorBoard
            self.writer.add_scalars("loss", {
                "train": train_metrics["loss"],
                "val": val_metrics["loss"]
            }, epoch)

            self.writer.add_scalars("r2", {
                "train": train_metrics["r2"],
                "val": val_metrics["r2"]
            }, epoch)

            self.writer.add_scalar("mae/val", val_metrics["mae"], epoch)
            self.writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            # 记录历史
            epoch_time = time.time() - start_time
            history.append({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_r2": train_metrics["r2"],
                "val_loss": val_metrics["loss"],
                "val_r2": val_metrics["r2"],
                "val_mae": val_metrics["mae"],
                "time": epoch_time
            })

            # 打印结果
            flush_print(f"\nEpoch {epoch}: "
                       f"train_loss={train_metrics['loss']:.4f}, "
                       f"val_loss={val_metrics['loss']:.4f}, "
                       f"val_r2={val_metrics['r2']:.4f}, "
                       f"val_mae={val_metrics['mae']:.2f}%, "
                       f"time={epoch_time:.1f}s")

            # 保存最佳模型
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "config": self.config,
                    "val_metrics": val_metrics,
                }, self.output_dir / "best.pt")
                flush_print(f"   ★ 新的最佳模型!")

        # 保存结果
        results = {
            "config": self.config,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_metrics": {
                "val_r2": history[-1]["val_r2"],
                "val_mae": history[-1]["val_mae"],
            },
            "history": history
        }

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        # 保存历史为CSV (方便查看)
        pd.DataFrame(history).to_csv(self.output_dir / "history.csv", index=False)

        self.writer.close()

        flush_print(f"\n{'='*60}")
        flush_print(f"训练完成!")
        flush_print(f"最佳Epoch: {best_epoch}, 最佳验证Loss: {best_val_loss:.4f}")
        flush_print(f"最终 R²: {history[-1]['val_r2']:.4f}, MAE: {history[-1]['val_mae']:.2f}%")
        flush_print(f"结果保存至: {self.output_dir}")
        flush_print(f"{'='*60}\n")

        return results


def run_experiment_suite(experiments, base_output_dir, num_epochs=5,
                         batch_size=4, num_workers=0, subset=None):
    """运行一系列实验"""
    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for i, exp in enumerate(experiments):
        exp_name = exp.get("name", f"exp_{i:03d}")
        flush_print(f"\n{'#'*60}")
        flush_print(f"实验 {i+1}/{len(experiments)}: {exp_name}")
        flush_print(f"{'#'*60}")

        config = {k: v for k, v in exp.items() if k != "name"}
        config["name"] = exp_name
        output_dir = base_output_dir / exp_name

        try:
            trainer = TensorBoardTrainer(config, output_dir)
            results = trainer.train(
                num_epochs=num_epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                subset=subset
            )
            results["experiment_name"] = exp_name
            all_results.append(results)
        except Exception as e:
            flush_print(f"实验 {exp_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "experiment_name": exp_name,
                "config": config,
                "error": str(e)
            })

    # 汇总结果
    flush_print("\n" + "="*80)
    flush_print("实验汇总")
    flush_print("="*80)

    summary = []
    for r in all_results:
        if "error" not in r:
            summary.append({
                "实验": r["experiment_name"],
                "预训练": "是" if r["config"]["pretrained"] else "否",
                "最佳Epoch": r["best_epoch"],
                "R²": f"{r['final_metrics']['val_r2']:.4f}",
                "MAE": f"{r['final_metrics']['val_mae']:.2f}%",
            })

    if summary:
        summary_df = pd.DataFrame(summary)
        flush_print(summary_df.to_string(index=False))
        summary_df.to_csv(base_output_dir / "experiment_summary.csv", index=False)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EchoNet-Dynamic 超参数调优")
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "recommended", "single"],
                        help="实验模式: quick(快速对比), recommended(推荐组合), single(单个实验)")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--subset", type=int, default=None,
                        help="使用训练集子集(样本数), 如 --subset 500")
    parser.add_argument("--output_dir", type=str, default="output/tuning_runs",
                        help="输出目录")

    # 单个实验参数
    parser.add_argument("--model_name", type=str, default="r2plus1d_18")
    parser.add_argument("--pretrained", action="store_true", help="使用预训练")
    parser.add_argument("--scratch", action="store_true", help="从头训练")
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp

    if args.mode == "quick":
        flush_print("模式: 快速对比实验 (预训练 vs 从头训练)")
        experiments = QUICK_EXPERIMENTS
    elif args.mode == "recommended":
        flush_print("模式: 推荐实验组合")
        experiments = RECOMMENDED_EXPERIMENTS
    else:  # single
        pretrained = args.pretrained and not args.scratch
        experiments = [{
            "name": f"{args.model_name}_{'pt' if pretrained else 'scratch'}",
            "model_name": args.model_name,
            "pretrained": pretrained,
            "lr": args.lr,
            "frames": 32,
            "period": 2,
            "weight_decay": 1e-4,
        }]

    flush_print(f"实验数量: {len(experiments)}")
    flush_print(f"每个实验 Epochs: {args.epochs}")
    flush_print(f"Batch Size: {args.batch_size}")
    if args.subset:
        flush_print(f"训练子集: {args.subset} 样本")
    flush_print(f"输出目录: {output_dir}")

    run_experiment_suite(
        experiments,
        base_output_dir=output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=0,  # Windows兼容
        subset=args.subset
    )
