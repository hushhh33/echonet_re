"""
EchoNet-Dynamic 完整实验套件 (服务器版)
=====================================
功能：
1. 精确复现论文结果
2. 完整Ablation Study
3. 断点保护 (自动保存进度，支持恢复)
4. TensorBoard可视化

服务器配置: 62GB RAM, 32GB VRAM
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 设置缓存目录
os.environ['TORCH_HOME'] = str(Path(__file__).parent.parent / '.cache')

sys.path.insert(0, str(Path(__file__).parent.parent))
import echonet


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# ============================================================
#                    实验配置
# ============================================================

# 精确复现论文
REPRODUCTION = [
    {"name": "00_paper_reproduction", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4, "epochs": 45,
     "description": "精确复现论文结果 (45 epochs)"},
]

# 核心Ablation (必做)
CORE_ABLATION = [
    # 预训练对比
    {"name": "01_pretrained", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4, "epochs": 15,
     "group": "pretrain"},
    {"name": "02_scratch", "model_name": "r2plus1d_18", "pretrained": False,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4, "epochs": 15,
     "group": "pretrain"},

    # 学习率对比
    {"name": "03_lr_1e-3", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-3, "frames": 32, "period": 2, "weight_decay": 1e-4, "epochs": 15,
     "group": "learning_rate"},
    {"name": "04_lr_1e-4", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4, "epochs": 15,
     "group": "learning_rate"},
    {"name": "05_lr_1e-5", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-5, "frames": 32, "period": 2, "weight_decay": 1e-4, "epochs": 15,
     "group": "learning_rate"},

    # 模型架构对比
    {"name": "06_r2plus1d", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4, "epochs": 15,
     "group": "architecture"},
    {"name": "07_r3d", "model_name": "r3d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4, "epochs": 15,
     "group": "architecture"},
    {"name": "08_mc3", "model_name": "mc3_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4, "epochs": 15,
     "group": "architecture"},
]

# 时序参数Ablation
TEMPORAL_ABLATION = [
    # 帧数对比
    {"name": "09_frames_16", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 16, "period": 2, "weight_decay": 1e-4, "epochs": 15,
     "group": "frames"},
    {"name": "10_frames_32", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4, "epochs": 15,
     "group": "frames"},
    {"name": "11_frames_64", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 64, "period": 2, "weight_decay": 1e-4, "epochs": 15,
     "group": "frames"},

    # 采样间隔对比
    {"name": "12_period_1", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 1, "weight_decay": 1e-4, "epochs": 15,
     "group": "period"},
    {"name": "13_period_2", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4, "epochs": 15,
     "group": "period"},
    {"name": "14_period_4", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 4, "weight_decay": 1e-4, "epochs": 15,
     "group": "period"},
]

# 快速验证 (2个实验)
QUICK_TEST = [
    {"name": "quick_pretrained", "model_name": "r2plus1d_18", "pretrained": True,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4, "epochs": 5},
    {"name": "quick_scratch", "model_name": "r2plus1d_18", "pretrained": False,
     "lr": 1e-4, "frames": 32, "period": 2, "weight_decay": 1e-4, "epochs": 5},
]

# 实验组合
EXPERIMENT_SETS = {
    "quick": QUICK_TEST,
    "reproduce": REPRODUCTION,
    "core": CORE_ABLATION,
    "temporal": TEMPORAL_ABLATION,
    "full": REPRODUCTION + CORE_ABLATION + TEMPORAL_ABLATION,
}


# ============================================================
#                    训练器
# ============================================================

class RobustTrainer:
    """带断点保护的训练器"""

    def __init__(self, config, output_dir, data_dir=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir or echonet.config.DATA_DIR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        self.writer.add_text("config", json.dumps(config, indent=2))

        # 断点文件
        self.checkpoint_path = self.output_dir / "checkpoint.pt"
        self.progress_path = self.output_dir / "progress.json"

    def build_model(self):
        model_name = self.config["model_name"]
        pretrained = self.config["pretrained"]

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
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.fc.bias.data[0] = 55.6

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(self.device)

        return model

    def get_dataloaders(self, batch_size=32, num_workers=8):
        # 计算统计信息
        temp_dataset = echonet.datasets.Echo(root=self.data_dir, split="train")
        indices = np.random.choice(len(temp_dataset), min(128, len(temp_dataset)), replace=False).tolist()
        stat_subset = torch.utils.data.Subset(temp_dataset, indices)
        dataloader = torch.utils.data.DataLoader(stat_subset, batch_size=16, num_workers=4)

        n, s1, s2 = 0, 0., 0.
        for (x, *_) in dataloader:
            x = x.transpose(0, 1).contiguous().view(3, -1)
            n += x.shape[1]
            s1 += torch.sum(x, dim=1).numpy()
            s2 += torch.sum(x ** 2, dim=1).numpy()

        mean = (s1 / n).astype(np.float32)
        std = np.sqrt(s2 / n - (s1 / n) ** 2).astype(np.float32)
        del temp_dataset, stat_subset

        kwargs = {
            "target_type": "EF",
            "mean": mean,
            "std": std,
            "length": self.config["frames"],
            "period": self.config["period"],
        }

        train_dataset = echonet.datasets.Echo(root=self.data_dir, split="train", **kwargs, pad=12)
        val_dataset = echonet.datasets.Echo(root=self.data_dir, split="val", **kwargs)
        test_dataset = echonet.datasets.Echo(root=self.data_dir, split="test", **kwargs)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        return train_loader, val_loader, test_loader

    def save_checkpoint(self, model, optimizer, scheduler, epoch, best_val_loss, history):
        """保存断点"""
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": self.config,
        }, self.checkpoint_path)

        with open(self.progress_path, "w") as f:
            json.dump({"epoch": epoch, "history": history}, f)

    def load_checkpoint(self, model, optimizer, scheduler):
        """加载断点"""
        if self.checkpoint_path.exists():
            flush_print(f"  [恢复] 发现断点，正在恢复...")
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            with open(self.progress_path) as f:
                progress = json.load(f)

            flush_print(f"  [恢复] 从 epoch {checkpoint['epoch']+1} 继续")
            return checkpoint["epoch"] + 1, checkpoint["best_val_loss"], progress["history"]

        return 0, float("inf"), []

    def train_epoch(self, model, dataloader, optimizer, epoch, total_epochs):
        model.train()
        total_loss, n = 0, 0
        all_preds, all_targets = [], []

        desc = f"Epoch {epoch+1}/{total_epochs} [Train]"
        pbar = tqdm(dataloader, desc=desc, ncols=120,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

        for X, y in pbar:
            X, y = X.to(self.device), y.to(self.device)

            optimizer.zero_grad()
            out = model(X).view(-1)
            loss = nn.functional.mse_loss(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            n += X.size(0)
            all_preds.extend(out.detach().cpu().numpy())
            all_targets.extend(y.cpu().numpy())

            current_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
            pbar.set_postfix({"loss": f"{total_loss/n:.2f}", "mae": f"{current_mae:.2f}%"})

        return {
            "loss": total_loss / n,
            "r2": sklearn.metrics.r2_score(all_targets, all_preds),
            "mae": sklearn.metrics.mean_absolute_error(all_targets, all_preds)
        }

    @torch.no_grad()
    def evaluate(self, model, dataloader, desc="Eval"):
        model.eval()
        total_loss, n = 0, 0
        all_preds, all_targets = [], []

        for X, y in tqdm(dataloader, desc=desc, ncols=100, leave=False):
            X, y = X.to(self.device), y.to(self.device)
            out = model(X).view(-1)
            loss = nn.functional.mse_loss(out, y)

            total_loss += loss.item() * X.size(0)
            n += X.size(0)
            all_preds.extend(out.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        return {
            "loss": total_loss / n,
            "r2": sklearn.metrics.r2_score(all_targets, all_preds),
            "mae": sklearn.metrics.mean_absolute_error(all_targets, all_preds),
            "rmse": np.sqrt(sklearn.metrics.mean_squared_error(all_targets, all_preds)),
            "preds": all_preds,
            "targets": all_targets
        }

    def train(self, num_epochs=None, batch_size=32, num_workers=8):
        """完整训练 (支持断点恢复)"""
        if num_epochs is None:
            num_epochs = self.config.get("epochs", 15)

        start_time = time.time()

        flush_print(f"\n{'='*70}")
        flush_print(f"  Experiment: {self.config.get('name', 'unnamed')}")
        flush_print(f"  Model: {self.config['model_name']}, Pretrained: {self.config['pretrained']}")
        flush_print(f"  LR: {self.config['lr']}, Frames: {self.config['frames']}, Period: {self.config['period']}")
        flush_print(f"  Epochs: {num_epochs}, Batch: {batch_size}")
        flush_print(f"{'='*70}")

        model = self.build_model()
        train_loader, val_loader, test_loader = self.get_dataloaders(batch_size, num_workers)

        flush_print(f"  Data: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

        optimizer = torch.optim.SGD(model.parameters(), lr=self.config["lr"],
                                     momentum=0.9, weight_decay=self.config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15)

        # 尝试恢复断点
        start_epoch, best_val_loss, history = self.load_checkpoint(model, optimizer, scheduler)
        best_epoch = start_epoch - 1 if start_epoch > 0 else 0

        flush_print(f"{'='*70}\n")

        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()

            train_metrics = self.train_epoch(model, train_loader, optimizer, epoch, num_epochs)
            val_metrics = self.evaluate(model, val_loader, "Val")

            scheduler.step()
            epoch_time = time.time() - epoch_start

            # TensorBoard
            self.writer.add_scalars("loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
            self.writer.add_scalars("r2", {"train": train_metrics["r2"], "val": val_metrics["r2"]}, epoch)
            self.writer.add_scalars("mae", {"train": train_metrics["mae"], "val": val_metrics["mae"]}, epoch)
            self.writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            history.append({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_r2": train_metrics["r2"],
                "val_loss": val_metrics["loss"],
                "val_r2": val_metrics["r2"],
                "val_mae": val_metrics["mae"],
                "time": epoch_time
            })

            # 进度显示
            eta = (num_epochs - epoch - 1) * epoch_time
            is_best = val_metrics["loss"] < best_val_loss
            best_marker = " *BEST*" if is_best else ""

            flush_print(f"  [{epoch+1:2d}/{num_epochs}] "
                       f"loss: {train_metrics['loss']:.2f}/{val_metrics['loss']:.2f} | "
                       f"R2: {val_metrics['r2']:.4f} | "
                       f"MAE: {val_metrics['mae']:.2f}% | "
                       f"{epoch_time:.0f}s | "
                       f"ETA: {str(timedelta(seconds=int(eta)))}{best_marker}")

            # 保存最佳模型
            if is_best:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "config": self.config,
                    "metrics": val_metrics
                }, self.output_dir / "best.pt")

            # 保存断点 (每个epoch)
            self.save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, history)

        # 测试集评估
        flush_print(f"\n  Loading best model from epoch {best_epoch+1}...")
        checkpoint = torch.load(self.output_dir / "best.pt", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

        test_metrics = self.evaluate(model, test_loader, "Test")

        total_time = time.time() - start_time

        flush_print(f"\n{'='*70}")
        flush_print(f"  RESULTS: {self.config.get('name', 'unnamed')}")
        flush_print(f"  Best Epoch: {best_epoch+1}")
        flush_print(f"  Test R2:   {test_metrics['r2']:.4f}")
        flush_print(f"  Test MAE:  {test_metrics['mae']:.2f}%")
        flush_print(f"  Test RMSE: {test_metrics['rmse']:.2f}%")
        flush_print(f"  Time: {str(timedelta(seconds=int(total_time)))}")
        flush_print(f"{'='*70}\n")

        # 保存结果
        results = {
            "config": self.config,
            "best_epoch": best_epoch,
            "test_metrics": {
                "r2": test_metrics["r2"],
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"]
            },
            "history": history,
            "total_time_seconds": total_time
        }

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        pd.DataFrame(history).to_csv(self.output_dir / "history.csv", index=False)
        pd.DataFrame({
            "target": test_metrics["targets"],
            "prediction": test_metrics["preds"]
        }).to_csv(self.output_dir / "test_predictions.csv", index=False)

        # 清理断点文件 (训练完成)
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        if self.progress_path.exists():
            self.progress_path.unlink()

        self.writer.close()
        return results


# ============================================================
#                    实验运行器
# ============================================================

def run_experiments(experiments, output_dir, batch_size=32, num_workers=8, resume=True):
    """运行实验套件 (支持断点恢复)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 进度文件
    progress_file = output_dir / "suite_progress.json"

    # 加载已完成的实验
    completed = set()
    if resume and progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
            completed = set(progress.get("completed", []))
        flush_print(f"[恢复] 已完成 {len(completed)} 个实验，继续执行剩余实验...")

    total_start = time.time()
    all_results = []

    flush_print("\n" + "="*70)
    flush_print("  ECHONET-DYNAMIC EXPERIMENT SUITE")
    flush_print("="*70)
    flush_print(f"  Total: {len(experiments)} experiments")
    flush_print(f"  Completed: {len(completed)}")
    flush_print(f"  Remaining: {len(experiments) - len(completed)}")
    flush_print(f"  Output: {output_dir}")
    flush_print("="*70 + "\n")

    for i, exp in enumerate(experiments):
        exp_name = exp["name"]

        # 跳过已完成的实验
        if exp_name in completed:
            flush_print(f"[{i+1}/{len(experiments)}] {exp_name} - SKIPPED (already completed)")
            # 加载已有结果
            result_file = output_dir / exp_name / "results.json"
            if result_file.exists():
                with open(result_file) as f:
                    all_results.append(json.load(f))
            continue

        flush_print(f"\n{'#'*70}")
        flush_print(f"  EXPERIMENT {i+1}/{len(experiments)}: {exp_name}")
        flush_print(f"{'#'*70}")

        config = {k: v for k, v in exp.items()}
        exp_output = output_dir / exp_name
        num_epochs = exp.get("epochs", 15)

        try:
            trainer = RobustTrainer(config, exp_output)
            results = trainer.train(num_epochs=num_epochs, batch_size=batch_size, num_workers=num_workers)
            results["experiment_name"] = exp_name
            all_results.append(results)

            # 更新进度
            completed.add(exp_name)
            with open(progress_file, "w") as f:
                json.dump({"completed": list(completed)}, f)

        except Exception as e:
            flush_print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"experiment_name": exp_name, "error": str(e)})

    # 汇总
    total_time = time.time() - total_start

    flush_print("\n" + "="*70)
    flush_print("  FINAL SUMMARY")
    flush_print("="*70)

    summary = []
    for r in all_results:
        if "error" not in r and "test_metrics" in r:
            summary.append({
                "Experiment": r.get("experiment_name", r["config"]["name"]),
                "Model": r["config"]["model_name"],
                "Pretrained": "Yes" if r["config"]["pretrained"] else "No",
                "LR": r["config"]["lr"],
                "Frames": r["config"]["frames"],
                "Test_R2": f"{r['test_metrics']['r2']:.4f}",
                "Test_MAE": f"{r['test_metrics']['mae']:.2f}%",
                "Best_Epoch": r["best_epoch"] + 1
            })

    if summary:
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(output_dir / "final_summary.csv", index=False)

        # 找最佳
        best_idx = summary_df["Test_R2"].str.replace("%", "").astype(float).idxmax()
        flush_print(f"\n  BEST: {summary_df.iloc[best_idx]['Experiment']} (R2={summary_df.iloc[best_idx]['Test_R2']})")

    flush_print(f"\n  Total time: {str(timedelta(seconds=int(total_time)))}")
    flush_print(f"  Results: {output_dir}")
    flush_print("="*70 + "\n")

    return all_results


# ============================================================
#                    主程序
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EchoNet-Dynamic Experiment Suite")
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "reproduce", "core", "temporal", "full"],
                        help="实验模式")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Data workers")
    parser.add_argument("--output", type=str, default="output/experiments", help="Output dir")
    parser.add_argument("--no_resume", action="store_true", help="不恢复断点，重新开始")

    args = parser.parse_args()

    experiments = EXPERIMENT_SETS[args.mode]

    flush_print(f"\n实验模式: {args.mode}")
    flush_print(f"实验数量: {len(experiments)}")

    # 估算时间
    total_epochs = sum(exp.get("epochs", 15) for exp in experiments)
    est_time = total_epochs * 2  # 约2分钟每epoch
    flush_print(f"预计时间: ~{est_time}分钟 ({est_time/60:.1f}小时)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"{args.mode}_{timestamp}"

    run_experiments(
        experiments,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resume=not args.no_resume
    )
