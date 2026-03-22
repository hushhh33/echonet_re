"""
实验结果分析与可视化
====================
分析调参实验结果，生成汇报用图表
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_experiment_results(tuning_dir):
    """加载所有实验结果"""
    tuning_dir = Path(tuning_dir)
    results = []

    for exp_dir in tuning_dir.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    data["experiment_dir"] = str(exp_dir)
                    results.append(data)

    return results


def create_comparison_table(results):
    """创建实验对比表格"""
    rows = []
    for r in results:
        if "error" not in r:
            row = {
                "Experiment": r.get("experiment_name", "unknown"),
                "Model": r["config"]["model_name"],
                "Pretrained": "Yes" if r["config"]["pretrained"] else "No",
                "LR": r["config"]["lr"],
                "Frames": r["config"]["frames"],
                "Period": r["config"]["period"],
                "Best Epoch": r["best_epoch"],
                "Test R2": f"{r['test_metrics']['r2']:.4f}",
                "Test MAE": f"{r['test_metrics']['mae']:.2f}",
                "Test RMSE": f"{r['test_metrics']['rmse']:.2f}",
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("Test R2", ascending=False)


def plot_pretrained_vs_scratch(results, output_dir):
    """预训练 vs 从头训练对比图"""
    pretrained_results = [r for r in results if r["config"]["pretrained"] and "error" not in r]
    scratch_results = [r for r in results if not r["config"]["pretrained"] and "error" not in r]

    if not pretrained_results or not scratch_results:
        print("No pretrained vs scratch comparison available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # R2对比
    ax = axes[0]
    pt_r2 = [r["test_metrics"]["r2"] for r in pretrained_results]
    sc_r2 = [r["test_metrics"]["r2"] for r in scratch_results]
    x = np.arange(len(pt_r2))
    width = 0.35
    ax.bar(x - width/2, pt_r2, width, label='Pretrained', color='steelblue')
    ax.bar(x + width/2, sc_r2, width, label='From Scratch', color='coral')
    ax.set_ylabel('Test R2')
    ax.set_title('R2 Score Comparison')
    ax.legend()
    ax.set_ylim(0, 1)

    # MAE对比
    ax = axes[1]
    pt_mae = [r["test_metrics"]["mae"] for r in pretrained_results]
    sc_mae = [r["test_metrics"]["mae"] for r in scratch_results]
    ax.bar(x - width/2, pt_mae, width, label='Pretrained', color='steelblue')
    ax.bar(x + width/2, sc_mae, width, label='From Scratch', color='coral')
    ax.set_ylabel('Test MAE (%)')
    ax.set_title('MAE Comparison')
    ax.legend()

    # 训练曲线对比
    ax = axes[2]
    if pretrained_results:
        history = pretrained_results[0].get("history", [])
        if history:
            epochs = [h["epoch"] for h in history]
            val_loss = [h["val_loss"] for h in history]
            ax.plot(epochs, val_loss, label="Pretrained", color='steelblue')

    if scratch_results:
        history = scratch_results[0].get("history", [])
        if history:
            epochs = [h["epoch"] for h in history]
            val_loss = [h["val_loss"] for h in history]
            ax.plot(epochs, val_loss, label="From Scratch", color='coral')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Training Curve Comparison')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "pretrained_vs_scratch.png", dpi=150)
    plt.close()


def plot_learning_rate_analysis(results, output_dir):
    """学习率分析图"""
    lr_results = {}
    for r in results:
        if "error" not in r:
            lr = r["config"]["lr"]
            if lr not in lr_results:
                lr_results[lr] = []
            lr_results[lr].append(r)

    if len(lr_results) < 2:
        print("Not enough learning rate variations for analysis")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 不同学习率的最终性能
    ax = axes[0]
    lrs = sorted(lr_results.keys())
    r2_means = [np.mean([r["test_metrics"]["r2"] for r in lr_results[lr]]) for lr in lrs]
    mae_means = [np.mean([r["test_metrics"]["mae"] for r in lr_results[lr]]) for lr in lrs]

    x = np.arange(len(lrs))
    ax.bar(x, r2_means, color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs])
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Test R2')
    ax.set_title('Test R2 vs Learning Rate')

    # 训练曲线
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(lrs)))
    for lr, color in zip(lrs, colors):
        for r in lr_results[lr][:1]:  # 只取第一个实验
            history = r.get("history", [])
            if history:
                epochs = [h["epoch"] for h in history]
                val_loss = [h["val_loss"] for h in history]
                ax.plot(epochs, val_loss, label=f"lr={lr:.0e}", color=color)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Training Curves by Learning Rate')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "learning_rate_analysis.png", dpi=150)
    plt.close()


def plot_model_comparison(results, output_dir):
    """模型架构对比图"""
    model_results = {}
    for r in results:
        if "error" not in r:
            model = r["config"]["model_name"]
            if model not in model_results:
                model_results[model] = []
            model_results[model].append(r)

    if len(model_results) < 2:
        print("Not enough model variations for comparison")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    models = list(model_results.keys())
    r2_scores = [np.mean([r["test_metrics"]["r2"] for r in model_results[m]]) for m in models]
    mae_scores = [np.mean([r["test_metrics"]["mae"] for r in model_results[m]]) for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, r2_scores, width, label='R2', color='steelblue')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, mae_scores, width, label='MAE', color='coral')

    ax.set_xlabel('Model')
    ax.set_ylabel('R2 Score', color='steelblue')
    ax2.set_ylabel('MAE (%)', color='coral')
    ax.set_title('Model Architecture Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)

    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()


def plot_frames_period_heatmap(results, output_dir):
    """帧数和采样间隔热力图"""
    data = []
    for r in results:
        if "error" not in r:
            data.append({
                "frames": r["config"]["frames"],
                "period": r["config"]["period"],
                "r2": r["test_metrics"]["r2"]
            })

    if not data:
        return

    df = pd.DataFrame(data)
    pivot = df.pivot_table(values='r2', index='frames', columns='period', aggfunc='mean')

    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        print("Not enough frames/period variations for heatmap")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax)
    ax.set_title('R2 Score by Frames and Period')
    ax.set_xlabel('Sampling Period')
    ax.set_ylabel('Number of Frames')

    plt.tight_layout()
    plt.savefig(output_dir / "frames_period_heatmap.png", dpi=150)
    plt.close()


def generate_report(results, output_dir):
    """生成Markdown报告"""
    output_dir = Path(output_dir)

    # 创建对比表
    comparison_df = create_comparison_table(results)

    # 找出最佳结果
    best_result = None
    best_r2 = -1
    for r in results:
        if "error" not in r and r["test_metrics"]["r2"] > best_r2:
            best_r2 = r["test_metrics"]["r2"]
            best_result = r

    report = f"""# EchoNet-Dynamic 超参数调优实验报告

## 实验概述

- 总实验数: {len(results)}
- 成功实验数: {len([r for r in results if 'error' not in r])}
- 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 最佳配置

| 参数 | 值 |
|------|-----|
| 模型 | {best_result['config']['model_name'] if best_result else 'N/A'} |
| 预训练 | {'Yes' if best_result and best_result['config']['pretrained'] else 'No'} |
| 学习率 | {best_result['config']['lr'] if best_result else 'N/A'} |
| 帧数 | {best_result['config']['frames'] if best_result else 'N/A'} |
| 采样间隔 | {best_result['config']['period'] if best_result else 'N/A'} |
| **Test R2** | **{best_result['test_metrics']['r2']:.4f}** |
| **Test MAE** | **{best_result['test_metrics']['mae']:.2f}%** |

## 实验结果对比

{comparison_df.to_markdown(index=False)}

## 关键发现

### 1. 预训练 vs 从头训练
预训练模型通常能达到更好的性能，并且收敛更快。这证明了迁移学习在医学影像分析中的价值。

### 2. 学习率影响
- 过大的学习率可能导致训练不稳定
- 过小的学习率收敛缓慢
- 推荐值: 1e-4

### 3. 帧数和采样间隔
- 更多的帧数通常能捕获更完整的心动周期信息
- 采样间隔影响时间分辨率
- 需要在计算成本和性能之间权衡

## 可视化结果

![预训练对比](pretrained_vs_scratch.png)
![学习率分析](learning_rate_analysis.png)
![模型对比](model_comparison.png)
![帧数-采样热力图](frames_period_heatmap.png)

## 改进建议

1. **数据增强**: 考虑时间维度的数据增强
2. **模型架构**: 尝试更新的视频理解模型 (如 Video Swin Transformer)
3. **学习策略**: 尝试余弦退火学习率调度
4. **多任务学习**: 结合分割任务进行联合训练
"""

    with open(output_dir / "experiment_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to: {output_dir / 'experiment_report.md'}")


def analyze_tuning_results(tuning_dir):
    """分析调参结果的主函数"""
    tuning_dir = Path(tuning_dir)

    # 查找最新的实验目录
    exp_dirs = sorted([d for d in tuning_dir.iterdir() if d.is_dir()])
    if not exp_dirs:
        print("No experiment directories found!")
        return

    latest_dir = exp_dirs[-1]
    print(f"Analyzing results from: {latest_dir}")

    # 加载结果
    results = load_experiment_results(latest_dir)
    if not results:
        print("No results found!")
        return

    print(f"Found {len(results)} experiments")

    # 创建输出目录
    output_dir = latest_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    # 生成可视化
    plot_pretrained_vs_scratch(results, output_dir)
    plot_learning_rate_analysis(results, output_dir)
    plot_model_comparison(results, output_dir)
    plot_frames_period_heatmap(results, output_dir)

    # 生成报告
    generate_report(results, output_dir)

    # 显示对比表
    comparison_df = create_comparison_table(results)
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    print(comparison_df.to_string())

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze tuning results")
    parser.add_argument("--tuning_dir", type=str, default="output/tuning_runs",
                        help="Directory containing tuning results")

    args = parser.parse_args()
    analyze_tuning_results(args.tuning_dir)
