#!/usr/bin/env python3
"""
Quick smoke-test training + inference for EF regression.

This is intended for fast validation (e.g., CPU-only) and does NOT reproduce
paper results. It trains on small subsets and writes a predictions CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from typing import Iterable, Tuple

import numpy as np
import torch
import torchvision

import echonet


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _pick_indices(n_total: int, n_pick: int, seed: int) -> np.ndarray:
    n_pick = min(n_pick, n_total)
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n_pick, replace=False)


def _make_model(model_name: str, pretrained: bool, device: torch.device) -> torch.nn.Module:
    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6
    model.to(device)
    return model


@torch.no_grad()
def _infer(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    filenames: list[str] = []
    y_true: list[float] = []
    y_pred: list[float] = []

    for x, target in dataloader:
        x = x.to(device)
        pred = model(x).view(-1).detach().cpu().numpy()

        if isinstance(target, (tuple, list)) and len(target) == 2:
            batch_filenames, batch_ef = target
        else:
            raise ValueError("Expected target=(Filename, EF) for inference.")

        if isinstance(batch_filenames, (tuple, list)):
            batch_filenames = list(batch_filenames)
        else:
            batch_filenames = [str(batch_filenames)]

        if torch.is_tensor(batch_ef):
            batch_ef_np = batch_ef.detach().cpu().numpy().reshape(-1)
        else:
            batch_ef_np = np.asarray(batch_ef, dtype=np.float32).reshape(-1)

        if len(batch_filenames) != pred.shape[0] or batch_ef_np.shape[0] != pred.shape[0]:
            raise ValueError("Batch size mismatch while collating inference targets.")

        filenames.extend(batch_filenames)
        y_true.extend(batch_ef_np.tolist())
        y_pred.extend(pred.tolist())

    return filenames, np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, help="Dataset root (contains FileList.csv, VolumeTracings.csv, Videos/).")
    parser.add_argument("--output", default=os.path.join("output", "quick_ef_smoke"), help="Output directory.")

    parser.add_argument("--model_name", default="r2plus1d_18")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--random_init", action="store_true", default=False)

    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--period", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train_samples", type=int, default=8)
    parser.add_argument("--val_samples", type=int, default=8)
    parser.add_argument("--test_samples", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    _seed_everything(args.seed)

    os.makedirs(args.output, exist_ok=True)

    device = torch.device(args.device)
    pretrained = bool(args.pretrained and (not args.random_init))

    # Compute mean/std on training split (subsampled internally by get_mean_and_std).
    # Use num_workers=0 for Windows compatibility (avoids multiprocessing pipe permission issues).
    mean, std = echonet.utils.get_mean_and_std(
        echonet.datasets.Echo(root=args.data_dir, split="train"),
        num_workers=0,
    )

    ds_kwargs = {
        "target_type": "EF",
        "mean": mean,
        "std": std,
        "length": args.frames,
        "period": args.period,
    }

    train_full = echonet.datasets.Echo(root=args.data_dir, split="train", **ds_kwargs, pad=12)
    val_full = echonet.datasets.Echo(root=args.data_dir, split="val", **ds_kwargs)
    test_full = echonet.datasets.Echo(root=args.data_dir, split="test", **ds_kwargs)

    train_idx = _pick_indices(len(train_full), args.train_samples, args.seed + 1)
    val_idx = _pick_indices(len(val_full), args.val_samples, args.seed + 2)
    test_idx = _pick_indices(len(test_full), args.test_samples, args.seed + 3)

    train_ds = torch.utils.data.Subset(train_full, train_idx.tolist())
    val_ds = torch.utils.data.Subset(val_full, val_idx.tolist())

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    model = _make_model(args.model_name, pretrained=pretrained, device=device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

    best_val = float("inf")
    log_path = os.path.join(args.output, "log.csv")
    ckpt_path = os.path.join(args.output, "checkpoint.pt")
    best_path = os.path.join(args.output, "best.pt")

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if os.stat(log_path).st_size == 0:
            writer.writerow(["epoch", "phase", "loss", "seconds", "n_samples"])

        for epoch in range(args.epochs):
            # Train (limited steps)
            model.train(True)
            start = time.time()
            total_loss = 0.0
            n = 0
            for step, (x, y) in enumerate(train_loader):
                if step >= args.steps_per_epoch:
                    break
                x = x.to(device)
                y = y.to(device).float().view(-1)
                pred = model(x).view(-1)
                loss = torch.nn.functional.mse_loss(pred, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                bs = x.shape[0]
                total_loss += loss.item() * bs
                n += bs

            train_loss = total_loss / max(1, n)
            writer.writerow([epoch, "train", f"{train_loss:.6f}", f"{time.time() - start:.2f}", n])
            f.flush()

            # Val (full val subset)
            model.train(False)
            start = time.time()
            total_loss = 0.0
            n = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device).float().view(-1)
                    pred = model(x).view(-1)
                    loss = torch.nn.functional.mse_loss(pred, y)
                    bs = x.shape[0]
                    total_loss += loss.item() * bs
                    n += bs

            val_loss = total_loss / max(1, n)
            writer.writerow([epoch, "val", f"{val_loss:.6f}", f"{time.time() - start:.2f}", n])
            f.flush()

            save = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "frames": args.frames,
                "period": args.period,
                "mean": np.asarray(mean),
                "std": np.asarray(std),
                "model_name": args.model_name,
                "pretrained": pretrained,
                "val_loss": val_loss,
            }
            torch.save(save, ckpt_path)
            if val_loss < best_val:
                torch.save(save, best_path)
                best_val = val_loss

    # Inference on a small test subset; include filenames in targets.
    test_ds = torch.utils.data.Subset(
        echonet.datasets.Echo(
            root=args.data_dir,
            split="test",
            target_type=["Filename", "EF"],
            mean=mean,
            std=std,
            length=args.frames,
            period=args.period,
        ),
        test_idx.tolist(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    # Load best checkpoint for inference.
    # PyTorch 2.6+ defaults `weights_only=True` which can reject non-tensor
    # objects. This checkpoint is created locally by this script, so we load
    # with `weights_only=False`.
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])

    filenames, y, yhat = _infer(model, test_loader, device=device)
    mae = float(np.mean(np.abs(yhat - y))) if y.size else float("nan")
    r2 = float(1 - np.sum((yhat - y) ** 2) / np.sum((y - y.mean()) ** 2)) if y.size and np.sum((y - y.mean()) ** 2) > 0 else float("nan")

    pred_path = os.path.join(args.output, "test_predictions.csv")
    with open(pred_path, "w", newline="") as g:
        writer = csv.writer(g)
        writer.writerow(["Filename", "EF", "Prediction"])
        for fn, yt, yp in zip(filenames, y.tolist(), yhat.tolist()):
            writer.writerow([fn, f"{yt:.4f}", f"{yp:.4f}"])

    metrics_path = os.path.join(args.output, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as g:
        g.write(f"test_samples={len(test_ds)}\n")
        g.write(f"mae={mae:.4f}\n")
        g.write(f"r2={r2:.4f}\n")

    print(f"[OK] wrote: {log_path}")
    print(f"[OK] wrote: {best_path}")
    print(f"[OK] wrote: {pred_path}")
    print(f"[OK] test MAE={mae:.4f} R2={r2:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
