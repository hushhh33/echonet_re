"""Test GPU training speed"""
import torch
import time

print("Testing GPU speed...")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test data (batch_size=4, 3 channels, 32 frames, 112x112)
    x = torch.randn(4, 3, 32, 112, 112).cuda()

    # Load model
    import torchvision
    model = torchvision.models.video.r2plus1d_18(weights=None).cuda()
    model.eval()

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)

    # Inference timing
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    torch.cuda.synchronize()
    inference_time = (time.time() - start) / 10

    print(f"\nInference: {inference_time*1000:.1f}ms per batch")
    print(f"Estimated inference per epoch (1865 batches): {1865 * inference_time / 60:.1f} min")

    # Training speed with gradients
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    train_time = (time.time() - start) / 5

    print(f"Training (with backprop): {train_time*1000:.1f}ms per batch")
    print(f"Estimated training per epoch: {1865 * train_time / 60:.1f} min")

    print("\n=== Summary ===")
    print(f"5 epochs would take: {5 * 1865 * train_time / 60:.0f} min")
    print(f"Note: Data loading adds extra time (maybe 2-3x slower)")
else:
    print("CUDA not available!")
