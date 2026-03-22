#!/bin/bash
# EchoNet-Dynamic 服务器配置脚本
# 使用方法: bash setup_server.sh

echo "======================================"
echo "EchoNet-Dynamic 服务器配置"
echo "======================================"


cd /root/autodl-tmp/cardio/dynamic-master/dynamic-master
echo "[2/4] 工作目录: $(pwd)"

# 3. 检查数据集
echo "[3/4] 检查数据集..."
DATA_DIR="/root/autodl-tmp/cardio/EchoNet-Dynamic"
if [ -d "$DATA_DIR/Videos" ]; then
    VIDEO_COUNT=$(ls -1 "$DATA_DIR/Videos" | wc -l)
    echo "  数据集路径: $DATA_DIR"
    echo "  视频数量: $VIDEO_COUNT"
else
    echo "  警告: 数据集目录不存在!"
    echo "  请确保数据集在: $DATA_DIR"
fi

# 4. 安装项目
echo "[4/4] 安装echonet包..."
pip install -e . -q
pip install tensorboard -q
echo "  安装完成"

echo ""
echo "======================================"
echo "配置完成!"
echo "======================================"
echo ""
echo "运行训练命令:"
echo "  cd /root/autodl-tmp/cardio/dynamic-master/dynamic-master"
echo ""
echo "  # 快速对比实验 (约20分钟)"
echo "  python scripts/hyperparameter_tuning.py --mode quick --epochs 5 --subset 500"
echo ""
echo "  # 完整训练"
echo "  echonet video --num_epochs 5 --batch_size 8"
echo ""
echo "  # 查看TensorBoard"
echo "  tensorboard --logdir=output/tuning_runs --port 6006"
echo ""
