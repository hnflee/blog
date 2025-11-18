#!/bin/bash
# 一键设置和运行推理脚本

echo "================================================================================"
echo "          多航班动态定价推理 - 自动安装和运行"
echo "================================================================================"

# 配置你的工作目录
WORK_DIR="/Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch"
CHECKPOINT="/Users/lifeng/ray_results/FlightPPO_Final_2025/PPO_multi_flight_v0_f4f6f_00000_0_2025-11-18_16-31-19/checkpoint_000000"

echo ""
echo "[步骤1] 检查工作目录..."
if [ ! -d "$WORK_DIR" ]; then
    echo "❌ 工作目录不存在: $WORK_DIR"
    echo "请修改脚本中的 WORK_DIR 变量"
    exit 1
fi
echo "✓ 工作目录存在: $WORK_DIR"

echo ""
echo "[步骤2] 检查环境文件..."
if [ ! -f "$WORK_DIR/qln_multi_flight_game_v2.py" ]; then
    echo "❌ 环境文件不存在: qln_multi_flight_game_v2.py"
    exit 1
fi
echo "✓ 环境文件存在"

echo ""
echo "[步骤3] 检查 Checkpoint..."
if [ ! -d "$CHECKPOINT" ]; then
    echo "❌ Checkpoint 不存在: $CHECKPOINT"
    echo "请修改脚本中的 CHECKPOINT 变量"
    exit 1
fi
echo "✓ Checkpoint 存在"

echo ""
echo "[步骤4] 复制推理脚本..."
if [ -f "./flight_inference.py" ]; then
    cp ./flight_inference.py "$WORK_DIR/"
    echo "✓ 推理脚本已复制到工作目录"
else
    echo "❌ 找不到 flight_inference.py"
    echo "请确保在正确的目录运行此脚本"
    exit 1
fi

echo ""
echo "[步骤5] 清理旧的 Ray 进程和缓存..."
ray stop 2>/dev/null
find "$WORK_DIR" -name "*.pyc" -delete 2>/dev/null
find "$WORK_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "✓ 清理完成"

echo ""
echo "[步骤6] 进入工作目录..."
cd "$WORK_DIR" || exit 1
echo "✓ 当前目录: $(pwd)"

echo ""
echo "================================================================================"
echo "          准备就绪！开始运行推理..."
echo "================================================================================"
echo ""

# 运行推理
# 你可以在这里添加自定义参数
python flight_inference.py --episodes 10

# 或者使用自定义参数：
# python flight_inference.py --flight1_days 4 --flight2_capacity 90

echo ""
echo "================================================================================"
echo "          完成！"
echo "================================================================================"
