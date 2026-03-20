#!/bin/bash
# Qwen3-Omni 部署脚本
# 用法: bash deploy_omni.sh [模型路径] [端口] [GPU]

MODEL_PATH=${1:-"Qwen/Qwen3-Omni-30B-A3B"}
PORT=${2:-9001}
DEVICE=${3:-"cuda:0"}

echo "============================================"
echo "  Qwen3-Omni 音频分析服务"
echo "  模型: ${MODEL_PATH}"
echo "  端口: ${PORT}"
echo "  设备: ${DEVICE}"
echo "============================================"

python $(dirname $0)/../inference/omni_analyzer.py \
    --model_path ${MODEL_PATH} \
    --device ${DEVICE} \
    --port ${PORT}
