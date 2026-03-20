#!/bin/bash
# Qwen3-ASR vLLM 部署脚本
# 用法: bash deploy_vllm.sh [模型路径] [端口] [GPU数量]

MODEL_PATH=${1:-"./output_dpo/final"}
PORT=${2:-8000}
NUM_GPUS=${3:-1}
HOST="0.0.0.0"

echo "============================================"
echo "  Qwen3-ASR vLLM 部署"
echo "  模型: ${MODEL_PATH}"
echo "  地址: ${HOST}:${PORT}"
echo "  GPU:  ${NUM_GPUS}"
echo "============================================"

# 检查 qwen-asr[vllm]
if ! python -c "import qwen_asr" 2>/dev/null; then
    echo "[WARN] 安装 qwen-asr..."
    pip install -U "qwen-asr[vllm]"
fi

# 优先使用 qwen-asr-serve（官方封装）
if command -v qwen-asr-serve &>/dev/null; then
    echo "[INFO] 使用 qwen-asr-serve 启动..."
    qwen-asr-serve "${MODEL_PATH}" \
        --host ${HOST} \
        --port ${PORT} \
        --gpu-memory-utilization 0.85 \
        --max-model-len 4096 \
        --tensor-parallel-size ${NUM_GPUS} \
        --dtype bfloat16 \
        --trust-remote-code
else
    echo "[INFO] 使用 vllm serve 启动..."
    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --host ${HOST} \
        --port ${PORT} \
        --gpu-memory-utilization 0.85 \
        --max-model-len 4096 \
        --tensor-parallel-size ${NUM_GPUS} \
        --dtype bfloat16 \
        --trust-remote-code
fi
