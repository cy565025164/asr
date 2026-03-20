#!/bin/bash
# Qwen3-Omni vLLM 部署
# 注意：需要安装 qwen3_omni 分支的 vLLM
# git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git
#
# 用法: bash deploy_vllm.sh [模型路径] [端口] [GPU数量]

MODEL_PATH=${1:-"Qwen/Qwen3-Omni-30B-A3B-Instruct"}
PORT=${2:-8000}
NUM_GPUS=${3:-2}

echo "============================================"
echo "  Qwen3-Omni vLLM 部署"
echo "  模型: ${MODEL_PATH}"
echo "  端口: ${PORT}"
echo "  GPU:  ${NUM_GPUS}"
echo "============================================"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port ${PORT} \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --tensor-parallel-size ${NUM_GPUS} \
    --dtype bfloat16 \
    --trust-remote-code \
    --limit-mm-per-prompt "audio=5"
