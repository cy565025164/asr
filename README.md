# Qwen3-ASR 电销场景后训练

基于 Qwen3-ASR 的电销对话场景 ASR 后训练方案，支持 SFT 和 DPO 两种训练方式。

核心思路：将**销售员的话术原文作为 context**，帮助模型更准确识别用户语音。

## 项目结构

```
├── data/                       # 数据相关
│   ├── README.md               # 数据格式说明
│   ├── sample_sft.jsonl        # SFT 示例数据
│   ├── sample_dpo.jsonl        # DPO 示例数据
│   ├── build_sft_dataset.py    # SFT 数据集构建工具
│   └── build_dpo_dataset.py    # DPO 数据集构建工具
├── training/                   # 训练脚本
│   ├── train_sft.py            # SFT 训练（对齐官方）
│   ├── train_dpo.py            # DPO 训练
│   ├── ds_config_zero2.json    # DeepSpeed ZeRO-2
│   └── ds_config_zero3.json    # DeepSpeed ZeRO-3
├── deploy/                     # 部署
│   ├── deploy_vllm.sh          # vLLM 服务部署
│   └── deploy_local.py         # 本地 SDK 部署
├── inference/                  # 推理
│   ├── asr_client.py           # 推理客户端
│   └── asr_server.py           # HTTP 服务
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境安装

```bash
pip install -U qwen-asr datasets librosa soundfile
pip install -U flash-attn --no-build-isolation  # 可选，加速
```

### 2. 准备数据

SFT 数据格式（对齐官方，prompt 字段放销售员话术）：
```json
{"audio": "/data/audio/user_turn.wav", "text": "language Chinese<asr_text>嗯，我想了解一下你们那个百万医疗险", "prompt": "销售员：请问您对我们的保险产品感兴趣吗？"}
```

DPO 数据格式：
```json
{"audio": "/data/audio/user_turn.wav", "prompt": "销售员：请问您对我们的保险产品感兴趣吗？", "chosen": "language Chinese<asr_text>嗯，我想了解一下百万医疗险", "rejected": "language Chinese<asr_text>嗯我想了解一下百万医疗"}
```

从电销录音构建数据集：
```bash
# SFT 数据
python data/build_sft_dataset.py \
    --annotation_file annotations.jsonl \
    --output_file data/train_sft.jsonl

# DPO 数据（需要模型推理生成 rejected）
python data/build_dpo_dataset.py \
    --annotation_file annotations.jsonl \
    --model_path Qwen/Qwen3-ASR-1.7B \
    --output_file data/train_dpo.jsonl
```

### 3. SFT 训练

```bash
# 单卡
python training/train_sft.py \
    --model_path Qwen/Qwen3-ASR-1.7B \
    --train_file data/train_sft.jsonl \
    --output_dir ./output_sft \
    --batch_size 32 --grad_acc 4 --lr 2e-5 --epochs 3

# 多卡
torchrun --nproc_per_node=4 training/train_sft.py \
    --model_path Qwen/Qwen3-ASR-1.7B \
    --train_file data/train_sft.jsonl \
    --output_dir ./output_sft \
    --batch_size 32 --grad_acc 4 --lr 2e-5 --epochs 3
```

### 4. DPO 训练（在 SFT 基础上）

```bash
python training/train_dpo.py \
    --model_path ./output_sft/final \
    --train_file data/train_dpo.jsonl \
    --output_dir ./output_dpo \
    --batch_size 8 --grad_acc 8 --lr 5e-7 --beta 0.1 --epochs 1
```

### 5. 部署

```bash
# vLLM 服务
bash deploy/deploy_vllm.sh ./output_dpo/final 8000 1

# 本地 SDK
python deploy/deploy_local.py --model_path ./output_dpo/final --port 9000
```

### 6. 推理

```bash
# 本地 SDK 模式（推荐，完整支持 context）
python inference/asr_client.py --local \
    --model_path ./output_dpo/final \
    --audio user.wav \
    --context "销售员：请问您对我们的保险产品感兴趣吗？"

# vLLM API 模式
python inference/asr_client.py \
    --base_url http://localhost:8000 \
    --audio user.wav \
    --context "销售员：请问您对我们的保险产品感兴趣吗？"

# 批量识别
python inference/asr_client.py --local \
    --model_path ./output_dpo/final \
    --audio_dir /data/test_audios/ \
    --context_file contexts.jsonl \
    --output results.jsonl
```

## Context 机制说明

Qwen3-ASR 原生支持 `context` 参数，直接传入销售员的话术原文：

```python
results = model.transcribe(
    audio="user_audio.wav",
    context="销售员：请问您对我们的保险产品感兴趣吗？",
    language="Chinese",
)
```

模型会利用 context 信息偏向识别相关领域词汇。在电销场景中：
- 销售员提到"百万医疗险" → 用户回复中更容易正确识别这个词
- 销售员提到"免赔额" → 减少同音字误识别

两层优化策略：
| 层级 | 方式 | 效果 | 成本 |
|------|------|------|------|
| 推理时 | context 参数传销售员话术 | 即时生效，提升术语识别 | 零成本 |
| 训练时 | SFT/DPO 后训练 | 深度优化整体识别质量 | 需要标注数据 + GPU |
