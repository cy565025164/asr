# Qwen3-ASR DPO 后训练

## 概述

基于 Qwen3-ASR 的 DPO 后训练方案，专门针对**电销对话场景**，通过引入销售员上下文提升用户语音识别准确率。

## 核心思路

```
输入 = 销售员对话上下文（文本） + 用户语音（音频）
chosen = 正确的用户话术转录（人工标注）
rejected = 错误的转录（模型原始输出 / 扰动生成）

DPO 目标：让模型在给定上下文的情况下，更倾向于生成正确的转录
```

## 文件结构

```
qwen3_asr_dpo/
├── train_asr_dpo.py          # 主训练脚本
├── build_dpo_dataset.py      # 数据集构建工具
├── ds_config_zero2.json      # DeepSpeed ZeRO-2 配置
├── ds_config_zero3.json      # DeepSpeed ZeRO-3 配置（显存不足时）
└── README.md                 # 本文件
```

## 数据格式

### DPO 训练数据 (JSONL)

```json
{
    "audio_path": "/data/audio/call_001_user_03.wav",
    "sales_context": "请问您对我们的保险产品感兴趣吗？",
    "chosen": "嗯，我想了解一下你们那个百万医疗险，就是住院能报销的那种",
    "rejected": "嗯我想了解一下你们那个百万医疗险就是住院能报销那种",
    "dialogue_history": [
        {"role": "sales", "text": "您好，我是XX保险的客服小王"},
        {"role": "user", "text": "嗯你好"},
        {"role": "sales", "text": "请问您对我们的保险产品感兴趣吗？"}
    ]
}
```

字段说明：
- `audio_path`: 用户语音文件路径（wav/mp3/flac）
- `sales_context`: 销售员最近一句话（必填）
- `chosen`: 正确转录（人工标注）
- `rejected`: 错误转录（模型输出 / 扰动生成）
- `dialogue_history`: 完整对话历史（可选，提供更丰富上下文）

### 标注文件格式（用于构建 DPO 数据）

```json
{
    "call_id": "call_001",
    "turn_id": 3,
    "audio_path": "/data/audio/call_001_user_03.wav",
    "correct_text": "嗯，我想了解一下你们那个百万医疗险",
    "sales_context": "请问您对我们的保险产品感兴趣吗？",
    "dialogue_history": [...]
}
```

## 使用方法

### 第 1 步：构建 DPO 数据集

**方式 A：用模型推理生成 rejected（推荐）**

用 Qwen3-ASR 原始模型跑推理，模型的错误输出作为 rejected：

```bash
python build_dpo_dataset.py \
    --annotation_file /data/annotations.jsonl \
    --output_file /data/dpo_train.jsonl \
    --model_path /path/to/Qwen3-ASR-1.7B
```

**方式 B：用文本扰动生成 rejected**

不需要 GPU，通过去标点、交换字符等方式构造错误样本：

```bash
python build_dpo_dataset.py \
    --annotation_file /data/annotations.jsonl \
    --output_file /data/dpo_train.jsonl \
    --use_perturbation
```

### 第 2 步：训练

**单卡训练：**

```bash
python train_asr_dpo.py \
    --model_name_or_path /path/to/Qwen3-ASR-1.7B \
    --dataset_path /data/dpo_train.jsonl \
    --output_dir ./output_asr_dpo \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --bf16 \
    --gradient_checkpointing
```

**多卡 + DeepSpeed：**

```bash
deepspeed --num_gpus=4 train_asr_dpo.py \
    --model_name_or_path /path/to/Qwen3-ASR-1.7B \
    --dataset_path /data/dpo_train.jsonl \
    --output_dir ./output_asr_dpo \
    --deepspeed ds_config_zero2.json \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --bf16 \
    --gradient_checkpointing
```

### 第 3 步：推理验证

```python
from qwen_asr import Qwen3ASRModel
import torch

model = Qwen3ASRModel.from_pretrained(
    "./output_asr_dpo/final",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = model.transcribe(
    audio="/data/test_audio.wav",
    language="Chinese",
)
print(results[0].text)
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beta` | 0.1 | DPO 温度。越小模型偏离 reference 越大 |
| `learning_rate` | 5e-7 | 文本解码器学习率 |
| `lr_audio_encoder` | 0.0 | 音频编码器学习率。0 = 冻结（推荐） |
| `loss_type` | sigmoid | DPO loss 类型：sigmoid / hinge / ipo |
| `max_history_turns` | 3 | 最多使用几轮对话历史作为上下文 |
| `max_audio_len` | 480000 | 最大音频长度（采样点数，30s @ 16kHz） |
| `max_length` | 2048 | 最大 token 序列长度 |

## 设计要点

### 为什么冻结音频编码器？

1. 音频编码器已在大规模数据上预训练，特征提取能力强
2. DPO 的目标是调整"在上下文条件下的文本生成偏好"，主要作用在解码器
3. 冻结编码器可以大幅减少显存和训练时间
4. 如需微调编码器，设置 `--lr_audio_encoder 1e-8`（用很小的学习率）

### 上下文注入方式

Prompt 结构：
```
<|im_start|>system
你是一个电销场景的语音识别助手。<|im_end|>
<|im_start|>user
【对话上下文】
销售员：您好，请问您对我们的产品感兴趣吗？
用户：嗯你好
销售员：那我给您介绍一下我们最新的百万医疗险
【请识别以下用户语音】
<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>
<|im_start|>assistant
{chosen / rejected 转录文本}<|im_end|>
```

上下文帮助模型：
- 预判用户可能说的领域词汇（如"百万医疗险"）
- 理解对话轮次关系，减少断句错误
- 区分相似发音词（如"报销" vs "保险"）

### rejected 样本来源

推荐混合使用：
1. **模型推理输出**（60%）：最真实的错误分布
2. **标点/断句扰动**（20%）：针对停顿、标点问题
3. **同音字替换**（10%）：针对发音相似的错误
4. **漏字/多字**（10%）：针对拟声词问题

## 环境依赖

```bash
pip install torch transformers>=4.57.0 qwen-asr
pip install trl datasets accelerate deepspeed
pip install librosa soundfile
pip install flash-attn --no-build-isolation  # 可选，加速
```
