# Qwen3-Omni 电销场景后训练

基于 **Qwen3-Omni-30B-A3B-Instruct** 的电销对话场景后训练方案。
单模型同时完成 ASR 语音识别和音频理解（性别/年龄/情绪/购买意向）。

## 核心思路

```
                                    ┌→ ASR 转录文字
用户音频 + 销售员上下文 → Qwen3-Omni ─┤
                                    └→ 音频分析（性别/年龄/情绪/意向）
```

通过 prompt 控制输出模式：ASR 模式、分析模式、或两者同时输出。

## 项目结构

```
├── data/
│   ├── README.md                  # 数据格式说明
│   ├── sample_sft.jsonl           # SFT 示例数据
│   ├── sample_dpo.jsonl           # DPO 示例数据
│   ├── build_sft_dataset.py       # SFT 数据集构建
│   └── build_dpo_dataset.py       # DPO 数据集构建
├── training/
│   ├── train_sft.py               # SFT 训练
│   ├── train_dpo.py               # DPO 训练
│   ├── ds_config_zero2.json
│   └── ds_config_zero3.json
├── deploy/
│   ├── deploy_vllm.sh             # vLLM 部署
│   └── deploy_transformers.py     # Transformers 本地部署
├── inference/
│   ├── client.py                  # 推理客户端
│   └── server.py                  # HTTP 服务
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境安装

```bash
# transformers 需从源码安装（Qwen3-Omni 支持已合并但未发 PyPI）
pip install git+https://github.com/huggingface/transformers
pip install accelerate qwen-omni-utils librosa soundfile datasets
pip install flash-attn --no-build-isolation  # 推荐
```

### 2. 数据格式

SFT 数据（对齐 Qwen3-Omni chat 格式）：
```json
{
  "messages": [
    {"role": "system", "content": "你是电销场景语音识别助手。销售员上一句：请问您对保险产品感兴趣吗？"},
    {"role": "user", "content": [{"type": "audio", "audio": "/data/user.wav"}, {"type": "text", "text": "请识别用户语音并分析说话人特征"}]},
    {"role": "assistant", "content": "【ASR】嗯，我想了解一下你们那个百万医疗险\n【分析】性别：女 | 年龄段：中年 | 情绪：积极 | 购买意向：有兴趣"}
  ]
}
```

### 3. SFT 训练

```bash
# 单卡
python training/train_sft.py \
    --model_path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --train_file data/train_sft.jsonl \
    --output_dir ./output_sft \
    --batch_size 1 --grad_acc 16 --lr 1e-5 --epochs 3

# 多卡
torchrun --nproc_per_node=4 training/train_sft.py \
    --model_path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --train_file data/train_sft.jsonl \
    --output_dir ./output_sft \
    --batch_size 1 --grad_acc 16 --lr 1e-5 --epochs 3
```

### 4. DPO 训练

```bash
python training/train_dpo.py \
    --model_path ./output_sft/final \
    --train_file data/train_dpo.jsonl \
    --output_dir ./output_dpo \
    --batch_size 1 --grad_acc 8 --lr 5e-7 --beta 0.1
```

### 5. 部署

```bash
# vLLM（推荐，需安装 qwen3_omni 分支）
bash deploy/deploy_vllm.sh ./output_dpo/final 8000 2

# Transformers 本地部署
python deploy/deploy_transformers.py --model_path ./output_dpo/final --port 9000
```

### 6. 推理

```bash
# ASR 模式
python inference/client.py --model_path ./output_dpo/final \
    --audio user.wav --context "销售员：请问您需要什么产品？" --mode asr

# 分析模式
python inference/client.py --model_path ./output_dpo/final \
    --audio user.wav --context "销售员：请问您需要什么产品？" --mode analyze

# 全量模式（ASR + 分析）
python inference/client.py --model_path ./output_dpo/final \
    --audio user.wav --context "销售员：请问您需要什么产品？" --mode full
```

## Prompt 设计

### ASR 模式
```
System: 你是电销场景语音识别助手。销售员上一句：{sales_context}
User: [audio] 请准确识别用户语音内容。
```

### 分析模式
```
System: 你是电销场景语音分析助手。销售员上一句：{sales_context}
User: [audio] 请分析说话人的性别、年龄段、情绪和购买意向。
```

### 全量模式
```
System: 你是电销场景语音识别与分析助手。销售员上一句：{sales_context}
User: [audio] 请先识别用户语音内容，然后分析说话人特征。
```
