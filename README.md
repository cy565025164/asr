# Qwen3-Omni 电销场景多任务后训练

基于 **Qwen3-Omni-30B-A3B-Instruct** 的电销场景多任务后训练方案。
同一基座模型，各任务数据独立，客户端按需组合。

## 架构

```
                         ┌─ ASR 语音识别     (独立数据/标注)
用户音频 → Qwen3-Omni ──┼─ 情绪识别         (独立数据/标注)
   +                     ├─ 性别识别         (独立数据/标注)
 销售上下文              ├─ 年龄段识别       (独立数据/标注)
                         └─ 任务N...        (未来扩展)
```

**设计原则：**
- 基座模型唯一：Qwen3-Omni-30B-A3B-Instruct
- 各任务数据独立标注、独立文件
- 训练时混合、推理时按需选择
- 新增任务只需：加 `tasks.json` 定义 + 准备标注数据

## 项目结构

```
├── data/
│   ├── tasks.json                        # 任务定义（system prompt / 输出格式）
│   ├── sample_annotations_asr.jsonl      # ASR 标注样例
│   ├── sample_annotations_emotion.jsonl  # 情绪标注样例
│   ├── sample_annotations_gender.jsonl   # 性别标注样例
│   ├── sample_annotations_age.jsonl      # 年龄标注样例
│   ├── build_sft_dataset.py              # 多任务 SFT 数据构建
│   └── build_dpo_dataset.py              # DPO 数据构建
├── training/
│   ├── train_sft.py                      # SFT 训练
│   ├── train_dpo.py                      # DPO 训练
│   ├── ds_config_zero2.json
│   └── ds_config_zero3.json
├── deploy/
│   ├── deploy_vllm.sh                    # vLLM 部署
│   └── deploy_transformers.py            # Transformers 本地部署
├── inference/
│   ├── client.py                         # 多任务推理客户端
│   └── server.py                         # HTTP 服务
└── requirements.txt
```

## 快速开始

### 1. 环境安装

```bash
pip install git+https://github.com/huggingface/transformers
pip install accelerate qwen-omni-utils librosa soundfile datasets
pip install flash-attn --no-build-isolation
```

### 2. 准备标注数据

各任务独立标注，每行一个 JSON：

**ASR** (`annotations_asr.jsonl`):
```json
{"audio_path": "call_001.wav", "sales_context": "请问您对保险感兴趣吗", "text": "嗯，我想了解一下百万医疗险"}
```

**情绪** (`annotations_emotion.jsonl`):
```json
{"audio_path": "call_001.wav", "sales_context": "请问您对保险感兴趣吗", "emotion": "积极", "intensity": "中", "confidence": 0.85}
```

**性别** (`annotations_gender.jsonl`):
```json
{"audio_path": "call_001.wav", "gender": "女", "confidence": 0.95}
```

**年龄** (`annotations_age.jsonl`):
```json
{"audio_path": "call_001.wav", "age_group": "中年", "confidence": 0.8}
```

### 3. 构建训练数据

```bash
# 多任务混合，按比例
python data/build_sft_dataset.py \
    --tasks asr:annotations_asr.jsonl:0.5 \
            emotion:annotations_emotion.jsonl:0.2 \
            gender:annotations_gender.jsonl:0.15 \
            age:annotations_age.jsonl:0.15 \
    --output train_sft.jsonl

# 也可以只构建单个任务
python data/build_sft_dataset.py \
    --tasks asr:annotations_asr.jsonl:1.0 \
    --output train_asr_only.jsonl
```

### 4. SFT 训练

```bash
# 单卡
python training/train_sft.py \
    --model_path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --train_file train_sft.jsonl \
    --output_dir ./output_sft \
    --batch_size 1 --grad_acc 16 --lr 1e-5 --epochs 3

# 多卡 + DeepSpeed ZeRO-2
deepspeed --num_gpus=4 training/train_sft.py \
    --model_path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --train_file train_sft.jsonl \
    --output_dir ./output_sft \
    --deepspeed training/ds_config_zero2.json
```

### 5. DPO 训练

```bash
# ASR DPO（文本扰动生成 rejected）
python data/build_dpo_dataset.py \
    --task asr --annotation_file annotations_asr.jsonl --output train_dpo_asr.jsonl

# 训练
python training/train_dpo.py \
    --model_path ./output_sft/final \
    --train_file train_dpo_asr.jsonl \
    --output_dir ./output_dpo \
    --beta 0.1
```

### 6. 推理

```bash
# 只做 ASR
python inference/client.py --local --model_path ./output_sft/final \
    --audio user.wav --context "销售员上一句" --tasks asr

# ASR + 情绪
python inference/client.py --local --model_path ./output_sft/final \
    --audio user.wav --tasks asr emotion

# 全部任务
python inference/client.py --local --model_path ./output_sft/final \
    --audio user.wav --tasks asr emotion gender age
```

### 7. 部署 HTTP 服务

```bash
python inference/server.py --model_path ./output_sft/final --port 9000

# 调用
curl -X POST http://localhost:9000/infer \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "user.wav", "context": "销售员上一句", "tasks": ["asr", "emotion"]}'

# 查看支持的任务
curl http://localhost:9000/tasks
```

## 扩展新任务

只需两步：

1. **在 `data/tasks.json` 添加任务定义：**
```json
{
  "purchase_intent": {
    "name": "购买意向识别",
    "system": "你是购买意向分析助手，根据用户语音判断购买意向。",
    "user_prompt": "请判断说话人的购买意向。输出JSON格式。",
    "output_type": "json",
    "annotation_fields": ["intent", "confidence"]
  }
}
```

2. **准备标注数据并构建：**
```bash
python data/build_sft_dataset.py \
    --tasks asr:ann_asr.jsonl:0.4 \
            emotion:ann_emotion.jsonl:0.15 \
            gender:ann_gender.jsonl:0.15 \
            age:ann_age.jsonl:0.15 \
            purchase_intent:ann_intent.jsonl:0.15 \
    --output train_sft.jsonl
```

无需改模型代码、训练代码或推理代码。
