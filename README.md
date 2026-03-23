# Qwen3-Omni 电销场景多任务后训练

基于 **Qwen3-Omni-30B-A3B-Instruct** 的电销场景多任务后训练方案。
同一基座模型，各任务数据独立，客户端按需组合。

## 架构

```
                         ┌─ ASR 语音识别
用户音频 → Qwen3-Omni ──┼─ 情绪识别
   +                     ├─ 性别识别
 销售上下文              ├─ 年龄段识别
                         └─ 任务N...（未来扩展）
```

**设计原则：**
- 基座模型唯一：Qwen3-Omni-30B-A3B-Instruct
- 各任务数据独立标注、独立构建
- 训练时混合、推理时按需选择
- 新增任务只需：加 `tasks.json` 定义 + 准备标注数据

## 项目结构

```
├── datasets/                             # 标注数据（gitignore）
│   ├── batch_001/
│   │   ├── 训练数据.csv
│   │   └── audios/{audio_id}/{audio_name}
│   └── batch_002/
│       └── ...
├── data/
│   ├── tasks.json                        # 任务定义（system prompt / 输出格式）
│   ├── build_dataset.py                  # 统一数据构建（SFT + DPO）
│   └── README.md                         # 数据格式详细说明
├── training/
│   ├── train_sft.py                      # SFT 训练
│   ├── train_dpo.py                      # DPO 训练
│   ├── ds_config_zero2.json
│   └── ds_config_zero3.json
├── deploy/
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

将标注文件夹放到 `datasets/` 下，每个子文件夹包含 `训练数据.csv` 和 `audios/` 目录。

CSV 字段：`audio_id`, `audio_name`, `text`(人工标注), `性别`, `context`(销售员话术), `model_text`(模型识别结果)

详见 [data/README.md](data/README.md)

### 3. 构建训练数据

```bash
# SFT - ASR
python data/build_dataset.py sft --task asr --output train_asr.jsonl

# SFT - 性别
python data/build_dataset.py sft --task gender --output train_gender.jsonl

# DPO - ASR（优先用 model_text 作 rejected，缺失时文本扰动）
python data/build_dataset.py dpo --task asr --output dpo_asr.jsonl

# DPO - 性别（取反作 rejected：男↔女）
python data/build_dataset.py dpo --task gender --output dpo_gender.jsonl

# 指定其他数据目录
python data/build_dataset.py sft --data_dir /path/to/other --task asr --output train.jsonl
```

### 4. SFT 训练

```bash
# 单卡
python training/train_sft.py \
    --model_path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --train_file train_asr.jsonl \
    --output_dir ./output_sft \
    --batch_size 1 --grad_acc 16 --lr 1e-5 --epochs 3

# 多卡 + DeepSpeed ZeRO-2
deepspeed --num_gpus=4 training/train_sft.py \
    --model_path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --train_file train_asr.jsonl \
    --output_dir ./output_sft \
    --deepspeed training/ds_config_zero2.json
```

### 5. DPO 训练

```bash
# 在 SFT 模型基础上进行 DPO 偏好对齐
python training/train_dpo.py \
    --model_path ./output_sft/final \
    --train_file dpo_asr.jsonl \
    --output_dir ./output_dpo \
    --beta 0.1 --lr 5e-7 --epochs 1

# 多卡 + DeepSpeed
deepspeed --num_gpus=4 training/train_dpo.py \
    --model_path ./output_sft/final \
    --train_file dpo_asr.jsonl \
    --output_dir ./output_dpo \
    --beta 0.1 \
    --deepspeed training/ds_config_zero2.json
```

DPO 训练参数说明：
- `--beta`：KL 散度惩罚系数（默认 0.1，越大越保守）
- `--loss_type`：损失函数类型（sigmoid / hinge / ipo）
- `--lr`：学习率（建议 5e-7，比 SFT 小一个数量级）
- `--epochs`：通常 1 轮即可

### 6. 推理

```bash
# 单任务 ASR
python inference/client.py --local --model_path ./output_dpo/final \
    --audio user.wav --context "销售员上一句" --tasks asr

# 多任务
python inference/client.py --local --model_path ./output_dpo/final \
    --audio user.wav --tasks asr emotion gender
```

### 7. 部署 HTTP 服务

```bash
python inference/server.py --model_path ./output_dpo/final --port 9000

curl -X POST http://localhost:9000/infer \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "user.wav", "context": "销售员上一句", "tasks": ["asr"]}'
```

## 训练流程

```
标注数据 CSV  ──→  build_dataset.py sft  ──→  train_sft.py  ──→  SFT 模型
     │                                                              │
     └─────────→  build_dataset.py dpo  ──→  train_dpo.py  ──→  DPO 模型（最终）
```

1. **SFT 阶段**：基座模型学会按指令格式输出（ASR 转录、性别判断等）
2. **DPO 阶段**：在 SFT 基础上用偏好对齐，让模型更倾向正确输出

## 扩展新任务

1. 在 `data/tasks.json` 添加任务定义
2. 准备标注数据
3. 构建 + 训练，无需改模型代码
