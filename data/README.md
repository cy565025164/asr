# 数据格式说明

## 设计原则

**一个基座模型，一份训练数据，每条标注就是一个任务，通过 prompt 区分。**

---

## 标注数据目录结构

标注数据放在项目根目录 `datasets/` 下，每个子文件夹包含 `训练数据.csv` 和 `audios/` 目录：

```
datasets/
├── batch_001/
│   ├── 训练数据.csv
│   ├── 其他.csv          # 会被忽略，只读 训练数据.csv
│   └── audios/
│       ├── 20034766766462853315/
│       │   ├── 20034766766462853315_0_1.wav
│       │   └── ...
│       └── 20034766776285388800/
│           └── ...
├── batch_002/
│   ├── 训练数据.csv
│   └── audios/
│       └── ...
└── ...
```

### CSV 字段说明

| 字段 | 说明 | 示例 |
|------|------|------|
| `audio_id` | 录音文件夹名称 | `20034766766462853315` |
| `audio_name` | 音频文件名 | `20034766766462853315_0_1.wav` |
| `text` | 人工标注文本（ASR chosen） | `不是，我在想没电了呢` |
| `文本拿不准` | 标注是否确定 | `否` |
| `性别` | 说话人性别（gender chosen） | `女` |
| `context` | 销售员上一句话术（可为空） | `您好，请问是张先生吗` |
| `model_text` | 模型识别文本（ASR rejected） | `嘿，哎，家不是，我在想没电了呢...` |

音频路径自动拼接为：`{子文件夹}/audios/{audio_id}/{audio_name}`

---

## 构建训练数据

### 默认从 datasets/ 读取

```bash
# SFT - 只生成 ASR
python data/build_dataset.py sft --task asr --output train_asr.jsonl

# SFT - 只生成性别
python data/build_dataset.py sft --task gender --output train_gender.jsonl

# DPO - ASR（优先用 model_text 作 rejected，缺失时文本扰动）
python data/build_dataset.py dpo --task asr --output dpo_asr.jsonl

# DPO - 性别（取反作 rejected：男↔女）
python data/build_dataset.py dpo --task gender --output dpo_gender.jsonl

# 不加 --task 则全部任务一起生成
python data/build_dataset.py sft --output train.jsonl
```

### 指定其他数据目录

```bash
python data/build_dataset.py sft --data_dir /path/to/other_data --output train.jsonl
```

### 兼容旧格式 jsonl

```bash
python data/build_dataset.py sft --annotations annotations.jsonl --output train.jsonl
```

---

## DPO Rejected 生成策略

### ASR 任务
优先级从高到低：
1. **model_text**：CSV 中模型识别的原始输出，最接近真实错误分布
2. **模型推理**：用 `--model_path` 指定 SFT 模型重新推理生成
3. **文本扰动**：随机组合以下方式生成坏样本
   - 随机去掉部分标点符号（约 50% 概率保留每个标点）
   - 相邻字符交换
   - 随机删字

### 性别任务
- 直接取反：男→女，女→男

当 chosen 和 rejected 相同时，自动 fallback 到文本扰动兜底，尽量不丢弃数据。

---

## 任务配置

任务定义见 `data/tasks.json`，目前支持：

| 任务 | 说明 | 输出类型 |
|------|------|----------|
| `asr` | 语音识别 | text |
| `emotion` | 情绪识别 | json |
| `gender` | 性别识别 | json |
| `age` | 年龄段识别 | json |
