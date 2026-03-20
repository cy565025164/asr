# 数据格式说明

## 设计原则

**一个基座模型，多个独立任务，每个任务独立标注、独立数据集。**

```
Qwen3-Omni-30B-A3B-Instruct（基座）
  ├── 任务1: ASR 语音识别
  ├── 任务2: 情绪识别
  ├── 任务3: 性别识别
  ├── 任务4: 年龄段识别
  └── 任务N: 未来新增...
```

训练时多任务数据混合，但**各任务数据独立准备、独立标注**。
推理时客户端**按需选择**要执行的任务组合。

---

## 任务定义 (tasks.json)

每个任务有独立的 task_id、system prompt、输出格式：

```json
{
  "asr": {
    "name": "语音识别",
    "system": "你是语音识别助手，请准确转录用户语音内容。",
    "user_prompt": "请准确识别语音内容。",
    "output_format": "纯文本转录"
  },
  "emotion": {
    "name": "情绪识别",
    "system": "你是语音情绪分析助手，根据说话人的语气、语调、语速判断情绪。",
    "user_prompt": "请判断说话人的情绪状态。",
    "output_format": "JSON: {emotion, intensity, confidence}"
  },
  "gender": {
    "name": "性别识别",
    "system": "你是语音性别识别助手，根据声音特征判断说话人性别。",
    "user_prompt": "请判断说话人的性别。",
    "output_format": "JSON: {gender, confidence}"
  },
  "age": {
    "name": "年龄段识别",
    "system": "你是语音年龄识别助手，根据声音特征判断说话人的大致年龄段。",
    "user_prompt": "请判断说话人的年龄段。",
    "output_format": "JSON: {age_group, confidence}"
  }
}
```

---

## SFT 数据格式

### 每个任务独立一个标注文件

**ASR 标注** (`annotations_asr.jsonl`):
```json
{"audio_path": "/data/audio/call_001_user_01.wav", "sales_context": "您好请问方便接听吗", "text": "嗯你好，我想了解一下保险"}
```

**情绪标注** (`annotations_emotion.jsonl`):
```json
{"audio_path": "/data/audio/call_001_user_01.wav", "sales_context": "您好请问方便接听吗", "emotion": "平静", "intensity": "低", "confidence": 0.9}
```

**性别标注** (`annotations_gender.jsonl`):
```json
{"audio_path": "/data/audio/call_001_user_01.wav", "gender": "女", "confidence": 0.95}
```

**年龄标注** (`annotations_age.jsonl`):
```json
{"audio_path": "/data/audio/call_001_user_01.wav", "age_group": "中年", "confidence": 0.8}
```

### 构建后的 SFT 训练数据

每条标注转为独立的 chat message：

```json
{"task": "asr", "messages": [
  {"role": "system", "content": "你是语音识别助手，请准确转录用户语音内容。销售员上一句：您好请问方便接听吗"},
  {"role": "user", "content": [{"type": "audio", "audio": "/data/audio/call_001_user_01.wav"}, {"type": "text", "text": "请准确识别语音内容。"}]},
  {"role": "assistant", "content": "嗯你好，我想了解一下保险"}
]}
```

```json
{"task": "emotion", "messages": [
  {"role": "system", "content": "你是语音情绪分析助手，根据说话人的语气、语调、语速判断情绪。销售员上一句：您好请问方便接听吗"},
  {"role": "user", "content": [{"type": "audio", "audio": "/data/audio/call_001_user_01.wav"}, {"type": "text", "text": "请判断说话人的情绪状态。"}]},
  {"role": "assistant", "content": "{\"emotion\":\"平静\",\"intensity\":\"低\",\"confidence\":0.9}"}
]}
```

---

## DPO 数据格式

同样按任务独立：

```json
{"task": "asr", "messages_prefix": [...], "chosen": "嗯你好，我想了解一下保险", "rejected": "嗯你好我想了解一下保险"}
```

---

## 训练数据混合

`build_sft_dataset.py` 将各任务数据合并，支持设置混合比例：

```bash
python data/build_sft_dataset.py \
    --tasks asr:annotations_asr.jsonl:0.5 \
            emotion:annotations_emotion.jsonl:0.2 \
            gender:annotations_gender.jsonl:0.15 \
            age:annotations_age.jsonl:0.15 \
    --output train_sft.jsonl
```

---

## 推理：客户端按需选择

```python
# 只要 ASR
result = client.infer(audio, context="...", tasks=["asr"])

# ASR + 情绪
result = client.infer(audio, context="...", tasks=["asr", "emotion"])

# 全部
result = client.infer(audio, context="...", tasks=["asr", "emotion", "gender", "age"])
```

返回：
```json
{
  "asr": {"text": "嗯你好，我想了解一下保险", "latency_ms": 85},
  "emotion": {"emotion": "平静", "intensity": "低", "confidence": 0.9, "latency_ms": 92},
  "gender": {"gender": "女", "confidence": 0.95, "latency_ms": 78},
  "age": {"age_group": "中年", "confidence": 0.8, "latency_ms": 80}
}
```
