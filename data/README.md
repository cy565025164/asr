# 数据格式说明

## 设计原则

**一个基座模型，一份训练数据，每条标注就是一个任务，通过 prompt 区分。**

---

## 标注文件格式

每条标注包含 `task` 字段指定任务类型，加对应任务的标签字段：

**ASR 任务：**
```json
{"audio_path": "/data/audio/call_001.wav", "sales_context": "请问您对保险感兴趣吗", "task": "asr", "text": "嗯，我想了解一下百万医疗险"}
```

**情绪识别：**
```json
{"audio_path": "/data/audio/call_001.wav", "sales_context": "请问您对保险感兴趣吗", "task": "emotion", "emotion": "积极", "emotion_intensity": "中"}
```

**性别识别：**
```json
{"audio_path": "/data/audio/call_001.wav", "task": "gender", "gender": "女"}
```

**年龄识别：**
```json
{"audio_path": "/data/audio/call_001.wav", "task": "age", "age_group": "中年"}
```

所有任务混在同一个 `.jsonl` 文件里。

---

## 构建训练数据

```bash
# SFT
python data/build_dataset.py sft --annotations annotations.jsonl --output train.jsonl

# DPO
python data/build_dataset.py dpo --annotations annotations.jsonl --output train_dpo.jsonl
```
