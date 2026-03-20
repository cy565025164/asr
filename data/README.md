# 数据格式说明

## 设计原则

**一个基座模型，一份训练数据，通过 prompt 区分任务。**

所有任务的训练样本合并到同一个 `.jsonl`，模型根据 system prompt + user prompt 来执行不同任务。

---

## 标注文件格式

一条标注可以包含多个任务的标签，按需填写：

```json
{
  "audio_path": "/data/audio/call_001_user_02.wav",
  "sales_context": "请问您对我们的保险产品感兴趣吗？",
  "text": "嗯，我想了解一下你们那个百万医疗险",
  "emotion": "积极",
  "emotion_intensity": "中",
  "gender": "女",
  "age_group": "中年"
}
```

- `text` → ASR 任务标签（必填）
- `emotion` / `emotion_intensity` → 情绪任务标签（选填）
- `gender` → 性别任务标签（选填）
- `age_group` → 年龄任务标签（选填）
- 有哪个标签就生成哪个任务的训练数据

---

## 构建后的训练数据 (train.jsonl)

`build_dataset.py` 会把一条标注展开为多条训练样本（按标签有无）：

```jsonl
{"task":"asr","messages":[{"role":"system","content":"你是语音识别助手...销售员上一句：..."},{"role":"user","content":[{"type":"audio","audio":"..."},{"type":"text","text":"请准确识别语音内容。"}]},{"role":"assistant","content":"嗯，我想了解一下你们那个百万医疗险"}]}
{"task":"emotion","messages":[{"role":"system","content":"你是语音情绪分析助手..."},{"role":"user","content":[{"type":"audio","audio":"..."},{"type":"text","text":"请判断说话人的情绪状态。"}]},{"role":"assistant","content":"{\"emotion\":\"积极\",\"intensity\":\"中\"}"}]}
{"task":"gender","messages":[...]}
{"task":"age","messages":[...]}
```

全部混在一个文件里，直接喂给 `train_sft.py`。

---

## DPO 数据 (train_dpo.jsonl)

同样一个文件，`task` 字段标记任务类型：

```json
{"task":"asr","messages_prefix":[...],"chosen":"嗯，我想了解一下百万医疗险","rejected":"嗯我想了解一下百万医疗险"}
```

---

## 用法

```bash
# 从标注文件构建 SFT 数据（一个 jsonl 出来）
python data/build_dataset.py sft \
    --annotations annotations.jsonl \
    --output train.jsonl

# 构建 DPO 数据
python data/build_dataset.py dpo \
    --annotations annotations.jsonl \
    --output train_dpo.jsonl

# 训练
python training/train_sft.py --train_file train.jsonl ...
```
