# 数据格式说明

## SFT 数据格式

对齐 Qwen3-ASR 官方格式，每行一个 JSON：

```json
{"audio": "/data/audio/call_001_user_03.wav", "text": "language Chinese<asr_text>嗯，我想了解一下你们那个百万医疗险", "prompt": "销售员：请问您对我们的保险产品感兴趣吗？"}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `audio` | ✅ | 用户语音文件路径（wav/mp3/flac） |
| `text` | ✅ | 转录标签，格式：`language {语言}<asr_text>{转录文本}` |
| `prompt` | 可选 | 销售员上一句话术，作为 system message 注入模型 |

### text 字段格式
- 有语言标注：`language Chinese<asr_text>转录文本`
- 无语言标注：`language None<asr_text>转录文本`

## DPO 数据格式

```json
{"audio": "/data/audio/call_001_user_03.wav", "prompt": "销售员：请问您对我们的保险产品感兴趣吗？", "chosen": "language Chinese<asr_text>嗯，我想了解一下百万医疗险", "rejected": "language Chinese<asr_text>嗯我想了解一下百万医疗"}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `audio` | ✅ | 用户语音文件路径 |
| `prompt` | 可选 | 销售员上一句话术 |
| `chosen` | ✅ | 正确转录（人工标注） |
| `rejected` | ✅ | 错误转录（模型输出 / 扰动生成） |

## 标注文件格式（用于构建数据集）

```json
{"call_id": "call_001", "turn_id": 3, "audio_path": "/data/audio/call_001_user_03.wav", "correct_text": "嗯，我想了解一下你们那个百万医疗险", "sales_context": "请问您对我们的保险产品感兴趣吗？", "language": "Chinese"}
```
