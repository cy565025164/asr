# 数据格式说明

## SFT 数据格式

对齐 Qwen3-Omni chat template，每行一个 JSON：

```json
{
  "messages": [
    {"role": "system", "content": "你是电销场景语音识别助手。销售员上一句：请问您对保险产品感兴趣吗？"},
    {"role": "user", "content": [{"type": "audio", "audio": "/data/audio/user.wav"}, {"type": "text", "text": "请识别用户语音并分析说话人特征。"}]},
    {"role": "assistant", "content": "【ASR】嗯，我想了解一下你们那个百万医疗险\n【分析】性别：女 | 年龄段：中年 | 情绪：积极 | 购买意向：有兴趣"}
  ]
}
```

### 三种任务模式

| 模式 | user text | assistant 输出格式 |
|------|-----------|-------------------|
| ASR | 请准确识别用户语音内容。 | 嗯，我想了解百万医疗险 |
| 分析 | 请分析说话人的性别、年龄段、情绪和购买意向。 | {"gender":"女","age_group":"中年","emotion":"积极","purchase_intent":"有兴趣"} |
| 全量 | 请先识别用户语音内容，然后分析说话人特征。 | 【ASR】...\n【分析】... |

## DPO 数据格式

```json
{
  "messages_prefix": [
    {"role": "system", "content": "你是电销场景语音识别助手。销售员上一句：请问您对保险产品感兴趣吗？"},
    {"role": "user", "content": [{"type": "audio", "audio": "/data/audio/user.wav"}, {"type": "text", "text": "请准确识别用户语音内容。"}]}
  ],
  "chosen": "嗯，我想了解一下你们那个百万医疗险，就是住院能报销的那种",
  "rejected": "嗯我想了解一下你们那个百万医疗险就是住院能报销那种"
}
```

## 标注文件格式（用于构建数据集）

```json
{
  "audio_path": "/data/audio/call_001_user_03.wav",
  "correct_text": "嗯，我想了解一下你们那个百万医疗险",
  "sales_context": "请问您对我们的保险产品感兴趣吗？",
  "gender": "女",
  "age_group": "中年",
  "emotion": "积极",
  "purchase_intent": "有兴趣"
}
```
