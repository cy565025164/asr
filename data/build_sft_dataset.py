"""
从电销录音标注文件构建 SFT 训练数据集
输出格式对齐 Qwen3-Omni chat template
"""

import json
import random
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 三种任务 prompt
TASK_PROMPTS = {
    "asr": "请准确识别用户语音内容。",
    "analyze": "请分析说话人的性别、年龄段、情绪和购买意向。输出JSON格式。",
    "full": "请先识别用户语音内容，然后分析说话人特征。",
}


def build_sft_item(ann: dict, task: str) -> dict:
    """构建一条 SFT 数据"""
    sales_ctx = ann.get("sales_context", "")
    system_msg = f"你是电销场景语音识别与分析助手。"
    if sales_ctx:
        system_msg += f"销售员上一句：{sales_ctx}"

    user_content = [
        {"type": "audio", "audio": ann["audio_path"]},
        {"type": "text", "text": TASK_PROMPTS[task]},
    ]

    # 构建 assistant 回复
    if task == "asr":
        assistant = ann["correct_text"]
    elif task == "analyze":
        assistant = json.dumps({
            "gender": ann.get("gender", "未知"),
            "age_group": ann.get("age_group", "未知"),
            "emotion": ann.get("emotion", "平静"),
            "purchase_intent": ann.get("purchase_intent", "未知"),
        }, ensure_ascii=False)
    else:  # full
        analysis = f"性别：{ann.get('gender', '未知')} | 年龄段：{ann.get('age_group', '未知')} | 情绪：{ann.get('emotion', '平静')} | 购买意向：{ann.get('purchase_intent', '未知')}"
        assistant = f"【ASR】{ann['correct_text']}\n【分析】{analysis}"

    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant},
        ]
    }


def main():
    p = argparse.ArgumentParser("构建 SFT 数据集")
    p.add_argument("--annotation_file", type=str, required=True)
    p.add_argument("--output_file", type=str, default="train_sft.jsonl")
    p.add_argument("--task_ratio", type=str, default="asr:0.5,analyze:0.2,full:0.3",
                   help="任务比例，如 asr:0.5,analyze:0.2,full:0.3")
    args = p.parse_args()

    # 解析任务比例
    ratios = {}
    for item in args.task_ratio.split(","):
        task, ratio = item.split(":")
        ratios[task] = float(ratio)

    tasks = list(ratios.keys())
    weights = list(ratios.values())

    annotations = []
    with open(args.annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    logger.info(f"Loaded {len(annotations)} annotations")

    count = 0
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for ann in annotations:
            # 按比例分配任务
            task = random.choices(tasks, weights=weights, k=1)[0]

            # 分析任务需要有标注
            if task in ("analyze", "full") and not ann.get("gender"):
                task = "asr"

            sft_item = build_sft_item(ann, task)
            fout.write(json.dumps(sft_item, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Built {count} SFT examples, saved to {args.output_file}")


if __name__ == "__main__":
    main()
