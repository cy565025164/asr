"""
多任务 SFT 数据集构建

各任务独立标注文件，合并为统一训练数据。
支持按比例混合。

用法：
    python build_sft_dataset.py \
        --tasks asr:annotations_asr.jsonl:0.5 \
                emotion:annotations_emotion.jsonl:0.2 \
                gender:annotations_gender.jsonl:0.15 \
                age:annotations_age.jsonl:0.15 \
        --output train_sft.jsonl

    # 也可以只构建单个任务
    python build_sft_dataset.py --tasks asr:annotations_asr.jsonl:1.0 --output train_asr.jsonl
"""

import json
import random
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 加载任务定义
TASKS_FILE = Path(__file__).parent / "tasks.json"


def load_tasks_config():
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sft_message(task_id: str, task_cfg: dict, ann: dict) -> dict:
    """根据任务定义和标注数据构建一条 SFT messages"""

    # system prompt: 任务定义 + 可选销售上下文
    system_text = task_cfg["system"]
    sales_ctx = ann.get("sales_context", "")
    if sales_ctx:
        system_text += f"\n销售员上一句：{sales_ctx}"

    # user content: audio + task prompt
    user_content = [
        {"type": "audio", "audio": ann["audio_path"]},
        {"type": "text", "text": task_cfg["user_prompt"]},
    ]

    # assistant 回复
    if task_cfg["output_type"] == "text":
        # ASR: 直接输出转录文本
        assistant_text = ann["text"]
    else:
        # JSON 输出: 从标注中提取对应字段
        output = {}
        for field in task_cfg["annotation_fields"]:
            if field in ann:
                output[field] = ann[field]
        assistant_text = json.dumps(output, ensure_ascii=False)

    return {
        "task": task_id,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_text},
        ],
    }


def main():
    p = argparse.ArgumentParser("多任务 SFT 数据集构建")
    p.add_argument(
        "--tasks", nargs="+", required=True,
        help="任务定义，格式 task_id:annotation_file:ratio，如 asr:ann_asr.jsonl:0.5",
    )
    p.add_argument("--output", type=str, default="train_sft.jsonl")
    p.add_argument("--max_samples", type=int, default=0, help="最大样本数，0=不限")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    tasks_config = load_tasks_config()

    # 解析任务参数
    task_specs = []
    for spec in args.tasks:
        parts = spec.split(":")
        if len(parts) == 3:
            tid, fpath, ratio = parts[0], parts[1], float(parts[2])
        elif len(parts) == 2:
            tid, fpath, ratio = parts[0], parts[1], 1.0
        else:
            raise ValueError(f"格式错误: {spec}，应为 task_id:file:ratio")

        if tid not in tasks_config:
            raise ValueError(f"未知任务: {tid}，可用: {list(tasks_config.keys())}")
        task_specs.append((tid, fpath, ratio))

    # 加载各任务标注并构建 SFT 数据
    all_items = []
    for tid, fpath, ratio in task_specs:
        annotations = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line))

        # 按比例采样
        if ratio < 1.0:
            n = max(1, int(len(annotations) * ratio))
            annotations = random.sample(annotations, min(n, len(annotations)))

        task_cfg = tasks_config[tid]
        for ann in annotations:
            item = build_sft_message(tid, task_cfg, ann)
            all_items.append(item)

        logger.info(f"  {tid}: {len(annotations)} samples")

    # 打乱
    random.shuffle(all_items)

    if args.max_samples > 0:
        all_items = all_items[:args.max_samples]

    with open(args.output, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计
    task_counts = {}
    for item in all_items:
        t = item["task"]
        task_counts[t] = task_counts.get(t, 0) + 1
    logger.info(f"Total: {len(all_items)} samples → {args.output}")
    for t, c in sorted(task_counts.items()):
        logger.info(f"  {t}: {c} ({c/len(all_items)*100:.1f}%)")


if __name__ == "__main__":
    main()
