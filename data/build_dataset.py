"""
统一数据集构建工具

每条标注自带 task 字段，直接构建对应任务的训练样本。
所有任务混在一个 jsonl 里，模型通过 prompt 区分。

用法：
    # 构建 SFT
    python build_dataset.py sft --annotations annotations.jsonl --output train.jsonl

    # 构建 DPO（文本扰动）
    python build_dataset.py dpo --annotations annotations.jsonl --output train_dpo.jsonl

    # 构建 DPO（模型推理生成 rejected）
    python build_dataset.py dpo --annotations annotations.jsonl --model_path ./output_sft/final --output train_dpo.jsonl
"""

import json
import random
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TASKS_FILE = Path(__file__).parent / "tasks.json"


def load_tasks_config():
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============ 构建训练样本 ============

def build_assistant_output(task_id: str, task_cfg: dict, ann: dict) -> str:
    """根据任务类型构建 assistant 回复"""
    if task_cfg["output_type"] == "text":
        return ann["text"]
    output = {}
    for field in task_cfg["annotation_fields"]:
        if field in ann:
            output[field] = ann[field]
    return json.dumps(output, ensure_ascii=False)


def build_messages(task_id: str, task_cfg: dict, ann: dict) -> list:
    """构建 chat messages"""
    system_text = task_cfg["system"]
    # 只有 ASR 等需要上下文的任务才注入销售员话术
    if task_cfg.get("use_context", False):
        sales_ctx = ann.get("sales_context", "")
        if sales_ctx:
            system_text += f"\n销售员上一句：{sales_ctx}"

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [
            {"type": "audio", "audio": ann["audio_path"]},
            {"type": "text", "text": task_cfg["user_prompt"]},
        ]},
        {"role": "assistant", "content": build_assistant_output(task_id, task_cfg, ann)},
    ]


# ============ SFT ============

def build_sft(annotations, tasks_config):
    items = []
    skipped = 0
    for ann in annotations:
        task_id = ann.get("task")
        if not task_id or task_id not in tasks_config:
            skipped += 1
            continue
        task_cfg = tasks_config[task_id]
        items.append({
            "task": task_id,
            "messages": build_messages(task_id, task_cfg, ann),
        })
    if skipped:
        logger.warning(f"Skipped {skipped} entries (missing/unknown task)")
    return items


# ============ DPO ============

def perturb_text(text: str) -> str:
    methods = [_remove_punct, _swap_chars, _drop_char]
    result = text
    for fn in random.sample(methods, min(random.randint(1, 2), len(methods))):
        result = fn(result)
    return result

def _remove_punct(t):
    return "".join(c for c in t if c not in "，。！？、；：""''（）,.")

def _swap_chars(t):
    chars = list(t)
    if len(chars) >= 4:
        i = random.randint(1, len(chars) - 3)
        chars[i], chars[i+1] = chars[i+1], chars[i]
    return "".join(chars)

def _drop_char(t):
    chars = list(t)
    if len(chars) >= 4:
        chars.pop(random.randint(1, len(chars) - 2))
    return "".join(chars)


def build_dpo(annotations, tasks_config):
    items = []
    for ann in annotations:
        task_id = ann.get("task")
        if not task_id or task_id not in tasks_config:
            continue
        task_cfg = tasks_config[task_id]
        chosen = build_assistant_output(task_id, task_cfg, ann)
        rejected = perturb_text(chosen)
        if rejected.strip() == chosen.strip():
            continue

        msgs = build_messages(task_id, task_cfg, ann)
        items.append({
            "task": task_id,
            "messages_prefix": msgs[:2],  # system + user
            "chosen": chosen,
            "rejected": rejected,
        })
    return items


def build_dpo_with_model(annotations, tasks_config, model_path):
    """用模型推理生成 rejected"""
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    logger.info(f"Loading model from {model_path}")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
    )
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    items = []
    valid = [(a, a["task"]) for a in annotations if a.get("task") in tasks_config]

    for i, (ann, task_id) in enumerate(valid):
        task_cfg = tasks_config[task_id]
        chosen = build_assistant_output(task_id, task_cfg, ann)
        msgs = build_messages(task_id, task_cfg, ann)
        prefix_msgs = msgs[:2]

        text = processor.apply_chat_template(prefix_msgs, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(prefix_msgs, use_audio_in_video=False)
        inputs = processor(text=text, audio=audios, images=images, videos=videos,
                          return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)

        with torch.no_grad():
            text_ids, _ = model.generate(**inputs, return_audio=False,
                                         thinker_return_dict_in_generate=True, max_new_tokens=256)
        output = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)[0].strip()

        rejected = output if output.strip() != chosen.strip() else perturb_text(chosen)
        if rejected.strip() != chosen.strip():
            items.append({
                "task": task_id,
                "messages_prefix": prefix_msgs,
                "chosen": chosen,
                "rejected": rejected,
            })

        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{len(valid)}")

    return items


# ============ 主入口 ============

def main():
    p = argparse.ArgumentParser("统一数据集构建")
    sub = p.add_subparsers(dest="cmd")

    sft_p = sub.add_parser("sft", help="构建 SFT 数据")
    sft_p.add_argument("--annotations", required=True)
    sft_p.add_argument("--output", default="train.jsonl")
    sft_p.add_argument("--seed", type=int, default=42)

    dpo_p = sub.add_parser("dpo", help="构建 DPO 数据")
    dpo_p.add_argument("--annotations", required=True)
    dpo_p.add_argument("--output", default="train_dpo.jsonl")
    dpo_p.add_argument("--model_path", default=None)
    dpo_p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return

    random.seed(args.seed)
    tasks_config = load_tasks_config()

    annotations = []
    with open(args.annotations, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    logger.info(f"Loaded {len(annotations)} annotations")

    if args.cmd == "sft":
        items = build_sft(annotations, tasks_config)
    elif args.cmd == "dpo":
        if args.model_path:
            items = build_dpo_with_model(annotations, tasks_config, args.model_path)
        else:
            items = build_dpo(annotations, tasks_config)

    random.shuffle(items)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    task_counts = {}
    for item in items:
        t = item["task"]
        task_counts[t] = task_counts.get(t, 0) + 1
    logger.info(f"Total: {len(items)} → {args.output}")
    for t, c in sorted(task_counts.items()):
        logger.info(f"  {t}: {c}")


if __name__ == "__main__":
    main()
