"""
多任务 DPO 数据集构建

当前主要针对 ASR 任务（文本扰动或模型推理生成 rejected）。
其他任务可扩展。

用法：
    # 文本扰动生成 rejected
    python build_dpo_dataset.py --task asr --annotation_file annotations_asr.jsonl --output train_dpo.jsonl

    # 用模型推理生成 rejected
    python build_dpo_dataset.py --task asr --annotation_file annotations_asr.jsonl \
        --model_path Qwen/Qwen3-Omni-30B-A3B-Instruct --output train_dpo.jsonl
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


# ============ 文本扰动 ============

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


# ============ 构建 DPO 项 ============

def build_dpo_item(task_id: str, task_cfg: dict, ann: dict, rejected: str) -> dict:
    system_text = task_cfg["system"]
    sales_ctx = ann.get("sales_context", "")
    if sales_ctx:
        system_text += f"\n销售员上一句：{sales_ctx}"

    # chosen
    if task_cfg["output_type"] == "text":
        chosen = ann["text"]
    else:
        output = {f: ann[f] for f in task_cfg["annotation_fields"] if f in ann}
        chosen = json.dumps(output, ensure_ascii=False)

    return {
        "task": task_id,
        "messages_prefix": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [
                {"type": "audio", "audio": ann["audio_path"]},
                {"type": "text", "text": task_cfg["user_prompt"]},
            ]},
        ],
        "chosen": chosen,
        "rejected": rejected,
    }


def build_with_model(task_id, task_cfg, annotations, model_path):
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

    results = []
    for i, ann in enumerate(annotations):
        system_text = task_cfg["system"]
        sales_ctx = ann.get("sales_context", "")
        if sales_ctx:
            system_text += f"\n销售员上一句：{sales_ctx}"

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [
                {"type": "audio", "audio": ann["audio_path"]},
                {"type": "text", "text": task_cfg["user_prompt"]},
            ]},
        ]

        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs = processor(text=text, audio=audios, images=images, videos=videos,
                          return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)

        with torch.no_grad():
            text_ids, _ = model.generate(**inputs, return_audio=False,
                                         thinker_return_dict_in_generate=True, max_new_tokens=256)
        output = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)[0].strip()

        # chosen text
        if task_cfg["output_type"] == "text":
            chosen = ann["text"]
        else:
            chosen = json.dumps({f: ann[f] for f in task_cfg["annotation_fields"] if f in ann}, ensure_ascii=False)

        rejected = output if output.strip() != chosen.strip() else perturb_text(chosen)
        if rejected.strip() != chosen.strip():
            results.append(build_dpo_item(task_id, task_cfg, ann, rejected))

        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{len(annotations)}")

    return results


def main():
    p = argparse.ArgumentParser("多任务 DPO 数据集构建")
    p.add_argument("--task", type=str, required=True, help="任务ID: asr, emotion, gender, age")
    p.add_argument("--annotation_file", type=str, required=True)
    p.add_argument("--output", type=str, default="train_dpo.jsonl")
    p.add_argument("--model_path", type=str, default=None, help="模型路径，不传用文本扰动")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    tasks_config = load_tasks_config()

    if args.task not in tasks_config:
        raise ValueError(f"未知任务: {args.task}，可用: {list(tasks_config.keys())}")
    task_cfg = tasks_config[args.task]

    annotations = []
    with open(args.annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    logger.info(f"Loaded {len(annotations)} annotations for task={args.task}")

    if args.model_path:
        dpo_data = build_with_model(args.task, task_cfg, annotations, args.model_path)
    else:
        dpo_data = []
        for ann in annotations:
            if task_cfg["output_type"] == "text":
                chosen = ann["text"]
            else:
                chosen = json.dumps({f: ann[f] for f in task_cfg["annotation_fields"] if f in ann}, ensure_ascii=False)

            rejected = perturb_text(chosen)
            if rejected.strip() != chosen.strip():
                dpo_data.append(build_dpo_item(args.task, task_cfg, ann, rejected))

    with open(args.output, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Built {len(dpo_data)} DPO examples → {args.output}")


if __name__ == "__main__":
    main()
