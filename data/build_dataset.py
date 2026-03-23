"""
统一数据集构建工具

支持从标注文件夹批量读取 CSV + 音频，构建 SFT / DPO 训练数据。

数据目录结构：
    data_dir/
    ├── batch_001/
    │   ├── *.csv          # 标注文件（audio_id, audio_name, text, 文本拿不准, 性别, context, model_text）
    │   └── audios/
    │       └── {audio_id}/
    │           └── {audio_name}
    ├── batch_002/
    │   ├── ...

用法：
    # 构建 SFT
    python build_dataset.py sft --data_dir /path/to/data --output train.jsonl

    # 构建 DPO（使用 CSV 中的 model_text 作为 rejected，缺失时文本扰动）
    python build_dataset.py dpo --data_dir /path/to/data --output train_dpo.jsonl

    # 构建 DPO（模型推理生成 rejected）
    python build_dataset.py dpo --data_dir /path/to/data --model_path ./output_sft/final --output train_dpo.jsonl
"""

import csv
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


# ============ 数据加载 ============

def load_annotations_from_dir(data_dir: str) -> list:
    """从数据目录批量读取所有子文件夹中的 CSV 标注"""
    data_path = Path(data_dir)
    annotations = []
    skipped = 0

    for subfolder in sorted(data_path.iterdir()):
        if not subfolder.is_dir():
            continue

        # 找 CSV 文件
        csv_files = list(subfolder.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV found in {subfolder.name}, skipping")
            continue

        audios_dir = subfolder / "audios"

        for csv_file in csv_files:
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    audio_id = row.get("audio_id", "").strip()
                    audio_name = row.get("audio_name", "").strip()
                    text = row.get("text", "").strip()

                    if not audio_id or not audio_name or not text:
                        skipped += 1
                        continue

                    audio_path = audios_dir / audio_id / audio_name
                    if not audio_path.exists():
                        skipped += 1
                        continue

                    ann = {
                        "task": "asr",
                        "audio_path": str(audio_path),
                        "text": text,
                        "sales_context": row.get("context", "").strip(),
                        "model_text": row.get("model_text", "").strip(),
                        "gender": row.get("性别", "").strip(),
                        "uncertain": row.get("文本拿不准", "").strip(),
                    }
                    annotations.append(ann)

    if skipped:
        logger.warning(f"Skipped {skipped} entries (missing fields or audio file)")
    logger.info(f"Loaded {len(annotations)} annotations from {data_dir}")
    return annotations


def load_annotations(args) -> list:
    """根据参数加载标注：支持 --data_dir（CSV文件夹）和 --annotations（jsonl，兼容旧格式）"""
    if hasattr(args, "data_dir") and args.data_dir:
        return load_annotations_from_dir(args.data_dir)

    annotations = []
    with open(args.annotations, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    logger.info(f"Loaded {len(annotations)} annotations from {args.annotations}")
    return annotations


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
    """随机去掉部分标点（而非全部），让 rejected 更贴近真实错误"""
    punct_set = set("，。！？、；：""''（）,.")
    return "".join(
        c for c in t
        if c not in punct_set or random.random() > 0.5
    )


def _swap_chars(t):
    chars = list(t)
    if len(chars) >= 4:
        i = random.randint(1, len(chars) - 3)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _drop_char(t):
    chars = list(t)
    if len(chars) >= 4:
        chars.pop(random.randint(1, len(chars) - 2))
    return "".join(chars)


def build_dpo(annotations, tasks_config):
    """构建 DPO 数据，优先用 CSV 中的 model_text 作为 rejected，缺失时文本扰动"""
    items = []
    from_model = 0
    from_perturb = 0

    for ann in annotations:
        task_id = ann.get("task")
        if not task_id or task_id not in tasks_config:
            continue
        task_cfg = tasks_config[task_id]
        chosen = build_assistant_output(task_id, task_cfg, ann)

        # 优先用标注数据中的 model_text 作为 rejected
        model_text = ann.get("model_text", "").strip()
        if model_text and model_text != chosen.strip():
            rejected = model_text
            from_model += 1
        else:
            rejected = perturb_text(chosen)
            from_perturb += 1

        # chosen == rejected 时用扰动兜底，尽量保留数据
        if rejected.strip() == chosen.strip():
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

    logger.info(f"DPO sources: {from_model} from model_text, {from_perturb} from perturbation")
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

def add_data_args(parser):
    """添加数据源参数（互斥：--data_dir 或 --annotations）"""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data_dir", help="标注数据根目录（含多个子文件夹，每个有 CSV + audios/）")
    group.add_argument("--annotations", help="标注 jsonl 文件（兼容旧格式）")


def main():
    p = argparse.ArgumentParser("统一数据集构建")
    sub = p.add_subparsers(dest="cmd")

    sft_p = sub.add_parser("sft", help="构建 SFT 数据")
    add_data_args(sft_p)
    sft_p.add_argument("--output", default="train.jsonl")
    sft_p.add_argument("--seed", type=int, default=42)

    dpo_p = sub.add_parser("dpo", help="构建 DPO 数据")
    add_data_args(dpo_p)
    dpo_p.add_argument("--output", default="train_dpo.jsonl")
    dpo_p.add_argument("--model_path", default=None)
    dpo_p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return

    random.seed(args.seed)
    tasks_config = load_tasks_config()
    annotations = load_annotations(args)

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
