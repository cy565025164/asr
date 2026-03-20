"""
构建 DPO 训练数据集

方式 A：用 Qwen3-ASR 模型推理生成 rejected（模型错误输出）
方式 B：通过文本扰动生成 rejected（去标点、交换字符等）

标注文件格式 (JSONL):
{"audio_path": "...", "correct_text": "...", "sales_context": "...", "language": "Chinese"}

输出格式:
{"audio": "...", "prompt": "销售员：...", "chosen": "language Chinese<asr_text>...", "rejected": "language Chinese<asr_text>..."}
"""

import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def perturb_text(text: str) -> str:
    """通过扰动生成错误样本"""
    methods = [_remove_punct, _swap_chars, _drop_char]
    # 随机应用 1-2 种扰动
    n = random.randint(1, 2)
    result = text
    for fn in random.sample(methods, min(n, len(methods))):
        result = fn(result)
    return result


def _remove_punct(text: str) -> str:
    """去除标点"""
    puncts = "，。！？、；：""''（）《》【】,."
    return "".join(c for c in text if c not in puncts)


def _swap_chars(text: str) -> str:
    """交换相邻字符"""
    chars = list(text)
    if len(chars) >= 4:
        idx = random.randint(1, len(chars) - 3)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def _drop_char(text: str) -> str:
    """随机删除一个字符"""
    chars = list(text)
    if len(chars) >= 4:
        idx = random.randint(1, len(chars) - 2)
        chars.pop(idx)
    return "".join(chars)


def build_with_model(
    annotations: List[dict],
    model_path: str,
    language: str = "Chinese",
) -> List[dict]:
    """用模型推理生成 rejected 样本"""
    import torch
    from qwen_asr import Qwen3ASRModel

    logger.info(f"Loading model from {model_path}")
    model = Qwen3ASRModel.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0",
        max_inference_batch_size=32, max_new_tokens=256,
    )

    # 批量推理
    audio_paths = [a["audio_path"] for a in annotations]
    contexts = [a.get("sales_context", "") for a in annotations]
    # 将销售员话术直接作为 context 传入
    contexts = [f"销售员：{c}" if c else "" for c in contexts]

    logger.info(f"Running model inference on {len(audio_paths)} samples...")
    batch_size = 32
    model_outputs = []
    for i in range(0, len(audio_paths), batch_size):
        batch_audio = audio_paths[i:i + batch_size]
        batch_ctx = contexts[i:i + batch_size]
        results = model.transcribe(
            audio=batch_audio,
            context=batch_ctx,
            language=language,
        )
        model_outputs.extend([r.text for r in results])
        logger.info(f"  {min(i + batch_size, len(audio_paths))}/{len(audio_paths)}")

    dpo_data = []
    for ann, model_text in zip(annotations, model_outputs):
        correct = ann["correct_text"]
        lang = ann.get("language", language)

        # 如果模型输出和正确文本一样，用扰动方法
        rejected_text = model_text if model_text.strip() != correct.strip() else perturb_text(correct)

        if rejected_text.strip() == correct.strip():
            continue

        dpo_data.append({
            "audio": ann["audio_path"],
            "prompt": f"销售员：{ann.get('sales_context', '')}" if ann.get("sales_context") else "",
            "chosen": f"language {lang}<asr_text>{correct}",
            "rejected": f"language {lang}<asr_text>{rejected_text}",
        })
    return dpo_data


def build_with_perturbation(annotations: List[dict], language: str = "Chinese") -> List[dict]:
    """用文本扰动生成 rejected 样本"""
    dpo_data = []
    for ann in annotations:
        correct = ann["correct_text"]
        lang = ann.get("language", language)
        rejected = perturb_text(correct)

        if rejected.strip() == correct.strip():
            continue

        dpo_data.append({
            "audio": ann["audio_path"],
            "prompt": f"销售员：{ann.get('sales_context', '')}" if ann.get("sales_context") else "",
            "chosen": f"language {lang}<asr_text>{correct}",
            "rejected": f"language {lang}<asr_text>{rejected}",
        })
    return dpo_data


def main():
    parser = argparse.ArgumentParser(description="构建 DPO 数据集")
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="train_dpo.jsonl")
    parser.add_argument("--model_path", type=str, default=None,
                        help="ASR 模型路径，用于生成 rejected（不传则用扰动方式）")
    parser.add_argument("--language", type=str, default="Chinese")
    args = parser.parse_args()

    # 加载标注
    annotations = []
    with open(args.annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    logger.info(f"Loaded {len(annotations)} annotations")

    if args.model_path:
        dpo_data = build_with_model(annotations, args.model_path, args.language)
    else:
        dpo_data = build_with_perturbation(annotations, args.language)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Built {len(dpo_data)} DPO examples, saved to {args.output_file}")


if __name__ == "__main__":
    main()
