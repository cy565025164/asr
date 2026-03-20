"""
DPO 数据集构建工具

功能：
1. 从电销对话录音中提取用户话术片段
2. 用 Qwen3-ASR 推理生成 rejected 样本（模型原始输出作为错误样本）
3. 结合人工标注的正确转录作为 chosen 样本
4. 输出标准 DPO JSONL 格式

用法：
  python build_dpo_dataset.py \
      --input_dir /data/calls/ \
      --annotation_file /data/annotations.jsonl \
      --output_file /data/dpo_train.jsonl \
      --model_path Qwen/Qwen3-ASR-1.7B
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Optional

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_annotations(annotation_file: str) -> List[dict]:
    """
    加载标注文件

    标注文件格式 (JSONL):
    {
        "call_id": "call_001",
        "turn_id": 3,
        "audio_path": "/data/audio/call_001_user_03.wav",
        "correct_text": "嗯，我想了解一下你们那个百万医疗险",
        "dialogue_history": [
            {"role": "sales", "text": "您好，我是XX保险的客服小王"},
            {"role": "user", "text": "嗯你好"},
            {"role": "sales", "text": "请问您对我们的保险产品感兴趣吗？"}
        ],
        "sales_context": "请问您对我们的保险产品感兴趣吗？"
    }
    """
    annotations = []
    with open(annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))
    logger.info(f"Loaded {len(annotations)} annotations")
    return annotations


def generate_rejected_with_model(
    model,
    audio_paths: List[str],
    batch_size: int = 16,
) -> List[str]:
    """
    用 Qwen3-ASR 模型推理，生成 rejected 样本
    模型的原始输出（未经修正）作为错误样本
    """
    results = []
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]
        batch_results = model.transcribe(
            audio=batch_paths,
            language="Chinese",
        )
        for r in batch_results:
            results.append(r.text)
        logger.info(f"  Processed {min(i + batch_size, len(audio_paths))}/{len(audio_paths)}")
    return results


def generate_rejected_by_perturbation(
    correct_text: str,
    perturbation_type: str = "random",
) -> str:
    """
    通过扰动正确文本生成 rejected 样本（备选方案）

    扰动类型：
    - remove_punct: 去除标点
    - swap_chars: 交换相邻字符
    - drop_chars: 随机删除字符
    - homophone: 同音字替换（需要同音字典）
    """
    import random

    if perturbation_type == "remove_punct":
        # 去除标点符号
        puncts = "，。！？、；：""''（）《》【】"
        return "".join(c for c in correct_text if c not in puncts)

    elif perturbation_type == "swap_chars":
        chars = list(correct_text)
        if len(chars) >= 2:
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    elif perturbation_type == "drop_chars":
        chars = list(correct_text)
        if len(chars) >= 3:
            idx = random.randint(0, len(chars) - 1)
            chars.pop(idx)
        return "".join(chars)

    else:
        # 混合扰动
        methods = ["remove_punct", "swap_chars", "drop_chars"]
        return generate_rejected_by_perturbation(
            correct_text, random.choice(methods)
        )


def build_dpo_dataset(
    annotations: List[dict],
    model=None,
    use_model_rejected: bool = True,
    output_file: str = "dpo_train.jsonl",
):
    """构建 DPO 数据集"""
    dpo_data = []

    if use_model_rejected and model is not None:
        # 批量推理获取 rejected
        audio_paths = [a["audio_path"] for a in annotations]
        logger.info("Generating rejected samples with model...")
        model_outputs = generate_rejected_with_model(model, audio_paths)
    else:
        model_outputs = None

    for idx, ann in enumerate(annotations):
        chosen = ann["correct_text"]

        # 获取 rejected
        if model_outputs is not None:
            rejected = model_outputs[idx]
            # 如果模型输出和正确文本完全一样，用扰动方法生成
            if rejected.strip() == chosen.strip():
                rejected = generate_rejected_by_perturbation(chosen)
        else:
            rejected = generate_rejected_by_perturbation(chosen)

        # 跳过 chosen == rejected 的样本
        if chosen.strip() == rejected.strip():
            continue

        dpo_item = {
            "audio_path": ann["audio_path"],
            "sales_context": ann.get("sales_context", ""),
            "chosen": chosen,
            "rejected": rejected,
        }

        # 添加对话历史（如果有）
        if "dialogue_history" in ann:
            dpo_item["dialogue_history"] = ann["dialogue_history"]

        dpo_data.append(dpo_item)

    # 写入文件
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Built {len(dpo_data)} DPO examples, saved to {output_file}")
    return dpo_data


def main():
    parser = argparse.ArgumentParser(description="Build DPO dataset for Qwen3-ASR")
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="dpo_train.jsonl")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Qwen3-ASR model path for generating rejected samples")
    parser.add_argument("--use_perturbation", action="store_true",
                        help="Use text perturbation instead of model inference")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    annotations = load_annotations(args.annotation_file)

    model = None
    use_model = not args.use_perturbation and args.model_path is not None
    if use_model:
        logger.info(f"Loading ASR model from {args.model_path}")
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )

    build_dpo_dataset(
        annotations=annotations,
        model=model,
        use_model_rejected=use_model,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
