"""
构建 DPO 训练数据集
使用 Qwen3-Omni 模型推理生成 rejected，或文本扰动
"""

import json
import random
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def perturb_text(text: str) -> str:
    """通过扰动生成错误样本"""
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


def build_dpo_item(ann: dict, rejected_text: str) -> dict:
    """构建一条 DPO 数据"""
    sales_ctx = ann.get("sales_context", "")
    system_msg = "你是电销场景语音识别助手。"
    if sales_ctx:
        system_msg += f"销售员上一句：{sales_ctx}"

    return {
        "messages_prefix": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [
                {"type": "audio", "audio": ann["audio_path"]},
                {"type": "text", "text": "请准确识别用户语音内容。"},
            ]},
        ],
        "chosen": ann["correct_text"],
        "rejected": rejected_text,
    }


def build_with_model(annotations, model_path):
    """用 Qwen3-Omni 推理生成 rejected"""
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    logger.info(f"Loading model from {model_path}")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
    )
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    dpo_data = []
    for i, ann in enumerate(annotations):
        sales_ctx = ann.get("sales_context", "")
        messages = [
            {"role": "system", "content": f"你是电销场景语音识别助手。销售员上一句：{sales_ctx}" if sales_ctx else "你是语音识别助手。"},
            {"role": "user", "content": [
                {"type": "audio", "audio": ann["audio_path"]},
                {"type": "text", "text": "请准确识别用户语音内容。"},
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

        rejected = output if output.strip() != ann["correct_text"].strip() else perturb_text(ann["correct_text"])
        if rejected.strip() == ann["correct_text"].strip():
            continue

        dpo_data.append(build_dpo_item(ann, rejected))

        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{len(annotations)}")

    return dpo_data


def main():
    p = argparse.ArgumentParser("构建 DPO 数据集")
    p.add_argument("--annotation_file", type=str, required=True)
    p.add_argument("--output_file", type=str, default="train_dpo.jsonl")
    p.add_argument("--model_path", type=str, default=None, help="模型路径，不传则用文本扰动")
    args = p.parse_args()

    annotations = []
    with open(args.annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    logger.info(f"Loaded {len(annotations)} annotations")

    if args.model_path:
        dpo_data = build_with_model(annotations, args.model_path)
    else:
        dpo_data = []
        for ann in annotations:
            rejected = perturb_text(ann["correct_text"])
            if rejected.strip() != ann["correct_text"].strip():
                dpo_data.append(build_dpo_item(ann, rejected))

    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Built {len(dpo_data)} DPO examples, saved to {args.output_file}")


if __name__ == "__main__":
    main()
