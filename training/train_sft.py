# coding=utf-8
"""
Qwen3-Omni SFT 训练脚本

基于 Qwen3OmniMoeForConditionalGeneration 进行 SFT 微调。
数据格式：标准 Qwen3-Omni chat messages（支持 audio + text 多模态输入）。
训练时禁用 talker（只训 thinker，节省显存）。

用法：
  单卡: python train_sft.py --model_path Qwen/Qwen3-Omni-30B-A3B-Instruct --train_file train_sft.jsonl
  多卡: torchrun --nproc_per_node=N train_sft.py ...
"""

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from qwen_omni_utils import process_mm_info


# ============================================================
# 工具函数
# ============================================================

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step, best_path = None, None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step, best_path = step, path
    return best_path


def copy_config_files(src_dir: str, dst_dir: str):
    """复制推理所需的配置文件"""
    os.makedirs(dst_dir, exist_ok=True)
    for fn in [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "processor_config.json", "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json",
    ]:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


# ============================================================
# 数据预处理
# ============================================================

def preprocess_example(ex: Dict, processor: Qwen3OmniMoeProcessor) -> Dict:
    """预处理单条数据：构建 prefix_text 和 target"""
    messages = ex["messages"]

    # 分离 prefix（system + user）和 target（assistant）
    prefix_msgs = [m for m in messages if m["role"] != "assistant"]
    target_text = ""
    for m in messages:
        if m["role"] == "assistant":
            target_text = m["content"] if isinstance(m["content"], str) else m["content"]
            break

    prefix_text = processor.apply_chat_template(
        [prefix_msgs], add_generation_prompt=True, tokenize=False
    )[0]

    return {
        "prefix_text": prefix_text,
        "target": target_text,
        "messages": messages,  # 保留原始 messages 用于提取音频
    }


@dataclass
class DataCollatorForOmniSFT:
    """
    数据整理器：处理多模态输入，构建 labels
    prefix 部分（system + user + audio）标为 -100
    """
    processor: Qwen3OmniMoeProcessor

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]
        messages_list = [f["messages"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

        # 提取音频
        all_audios = []
        for msgs in messages_list:
            audios, _, _ = process_mm_info(msgs, use_audio_in_video=False)
            all_audios.append(audios)

        # 处理完整序列
        full_inputs = self.processor(
            text=full_texts,
            audio=[a for audios in all_audios for a in (audios if audios else [])],
            return_tensors="pt", padding=True, truncation=False,
        )

        # 处理 prefix 序列（确定 label mask 长度）
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=[a for audios in all_audios for a in (audios if audios else [])],
            return_tensors="pt", padding=True, truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()

        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


# ============================================================
# Trainer
# ============================================================

class CastDtypeTrainer(Trainer):
    """确保浮点输入与模型 dtype 一致"""
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


class SaveConfigCallback(TrainerCallback):
    """确保每个 checkpoint 可直接推理"""
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        copy_config_files(self.base_model_path, ckpt_dir)
        return control


# ============================================================
# 主函数
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("Qwen3-Omni SFT Training")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./output_sft")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_acc", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    # 加载模型（禁用 talker 节省显存）
    print(f"[INFO] Loading model from {args.model_path}")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()

    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    # 启用 gradient checkpointing
    model.gradient_checkpointing_enable()

    # 加载数据
    raw_ds = load_dataset("json", data_files={
        "train": args.train_file,
        **({} if not args.eval_file else {"validation": args.eval_file}),
    })

    ds = raw_ds.map(
        lambda ex: preprocess_example(ex, processor),
        num_proc=1,
    )

    keep = {"prefix_text", "target", "messages"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForOmniSFT(processor=processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps" if args.eval_file else "no",
        eval_steps=args.save_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = CastDtypeTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[SaveConfigCallback(args.model_path)],
    )

    resume_from = (args.resume_from or "").strip()
    if not resume_from and args.resume == 1:
        resume_from = find_latest_checkpoint(args.output_dir) or ""

    if resume_from:
        print(f"[resume] {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    copy_config_files(args.model_path, final_dir)
    print(f"[done] Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
