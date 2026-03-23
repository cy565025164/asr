# coding=utf-8
"""
Qwen3-Omni DPO 训练脚本

在 SFT 基础上进行 DPO 偏好对齐。
数据格式：messages_prefix + chosen/rejected。

用法：
  python train_dpo.py --model_path ./output_sft/final --train_file train_dpo.jsonl --output_dir ./output_dpo
"""

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
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

def find_latest_checkpoint(output_dir):
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step, best_path = None, None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m: continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step, best_path = step, path
    return best_path

def copy_config_files(src, dst):
    os.makedirs(dst, exist_ok=True)
    for fn in ["config.json", "generation_config.json", "preprocessor_config.json",
                "processor_config.json", "tokenizer_config.json", "tokenizer.json",
                "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json"]:
        s = os.path.join(src, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst, fn))


# ============================================================
# 数据预处理
# ============================================================

def preprocess_dpo(ex, processor):
    """预处理 DPO 数据"""
    prefix_msgs = ex["messages_prefix"]
    prefix_text = processor.apply_chat_template(
        [prefix_msgs], add_generation_prompt=True, tokenize=False
    )[0]
    return {
        "prefix_text": prefix_text,
        "chosen": ex["chosen"],
        "rejected": ex["rejected"],
        "messages_prefix": prefix_msgs,
    }


@dataclass
class DataCollatorForDPO:
    """DPO 数据整理器：为 chosen/rejected 分别构建输入"""
    processor: Qwen3OmniMoeProcessor

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        prefix_texts = [f["prefix_text"] for f in features]
        chosen_targets = [f["chosen"] for f in features]
        rejected_targets = [f["rejected"] for f in features]
        msgs_list = [f["messages_prefix"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""

        # 提取音频
        all_audios = []
        for msgs in msgs_list:
            audios, _, _ = process_mm_info(msgs, use_audio_in_video=False)
            all_audios.append(audios)
        flat_audios = [a for audios in all_audios for a in (audios if audios else [])]

        # 构建 chosen 和 rejected 的完整序列
        chosen_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, chosen_targets)]
        rejected_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, rejected_targets)]

        chosen_inputs = self.processor(
            text=chosen_texts, audio=flat_audios,
            return_tensors="pt", padding=True, truncation=False,
        )
        rejected_inputs = self.processor(
            text=rejected_texts, audio=flat_audios,
            return_tensors="pt", padding=True, truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts, audio=flat_audios,
            return_tensors="pt", padding=True, truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        pad_id = self.processor.tokenizer.pad_token_id

        for inputs_dict in [chosen_inputs, rejected_inputs]:
            labels = inputs_dict["input_ids"].clone()
            for i, pl in enumerate(prefix_lens):
                labels[i, :pl] = -100
            if pad_id is not None:
                labels[labels == pad_id] = -100
            inputs_dict["labels"] = labels

        batch = {}
        for k, v in chosen_inputs.items():
            batch[f"chosen_{k}"] = v
        for k, v in rejected_inputs.items():
            batch[f"rejected_{k}"] = v
        return batch


# ============================================================
# DPO Trainer
# ============================================================

class DPOTrainer(Trainer):
    def __init__(self, ref_model=None, beta=0.1, loss_type="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.loss_type = loss_type
        if self.ref_model:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        dtype = getattr(self.model, "dtype", None)
        if dtype:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=dtype)
        return inputs

    def _logprobs(self, model, input_ids, attention_mask, labels, **extra):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, **extra)
        logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        log_probs = F.log_softmax(logits, dim=-1)
        per_token = torch.gather(log_probs, -1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        mask = (shift_labels != -100).float()
        return (per_token * mask).sum(-1)

    def _extract_kwargs(self, inputs, prefix):
        """提取 chosen_ 或 rejected_ 前缀的输入"""
        kw = {}
        for k, v in inputs.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                kw[new_k] = v
        return kw

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_kw = self._extract_kwargs(inputs, "chosen_")
        rejected_kw = self._extract_kwargs(inputs, "rejected_")

        p_chosen = self._logprobs(model, **chosen_kw)
        p_rejected = self._logprobs(model, **rejected_kw)

        with torch.no_grad():
            r_chosen = self._logprobs(self.ref_model, **chosen_kw)
            r_rejected = self._logprobs(self.ref_model, **rejected_kw)

        margins = self.beta * ((p_chosen - r_chosen) - (p_rejected - r_rejected))

        if self.loss_type == "sigmoid":
            loss = -F.logsigmoid(margins).mean()
        elif self.loss_type == "hinge":
            loss = torch.relu(1 - margins).mean()
        elif self.loss_type == "ipo":
            loss = ((margins - 1 / (2 * self.beta)) ** 2).mean()
        else:
            raise ValueError(f"Unknown loss: {self.loss_type}")

        if self.state.global_step % self.args.logging_steps == 0:
            acc = (p_chosen > p_rejected).float().mean().item()
            self.log({
                "dpo_loss": loss.item(),
                "rewards/margins": margins.mean().item(),
                "rewards/accuracies": acc,
            })

        return (loss, {"logits": p_chosen}) if return_outputs else loss


class SaveConfigCallback(TrainerCallback):
    def __init__(self, base_model_path):
        self.base_model_path = base_model_path
    def on_save(self, args, state, control, **kwargs):
        if args.process_index == 0:
            copy_config_files(self.base_model_path, os.path.join(args.output_dir, f"checkpoint-{state.global_step}"))
        return control


# ============================================================
# 主函数
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("Qwen3-Omni DPO Training")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./output_dpo")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--loss_type", type=str, default="sigmoid", choices=["sigmoid", "hinge", "ipo"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_acc", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Policy model
    print(f"[INFO] Loading policy model from {args.model_path}")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, device_map=None, attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)
    model.gradient_checkpointing_enable()

    # Reference model
    print(f"[INFO] Loading reference model")
    ref_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, device_map=None, attn_implementation="flash_attention_2",
    )
    ref_model.disable_talker()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # 数据
    raw_ds = load_dataset("json", data_files={
        "train": args.train_file,
        **({} if not args.eval_file else {"validation": args.eval_file}),
    })
    ds = raw_ds.map(lambda ex: preprocess_dpo(ex, processor), num_proc=1)
    keep = {"prefix_text", "chosen", "rejected", "messages_prefix"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        bf16=use_bf16, fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
    )

    trainer = DPOTrainer(
        ref_model=ref_model, beta=args.beta, loss_type=args.loss_type,
        model=model, args=training_args,
        train_dataset=ds["train"], eval_dataset=ds.get("validation"),
        data_collator=DataCollatorForDPO(processor=processor),
        tokenizer=processor.tokenizer,
        callbacks=[SaveConfigCallback(args.model_path)],
    )

    resume_from = (args.resume_from or "").strip()
    if not resume_from and args.resume == 1:
        resume_from = find_latest_checkpoint(args.output_dir) or ""
    trainer.train(resume_from_checkpoint=resume_from if resume_from else None)

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    copy_config_files(args.model_path, final_dir)
    print(f"[done] {final_dir}")


if __name__ == "__main__":
    main()
