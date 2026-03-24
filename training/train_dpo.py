# coding=utf-8
"""
Qwen3-Omni DPO 训练脚本

在 SFT 基础上进行 DPO 偏好对齐。
数据格式：messages_prefix + chosen/rejected。

方案：训练前预计算 ref logprobs，训练时只跑 policy model。
避免 DeepSpeed ZeRO-3 与独立 ref model 的通信冲突。

用法：
  python train_dpo.py --model_path ./output_sft/final --train_file train_dpo.jsonl --output_dir ./output_dpo
"""

import argparse
import gc
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
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step, best_path = step, path
    return best_path


def copy_config_files(src, dst):
    os.makedirs(dst, exist_ok=True)
    for fn in [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "processor_config.json", "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json",
    ]:
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


# ============================================================
# 预计算 ref logprobs
# ============================================================

def _compute_logprobs_single(thinker, input_ids, attention_mask, labels, extra):
    """对单条数据计算 sum log prob（只在有 label 的 token 上）"""
    with torch.no_grad():
        outputs = thinker(input_ids=input_ids, attention_mask=attention_mask, **extra)
    logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(logits, dim=-1)
    per_token = torch.gather(log_probs, -1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    return (per_token * mask).sum(-1).item()


def precompute_ref_logprobs(model_path, dtype, processor, dataset, max_length=2048):
    """
    加载 ref model，逐条计算 chosen/rejected 的 logprobs，然后释放模型。
    返回两个 list: ref_chosen_logprobs, ref_rejected_logprobs
    """
    print("[INFO] Loading reference model for precomputation...")
    ref_full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, device_map=None, attn_implementation="sdpa",
    )
    ref_full.disable_talker()
    ref_thinker = ref_full.thinker
    ref_thinker.eval()

    # 放到第一张卡
    device = "cuda:0"
    ref_thinker = ref_thinker.to(device)

    eos = processor.tokenizer.eos_token or ""
    pad_id = processor.tokenizer.pad_token_id

    ref_chosen_lps = []
    ref_rejected_lps = []
    total = len(dataset)

    for idx in range(total):
        ex = dataset[idx]
        prefix_text = ex["prefix_text"]
        msgs = ex["messages_prefix"]

        # 提取音频
        audios, _, _ = process_mm_info(msgs, use_audio_in_video=False)
        flat_audios = audios if audios else []

        chosen_text = prefix_text + ex["chosen"] + eos
        rejected_text = prefix_text + ex["rejected"] + eos

        # prefix 长度
        prefix_enc = processor(
            text=[prefix_text], audio=flat_audios,
            return_tensors="pt", padding=False, truncation=True,
            max_length=max_length,
        )
        prefix_len = prefix_enc["attention_mask"].sum().item()

        for text, lp_list in [(chosen_text, ref_chosen_lps), (rejected_text, ref_rejected_lps)]:
            enc = processor(
                text=[text], audio=flat_audios,
                return_tensors="pt", padding=False, truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[0, :prefix_len] = -100
            if pad_id is not None:
                labels[labels == pad_id] = -100

            # 提取非 input_ids/attention_mask 的额外参数（如 audio features）
            extra = {}
            for k, v in enc.items():
                if k not in ("input_ids", "attention_mask", "labels"):
                    extra[k] = v.to(device) if torch.is_tensor(v) else v

            lp = _compute_logprobs_single(ref_thinker, input_ids, attention_mask, labels, extra)
            lp_list.append(lp)

        if (idx + 1) % 50 == 0 or idx == total - 1:
            print(f"  [ref precompute] {idx + 1}/{total}")

    # 释放 ref model
    del ref_thinker, ref_full
    gc.collect()
    torch.cuda.empty_cache()
    print("[INFO] Reference model released.")

    return ref_chosen_lps, ref_rejected_lps


# ============================================================
# Data Collator
# ============================================================

@dataclass
class DataCollatorForDPO:
    """DPO 数据整理器：为 chosen/rejected 分别构建输入，附带预计算的 ref logprobs"""
    processor: Qwen3OmniMoeProcessor
    max_length: int = 2048

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
            return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_length,
        )
        rejected_inputs = self.processor(
            text=rejected_texts, audio=flat_audios,
            return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_length,
        )
        prefix_inputs = self.processor(
            text=prefix_texts, audio=flat_audios,
            return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_length,
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

        # 预计算的 ref logprobs
        batch["ref_chosen_logprobs"] = torch.tensor(
            [f["ref_chosen_logprobs"] for f in features], dtype=torch.float32
        )
        batch["ref_rejected_logprobs"] = torch.tensor(
            [f["ref_rejected_logprobs"] for f in features], dtype=torch.float32
        )
        return batch


# ============================================================
# DPO Trainer（无需 ref model）
# ============================================================

class DPOTrainer(Trainer):
    def __init__(self, beta=0.1, loss_type="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.loss_type = loss_type

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

        # 预计算的 ref logprobs，直接从 batch 中读取
        r_chosen = inputs["ref_chosen_logprobs"].to(p_chosen.device, dtype=p_chosen.dtype)
        r_rejected = inputs["ref_rejected_logprobs"].to(p_rejected.device, dtype=p_rejected.dtype)

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
            copy_config_files(
                self.base_model_path,
                os.path.join(args.output_dir, f"checkpoint-{state.global_step}"),
            )
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
    p.add_argument("--max_length", type=int, default=2048, help="最大序列长度，超出截断（防 OOM）")
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    # ---- 数据加载 ----
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

    # ---- 预计算 ref logprobs（仅 rank 0 计算，其余等待） ----
    cache_path = os.path.join(args.output_dir, "ref_logprobs.pt")
    os.makedirs(args.output_dir, exist_ok=True)

    if local_rank == 0:
        if os.path.exists(cache_path):
            print(f"[INFO] Loading cached ref logprobs from {cache_path}")
            cached = torch.load(cache_path, weights_only=True)
            ref_chosen_lps = cached["ref_chosen_logprobs"]
            ref_rejected_lps = cached["ref_rejected_logprobs"]
        else:
            ref_chosen_lps, ref_rejected_lps = precompute_ref_logprobs(
                args.model_path, dtype, processor, ds["train"], max_length=args.max_length
            )
            torch.save({
                "ref_chosen_logprobs": ref_chosen_lps,
                "ref_rejected_logprobs": ref_rejected_lps,
            }, cache_path)
            print(f"[INFO] Ref logprobs cached to {cache_path}")

    # 同步：等 rank 0 算完
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if local_rank != 0:
        cached = torch.load(cache_path, weights_only=True)
        ref_chosen_lps = cached["ref_chosen_logprobs"]
        ref_rejected_lps = cached["ref_rejected_logprobs"]

    # 把 ref logprobs 加到 dataset
    ds["train"] = ds["train"].add_column("ref_chosen_logprobs", ref_chosen_lps)
    ds["train"] = ds["train"].add_column("ref_rejected_logprobs", ref_rejected_lps)

    # ---- 加载 policy model ----
    print(f"[INFO] Loading policy model from {args.model_path}")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, device_map=None, attn_implementation="sdpa",
    )
    model.disable_talker()

    # 直接用 thinker 子模块：它有完整的 forward(input_ids, ...)
    thinker = model.thinker
    thinker.gradient_checkpointing_enable()

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
        beta=args.beta, loss_type=args.loss_type,
        model=thinker, args=training_args,
        train_dataset=ds["train"], eval_dataset=ds.get("validation"),
        data_collator=DataCollatorForDPO(processor=processor, max_length=args.max_length),
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
