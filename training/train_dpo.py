# coding=utf-8
"""
Qwen3-ASR DPO 训练脚本

基于官方 SFT 架构改造，使用 DPO 进行偏好对齐。
适用于 SFT 训练后的进一步优化。

用法：
  python train_dpo.py \
      --model_path ./output_sft/final \
      --train_file train_dpo.jsonl \
      --output_dir ./output_dpo \
      --batch_size 8 --grad_acc 8 --lr 5e-7 --beta 0.1
"""

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
import torch.nn.functional as F
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)


# ============================================================
# 工具函数（与 train_sft.py 一致）
# ============================================================

def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Cannot patch forward: model has no `.thinker.forward`.")

    def forward(self, input_ids=None, attention_mask=None, input_features=None,
                feature_attention_mask=None, labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features, feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs,
        )
    cls.forward = forward
    cls._forward_patched = True


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


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def copy_required_hf_files(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    for fn in [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "processor_config.json", "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json",
    ]:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)
        copy_required_hf_files(self.base_model_path, ckpt_dir)
        return control


# ============================================================
# DPO 数据预处理
# ============================================================

def make_dpo_preprocess_fn(processor):
    """预处理 DPO 数据：为 chosen 和 rejected 分别构建 prefix_text"""
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        prefix_msgs = build_prefix_messages(prompt, None)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
            "prefix_text": prefix_text,
        }
    return _preprocess


@dataclass
class DataCollatorForDPO:
    """
    DPO 数据整理器：
    为 chosen 和 rejected 分别构建完整序列，
    prefix 部分标记为 -100（不计算 loss）
    """
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        chosen_targets = [f["chosen"] for f in features]
        rejected_targets = [f["rejected"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        # 构建 chosen 序列
        chosen_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, chosen_targets)]
        chosen_inputs = self.processor(
            text=chosen_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )

        # 构建 rejected 序列
        rejected_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, rejected_targets)]
        rejected_inputs = self.processor(
            text=rejected_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )

        # prefix 长度（用于 label mask）
        prefix_inputs = self.processor(
            text=prefix_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()

        # 构建 labels
        pad_id = self.processor.tokenizer.pad_token_id
        for inputs_dict, name in [(chosen_inputs, "chosen"), (rejected_inputs, "rejected")]:
            labels = inputs_dict["input_ids"].clone()
            for i, pl in enumerate(prefix_lens):
                labels[i, :pl] = -100
            if pad_id is not None:
                labels[labels == pad_id] = -100
            inputs_dict["labels"] = labels

        # 返回包含 chosen 和 rejected 的 batch
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
    """
    自定义 DPO Trainer

    DPO Loss:
    L = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
    """
    def __init__(self, ref_model=None, beta=0.1, loss_type="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.loss_type = loss_type

        # 冻结 reference model
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def _prepare_inputs(self, inputs):
        """确保浮点输入与模型 dtype 一致"""
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs

    def _compute_logprobs(self, model, input_ids, attention_mask, labels,
                          input_features=None, feature_attention_mask=None):
        """计算 response 部分的对数概率"""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        logits = outputs.logits  # [batch, seq_len, vocab]

        # shift: logits[:-1] 预测 labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs, dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)

        # 只在 response 部分计算（label != -100）
        loss_mask = (shift_labels != -100).float()
        per_seq_logps = (per_token_logps * loss_mask).sum(dim=-1)
        return per_seq_logps

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算 DPO loss"""
        # 提取 chosen 和 rejected 的输入
        chosen_kwargs = {
            "input_ids": inputs["chosen_input_ids"],
            "attention_mask": inputs["chosen_attention_mask"],
            "labels": inputs["chosen_labels"],
        }
        rejected_kwargs = {
            "input_ids": inputs["rejected_input_ids"],
            "attention_mask": inputs["rejected_attention_mask"],
            "labels": inputs["rejected_labels"],
        }

        # 音频特征（chosen 和 rejected 共享同一段音频）
        if "chosen_input_features" in inputs:
            chosen_kwargs["input_features"] = inputs["chosen_input_features"]
            rejected_kwargs["input_features"] = inputs["rejected_input_features"]
        if "chosen_feature_attention_mask" in inputs:
            chosen_kwargs["feature_attention_mask"] = inputs["chosen_feature_attention_mask"]
            rejected_kwargs["feature_attention_mask"] = inputs["rejected_feature_attention_mask"]

        # Policy model log probs
        policy_chosen_logps = self._compute_logprobs(model, **chosen_kwargs)
        policy_rejected_logps = self._compute_logprobs(model, **rejected_kwargs)

        # Reference model log probs
        with torch.no_grad():
            ref_chosen_logps = self._compute_logprobs(self.ref_model, **chosen_kwargs)
            ref_rejected_logps = self._compute_logprobs(self.ref_model, **rejected_kwargs)

        # DPO loss
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
        margins = chosen_rewards - rejected_rewards

        if self.loss_type == "sigmoid":
            loss = -F.logsigmoid(margins).mean()
        elif self.loss_type == "hinge":
            loss = torch.relu(1 - margins).mean()
        elif self.loss_type == "ipo":
            loss = ((margins - 1 / (2 * self.beta)) ** 2).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 日志
        if self.state.global_step % self.args.logging_steps == 0:
            acc = (policy_chosen_logps > policy_rejected_logps).float().mean().item()
            self.log({
                "dpo_loss": loss.item(),
                "rewards/chosen": chosen_rewards.mean().item(),
                "rewards/rejected": rejected_rewards.mean().item(),
                "rewards/margins": margins.mean().item(),
                "rewards/accuracies": acc,
            })

        if return_outputs:
            return loss, {"logits": policy_chosen_logps}
        return loss


# ============================================================
# 参数解析
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR DPO Training")
    p.add_argument("--model_path", type=str, required=True, help="SFT 训练后的模型路径")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./output_dpo")
    p.add_argument("--sr", type=int, default=16000)
    # DPO 参数
    p.add_argument("--beta", type=float, default=0.1, help="DPO 温度")
    p.add_argument("--loss_type", type=str, default="sigmoid", choices=["sigmoid", "hinge", "ipo"])
    # 训练超参
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_acc", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    # 保存
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=3)
    # 恢复
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    return p.parse_args()


# ============================================================
# 主函数
# ============================================================

def main():
    args_cli = parse_args()
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    # 加载 policy model
    print(f"[INFO] Loading policy model from {args_cli.model_path}")
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # 加载 reference model（冻结副本）
    print(f"[INFO] Loading reference model from {args_cli.model_path}")
    ref_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    ref_model = ref_wrapper.model
    patch_outer_forward(ref_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 加载数据集
    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({} if not args_cli.eval_file else {"validation": args_cli.eval_file}),
        },
    )
    ds = raw_ds.map(make_dpo_preprocess_fn(processor), num_proc=1)

    keep = {"prompt", "audio", "chosen", "rejected", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForDPO(processor=processor, sampling_rate=args_cli.sr)

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        save_strategy="steps",
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps" if args_cli.eval_file else "no",
        eval_steps=args_cli.save_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        ref_model=ref_model,
        beta=args_cli.beta,
        loss_type=args_cli.loss_type,
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeCheckpointInferableCallback(base_model_path=args_cli.model_path)],
    )

    # 恢复训练
    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        print(f"[resume] {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # 保存最终模型
    final_dir = os.path.join(args_cli.output_dir, "final")
    trainer.save_model(final_dir)
    copy_required_hf_files(args_cli.model_path, final_dir)
    print(f"[done] Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
