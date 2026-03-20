"""
Qwen3-ASR DPO 后训练脚本
场景：电销对话，利用销售员上下文提升用户语音识别准确率
架构：Qwen3ASRForConditionalGeneration (Audio Encoder + Qwen3 Decoder)

DPO 思路：
- 输入：用户音频 + 销售员文本上下文（作为 prompt 的一部分）
- chosen：正确的用户话术转录
- rejected：错误的转录（模型原始输出 / 人工标注的错误样本）

用法：
  单卡：python train_asr_dpo.py --config config.yaml
  多卡：deepspeed --num_gpus=N train_asr_dpo.py --config config.yaml --deepspeed ds_config.json
"""

import os
import json
import copy
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import soundfile as sf
import librosa

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. 数据集定义
# ============================================================

"""
数据格式 (JSONL):
{
    "audio_path": "/data/audio/call_001_user_turn_03.wav",
    "sales_context": "销售员：您好，请问您对我们的保险产品感兴趣吗？",
    "chosen": "嗯，我想了解一下你们那个百万医疗险，就是住院能报销的那种",
    "rejected": "嗯我想了解一下你们那个百万医疗险就是住院能报销的那种",
    "dialogue_history": [
        {"role": "sales", "text": "您好，我是XX保险的客服小王"},
        {"role": "user", "text": "嗯你好"},
        {"role": "sales", "text": "请问您对我们的保险产品感兴趣吗？"}
    ]
}

说明：
- audio_path: 当前用户说话的音频文件路径
- sales_context: 紧邻当前用户话术之前的销售员文本（最关键的上下文）
- chosen: 正确的转录文本（人工标注）
- rejected: 错误的转录文本（可以是模型原始输出、或人工构造的错误样本）
- dialogue_history (可选): 更完整的对话历史，按轮次记录
"""


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """加载音频文件并重采样到目标采样率"""
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # 多声道转单声道
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 128,
) -> torch.Tensor:
    """计算 Mel 频谱图，与 Qwen3-ASR 音频编码器对齐"""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # 归一化到 [-1, 1]
    log_mel = (log_mel + 40) / 40
    return torch.from_numpy(log_mel).float()


class ASRDPODataset(Dataset):
    """ASR DPO 数据集"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_audio_len: int = 480000,  # 30s @ 16kHz
        max_text_len: int = 512,
        use_dialogue_history: bool = True,
        max_history_turns: int = 3,
    ):
        self.tokenizer = tokenizer
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.use_dialogue_history = use_dialogue_history
        self.max_history_turns = max_history_turns

        # 加载数据
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        logger.info(f"Loaded {len(self.data)} DPO examples from {data_path}")

    def _build_context_prompt(self, item: dict) -> str:
        """
        构建包含销售员上下文的 prompt
        关键：将销售员的文本话术作为指令上下文，帮助模型理解对话场景
        """
        parts = []

        # 系统指令
        parts.append(
            "你是一个电销场景的语音识别助手。"
            "请根据提供的对话上下文，准确识别用户的语音内容。"
        )

        # 对话历史（如果有）
        if self.use_dialogue_history and "dialogue_history" in item:
            history = item["dialogue_history"]
            # 只取最近 N 轮
            history = history[-self.max_history_turns :]
            if history:
                parts.append("\n【对话上下文】")
                for turn in history:
                    role = "销售员" if turn["role"] == "sales" else "用户"
                    parts.append(f"{role}：{turn['text']}")
        elif "sales_context" in item and item["sales_context"]:
            # 没有完整历史时，使用 sales_context
            parts.append(f"\n【销售员上一句话】\n{item['sales_context']}")

        parts.append("\n【请识别以下用户语音】")

        return "\n".join(parts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. 加载音频
        audio = load_audio(item["audio_path"])
        if len(audio) > self.max_audio_len:
            audio = audio[: self.max_audio_len]

        # 2. 计算 mel 频谱
        mel = compute_mel_spectrogram(audio)

        # 3. 构建上下文 prompt
        context_prompt = self._build_context_prompt(item)

        # 4. Tokenize chosen & rejected
        chosen_text = item["chosen"]
        rejected_text = item["rejected"]

        return {
            "audio": audio,
            "mel": mel,
            "context_prompt": context_prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }


# ============================================================
# 2. DPO Trainer（适配多模态 ASR）
# ============================================================


class ASRDPOTrainer:
    """
    自定义 DPO Trainer，适配 Qwen3-ASR 多模态架构
    
    核心思路：
    1. 冻结音频编码器（或低学习率微调）
    2. 对文本解码器部分应用 DPO 训练
    3. prompt 部分（音频特征 + 上下文文本）不计算 loss
    4. 只在 response 部分（转录文本）计算 DPO loss
    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        processor,
        args: dict,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.args = args
        self.beta = args.get("beta", 0.1)
        self.loss_type = args.get("loss_type", "sigmoid")

        # 冻结 reference model
        if self.ref_model is not None:
            for param in self.ref_model.parameters():
                param.requires_grad = False

        # 可选：冻结音频编码器
        if args.get("freeze_audio_encoder", True):
            self._freeze_audio_encoder()

    def _freeze_audio_encoder(self):
        """冻结音频编码器参数"""
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if "audio" in name.lower() or "encoder" in name.lower():
                param.requires_grad = False
                frozen_count += 1
        logger.info(f"Froze {frozen_count} audio encoder parameters")

    def _build_model_inputs(
        self,
        audio_array: np.ndarray,
        context_prompt: str,
        response_text: str,
    ) -> Dict[str, torch.Tensor]:
        """
        构建模型输入，包含音频特征和文本

        Qwen3-ASR 的输入格式：
        [system] [context_text] [audio_start] [audio_features] [audio_end] [response_text]

        其中 audio_start/audio_end 是特殊 token (151669/151670)
        audio_features 由音频编码器产生，用 audio_token (151676) 占位
        """
        # 使用 processor 处理音频（如果可用）
        # 否则手动构建输入

        # 构建完整的对话文本
        # prompt 部分（不计算 loss）
        prompt_text = context_prompt

        # 用特殊标记包裹音频
        AUDIO_START = "<|audio_bos|>"
        AUDIO_END = "<|audio_eos|>"

        # 完整输入文本
        full_text = f"{prompt_text}\n{AUDIO_START}{{audio}}{AUDIO_END}\n{response_text}"

        return {
            "audio": audio_array,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "full_text": full_text,
        }

    def compute_logprobs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算给定 response 的对数概率

        Args:
            model: 模型
            input_ids: 输入 token ids [batch, seq_len]
            attention_mask: 注意力掩码
            labels: 标签（prompt 部分为 -100）
            audio_features: 预计算的音频特征

        Returns:
            per-sequence 对数概率 [batch]
        """
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_features=audio_features,
            )

        logits = outputs.logits  # [batch, seq_len, vocab]
        # Shift: logits[:-1] 预测 labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # 计算每个 token 的 log prob
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # 只在 label != -100 的位置计算（即 response 部分）
        loss_mask = (shift_labels != -100).float()
        per_seq_logps = (per_token_logps * loss_mask).sum(dim=-1)

        return per_seq_logps

    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 DPO Loss

        L_DPO = -log σ(β * (log π(y_w|x) / π_ref(y_w|x) - log π(y_l|x) / π_ref(y_l|x)))
        """
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (
            policy_rejected_logps - reference_rejected_logps
        )
        margins = chosen_rewards - rejected_rewards

        if self.loss_type == "sigmoid":
            loss = -F.logsigmoid(margins)
        elif self.loss_type == "hinge":
            loss = torch.relu(1 - margins)
        elif self.loss_type == "ipo":
            loss = (margins - 1 / (2 * self.beta)) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean()


# ============================================================
# 3. 完整训练流程（基于 TRL 风格封装）
# ============================================================


def prepare_dpo_batch(
    batch: List[dict],
    tokenizer,
    processor,
    model,
    max_length: int = 2048,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    将一个 batch 的数据处理成模型可接受的输入

    关键：音频特征通过音频编码器提取后，替换 input_ids 中的 audio_token 占位符
    """
    chosen_input_ids_list = []
    chosen_labels_list = []
    rejected_input_ids_list = []
    rejected_labels_list = []
    audio_features_list = []

    for item in batch:
        context = item["context_prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        audio = item["audio"]

        # ---- 构建 prompt + response ----
        # Prompt 部分（包含上下文 + 音频占位符）
        prompt_template = (
            f"<|im_start|>system\n你是一个电销场景的语音识别助手。<|im_end|>\n"
            f"<|im_start|>user\n{context}\n"
            f"<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        prompt_ids = tokenizer.encode(prompt_template, add_special_tokens=False)
        chosen_ids = tokenizer.encode(
            chosen + "<|im_end|>", add_special_tokens=False
        )
        rejected_ids = tokenizer.encode(
            rejected + "<|im_end|>", add_special_tokens=False
        )

        # 完整序列
        chosen_full = prompt_ids + chosen_ids
        rejected_full = prompt_ids + rejected_ids

        # Labels：prompt 部分 = -100，response 部分 = token_ids
        chosen_labels = [-100] * len(prompt_ids) + chosen_ids
        rejected_labels = [-100] * len(prompt_ids) + rejected_ids

        # 截断
        chosen_full = chosen_full[:max_length]
        chosen_labels = chosen_labels[:max_length]
        rejected_full = rejected_full[:max_length]
        rejected_labels = rejected_labels[:max_length]

        chosen_input_ids_list.append(torch.tensor(chosen_full, dtype=torch.long))
        chosen_labels_list.append(torch.tensor(chosen_labels, dtype=torch.long))
        rejected_input_ids_list.append(torch.tensor(rejected_full, dtype=torch.long))
        rejected_labels_list.append(torch.tensor(rejected_labels, dtype=torch.long))

        # 处理音频特征（通过模型的音频编码器）
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        audio_features_list.append(audio_tensor)

    # Pad sequences
    def pad_sequences(seqs, pad_value=0):
        max_len = max(s.size(0) for s in seqs)
        padded = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
        masks = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, : s.size(0)] = s
            masks[i, : s.size(0)] = 1
        return padded, masks

    chosen_input_ids, chosen_attention_mask = pad_sequences(
        chosen_input_ids_list, pad_value=tokenizer.pad_token_id or 0
    )
    chosen_labels, _ = pad_sequences(chosen_labels_list, pad_value=-100)
    rejected_input_ids, rejected_attention_mask = pad_sequences(
        rejected_input_ids_list, pad_value=tokenizer.pad_token_id or 0
    )
    rejected_labels, _ = pad_sequences(rejected_labels_list, pad_value=-100)

    return {
        "chosen_input_ids": chosen_input_ids.to(device),
        "chosen_attention_mask": chosen_attention_mask.to(device),
        "chosen_labels": chosen_labels.to(device),
        "rejected_input_ids": rejected_input_ids.to(device),
        "rejected_attention_mask": rejected_attention_mask.to(device),
        "rejected_labels": rejected_labels.to(device),
        "audio_list": audio_features_list,
    }


# ============================================================
# 4. 主训练循环
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR DPO Training")

    # 模型参数
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Qwen3-ASR model path (e.g., Qwen/Qwen3-ASR-1.7B or local path)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to DPO JSONL dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_asr_dpo",
        help="Output directory",
    )

    # DPO 参数
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "hinge", "ipo"],
    )

    # 训练参数
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lr_audio_encoder", type=float, default=0.0,
                        help="Audio encoder LR. 0 = frozen.")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_audio_len", type=int, default=480000)
    parser.add_argument("--max_history_turns", type=int, default=3)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--eval_split_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    # ---- 设置设备 ----
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 加载 tokenizer ----
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 加载模型 ----
    logger.info(f"Loading model from {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto" if args.local_rank == -1 else None,
    )
    model.warnings_issued = {}  # 兼容性修复

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ---- 加载 reference model (冻结) ----
    logger.info("Loading reference model (frozen copy)")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto" if args.local_rank == -1 else None,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # ---- 处理音频编码器冻结/微调 ----
    freeze_audio = args.lr_audio_encoder == 0.0
    if freeze_audio:
        frozen_count = 0
        for name, param in model.named_parameters():
            if "audio" in name.lower() or "encoder" in name.lower():
                param.requires_grad = False
                frozen_count += 1
        logger.info(f"Froze {frozen_count} audio encoder parameters")

    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ---- 加载数据集 ----
    logger.info(f"Loading dataset from {args.dataset_path}")
    full_dataset = ASRDPODataset(
        data_path=args.dataset_path,
        tokenizer=tokenizer,
        max_audio_len=args.max_audio_len,
        max_text_len=args.max_length,
        use_dialogue_history=True,
        max_history_turns=args.max_history_turns,
    )

    # 划分训练/验证集
    eval_size = max(1, int(len(full_dataset) * args.eval_split_ratio))
    train_size = len(full_dataset) - eval_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info(f"Train: {train_size}, Eval: {eval_size}")

    # ---- 创建 DPO Trainer ----
    dpo_trainer = ASRDPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        processor=None,
        args={
            "beta": args.beta,
            "loss_type": args.loss_type,
            "freeze_audio_encoder": freeze_audio,
        },
    )

    # ---- 优化器 ----
    if not freeze_audio and args.lr_audio_encoder > 0:
        # 分层学习率
        audio_params = []
        text_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "audio" in name.lower() or "encoder" in name.lower():
                audio_params.append(param)
            else:
                text_params.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": text_params, "lr": args.learning_rate},
                {"params": audio_params, "lr": args.lr_audio_encoder},
            ],
            weight_decay=0.01,
        )
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.learning_rate, weight_decay=0.01
        )

    # ---- 学习率调度 ----
    from transformers import get_cosine_schedule_with_warmup

    total_steps = (
        len(train_dataset)
        // args.per_device_train_batch_size
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # ---- DataLoader ----
    def collate_fn(batch):
        return batch  # 在训练循环中处理

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # ---- 训练循环 ----
    logger.info("=" * 60)
    logger.info("Starting DPO training")
    logger.info(f"  Epochs: {args.num_train_epochs}")
    logger.info(f"  Batch size: {args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Beta: {args.beta}")
    logger.info(f"  Loss type: {args.loss_type}")
    logger.info("=" * 60)

    global_step = 0
    best_eval_accuracy = 0.0
    scaler = torch.amp.GradScaler("cuda") if not args.bf16 else None

    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            # 处理 batch
            processed = prepare_dpo_batch(
                batch=batch,
                tokenizer=tokenizer,
                processor=None,
                model=model,
                max_length=args.max_length,
                device=device,
            )

            # 计算 policy model 的 log probs
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                policy_chosen_logps = dpo_trainer.compute_logprobs(
                    model=model,
                    input_ids=processed["chosen_input_ids"],
                    attention_mask=processed["chosen_attention_mask"],
                    labels=processed["chosen_labels"],
                )
                policy_rejected_logps = dpo_trainer.compute_logprobs(
                    model=model,
                    input_ids=processed["rejected_input_ids"],
                    attention_mask=processed["rejected_attention_mask"],
                    labels=processed["rejected_labels"],
                )

            # 计算 reference model 的 log probs（不需要梯度）
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    ref_chosen_logps = dpo_trainer.compute_logprobs(
                        model=ref_model,
                        input_ids=processed["chosen_input_ids"],
                        attention_mask=processed["chosen_attention_mask"],
                        labels=processed["chosen_labels"],
                    )
                    ref_rejected_logps = dpo_trainer.compute_logprobs(
                        model=ref_model,
                        input_ids=processed["rejected_input_ids"],
                        attention_mask=processed["rejected_attention_mask"],
                        labels=processed["rejected_labels"],
                    )

            # 计算 DPO loss
            loss, chosen_reward, rejected_reward = dpo_trainer.dpo_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                reference_chosen_logps=ref_chosen_logps,
                reference_rejected_logps=ref_rejected_logps,
            )

            # 梯度累积
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            # Accuracy
            with torch.no_grad():
                acc = (
                    (policy_chosen_logps > policy_rejected_logps).float().mean().item()
                )

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            epoch_acc += acc
            epoch_count += 1

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss / epoch_count
                    avg_acc = epoch_acc / epoch_count
                    lr = scheduler.get_last_lr()[0]
                    margin = chosen_reward.item() - rejected_reward.item()
                    logger.info(
                        f"Step {global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Acc: {avg_acc:.3f} | "
                        f"Margin: {margin:.4f} | "
                        f"Chosen_R: {chosen_reward.item():.4f} | "
                        f"Rejected_R: {rejected_reward.item():.4f} | "
                        f"LR: {lr:.2e}"
                    )

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    logger.info(f"Saved checkpoint to {ckpt_dir}")

        # ---- Eval at end of epoch ----
        model.eval()
        eval_loss = 0.0
        eval_acc = 0.0
        eval_count = 0

        with torch.no_grad():
            for batch in eval_loader:
                processed = prepare_dpo_batch(
                    batch=batch,
                    tokenizer=tokenizer,
                    processor=None,
                    model=model,
                    max_length=args.max_length,
                    device=device,
                )

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    p_chosen = dpo_trainer.compute_logprobs(
                        model, processed["chosen_input_ids"],
                        processed["chosen_attention_mask"],
                        processed["chosen_labels"],
                    )
                    p_rejected = dpo_trainer.compute_logprobs(
                        model, processed["rejected_input_ids"],
                        processed["rejected_attention_mask"],
                        processed["rejected_labels"],
                    )
                    r_chosen = dpo_trainer.compute_logprobs(
                        ref_model, processed["chosen_input_ids"],
                        processed["chosen_attention_mask"],
                        processed["chosen_labels"],
                    )
                    r_rejected = dpo_trainer.compute_logprobs(
                        ref_model, processed["rejected_input_ids"],
                        processed["rejected_attention_mask"],
                        processed["rejected_labels"],
                    )

                loss, _, _ = dpo_trainer.dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected)
                acc = (p_chosen > p_rejected).float().mean().item()

                eval_loss += loss.item()
                eval_acc += acc
                eval_count += 1

        avg_eval_loss = eval_loss / max(eval_count, 1)
        avg_eval_acc = eval_acc / max(eval_count, 1)
        logger.info(
            f"Epoch {epoch+1}/{args.num_train_epochs} | "
            f"Eval Loss: {avg_eval_loss:.4f} | "
            f"Eval Accuracy: {avg_eval_acc:.3f}"
        )

        # 保存最佳模型
        if avg_eval_acc > best_eval_accuracy:
            best_eval_accuracy = avg_eval_acc
            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logger.info(f"New best model (acc={avg_eval_acc:.3f}) saved to {best_dir}")

    # ---- 保存最终模型 ----
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Training complete! Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
