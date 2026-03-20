"""
Qwen3-Omni 推理客户端

支持本地 Transformers 推理和远程 API 调用。
三种模式：asr / analyze / full

用法：
    # 本地推理
    python client.py --local --model_path ./output_dpo/final \
        --audio user.wav --context "销售员：请问您需要什么产品？" --mode full

    # 远程 API
    python client.py --api_url http://localhost:9000 \
        --audio user.wav --mode asr

    # 批量
    python client.py --local --model_path ./output_dpo/final \
        --audio_dir /data/audios/ --mode asr --output results.jsonl
"""

import json
import time
import argparse
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


PROMPTS = {
    "asr": "请准确识别用户语音内容。",
    "analyze": "请分析说话人的性别、年龄段、情绪和购买意向。输出JSON格式。",
    "full": "请先识别用户语音内容，然后分析说话人特征。",
}


class LocalClient:
    """本地 Transformers 推理"""
    def __init__(self, model_path, device="auto"):
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        from qwen_omni_utils import process_mm_info
        self._process_mm_info = process_mm_info

        logger.info(f"Loading model from {model_path}")
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map=device, attn_implementation="flash_attention_2",
        )
        self.model.disable_talker()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        logger.info("Model loaded")

    def infer(self, audio_path, context="", mode="asr", max_tokens=512):
        import torch

        system_msg = "你是电销场景语音识别与分析助手。"
        if context:
            system_msg += f"销售员上一句：{context}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": PROMPTS.get(mode, PROMPTS["asr"])},
            ]},
        ]

        start = time.time()
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = self._process_mm_info(messages, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                               return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        with torch.no_grad():
            text_ids, _ = self.model.generate(
                **inputs, return_audio=False,
                thinker_return_dict_in_generate=True, max_new_tokens=max_tokens,
            )
        output = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        return {"output": output, "mode": mode, "latency_ms": int((time.time() - start) * 1000)}


class APIClient:
    """远程 API 客户端"""
    def __init__(self, api_url="http://localhost:9000", timeout=120):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def infer(self, audio_path, context="", mode="asr", max_tokens=512):
        try:
            resp = requests.post(f"{self.api_url}/infer", json={
                "audio_path": audio_path, "context": context,
                "mode": mode, "max_tokens": max_tokens,
            }, timeout=self.timeout)
            return resp.json()
        except Exception as e:
            return {"output": "", "error": str(e)}


def main():
    p = argparse.ArgumentParser("Qwen3-Omni Client")
    p.add_argument("--local", action="store_true")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--api_url", type=str, default="http://localhost:9000")
    p.add_argument("--audio", type=str)
    p.add_argument("--audio_dir", type=str)
    p.add_argument("--context", type=str, default="")
    p.add_argument("--mode", type=str, default="asr", choices=["asr", "analyze", "full"])
    p.add_argument("--output", type=str, default="results.jsonl")
    args = p.parse_args()

    client = LocalClient(args.model_path) if args.local else APIClient(args.api_url)

    if args.audio:
        result = client.infer(args.audio, context=args.context, mode=args.mode)
        print(f"\n{'='*50}")
        print(f"  模式: {args.mode}")
        print(f"  结果: {result['output']}")
        print(f"  耗时: {result.get('latency_ms', 'N/A')}ms")
        if args.context:
            print(f"  上下文: {args.context}")
        print(f"{'='*50}\n")
        return

    if args.audio_dir:
        audio_paths = sorted(
            str(p) for p in Path(args.audio_dir).glob("*")
            if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
        )
        logger.info(f"Found {len(audio_paths)} files")
        with open(args.output, "w") as f:
            for i, ap in enumerate(audio_paths):
                r = client.infer(ap, context=args.context, mode=args.mode)
                r["audio_path"] = ap
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if (i + 1) % 10 == 0:
                    logger.info(f"  {i+1}/{len(audio_paths)}")
        logger.info(f"Saved to {args.output}")
        return

    p.print_help()


if __name__ == "__main__":
    main()
