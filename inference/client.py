"""
Qwen3-Omni 多任务推理客户端

客户端按需选择要执行的任务组合，每个任务独立调用模型。

用法：
    # 本地 - 只做 ASR
    python client.py --local --model_path ./output_sft/final \
        --audio user.wav --context "销售员：请问您需要什么产品？" --tasks asr

    # 本地 - ASR + 情绪
    python client.py --local --model_path ./output_sft/final \
        --audio user.wav --tasks asr emotion

    # 本地 - 全部任务
    python client.py --local --model_path ./output_sft/final \
        --audio user.wav --tasks asr emotion gender age

    # 远程 API
    python client.py --api_url http://localhost:9000 \
        --audio user.wav --tasks asr emotion

    # 批量
    python client.py --local --model_path ./output_sft/final \
        --audio_dir /data/audios/ --tasks asr --output results.jsonl
"""

import json
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 任务定义（与 tasks.json 一致，内联避免文件依赖）
TASK_DEFS = {
    "asr": {
        "system": "你是语音识别助手，请准确转录用户语音内容。",
        "user_prompt": "请准确识别语音内容。",
    },
    "emotion": {
        "system": "你是语音情绪分析助手，根据说话人的语气、语调、语速判断情绪。",
        "user_prompt": "请判断说话人的情绪状态。输出JSON格式。",
    },
    "gender": {
        "system": "你是语音性别识别助手，根据声音特征判断说话人性别。",
        "user_prompt": "请判断说话人的性别。输出JSON格式。",
    },
    "age": {
        "system": "你是语音年龄识别助手，根据声音特征判断说话人的大致年龄段。",
        "user_prompt": "请判断说话人的年龄段。输出JSON格式。",
    },
}


class LocalClient:
    """本地 Transformers 推理"""
    def __init__(self, model_path, device="auto"):
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        from qwen_omni_utils import process_mm_info
        self._process_mm_info = process_mm_info
        self._torch = torch

        logger.info(f"Loading model from {model_path}")
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map=device, attn_implementation="flash_attention_2",
        )
        self.model.disable_talker()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        logger.info("Model loaded")

    def _infer_single_task(self, audio_path, task_id, context="", max_tokens=512):
        """执行单个任务"""
        task_def = TASK_DEFS[task_id]
        system_text = task_def["system"]
        # 只有 ASR 需要销售员上下文
        if context and task_id == "asr":
            system_text += f"\n销售员上一句：{context}"

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": task_def["user_prompt"]},
            ]},
        ]

        start = time.time()
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = self._process_mm_info(messages, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                               return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        with self._torch.no_grad():
            text_ids, _ = self.model.generate(
                **inputs, return_audio=False,
                thinker_return_dict_in_generate=True, max_new_tokens=max_tokens,
            )
        output = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        latency_ms = int((time.time() - start) * 1000)

        # JSON 任务尝试解析
        if task_id != "asr":
            try:
                parsed = json.loads(output)
                parsed["latency_ms"] = latency_ms
                return parsed
            except json.JSONDecodeError:
                pass

        return {"text" if task_id == "asr" else "raw": output, "latency_ms": latency_ms}

    def infer(self, audio_path, context="", tasks=None, max_tokens=512):
        """执行多个任务，返回各任务结果"""
        tasks = tasks or ["asr"]
        results = {}
        for task_id in tasks:
            if task_id not in TASK_DEFS:
                results[task_id] = {"error": f"未知任务: {task_id}"}
                continue
            results[task_id] = self._infer_single_task(audio_path, task_id, context, max_tokens)
        return results


class APIClient:
    """远程 API 客户端"""
    def __init__(self, api_url="http://localhost:9000", timeout=120):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def infer(self, audio_path, context="", tasks=None, max_tokens=512):
        tasks = tasks or ["asr"]
        try:
            resp = requests.post(f"{self.api_url}/infer", json={
                "audio_path": audio_path, "context": context,
                "tasks": tasks, "max_tokens": max_tokens,
            }, timeout=self.timeout)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}


def main():
    p = argparse.ArgumentParser("Qwen3-Omni Multi-Task Client")
    p.add_argument("--local", action="store_true")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--api_url", type=str, default="http://localhost:9000")
    p.add_argument("--audio", type=str)
    p.add_argument("--audio_dir", type=str)
    p.add_argument("--context", type=str, default="")
    p.add_argument("--tasks", nargs="+", default=["asr"], choices=list(TASK_DEFS.keys()))
    p.add_argument("--output", type=str, default="results.jsonl")
    args = p.parse_args()

    client = LocalClient(args.model_path) if args.local else APIClient(args.api_url)

    if args.audio:
        result = client.infer(args.audio, context=args.context, tasks=args.tasks)
        print(f"\n{'='*60}")
        print(f"  任务: {', '.join(args.tasks)}")
        if args.context:
            print(f"  上下文: {args.context}")
        print(f"{'='*60}")
        for task_id, task_result in result.items():
            print(f"\n  [{task_id}]")
            for k, v in (task_result.items() if isinstance(task_result, dict) else [("result", task_result)]):
                print(f"    {k}: {v}")
        print(f"\n{'='*60}\n")
        return

    if args.audio_dir:
        audio_paths = sorted(
            str(p) for p in Path(args.audio_dir).glob("*")
            if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
        )
        logger.info(f"Found {len(audio_paths)} files, tasks={args.tasks}")
        with open(args.output, "w") as f:
            for i, ap in enumerate(audio_paths):
                r = client.infer(ap, context=args.context, tasks=args.tasks)
                r["_audio_path"] = ap
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if (i + 1) % 10 == 0:
                    logger.info(f"  {i+1}/{len(audio_paths)}")
        logger.info(f"Saved to {args.output}")
        return

    p.print_help()


if __name__ == "__main__":
    main()
