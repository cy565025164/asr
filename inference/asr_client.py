"""
Qwen3-ASR 推理客户端

支持两种模式：
  1. 本地 SDK 模式（--local）：完整支持 context 参数
  2. vLLM API 模式：通过 OpenAI 兼容 API 调用

用法：
    # 本地模式（推荐）
    python asr_client.py --local --model_path ./output_dpo/final \
        --audio user.wav --context "销售员：请问您需要什么产品？"

    # vLLM 模式
    python asr_client.py --base_url http://localhost:8000 \
        --audio user.wav

    # 批量识别
    python asr_client.py --local --model_path ./output_dpo/final \
        --audio_dir /data/audios/ --context_file contexts.jsonl --output results.jsonl
"""

import json
import time
import base64
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 本地 SDK 客户端
# ============================================================

class LocalClient:
    """本地推理，使用 qwen-asr SDK，完整支持 context"""

    def __init__(self, model_path: str, device: str = "cuda:0",
                 forced_aligner: str = None):
        import torch
        from qwen_asr import Qwen3ASRModel

        logger.info(f"Loading model from {model_path}")
        kw = {}
        if forced_aligner:
            kw = {
                "forced_aligner": forced_aligner,
                "forced_aligner_kwargs": dict(dtype=torch.bfloat16, device_map=device),
            }
        self.model = Qwen3ASRModel.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device,
            max_inference_batch_size=32, max_new_tokens=512, **kw,
        )
        logger.info("Model loaded")

    def transcribe(self, audio, context="", language="Chinese",
                   return_timestamps=False) -> Dict:
        start = time.time()
        results = self.model.transcribe(
            audio=audio, context=context or "",
            language=language, return_time_stamps=return_timestamps,
        )
        latency = int((time.time() - start) * 1000)
        r = results[0]
        out = {"text": r.text, "language": r.language, "latency_ms": latency}
        if return_timestamps and r.time_stamps:
            out["timestamps"] = [
                {"text": ts.text, "start": ts.start_time, "end": ts.end_time}
                for ts in r.time_stamps
            ]
        return out

    def transcribe_batch(self, audio_paths, contexts=None,
                         language="Chinese", return_timestamps=False) -> List[Dict]:
        contexts = contexts or [""] * len(audio_paths)
        start = time.time()
        results = self.model.transcribe(
            audio=audio_paths, context=contexts,
            language=language, return_time_stamps=return_timestamps,
        )
        total_ms = int((time.time() - start) * 1000)
        outputs = []
        for i, r in enumerate(results):
            out = {
                "audio_path": audio_paths[i], "text": r.text,
                "language": r.language, "latency_ms": total_ms // len(audio_paths),
            }
            outputs.append(out)
        logger.info(f"Batch: {len(audio_paths)} files, {total_ms}ms total")
        return outputs


# ============================================================
# vLLM API 客户端
# ============================================================

class VLLMClient:
    """vLLM OpenAI 兼容 API 客户端"""

    def __init__(self, base_url="http://localhost:8000", timeout=60):
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/v1/chat/completions"
        self.timeout = timeout
        self.model_id = self._get_model_id()

    def _get_model_id(self):
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            models = resp.json().get("data", [])
            if models:
                mid = models[0]["id"]
                logger.info(f"Connected to vLLM, model: {mid}")
                return mid
        except Exception:
            logger.warning(f"Cannot connect to {self.base_url}")
        return None

    def transcribe(self, audio_path=None, audio_url=None,
                   context="", language="Chinese") -> Dict:
        messages = [{"role": "system", "content": context or ""}]

        user_content = []
        if audio_path:
            with open(audio_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = Path(audio_path).suffix.lower()
            mime = {".wav": "audio/wav", ".mp3": "audio/mp3", ".flac": "audio/flac"}.get(ext, "audio/wav")
            user_content.append({"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{b64}"}})
        elif audio_url:
            user_content.append({"type": "audio_url", "audio_url": {"url": audio_url}})

        messages.append({"role": "user", "content": user_content})

        payload = {"messages": messages, "temperature": 0, "max_tokens": 512}
        if self.model_id:
            payload["model"] = self.model_id

        start = time.time()
        try:
            resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return {"text": "", "error": str(e), "latency_ms": 0}

        latency = int((time.time() - start) * 1000)

        text = content
        try:
            from qwen_asr import parse_asr_output
            _, text = parse_asr_output(content)
        except Exception:
            pass

        return {"text": text, "language": language, "latency_ms": latency}

    def transcribe_batch(self, audio_paths, contexts=None, max_workers=4) -> List[Dict]:
        contexts = contexts or [""] * len(audio_paths)
        results = [None] * len(audio_paths)

        def _run(i, p, c):
            r = self.transcribe(audio_path=p, context=c)
            r["audio_path"] = p
            return i, r

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_run, i, p, c): i for i, (p, c) in enumerate(zip(audio_paths, contexts))}
            for f in as_completed(futs):
                i, r = f.result()
                results[i] = r
                done += 1
                if done % 10 == 0:
                    logger.info(f"Progress: {done}/{len(audio_paths)}")
        return results


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser("Qwen3-ASR 推理客户端")
    p.add_argument("--local", action="store_true", help="本地 SDK 模式")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--forced_aligner", type=str, default=None)
    p.add_argument("--base_url", type=str, default="http://localhost:8000")
    p.add_argument("--audio", type=str, help="单条音频路径")
    p.add_argument("--audio_url", type=str)
    p.add_argument("--context", type=str, default="", help="销售员话术原文")
    p.add_argument("--language", type=str, default="Chinese")
    p.add_argument("--timestamps", action="store_true")
    p.add_argument("--audio_dir", type=str, help="批量识别目录")
    p.add_argument("--context_file", type=str, help="上下文文件 (JSONL): {audio_path, context}")
    p.add_argument("--output", type=str, default="asr_results.jsonl")
    p.add_argument("--max_workers", type=int, default=4)
    args = p.parse_args()

    # 创建客户端
    if args.local:
        client = LocalClient(args.model_path, forced_aligner=args.forced_aligner)
    else:
        client = VLLMClient(args.base_url)

    # 单条识别
    if args.audio or args.audio_url:
        if args.local:
            result = client.transcribe(
                audio=args.audio or args.audio_url,
                context=args.context, language=args.language,
                return_timestamps=args.timestamps,
            )
        else:
            result = client.transcribe(
                audio_path=args.audio, audio_url=args.audio_url,
                context=args.context,
            )
        print(f"\n{'='*50}")
        print(f"  结果: {result['text']}")
        print(f"  语言: {result.get('language', 'N/A')}")
        print(f"  耗时: {result['latency_ms']}ms")
        if args.context:
            print(f"  上下文: {args.context}")
        print(f"{'='*50}\n")
        return

    # 批量识别
    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
        audio_paths = sorted(
            str(p) for p in audio_dir.glob("*")
            if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
        )
        logger.info(f"Found {len(audio_paths)} audio files")

        contexts = [args.context] * len(audio_paths)
        if args.context_file:
            ctx_map = {}
            with open(args.context_file, "r") as f:
                for line in f:
                    it = json.loads(line.strip())
                    ctx_map[it["audio_path"]] = it.get("context", "")
            contexts = [ctx_map.get(p, args.context) for p in audio_paths]

        if args.local:
            results = client.transcribe_batch(audio_paths, contexts, args.language, args.timestamps)
        else:
            results = client.transcribe_batch(audio_paths, contexts, args.max_workers)

        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Saved to {args.output}")
        return

    p.print_help()


if __name__ == "__main__":
    main()
