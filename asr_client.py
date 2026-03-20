"""
Qwen3-ASR 推理客户端（支持 context 热词 + vLLM 服务 + 本地推理）

支持两种推理方式：
  1. vLLM 服务模式：调用 deploy_vllm.sh 启动的 OpenAI 兼容 API
  2. 本地推理模式：直接加载模型，使用 qwen-asr SDK（支持 context 热词）

用法：
    # === vLLM 服务模式 ===
    # 单条识别
    python asr_client.py --audio /path/to/audio.wav --context "百万医疗险 保险"

    # 批量识别
    python asr_client.py --audio_dir /data/audios/ --output results.jsonl

    # 启动 HTTP 服务
    python asr_client.py --serve --port 9000

    # === 本地推理模式（推荐，支持 context 热词） ===
    python asr_client.py --local --model_path /path/to/model --audio /path/to/audio.wav --context "百万医疗险"

    # 本地批量识别
    python asr_client.py --local --model_path /path/to/model --audio_dir /data/audios/ --context "百万医疗险 保费"
"""

import os
import json
import time
import base64
import argparse
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 热词/上下文工具
# ============================================================

def extract_keywords_from_sales_text(sales_text: str) -> str:
    """
    从销售员话术中提取关键词，作为 ASR context

    Qwen3-ASR 的 context 参数接受空格分隔的关键词/术语，
    模型会在解码时偏向识别这些词。

    示例：
        输入: "您好，我给您介绍一下我们的百万医疗险，住院报销比例可达100%"
        输出: "百万医疗险 住院报销 100%"
    """
    import re

    # 简单提取：去除常见口语词，保留有意义的名词/术语
    stopwords = {
        "您好", "你好", "请问", "我们", "我是", "的", "了", "吗", "呢", "吧",
        "是", "有", "在", "就是", "那个", "这个", "一下", "可以", "能", "会",
        "给您", "跟您", "帮您", "对", "嗯", "啊", "哦", "好的", "行",
    }

    # 按标点分割
    segments = re.split(r'[，。！？、；：""''（）《》【】\s]+', sales_text)
    keywords = []
    for seg in segments:
        seg = seg.strip()
        if seg and seg not in stopwords and len(seg) >= 2:
            keywords.append(seg)

    return " ".join(keywords)


def load_hotwords_file(hotwords_path: str) -> str:
    """
    从热词文件加载，每行一个热词

    文件格式：
        百万医疗险
        住院报销
        免赔额
        等待期
    """
    hotwords = []
    with open(hotwords_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith("#"):
                hotwords.append(word)
    return " ".join(hotwords)


# ============================================================
# 本地推理客户端（使用 qwen-asr SDK，支持 context）
# ============================================================

class Qwen3ASRLocalClient:
    """
    本地推理客户端，直接使用 qwen-asr SDK

    优势：完整支持 context 参数，可传入热词提升术语识别率
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda:0",
        forced_aligner: Optional[str] = None,
        max_batch_size: int = 32,
        max_new_tokens: int = 512,
    ):
        import torch
        from qwen_asr import Qwen3ASRModel

        logger.info(f"Loading Qwen3-ASR from {model_path}")
        aligner_kwargs = {}
        if forced_aligner:
            aligner_kwargs = {
                "forced_aligner": forced_aligner,
                "forced_aligner_kwargs": dict(
                    dtype=torch.bfloat16,
                    device_map=device,
                ),
            }

        self.model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            max_inference_batch_size=max_batch_size,
            max_new_tokens=max_new_tokens,
            **aligner_kwargs,
        )
        logger.info("Model loaded successfully")

    def transcribe(
        self,
        audio_path: Optional[str] = None,
        audio_url: Optional[str] = None,
        context: Optional[str] = None,
        language: Optional[str] = "Chinese",
        return_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """
        单条识别

        Args:
            audio_path: 本地文件路径
            audio_url: 远程 URL
            context: 热词/术语，空格分隔（如 "百万医疗险 住院报销 免赔额"）
            language: 语言
            return_timestamps: 是否返回时间戳
        """
        audio_input = audio_path or audio_url
        if not audio_input:
            return {"text": "", "error": "No audio input provided"}

        start_time = time.time()
        results = self.model.transcribe(
            audio=audio_input,
            context=context or "",
            language=language,
            return_time_stamps=return_timestamps,
        )
        latency_ms = int((time.time() - start_time) * 1000)

        r = results[0]
        output = {
            "text": r.text,
            "language": r.language,
            "latency_ms": latency_ms,
        }
        if return_timestamps and r.time_stamps:
            output["timestamps"] = [
                {"text": ts.text, "start": ts.start_time, "end": ts.end_time}
                for ts in r.time_stamps
            ]
        return output

    def transcribe_batch(
        self,
        audio_paths: List[str],
        contexts: Optional[List[str]] = None,
        language: Optional[str] = "Chinese",
        return_timestamps: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        批量识别（SDK 原生批量，效率最高）
        """
        contexts = contexts or [""] * len(audio_paths)
        languages = [language] * len(audio_paths)

        start_time = time.time()
        results = self.model.transcribe(
            audio=audio_paths,
            context=contexts,
            language=languages,
            return_time_stamps=return_timestamps,
        )
        total_latency = int((time.time() - start_time) * 1000)

        outputs = []
        for i, r in enumerate(results):
            out = {
                "audio_path": audio_paths[i],
                "text": r.text,
                "language": r.language,
                "latency_ms": total_latency // len(audio_paths),
            }
            if return_timestamps and r.time_stamps:
                out["timestamps"] = [
                    {"text": ts.text, "start": ts.start_time, "end": ts.end_time}
                    for ts in r.time_stamps
                ]
            outputs.append(out)

        logger.info(
            f"Batch done: {len(audio_paths)} files, "
            f"total {total_latency}ms, avg {total_latency//len(audio_paths)}ms/file"
        )
        return outputs


# ============================================================
# vLLM 远程推理客户端
# ============================================================

class Qwen3ASRVLLMClient:
    """vLLM 服务推理客户端（OpenAI 兼容 API）"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 60,
        language: str = "Chinese",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/v1/chat/completions"
        self.timeout = timeout
        self.language = language
        self.model_id = None
        self._check_connection()

    def _check_connection(self):
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    self.model_id = models[0]["id"]
                    logger.info(f"Connected to vLLM. Model: {self.model_id}")
                    return
        except requests.ConnectionError:
            logger.warning(f"Cannot connect to {self.base_url}")

    def _encode_audio_base64(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def transcribe(
        self,
        audio_path: Optional[str] = None,
        audio_url: Optional[str] = None,
        audio_base64: Optional[str] = None,
        context: Optional[str] = None,
        dialogue_history: Optional[List[Dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        单条识别

        注意：vLLM 模式下 context 通过 system prompt 注入，
        效果不如本地推理的原生 context 参数。
        """
        messages = []

        # System message
        system_text = "你是一个电销场景的语音识别助手。请准确识别用户的语音内容。"
        if context:
            system_text += f"\n请特别注意以下术语的准确识别：{context}"
        messages.append({"role": "system", "content": system_text})

        # User content
        user_content = []

        # 对话上下文
        context_text = ""
        if dialogue_history:
            context_text = "【对话上下文】\n"
            for turn in dialogue_history:
                role = "销售员" if turn["role"] == "sales" else "用户"
                context_text += f"{role}：{turn['text']}\n"
            context_text += "\n【请识别以下用户语音】"
        if context_text:
            user_content.append({"type": "text", "text": context_text})

        # 音频
        if audio_path:
            audio_b64 = self._encode_audio_base64(audio_path)
            ext = Path(audio_path).suffix.lower()
            mime_map = {".wav": "audio/wav", ".mp3": "audio/mp3", ".flac": "audio/flac"}
            mime = mime_map.get(ext, "audio/wav")
            user_content.append({
                "type": "audio_url",
                "audio_url": {"url": f"data:{mime};base64,{audio_b64}"},
            })
        elif audio_url:
            user_content.append({
                "type": "audio_url",
                "audio_url": {"url": audio_url},
            })
        elif audio_base64:
            user_content.append({
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"},
            })

        messages.append({"role": "user", "content": user_content})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.model_id:
            payload["model"] = self.model_id

        start_time = time.time()
        try:
            resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            return {"text": "", "error": str(e), "latency_ms": 0}

        latency_ms = int((time.time() - start_time) * 1000)
        result = resp.json()
        content = result["choices"][0]["message"]["content"]

        text = content
        language = self.language
        try:
            from qwen_asr import parse_asr_output
            language, text = parse_asr_output(content)
        except (ImportError, Exception):
            pass

        return {
            "text": text,
            "language": language,
            "raw_output": content,
            "latency_ms": latency_ms,
            "usage": result.get("usage", {}),
        }

    def transcribe_batch(
        self,
        audio_paths: List[str],
        contexts: Optional[List[str]] = None,
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        results = [None] * len(audio_paths)
        contexts = contexts or [None] * len(audio_paths)

        def _process(idx, path, ctx):
            result = self.transcribe(audio_path=path, context=ctx)
            result["audio_path"] = path
            return idx, result

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process, i, p, c): i
                for i, (p, c) in enumerate(zip(audio_paths, contexts))
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(audio_paths)}")

        return results


# ============================================================
# HTTP 服务（封装两种客户端）
# ============================================================

def create_http_server(client, port: int = 9000):
    """
    HTTP 服务，供业务系统调用

    POST /asr
    {
        "audio_path": "/path/to/audio.wav",   # 或 audio_url / audio_base64
        "context": "百万医疗险 住院报销",       # 热词（空格分隔）
        "sales_text": "您好，给您介绍百万医疗险",  # 或直接传销售员原文，自动提取热词
        "dialogue_history": [...]               # 可选
    }

    POST /asr/batch
    {
        "items": [{"audio_path": "...", "context": "..."}],
        "max_workers": 4
    }

    GET /health
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class ASRHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            if self.path == "/asr":
                # 支持传 sales_text 自动提取热词
                context = data.get("context")
                if not context and data.get("sales_text"):
                    context = extract_keywords_from_sales_text(data["sales_text"])

                result = client.transcribe(
                    audio_path=data.get("audio_path"),
                    audio_url=data.get("audio_url"),
                    audio_base64=data.get("audio_base64"),
                    context=context,
                    dialogue_history=data.get("dialogue_history"),
                )
                if context:
                    result["context_used"] = context
                self._respond(200, result)

            elif self.path == "/asr/batch":
                items = data.get("items", [])
                audio_paths = [it.get("audio_path") for it in items]
                contexts = []
                for it in items:
                    ctx = it.get("context")
                    if not ctx and it.get("sales_text"):
                        ctx = extract_keywords_from_sales_text(it["sales_text"])
                    contexts.append(ctx)
                results = client.transcribe_batch(
                    audio_paths=audio_paths,
                    contexts=contexts,
                    max_workers=data.get("max_workers", 4),
                )
                self._respond(200, {"results": results})

            elif self.path == "/health":
                self._respond(200, {"status": "ok"})
            else:
                self._respond(404, {"error": "not found"})

        def do_GET(self):
            if self.path == "/health":
                self._respond(200, {"status": "ok"})
            else:
                self._respond(404, {"error": "not found"})

        def _respond(self, code, data):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

        def log_message(self, format, *args):
            logger.info(f"{self.address_string()} - {format % args}")

    server = HTTPServer(("0.0.0.0", port), ASRHandler)
    logger.info(f"ASR HTTP service on port {port}")
    logger.info(f"  POST /asr         单条识别")
    logger.info(f"  POST /asr/batch   批量识别")
    logger.info(f"  GET  /health      健康检查")
    server.serve_forever()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR 推理客户端")

    # 模式选择
    parser.add_argument("--local", action="store_true",
                        help="本地推理模式（推荐，完整支持 context 热词）")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B",
                        help="本地模式的模型路径")
    parser.add_argument("--forced_aligner", type=str, default=None,
                        help="ForcedAligner 模型路径（可选，用于时间戳）")

    # vLLM 服务模式
    parser.add_argument("--base_url", type=str, default="http://localhost:8000",
                        help="vLLM 服务地址")
    parser.add_argument("--timeout", type=int, default=60)

    # 识别参数
    parser.add_argument("--audio", type=str, help="音频文件路径")
    parser.add_argument("--audio_url", type=str, help="音频 URL")
    parser.add_argument("--context", type=str,
                        help="热词/术语，空格分隔（如 '百万医疗险 住院报销'）")
    parser.add_argument("--hotwords_file", type=str,
                        help="热词文件路径（每行一个热词）")
    parser.add_argument("--sales_text", type=str,
                        help="销售员原文（自动提取热词）")
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--timestamps", action="store_true",
                        help="返回时间戳（本地模式 + ForcedAligner）")

    # 批量
    parser.add_argument("--audio_dir", type=str, help="音频目录")
    parser.add_argument("--output", type=str, default="asr_results.jsonl")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--context_file", type=str,
                        help="上下文文件 (JSONL): {audio_path, context/sales_text}")

    # HTTP 服务
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=9000)

    args = parser.parse_args()

    # ---- 合并 context 来源 ----
    context = args.context or ""
    if args.hotwords_file:
        context = load_hotwords_file(args.hotwords_file)
    elif args.sales_text and not context:
        context = extract_keywords_from_sales_text(args.sales_text)

    if context:
        logger.info(f"Context keywords: {context}")

    # ---- 创建客户端 ----
    if args.local:
        client = Qwen3ASRLocalClient(
            model_path=args.model_path,
            forced_aligner=args.forced_aligner,
        )
    else:
        client = Qwen3ASRVLLMClient(
            base_url=args.base_url,
            timeout=args.timeout,
            language=args.language,
        )

    # ---- HTTP 服务 ----
    if args.serve:
        create_http_server(client, port=args.port)
        return

    # ---- 单条识别 ----
    if args.audio or args.audio_url:
        if args.local:
            result = client.transcribe(
                audio_path=args.audio,
                audio_url=args.audio_url,
                context=context,
                language=args.language,
                return_timestamps=args.timestamps,
            )
        else:
            result = client.transcribe(
                audio_path=args.audio,
                audio_url=args.audio_url,
                context=context,
            )
        print(f"\n{'='*50}")
        print(f"  识别结果: {result['text']}")
        print(f"  语言:     {result.get('language', 'N/A')}")
        print(f"  耗时:     {result['latency_ms']}ms")
        if context:
            print(f"  热词:     {context}")
        if result.get("timestamps"):
            print(f"  时间戳:   {len(result['timestamps'])} segments")
        if result.get("error"):
            print(f"  错误:     {result['error']}")
        print(f"{'='*50}\n")
        return

    # ---- 批量识别 ----
    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
        audio_paths = sorted(
            str(p) for p in audio_dir.glob("*")
            if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        )
        logger.info(f"Found {len(audio_paths)} audio files")

        # 加载每条音频的上下文
        contexts = [context] * len(audio_paths)
        if args.context_file:
            context_map = {}
            with open(args.context_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    ctx = item.get("context", "")
                    if not ctx and item.get("sales_text"):
                        ctx = extract_keywords_from_sales_text(item["sales_text"])
                    context_map[item["audio_path"]] = ctx
            contexts = [context_map.get(p, context) for p in audio_paths]

        if args.local:
            results = client.transcribe_batch(
                audio_paths=audio_paths,
                contexts=contexts,
                language=args.language,
                return_timestamps=args.timestamps,
            )
        else:
            results = client.transcribe_batch(
                audio_paths=audio_paths,
                contexts=contexts,
                max_workers=args.max_workers,
            )

        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        total = len(results)
        errors = sum(1 for r in results if r.get("error"))
        avg_lat = sum(r.get("latency_ms", 0) for r in results) / max(total, 1)
        logger.info(f"Done: {total} files, {errors} errors, avg {avg_lat:.0f}ms")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
