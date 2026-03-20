"""
Qwen3-ASR 推理客户端
支持：单条识别、批量识别、带上下文识别、流式识别

服务端：通过 deploy_vllm.sh 启动的 vLLM 服务

用法：
    # 单条识别
    python asr_client.py --audio /path/to/audio.wav

    # 带销售员上下文识别
    python asr_client.py --audio /path/to/audio.wav --context "销售员：请问您需要什么产品？"

    # 批量识别（目录下所有 wav 文件）
    python asr_client.py --audio_dir /path/to/audio_dir/ --output results.jsonl

    # 启动 HTTP 服务（供业务系统调用）
    python asr_client.py --serve --port 9000
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
# ASR 客户端核心类
# ============================================================

class Qwen3ASRClient:
    """Qwen3-ASR vLLM 推理客户端"""

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

        # 检查连接
        self._check_connection()

    def _check_connection(self):
        """检查 vLLM 服务是否可用"""
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    self.model_id = models[0]["id"]
                    logger.info(f"Connected to vLLM. Model: {self.model_id}")
                    return
            logger.warning(f"vLLM service responded but no models found")
        except requests.ConnectionError:
            logger.warning(f"Cannot connect to {self.base_url}, will retry on first call")
        self.model_id = None

    def _encode_audio_base64(self, audio_path: str) -> str:
        """将音频文件编码为 base64"""
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode("utf-8")

    def _build_messages(
        self,
        audio_path: Optional[str] = None,
        audio_url: Optional[str] = None,
        audio_base64: Optional[str] = None,
        context: Optional[str] = None,
        dialogue_history: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        构建 chat messages

        支持三种音频输入方式：
        1. audio_path: 本地文件路径（自动转 base64）
        2. audio_url: 远程 URL
        3. audio_base64: 已编码的 base64 字符串
        """
        messages = []

        # System message（含上下文指令）
        system_text = "你是一个电销场景的语音识别助手。请准确识别用户的语音内容。"
        if context or dialogue_history:
            system_text += "\n请结合以下对话上下文进行识别："
        messages.append({"role": "system", "content": system_text})

        # User message
        user_content = []

        # 添加上下文（文本部分）
        context_text = ""
        if dialogue_history:
            context_text = "【对话上下文】\n"
            for turn in dialogue_history:
                role = "销售员" if turn["role"] == "sales" else "用户"
                context_text += f"{role}：{turn['text']}\n"
            context_text += "\n【请识别以下用户语音】"
        elif context:
            context_text = f"【销售员上一句话】\n{context}\n\n【请识别以下用户语音】"

        if context_text:
            user_content.append({"type": "text", "text": context_text})

        # 添加音频
        if audio_path:
            audio_b64 = self._encode_audio_base64(audio_path)
            # 根据文件扩展名确定 MIME 类型
            ext = Path(audio_path).suffix.lower()
            mime_map = {".wav": "audio/wav", ".mp3": "audio/mp3", ".flac": "audio/flac", ".ogg": "audio/ogg"}
            mime_type = mime_map.get(ext, "audio/wav")
            user_content.append({
                "type": "audio_url",
                "audio_url": {"url": f"data:{mime_type};base64,{audio_b64}"},
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
        return messages

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
        单条音频识别

        Args:
            audio_path: 本地音频文件路径
            audio_url: 远程音频 URL
            audio_base64: base64 编码的音频数据
            context: 销售员上下文文本
            dialogue_history: 完整对话历史
            temperature: 生成温度（ASR 建议 0）
            max_tokens: 最大生成 token 数

        Returns:
            {
                "text": "识别结果文本",
                "language": "Chinese",
                "latency_ms": 123,
                "usage": {...}
            }
        """
        messages = self._build_messages(
            audio_path=audio_path,
            audio_url=audio_url,
            audio_base64=audio_base64,
            context=context,
            dialogue_history=dialogue_history,
        )

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.model_id:
            payload["model"] = self.model_id

        start_time = time.time()
        try:
            resp = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            return {"text": "", "error": str(e), "latency_ms": 0}

        latency_ms = int((time.time() - start_time) * 1000)
        result = resp.json()

        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})

        # 解析 ASR 输出（尝试用 qwen_asr 的解析器）
        language = self.language
        text = content
        try:
            from qwen_asr import parse_asr_output
            language, text = parse_asr_output(content)
        except (ImportError, Exception):
            # 如果没有 qwen_asr 包，直接用原始输出
            pass

        return {
            "text": text,
            "language": language,
            "raw_output": content,
            "latency_ms": latency_ms,
            "usage": usage,
        }

    def transcribe_batch(
        self,
        audio_paths: List[str],
        contexts: Optional[List[str]] = None,
        max_workers: int = 4,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量音频识别（并发请求）

        Args:
            audio_paths: 音频文件路径列表
            contexts: 对应的上下文列表（可选）
            max_workers: 并发线程数
            show_progress: 是否显示进度

        Returns:
            识别结果列表
        """
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
                if show_progress and completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(audio_paths)}")

        return results


# ============================================================
# HTTP 服务（供业务系统调用）
# ============================================================

def create_http_server(client: Qwen3ASRClient, port: int = 9000):
    """
    启动一个简单的 HTTP 服务，封装 ASR 调用

    API 接口：
        POST /asr
        Content-Type: application/json
        Body: {
            "audio_path": "/path/to/audio.wav",     # 服务器本地路径
            "audio_url": "https://...",              # 或远程 URL
            "audio_base64": "...",                   # 或 base64 数据
            "context": "销售员上一句话",               # 可选
            "dialogue_history": [...]                 # 可选
        }

        POST /asr/batch
        Body: {
            "items": [
                {"audio_path": "...", "context": "..."},
                ...
            ]
        }
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class ASRHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            if self.path == "/asr":
                result = client.transcribe(
                    audio_path=data.get("audio_path"),
                    audio_url=data.get("audio_url"),
                    audio_base64=data.get("audio_base64"),
                    context=data.get("context"),
                    dialogue_history=data.get("dialogue_history"),
                )
                self._respond(200, result)

            elif self.path == "/asr/batch":
                items = data.get("items", [])
                audio_paths = [it.get("audio_path") for it in items]
                contexts = [it.get("context") for it in items]
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
    logger.info(f"ASR HTTP service started on port {port}")
    logger.info(f"  POST /asr         - 单条识别")
    logger.info(f"  POST /asr/batch   - 批量识别")
    logger.info(f"  GET  /health      - 健康检查")
    server.serve_forever()


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR 推理客户端")

    # 连接参数
    parser.add_argument("--base_url", type=str, default="http://localhost:8000",
                        help="vLLM 服务地址")
    parser.add_argument("--timeout", type=int, default=60)

    # 单条识别
    parser.add_argument("--audio", type=str, help="音频文件路径")
    parser.add_argument("--audio_url", type=str, help="音频 URL")
    parser.add_argument("--context", type=str, help="销售员上下文文本")
    parser.add_argument("--language", type=str, default="Chinese")

    # 批量识别
    parser.add_argument("--audio_dir", type=str, help="音频目录（批量识别）")
    parser.add_argument("--output", type=str, default="asr_results.jsonl",
                        help="批量识别结果输出文件")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="并发线程数")

    # 上下文文件（批量时使用）
    parser.add_argument("--context_file", type=str,
                        help="上下文文件 (JSONL)，每行 {audio_path, context}")

    # HTTP 服务模式
    parser.add_argument("--serve", action="store_true",
                        help="启动 HTTP 服务模式")
    parser.add_argument("--port", type=int, default=9000,
                        help="HTTP 服务端口")

    args = parser.parse_args()

    # 创建客户端
    client = Qwen3ASRClient(
        base_url=args.base_url,
        timeout=args.timeout,
        language=args.language,
    )

    # ---- HTTP 服务模式 ----
    if args.serve:
        create_http_server(client, port=args.port)
        return

    # ---- 单条识别 ----
    if args.audio or args.audio_url:
        result = client.transcribe(
            audio_path=args.audio,
            audio_url=args.audio_url,
            context=args.context,
        )
        print(f"\n{'='*50}")
        print(f"  识别结果: {result['text']}")
        print(f"  语言:     {result.get('language', 'N/A')}")
        print(f"  耗时:     {result['latency_ms']}ms")
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
        logger.info(f"Found {len(audio_paths)} audio files in {audio_dir}")

        # 加载上下文（如果有）
        contexts = [None] * len(audio_paths)
        if args.context_file:
            context_map = {}
            with open(args.context_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    context_map[item["audio_path"]] = item.get("context", "")
            contexts = [context_map.get(p) for p in audio_paths]
            logger.info(f"Loaded {len(context_map)} context entries")

        # 批量推理
        results = client.transcribe_batch(
            audio_paths=audio_paths,
            contexts=contexts,
            max_workers=args.max_workers,
        )

        # 保存结果
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info(f"Results saved to {args.output}")

        # 统计
        total = len(results)
        errors = sum(1 for r in results if r.get("error"))
        avg_latency = sum(r["latency_ms"] for r in results) / max(total, 1)
        logger.info(f"Total: {total}, Errors: {errors}, Avg latency: {avg_latency:.0f}ms")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
