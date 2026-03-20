"""
Qwen3-ASR HTTP 推理服务

封装本地 SDK 或 vLLM 后端，提供统一的 HTTP API。

用法:
    # 本地 SDK 后端
    python asr_server.py --backend local --model_path ./output_dpo/final --port 9000

    # vLLM 后端（需要先启动 vLLM 服务）
    python asr_server.py --backend vllm --vllm_url http://localhost:8000 --port 9000

API:
    POST /asr
    {
        "audio_path": "/path/to/audio.wav",
        "context": "销售员：请问您对保险产品感兴趣吗？",
        "language": "Chinese"
    }

    POST /asr/batch
    {
        "items": [{"audio_path": "...", "context": "..."}],
        "language": "Chinese"
    }

    GET /health
"""

import json
import argparse
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_backend(args):
    if args.backend == "local":
        from asr_client import LocalClient
        return LocalClient(args.model_path, forced_aligner=args.forced_aligner)
    else:
        from asr_client import VLLMClient
        return VLLMClient(args.vllm_url)


class Handler(BaseHTTPRequestHandler):
    backend = None

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))

        if self.path == "/asr":
            audio = body.get("audio_path") or body.get("audio_url")
            context = body.get("context", "")
            language = body.get("language", "Chinese")

            if hasattr(self.backend, "model"):
                # 本地模式
                result = self.backend.transcribe(
                    audio=audio, context=context, language=language,
                    return_timestamps=body.get("return_timestamps", False),
                )
            else:
                # vLLM 模式
                result = self.backend.transcribe(
                    audio_path=body.get("audio_path"),
                    audio_url=body.get("audio_url"),
                    context=context,
                )
            self._respond(200, result)

        elif self.path == "/asr/batch":
            items = body.get("items", [])
            audios = [it.get("audio_path") or it.get("audio_url") for it in items]
            contexts = [it.get("context", "") for it in items]
            language = body.get("language", "Chinese")

            results = self.backend.transcribe_batch(audios, contexts)
            self._respond(200, {"results": results})

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
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def log_message(self, fmt, *args):
        logger.info(f"{self.address_string()} {fmt % args}")


def main():
    p = argparse.ArgumentParser("Qwen3-ASR HTTP Server")
    p.add_argument("--backend", choices=["local", "vllm"], default="local")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--forced_aligner", type=str, default=None)
    p.add_argument("--vllm_url", type=str, default="http://localhost:8000")
    p.add_argument("--port", type=int, default=9000)
    args = p.parse_args()

    Handler.backend = create_backend(args)
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    logger.info(f"ASR server on port {args.port} (backend={args.backend})")
    logger.info(f"  POST /asr        单条识别")
    logger.info(f"  POST /asr/batch  批量识别")
    logger.info(f"  GET  /health     健康检查")
    server.serve_forever()


if __name__ == "__main__":
    main()
