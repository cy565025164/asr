"""
Qwen3-ASR 本地 SDK 部署（支持 context 热词）

启动一个 HTTP 服务，内部使用 qwen-asr SDK 进行推理，
完整支持 context 参数。

用法:
    python deploy_local.py --model_path ./output_dpo/final --port 9000
"""

import json
import time
import argparse
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
from qwen_asr import Qwen3ASRModel

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ASRService:
    """ASR 推理服务"""

    def __init__(self, model_path: str, device: str = "cuda:0",
                 forced_aligner: str = None, max_batch_size: int = 32):
        logger.info(f"Loading model from {model_path}")
        aligner_kwargs = {}
        if forced_aligner:
            aligner_kwargs = {
                "forced_aligner": forced_aligner,
                "forced_aligner_kwargs": dict(dtype=torch.bfloat16, device_map=device),
            }

        self.model = Qwen3ASRModel.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device,
            max_inference_batch_size=max_batch_size, max_new_tokens=512,
            **aligner_kwargs,
        )
        logger.info("Model loaded")

    def transcribe(self, audio, context="", language="Chinese", return_timestamps=False):
        """
        识别音频

        Args:
            audio: 文件路径 / URL / 路径列表
            context: 销售员话术原文（直接传给模型 context 参数）
            language: 语言
            return_timestamps: 是否返回时间戳
        """
        start = time.time()
        is_batch = isinstance(audio, list)

        results = self.model.transcribe(
            audio=audio,
            context=context if is_batch else (context or ""),
            language=language,
            return_time_stamps=return_timestamps,
        )

        latency_ms = int((time.time() - start) * 1000)
        outputs = []
        for r in results:
            out = {"text": r.text, "language": r.language, "latency_ms": latency_ms}
            if return_timestamps and r.time_stamps:
                out["timestamps"] = [
                    {"text": ts.text, "start": ts.start_time, "end": ts.end_time}
                    for ts in r.time_stamps
                ]
            outputs.append(out)
        return outputs if is_batch else outputs[0]


def create_server(service: ASRService, port: int):
    """创建 HTTP 服务"""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))

            if self.path == "/asr":
                result = service.transcribe(
                    audio=body.get("audio_path") or body.get("audio_url"),
                    context=body.get("context", ""),
                    language=body.get("language", "Chinese"),
                    return_timestamps=body.get("return_timestamps", False),
                )
                self._respond(200, result)

            elif self.path == "/asr/batch":
                items = body.get("items", [])
                audios = [it.get("audio_path") or it.get("audio_url") for it in items]
                contexts = [it.get("context", "") for it in items]
                results = service.transcribe(
                    audio=audios, context=contexts,
                    language=body.get("language", "Chinese"),
                )
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

    server = HTTPServer(("0.0.0.0", port), Handler)
    logger.info(f"ASR service on port {port}")
    logger.info(f"  POST /asr        单条识别")
    logger.info(f"  POST /asr/batch  批量识别")
    logger.info(f"  GET  /health     健康检查")
    server.serve_forever()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--forced_aligner", type=str, default=None)
    p.add_argument("--port", type=int, default=9000)
    args = p.parse_args()

    service = ASRService(args.model_path, args.device, args.forced_aligner)
    create_server(service, args.port)


if __name__ == "__main__":
    main()
