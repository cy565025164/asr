"""
Qwen3-Omni Transformers 本地部署

用法: python deploy_transformers.py --model_path Qwen/Qwen3-Omni-30B-A3B-Instruct --port 9000
"""

import json
import time
import argparse
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# 预定义 prompt
PROMPTS = {
    "asr": "请准确识别用户语音内容。",
    "analyze": "请分析说话人的性别、年龄段、情绪和购买意向。输出JSON格式。",
    "full": "请先识别用户语音内容，然后分析说话人特征。",
}


class OmniService:
    def __init__(self, model_path, device="auto"):
        logger.info(f"Loading Qwen3-Omni from {model_path}")
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map=device,
            attn_implementation="flash_attention_2",
        )
        self.model.disable_talker()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        logger.info("Model loaded")

    def infer(self, audio_path, context="", mode="asr", max_tokens=512):
        """
        推理

        Args:
            audio_path: 音频路径或 URL
            context: 销售员上下文
            mode: asr / analyze / full
            max_tokens: 最大生成 token
        """
        system_msg = "你是电销场景语音识别与分析助手。"
        if context:
            system_msg += f"销售员上一句：{context}"

        task_prompt = PROMPTS.get(mode, PROMPTS["asr"])

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": task_prompt},
            ]},
        ]

        start = time.time()

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        with torch.no_grad():
            text_ids, _ = self.model.generate(
                **inputs, return_audio=False,
                thinker_return_dict_in_generate=True,
                max_new_tokens=max_tokens,
            )

        output = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        latency_ms = int((time.time() - start) * 1000)
        return {"output": output, "mode": mode, "latency_ms": latency_ms}


def create_server(service, port):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))

            if self.path == "/infer":
                result = service.infer(
                    audio_path=body["audio_path"],
                    context=body.get("context", ""),
                    mode=body.get("mode", "asr"),
                    max_tokens=body.get("max_tokens", 512),
                )
                self._respond(200, result)
            elif self.path == "/infer/batch":
                results = []
                for item in body.get("items", []):
                    r = service.infer(
                        audio_path=item["audio_path"],
                        context=item.get("context", ""),
                        mode=item.get("mode", "asr"),
                    )
                    r["audio_path"] = item["audio_path"]
                    results.append(r)
                self._respond(200, {"results": results})
            else:
                self._respond(404, {"error": "not found"})

        def do_GET(self):
            self._respond(200, {"status": "ok"}) if self.path == "/health" else self._respond(404, {})

        def _respond(self, code, data):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

        def log_message(self, fmt, *args):
            logger.info(f"{self.address_string()} {fmt % args}")

    server = HTTPServer(("0.0.0.0", port), Handler)
    logger.info(f"Qwen3-Omni service on port {port}")
    logger.info(f"  POST /infer        单条推理 (mode: asr/analyze/full)")
    logger.info(f"  POST /infer/batch  批量推理")
    server.serve_forever()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--port", type=int, default=9000)
    args = p.parse_args()
    service = OmniService(args.model_path)
    create_server(service, args.port)


if __name__ == "__main__":
    main()
