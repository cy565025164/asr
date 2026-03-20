"""
Qwen3-Omni 多任务 HTTP 服务

接口：
    POST /infer  {"audio_path": "...", "context": "...", "tasks": ["asr", "emotion"], "max_tokens": 512}
    GET  /health
    GET  /tasks   返回支持的任务列表

用法：
    python server.py --model_path ./output_sft/final --port 9000
"""

import json
import time
import argparse
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 加载任务定义
TASKS_FILE = Path(__file__).parent.parent / "data" / "tasks.json"


def load_tasks_config():
    if TASKS_FILE.exists():
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # fallback
    return {
        "asr": {
            "system": "你是语音识别助手，请准确转录用户语音内容。",
            "user_prompt": "请准确识别语音内容。",
            "output_type": "text",
        },
        "emotion": {
            "system": "你是语音情绪分析助手，根据说话人的语气、语调、语速判断情绪。",
            "user_prompt": "请判断说话人的情绪状态。输出JSON格式。",
            "output_type": "json",
        },
        "gender": {
            "system": "你是语音性别识别助手，根据声音特征判断说话人性别。",
            "user_prompt": "请判断说话人的性别。输出JSON格式。",
            "output_type": "json",
        },
        "age": {
            "system": "你是语音年龄识别助手，根据声音特征判断说话人的大致年龄段。",
            "user_prompt": "请判断说话人的年龄段。输出JSON格式。",
            "output_type": "json",
        },
    }


class OmniService:
    def __init__(self, model_path, device="auto"):
        self.tasks_config = load_tasks_config()

        logger.info(f"Loading Qwen3-Omni from {model_path}")
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map=device,
            attn_implementation="flash_attention_2",
        )
        self.model.disable_talker()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        logger.info(f"Model loaded. Supported tasks: {list(self.tasks_config.keys())}")

    def infer_task(self, audio_path, task_id, context="", max_tokens=512):
        """执行单个任务"""
        task_cfg = self.tasks_config[task_id]

        system_text = task_cfg["system"]
        if context:
            system_text += f"\n销售员上一句：{context}"

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": task_cfg["user_prompt"]},
            ]},
        ]

        start = time.time()
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
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

        latency_ms = int((time.time() - start) * 1000)

        # JSON 任务尝试解析
        if task_cfg.get("output_type") == "json":
            try:
                parsed = json.loads(output)
                parsed["latency_ms"] = latency_ms
                return parsed
            except json.JSONDecodeError:
                return {"raw": output, "latency_ms": latency_ms}

        return {"text": output, "latency_ms": latency_ms}

    def infer(self, audio_path, tasks, context="", max_tokens=512):
        """执行多个任务"""
        results = {}
        for task_id in tasks:
            if task_id not in self.tasks_config:
                results[task_id] = {"error": f"未知任务: {task_id}"}
                continue
            results[task_id] = self.infer_task(audio_path, task_id, context, max_tokens)
        return results


def create_server(service, port):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))

            if self.path == "/infer":
                tasks = body.get("tasks", ["asr"])
                result = service.infer(
                    audio_path=body["audio_path"],
                    tasks=tasks,
                    context=body.get("context", ""),
                    max_tokens=body.get("max_tokens", 512),
                )
                self._respond(200, result)
            else:
                self._respond(404, {"error": "not found"})

        def do_GET(self):
            if self.path == "/health":
                self._respond(200, {"status": "ok"})
            elif self.path == "/tasks":
                self._respond(200, {
                    "tasks": {k: {"name": v.get("name", k), "system": v["system"]}
                              for k, v in service.tasks_config.items()}
                })
            else:
                self._respond(404, {})

        def _respond(self, code, data):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

        def log_message(self, fmt, *args):
            logger.info(f"{self.address_string()} {fmt % args}")

    server = HTTPServer(("0.0.0.0", port), Handler)
    logger.info(f"Qwen3-Omni service on :{port}")
    logger.info(f"  POST /infer   多任务推理 (tasks: {list(service.tasks_config.keys())})")
    logger.info(f"  GET  /tasks   查看支持的任务")
    logger.info(f"  GET  /health  健康检查")
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
