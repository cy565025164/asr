"""
双模型并行推理调度器

实时链路：Qwen3-ASR（ASR + context）→ 毫秒级返回转文字
异步链路：Qwen3-Omni（性别/年龄/情绪）→ 异步回写标签

同一段音频同时丢给两个模型，ASR 不等 Omni。

用法：
    # 启动调度器（需要两个后端服务已启动）
    python pipeline.py --asr_url http://localhost:9000 --omni_url http://localhost:9001 --port 8080

    # 或直接本地加载两个模型
    python pipeline.py --local \
        --asr_model ./output_dpo/final \
        --omni_model Qwen/Qwen3-Omni-30B-A3B \
        --port 8080

API:
    POST /recognize
    {
        "audio_path": "/data/audio/user.wav",
        "context": "销售员：请问您需要什么产品？",
        "need_analysis": true
    }

    返回：
    {
        "asr": {"text": "嗯，我想了解百万医疗险", "language": "Chinese", "latency_ms": 120},
        "analysis": {"status": "pending", "task_id": "xxx"}
    }

    GET /analysis/{task_id}
    返回异步分析结果
"""

import json
import time
import uuid
import argparse
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import requests

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 异步任务存储
# ============================================================

class TaskStore:
    """简单的内存任务存储"""
    def __init__(self):
        self._tasks: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def create(self, task_id: str):
        with self._lock:
            self._tasks[task_id] = {"status": "pending", "result": None}

    def complete(self, task_id: str, result: Dict):
        with self._lock:
            self._tasks[task_id] = {"status": "completed", "result": result}

    def fail(self, task_id: str, error: str):
        with self._lock:
            self._tasks[task_id] = {"status": "failed", "error": error}

    def get(self, task_id: str) -> Optional[Dict]:
        with self._lock:
            return self._tasks.get(task_id)


# ============================================================
# Pipeline
# ============================================================

class DualModelPipeline:
    """双模型并行推理 Pipeline"""

    def __init__(self, asr_url: str, omni_url: str):
        self.asr_url = asr_url.rstrip("/")
        self.omni_url = omni_url.rstrip("/")
        self.task_store = TaskStore()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def recognize(self, audio_path: str, context: str = "",
                  need_analysis: bool = True) -> Dict[str, Any]:
        """
        主入口：同步返回 ASR，异步提交 Omni 分析

        Returns:
            {
                "asr": {...},           # ASR 结果（同步）
                "analysis": {           # 分析任务（异步）
                    "status": "pending",
                    "task_id": "xxx"
                }
            }
        """
        result = {}

        # 1. 同步调用 ASR（实时链路）
        try:
            asr_resp = requests.post(
                f"{self.asr_url}/asr",
                json={"audio_path": audio_path, "context": context},
                timeout=30,
            )
            result["asr"] = asr_resp.json()
        except Exception as e:
            result["asr"] = {"error": str(e)}

        # 2. 异步提交 Omni 分析
        if need_analysis:
            task_id = str(uuid.uuid4())[:8]
            self.task_store.create(task_id)
            self.executor.submit(self._run_analysis, task_id, audio_path, context)
            result["analysis"] = {"status": "pending", "task_id": task_id}
        else:
            result["analysis"] = {"status": "skipped"}

        return result

    def _run_analysis(self, task_id: str, audio_path: str, context: str):
        """后台执行 Omni 分析"""
        try:
            resp = requests.post(
                f"{self.omni_url}/analyze",
                json={"audio_path": audio_path, "sales_context": context},
                timeout=120,
            )
            self.task_store.complete(task_id, resp.json())
            logger.info(f"Analysis {task_id} completed")
        except Exception as e:
            self.task_store.fail(task_id, str(e))
            logger.error(f"Analysis {task_id} failed: {e}")

    def get_analysis(self, task_id: str) -> Optional[Dict]:
        return self.task_store.get(task_id)


# ============================================================
# HTTP 服务
# ============================================================

def create_server(pipeline: DualModelPipeline, port: int):

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))

            if self.path == "/recognize":
                result = pipeline.recognize(
                    audio_path=body["audio_path"],
                    context=body.get("context", ""),
                    need_analysis=body.get("need_analysis", True),
                )
                self._respond(200, result)
            else:
                self._respond(404, {"error": "not found"})

        def do_GET(self):
            if self.path.startswith("/analysis/"):
                task_id = self.path.split("/")[-1]
                result = pipeline.get_analysis(task_id)
                if result:
                    self._respond(200, result)
                else:
                    self._respond(404, {"error": "task not found"})
            elif self.path == "/health":
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
    logger.info(f"Pipeline service on port {port}")
    logger.info(f"  ASR backend:  {pipeline.asr_url}")
    logger.info(f"  Omni backend: {pipeline.omni_url}")
    logger.info(f"  POST /recognize       实时ASR + 异步分析")
    logger.info(f"  GET  /analysis/{{id}}   查询分析结果")
    server.serve_forever()


def main():
    p = argparse.ArgumentParser("Dual-Model Pipeline")
    p.add_argument("--asr_url", type=str, default="http://localhost:9000",
                   help="ASR 服务地址（deploy_local.py 或 vLLM）")
    p.add_argument("--omni_url", type=str, default="http://localhost:9001",
                   help="Omni 分析服务地址（omni_analyzer.py）")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    pipeline = DualModelPipeline(args.asr_url, args.omni_url)
    create_server(pipeline, args.port)


if __name__ == "__main__":
    main()
