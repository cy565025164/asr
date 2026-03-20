"""
Qwen3-Omni 音频理解服务

异步分析音频的说话人属性：性别、年龄段、情绪状态。
与 Qwen3-ASR 实时识别并行运行，不阻塞主链路。

用法：
    # 启动服务
    python omni_analyzer.py --model_path Qwen/Qwen3-Omni-30B-A3B --port 9001

    # 调用
    curl -X POST http://localhost:9001/analyze \
        -H "Content-Type: application/json" \
        -d '{"audio_path": "/data/audio/user.wav", "sales_context": "销售员：请问您需要什么产品？"}'
"""

import json
import time
import argparse
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any

import torch

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Qwen3-Omni 分析器
# ============================================================

ANALYZE_PROMPT = """请分析这段电销对话中用户的语音，输出以下信息：

1. 性别：男/女
2. 年龄段：青年(18-35)/中年(36-55)/老年(55+)
3. 情绪状态：平静/积极/犹豫/不耐烦/愤怒/开心
4. 情绪强度：低/中/高
5. 语速感受：慢/正常/快

请严格以 JSON 格式输出，不要输出其他内容：
{"gender": "男/女", "age_group": "青年/中年/老年", "emotion": "...", "emotion_intensity": "低/中/高", "speech_rate": "慢/正常/快"}"""

ANALYZE_WITH_CONTEXT_PROMPT = """请分析这段电销对话中用户的语音，输出以下信息。

对话上下文：
{context}

分析维度：
1. 性别：男/女
2. 年龄段：青年(18-35)/中年(36-55)/老年(55+)
3. 情绪状态：平静/积极/犹豫/不耐烦/愤怒/开心
4. 情绪强度：低/中/高
5. 语速感受：慢/正常/快
6. 购买意向：无意向/观望/有兴趣/强意向

请严格以 JSON 格式输出，不要输出其他内容：
{"gender": "男/女", "age_group": "青年/中年/老年", "emotion": "...", "emotion_intensity": "低/中/高", "speech_rate": "慢/正常/快", "purchase_intent": "..."}"""


class OmniAnalyzer:
    """Qwen3-Omni 音频分析器"""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info(f"Loading Qwen3-Omni from {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        logger.info("Qwen3-Omni loaded")

    def analyze(
        self,
        audio_path: str,
        sales_context: str = "",
    ) -> Dict[str, Any]:
        """
        分析音频中的说话人属性

        Args:
            audio_path: 音频文件路径
            sales_context: 销售员上下文（可选，用于判断购买意向）

        Returns:
            {"gender": "...", "age_group": "...", "emotion": "...", ...}
        """
        # 构建 prompt
        if sales_context:
            prompt = ANALYZE_WITH_CONTEXT_PROMPT.format(context=sales_context)
        else:
            prompt = ANALYZE_PROMPT

        messages = [
            {"role": "system", "content": "你是一个专业的语音分析助手。"},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ]},
        ]

        start = time.time()

        # 处理输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text, audios=[audio_path],
            return_tensors="pt", padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
            )

        # 只取生成部分
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated, skip_special_tokens=True).strip()

        latency_ms = int((time.time() - start) * 1000)

        # 解析 JSON
        result = self._parse_response(response)
        result["latency_ms"] = latency_ms
        result["raw_output"] = response

        return result

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """尝试从模型输出中解析 JSON"""
        import re
        # 尝试提取 JSON
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # 解析失败，返回原始文本
        return {"parse_error": True, "raw": response}


# ============================================================
# HTTP 服务
# ============================================================

def create_server(analyzer: OmniAnalyzer, port: int):

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))

            if self.path == "/analyze":
                result = analyzer.analyze(
                    audio_path=body["audio_path"],
                    sales_context=body.get("sales_context", ""),
                )
                self._respond(200, result)

            elif self.path == "/analyze/batch":
                items = body.get("items", [])
                results = []
                for item in items:
                    r = analyzer.analyze(
                        audio_path=item["audio_path"],
                        sales_context=item.get("sales_context", ""),
                    )
                    r["audio_path"] = item["audio_path"]
                    results.append(r)
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
    logger.info(f"Omni analyzer on port {port}")
    logger.info(f"  POST /analyze        单条分析")
    logger.info(f"  POST /analyze/batch   批量分析")
    server.serve_forever()


def main():
    p = argparse.ArgumentParser("Qwen3-Omni Audio Analyzer")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-Omni-30B-A3B")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--port", type=int, default=9001)
    args = p.parse_args()

    analyzer = OmniAnalyzer(args.model_path, args.device)
    create_server(analyzer, args.port)


if __name__ == "__main__":
    main()
