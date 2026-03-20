"""
从电销录音标注文件构建 SFT 训练数据集

标注文件格式 (JSONL):
{"call_id": "call_001", "turn_id": 3, "audio_path": "/data/audio/call_001_user_03.wav",
 "correct_text": "嗯，我想了解一下你们那个百万医疗险", "sales_context": "请问您对我们的保险产品感兴趣吗？",
 "language": "Chinese"}

输出格式（对齐官方）:
{"audio": "...", "text": "language Chinese<asr_text>...", "prompt": "销售员：..."}
"""

import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_sft_dataset(annotation_file: str, output_file: str, language: str = "Chinese"):
    """将标注文件转换为 SFT 训练格式"""
    count = 0
    skipped = 0

    with open(annotation_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ann = json.loads(line)

            audio_path = ann.get("audio_path", "")
            correct_text = ann.get("correct_text", "").strip()
            sales_context = ann.get("sales_context", "").strip()
            lang = ann.get("language", language)

            if not audio_path or not correct_text:
                skipped += 1
                continue

            # 检查音频文件是否存在
            if not Path(audio_path).exists():
                logger.warning(f"Audio not found: {audio_path}")
                skipped += 1
                continue

            # 构建 SFT 格式
            sft_item = {
                "audio": audio_path,
                "text": f"language {lang}<asr_text>{correct_text}",
                "prompt": f"销售员：{sales_context}" if sales_context else "",
            }
            fout.write(json.dumps(sft_item, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Built {count} SFT examples (skipped {skipped}), saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="构建 SFT 数据集")
    parser.add_argument("--annotation_file", type=str, required=True, help="标注文件路径")
    parser.add_argument("--output_file", type=str, default="train_sft.jsonl", help="输出文件路径")
    parser.add_argument("--language", type=str, default="Chinese", help="默认语言")
    args = parser.parse_args()

    build_sft_dataset(args.annotation_file, args.output_file, args.language)


if __name__ == "__main__":
    main()
