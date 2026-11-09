"""Run TrOCR on one handwritten line image."""
from __future__ import annotations

import argparse
import sys

from src.recognition.trocr import DEFAULT_MODEL_NAME, TrOCRRecognizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Nhận dạng một ảnh dòng chữ viết tay")
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh crop một dòng")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", help="auto, cpu hoặc cuda")
    args = parser.parse_args()
    try:
        recognizer = TrOCRRecognizer(model_name=args.model_name, device=args.device)
        print(f"Device: {recognizer.device}")
        print(f"Prediction: {recognizer.recognize(args.image)}")
    except Exception as exc:
        print(f"Lỗi nhận dạng: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
