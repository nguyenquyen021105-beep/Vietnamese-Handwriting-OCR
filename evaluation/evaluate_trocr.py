"""Evaluate TrOCR predictions from an image/reference CSV."""
from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from pathlib import Path

from evaluation.metrics import compute_metrics
from src.recognition.trocr import DEFAULT_MODEL_NAME, TrOCRRecognizer


def _value(row: dict[str, str], names: tuple[str, ...], row_number: int) -> str:
    for name in names:
        if row.get(name) is not None:
            return row[name]
    raise ValueError(f"CSV thiếu cột {names} tại dòng {row_number}.")


def evaluate(csv_path: Path, recognizer: TrOCRRecognizer, output_path: Path | None) -> dict:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            image_name = _value(row, ("image_path", "image", "filename"), row_number)
            reference = _value(row, ("reference", "text", "label"), row_number)
            image_path = Path(image_name)
            if not image_path.is_absolute():
                image_path = csv_path.parent / image_path
            prediction = recognizer.recognize(image_path)
            reference = unicodedata.normalize("NFC", reference)
            prediction = unicodedata.normalize("NFC", prediction)
            scores = compute_metrics(reference, prediction)
            rows.append({"image_path": str(image_path), "reference": reference,
                         "prediction": prediction, **scores})
    if not rows:
        raise ValueError("CSV không có dòng dữ liệu.")
    separator = "\n"
    reference_all = separator.join(row["reference"] for row in rows)
    prediction_all = separator.join(row["prediction"] for row in rows)
    total = compute_metrics(reference_all, prediction_all)
    if output_path is not None:
        fieldnames = ["image_path", "reference", "prediction", "cer", "wer"]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return {"rows": rows, "total": total}


def main() -> int:
    parser = argparse.ArgumentParser(description="Đánh giá CER/WER TrOCR")
    parser.add_argument("--csv", required=True, help="CSV có image_path và reference")
    parser.add_argument("--output", help="CSV xuất prediction và điểm từng dòng")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        recognizer = TrOCRRecognizer(model_name=args.model_name, device=args.device)
        result = evaluate(Path(args.csv), recognizer,
                          Path(args.output) if args.output else None)
    except Exception as exc:
        print(f"Lỗi đánh giá: {exc}", file=sys.stderr)
        return 1
    print(f"Rows: {len(result['rows'])}")
    print(f"Total CER: {result['total']['cer']:.6f}")
    print(f"Total WER: {result['total']['wer']:.6f}")
    if args.output:
        print(f"Đã ghi: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
