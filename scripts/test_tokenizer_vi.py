"""Smoke-test the original TrOCR tokenizer with Vietnamese text."""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from pathlib import Path

DEFAULT_MODEL_NAME = "microsoft/trocr-base-handwritten"
DEFAULT_REPORT = Path("reports/t01_tokenizer_results.csv")


def build_samples() -> list[str]:
    """Return the 20 deterministic strings used by the T01 tokenizer check."""
    nfd = unicodedata.normalize("NFD", "Tiếng Việt dùng dấu tổ hợp: ấ, ề, ộ, ử, ỹ.")
    return [
        "Tiếng Việt có dấu: ă â ê ô ơ ư đ.",
        "Việt Nam / VIỆT NAM / việt nam.",
        "Năm 2026, số 12,5 và ngày 04/11/2026.",
        "TP.HCM, PGS.TS., UBND và HĐND.",
        nfd,
        "Cà phê sữa đá, bánh mì, phở và cơm tấm.",
        "ĐẶC BIỆT: Ắ, Ằ, Ấ, Ầ, Ắ, Ẽ, Ỉ, Ố, Ồ, Ỗ, Ờ, Ừ, Ỷ.",
        "Mã đơn hàng VN-2026-001234; tổng tiền: 1.250.000₫.",
        "Ngày sinh: 01-02-1990; hạn dùng: 31/12/2030.",
        "Email: support@example.vn — Hotline: (+84) 1900 1234.",
        "TS. Nguyễn Văn A phát biểu: “Chất lượng là ưu tiên số một.”",
        "Lưu ý: không tự động sửa tên riêng, chữ số hoặc viết tắt.",
        "A/B testing: accuracy 98,5%; loss = 0,042; batch_size=16.",
        "Câu dài: Sáng thứ Hai, nhóm nghiên cứu kiểm tra dữ liệu, mô hình, kết quả và báo cáo cuối kỳ.",
        "Hà Nội — Đà Nẵng — Thành phố Hồ Chí Minh — Cần Thơ.",
        "Viết tắt: CNTT, AI, OCR, PDF, CSV, API, GPU và CPU.",
        "Ký tự đặc biệt: (a), [b], {c}, #hashtag, @user, +84, 50%.",
        "Trời hôm nay đẹp quá! Bạn có khỏe không? Mình ổn, cảm ơn nhé.",
        "Số thực: -3,14; phân số 1/2; khoảng 10–20; nhiệt độ 36°C.",
        "Kết thúc thử nghiệm — tiếng Việt chuẩn NFC: Ảnh, nhận dạng, huấn luyện.",
    ]


def run_check(tokenizer, samples: list[str]) -> list[dict[str, object]]:
    results = []
    for index, original in enumerate(samples, start=1):
        nfc_input = unicodedata.normalize("NFC", original)
        token_ids = tokenizer.encode(nfc_input, add_special_tokens=True)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        decoded_nfc = unicodedata.normalize("NFC", decoded)
        results.append(
            {
                "id": index,
                "input_text": original,
                "decoded_text": decoded_nfc,
                "token_count": len(token_ids),
                "nfc_equal": decoded_nfc == nfc_input,
            }
        )
    return results


def write_report(output_path: Path, results: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "input_text", "decoded_text", "token_count", "nfc_equal"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Kiểm tra tokenizer TrOCR tiếng Việt")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    try:
        from transformers import TrOCRProcessor

        processor = TrOCRProcessor.from_pretrained(args.model_name)
    except Exception as exc:
        print(f"Không thể tải processor/tokenizer: {exc}", file=sys.stderr)
        return 1

    results = run_check(processor.tokenizer, build_samples())
    write_report(args.output, results)

    token_counts = [int(row["token_count"]) for row in results]
    pass_count = sum(bool(row["nfc_equal"]) for row in results)
    fail_count = len(results) - pass_count
    for row in results:
        status = "PASS" if row["nfc_equal"] else "FAIL"
        print(f"{status} {row['id']:02d}: tokens={row['token_count']}")
    print(f"PASS: {pass_count} | FAIL: {fail_count}")
    print(f"Token trung bình: {sum(token_counts) / len(token_counts):.2f}")
    print(f"Token lớn nhất: {max(token_counts)}")
    print(f"Report: {args.output}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
