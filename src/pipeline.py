"""TODO: assemble preprocessing → line detection → TrOCR → correction.
Do not silently replace Faithful OCR with corrected text.
"""
from dataclasses import dataclass

@dataclass
class OCRResult:
    faithful_text: str
    corrected_text: str | None = None


def run_pipeline(image_path: str) -> OCRResult:
    raise NotImplementedError("End-to-end pipeline not integrated")
