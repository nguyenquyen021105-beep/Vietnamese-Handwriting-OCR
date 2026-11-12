"""Inference wrapper for Hugging Face TrOCR line recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import unicodedata

from PIL import Image
import torch

DEFAULT_MODEL_NAME = "microsoft/trocr-base-handwritten"


class TrOCRRecognizer:
    """Recognize one handwritten line with the original TrOCR processor."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None,
                 processor: Any | None = None, model: Any | None = None) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.processor = processor
        self.model = model
        if self.processor is None or self.model is None:
            self._load_components()
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device is None or device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            resolved = torch.device(device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Thiết bị không hợp lệ: {device!r}") from exc
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA được yêu cầu nhưng không khả dụng.")
        return resolved

    def _load_components(self) -> None:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise ImportError("Cần cài transformers: pip install -r requirements.txt") from exc
        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

    @staticmethod
    def _load_image(line_image: Image.Image | str | Path) -> Image.Image:
        if isinstance(line_image, (str, Path)):
            image_path = Path(line_image)
            if not image_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy ảnh dòng: {image_path}")
            with Image.open(image_path) as image:
                return image.convert("RGB")
        if isinstance(line_image, Image.Image):
            return line_image.convert("RGB")
        raise TypeError("line_image phải là PIL.Image.Image hoặc đường dẫn ảnh.")

    def recognize(self, line_image: Image.Image | str | Path) -> str:
        image = self._load_image(line_image)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return unicodedata.normalize("NFC", text).strip()
    
    def recognize_line(
        self,
        image: Image.Image | str | Path) -> dict:
        """
        Recognize a single handwritten text line.

        Returns:
            dict: OCR result for API integration.
        """
        text = self.recognize(image)

        return {
            "text": text,
            "confidence": None,
            "model": self.model_name,
            "status": "success"
        }

    def recognize_line_from_path(
        self,
        image_path: str | Path) -> dict:
        """
        Recognize a handwritten line from an image path.
        """
        return self.recognize_line(image_path)
