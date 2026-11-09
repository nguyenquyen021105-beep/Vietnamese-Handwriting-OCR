from pathlib import Path
import unicodedata

import pytest
import torch
from PIL import Image

from src.recognition.trocr import TrOCRRecognizer


class FakeProcessor:
    def __call__(self, images, return_tensors):
        assert images.mode == "RGB"
        return {"pixel_values": torch.zeros((1, 3, 8, 8))}

    def batch_decode(self, generated_ids, skip_special_tokens):
        assert skip_special_tokens is True
        return ["Tiếng Việt"]


class FakeModel:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.was_eval = True
        return self

    def generate(self, pixel_values):
        self.generated_with = pixel_values
        return torch.tensor([[1, 2]])


def make_recognizer():
    return TrOCRRecognizer(device="cpu", processor=FakeProcessor(), model=FakeModel())


def test_recognize_normalizes_nfc_and_calls_generate():
    recognizer = make_recognizer()
    prediction = recognizer.recognize(Image.new("L", (8, 8), color=255))
    assert prediction == "Tiếng Việt"
    assert unicodedata.is_normalized("NFC", prediction)
    assert recognizer.model.was_eval is True
    assert recognizer.model.generated_with.device.type == "cpu"


def test_recognize_accepts_image_path(tmp_path: Path):
    image_path = tmp_path / "line.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    assert make_recognizer().recognize(image_path) == "Tiếng Việt"


def test_recognize_raises_for_missing_image(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        make_recognizer().recognize(tmp_path / "missing.png")


def test_recognize_rejects_invalid_image_input():
    with pytest.raises(TypeError):
        make_recognizer().recognize(object())
