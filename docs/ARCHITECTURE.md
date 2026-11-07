# Architecture
OpenCV → DBNet line grouping → TrOCR-Base → faithful text → ViT5 controlled edits → corrected text → export.

TrOCR-Small is a baseline/fallback. Keep original OCR and per-line coordinates, confidence, edit logs.
