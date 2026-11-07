# Training checklist
1. Normalize ground truth with NFC.
2. Split by writer_id; reserve independent test.
3. Smoke-test TrOCR-Small on 32–64 lines.
4. Fine-tune TrOCR-Base with original tokenizer as baseline.
5. Evaluate CER/WER and diacritic/digit/abbreviation accuracy.
6. Produce out-of-fold OCR predictions for ViT5 training.
7. Compare Faithful OCR vs AI Corrected on held-out test.
