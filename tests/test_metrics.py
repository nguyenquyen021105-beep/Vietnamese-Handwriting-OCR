from evaluation.metrics import normalize_nfc, compute_metrics

def test_unicode_nfc():
    assert normalize_nfc("a\u0301") == "á"

def test_exact_match():
    assert compute_metrics("Xin chào", "Xin chào") == {"cer": 0.0, "wer": 0.0}
