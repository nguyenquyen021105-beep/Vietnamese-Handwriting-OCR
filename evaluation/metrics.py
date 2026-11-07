import unicodedata
from jiwer import cer, wer

def normalize_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def compute_metrics(reference: str, prediction: str) -> dict:
    reference, prediction = normalize_nfc(reference), normalize_nfc(prediction)
    return {"cer": cer(reference, prediction), "wer": wer(reference, prediction)}
