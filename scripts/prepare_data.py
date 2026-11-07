"""TODO TV2: NFC normalize labels, validate paths, writer-disjoint split."""
import unicodedata

def normalize_label(s: str) -> str:
    return unicodedata.normalize("NFC", s)
