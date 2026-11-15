# Vietnamese Handwriting OCR & Correction

Hệ thống số hóa tài liệu viết tay tiếng Việt, xuất **Faithful OCR** và **AI Corrected**.

## Kiến trúc
Image/PDF → OpenCV → line detection (DBNet + grouping) → TrOCR → text reconstruction → Faithful OCR → ViT5 controlled correction → AI Corrected → DOCX/TXT/PDF.

> Đây là **starter repository**. TrOCR, DBNet và ViT5 chưa được huấn luyện hoặc tích hợp hoàn chỉnh. API demo chỉ trả kết quả placeholder, không phải OCR thực.

## Phân công
- TV1: `src/recognition/`, `training/train_trocr.py`, `evaluation/`
- TV2: `src/preprocessing/`, `src/detection/`, `data/`, `scripts/prepare_data.py`
- TV3: `src/correction/`, `src/export/`, `app/`, `api/`

## Khởi động (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
streamlit run app/streamlit_app.py
```
Chạy API (terminal khác): `uvicorn api.main:app --reload`.

## Dữ liệu
- `data/raw/`: ảnh/PDF gốc 
- `data/processed/`: ảnh đã xử lý và crop dòng
- `data/annotations/`: nhãn ảnh-dòng 
- `models/`: checkpoint lớn 
- `outputs/`: file xuất 
