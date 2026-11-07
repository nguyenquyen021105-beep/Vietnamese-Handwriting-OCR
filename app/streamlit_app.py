import streamlit as st

st.set_page_config(page_title="Vietnamese Handwriting OCR", layout="wide")
st.title("Vietnamese Handwriting OCR & Correction")
st.info("Starter UI: chưa tích hợp TrOCR/ViT5. Không tạo kết quả OCR giả.")
file = st.file_uploader("Chọn ảnh hoặc PDF", type=["png", "jpg", "jpeg", "pdf"])
if file is not None:
    st.success(f"Đã chọn: {file.name}")
    st.warning("Bước tiếp theo: kết nối pipeline trong src/pipeline.py")
