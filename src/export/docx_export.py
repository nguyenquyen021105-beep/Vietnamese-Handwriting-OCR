from docx import Document

def export_both(faithful_text: str, corrected_text: str, output_path: str):
    doc = Document()
    doc.add_heading("Faithful OCR", level=1)
    doc.add_paragraph(faithful_text)
    doc.add_heading("AI Corrected", level=1)
    doc.add_paragraph(corrected_text)
    doc.save(output_path)
