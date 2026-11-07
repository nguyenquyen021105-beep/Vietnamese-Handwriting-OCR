from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="Vietnamese Handwriting OCR - Starter")

@app.get("/health")
def health():
    return {"status": "ok", "ocr_ready": False}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    raise HTTPException(status_code=501, detail="OCR model is not integrated yet")
