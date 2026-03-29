from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from nlp.inference import BioBERTRiskScorer
from nlp.pdf_utils import extract_text_from_pdf_bytes
from nlp.risk_fusion import combine_scores

app = FastAPI(title="Smart Health NLP API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

scorer: BioBERTRiskScorer | None = None


def get_scorer() -> BioBERTRiskScorer:
    global scorer
    if scorer is None:
        scorer = BioBERTRiskScorer()
    return scorer


@app.get("/health")
def health():
    return {"status": "ok", "service": "nlp"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    extracted_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")

    try:
        result = get_scorer().score_text(extracted_text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "nlp_score": result.nlp_score,
        "confidence": result.confidence,
        "predicted_label": result.predicted_label,
        "extracted_text_preview": extracted_text[:500],
    }


@app.post("/predict-risk")
async def predict_risk(
    physiological_score: float = Form(...),
    file: UploadFile = File(...),
):
    if not 0.0 <= physiological_score <= 1.0:
        raise HTTPException(status_code=400, detail="physiological_score must be between 0 and 1.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    extracted_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")

    try:
        nlp_result = get_scorer().score_text(extracted_text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    fusion = combine_scores(
        physiological_score=physiological_score,
        nlp_score=nlp_result.nlp_score,
        nlp_confidence=nlp_result.confidence,
    )
    return {
        "physiological_score": round(float(physiological_score), 4),
        "nlp_score": round(nlp_result.nlp_score, 4),
        "final_score": fusion.final_score,
        "risk_category": fusion.risk_category,
        "confidence": fusion.confidence,
    }
