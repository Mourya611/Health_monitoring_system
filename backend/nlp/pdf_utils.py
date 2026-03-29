from __future__ import annotations

import io

import fitz
import pdfplumber


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text_parts = []

    # Primary extractor: PyMuPDF
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text_parts.append(page.get_text("text"))
    except Exception:
        text_parts = []

    text = "\n".join(text_parts).strip()
    if text:
        return text

    # Fallback extractor: pdfplumber
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        fallback_parts = []
        for page in pdf.pages:
            fallback_parts.append(page.extract_text() or "")
    return "\n".join(fallback_parts).strip()

