from __future__ import annotations

import io
from functools import lru_cache

import fitz
import pdfplumber
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class PdfTextExtractionError(RuntimeError):
    pass


OCR_MODEL_ID = "microsoft/trocr-small-printed"


@lru_cache(maxsize=1)
def _get_ocr_components() -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    processor = TrOCRProcessor.from_pretrained(OCR_MODEL_ID, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(OCR_MODEL_ID)
    model.eval()
    return processor, model


def _detect_scanned_pdf(pdf_bytes: bytes) -> tuple[bool, int]:
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            total_pages = doc.page_count
            image_only_pages = 0
            for page in doc:
                text = (page.get_text("text") or "").strip()
                image_count = len(page.get_images(full=True))
                if not text and image_count > 0:
                    image_only_pages += 1
            return total_pages > 0 and image_only_pages == total_pages, total_pages
    except Exception:
        return False, 0


def _ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    processor, model = _get_ocr_components()
    page_texts = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            with torch.inference_mode():
                generated_ids = model.generate(pixel_values, max_new_tokens=512)
            page_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if page_text:
                page_texts.append(page_text)

    return "\n".join(page_texts).strip()


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
    fallback_text = "\n".join(fallback_parts).strip()
    if fallback_text:
        return fallback_text

    is_scanned_pdf, total_pages = _detect_scanned_pdf(pdf_bytes)
    if is_scanned_pdf:
        try:
            ocr_text = _ocr_pdf_bytes(pdf_bytes)
        except Exception as exc:
            raise PdfTextExtractionError(
                "This PDF appears to be scanned/image-only, but OCR failed to run. "
                f"Details: {exc}"
            ) from exc
        if ocr_text:
            return ocr_text
        raise PdfTextExtractionError(
            f"This PDF appears to be scanned/image-only across {total_pages} page(s), but OCR did not "
            "recover any readable text."
        )

    raise PdfTextExtractionError(
        "No readable text found in PDF. The file may be image-based, encrypted, or contain unsupported text encoding."
    )
